"""
Based on https://github.com/ray-project/kuberay/blob/master/ray-operator/config/samples/vllm/serve.py
and https://docs.ray.io/en/latest/serve/tutorials/vllm-example.html
"""

import os

from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_engine import BaseModelPath, PromptAdapterPath
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger


CACHE_LOCATION = "/home/ray/cache"
logger = logging.getLogger("ray.serve")
os.environ["HF_HUB_CACHE"] = CACHE_LOCATION


app = FastAPI()


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[list[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        # AsyncEngineArgs (https://www.restack.io/p/vllm-answer-async-engine-args-cat-ai)
        # combines General Engine Arguments:
        # --model: Specifies the model to be served. This argument is mandatory and must point to a valid
        # third-party model.
        # --port: Defines the port on which the server will listen for incoming requests. The default is
        # 8080, but this can be changed based on user requirements.
        # --timeout: Sets the maximum time (in seconds) that the server will wait for a request to complete
        # before timing out. This is crucial for managing long-running requests.
        # with additional arguments that enhance the performance and responsiveness of the model serving:
        # --async: Enables asynchronous processing, allowing the server to handle multiple requests concurrently.
        # This is particularly useful for high-throughput applications.
        # --max-concurrent-requests: Limits the number of concurrent requests that can be processed. Setting
        # this value helps prevent server overload and ensures stable performance.
        # --queue-size: Defines the maximum number of requests that can be queued for processing.
        # This is important for managing bursts of incoming requests without dropping them.
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        # We are using here AsyncLLMEngine (https://www.restack.io/p/vllm-answer-async-llm-engine-cat-ai)
        # designed to facilitate asynchronous operations for large language models. This engine allows for
        # efficient handling of multiple requests simultaneously, optimizing resource utilization and
        # reducing latency in model inference. By leveraging asynchronous programming paradigms,
        # the AsyncLLMEngine enhances the throughput of model queries, making it particularly suitable
        # for applications requiring high responsiveness.
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        # If we do not have serving chat, create one
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            # serving models
            base_model_paths = [
                BaseModelPath(name=name, model_path=CACHE_LOCATION)
                for name in served_model_names
            ]
            # çreate (and remember) serving chat
            self.openai_serving_chat = OpenAIServingChat(
                engine_client=self.engine,
                model_config=model_config,
                base_model_paths=base_model_paths,
                response_role=self.response_role,
                lora_modules=self.lora_modules,
                chat_template=self.chat_template,
                prompt_adapters=self.prompt_adapters,
                request_logger=self.request_logger,
            )

        logger.info(f"Request: {request}")
        # Get result
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        # process result
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently, uses argparse because vLLM doesn't expose Python models for all the
    config options we want to support.
    """
    """FlexibleArgumentParser is an argumentParser that allows both underscore and dash in names."""
    parser = FlexibleArgumentParser(description="vLLM CLI")
    # defines parsing arguments for VLLM
    parser = make_arg_parser(parser)
    # build cli like string
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    # parse arguments
    parsed_args = parse_vllm_args(cli_args)
    # get engine args
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    # build an app
    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
    )


model = build_app(
    {
        "model": os.environ["MODEL_ID"],
        "tensor-parallel-size": os.environ["TENSOR_PARALLELISM"],
        "pipeline-parallel-size": os.environ["PIPELINE_PARALLELISM"],
    }
)
