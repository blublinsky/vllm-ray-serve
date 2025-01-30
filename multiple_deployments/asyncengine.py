import os
import logging
import asyncio
import nest_asyncio
from typing import Any

from ray import serve

from async_llm_engine_support import AsyncLLMEngineDeployment, AsyncLLMEngineProxy

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

CACHE_LOCATION = "/home/ray/cache"
logger = logging.getLogger("ray.serve")
os.environ["HF_HUB_CACHE"] = CACHE_LOCATION
nest_asyncio.apply()


async def gen(engine: AsyncLLMEngineProxy, example_input: dict[str, Any], r_id: int) -> list[str]:
    results_generator = engine.generate(
        prompt=example_input["prompt"],
        sampling_params=SamplingParams(temperature=example_input["temperature"]),
        request_id=str(r_id),
    )
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    return text_outputs


async def main():

    args = AsyncEngineArgs(model="meta-llama/Meta-Llama-3-8B-Instruct", tensor_parallel_size=1,
                           gpu_memory_utilization=0.9, enforce_eager=True)
    app = AsyncLLMEngineDeployment.bind(engine_args=args)
    handle = serve.run(app)
    example_inputs = [
        {
            "prompt": "About 200 words, please give me some tourist information about Tokyo.",
            "temperature": 0.9,
        },
        {
            "prompt": "About 200 words, please give me some tourist information about Osaka.",
            "temperature": 0.9,
        },
    ]

    proxy = AsyncLLMEngineProxy(engine=handle)
    for i in range(len(example_inputs)):
        result = await gen(engine=proxy, example_input=example_inputs[i], r_id=i)
        logger.info(f"Generation result is  {result}")

    logger.info(f"engine error {proxy.errored}")

    logger.info(f"Tracing enabled {await proxy.is_tracing_enabled()}")

    decoding_config = await handle.get_decoding_config.remote()
    logger.info(f"decoding config {decoding_config}")

    lora_config = await handle.get_lora_config.remote()
    logger.info(f"LoRA config {lora_config}")

    model_config = await proxy.get_model_config()
    logger.info(f"model config {model_config}")

    para_config = await handle.get_parallel_config.remote()
    logger.info(f"parallel config {para_config}")

    schedule_config = await handle.get_scheduler_config.remote()
    logger.info(f"scheduler config {schedule_config}")

"""
    tokenizer = await proxy.get_tokenizer()
    logger.info(f"tokenizer {tokenizer}")
"""

if __name__ == "__main__":
    asyncio.run(main())
