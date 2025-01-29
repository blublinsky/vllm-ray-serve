# This are experiments of running VLLM in Ray serve

* [pure VLLM](pure_vllm) - is wrapping VLLM code as-is into deployment
* [multiple-deployments](multiple_deployments) - splitting VLLM code into 2 separate deployments - 
Async LLM engine and OpenAI HTTP server

## Why bother with multiple deployments?

Some considerations:

* Different resource requirements
* Support for separate scaling for both deployments
* Ability to add additional processing around Async LLM engine