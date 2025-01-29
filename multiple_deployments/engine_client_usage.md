# Engine client usage in OpenAIServingChat

OpenAIServingChat is using a a parameter engine_client to know about the client (AsyncLLMEngine in our case)
to interact with VLLM implementation.

Here we will summarize the API used by this: 

* Check whether client is in error (L114)
```python
if self.engine_client.errored:
            raise self.engine_client.dead_error

```
* Get tokenizer (l125)
```python
tokenizer = await self.engine_client.get_tokenizer(lora_request)
```
* Get generator (L214)
```python
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                    )
```
* get generator (L220)
```python
                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        prompt_adapter_request=prompt_adapter_request,
                        priority=request.priority,
                    )
```
It is also used by its parent class - `OpenAIServing` for a single call:
* check idf tracing is enabled (L489)
```python
       is_tracing_enabled = await self.engine_client.is_tracing_enabled()
```

