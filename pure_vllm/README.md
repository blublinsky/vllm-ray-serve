# Very simple Ray serve with VLLM

This is a very simple Ray serve implementation of Ray serve with VLLM, based on 
[Ray documentation](https://docs.ray.io/en/latest/serve/tutorials/vllm-example.html) and Kuberay
[example](https://github.com/ray-project/kuberay/tree/master/ray-operator/config/samples/vllm), updated 
to the latest version of Ray (2.40) and VLLM (0.6.6.post1).

Implementation consists of 3 main parts:
* [support](../support) - things that can be used for any Ray serve with VLLM implementation
* [pure VLLM](.) - current implementation
* [client](../client) - client for testing

## Support

Running VLLM is not supported on MAC, so we introduce a "playground" [pod](../support/pod.yaml), 
that we can use on OS cluster to test our implementation code. It is based on Ray docker. To test 
VLLM implementations we need to:
* Install VLLM
* Set `HF_TOKEN` to enable access to models
* Uninstall `pynvml` (improves performance)
Once this is done we can use `serve` cli command to test our implementation

For running on OS cluster we use `hf-secret` [secret](../support/hf_secret.yaml) to store HF token
that is used by Ray cluster

## Pure VLLM implementation

This contains 2 files:
* [vllm_serve](vllm_serve_066_post1.py) containing the actual implementation, updated to current software versions
* [reployment](ray-service.vllm.yaml) based on KubeRay cluster and adapted to our OS cluster

## Client

A very simple [client](../client/query.py) is implemented to test OpenAI protocol

