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


## Pure VLLM implementation


## Client

