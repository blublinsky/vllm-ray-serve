# Very simple Ray serve with VLLM

This is a very simple Ray serve implementation of Ray serve with VLLM, based on 
[Ray documentation](https://docs.ray.io/en/latest/serve/tutorials/vllm-example.html) and Kuberay
[example](https://github.com/ray-project/kuberay/tree/master/ray-operator/config/samples/vllm), updated 
to the latest version of Ray (2.40) and VLLM (see below).

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

### Software versions

Both Ray and VLLM project is extremely active, so versions are created very fast and are not necessarily backward 
compatible. What is even more frustrating is depending on the platform where you install `pip install vllm` installs 
different versions. Currently Linux/GPU installs version 0.6.6.post1, while running the same command on mac, installs
version 0.6.3.post1 which has slightly different APIs. Even more complex is that both VLLM and Ray are using 
same libraries and when used in virtual environment (the way Ray does it) it is not guaranteed that you will
get compatible versions.

As an example, I made both VLLM 0.6.6.post1 and 0.6.3.post1 work on playground pod, but not on Ray cluster.

This means:
* One has to be very careful to make sure that he is using well defined versions of both Ray and VLLM.
* While using "standard" Ray image is very convenient for quick testing, in reality we would need to build 
custom Ray images, including VLLM install (this will improve performance of the service creation, which is
currently in the order of 7 mins) and modifying all required libraries. 
* It is highly recommended to do initial testing using `playground` pod and only then build the actual image.

## Client

A very simple [client](../client/query.py) is implemented to test OpenAI protocol. 

Note that once the service is ready KubeRay creates a specialized service for serving, in our case
`llama-3-8b-serve-svc` which is balancing across all the HTTP listeners (Proxy actors). This is the 
service that should be used to access model serving

