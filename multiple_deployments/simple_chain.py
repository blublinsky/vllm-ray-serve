"""
This based on this documentation:
    https://docs.ray.io/en/latest/serve/model_composition.html#chaining-deploymenthandle-calls
"""
import uuid
from ray import serve
from ray.serve.handle import DeploymentHandle


class AdderImpl:
    def __init__(self, increment: int):
        self._increment = increment
        self.id = str(uuid.uuid4())
        print(f"Created adder with id {self.id}")

    def add(self, val: int) -> int:
        print(f"invoking adder with id {self.id}")
        return val + self._increment


class MultiplierImpl:
    def __init__(self, multiple: int):
        self._multiple = multiple
        self.id = str(uuid.uuid4())
        print(f"Created multiplier with id {self.id}")

    def multiply(self, val: int) -> int:
        print(f"invoking multiplier with id {self.id}")
        return val * self._multiple


class IngressImpl:
    def __init__(self, adder: DeploymentHandle, multiplier: DeploymentHandle):
        self._adder = adder
        self._multiplier = multiplier
        self.id = str(uuid.uuid4())
        print(f"Created ingress with id {self.id}")

    async def sum_multiply(self, input: int) -> int:
        print(f"invoking ingress with id {self.id}")
        adder_response = self._adder.add.remote(input)
        # Pass the adder response directly into the multipler (no `await` needed).
        multiplier_response = self._multiplier.multiply.remote(
            adder_response
        )
        # `await` the final chained response.
        return await multiplier_response

Adder = serve.deployment(_func_or_class=AdderImpl, name="adder", num_replicas=2, ray_actor_options={"num_cpus": .2})
Multiplier = serve.deployment(_func_or_class=MultiplierImpl, name="multiplier", num_replicas=3, ray_actor_options={"num_cpus": .2})
Ingress = serve.deployment(_func_or_class=IngressImpl, name="ingress", num_replicas=2, ray_actor_options={"num_cpus": .2})

app = Ingress.bind(
    Adder.bind(increment=1),
    Multiplier.bind(multiple=2),
)

handle: DeploymentHandle = serve.run(app)

# Run several times to check load balancing
for _ in range(10):
    response = handle.sum_multiply.remote(5).result()
    print(f"Response {response}")