"""Exercise real Thor GPU kernels before installing or loading any policy."""

import importlib.metadata
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np


def main():
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU backend required, found {devices}")
    results = []
    for dtype in (jnp.float32, jnp.bfloat16):
        # Exact small integers distinguish execution correctness from timing.
        x = jnp.ones((256, 256), dtype=dtype)
        compute = jax.jit(lambda a: jnp.matmul(a, a, precision=jax.lax.Precision.HIGHEST))
        start = time.monotonic()
        y = compute(x).block_until_ready()
        compile_and_run = time.monotonic() - start
        start = time.monotonic()
        y = compute(x).block_until_ready()
        run = time.monotonic() - start
        np.testing.assert_array_equal(np.asarray(y, dtype=np.float32), np.full((256, 256), 256))
        # Exercise random generation and non-GEMM kernels too.
        z = jax.jit(lambda key, dt=dtype: jnp.tanh(jax.random.normal(key, (256, 256), dtype=dt)))(
            jax.random.key(0)
        ).block_until_ready()
        if not np.isfinite(np.asarray(z, dtype=np.float32)).all():
            raise RuntimeError("Non-finite GPU random/nonlinear output")
        results.append({"dtype": str(dtype), "compile_and_run_s": compile_and_run, "run_s": run})
    packages = {}
    for name in ("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    print(
        json.dumps(
            {
                "stage": "gpu_kernels_only_not_policy_validation",
                "python": platform.python_version(),
                "architecture": platform.machine(),
                "devices": [str(d) for d in devices],
                "packages": packages,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
