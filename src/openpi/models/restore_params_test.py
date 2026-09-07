"""Keep old/new Orbax metadata compatible without changing weight precision."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("dtype", [None, jnp.float32, jnp.bfloat16])
def test_restore_metadata_compatibility(monkeypatch, tmp_path, wrapped, dtype):
    tree = {"params": {"layer": {"value": np.ones((2, 3), dtype=np.float32)}}}

    class Checkpointer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def metadata(self, _path):
            return types.SimpleNamespace(item_metadata=tree) if wrapped else tree

        def restore(self, _path, args):
            assert args.item is not None
            leaves = jax.tree.leaves(args.restore_args)
            assert all(a.dtype is dtype and a.restore_type is np.ndarray for a in leaves)
            return tree

    monkeypatch.setattr(model.ocp, "PyTreeCheckpointer", Checkpointer)
    result = model.restore_params(tmp_path, restore_type=np.ndarray, dtype=dtype)
    np.testing.assert_array_equal(result["layer"], tree["params"]["layer"]["value"])
