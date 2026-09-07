import os
import subprocess
import sys


def test_jax_policy_import_does_not_require_optional_backends():
    code = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'lerobot', 'transformers', 'safetensors'}:
            raise ModuleNotFoundError('Optional backend unexpectedly imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from openpi.policies import policy_config
from openpi.training import config
assert config.get_config('pi05_yam').model.pi05
assert 'openpi.models_pytorch.pi0_pytorch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], env={**os.environ, "JAX_PLATFORMS": "cpu"}, check=True, timeout=60)
