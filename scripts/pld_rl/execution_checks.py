"""Standard-library checks for training location guards; never trains."""
# ruff: noqa: PT009, PT027
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.execution import require_training_runtime


class ExecutionChecks(unittest.TestCase):
    def test_slurm_requires_allocation(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            require_training_runtime("slurm", cuda_available=True)

    def test_slurm_preserved(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "fixture"}, clear=True):
            self.assertEqual(require_training_runtime("slurm", cuda_available=True)["job"], "fixture")

    def test_no_cpu_fallback(self):
        with self.assertRaises(RuntimeError):
            require_training_runtime("thor-offline", cuda_available=False)

    def test_thor_optin_does_not_allow_workstation(self):
        with patch.dict(os.environ, {"YAM_RL_TRAINING_TARGET": "thor-offline"}, clear=True), \
                patch("yam_pld_rl.execution.platform.machine", return_value="x86_64"), \
                self.assertRaises(RuntimeError):
            require_training_runtime("thor-offline", cuda_available=True, capability=(11, 0))

    def test_thor_container_and_gpu_required(self):
        with patch.dict(os.environ, {"YAM_RL_TRAINING_TARGET": "thor-offline"}, clear=True), \
                patch("yam_pld_rl.execution.platform.machine", return_value="aarch64"), \
                patch("yam_pld_rl.execution.Path.is_file", return_value=True):
            self.assertEqual(require_training_runtime("thor-offline", cuda_available=True, capability=(11, 0))["target"], "thor-offline")
            with self.assertRaises(RuntimeError):
                require_training_runtime("thor-offline", cuda_available=True, capability=(9, 0))
        with patch.dict(os.environ, {"YAM_RL_TRAINING_TARGET": "thor-offline"}, clear=True), \
                patch("yam_pld_rl.execution.platform.machine", return_value="aarch64"), \
                patch("yam_pld_rl.execution.Path.is_file", return_value=False), self.assertRaises(RuntimeError):
            require_training_runtime("thor-offline", cuda_available=True, capability=(11, 0))


if __name__ == "__main__":
    unittest.main()
