"""Synthetic process tests only: never import or run a trainer/GPU workload."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_training_with_evidence as runner


class EvidenceRunnerTest(unittest.TestCase):
    def test_preserves_training_environment_and_appends_xla_flags(self):
        env = runner.diagnostic_environment(
            Path("/tmp/incident"),
            {"XLA_FLAGS": "--existing=true", "LEGO_BATCH_SIZE": "32", "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        )
        assert env["XLA_FLAGS"].startswith("--existing=true ")
        assert env["LEGO_BATCH_SIZE"] == "32"
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert env["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] == "1"
        assert "%h.%p" in env["NCCL_DEBUG_FILE"]

    def run_synthetic(self, code):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "incident"
            args = SimpleNamespace(
                output=output,
                since="2026-10-22",
                metadata_python=sys.executable,
                source_file=[],
                command=[sys.executable, "-c", code],
            )
            original_popen = subprocess.Popen

            def fake_monitor(command, **kwargs):
                if command[0] in ("nvidia-smi", "dmesg"):
                    command = [sys.executable, "-c", "print('synthetic monitor')"]
                return original_popen(command, **kwargs)

            def fake_collect(snapshot):
                snapshot.output.mkdir()

            with (
                patch.object(runner, "collect", fake_collect) as _collector,
                patch.object(runner.subprocess, "Popen", fake_monitor),
                patch.object(runner, "host_sample", return_value={}),
                patch.object(runner, "process_snapshot", return_value={}),
            ):
                status = runner.run(args)
            record = json.loads((output / "exit.json").read_text())
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            assert sum(e["event"] == "trainer_started" for e in events) == 1
            assert (output / "after").exists()
            return status, record

    def test_preserves_child_exit_and_does_not_retry(self):
        status, record = self.run_synthetic("print('synthetic child'); raise SystemExit(17)")
        assert status == 17
        assert record["returncode"] == 17
        assert record["signal"] is None

    @unittest.skipUnless(os.name == "posix", "POSIX signals")
    def test_records_signal_separately_from_wrapper_exit(self):
        status, record = self.run_synthetic("import os, signal; os.kill(os.getpid(), signal.SIGTERM)")
        assert status == 143
        assert record["returncode"] == -15
        assert record["signal"] == 15


if __name__ == "__main__":
    unittest.main()
