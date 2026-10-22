"""Offline collector checks; no GPU or training imports/execution."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

from collect_training_incident import Capture
from collect_training_incident import selected_environment
import pytest


class IncidentCaptureTest(unittest.TestCase):
    def test_environment_excludes_credentials_and_unrelated_settings(self):
        env = selected_environment(
            b"NCCL_DEBUG=INFO\0HF_TOKEN=private\0AWS_SECRET_ACCESS_KEY=private\0CUDA_VISIBLE_DEVICES=1\0"
        )
        assert env == {"NCCL_DEBUG": "INFO", "CUDA_VISIBLE_DEVICES": "1"}

    def test_refuses_to_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "incident"
            Capture(target)
            (target / "original").write_text("keep")
            with pytest.raises(FileExistsError):
                Capture(target)
            assert (target / "original").read_text() == "keep"

    def test_failed_missing_and_timed_out_commands_keep_output(self):
        with tempfile.TemporaryDirectory() as root:
            capture = Capture(Path(root) / "incident", timeout=0.2)
            capture.command("failed", [sys.executable, "-c", "print('evidence', flush=True); raise SystemExit(9)"])
            capture.command("missing", [str(Path(root) / "absent-command")])
            capture.command(
                "timeout", [sys.executable, "-c", "import time; print('before timeout', flush=True); time.sleep(30)"]
            )
            assert [r["status"] for r in capture.records] == ["command_failed", "unavailable", "timeout"]
            assert capture.records[0]["returncode"] == 9
            assert "before timeout" in (capture.output / "timeout.stdout.txt").read_text()

    def test_preserves_full_log_and_records_missing_file(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "train.log"
            data = b"first failure\n" + b"x" * (2 * 1024 * 1024) + b"\nlast line\n"
            source.write_bytes(data)
            capture = Capture(Path(root) / "incident")
            capture.snapshot("log.txt", source)
            capture.snapshot("missing.txt", Path(root) / "absent")
            assert (capture.output / "log.txt").read_bytes() == data
            assert capture.records[0]["complete_initial_extent"]
            assert not capture.records[0]["changed_during_copy"]
            assert capture.records[1]["status"] == "unavailable"
            args = type("Args", (), {"job_id": "42", "pid": None, "since": "2026-10-22"})()
            capture.finish(args)
            manifest = json.loads((capture.output / "manifest.json").read_text())
            assert manifest["files"]["log.txt"]["bytes"] == len(data)


if __name__ == "__main__":
    unittest.main()
