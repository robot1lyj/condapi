# Standard-library unittest entrypoint is intentional.
# ruff: noqa: PT009, PT027
"""CPU orchestration tests using fake native functions; no models or training."""

import copy
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

RUNNER_PATH = Path(__file__).with_name("run_flora_tshirt_batch.py")
SPEC = importlib.util.spec_from_file_location("flora_batch_under_test", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
MODELS = {
    "tempseg": {"repo": "official/tempseg", "revision": "fixed-tempseg"},
    "captioner": {"repo": "official/captioner", "revision": "fixed-captioner"},
}


class FakeEngine:
    """Stand in for native CPU media validation, keeping real file SHA checks."""

    @staticmethod
    def _inputs(video, profile, left_wrist, right_wrist, yam32, instruction):
        videos = {
            role: SimpleNamespace(
                path=path,
                sha256=runner.file_sha256(path),
                rate=Fraction(30),
                duration_s=3.0,
            )
            for role, path in (("main", video), ("left_wrist", left_wrist), ("right_wrist", right_wrist))
        }
        return videos, "YAM"


class BatchProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="flora-batch-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patch = mock.patch.object(runner, "event")
        self.events = patch.start()
        self.addCleanup(patch.stop)
        self.loads, self.operations, self.sessions = [], [], []
        self.caption_status = "complete"
        self.fail_caption = False
        self.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=mock.Mock()))
        owner = self

        class NativeSession:
            def __init__(self):
                self.model = self.proc = None
                self.audit = {}
                owner.sessions.append(self)

            def close(self):
                self.model = self.proc = None

            def load(self, kind):
                # Native Session.load dynamically calls self.close before loading.
                # Exercise that behavior so the subclass cache cannot rely on a
                # simpler fake that leaves loaded_kind untouched.
                self.close()
                owner.loads.append(kind)
                self.model, self.proc = object(), object()
                return self.model, self.proc

        self.native_session = NativeSession
        self.jobs = self.make_jobs()

    def make_jobs(self):
        rows = []
        for name in ("one", "two"):
            paths = {}
            for key in ("video", "left_wrist", "right_wrist"):
                path = self.root / f"{name}-{key}.mp4"
                path.write_bytes((name + ":" + key).encode())
                paths[key] = path.name
            rows.append(dict(id=name, profile="yam", instruction="fold and stack the t-shirts", **paths))
        self.manifest = self.root / "jobs.jsonl"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        jobs = runner.load_jobs(self.manifest, None)
        runner.prepare_jobs(jobs, FakeEngine(), copy.deepcopy(MODELS))
        return jobs

    def segment(self, videos, pool, session, instruction, progress):
        session.load("tempseg")
        self.operations.append(("tempseg", videos["main"].path.name))
        progress(0.25, desc="native temporal window")
        return (
            [
                {
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "metadata": {"segmentation_provenance": "fine_retry", "kept_native_metadata": True},
                }
            ],
            [
                {
                    "window": {"window_index": 0},
                    "frame_plan": {"source_frame_indices": [0, 7]},
                    "prompt_sha256": "a" * 64,
                    "output": '{"segments":[{"s":0,"e":40}]}',
                }
            ],
        )

    def caption(self, videos, segments, session, instruction, progress, profile):
        session.load("captioner")
        self.operations.append(("captioner", videos["main"].path.name))
        if self.fail_caption:
            raise RuntimeError("simulated caption interruption")
        progress(0.55, desc="native caption")
        caption = "Fold the T-shirt." if self.caption_status == "complete" else ""
        return (
            [{"start_s": 0.0, "end_s": 2.0, "caption": caption, "status": self.caption_status}],
            [{"event_id": "event_0000", "prompt_sha256": "b" * 64}],
        )

    def run_batch(self, mode="session-cache", stage="all", directory=None, jobs=None):
        output = directory or self.root / mode
        args = SimpleNamespace(output_dir=output, stage=stage, mode=mode)
        runner.run_stages(
            args, self.jobs if jobs is None else jobs, self.native_session, self.segment, self.caption, self.torch
        )
        return output

    def read(self, directory, filename):
        return json.loads((directory / filename).read_text())

    def test_baseline_reloads_for_each_job_and_runs_persisted_stages(self):
        output = self.run_batch(mode="baseline")
        self.assertEqual(self.loads, ["tempseg", "tempseg", "captioner", "captioner"])
        self.assertEqual([stage for stage, _ in self.operations], ["tempseg", "tempseg", "captioner", "captioner"])
        for job in ("one", "two"):
            self.assertEqual(self.read(output, job + ".tempseg.json")["timing"]["actual_model_loads"], 1)
            self.assertEqual(self.read(output, job + ".captioner.json")["timing"]["actual_model_loads"], 1)
        self.assertTrue(all(session.model is None and session.proc is None for session in self.sessions))

    def test_cache_reuses_same_kind_across_jobs_but_closes_at_stage_end(self):
        output = self.run_batch()
        self.assertEqual(self.loads, ["tempseg", "captioner"])
        for stage in ("tempseg", "captioner"):
            first = self.read(output, "one." + stage + ".json")["timing"]
            second = self.read(output, "two." + stage + ".json")["timing"]
            self.assertEqual((first["actual_model_loads"], first["reused_model_loads"]), (1, 0))
            self.assertEqual((second["actual_model_loads"], second["reused_model_loads"]), (0, 1))
        self.assertEqual(len(self.sessions), 2)
        self.assertTrue(all(session.model is None and session.proc is None for session in self.sessions))

    def test_modes_preserve_native_output_audit_and_provenance(self):
        baseline = self.run_batch(mode="baseline")
        candidate = self.run_batch()
        for name in ("one", "two"):
            left, right = self.read(baseline, name + ".json"), self.read(candidate, name + ".json")
            left.pop("elapsed_s")
            right.pop("elapsed_s")
            self.assertEqual(left, right)
            self.assertEqual(right["schema_version"], runner.NATIVE_SCHEMA)
            self.assertEqual(right["segments"][0]["segmentation_provenance"], "fine_retry")
            self.assertEqual(right["pipeline"]["temporal_windows"][0]["output"], '{"segments":[{"s":0,"e":40}]}')
            self.assertTrue(
                self.read(candidate, name + ".tempseg.json")["segments"][0]["metadata"]["kept_native_metadata"]
            )

    def test_resume_makes_no_new_native_calls_and_preserves_exact_final(self):
        output = self.run_batch()
        before = {path.name: path.read_bytes() for path in output.glob("*.json")}
        self.loads.clear()
        self.operations.clear()
        self.run_batch()
        self.assertEqual(self.loads, [])
        self.assertEqual(self.operations, [])
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.glob("*.json")})

    def test_caption_checkpoint_recovers_missing_final_without_inference(self):
        output = self.run_batch()
        final = output / "one.json"
        expected = final.read_bytes()
        final.unlink()
        self.loads.clear()
        self.operations.clear()
        self.run_batch()
        self.assertEqual(self.loads, [])
        self.assertEqual(self.operations, [])
        self.assertEqual(final.read_bytes(), expected)

    def test_caption_failure_keeps_tempseg_stage_resumable(self):
        self.fail_caption = True
        with self.assertRaisesRegex(RuntimeError, "simulated caption interruption"):
            self.run_batch()
        output = self.root / "session-cache"
        self.assertTrue((output / "one.tempseg.json").exists())
        self.assertTrue((output / "two.tempseg.json").exists())
        self.assertFalse((output / "one.json").exists())
        self.fail_caption = False
        self.loads.clear()
        self.operations.clear()
        self.run_batch()
        self.assertEqual(self.loads, ["captioner"])
        self.assertEqual([stage for stage, _ in self.operations], ["captioner", "captioner"])

    def test_instruction_revision_and_input_hash_changes_reject_resume(self):
        self.run_batch()
        for field in ("instruction", "model_revisions", "input_sha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.jobs)
                if field == "instruction":
                    changed[0]["identity"][field] = "different instruction"
                elif field == "model_revisions":
                    changed[0]["identity"][field]["tempseg"]["revision"] = "different revision"
                else:
                    changed[0]["identity"][field]["main"] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "contract/hash/revision mismatch"):
                    self.run_batch(jobs=changed)

    def test_source_file_change_during_run_is_rejected(self):
        self.jobs[0]["source_paths"][0].write_bytes(b"changed source payload")
        with self.assertRaisesRegex(RuntimeError, "Source media changed"):
            self.run_batch()
        self.assertEqual(self.loads, [])

    def test_tampered_final_is_rejected_without_overwriting(self):
        output = self.run_batch()
        final = output / "one.json"
        final.write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "differs from validated caption checkpoint"):
            self.run_batch()
        self.assertEqual(final.read_text(), "{}")

    def test_foreign_final_without_checkpoint_is_not_accepted(self):
        output = self.root / "session-cache"
        output.mkdir()
        (output / "one.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "no identity-bound caption checkpoint"):
            self.run_batch()
        self.assertEqual(self.loads, [])

    def test_incomplete_native_status_is_preserved(self):
        self.caption_status = "unsupported_target_frames"
        output = self.run_batch()
        row = self.read(output, "one.json")["segments"][0]
        self.assertEqual(row["status"], "unsupported_target_frames")
        self.assertEqual(row["caption"], "")
        completions = [call.kwargs for call in self.events.call_args_list if call.args[0] == "captioner_complete"]
        self.assertTrue(all(event["incomplete"] == 1 for event in completions))

    def test_worker_lock_rejects_second_worker_then_releases(self):
        path = self.root / "gpu.lock"
        with (
            runner.worker_lock(path),
            self.assertRaisesRegex(RuntimeError, "Another FLORA batch worker"),
            runner.worker_lock(path),
        ):
            self.fail("second worker acquired shared lock")
        with runner.worker_lock(path):
            pass

    def test_publication_is_0644_even_under_restrictive_umask_and_never_overwrites(self):
        path = self.root / "published.json"
        previous = os.umask(0o077)
        try:
            runner.publish_exclusive(path, {"ok": True})
        finally:
            os.umask(previous)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            runner.publish_exclusive(path, {"ok": False})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
        self.assertEqual(list(self.root.glob(".published.json.*.partial")), [])

    def test_manifest_rejects_stage_artifact_filename_collision(self):
        rows = [json.loads(line) for line in self.manifest.read_text().splitlines()]
        rows[1]["id"] = "one.tempseg"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "collides with another job's stage artifact"):
            runner.load_jobs(self.manifest, None)

    def test_validate_only_entrypoint_does_not_import_torch_or_create_outputs(self):
        flora = self.root / "fake-flora"
        package = flora / "captioner" / "batch_utils"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "config.py").write_text("MODELS=" + repr(MODELS) + "\n")
        (package / "api.py").write_text(
            textwrap.dedent("""\
            import hashlib
            from types import SimpleNamespace
            class DataEnginePipeline:
                @staticmethod
                def _inputs(video, profile, left_wrist, right_wrist, yam32, instruction):
                    cameras = (("main", video), ("left_wrist", left_wrist), ("right_wrist", right_wrist))
                    return {
                        role: SimpleNamespace(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), duration_s=3.0)
                        for role, path in cameras
                    }, "YAM"
            """)
        )
        trap = self.root / "import-trap"
        trap.mkdir()
        (trap / "torch.py").write_text("raise AssertionError('validate-only imported torch')\n")
        output, lock = self.root / "not-created", self.root / "not-created.lock"
        environment = dict(os.environ, PYTHONPATH=str(trap), CUDA_VISIBLE_DEVICES="")
        process = subprocess.run(
            [
                sys.executable,
                str(RUNNER_PATH),
                "--flora-root",
                str(flora),
                "--input-jsonl",
                str(self.manifest),
                "--output-dir",
                str(output),
                "--lock-file",
                str(lock),
                "--validate-only",
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn('"event": "validation_complete"', process.stdout)
        self.assertIn('"new_model_calls": 0', process.stdout)
        self.assertFalse(output.exists())
        self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
