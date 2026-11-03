# Standard-library unittest entrypoint is intentional.
# ruff: noqa: PT009, PT027
"""CPU-only artifact/log protocol tests; no model imports or GPU execution."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location(
    "compare_flora_runs_under_test", Path(__file__).with_name("compare_flora_runs.py")
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)

FIRST = "tshirt-train-128691-w000"
SECOND = "tshirt-train-069431-w000"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def tensor_digest(tensors):
    return text_sha(json.dumps(tensors, sort_keys=True, separators=(",", ":")))


def make_run(directory, job_order=(FIRST, SECOND), elapsed=10.0, mode="baseline"):
    """Create complete synthetic artifacts and phase-ordered native log events."""
    directory.mkdir()
    for position, job in enumerate(job_order):
        identity = {
            "duration_s": 3.0,
            "profile": "YAM",
            "camera_roles": ["main", "left_wrist", "right_wrist"],
            "input_sha256": {role: text_sha(job + "/" + role) for role in ("main", "left_wrist", "right_wrist")},
            "model_revisions": {"model": {"revision": "synthetic-fixed-revision"}},
            "instruction": "fold and stack the t-shirts",
            "yam32": False,
        }
        temporal = [
            {
                "output": '{"segments":[{"s":0,"e":40}]}',
                "prompt_sha256": text_sha(job + "/temporal-prompt"),
                "window": {"scope_id": job + ":fine-window:000000"},
                "frame_plan": {"source_frame_indices": [0, 7, 15]},
            }
        ]
        native = {
            "schema_version": compare.NATIVE_SCHEMA,
            **{
                key: identity[key]
                for key in ("duration_s", "profile", "camera_roles", "input_sha256", "model_revisions")
            },
            "elapsed_s": elapsed * 2,
            "source_clock": "seconds relative to the first native source presentation timestamp",
            "segments": [
                {
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "caption": "Fold the T-shirt.",
                    "status": "complete",
                    "segmentation_provenance": "fine",
                }
            ],
            "pipeline": {
                "temporal_windows": temporal,
                "caption_inputs": [
                    {
                        "event_id": "event_0000",
                        "prompt_sha256": text_sha(job + "/caption-prompt"),
                        "blocks": [{"role": "TARGET", "indices": [2, 6, 11]}],
                    }
                ],
            },
        }
        reused = mode == "session-cache" and position > 0
        timing = {
            "stage_wall_s": elapsed,
            "model_load_wall_s": 0.0 if reused else elapsed / 2,
            "actual_model_loads": int(not reused),
            "reused_model_loads": int(reused),
        }
        source = {
            "schema_version": compare.CHECKPOINT_SCHEMA,
            "id": job,
            "stage": "tempseg",
            "identity": identity,
            "segments": [
                {
                    "start_s": 0.0,
                    "end_s": 2.0,
                    "metadata": {"segmentation_provenance": "fine", "native_detail": [1, 2]},
                }
            ],
            "temporal_audit": temporal,
            "timing": timing,
            "mode": mode,
        }
        source_path = directory / (job + ".tempseg.json")
        write_json(source_path, source)
        write_json(
            directory / (job + ".captioner.json"),
            {
                "schema_version": compare.CHECKPOINT_SCHEMA,
                "id": job,
                "stage": "captioner",
                "identity": identity,
                "tempseg_sha256": compare.sha256(source_path),
                "native_result": native,
                "timing": timing,
                "mode": mode,
            },
        )
        write_json(directory / (job + ".json"), native)

    events = [("FLORA_BATCH", {"event": "run_start"})]
    for stage in ("tempseg", "captioner"):
        for job in job_order:
            tensors = {
                "input_ids": {
                    "dtype": "torch.int64",
                    "shape": [1, 3],
                    "sha256": text_sha(job + "/" + stage + "/input_ids"),
                }
            }
            events.extend(
                [
                    ("FLORA_BATCH", {"event": "progress", "id": job, "stage": stage}),
                    (
                        "FLORA_RUNTIME",
                        {
                            "event": "generation_start",
                            "input_tokens": 3,
                            "output_limit": 512,
                            "input_tensors": tensors,
                            "input_tensor_digest": tensor_digest(tensors),
                        },
                    ),
                    (
                        "FLORA_RUNTIME",
                        {
                            "event": "generation_end",
                            "input_tokens": 3,
                            "output_limit": 512,
                            "generated_tokens": 4,
                            "complete": True,
                            "elapsed_s": elapsed,
                            "peak_allocated_gib": 42.0,
                        },
                    ),
                ]
            )
    log = directory / "run.log"
    log.write_text("\n".join(marker + " " + json.dumps(value) for marker, value in events) + "\n")
    return log


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="flora-comparison-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.baseline_log = make_run(self.baseline)
        self.candidate_log = make_run(self.candidate, elapsed=5.0, mode="session-cache")
        self.args = SimpleNamespace(
            baseline_dir=self.baseline,
            candidate_dir=self.candidate,
            baseline_log=[self.baseline_log],
            candidate_log=[self.candidate_log],
            ids=None,
        )

    def edit_candidate_final(self, edit):
        path = self.candidate / (FIRST + ".json")
        value = json.loads(path.read_text())
        edit(value)
        write_json(path, value)

    def edit_log_event(self, event_name, edit):
        lines = self.candidate_log.read_text().splitlines()
        for index, line in enumerate(lines):
            marker, encoded = line.split(" ", 1)
            value = json.loads(encoded)
            if value["event"] == event_name:
                edit(value)
                lines[index] = marker + " " + json.dumps(value)
                self.candidate_log.write_text("\n".join(lines) + "\n")
                return
        self.fail("Missing fixture event " + event_name)

    def test_exact_parity_ignores_timing_and_model_load_counts(self):
        result = compare.compare(self.args)
        self.assertEqual(result["status"], "EXACT_PARITY")
        self.assertEqual(result["jobs"], 2)
        self.assertEqual(
            result["all_supplied_log_generation_wall_s"],
            {
                "baseline": 40.0,
                "candidate": 20.0,
            },
        )
        second = next(row for row in result["details"] if row["id"] == SECOND)
        self.assertEqual(second["timings"]["tempseg"]["candidate"]["reused_model_loads"], 1)

    def test_changed_caption_is_rejected(self):
        self.edit_candidate_final(lambda value: value["segments"][0].update(caption="Changed caption"))
        with self.assertRaisesRegex(ValueError, r"segments\[0\]\.caption"):
            compare.compare(self.args)

    def test_changed_raw_temporal_text_is_rejected(self):
        self.edit_candidate_final(
            lambda value: value["pipeline"]["temporal_windows"][0].update(output='{"segments":[]}')
        )
        with self.assertRaisesRegex(ValueError, r"temporal_windows\[0\]\.output"):
            compare.compare(self.args)

    def test_changed_tensor_with_valid_recomputed_digest_is_rejected(self):
        def edit(value):
            value["input_tensors"]["input_ids"]["sha256"] = "e" * 64
            value["input_tensor_digest"] = tensor_digest(value["input_tensors"])

        self.edit_log_event("generation_start", edit)
        with self.assertRaisesRegex(ValueError, "input tensor digests"):
            compare.compare(self.args)

    def test_incomplete_eos_generation_is_rejected(self):
        self.edit_log_event("generation_end", lambda value: value.update(complete=False))
        with self.assertRaisesRegex(ValueError, "generation is incomplete"):
            compare.compare(self.args)

    def test_missing_original_generation_log_is_rejected(self):
        self.candidate_log.write_text("")
        with self.assertRaisesRegex(ValueError, "native audit expects"):
            compare.compare(self.args)

    def test_reverse_candidate_matches_independent_baseline_subset(self):
        # The matched video is second in each candidate model stage, so its
        # fixture records actual reuse. Baseline contains only its fresh run.
        baseline = self.root / "fresh-matched"
        candidate = self.root / "reverse-candidate"
        args = SimpleNamespace(
            baseline_dir=baseline,
            candidate_dir=candidate,
            baseline_log=[make_run(baseline, (FIRST,))],
            candidate_log=[make_run(candidate, (SECOND, FIRST), elapsed=5.0, mode="session-cache")],
            ids=[FIRST],
        )
        result = compare.compare(args)
        self.assertEqual(result["status"], "EXACT_PARITY")
        self.assertEqual(result["jobs"], 1)
        row = result["details"][0]
        self.assertEqual(row["id"], FIRST)
        for stage in ("tempseg", "captioner"):
            self.assertEqual(row["timings"][stage]["candidate"]["reused_model_loads"], 1)

    def test_tampered_tempseg_checkpoint_chain_is_rejected(self):
        path = self.candidate / (FIRST + ".tempseg.json")
        value = json.loads(path.read_text())
        value["segments"][0]["metadata"]["native_detail"].append(3)
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "TempSeg SHA-256 mismatch"):
            compare.compare(self.args)

    def test_checkpoint_identity_mismatch_is_rejected(self):
        path = self.candidate / (FIRST + ".captioner.json")
        value = json.loads(path.read_text())
        value["identity"]["instruction"] = "Different task"
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "checkpoint identities"):
            compare.compare(self.args)

    def test_unbound_generation_is_rejected(self):
        lines = self.candidate_log.read_text().splitlines()
        del lines[1]  # Keep run_start, remove the first job/stage progress marker.
        self.candidate_log.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "unbound or overlapping"):
            compare.compare(self.args)

    def test_inconsistent_tensor_digest_is_rejected(self):
        self.edit_log_event("generation_start", lambda value: value.update(input_tensor_digest="f" * 64))
        with self.assertRaisesRegex(ValueError, "inconsistent input_tensor_digest"):
            compare.compare(self.args)

    def test_missing_complete_final_is_rejected(self):
        (self.candidate / (FIRST + ".json")).unlink()
        with self.assertRaisesRegex(ValueError, "missing complete native final"):
            compare.compare(self.args)


if __name__ == "__main__":
    unittest.main()
