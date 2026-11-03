# Standard-library unittest entrypoint is intentional.
# ruff: noqa: PT009, PT027
"""Standard-library fixtures for review boundaries; no model/data dependencies."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "fineart_review_under_test", Path(__file__).with_name("prepare_fineart_subtask_review.py")
)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class SubtaskReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fineart-review-test-")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.results = self.root / "results"
        self.results.mkdir()
        self.lineage = self.root / "lineage.json"
        self.metadata = self.root / "audit.jsonl"
        self.metadata.write_text(
            json.dumps({"pool": "train", "source_episode_index": 7, "length": 180, "fps": 30}) + "\n"
        )
        self.windows = [self.window("w0", 0, 120, 0, 90), self.window("w1", 60, 180, 90, 180)]
        self.write(self.lineage, {"status": "NATIVE_INPUTS_VALIDATED_SEMANTIC_REVIEW_PENDING", "windows": self.windows})
        self.result("w0", "Spread the T-shirt flat.")
        self.result("w1", "Fold the left side toward the center.")

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def window(wid, start, end, core_start, core_end):
        return {
            "id": wid,
            "pool": "train",
            "source_episode_index": 7,
            "parent_record_sha256": "parent-record-fixture",
            "parent_from_frame": start,
            "parent_to_frame": end,
            "core_from_frame": core_start,
            "core_to_frame": core_end,
            "local_to_parent_offset_s": start / 30,
            "fps": 30,
            "training_episode": False,
            "source_revision": "source-revision-fixture",
            "review_status": "SEMANTIC_REVIEW_PENDING",
            "cameras": {
                k: {"path": k + ".mp4", "sha256": k + "-hash", "frames": end - start}
                for k in ("top", "left_wrist", "right_wrist")
            },
        }

    def result(self, wid, caption, segments=None):
        value = {
            "schema_version": "flora_demo_annotation_v1",
            "duration_s": 4.0,
            "profile": "YAM",
            "input_sha256": {"main": "top-hash", "left_wrist": "left_wrist-hash", "right_wrist": "right_wrist-hash"},
            "model_revisions": {"captioner": {"revision": "fixture"}},
            "segments": segments
            if segments is not None
            else [{"start_s": 0.0, "end_s": 4.0, "caption": caption, "status": "complete"}],
        }
        self.write(self.results / (wid + ".json"), value)
        return value

    def draft(self):
        return review.make_draft(self.lineage, self.metadata, self.results)

    def approved_files(self):
        draft, form = self.draft()
        draft_path, form_path = self.root / "draft.json", self.root / "approved.json"
        self.write(draft_path, draft)
        form.update(
            draft_sha256=review.digest(draft_path),
            reviewer="fixture-agent",
            review_method="agent_visual_review_protocol_fixture",
            reviewed_at="2026-11-03T15:00:00+08:00",
        )
        for parent in form["parents"]:
            for interval in parent["intervals"]:
                interval.update(
                    review_status="APPROVED", reviewer="fixture-reviewer", note="Fixture video review decision."
                )
            for item in parent["issue_resolutions"]:
                item.update(resolution="Fixture semantic relabel after agent visual review.", reviewer="fixture-agent")
        self.write(form_path, form)
        # This is a protocol fixture, not a real parquet or a timestamp extraction test.
        parquet = self.root / "source.parquet"
        parquet.write_bytes(b"fixture-source-not-an-actual-parquet")
        timestamp_data = {
            "schema_version": 1,
            "parents": [
                {
                    "pool": "train",
                    "source_episode_index": 7,
                    "fps": 30,
                    "timestamps": [7.5 + i / 30 for i in range(180)],
                    "timestamp_source": {
                        "kind": "actual_parquet_timestamp_column",
                        "path": str(parquet),
                        "sha256": review.digest(parquet),
                        "column": "timestamp",
                    },
                }
            ],
        }
        timestamp_path = self.root / "timestamps.json"
        self.write(timestamp_path, timestamp_data)
        return draft_path, form_path, timestamp_path, form, timestamp_data

    def test_context_is_retained_but_core_owns_training_suggestions(self):
        draft, form = self.draft()
        self.assertEqual(draft["status"], "DRAFT_REQUIRES_SEMANTIC_REVIEW")
        parent = draft["parents"][0]
        self.assertEqual(parent["raw_labels"][0]["parent_interval"], [0, 120])
        self.assertEqual(parent["raw_labels"][1]["parent_interval"], [60, 180])
        self.assertEqual(parent["raw_labels"][0]["core_owned_interval"], [0, 90])
        self.assertEqual(parent["raw_labels"][1]["core_owned_interval"], [90, 180])
        self.assertFalse(parent["issues"])
        self.assertEqual(
            [(r["from_frame"], r["to_frame"]) for r in form["parents"][0]["intervals"]], [(0, 90), (90, 180)]
        )
        self.assertTrue(all(r["review_status"] == "UNREVIEWED" for r in form["parents"][0]["intervals"]))
        self.assertEqual(parent["windows"][0]["result_source"]["sha256"], review.digest(self.results / "w0.json"))

    def test_parallel_unknown_blank_incomplete_and_gaps_require_review(self):
        self.write(self.lineage, {"windows": [self.window("w0", 0, 180, 0, 180)]})
        value = self.result(
            "w0",
            None,
            [
                {"start_s": 0, "end_s": 2, "caption": "Spread the shirt.", "status": "complete"},
                {"start_s": 1, "end_s": 3, "caption": "unknown", "status": "complete"},
                {"start_s": 3.5, "end_s": 4, "caption": "", "status": "complete"},
                {"start_s": 4, "end_s": 5, "caption": "mistake", "status": "unsupported_target_frames"},
            ],
        )
        value["duration_s"] = 6
        self.write(self.results / "w0.json", value)
        draft, form = self.draft()
        codes = {i["code"] for i in draft["parents"][0]["issues"]}
        self.assertTrue(
            {
                "OVERLAPPING_RAW_LABELS",
                "NON_INSTRUCTION_OR_UNCERTAIN_LABEL",
                "EMPTY_INSTRUCTION",
                "INCOMPLETE_SEGMENT",
                "UNCOVERED_LABEL_FRAMES",
            }
            <= codes
        )
        self.assertEqual(len(draft["parents"][0]["raw_labels"]), 4)
        overlap = next(x for x in form["parents"][0]["intervals"] if x["from_frame"] == 30)
        self.assertIsNone(overlap["content"])
        self.assertEqual(len(overlap["source_label_ids"]), 2)

    def test_missing_window_is_not_filled_with_previous_instruction(self):
        self.write(self.lineage, {"windows": self.windows[:1]})
        (self.results / "w0.json").unlink()
        draft, form = self.draft()
        codes = {i["code"] for i in draft["parents"][0]["issues"]}
        self.assertTrue({"MISSING_FINAL_RESULT", "MISSING_WINDOW_CORE", "UNCOVERED_LABEL_FRAMES"} <= codes)
        self.assertTrue(all(r["content"] is None for r in form["parents"][0]["intervals"]))

    def test_local_boundaries_select_actual_frame_times_half_open(self):
        self.assertEqual(review.local_boundary(1 / 30), 1)
        self.assertEqual(review.local_boundary(0.01), 1)
        self.assertEqual(review.local_boundary(2 / 30), 2)

    def test_wrong_input_hash_preserves_raw_but_suppresses_suggestion(self):
        path = self.results / "w0.json"
        value = json.loads(path.read_text())
        value["input_sha256"]["main"] = "wrong-input"
        self.write(path, value)
        draft, form = self.draft()
        self.assertIn("RESULT_INPUT_HASH_MISMATCH", {x["code"] for x in draft["parents"][0]["issues"]})
        self.assertEqual(draft["parents"][0]["raw_labels"][0]["raw_segment"]["caption"], "Spread the T-shirt flat.")
        self.assertIsNone(form["parents"][0]["intervals"][0]["content"])

    def test_two_layers_are_preserved_without_selecting_one(self):
        path = self.results / "w0.json"
        value = json.loads(path.read_text())
        value["coarse_segments"] = [{"start_s": 0, "end_s": 4, "caption": "Fold the shirt."}]
        self.write(path, value)
        draft, form = self.draft()
        self.assertIn("MULTIPLE_LABEL_LAYERS_REQUIRE_SELECTION", {x["code"] for x in draft["parents"][0]["issues"]})
        self.assertEqual(
            draft["parents"][0]["windows"][0]["auxiliary_raw_label_layers"]["coarse_segments"], value["coarse_segments"]
        )
        self.assertIsNone(form["parents"][0]["intervals"][0]["content"])

    def test_incorrect_lineage_offset_is_rejected(self):
        changed = copy.deepcopy(self.windows)
        changed[1]["local_to_parent_offset_s"] = 1.0
        self.write(self.lineage, {"windows": changed})
        with self.assertRaisesRegex(ValueError, "offset"):
            self.draft()

    def test_export_uses_supplied_timestamp_instead_of_frame_div_fps(self):
        paths = self.approved_files()
        result = review.validate_review(*paths[:3])
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["review_method"], "agent_visual_review_protocol_fixture")
        self.assertEqual(
            result["parents"][0]["language_persistent"],
            [
                {
                    "role": "assistant",
                    "content": "Spread the T-shirt flat.",
                    "style": "subtask",
                    "timestamp": 7.5,
                    "camera": None,
                    "tool_calls": None,
                },
                {
                    "role": "assistant",
                    "content": "Fold the left side toward the center.",
                    "style": "subtask",
                    "timestamp": 10.5,
                    "camera": None,
                    "tool_calls": None,
                },
            ],
        )

    def test_unreviewed_or_incomplete_timeline_cannot_export(self):
        draft_path, form_path, ts_path, original, _ = self.approved_files()
        for mutation in (
            lambda f: f["parents"][0]["intervals"][0].update(review_status="UNREVIEWED"),
            lambda f: f["parents"][0]["intervals"][1].update(from_frame=91),
            lambda f: f["parents"][0]["intervals"][1].update(from_frame=89),
            lambda f: f["parents"][0]["intervals"].pop(),
            lambda f: f["parents"][0]["intervals"][0].update(content="unknown"),
            lambda f: f["parents"][0]["intervals"][0].update(note=""),
            lambda f: f.update(draft_sha256="changed"),
            lambda f: f.update(review_method=""),
        ):
            changed = copy.deepcopy(original)
            mutation(changed)
            self.write(form_path, changed)
            with self.assertRaises(ValueError):
                review.validate_review(draft_path, form_path, ts_path)

    def test_synthetic_missing_gapped_or_changed_source_timestamps_are_rejected(self):
        draft_path, form_path, ts_path, _, original = self.approved_files()
        for mutation in (
            lambda t: t["parents"][0]["timestamp_source"].update(kind="synthetic_frame_div_fps"),
            lambda t: t["parents"][0]["timestamps"].pop(),
            lambda t: t["parents"][0]["timestamps"].__setitem__(90, 11.0),
            lambda t: t["parents"][0]["timestamp_source"].update(sha256="changed"),
        ):
            changed = copy.deepcopy(original)
            mutation(changed)
            self.write(ts_path, changed)
            with self.assertRaises(ValueError):
                review.validate_review(draft_path, form_path, ts_path)

    def test_unresolved_issues_cannot_be_silently_dropped(self):
        (self.results / "w1.json").unlink()
        draft_path, form_path, ts_path, form, _ = self.approved_files()
        for row in form["parents"][0]["intervals"]:
            row["content"] = "Fold the shirt after inspecting the source video."
        form["parents"][0]["issue_resolutions"][0]["resolution"] = ""
        self.write(form_path, form)
        with self.assertRaisesRegex(ValueError, "Unresolved source issues"):
            review.validate_review(draft_path, form_path, ts_path)

    def test_cli_refuses_missing_timestamp_input_and_writes_no_output(self):
        output = self.root / "must-not-exist.json"
        result = subprocess.run(
            [sys.executable, "-B", str(Path(review.__file__)), "--validate-reviewed", "--output", str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("actual parquet", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
