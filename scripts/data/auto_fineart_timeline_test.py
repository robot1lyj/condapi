# Standard-library unittest entrypoint is intentional.
# ruff: noqa: PT009, PT027
"""Synthetic CPU source/decision protocol fixtures; no decoder/model/training."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "auto_timeline_under_test", Path(__file__).with_name("auto_fineart_timeline.py")
)
auto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto)
SELECTOR = {
    "model_id": "Qwen/Qwen3.5-9B",
    "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "implementation_version": "test-selector-v1",
    "prompt_sha256": "b" * 64,
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def segment(start, end, caption, status="complete"):
    return {"start_s": start / 30, "end_s": end / 30, "caption": caption, "status": status}


class AutomaticTimelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="auto-fineart-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, episodes):
        """Use the real native draft builder and its actual-file hash bindings."""
        dataset, annotation, results = [self.root / value for value in ("dataset", "annotation", "results")]
        for path in (dataset, annotation, results):
            path.mkdir()
        records, windows, times = [], [], []
        for eid, length, segments in episodes:
            videos, camera_receipts = {}, {}
            directory = dataset / "raw" / str(eid)
            directory.mkdir(parents=True)
            window_id = f"window-{eid}"
            for camera in auto.CAMERAS:
                parent = directory / (camera + ".mp4")
                parent.write_bytes(f"synthetic-parent:{eid}:{camera}".encode())
                videos[camera] = {
                    "artifact": {"path": str(parent.relative_to(dataset)), "sha256": auto.review.digest(parent)},
                    "offset_s": 0.0,
                    "metrics": {"frames": length},
                }
                clip = annotation / window_id / (camera + ".mp4")
                clip.parent.mkdir(exist_ok=True)
                clip.write_bytes(f"synthetic-window:{eid}:{camera}".encode())
                camera_receipts[camera] = {
                    "path": str(clip.relative_to(annotation)),
                    "sha256": auto.review.digest(clip),
                    "frames": length,
                }
            parquet = directory / "data.parquet"
            # The existing exporter verifies a real file's SHA and explicit
            # provenance; decoding actual Parquet is outside this stdlib test.
            parquet.write_bytes(f"synthetic-parquet-protocol:{eid}".encode())
            record = {
                "pool": "train",
                "source_episode_index": eid,
                "length": length,
                "fps": 30,
                "status": "PREPARED_REVIEW_PENDING",
                "videos": videos,
                "task": "fold and stack the t-shirts",
                "source_repo": "lerobot/abc_130k_v3_train",
                "source_revision": "68651e4929d9fb00f798937b2d62617cab5c771d",
                "data": {"path": str(parquet.relative_to(dataset)), "sha256": auto.review.digest(parquet)},
            }
            records.append(record)
            receipt = {
                "id": window_id,
                "pool": "train",
                "source_episode_index": eid,
                "fps": 30,
                "parent_record_sha256": auto.hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
                "parent_from_frame": 0,
                "parent_to_frame": length,
                "core_from_frame": 0,
                "core_to_frame": length,
                "local_to_parent_offset_s": 0.0,
                "training_episode": False,
                "cameras": camera_receipts,
            }
            windows.append(receipt)
            native = {
                "schema_version": "flora_demo_annotation_v1",
                "duration_s": length / 30,
                "profile": "YAM",
                "camera_roles": ["main", "left_wrist", "right_wrist"],
                "input_sha256": {
                    "main": camera_receipts["top"]["sha256"],
                    "left_wrist": camera_receipts["left_wrist"]["sha256"],
                    "right_wrist": camera_receipts["right_wrist"]["sha256"],
                },
                "model_revisions": {"tempseg": {"revision": "frozen"}},
                "segments": segments,
            }
            write(results / (window_id + ".json"), native)
            write(results / (window_id + ".tempseg.json"), {"stage": "tempseg", "segments": segments})
            times.append(
                {
                    "pool": "train",
                    "source_episode_index": eid,
                    "fps": 30,
                    "timestamps": [0.125 + index / 30 for index in range(length)],
                    "timestamp_source": {
                        "kind": "actual_parquet_timestamp_column",
                        "column": "timestamp",
                        "path": str(parquet.relative_to(self.root)),
                        "sha256": auto.review.digest(parquet),
                    },
                }
            )
        self.metadata = dataset / "audit-frozen.json"
        self.lineage = annotation / "lineage.json"
        self.results = results
        write(self.metadata, {"parents": records})
        write(self.lineage, {"parent_episodes": len(records), "annotation_windows": len(windows), "windows": windows})
        draft, _ = auto.review.make_draft(self.lineage, self.metadata, results)
        self.draft = self.root / "draft.json"
        write(self.draft, draft)
        self.plan_path = self.root / "plan.json"
        self.plan = auto.make_plan(self.draft)
        write(self.plan_path, self.plan)
        self.timestamps = self.root / "timestamps.json"
        write(self.timestamps, {"schema_version": 1, "parents": times})
        return self.plan

    def decisions(self, selections=None):
        choices = []
        for index, task in enumerate(self.plan["tasks"]):
            selected = (selections or {}).get(task["task_id"], "UNKNOWN")
            evidence = []
            for number, frame in enumerate(task["evidence"]["frames"]):
                image = self.root / "evidence" / f"task-{index}-frame-{number}.png"
                image.parent.mkdir(exist_ok=True)
                image.write_bytes(("synthetic-image-protocol:" + frame["frame_id"]).encode())
                evidence.append(
                    {"frame_id": frame["frame_id"], "path": str(image), "sha256": auto.review.digest(image)}
                )
            core = {
                "task_id": task["task_id"],
                "atom_id": task["atom_id"],
                "task_sha256": task["task_sha256"],
                "selected_id": selected,
                "input_video_sha256": {camera: task["evidence"]["videos"][camera]["sha256"] for camera in auto.CAMERAS},
            }
            receipt = self.root / "receipts" / f"task-{index}.json"
            write(receipt, dict(**core, selector=SELECTOR, evidence=evidence))
            choices.append(
                dict(
                    **core,
                    evidence_frame_ids=task["evidence_frame_ids"],
                    inference_receipt={"path": str(receipt), "sha256": auto.review.digest(receipt)},
                )
            )
        path = self.root / "decisions.json"
        write(
            path,
            {
                "schema_version": auto.DECISION_SCHEMA,
                "task_manifest_sha256": auto.review.digest(self.plan_path),
                "selector": SELECTOR,
                "decisions": choices,
            },
        )
        return path

    def conflict_fixture(self):
        return self.fixture([(1, 120, [segment(0, 90, "Lay out the shirt."), segment(30, 120, "Fold the sleeve.")])])

    def rebuild(self, policy_file=None):
        draft, _ = auto.review.make_draft(self.lineage, self.metadata, self.results)
        write(self.draft, draft)
        self.plan = auto.make_plan(self.draft, policy_file or auto.DEFAULT_POLICY)
        write(self.plan_path, self.plan)

    def seam_fixture(self):
        self.fixture([(1, 120, [segment(0, 120, "Lay out the shirt.")])])
        original = json.loads(self.lineage.read_text())["windows"][0]
        template = json.loads((self.results / "window-1.json").read_text())
        windows = []
        for name, start, end, core_start, core_end, caption in (
            ("seam-left", 0, 90, 0, 60, "Lay out the shirt."),
            ("seam-right", 30, 120, 60, 120, "Fold the sleeve."),
        ):
            cameras = {}
            for camera in auto.CAMERAS:
                path = self.lineage.parent / name / (camera + ".mp4")
                path.parent.mkdir(exist_ok=True)
                path.write_bytes((name + camera).encode())
                cameras[camera] = {
                    "path": str(path.relative_to(self.lineage.parent)),
                    "sha256": auto.review.digest(path),
                    "frames": end - start,
                }
            windows.append(
                {
                    **original,
                    "id": name,
                    "parent_from_frame": start,
                    "parent_to_frame": end,
                    "core_from_frame": core_start,
                    "core_to_frame": core_end,
                    "local_to_parent_offset_s": start / 30,
                    "cameras": cameras,
                }
            )
            native = {
                **template,
                "duration_s": (end - start) / 30,
                "segments": [segment(0, end - start, caption)],
                "input_sha256": {
                    "main": cameras["top"]["sha256"],
                    "left_wrist": cameras["left_wrist"]["sha256"],
                    "right_wrist": cameras["right_wrist"]["sha256"],
                },
            }
            write(self.results / (name + ".json"), native)
        write(self.lineage, {"parent_episodes": 1, "annotation_windows": 2, "windows": windows})
        self.rebuild()

    def test_cross_window_disagreement_routes_seam_atoms_and_preserves_real_changes(self):
        self.seam_fixture()
        self.assertEqual([(task["from_frame"], task["to_frame"]) for task in self.plan["tasks"]], [(30, 60), (60, 90)])
        selections = {}
        for task in self.plan["tasks"]:
            self.assertIn("WINDOW_SEAM", task["reason_codes"])
            self.assertEqual(task["seam_boundaries"], [60])
            caption = "Lay out the shirt." if task["from_frame"] < 60 else "Fold the sleeve."
            selections[task["task_id"]] = next(
                candidate["candidate_id"] for candidate in task["candidates"] if candidate["caption"] == caption
            )
        application = auto.apply_decisions(self.plan_path, self.decisions(selections))
        intervals = application["parents"][0]["intervals"]
        self.assertEqual(
            [(row["from_frame"], row["to_frame"], row["content"]) for row in intervals],
            [(0, 60, "Lay out the shirt."), (60, 120, "Fold the sleeve.")],
        )

    def test_same_caption_across_core_boundary_does_not_create_a_training_subtask_cut(self):
        self.seam_fixture()
        path = self.results / "seam-right.json"
        native = json.loads(path.read_text())
        native["segments"][0]["caption"] = " Lay   out the shirt. "
        write(path, native)
        self.rebuild()
        self.assertEqual(self.plan["tasks"], [])
        application = auto.apply_decisions(self.plan_path)
        intervals = application["parents"][0]["intervals"]
        self.assertEqual(len(intervals), 1)
        self.assertEqual((intervals[0]["from_frame"], intervals[0]["to_frame"]), (0, 120))
        self.assertEqual(intervals[0]["source_label_ids"], ["seam-left:s00000", "seam-right:s00000"])

    def test_actual_policy_is_loaded_hash_bound_and_unsupported_changes_rejected(self):
        self.conflict_fixture()
        path = self.root / "policy.json"
        policy = json.loads(auto.DEFAULT_POLICY.read_text())
        write(path, policy)
        self.rebuild(path)
        self.assertEqual(self.plan["policy_contract"], policy)
        self.assertEqual(self.plan["policy_source"]["sha256"], auto.review.digest(path))
        self.assertEqual(self.plan["policy"]["neighbor_frames"], policy["candidates"]["neighbor_max_distance_frames"])
        for group, field, value in (
            ("candidates", "neighbor_max_distance_frames", 149),
            ("candidates", "neighbor_max_distinct_captions", 3),
            ("evidence", "context_frames_each_side", 15),
            ("candidates", "include_UNKNOWN", False),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(policy)
                changed[group][field] = value
                write(path, changed)
                with self.assertRaises(ValueError):
                    auto.make_plan(self.draft, path)
        changed = copy.deepcopy(policy)
        changed["human_approval_required"] = True
        write(path, changed)
        with self.assertRaisesRegex(ValueError, "human/user approval"):
            auto.make_plan(self.draft, path)
        changed = copy.deepcopy(policy)
        changed["status"] = "NEW_POLICY_VERSION"
        write(path, changed)
        with self.assertRaisesRegex(ValueError, "Plan/source/policy version changed"):
            auto.apply_decisions(self.plan_path)

    def test_known_invalid_source_header_cannot_be_resolved_by_label_selection(self):
        self.fixture([(1, 120, [segment(0, 120, "Fold the shirt.")])])
        path = self.results / "window-1.json"
        native = json.loads(path.read_text())
        native["input_sha256"]["main"] = "0" * 64
        write(path, native)
        self.rebuild()
        self.assertEqual(self.plan["status"], "BLOCKED_SOURCE_CONTRACT")
        application = auto.apply_decisions(self.plan_path)
        self.assertEqual(application["parents"], [])
        self.assertIn(
            "RESULT_INPUT_HASH_MISMATCH", [issue["code"] for issue in application["pending_parents"][0]["hard_issues"]]
        )

    def test_whitespace_equivalent_overlap_is_deduplicated_and_exported_automatically(self):
        self.fixture(
            [
                (
                    1,
                    120,
                    [
                        segment(0, 60, "Pick up shirt."),
                        segment(30, 90, " Pick   up shirt. "),
                        segment(90, 120, "Fold shirt."),
                    ],
                )
            ]
        )
        self.assertEqual(self.plan["tasks"], [])
        atoms = self.plan["parents"][0]["atoms"]
        self.assertEqual(
            [(atom["from_frame"], atom["to_frame"]) for atom in atoms], [(0, 30), (30, 60), (60, 90), (90, 120)]
        )
        application = auto.apply_decisions(self.plan_path)
        self.assertEqual(application["review_method"], "automatic_candidate_selector")
        self.assertEqual(len(application["parents"][0]["intervals"]), 2)
        self.assertEqual(
            application["parents"][0]["intervals"][0]["source_label_ids"], ["window-1:s00000", "window-1:s00001"]
        )
        result = auto.export_resolved(self.plan_path, None, self.timestamps, self.root / "language.json")
        self.assertEqual(result["status"], "LANGUAGE_ROWS_VALIDATED_DATASET_RELEASE_PENDING")
        rows = result["parents"][0]["language_persistent"]
        self.assertEqual([row["timestamp"] for row in rows], [0.125, 3.125])
        self.assertTrue(
            all(
                row["role"] == "assistant"
                and row["style"] == "subtask"
                and row["camera"] is None
                and row["tool_calls"] is None
                for row in rows
            )
        )
        self.assertFalse(result["training_ready"])

    def test_boundary_id_regression_resolves_reset_conflicts_and_gap_without_fold_prior(self):
        bounds = [
            (0, 150),
            (150, 495),
            (345, 495),
            (495, 615),
            (615, 720),
            (615, 885),
            (885, 1050),
            (1065, 1275),
            (1275, 1434),
        ]
        self.fixture([(128691, 1434, [segment(a, b, f"Action {index}.") for index, (a, b) in enumerate(bounds)])])
        atoms = self.plan["parents"][0]["atoms"]
        self.assertEqual(
            [atom["from_frame"] for atom in atoms] + [atoms[-1]["to_frame"]],
            [0, 150, 345, 495, 615, 720, 885, 1050, 1065, 1275, 1434],
        )
        expected = {345: 2, 615: 4, 1050: 7}
        selections = {}
        for task in self.plan["tasks"]:
            source = f"window-128691:s{expected[task['from_frame']]:05d}"
            selections[task["task_id"]] = next(
                candidate["candidate_id"] for candidate in task["candidates"] if source in candidate["source_label_ids"]
            )
            self.assertEqual(task["task_sha256"], auto.task_sha(task))
            self.assertEqual(task["evidence_frame_ids"], [frame["frame_id"] for frame in task["evidence"]["frames"]])
        self.assertEqual(len(selections), 3)
        result = auto.export_resolved(
            self.plan_path, self.decisions(selections), self.timestamps, self.root / "language.json"
        )
        self.assertEqual(len(result["parents"][0]["language_persistent"]), 9)
        reviewed = json.loads((self.root / "language.reviewed.json").read_text())
        self.assertEqual(
            [row["content"] for row in reviewed["parents"][0]["intervals"]], [f"Action {index}." for index in range(9)]
        )
        repaired_gap = reviewed["parents"][0]["intervals"][7]
        self.assertEqual((repaired_gap["from_frame"], repaired_gap["to_frame"]), (1050, 1275))
        self.assertIn('"quality": "VLM_SELECTED_CANDIDATE"', repaired_gap["note"])

    def test_unknown_and_missing_decisions_remain_local_pending_without_forward_fill(self):
        self.conflict_fixture()
        missing = auto.apply_decisions(self.plan_path)
        self.assertEqual(missing["parents"], [])
        self.assertEqual(missing["pending_parents"][0]["local_selection_tasks"][0]["reason"], "MISSING_DECISION")
        result = auto.export_resolved(self.plan_path, self.decisions(), self.timestamps, self.root / "pending.json")
        self.assertEqual(result["status"], "PENDING_LOCAL_SELECTION")
        self.assertNotIn("language_persistent", json.dumps(result))
        self.assertEqual(result["pending_parents"][0]["local_selection_tasks"][0]["reason"], "UNKNOWN")
        self.assertFalse((self.root / "pending.reviewed.json").exists())

    def test_only_complete_resolved_parents_export_with_explicit_subset_scope(self):
        self.fixture(
            [
                (1, 120, [segment(0, 120, "Fold the shirt.")]),
                (2, 120, [segment(0, 45, "Lift the shirt."), segment(75, 120, "Fold the shirt.")]),
            ]
        )
        result = auto.export_resolved(self.plan_path, self.decisions(), self.timestamps, self.root / "subset.json")
        self.assertEqual([parent["source_episode_index"] for parent in result["parents"]], [1])
        self.assertEqual(result["automatic_scope"]["included_parents"], [["train", 1]])
        self.assertEqual(result["automatic_scope"]["pending_parents"], [["train", 2]])
        scoped = json.loads((self.root / "subset.scope.draft.json").read_text())
        self.assertEqual(len(scoped["parents"]), 1)
        queue = json.loads((self.root / "subset.queue.json").read_text())
        self.assertEqual(queue["pending_parents"][0]["length"], 120)

    def test_real_timestamp_input_and_original_parquet_binding_are_required(self):
        self.fixture([(1, 120, [segment(0, 120, "Fold the shirt.")])])
        result = auto.export_resolved(self.plan_path, None, None, self.root / "no-times.json")
        self.assertEqual(result["status"], "RESOLVED_TIMELINE_TIMESTAMP_EXPORT_PENDING")
        timestamps = json.loads(self.timestamps.read_text())
        fake = self.root / "other.parquet"
        fake.write_bytes(b"different source")
        timestamps["parents"][0]["timestamp_source"].update(path=str(fake), sha256=auto.review.digest(fake))
        write(self.timestamps, timestamps)
        with self.assertRaisesRegex(ValueError, "original parent parquet"):
            auto.export_resolved(self.plan_path, None, self.timestamps, self.root / "bad-times.json")
        self.assertFalse((self.root / "bad-times.json").exists())

    def test_source_camera_and_native_result_tampering_are_rejected(self):
        self.conflict_fixture()
        camera = Path(self.plan["tasks"][0]["evidence"]["videos"]["top"]["path"])
        original = camera.read_bytes()
        camera.write_bytes(b"different camera source")
        with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
            auto.apply_decisions(self.plan_path)
        camera.write_bytes(original)
        result = self.results / "window-1.json"
        native = json.loads(result.read_text())
        native["segments"][0]["caption"] = "Different native output."
        write(result, native)
        with self.assertRaisesRegex(ValueError, "source reconstruction"):
            auto.apply_decisions(self.plan_path)

    def test_plan_boundaries_cannot_be_edited_even_with_recomputed_task_hash(self):
        self.conflict_fixture()
        changed = copy.deepcopy(self.plan)
        changed["tasks"][0]["from_frame"] += 1
        changed["tasks"][0]["task_sha256"] = auto.task_sha(changed["tasks"][0])
        write(self.plan_path, changed)
        with self.assertRaisesRegex(ValueError, "Plan/source/policy version changed"):
            auto.apply_decisions(self.plan_path)

    def test_unoffered_candidate_free_time_or_wrong_frame_id_is_rejected(self):
        self.conflict_fixture()
        path = self.decisions()
        original = json.loads(path.read_text())
        for kind in ("candidate", "time", "frame", "task", "manifest"):
            with self.subTest(kind=kind):
                document = copy.deepcopy(original)
                decision = document["decisions"][0]
                if kind == "candidate":
                    decision["selected_id"] = "C999"
                elif kind == "time":
                    decision["from_frame"] = 1
                elif kind == "frame":
                    decision["evidence_frame_ids"][0] = "unplanned-frame"
                elif kind == "task":
                    decision["task_sha256"] = "0" * 64
                else:
                    document["task_manifest_sha256"] = "0" * 64
                write(path, document)
                with self.assertRaises(ValueError):
                    auto.apply_decisions(self.plan_path, path)

    def test_actual_evidence_image_and_selector_version_are_checked(self):
        self.conflict_fixture()
        path = self.decisions()
        document = json.loads(path.read_text())
        receipt = Path(document["decisions"][0]["inference_receipt"]["path"])
        payload = json.loads(receipt.read_text())
        image = Path(payload["evidence"][0]["path"])
        image.write_bytes(b"different image payload")
        with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
            auto.apply_decisions(self.plan_path, path)
        image.write_bytes(("synthetic-image-protocol:" + payload["evidence"][0]["frame_id"]).encode())
        document["selector"]["model_revision"] = "c" * 40
        write(path, document)
        with self.assertRaisesRegex(ValueError, "Selector model/revision|selector version differs"):
            auto.apply_decisions(self.plan_path, path)

    def test_receipt_selection_mismatch_is_rejected_after_rehash(self):
        self.conflict_fixture()
        path = self.decisions()
        document = json.loads(path.read_text())
        receipt = Path(document["decisions"][0]["inference_receipt"]["path"])
        payload = json.loads(receipt.read_text())
        payload["selected_id"] = self.plan["tasks"][0]["candidates"][0]["candidate_id"]
        write(receipt, payload)
        document["decisions"][0]["inference_receipt"]["sha256"] = auto.review.digest(receipt)
        write(path, document)
        with self.assertRaisesRegex(ValueError, "receipt selected_id mismatch"):
            auto.apply_decisions(self.plan_path, path)

    def test_short_unknown_fallback_preserves_all_actions_and_requires_materialization(self):
        bounds = [
            (0, 150),
            (150, 495),
            (345, 495),
            (495, 615),
            (615, 720),
            (615, 885),
            (885, 1050),
            (1065, 1275),
            (1275, 1434),
        ]
        self.fixture([(128691, 1434, [segment(a, b, f"Action {index}.") for index, (a, b) in enumerate(bounds)])])
        selections = {}
        for task in self.plan["tasks"]:
            if task["from_frame"] in (345, 615):
                ordinal = 2 if task["from_frame"] == 345 else 4
                ref = f"window-128691:s{ordinal:05d}"
                selections[task["task_id"]] = next(
                    candidate["candidate_id"]
                    for candidate in task["candidates"]
                    if ref in candidate["source_label_ids"]
                )
        decisions = self.decisions(selections)
        self.assertEqual(auto.apply_decisions(self.plan_path, decisions)["parents"], [])
        result = auto.export_resolved(
            self.plan_path, decisions, self.timestamps, self.root / "fallback.json", "native-empty-anchor-fallback"
        )
        self.assertEqual(result["status"], "NOT_MATERIALIZED_NOT_TRAINING_READY")
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["materialized"])
        parent = result["parents"][0]
        self.assertEqual(
            (parent["length"], parent["action_frames"], parent["valid_language_frames"]), (1434, 1434, 1419)
        )
        empty = parent["empty_language_anchor_intervals"]
        self.assertEqual([(row["from_frame"], row["to_frame"]) for row in empty], [(1050, 1065)])
        self.assertEqual(empty[0]["language_persistent"], [])
        self.assertEqual(empty[0]["language_events"], [])
        self.assertNotIn("content", empty[0])
        for interval in parent["valid_language_rows"]:
            self.assertTrue(interval["language_persistent"][0]["content"])
            self.assertNotIn("UNKNOWN", interval["language_persistent"][0]["content"])
        self.assertAlmostEqual(result["coverage"]["action_hours"], 1434 / 30 / 3600)
        self.assertAlmostEqual(result["coverage"]["valid_language_hours"], 1419 / 30 / 3600)
        self.assertTrue(result["materialization_contract"]["preserve_all_parent_video_action_frames_and_H50"])
        self.assertFalse(result["materialization_contract"]["null_or_empty_content_tombstones_allowed"])

    def test_fallback_accepts_exact_two_percent_and_thirty_frames(self):
        self.fixture([(1, 1500, [segment(0, 100, "Fold the shirt."), segment(130, 1500, "Fold the shirt.")])])
        result = auto.export_resolved(
            self.plan_path, self.decisions(), self.timestamps, self.root / "exact.json", "native-empty-anchor-fallback"
        )
        self.assertEqual(len(result["parents"]), 1)
        self.assertEqual(result["parents"][0]["unknown_anchor_frames"], 30)
        self.assertEqual(result["parents"][0]["unknown_fraction"], 0.02)

    def test_fallback_mixed_batch_preserves_fully_resolved_and_small_unknown_parents(self):
        self.fixture(
            [
                (1, 120, [segment(0, 120, "Fold the shirt.")]),
                (2, 1500, [segment(0, 100, "Fold the shirt."), segment(115, 1500, "Fold the shirt.")]),
            ]
        )
        result = auto.export_resolved(
            self.plan_path, self.decisions(), self.timestamps, self.root / "mixed.json", "native-empty-anchor-fallback"
        )
        self.assertEqual([parent["source_episode_index"] for parent in result["parents"]], [1, 2])
        self.assertEqual(result["parents"][0]["empty_language_anchor_intervals"], [])
        self.assertEqual(result["coverage"]["action_frames"], 1620)
        self.assertEqual(result["coverage"]["valid_language_frames"], 1605)
        self.assertEqual(result["pending_queue"]["parents"], 0)

    def test_excess_fraction_or_contiguous_run_keeps_whole_parent_pending(self):
        base = self.root
        for name, length, gap in (("fraction", 1499, 30), ("run", 2000, 31)):
            with self.subTest(name=name):
                self.root = base / name
                self.root.mkdir()
                self.fixture(
                    [(1, length, [segment(0, 100, "Fold the shirt."), segment(100 + gap, length, "Fold the shirt.")])]
                )
                result = auto.export_resolved(
                    self.plan_path,
                    self.decisions(),
                    self.timestamps,
                    self.root / "rejected.json",
                    "native-empty-anchor-fallback",
                )
                self.assertEqual(result["parents"], [])
                self.assertEqual(result["coverage"]["action_frames"], 0)
                queue = json.loads((self.root / "rejected.queue.json").read_text())
                self.assertEqual(queue["pending_parents"][0]["length"], length)
                self.assertEqual(queue["pending_parents"][0]["fallback_rejection"]["unknown_frames"], gap)

    def test_adjacent_unknown_atoms_are_one_run_before_applying_thirty_frame_limit(self):
        self.fixture(
            [
                (
                    1,
                    2000,
                    [
                        segment(0, 100, "Fold the shirt."),
                        segment(115, 120, "UNKNOWN"),
                        segment(131, 2000, "Fold the shirt."),
                    ],
                )
            ]
        )
        self.assertEqual(len(self.plan["tasks"]), 3)
        result = auto.export_resolved(
            self.plan_path, self.decisions(), self.timestamps, self.root / "runs.json", "native-empty-anchor-fallback"
        )
        self.assertEqual(result["parents"], [])
        queue = json.loads((self.root / "runs.queue.json").read_text())
        self.assertEqual(queue["pending_parents"][0]["fallback_rejection"]["max_contiguous_unknown_frames"], 31)

    def test_missing_decision_is_not_treated_as_permitted_unknown(self):
        self.fixture([(1, 1500, [segment(0, 100, "Fold the shirt."), segment(115, 1500, "Fold the shirt.")])])
        result = auto.export_resolved(
            self.plan_path, None, self.timestamps, self.root / "missing.json", "native-empty-anchor-fallback"
        )
        self.assertEqual(result["parents"], [])
        queue = json.loads((self.root / "missing.queue.json").read_text())
        self.assertEqual(queue["pending_parents"][0]["local_selection_tasks"][0]["reason"], "MISSING_DECISION")

    def test_fallback_requires_actual_timestamps_and_rejects_null_tombstone_policy(self):
        self.fixture([(1, 120, [segment(0, 120, "Fold the shirt.")])])
        with self.assertRaisesRegex(ValueError, "actual timestamps"):
            auto.export_resolved(
                self.plan_path, None, None, self.root / "no-times.json", "native-empty-anchor-fallback"
            )
        timestamps = json.loads(self.timestamps.read_text())
        timestamps["parents"][0]["timestamp_source"]["kind"] = "synthetic_frame_divided_by_fps"
        write(self.timestamps, timestamps)
        with self.assertRaisesRegex(ValueError, "Actual parquet timestamp provenance"):
            auto.export_resolved(
                self.plan_path, None, self.timestamps, self.root / "fake-times.json", "native-empty-anchor-fallback"
            )
        policy = copy.deepcopy(self.plan["policy_contract"])
        for tombstone in (None, [{"content": None}], [{"content": ""}]):
            with self.subTest(tombstone=tombstone):
                changed = copy.deepcopy(policy)
                changed["unknown_anchor_fallback"]["language_persistent"] = tombstone
                path = self.root / "bad-policy.json"
                write(path, changed)
                with self.assertRaisesRegex(ValueError, "null tombstones"):
                    auto.make_plan(self.draft, path)


if __name__ == "__main__":
    unittest.main()
