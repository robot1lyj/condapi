# Standard-library unittest entrypoint is intentional.
# ruff: noqa: PT009, PT027, PLC0415
"""Small PyArrow protocol fixtures; no model imports, video decode or training."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "materializer_under_test", Path(__file__).with_name("materialize_fineart_language.py")
)
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("PyArrow is required for Parquet language materialization")
        self.pa, self.pq = pa, pq
        directory = tempfile.TemporaryDirectory(prefix="fineart-materialize-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def fixture(self, *, existing_language=False):
        pa, pq = self.pa, self.pq
        length = 100
        timestamps = pa.array([index / 30 for index in range(length)], type=pa.float32())
        self.times = timestamps.to_pylist()
        table = pa.table(
            {
                "timestamp": timestamps,
                "frame_index": pa.array(range(length), type=pa.int64()),
                "episode_index": pa.array([12] * length, type=pa.int64()),
                "task_index": pa.array([7] * length, type=pa.int32()),
                "action": pa.array(
                    [[index + joint / 100 for joint in range(14)] for index in range(length)],
                    type=pa.list_(pa.float64(), 14),
                ),
                "observation.state": pa.array(
                    [[index - joint / 100 for joint in range(14)] for index in range(length)],
                    type=pa.list_(pa.float32(), 14),
                ),
                "observation.images.top_rgb": pa.array(
                    [{"path": "original-video.mp4", "bytes": b"unchanged\x00image"}] * length,
                    type=pa.struct([pa.field("path", pa.string()), pa.field("bytes", pa.binary())]),
                ),
            }
        ).replace_schema_metadata({b"source_dataset_metadata": b"preserved; not a dataset publication"})
        if existing_language:
            persistent, events = materializer.official_language_types(pa)
            old = {
                "role": "assistant",
                "content": "Old annotation.",
                "style": "subtask",
                "timestamp": self.times[0],
                "camera": None,
                "tool_calls": None,
            }
            event = {key: value for key, value in old.items() if key != "timestamp"}
            table = table.append_column(
                pa.field("language_persistent", persistent, metadata={b"owner": b"source-persistent"}),
                materializer.nested_array([[old]] * length, persistent, pa),
            )
            table = table.append_column(
                pa.field("language_events", events, metadata={b"owner": b"source-events"}),
                materializer.nested_array([[event]] * length, events, pa),
            )
        self.source = self.root / "original" / "data.parquet"
        self.source.parent.mkdir()
        pq.write_table(table, self.source)
        self.source_bytes = self.source.read_bytes()
        self.source_table = pq.ParquetFile(self.source).read()

        def native(content, start):
            return {
                "role": "assistant",
                "content": content,
                "style": "subtask",
                "timestamp": self.times[start],
                "camera": None,
                "tool_calls": None,
            }

        self.view = {
            "schema_version": materializer.VIEW_SCHEMA,
            "status": "NOT_MATERIALIZED_NOT_TRAINING_READY",
            "unknown_mode": "native-empty-anchor-fallback",
            "materialized": False,
            "training_ready": False,
            "fallback_policy": {
                "max_parent_unknown_fraction": 0.02,
                "max_contiguous_unknown_frames": 30,
                "language_persistent": [],
                "language_events": [],
                "materialization_required": True,
            },
            "materialization_contract": {
                "write_both_language_columns_for_every_parent_frame": True,
                "null_or_empty_content_tombstones_allowed": False,
                "preserve_all_parent_video_action_frames_and_H50": True,
            },
            "coverage": {"action_frames": 100, "valid_language_frames": 98, "empty_language_anchor_frames": 2},
            "parents": [
                {
                    "pool": "train",
                    "source_episode_index": 12,
                    "length": length,
                    "fps": 30,
                    "action_frames": 100,
                    "valid_language_frames": 98,
                    "unknown_anchor_frames": 2,
                    "materialization_required": True,
                    "training_ready": False,
                    "timestamp_source": {
                        "kind": "actual_parquet_timestamp_column",
                        "column": "timestamp",
                        "path": str(self.source),
                        "sha256": materializer.digest(self.source),
                    },
                    "valid_language_rows": [
                        {
                            "from_frame": 0,
                            "to_frame": 50,
                            "language_persistent": [native("Spread the shirt.", 0)],
                            "language_events": [],
                        },
                        {
                            "from_frame": 52,
                            "to_frame": 100,
                            "language_persistent": [native("Fold the sleeve.", 52)],
                            "language_events": [],
                        },
                    ],
                    "empty_language_anchor_intervals": [
                        {"from_frame": 50, "to_frame": 52, "language_persistent": [], "language_events": []}
                    ],
                }
            ],
        }
        self.view_path = self.root / "view.json"
        self.save_view()
        return table

    def save_view(self):
        self.view_path.write_text(json.dumps(self.view, indent=2) + "\n")

    def replace_source_fixture(self, table):
        self.pq.write_table(table, self.source)
        self.source_bytes = self.source.read_bytes()
        self.source_table = self.pq.ParquetFile(self.source).read()
        self.times = self.source_table["timestamp"].to_pylist()
        parent = self.view["parents"][0]
        parent["timestamp_source"]["sha256"] = materializer.digest(self.source)
        for interval in parent["valid_language_rows"]:
            interval["language_persistent"][0]["timestamp"] = self.times[interval["from_frame"]]
        self.save_view()

    def materialize(self, output="materialized"):
        return materializer.materialize(self.view_path, self.root / output)

    def test_missing_language_columns_use_fixed_native_schema_and_materialize_actual_empty_lists(self):
        self.fixture()
        result = self.materialize()
        self.assertEqual(result["status"], "LANGUAGE_COLUMNS_MATERIALIZED_NOT_DATASET_RELEASED")
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["dataset_release"])
        parent = result["parents"][0]
        path = self.root / "materialized" / parent["output_parquet"]["path"]
        restored = self.pq.ParquetFile(path).read()
        persistent, events = restored["language_persistent"].to_pylist(), restored["language_events"].to_pylist()
        timeline = [self.view["parents"][0]["valid_language_rows"][index]["language_persistent"][0] for index in (0, 1)]
        self.assertEqual(persistent[0], timeline)
        self.assertEqual(persistent[49], timeline)
        self.assertEqual(persistent[50:52], [[], []])
        self.assertEqual(persistent[52], timeline)
        self.assertEqual(persistent[-1], timeline)
        self.assertEqual(events, [[] for _ in range(100)])
        self.assertEqual(restored["language_persistent"].null_count, 0)
        self.assertEqual(restored["language_events"].null_count, 0)
        self.assertEqual(
            restored.schema.field("language_persistent").type.value_type.field("timestamp").type, self.pa.float32()
        )
        self.assertNotIn("timestamp", restored.schema.field("language_events").type.value_type.names)
        self.assertEqual(
            restored.schema.field("language_persistent").type.value_type.field("camera").type, self.pa.string()
        )
        self.assertTrue(
            self.pa.types.is_list(restored.schema.field("language_persistent").type.value_type.field("tool_calls").type)
        )
        self.assertEqual(parent["frame_audit"]["language_anchor_frames"], 98)
        self.assertTrue(parent["frame_audit"]["normal_anchors_keep_full_timeline"])
        self.assertEqual(parent["output_parquet"]["sha256"], materializer.digest(path))
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        for name in self.source_table.column_names:
            self.assertTrue(restored[name].equals(self.source_table[name]), name)
        self.assertEqual(restored.schema.metadata, self.source_table.schema.metadata)
        self.assertEqual(result["view_source"]["sha256"], materializer.digest(self.view_path))

    def test_existing_language_arrow_fields_are_reused_and_all_events_cleared(self):
        self.fixture(existing_language=True)
        result = self.materialize()
        parent = result["parents"][0]
        restored = self.pq.ParquetFile(self.root / "materialized" / parent["output_parquet"]["path"]).read()
        for name in materializer.LANGUAGE_COLUMNS:
            self.assertEqual(restored.schema.field(name), self.source_table.schema.field(name))
            self.assertEqual(parent["schema_source"][name], "source_arrow_schema")
        self.assertEqual(restored["language_events"].to_pylist(), [[] for _ in range(100)])
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_inferred_null_camera_and_tools_are_upgraded_only_in_new_language_columns(self):
        self.fixture()
        old = {
            "role": "assistant",
            "content": "Old source label.",
            "style": "subtask",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        }
        events = {key: value for key, value in old.items() if key != "timestamp"}
        table = self.source_table.append_column("language_persistent", self.pa.array([[old]] * 100))
        table = table.append_column("language_events", self.pa.array([[events]] * 100))
        self.assertEqual(table.schema.field("language_persistent").type.value_type.field("camera").type, self.pa.null())
        self.replace_source_fixture(table)
        result = self.materialize()
        parent = result["parents"][0]
        restored = self.pq.ParquetFile(self.root / "materialized" / parent["output_parquet"]["path"]).read()
        target = materializer.official_language_types(self.pa)
        self.assertTrue(parent["schema_upgraded"])
        for index, name in enumerate(materializer.LANGUAGE_COLUMNS):
            self.assertEqual(restored.schema.field(name).type, target[index])
            self.assertTrue(parent["language_schema_audit"][name]["schema_upgraded"])
            self.assertIn("null", parent["language_schema_audit"][name]["old_type"])
        for name in self.source_table.column_names:
            if name not in materializer.LANGUAGE_COLUMNS:
                self.assertTrue(restored[name].equals(self.source_table[name]), name)
        self.assertEqual(restored["language_persistent"].to_pylist()[50:52], [[], []])
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_float64_language_timestamp_schema_is_upgraded_to_fixed_float32(self):
        self.fixture()
        persistent, events = materializer.official_language_types(self.pa)
        row_type = self.pa.struct(
            [
                self.pa.field(
                    field.name, self.pa.float64() if field.name == "timestamp" else field.type, nullable=field.nullable
                )
                for field in persistent.value_type
            ]
        )
        old_type = self.pa.list_(row_type)
        old = {
            "role": "assistant",
            "content": "Old source label.",
            "style": "subtask",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        }
        table = self.source_table.append_column(
            "language_persistent", materializer.nested_array([[old]] * 100, old_type, self.pa)
        )
        table = table.append_column("language_events", materializer.nested_array([[]] * 100, events, self.pa))
        self.replace_source_fixture(table)
        parent = self.materialize()["parents"][0]
        restored = self.pq.ParquetFile(self.root / "materialized" / parent["output_parquet"]["path"]).read()
        self.assertTrue(parent["language_schema_audit"]["language_persistent"]["schema_upgraded"])
        self.assertIn("timestamp: double", parent["language_schema_audit"]["language_persistent"]["old_type"])
        self.assertEqual(restored.schema.field("language_persistent").type, persistent)
        self.assertTrue(restored["timestamp"].equals(self.source_table["timestamp"]))

    def test_float32_onset_cast_that_changes_active_mapping_still_blocks(self):
        self.fixture()
        index = self.source_table.schema.get_field_index("timestamp")
        table = self.source_table.set_column(
            index, "timestamp", self.pa.array([frame / 30 for frame in range(100)], type=self.pa.float64())
        )
        self.replace_source_fixture(table)
        with self.assertRaisesRegex(ValueError, "wrong unique active instruction"):
            self.materialize()
        self.assertFalse((self.root / "materialized" / "manifest.json").exists())
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_existing_output_directory_is_never_overwritten_or_adopted(self):
        self.fixture()
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "foreign.txt"
        sentinel.write_text("retain me")
        with self.assertRaisesRegex(ValueError, "new-directory only"):
            self.materialize("existing")
        self.assertEqual(sentinel.read_text(), "retain me")
        self.assertFalse((output / "ownership.json").exists())

    def test_source_hash_change_is_rejected_and_no_final_manifest_claimed(self):
        self.fixture()
        self.source.write_bytes(b"changed source artifact")
        with self.assertRaisesRegex(ValueError, "Source Parquet SHA-256 changed"):
            self.materialize()
        self.assertFalse((self.root / "materialized" / "manifest.json").exists())
        self.assertEqual(self.source.read_bytes(), b"changed source artifact")

    def test_view_gap_or_overlap_is_rejected_without_changing_source(self):
        self.fixture()
        original = copy.deepcopy(self.view)
        for name, boundary in (("gap", 53), ("overlap", 51)):
            with self.subTest(name=name):
                self.view = copy.deepcopy(original)
                self.view["parents"][0]["valid_language_rows"][1]["from_frame"] = boundary
                self.view["parents"][0]["valid_language_rows"][1]["language_persistent"][0]["timestamp"] = self.times[
                    boundary
                ]
                self.save_view()
                with self.assertRaisesRegex(ValueError, "overlap|cover every"):
                    self.materialize(name)
                self.assertFalse((self.root / name / "manifest.json").exists())
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_null_or_empty_content_tombstones_are_rejected(self):
        self.fixture()
        original = copy.deepcopy(self.view)
        for index, tombstone in enumerate((None, "", "UNKNOWN")):
            with self.subTest(tombstone=tombstone):
                self.view = copy.deepcopy(original)
                self.view["parents"][0]["valid_language_rows"][0]["language_persistent"][0]["content"] = tombstone
                self.save_view()
                with self.assertRaisesRegex(ValueError, "placeholder"):
                    self.materialize(f"bad-content-{index}")
        self.view = copy.deepcopy(original)
        self.view["parents"][0]["empty_language_anchor_intervals"][0]["language_persistent"] = None
        self.save_view()
        with self.assertRaisesRegex(ValueError, "both language columns"):
            self.materialize("bad-empty-column")

    def test_language_timestamp_must_equal_actual_source_column(self):
        self.fixture()
        self.view["parents"][0]["valid_language_rows"][1]["language_persistent"][0]["timestamp"] += 1 / 30
        self.save_view()
        with self.assertRaisesRegex(ValueError, "actual source start frame"):
            self.materialize()
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_aggregate_counts_are_checked_against_materialized_frames(self):
        self.fixture()
        self.view["coverage"]["valid_language_frames"] = 100
        self.save_view()
        with self.assertRaisesRegex(ValueError, "aggregate counts"):
            self.materialize()
        self.assertFalse((self.root / "materialized" / "manifest.json").exists())
        failure = json.loads((self.root / "materialized" / "failure.json").read_text())
        self.assertEqual(failure["status"], "MATERIALIZATION_INCOMPLETE")
        self.assertFalse(failure["training_ready"])


if __name__ == "__main__":
    unittest.main()
