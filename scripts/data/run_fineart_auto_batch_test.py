"""Fake-subprocess protocol checks; no real models, datasets or GPU execution."""

# Keep unittest so this protocol suite needs no pytest dependency in the Flora environment.
# ruff: noqa: PT009, PT027
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import run_fineart_auto_batch as batch


class AutoBatchProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fineart-batch-protocol-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / "deployed-tools"
        self.tools.mkdir()
        for name in batch.TOOLS:
            (self.tools / name).write_text("# fake protocol tool: " + name + "\n")
        self.lineage, self.audit, self.policy = [self.root / f for f in ("lineage.json", "audit.jsonl", "policy.json")]
        self.lineage.write_text('{"windows": []}\n')
        self.audit.write_text('{"pool":"train","source_episode_index":7}\n')
        self.policy.write_text('{"schema_version":"fixture"}\n')
        self.originals = {p: p.read_bytes() for p in (self.lineage, self.audit, self.policy)}
        for name in ("results", "model", "flora"):
            (self.root / name).mkdir()
        self.args = argparse.Namespace(
            lineage=self.lineage,
            parent_audit=self.audit,
            raw_results=self.root / "results",
            policy=self.policy,
            model_dir=self.root / "model",
            flora_root=self.root / "flora",
            output_root=self.root / "out",
            tools_dir=self.tools,
            lock_file=Path("/data/.flora-annotation-gpu.lock"),
            unknown_mode="native-empty-anchor-fallback",
            skip_materialize=False,
        )
        self.calls, self.tasks, self.fail, self.include_parent = [], 1, None, True

    @staticmethod
    def value(argv, flag):
        return argv[argv.index(flag) + 1]

    def fake_process(self, argv, stdout, stderr, check):
        name = Path(argv[1]).name
        phase = "plan" if name == batch.TOOLS[1] and "--plan" in argv else ("apply" if name == batch.TOOLS[1] else name)
        self.calls.append((phase, argv))
        if phase == self.fail:
            stderr.write("injected failure\n")
            return types.SimpleNamespace(returncode=9)
        if name == batch.TOOLS[0]:
            batch.write_json(
                self.value(argv, "--output"), {"parents": [{"pool": "train", "source_episode_index": 7, "length": 3}]}
            )
        elif phase == "plan":
            batch.write_json(self.value(argv, "--output"), {"tasks": [{"task_id": str(i)} for i in range(self.tasks)]})
        elif name == batch.TOOLS[2]:
            self.assertEqual(self.value(argv, "--lock-file"), "/data/.flora-annotation-gpu.lock")
            batch.write_json(Path(self.value(argv, "--output-dir")) / "decisions.json", {"decisions": []})
        elif phase == "apply":
            self.assertEqual(self.value(argv, "--unknown-mode"), "native-empty-anchor-fallback")
            self.assertEqual("--decisions" in argv, self.tasks > 0)
            batch.write_json(
                self.value(argv, "--output"),
                {
                    "status": "NOT_MATERIALIZED_NOT_TRAINING_READY",
                    "parents": [{"pool": "train", "source_episode_index": 7}] if self.include_parent else [],
                    "pending_queue": {"parents": 0 if self.include_parent else 1},
                    "coverage": {"action_frames": 3, "valid_language_frames": 3},
                },
            )
        elif name == batch.TOOLS[3]:
            batch.write_json(
                Path(self.value(argv, "--output-root")) / "manifest.json",
                {
                    "status": "LANGUAGE_COLUMNS_MATERIALIZED_NOT_DATASET_RELEASED",
                    "training_ready": False,
                    "dataset_release": False,
                    "counts": {"parents": 1, "action_frames": 3},
                },
            )
        stdout.write("fake subprocess completed\n")
        return types.SimpleNamespace(returncode=0)

    def fake_timestamps(self, draft, audit, output):
        self.assertTrue(Path(audit).parent == self.audit.parent)
        batch.write_json(output, {"schema_version": 1, "parents": []})
        return {"protocol_fixture_only": True}

    def execute(self):
        return batch.run(self.args, runner=self.fake_process, timestamp_extractor=self.fake_timestamps)

    def test_chain_freezes_original_bases_and_records_stages_without_ready_claim(self):
        result = self.execute()
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["dataset_release"])
        self.assertEqual([x[0] for x in self.calls], [batch.TOOLS[0], "plan", batch.TOOLS[2], "apply", batch.TOOLS[3]])
        ownership = batch.load(self.args.output_root / "ownership.json")
        for key, original in (("lineage", self.lineage), ("parent_audit", self.audit)):
            snapshot = Path(ownership[key]["path"])
            self.assertEqual(snapshot.parent, original.parent)
            self.assertEqual(snapshot.read_bytes(), self.originals[original])
        self.assertTrue((self.args.output_root / "code/prepare_fineart_subtask_review.py").is_file())
        self.assertTrue((self.args.output_root / "code/data/auto_fineart_timeline.py").is_file())
        self.assertEqual(len(result["stages"]), 7)
        self.assertTrue(all(s["exit_code"] == 0 and Path(s["stdout"]).is_file() for s in result["stages"]))
        for path, content in self.originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_rules_only_skips_selector_and_same_snapshot_is_reusable(self):
        self.tasks = 0
        self.args.model_dir = self.root / "no-model-needed-for-rules"
        self.args.flora_root = self.root / "no-flora-load-needed-for-rules"
        self.execute()
        snapshots = set(self.root.glob("*.fineart-snapshot-*"))
        self.args.output_root = self.root / "another-new-output"
        result = self.execute()
        self.assertEqual(set(self.root.glob("*.fineart-snapshot-*")), snapshots)
        self.assertFalse(any(name == batch.TOOLS[2] for name, _ in self.calls))
        self.assertEqual(next(x for x in result["stages"] if x["name"] == "selector")["status"], "SKIPPED")

    def test_nonzero_selector_stops_without_apply_and_preserves_failure_logs(self):
        self.fail = batch.TOOLS[2]
        with self.assertRaisesRegex(ValueError, "exited9|exited 9"):
            self.execute()
        self.assertFalse(any(name == "apply" for name, _ in self.calls))
        self.assertFalse((self.args.output_root / "materialized").exists())
        status = batch.load(self.args.output_root / "status.json")
        self.assertEqual(status["status"], "BATCH_FAILED_ARTIFACTS_RETAINED")
        self.assertEqual(status["stages"][-1]["exit_code"], 9)
        self.assertTrue((self.args.output_root / "failure.json").is_file())
        self.assertIn("injected failure", Path(status["stages"][-1]["stderr"]).read_text())

    def test_no_eligible_parents_keeps_queue_and_skips_materializer(self):
        self.include_parent = False
        result = self.execute()
        self.assertEqual(result["pending_parent_count"], 1)
        self.assertFalse(any(name == batch.TOOLS[3] for name, _ in self.calls))
        self.assertEqual(result["status"], "ANNOTATION_VIEW_NOT_MATERIALIZED_NOT_TRAINING_READY")

    def test_skip_materialize_and_existing_output_are_explicit(self):
        self.args.skip_materialize = True
        (self.tools / batch.TOOLS[3]).unlink()
        self.execute()
        self.assertFalse(any(name == batch.TOOLS[3] for name, _ in self.calls))
        before = (self.args.output_root / "summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "must be NEW"):
            self.execute()
        self.assertEqual((self.args.output_root / "summary.json").read_bytes(), before)

    def test_snapshot_prefix_collision_is_rejected_without_overwriting(self):
        content = self.lineage.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        target = self.lineage.with_name(f"lineage.fineart-snapshot-{sha[:20]}.json")
        target.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "different bytes"):
            self.execute()
        self.assertEqual(target.read_bytes(), b"different")


class TimestampProtocolTest(unittest.TestCase):
    def test_actual_values_and_source_frame_identity_are_used(self):
        with tempfile.TemporaryDirectory(prefix="fineart-timestamp-protocol-") as directory:
            root = Path(directory)
            data = root / "raw/data.parquet"
            data.parent.mkdir()
            data.write_bytes(b"fake parquet bytes; protocol only")
            meta = {
                "pool": "train",
                "source_episode_index": 7,
                "length": 3,
                "fps": 30,
                "data": {"path": "raw/data.parquet", "sha256": batch.digest(data)},
            }
            audit = root / "audit.jsonl"
            audit.write_text(json.dumps(meta) + "\n")
            draft = root / "draft.json"
            batch.write_json(
                draft,
                {
                    "parents": [
                        {
                            "pool": "train",
                            "source_episode_index": 7,
                            "length": 3,
                            "fps": 30,
                            "windows": [
                                {
                                    "parent_record_sha256": hashlib.sha256(
                                        json.dumps(meta, sort_keys=True).encode()
                                    ).hexdigest()
                                }
                            ],
                        }
                    ]
                },
            )
            values = {"episode_index": [7, 7, 7], "frame_index": [0, 1, 2], "timestamp": [8.0, 8 + 1 / 30, 8 + 2 / 30]}

            class Table:
                num_rows = 3

                def __getitem__(self, key):
                    return types.SimpleNamespace(to_pylist=lambda: values[key])

            def read_table(path, columns):
                self.assertEqual(Path(path), data)
                self.assertEqual(columns, ["episode_index", "frame_index", "timestamp"])
                return Table()

            pq = types.ModuleType("pyarrow.parquet")
            pq.read_table = read_table
            pa = types.ModuleType("pyarrow")
            pa.parquet = pq
            output = root / "timestamps.json"
            with patch.dict(sys.modules, {"pyarrow": pa, "pyarrow.parquet": pq}):
                batch.extract_actual_timestamps(draft, audit, output)
                actual = batch.load(output)["parents"][0]
                self.assertEqual(actual["timestamps"], values["timestamp"])
                self.assertEqual(actual["timestamp_source"]["sha256"], batch.digest(data))
                values["frame_index"] = [0, 2, 1]
                with self.assertRaisesRegex(ValueError, "frame_index"):
                    batch.extract_actual_timestamps(draft, audit, root / "must-not-exist.json")
                self.assertFalse((root / "must-not-exist.json").exists())


if __name__ == "__main__":
    unittest.main()
