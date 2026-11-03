# Standard-library unittest entrypoint is intentional.
# ruff: noqa: PT009, PT027
"""Stdlib receipt-interface tests, not actual Parquet scan certification."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "fineart_preparation_under_test", Path(__file__).with_name("prepare_fineart_tshirt_training.py")
)
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


class EmptyAnchorAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fineart-audit-interface-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.datasets, files = {}, []
        for pool, eid in (("train", 7), ("dev", 8)):
            path = root / pool / "data/chunk-000/file-000.parquet"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"synthetic receipt fixture; not an actual Parquet")
            files.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            self.datasets[pool] = (root / pool, {"total_frames": 1000, "features": {"language_events": {}}}, {eid})
        self.release = {"language_supervision_mode": "native-empty-anchor-fallback"}
        self.audit = {
            "mode": "native-empty-anchor-fallback",
            "materialization_status": "MATERIALIZED_AUDITED",
            "unresolved_overlaps": 0,
            "empty_instructions": 0,
            "same_timestamp_conflicts": 0,
            "action_coverage": {"train": 1.0, "dev": 1.0},
            "effective_language_coverage": {"train": 0.98, "dev": 1.0},
            "parents": [
                {
                    "pool": "train",
                    "source_episode_index": 7,
                    "length": 1000,
                    "empty_language_anchor_intervals": [
                        {"from_frame": 10, "to_frame": 30, "language_persistent": [], "language_events": []}
                    ],
                },
                {"pool": "dev", "source_episode_index": 8, "length": 1000, "empty_language_anchor_intervals": []},
            ],
            "parquet_scan": {
                "checked_all_rows": True,
                "empty_columns_exactly_at_unknown_anchors": True,
                "normal_anchors_keep_full_timeline": True,
                "only_language_columns_changed": True,
                "unknown_anchor_frames": {"train": 20, "dev": 0},
                "files": files,
            },
        }

    def test_explicit_fallback_interface_reports_coverage_separately(self):
        value = preparation.check_language_audit(self.release, self.audit, self.datasets)
        self.assertEqual(value["effective_language_coverage"]["train"], 0.98)
        self.assertEqual(value["action_coverage"]["train"], 1.0)

    def test_sidecar_false_coverage_null_and_missing_actual_scan_are_rejected(self):
        mutations = [
            lambda x: x.update(materialization_status="NOT_MATERIALIZED_NOT_TRAINING_READY"),
            lambda x: x["effective_language_coverage"].update(train=1.0),
            lambda x: x.update(coverage={"train": 1.0, "dev": 1.0}),
            lambda x: x["parents"][0]["empty_language_anchor_intervals"][0].update(language_persistent=None),
            lambda x: x["parquet_scan"].update(checked_all_rows=False),
            lambda x: x["parquet_scan"]["files"][0].update(sha256="changed"),
            lambda x: x["parents"][0]["empty_language_anchor_intervals"][0].update(to_frame=31),
        ]
        for mutate in mutations:
            audit = copy.deepcopy(self.audit)
            mutate(audit)
            with self.subTest(audit=audit), self.assertRaises(ValueError):
                preparation.check_language_audit(self.release, audit, self.datasets)

    def test_contiguous_limit_combines_adjacent_atoms(self):
        audit = copy.deepcopy(self.audit)
        audit["parents"][0]["length"] = 2000
        self.datasets["train"][1]["total_frames"] = 2000
        audit["parents"][0]["empty_language_anchor_intervals"] = [
            {"from_frame": 0, "to_frame": 20, "language_persistent": [], "language_events": []},
            {"from_frame": 20, "to_frame": 40, "language_persistent": [], "language_events": []},
        ]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            preparation.check_language_audit(self.release, audit, self.datasets)

    def test_strict_default_still_requires_full_language(self):
        strict = {
            "coverage": {"train": 1.0, "dev": 1.0},
            "unresolved_overlaps": 0,
            "empty_instructions": 0,
            "same_timestamp_conflicts": 0,
        }
        self.assertEqual(preparation.check_language_audit({}, strict, self.datasets)["mode"], "strict-parent")
        strict["coverage"]["train"] = 0.98
        with self.assertRaises(ValueError):
            preparation.check_language_audit({}, strict, self.datasets)


if __name__ == "__main__":
    unittest.main()
