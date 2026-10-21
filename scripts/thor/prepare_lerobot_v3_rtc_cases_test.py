"""CPU data-contract tests only; no model, training or device control."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import prepare_lerobot_v3_rtc_cases as module


class ReplayTests(unittest.TestCase):
    def test_paths_cannot_escape_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                module.checked_path(Path(directory), "../outside")

    def test_pts_mismatch_is_rejected(self):
        class Frame:
            pts = 0
            time_base = 1

        @contextmanager
        def opened(_path):
            yield mock.Mock(decode=lambda **_kwargs: iter([Frame()]))

        with mock.patch.object(module.av, "open", opened):
            with self.assertRaisesRegex(ValueError, "cover requested timestamps"):
                module.read_rgb(Path("unused"), {1: 1 / 30}, 30)

    def test_v3_offsets_prefixes_and_training_norm(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, checkpoint, output = base / "val", base / "checkpoint", base / "replay"
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            (checkpoint / "assets/yam").mkdir(parents=True)
            (checkpoint / "params").mkdir()
            (checkpoint / "params/_METADATA").write_text("{}")
            norm = b'{"test_only": true}'
            (checkpoint / "assets/yam/norm_stats.json").write_bytes(norm)
            contract = base / "training_contract.json"
            contract.write_text(json.dumps({"model": {"pi05": True, "action_horizon": 50,
                "rtc_training_max_delay": 10},
                "norm_sha256": hashlib.sha256(norm + b"\n").hexdigest()}))
            info = {"codebase_version": "v3.0", "robot_type": "yam", "fps": 30,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
            (root / "meta/info.json").write_text(json.dumps(info))
            entries = []
            actions = np.tile(np.arange(14, dtype=np.float32), (301, 1))
            for eid in range(3):
                entry = {"episode_index": eid, "tasks": ["test fixture only"], "length": 301,
                    "data/chunk_index": 0, "data/file_index": eid}
                data = root / f"data/chunk-000/file-{eid:03d}.parquet"
                data.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table({"observation.state": (actions * 0.1).tolist(),
                    "action": actions.tolist(), "episode_index": [eid] * 301,
                    "frame_index": list(range(301)), "timestamp": (np.arange(301) / 30).tolist()}), data)
                for key in module.VIEWS:
                    prefix = f"videos/{key}"
                    entry.update({f"{prefix}/chunk_index": 0, f"{prefix}/file_index": eid,
                        f"{prefix}/from_timestamp": 5.5, f"{prefix}/to_timestamp": 5.5 + 301 / 30})
                    video = root / f"videos/{key}/chunk-000/file-{eid:03d}.mp4"
                    video.parent.mkdir(parents=True, exist_ok=True)
                    video.write_bytes(b"mock video; decoder is patched")
                entries.append(entry)
            pq.write_table(pa.Table.from_pylist(entries), root / "meta/episodes/chunk-000/file-000.parquet")
            calls = []

            def rgb(_path, targets, fps):
                calls.append((targets, fps))
                return {index: np.zeros((224, 224, 3), dtype=np.uint8) for index in targets}

            with mock.patch.object(module, "read_rgb", rgb):
                module.prepare(root, checkpoint, contract, output, [0, 1, 2])
                with self.assertRaises(ValueError):
                    module.prepare(root, checkpoint, contract, output, [0, 1, 2])
            cases = json.loads((output / "cases.json").read_text())
            self.assertEqual(len(cases["cases"]), 9)
            self.assertEqual(cases["norm_identity"]["identity_relation"], "checkpoint_omitted_one_final_newline")
            self.assertEqual((output / "norm_stats.json").read_bytes(), norm)
            self.assertEqual(len(calls), 9)
            self.assertAlmostEqual(calls[0][0][100], 5.5 + 100 / 30)
            for case in cases["cases"]:
                delay = case["delay_steps"]
                prefix = np.load(output / case["committed_actions"], allow_pickle=False)
                np.testing.assert_array_equal(prefix, actions[:delay])
                self.assertEqual(case["target_start_tick"], case["observation_policy_tick"])
                provenance = json.loads((output / case["provenance"]).read_text())
                self.assertIn(case["source_episode"], provenance["source_files"])


if __name__ == "__main__":
    unittest.main()
