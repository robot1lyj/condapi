"""Native UUID/sidecar and probe rejection regression checks; no learner."""

# ruff: noqa: E402, PT009, PT027

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from curator_checks import native_fixture
from yam_pld_rl.native_recording import read_episode
from yam_pld_rl.probe_audit import audit_probe


def write_fixture(root):
    folder = root / "session" / "episode_000001"
    segment = folder / "segment_000000"
    segment.mkdir(parents=True)
    manifest = {"schema": "yam_hil_v2", "episode_id": "native-uuid", "outcome": "unknown",
                "pld": {"mode": "probe"}, "policy_events": "../policy_events.jsonl"}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with h5py.File(segment / "samples.h5", "w") as f:
        for key, value in (("time", [100.]), ("tick", [5]), ("epoch", [1]),
                           ("observation_valid", [True]), ("is_intervention", [False])):
            f[key] = value
            f[key + "__valid"] = [True]
        for key in ("measured_state", "observation_state", "submitted_action"):
            f[key] = np.full((1, 14), .25)
            f[key + "__valid"] = [True]
        f["video_indices"] = [[0, 0, 0]]
        f["camera_host_received_at"] = [[100., 100., 100.]]
        f["camera_host_received_at__valid"] = [[True, True, True]]
        f["details"] = [json.dumps({"obs_id": 7})]
        f["committed_rows"] = 1
    for view in ("top", "left", "right"):
        (segment / f"{view}.mp4").write_bytes(b"fixture-only-not-a-decodable-video")
    events = [{"event": "evaluation", "episode_id": ep, "details": {"evaluation": {"C": 1, "N": 2, "purpose": "evaluation_only"}}}
              for ep in ("foreign-uuid", "native-uuid")]
    (folder.parent / "policy_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return folder


def probe_fixture():
    _, rows, _, _ = native_fixture()
    identity = None
    for row in rows:
        row["observation_state"] = [0.] * 14
        reply = row["details"].get("policy_reply")
        if not reply:
            continue
        p, token = reply["pld"], reply["token"]
        p.update(mode="probe", replay_eligible=False, residual_applied=False, base=copy.deepcopy(reply["actions"]),
                 actions=copy.deepcopy(reply["actions"]))
        if row["tick"]:
            p["previous_command"] = rows[row["tick"] - 1]["submitted_action"]
        token["observed_at"] = row["time"] + .001
        row["observation_state"] = p["state"]
        reply["pld_request"] = {k: p[k] for k in ("schema", "mode", "run_id", "episode_id", "contract_sha256", "snapshot_sha256",
                                                "epoch", "request_id", "observation_id", "previous_command")}
        reply["request_context"] = {"previous_command_tick": row["tick"] - 1,
                                    "previous_command_submitted_at": row["time"] - .01,
                                    "committed_actions": reply["actions"][:p["delay_steps"]]}
        identity = {k: p[k] for k in ("mode", "run_id", "contract_sha256", "snapshot_sha256")}
    return rows, {"episode_id": "ep", "pld": identity}


class NativeChecks(unittest.TestCase):
    def test_native_uuid_filters_other_episode_and_hashes_all_sources(self):
        with tempfile.TemporaryDirectory() as root:
            folder = write_fixture(Path(root))
            rows, events, hashes = read_episode(folder)
            self.assertEqual(rows[0]["episode_id"], "native-uuid")
            self.assertNotEqual(rows[0]["episode_id"], folder.name)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["details"]["evaluation"]["purpose"], "evaluation_only")
            self.assertTrue(events[0]["audit_ref"].endswith("#line=2"))
            self.assertEqual(len(hashes), 6)

    def test_missing_sidecar_and_missing_uuid_reject(self):
        with tempfile.TemporaryDirectory() as root:
            folder = write_fixture(Path(root))
            (folder.parent / "policy_events.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "complete session"):
                read_episode(folder)
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest.pop("episode_id")
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "episode_id required"):
                read_episode(folder)

    def test_probe_audit_passes_and_does_not_create_rewards(self):
        rows, manifest = probe_fixture()
        report = audit_probe(rows, [], manifest)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["counts"]["accepted_probe_replies"], 4)
        self.assertFalse(report["training_ready"])
        self.assertIsNone(report["robot_success_rate"])

    def test_wrong_input_source_and_identity_fail_probe(self):
        for mutation in ("previous", "state", "identity", "source"):
            rows, manifest = probe_fixture()
            reply = rows[8]["details"]["policy_reply"]
            if mutation == "previous":
                reply["pld"]["previous_command"] = [9.] * 14
            elif mutation == "state":
                rows[8]["observation_state"] = [9.] * 14
            elif mutation == "identity":
                reply["pld"]["episode_id"] = "another-episode"
            else:
                rows[10]["submitted_action"] = [9.] * 14
            report = audit_probe(rows, [], manifest)
            self.assertEqual(report["status"], "failed", mutation)
            self.assertTrue(report["errors"])


if __name__ == "__main__":
    unittest.main()
