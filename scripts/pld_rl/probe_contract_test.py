"""Array-only probe deployment checks; no model inference or learner."""

# ruff: noqa: E402, PT009, PT027

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))
from curator_checks import native_fixture
from yam_pld_rl.curator import reconstruct_intervals
from yam_pld_rl.probe_runtime import ProbeRuntime


class ProbeChecks(unittest.TestCase):
    def setUp(self):
        self.runtime = ProbeRuntime(ROOT / "third_party/yam-fk-model/yam_linear4310.xml", base_weights_sha256="fixture",
                                    norm_sha256="fixture", engine_sha256="fixture", source_revision="fixture")
        self.obs = {"prompt": "fixture", "observation.state": np.zeros(14)}
        self.obs["observation.state"][[6,13]] = .9
        self.base = np.tile(self.obs["observation.state"], (50,1))
        self.base[5:,6] = .4
        self.rtc = {"observation_policy_tick": 0, "delay_steps": 3}
        self.request = {**self.runtime.metadata, "mode": "probe", "run_id": "r", "episode_id": "e", "epoch": 0,
                        "observation_id": 1, "request_id": 1, "previous_command": self.base[0]}

    def test_probe_is_bitwise_pi_and_has_no_training_contract(self):
        session = self.runtime.new_session()
        output = session(self.obs,self.rtc,self.request,{"actions": self.base})
        self.assertEqual(output["actions"].tobytes(), self.base.tobytes())
        self.assertEqual(output["pld"]["base"].tobytes(), self.base.tobytes())
        self.assertEqual(output["pld"]["arm"], "left")
        self.assertFalse(output["pld"]["residual_applied"])
        self.assertFalse(output["pld"]["replay_eligible"])
        self.assertNotIn("x",output["pld"])
        self.assertEqual(self.runtime.metadata["modes"],["off","probe"])

    def test_unsupported_modes_and_wrong_identity_reject(self):
        for mode in ("collect","eval","shadow"):
            with self.assertRaisesRegex(ValueError,"probe-only"):
                self.runtime.validate_request(self.obs,self.rtc,{**self.request,"mode": mode})
        with self.assertRaises(ValueError):
            self.runtime.validate_request(self.obs,self.rtc,{**self.request,"snapshot_sha256": "wrong"})

    def test_off_and_connections_reset_without_model_reload(self):
        session = self.runtime.new_session()
        first = session(self.obs,self.rtc,self.request,{"actions":self.base})
        output = {"actions":self.base}
        self.assertIs(session(self.obs,self.rtc,{"mode":"off"},output),output)
        second = session(self.obs,self.rtc,self.request,output)
        self.assertNotEqual(first["pld"]["window_id"],second["pld"]["window_id"])
        another = self.runtime.new_session()
        self.assertIsNone(another.gate.active)

    def test_probe_diagnostics_cannot_enter_sac_replay(self):
        _, rows, _, _ = native_fixture()
        for row in rows:
            p = (row["details"].get("policy_reply") or {}).get("pld")
            if p:
                p["replay_eligible"] = False
        windows, audit = reconstruct_intervals(rows)
        self.assertEqual(len(audit),3)
        self.assertTrue(all(r["receipt"]["status"]=="rejected" for r in windows))
        self.assertTrue(all("not SAC replay" in r["reason"] for r in audit))


if __name__ == "__main__":
    unittest.main()
