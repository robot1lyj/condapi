"""CPU geometry and recorded-interval tests; no training or robot connection."""
# ruff: noqa: PT009, PT027
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages/pld-rl/src"))
from block_fixtures import stats
from grasp_v3_checks import modern_contract
from grasp_v3_checks import modern_recording
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_contract import window_context
from yam_pld_rl.block_replay import compile_replay
from yam_pld_rl.collector import center_latent
from yam_pld_rl.collector import closing_collect_manifest
from yam_pld_rl.collector import execution_contract
from yam_pld_rl.collector import expanded_manifest
from yam_pld_rl.collector import native_trajectory_manifest
from yam_pld_rl.collector import smooth_manifest
from yam_pld_rl.collector import validate_collector
from yam_pld_rl.curator import associate_rewards


def parent():
    c = modern_contract()
    c.update(radius=[.02]*6+[.05]+[.02]*6+[.05], max_residual_step=[.008]*6+[.02]+[.008]*6+[.02])
    return {"contract": c, "phase": "candidate_requires_AB", "provenance": {"actor_updates": 500},
            "normalization": stats(), "encoder_sha256": "fixture", "actor_sha256": "unchanged-fixture"}


class CollectorChecks(unittest.TestCase):
    def test_closing_collector_preserves_parent_eval_and_weights(self):
        behavior = smooth_manifest(parent(), parent_snapshot="1"*64)
        learned = native_trajectory_manifest({"contract": behavior["contract"], "actor_sha256": "learned",
                                              "provenance": {"actor_updates": 500}}, behavior_snapshot="2"*64)
        saved = copy.deepcopy(learned)
        out = closing_collect_manifest(learned, parent_snapshot="3"*64)
        self.assertEqual(learned, saved)
        self.assertEqual(out["eval_contract"], learned["eval_contract"])
        self.assertEqual(out["actor_contract"], learned["actor_contract"])
        self.assertEqual(out["actor_sha256"], learned["actor_sha256"])
        self.assertEqual(out["contract"]["negative_radius"][6], .20)
        self.assertEqual(out["contract"]["positive_radius"][6], .05)
        self.assertEqual(out["contract"]["radius"][0], .08)
        self.assertFalse(out["collector"]["native_actor_coordinates"])
        broken = copy.deepcopy(out)
        broken["contract"]["negative_radius"][6] = .04
        with self.assertRaises(ValueError):
            validate_collector(broken)

    def test_new_actor_uses_its_trained_trajectory_coordinates(self):
        behavior = smooth_manifest(parent(), parent_snapshot="1"*64)
        learned = {"contract": behavior["contract"], "actor_sha256": "new-learned-weights"}
        out = native_trajectory_manifest(learned, behavior_snapshot="2"*64)
        self.assertEqual(out["actor_contract"], behavior["contract"])
        self.assertEqual(execution_contract(out, "eval")["radius"][0], .08)
        self.assertEqual(out["contract"]["action_mapping"], "signed_trajectory_v1")
        self.assertEqual(behavior["actor_contract"]["radius"][0], .02)
        self.assertNotIn("collector", learned)

    def test_eval_unchanged_and_mutation_isolated(self):
        old = parent()
        saved = copy.deepcopy(old)
        new = expanded_manifest(old, parent_snapshot="1"*64)
        self.assertEqual(old, saved)
        for mode in ("probe", "shadow", "eval", "off"):
            self.assertEqual(execution_contract(new, mode), old["contract"])
        self.assertEqual(execution_contract(new, "collect")["radius"][0], .08)
        self.assertEqual(new["actor_sha256"], old["actor_sha256"])
        for key, value in (("task", "different"), ("max_target_step", [.9]*14)):
            broken = copy.deepcopy(new)
            broken["contract"][key] = value
            with self.assertRaises(ValueError):
                validate_collector(broken)

    def test_signed_center_decoder_and_slew(self):
        old = parent()["contract"]
        new = expanded_manifest(parent(), parent_snapshot="1"*64)["contract"]
        rng = np.random.default_rng(372)
        for arm in ("left", "right"):
            off = 0 if arm == "left" else 7
            for endpoint in (.02, .5, .98):
                for _ in range(30):
                    base = np.full((50, 14), .4)
                    base[:, off+6] = endpoint
                    ap, an = feasible_bounds(base, base[0], arm, 10, old)
                    pos, neg = feasible_bounds(base, base[0], arm, 10, new)
                    mean = rng.uniform(-.9, .9, 7)
                    physical = map_action(mean, ap, an)*np.asarray(old["radius"][off:off+7])
                    latent = center_latent(physical, new["radius"][off:off+7], pos, neg)
                    expected = decode(base, map_action(mean, ap, an), arm, 10, old)
                    actual = decode(base, map_action(np.tanh(latent), pos, neg), arm, 10, new)
                    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
                    sampled = map_action(np.tanh(latent+rng.normal(size=7)*.5), pos, neg)
                    out = decode(base, sampled, arm, 10, new)
                    np.testing.assert_array_equal(out[:10], base[:10])
                    self.assertLessEqual(np.abs(np.diff(out-base, axis=0)[:, off:off+6]).max(), .03+1e-7)
        np.testing.assert_array_equal(center_latent(np.zeros(7), np.ones(7), np.zeros(7), np.zeros(7)), np.zeros(7))
        with self.assertRaises(ValueError):
            center_latent(np.ones(7), np.ones(7), np.zeros(7), np.zeros(7))

    def test_collector_reaches_native_reward_replay(self):
        manifest = expanded_manifest(parent(), parent_snapshot="1"*64)
        c, rows, records, candidates, reviews = modern_recording(contract_override=manifest["contract"])
        linked, labels, errors = associate_rewards(records, candidates, reviews, rows, "train")
        self.assertFalse(errors)
        for record in linked:
            p = record["proposal"]
            ac = manifest["actor_contract"]
            ap, an = feasible_bounds(p["base"], p["previous_command"], p["arm"], p["delay_steps"], ac)
            if not p["gate"]["apply_residual"]:
                ap, an = np.zeros(7), np.zeros(7)
            ax = observation_vector(np.asarray(p["x"])[:1536], p["state"], p["base"], p["previous_command"],
                                    p["arm"], p["delay_steps"], p["elapsed_seconds"], ap, stats(), negative_scale=an,
                                    context=window_context(p["gate"], p["start_tick"], p["base_z_m"][p["arm"]], 1/30))
            physical = map_action(np.full(7, .3), ap, an)*np.asarray(ac["radius"][:7])
            latent = center_latent(physical, c["radius"][:7], p["scale"], p["negative_scale"])
            p.update(execution_contract_sha256=fingerprint(c), actor_context=ax[2264:], actor_mean_u=[.3]*7,
                     actor_scale=ap, actor_negative_scale=an, actor_physical_center=physical,
                     collector_latent_center=latent, collector_latent_std=[.5]*6+[.2])
        data, audit = compile_replay(linked, labels, c, {"fixture": manifest})
        self.assertFalse(audit["excluded"])
        self.assertEqual(data["reward"].sum(), 1)
        self.assertTrue(np.any(data["action"]))
        linked[0]["proposal"]["actor_physical_center"] = np.ones(7)
        with self.assertRaisesRegex(ValueError, "collector provenance mismatch"):
            compile_replay(linked, labels, c, {"fixture": manifest})


if __name__ == "__main__":
    unittest.main()
