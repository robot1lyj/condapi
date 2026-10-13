"""CPU-only reward-policy checks; no model loading or learning."""

# ruff: noqa: E402, PT009

import copy
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src")]
from grasp_v3_checks import modern_recording
from yam_pld_rl.review_policy import automatic_reward
from yam_pld_rl.review_policy import merge_rewards


def fixture():
    candidate = next(c for c in modern_recording()[3] if c.get("outcome"))
    judged = {"suggested_label": "success", "confidence": "low", "audit_flags": ["dino_weak_or_mixed"],
              "dino": {"outcome": "failure"}}
    return candidate, judged


class ReviewPolicyChecks(unittest.TestCase):
    def test_weak_disagreeing_suggestion_is_used_with_provenance(self):
        candidate, judged = fixture()
        reward, _ = automatic_reward(candidate, judged)
        self.assertEqual(reward["status"], "held")
        self.assertFalse(reward["human_reviewed"])
        self.assertEqual(reward["model_confidence"], "low")
        self.assertEqual(reward["dino_suggestion"], "failure")

    def test_absent_suggestion_stays_pending_not_negative(self):
        candidate, _ = fixture()
        merged = merge_rewards([candidate], {}, {})
        self.assertFalse(merged["rewards"])
        self.assertEqual(len(merged["pending"]), 1)

    def test_missing_latest_evidence_does_not_become_success(self):
        candidate, judged = fixture()
        candidate["outcome"]["lifted_rows"] = candidate["outcome"]["lifted_rows"][:-1]
        self.assertIsNone(automatic_reward(candidate, judged)[0])

    def test_duplicate_frames_cannot_prove_held(self):
        candidate, judged = fixture()
        frames = [f for f in candidate["frames"] if f["row"] in candidate["outcome"]["fresh_rows"]]
        for frame in frames:
            frame["views"][candidate["arm"]]["video_frame_index"] = 1
        self.assertIsNone(automatic_reward(candidate, judged)[0])

    def test_human_override_and_exclusion_take_priority(self):
        candidate, judged = fixture()
        cid = candidate["candidate_id"]
        merged = merge_rewards([candidate], {cid: judged}, {cid: {"status": "empty"}})
        self.assertEqual(merged["rewards"][0]["status"], "empty")
        self.assertTrue(merged["rewards"][0]["human_reviewed"])
        excluded = merge_rewards([candidate], {cid: judged}, {cid: {"status": "exclude", "note": "container adjustment"}})
        self.assertFalse(excluded["rewards"])
        self.assertEqual(len(excluded["excluded"]), 1)

    def test_invalid_human_choice_blocks_automatic_fallback(self):
        candidate, judged = fixture()
        cid = candidate["candidate_id"]
        before = copy.deepcopy(candidate)
        merged = merge_rewards([candidate], {cid: judged}, {cid: {"status": "dropped"}})
        self.assertFalse(merged["rewards"])
        self.assertEqual(len(merged["pending"]), 1)
        self.assertEqual(candidate, before)


if __name__ == "__main__":
    unittest.main()
