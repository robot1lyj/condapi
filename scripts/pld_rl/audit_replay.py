"""Pure-array audit of rewards, causal frame references and the held-out split."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.data import action_coordinates
from yam_pld_rl.data import fit_normalization
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, key in [
        ("replay.npz", "replay_sha256"),
        ("rewards.json", "reward_sha256"),
        ("normalization.json", "normalization_sha256"),
    ]:
        assert sha256(root / filename) == manifest[key], filename
    with np.load(root / "replay.npz") as source:
        a = {key: source[key] for key in source.files}
    attempts = json.loads((root / "attempts.json").read_text())
    observations = json.loads((root / "observations.json").read_text())
    rewards = {x["candidate_id"]: x for x in json.loads((root / "rewards.json").read_text())}
    assert len(a["reward"]) == manifest["transitions"]
    assert np.all((a["dt"] > 0) & (a["dt"] <= 0.15))
    np.testing.assert_allclose(a["discount"], manifest["discount_per_second"] ** a["dt"])
    np.testing.assert_allclose(a["action_coordinates"], action_coordinates(a["action"], a["state"]))
    np.testing.assert_array_equal(a["base"], a["action"])
    for aid, attempt in enumerate(attempts):
        ids = np.flatnonzero(a["attempt"] == aid)
        assert len(ids) == attempt["transitions"]
        assert ids[0] == attempt["transition_start"]
        assert np.all(a["episode"][ids] == attempt["episode"])
        assert np.all(a["outcome"][ids] == attempt["reward"])
        assert attempt["reward"] == rewards[attempt["candidate_id"]]["approved_reward"]
        assert np.sum(a["terminal"][ids]) == 1
        assert a["terminal"][ids[-1]]
        assert a["row"][ids[-1]] + 1 == attempt["terminal_row"]
        assert np.all(a["reward"][ids[:-1]] == 0)
        assert a["reward"][ids[-1]] == attempt["reward"]
        for i in ids:
            now, nxt = observations[a["obs_index"][i]], observations[a["next_obs_index"][i]]
            assert now["episode"] == nxt["episode"] == attempt["episode"]
            assert now["row"] == a["row"][i]
            assert nxt["row"] == now["row"] + 1
        valid = ids[a["mc_valid"][ids]]
        expected = attempt["reward"] * manifest["discount_per_second"] ** (a["time"][ids[-1]] - a["time"][valid])
        np.testing.assert_allclose(a["mc_return"][valid], expected)
    stats = json.loads((root / "normalization.json").read_text())
    recalculated = fit_normalization(a["state"], a["action_coordinates"], a["episode"] != 1)
    assert stats == recalculated
    return {
        "passed": True,
        "replay_sha256": manifest["replay_sha256"],
        "attempts": len(attempts),
        "transitions": len(a["reward"]),
        "terminal_events": int(a["terminal"].sum()),
        "positive_rewards": int(a["reward"].sum()),
        "validation_episodes": [1],
        "no_gap_compaction": True,
        "reward_once_per_attempt": True,
        "normalization_train_only": True,
        "scope": "array/provenance audit; raw freshness and physical continuity are checked by the builder",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.replay)
    write_json(args.output, result)
    print(json.dumps(result))
