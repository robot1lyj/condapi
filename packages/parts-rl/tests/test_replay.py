# ruff: noqa: PT011, PT018
import copy
import json

import numpy as np
from parts_rl.data import Replay
from parts_rl.data import digest
from parts_rl.prepare import digest_feature
from parts_rl.prepare import prepare_runs
import pytest
from test_contract import declaration
from test_contract import sent
from vla_platform import parts as wire

h5py = pytest.importorskip("h5py")
REWARD = {
    "schema": "fixture-reward",
    "formula": "negative_absolute_height_error_v1",
    "height_scope": "whole_attempt",
    "height_weight": 1.0,
    "grasp_weight": 1.0,
}
SPLIT = {"train_groups": ["layout-train"], "holdout_groups": ["layout-held-out"]}


def write_json(path, value):
    path.write_text(json.dumps(value, default=lambda x: x.tolist(), allow_nan=False))


def write_jsonl(path, values):
    path.write_text(
        "".join(json.dumps(value, default=lambda x: x.tolist(), allow_nan=False) + "\n" for value in values)
    )


def publish(root):
    files = {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in root.rglob("*")
        if path.is_file() and path != root / "publication.json"
    }
    write_json(
        root / "publication.json",
        {
            "schema": "yam_parts_raw_v1",
            "run_id": "fixture",
            "client_complete": True,
            "training_ready": False,
            "gaps": [],
            "files": files,
        },
    )


def package(root, arm="left", result="success", *, repeated_sdk=False, omit_base=False):
    root.mkdir()
    config = {"reward": REWARD, "max_feedback_age_s": 0.1, "max_confirmation_gap_s": 0.1, "confirm_s": 0.05}
    write_json(
        root / "run.json",
        {
            "schema": "yam_parts_raw_v1",
            "run_id": "fixture",
            "mock": False,
            "mode": "collect",
            "split_role": "train",
            "layout_group_id": "layout-train",
            "contract_sha": "fixture-contract",
            "config": config,
        },
    )
    requests = []
    actions = np.zeros((50, 14), np.float64)
    with h5py.File(root / "requests.h5", "w") as file:
        for request_id, k in ((1, 0), (2, 2)):
            payload = sent("collect", k, request_id, arm)
            group_path = f"/requests/e0_r{request_id}"
            group = file.create_group(group_path)
            group["actions_native"] = actions
            for name, value in payload["scheduler"].items():
                group["scheduler/" + name] = value
            payload["scheduler"] = {}
            group["features/z"] = z = np.array([0.25, 0.5], np.float32)
            for side in wire.INDICES:
                group[side + "/u"] = np.full((50, 6), 0.1 * request_id, np.float32)
                group[side + "/B_rad"] = np.full(6, 0.001, np.float64)
                group[side + "/editable_mask"] = np.ones(50, bool)
            requests.append(
                {
                    "context": payload["context"],
                    "parts_request": payload,
                    "hdf5_group": group_path,
                    "observation_state": np.zeros(14),
                    "features": {"sha256": digest_feature(z), "feature_schema_id": "fixture-feature"},
                    "behavior_snapshot_id": "fixture-behavior",
                    "candidate_metadata": {side: {"actor_snapshot_id": "fixture-" + side} for side in wire.INDICES},
                }
            )
    for request in requests:
        k = request["context"]["observation_policy_tick"]
        request["video_refs"] = {
            "episode_id": "e1",
            "episode_path": "episodes/e1",
            "segment": "segment_000",
            "tick": k,
            "frame_indices": dict.fromkeys(("top", "left", "right"), k),
        }
    write_jsonl(root / "requests.jsonl", requests)
    details, submitted, feedbacks = [], [], []
    for tick in range(5):
        value = sent(arm=arm)["arms"][arm]
        value["error_m"] = 0.04 - 0.001 * tick
        value["height_m"] = value["h_goal_m"] + value["error_m"]
        feedback = {
            "valid": True,
            "sampled_at": 1 + tick / 30,
            "feedback_age_s": 0.0,
            "sdk_updated_at": 100.0 if repeated_sdk else 100.0 + tick,
            "effort_nm": -0.8,
        }
        value["force_feedback"] = feedback
        feedbacks.append(feedback)
        request_id = 1 if tick < 2 else 2
        k = 0 if tick < 2 else 2
        residual = np.zeros(14)
        residual[wire.INDICES[arm]] = np.float32(0.1 * request_id) * 0.001
        submitted.append(residual)
        part = {
            "attempt_id": "a1",
            "active_arm": arm,
            "physical_residual_rad": residual,
            "arms": {arm: value},
            "selection": {
                "request": {"epoch": 0, "request_id": request_id},
                "model_index": tick - k,
                "target_tick": tick,
            },
            "candidate_ref": {
                "request_epoch": 0,
                "request_id": request_id,
                "model_index": tick - k,
                "arm": arm,
                "actor_snapshot_id": "fixture-" + arm,
                "behavior_snapshot_id": "fixture-behavior",
            },
        }
        if not omit_base:
            part["base_target"] = np.zeros(14)
        details.append({"parts": part, "observation_valid": True})
    attempt = {
        "arm": arm,
        "attempt_id": "a1",
        "epoch": 0,
        "result": result,
        "entry_tick": 0,
        "terminal_tick": 4,
        "closure_tick": 1,
        "event_id": "terminal",
        "handback_effective_tick": 4,
        "grasp_reward": int(result == "success"),
    }
    write_jsonl(root / "attempts.jsonl", [attempt])
    write_jsonl(
        root / "events.jsonl",
        [
            {
                "event_id": "terminal",
                "attempt_id": "a1",
                "kind": result,
                "tick": 4,
                "time": 1 + 4 / 30,
                "feedback": feedbacks[-1],
            }
        ],
    )
    episode = root / "episodes/e1"
    segment = episode / "segment_000"
    segment.mkdir(parents=True)
    write_json(
        episode / "manifest.json",
        {
            "schema": "yam_hil_v2",
            "episode_id": "e1",
            "segments": [
                {"path": "segment_000", "steps": 5, "video_frames": dict.fromkeys(("top", "left", "right"), 5)}
            ],
        },
    )
    for role in ("top", "left", "right"):
        (segment / (role + ".mp4")).write_bytes(b"synthetic protocol fixture; no video decoding")
    with h5py.File(segment / "samples.h5", "w") as file:
        file["tick"] = np.arange(5)
        file["obs_id"] = [1, 1, 2, 2, 2]
        file["observation_state"] = np.zeros((5, 14))
        file["epoch"] = np.zeros(5, np.int64)
        file["time"] = 1 + np.arange(5) / 30
        file["submitted_action"] = np.asarray(submitted)
        file["measured_state"] = np.zeros((5, 14))
        file["committed_rows"] = 5
        file.create_dataset(
            "details",
            data=[json.dumps(value, default=lambda x: x.tolist()) for value in details],
            dtype=h5py.string_dtype(),
        )
    publish(root)
    return root


@pytest.mark.parametrize("arm", ["left", "right"])
def test_prepare_real_layout_without_any_model_or_training(tmp_path, arm):
    root = package(tmp_path / "raw", arm=arm)
    before = digest(root / "publication.json")
    report = prepare_runs([root], tmp_path / "replay", declaration(["collect"]), REWARD, SPLIT, 0.5)
    assert report["ready_rows"] == {arm: 2}
    assert report["excluded_attempts"] == 0
    assert digest(root / "publication.json") == before
    replay = Replay(tmp_path / f"replay/{arm}/READY.json", arm)
    assert len(replay) == 2
    np.testing.assert_array_equal(replay.arrays["elapsed_steps"], [2, 2])
    np.testing.assert_array_equal(replay.arrays["bootstrap"], [True, False])
    assert replay.arrays["success"].all()  # Intermediate successful-attempt rows may bootstrap.
    # Commands [k,next_tick), feedback/rewards (k,next_tick]. Never count the next plan's command.
    for mask in replay.arrays["executed_mask"].reshape(2, 50, 6):
        assert mask[:2].all() and not mask[2:].any()
    np.testing.assert_allclose(replay.arrays["reward"], [-0.039 + 0.5 * (-0.038), -0.037 + 0.5 * (1 - 0.036)])
    assert not replay.arrays["next_action_mask"][-1].any()


@pytest.mark.parametrize("gap", ["duplicate_sdk", "missing_base", "canceled"])
def test_incomplete_attempt_never_ready(tmp_path, gap):
    root = package(
        tmp_path / "raw",
        repeated_sdk=gap == "duplicate_sdk",
        omit_base=gap == "missing_base",
        result="canceled" if gap == "canceled" else "success",
    )
    report = prepare_runs([root], tmp_path / "replay", declaration(["collect"]), REWARD, SPLIT, 0.5)
    assert report["ready_rows"] == {}
    assert not list((tmp_path / "replay").rglob("READY.json"))
    audit = json.loads((tmp_path / "replay/audit.json").read_text())
    assert audit["excluded"]


@pytest.mark.parametrize("invalid", ["mock", "bad_hash", "escape", "holdout", "reward"])
def test_publication_and_split_gates(tmp_path, invalid):
    root = package(tmp_path / "raw")
    if invalid in ("mock", "holdout"):
        run = json.loads((root / "run.json").read_text())
        run["mock" if invalid == "mock" else "layout_group_id"] = True if invalid == "mock" else "layout-held-out"
        write_json(root / "run.json", run)
        publish(root)
    elif invalid == "bad_hash":
        (root / "events.jsonl").write_text("{}\n")
    elif invalid == "escape":
        publication = json.loads((root / "publication.json").read_text())
        publication["files"]["../outside"] = {"bytes": 0, "sha256": "bad"}
        write_json(root / "publication.json", publication)
    reward = copy.deepcopy(REWARD)
    if invalid == "reward":
        reward["grasp_weight"] = 2.0
    if invalid == "holdout":
        assert (
            prepare_runs([root], tmp_path / "replay", declaration(["collect"]), reward, SPLIT, 0.5)["ready_rows"] == {}
        )
    else:
        with pytest.raises(ValueError):
            prepare_runs([root], tmp_path / "replay", declaration(["collect"]), reward, SPLIT, 0.5)


def test_pre_entry_candidate_applied_later_keeps_causal_interval(tmp_path):
    root = package(tmp_path / "raw")
    attempts = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
    attempts[0]["entry_tick"] = 1
    write_jsonl(root / "attempts.jsonl", attempts)
    requests = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
    requests[0]["parts_request"]["active_arm"] = None
    requests[0]["parts_request"]["arms"]["left"].update(attempt_id=None, phase="READY")
    write_jsonl(root / "requests.jsonl", requests)
    with h5py.File(root / "episodes/e1/segment_000/samples.h5", "r+") as file:
        detail = json.loads(file["details"][0])
        detail["parts"].update(attempt_id=None, active_arm=None, candidate_ref=None)
        file["details"][0] = json.dumps(detail)
        file["submitted_action"][0] = np.zeros(14)
    publish(root)
    result = prepare_runs([root], tmp_path / "replay", declaration(["collect"]), REWARD, SPLIT, 0.5)
    assert result["ready_rows"] == {"left": 2}
    replay = Replay(tmp_path / "replay/left/READY.json", "left")
    np.testing.assert_allclose(replay.arrays["reward"][0], 0.5 * (-0.038))
    mask = replay.arrays["executed_mask"][0].reshape(50, 6)
    assert not mask[0].any()
    assert mask[1].all()
    assert not mask[2:].any()


def test_descent_reward_uses_previous_command_phase(tmp_path):
    root = package(tmp_path / "raw")
    reward = {**REWARD, "height_scope": "active_descent"}
    run = json.loads((root / "run.json").read_text())
    run["config"]["reward"] = reward
    write_json(root / "run.json", run)
    with h5py.File(root / "episodes/e1/segment_000/samples.h5", "r+") as file:
        for i in range(2, 5):
            detail = json.loads(file["details"][i])
            detail["parts"]["arms"]["left"]["phase"] = "ACTIVE_CLOSURE"
            file["details"][i] = json.dumps(detail)
    publish(root)
    assert prepare_runs([root], tmp_path / "replay", declaration(["collect"]), reward, SPLIT, 0.5)["ready_rows"]
    replay = Replay(tmp_path / "replay/left/READY.json", "left")
    np.testing.assert_allclose(replay.arrays["reward"], [-0.039 + 0.5 * (-0.038), 0.5])


def test_threshold_equality_is_not_success(tmp_path):
    root = package(tmp_path / "raw")
    with h5py.File(root / "episodes/e1/segment_000/samples.h5", "r+") as file:
        for i in range(5):
            detail = json.loads(file["details"][i])
            detail["parts"]["arms"]["left"]["force_feedback"]["effort_nm"] = -0.65
            file["details"][i] = json.dumps(detail)
    publish(root)
    assert prepare_runs([root], tmp_path / "replay", declaration(["collect"]), REWARD, SPLIT, 0.5)["ready_rows"] == {}
