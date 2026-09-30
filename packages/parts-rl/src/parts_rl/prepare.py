"""Build decision replay from immutable client HDF5/JSONL packages.

No model inference or optimizer runs here. Partial/canceled attempts are retained
in the audit report, never repaired by fabricating actions or next states.
"""

import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from vla_platform import parts as wire

from .data import digest
from .data import discounted_return
from .state import NON_VISUAL_DIM
from .state import STATE_SCHEMA
from .state import encode_state


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def tick_rows(root):
    import h5py  # noqa: PLC0415 - data dependency is optional for pure helpers.

    rows = {}
    for manifest_path in sorted((root / "episodes").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "yam_hil_v2":
            raise ValueError("Unknown raw episode schema")
        for segment in manifest["segments"]:
            path = (manifest_path.parent / segment["path"] / "samples.h5").resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Unsafe/missing samples file")
            with h5py.File(path, "r") as file:
                count = len(file["tick"])
                if int(file["committed_rows"][()]) != count or segment["steps"] != count:
                    raise ValueError("Uncommitted/incomplete raw segment")
                for i in range(count):
                    value = file["details"][i]
                    row = json.loads(value.decode() if isinstance(value, bytes) else value)
                    for key in (
                        "tick",
                        "epoch",
                        "time",
                        "submitted_action",
                        "measured_state",
                        "obs_id",
                        "observation_state",
                    ):
                        valid = key in file and (key + "__valid" not in file or bool(file[key + "__valid"][i]))
                        if not valid and key in ("tick", "epoch", "time"):
                            raise ValueError(f"Invalid raw {key}")
                        row[key] = file[key][i].tolist() if valid else None
                    key = (row["epoch"], row["tick"])
                    if key in rows:
                        raise ValueError("Duplicate epoch/tick across raw episodes")
                    row["source"] = {"path": path.relative_to(root).as_posix(), "row": i}
                    rows[key] = row
    return rows


def verify_observation(root, request, rows):
    refs = wire.object_value(request.get("video_refs"), "video_refs")
    context = request["context"]
    observed = rows[(context["epoch"], refs["tick"])]
    if (
        observed["obs_id"] != context["observation_id"]
        or observed.get("observation_valid") is not True
        or not np.array_equal(observed["observation_state"], request["observation_state"])
    ):
        raise ValueError("Request observation/frame/state source mismatch")
    episode = (root / refs["episode_path"]).resolve()
    if not episode.is_relative_to(root):
        raise ValueError("Unsafe video source")
    manifest = json.loads((episode / "manifest.json").read_text())
    segments = [value for value in manifest["segments"] if value["path"] == refs["segment"]]
    if len(segments) != 1 or manifest["episode_id"] != refs["episode_id"]:
        raise ValueError("Video segment identity mismatch")
    segment = segments[0]
    for role in ("top", "left", "right"):
        index = wire.integer(refs["frame_indices"][role], "video frame")
        path = (episode / segment["path"] / (role + ".mp4")).resolve()
        if not path.is_relative_to(root) or not path.is_file() or index >= segment["video_frames"][role]:
            raise ValueError("Request video frame missing/out of bounds")


def force_fresh(value, now, config):
    if not value or value.get("valid") is not True:
        return False
    try:
        age = wire.finite(value["feedback_age_s"], "feedback_age_s")
        sampled = wire.finite(value["sampled_at"], "sampled_at")
        wire.finite(value["effort_nm"], "effort_nm")
        wire.finite(value["sdk_updated_at"], "sdk_updated_at")
        maximum = wire.finite(config["max_feedback_age_s"], "max_feedback_age_s")
    except (ValueError, KeyError, TypeError):
        return False
    return 0 <= now - sampled <= maximum and 0 <= age + now - sampled <= maximum


def verify_success(attempt, rows, terminal_event, config):
    closure = attempt.get("closure_tick")
    if type(closure) is not int or closure > attempt["terminal_tick"]:
        raise ValueError("Success requires closure before terminal")
    started, last_time, last_sdk, confirmed = None, None, None, False
    for tick in range(closure + 1, attempt["terminal_tick"] + 1):
        row = rows[(attempt["epoch"], tick)]
        feedback = row.get("parts", {}).get("arms", {}).get(attempt["arm"], {}).get("force_feedback")
        now = row["time"]
        if not force_fresh(feedback, now, config) or abs(feedback["effort_nm"]) <= 0.65:
            started, last_time = None, None
            continue
        sdk = feedback["sdk_updated_at"]
        if last_sdk is not None and sdk <= last_sdk:
            started, last_time = None, None
            continue
        last_sdk = sdk
        if last_time is None or now - last_time > config["max_confirmation_gap_s"]:
            started = now
        last_time = now
        if now - started >= config["confirm_s"]:
            confirmed = True
    feedback = terminal_event.get("feedback")
    if (
        not confirmed
        or not force_fresh(feedback, terminal_event["time"], config)
        or abs(feedback["effort_nm"]) <= 0.65
        or attempt.get("handback_effective_tick") != attempt["terminal_tick"]
    ):
        raise ValueError("Success feedback/confirmation/handback does not verify")


def prepare_runs(run_paths, output, declaration, recipe, split, gamma):
    import h5py  # noqa: PLC0415 - data dependency is optional for pure helpers.

    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Replay output must be a new directory")
    wire.contract(declaration)
    wire.height_reward(recipe, 0)
    if not 0 < gamma <= 1:
        raise ValueError("Invalid per-control-tick gamma")
    train_groups, holdout_groups = set(split["train_groups"]), set(split["holdout_groups"])
    if not train_groups or not holdout_groups or train_groups & holdout_groups:
        raise ValueError("Explicit disjoint train/holdout groups required")
    examples = {arm: [] for arm in wire.INDICES}
    sources, audit, attempt_labels = [], [], {arm: [] for arm in wire.INDICES}
    feature_dim = None
    run_ids = set()
    for run_path in run_paths:
        root = Path(run_path).resolve()
        file_audit = wire.audit_publication(root)
        run = json.loads((root / "run.json").read_text())
        if run["run_id"] in run_ids:
            raise ValueError("Duplicate run identity; do not import a retransmission twice")
        run_ids.add(run["run_id"])
        group_id = run.get("layout_group_id")
        if group_id in holdout_groups:
            audit.append({"run_id": run["run_id"], "excluded": "holdout"})
            continue
        if (
            group_id not in train_groups
            or run.get("split_role") != "train"
            or run.get("mode") != "collect"
            or run.get("contract_sha") != declaration["contract_sha"]
        ):
            raise ValueError("Run group/role/mode/contract mismatch")
        if run.get("config", {}).get("reward") != recipe:
            raise ValueError("Recorded reward recipe differs from preparation recipe")
        rows = tick_rows(root)
        event_rows = jsonl(root / "events.jsonl")
        events = {item["event_id"]: item for item in event_rows}
        if len(events) != len(event_rows):
            raise ValueError("Duplicate reward event IDs")
        requests = jsonl(root / "requests.jsonl")
        attempts = jsonl(root / "attempts.jsonl")
        if len({item["attempt_id"] for item in attempts}) != len(attempts):
            raise ValueError("Duplicate attempt identities")
        sources.append(
            {"run_id": run["run_id"], "root": str(root), "publication_sha256": file_audit["publication_sha256"]}
        )
        with h5py.File(root / "requests.h5", "r") as file:
            for attempt in attempts:
                arm = attempt.get("arm")
                try:
                    if arm not in examples or attempt.get("result") not in ("success", "failure"):
                        raise ValueError("canceled/unknown attempt")
                    start, stop = attempt["entry_tick"], attempt["terminal_tick"]
                    epoch = attempt["epoch"]
                    sequence = [rows[(epoch, tick)] for tick in range(start, stop + 1)]
                    if any(b["time"] <= a["time"] for a, b in itertools.pairwise(sequence)):
                        raise ValueError("Attempt clock not monotonic")
                    terminal_event = events[attempt["event_id"]]
                    if (
                        terminal_event.get("attempt_id") != attempt["attempt_id"]
                        or terminal_event.get("kind") != attempt["result"]
                        or terminal_event.get("tick") != stop
                    ):
                        raise ValueError("Attempt terminal event mismatch")
                    if attempt["result"] == "success":
                        verify_success(attempt, rows, terminal_event, run["config"])
                    if attempt.get("grasp_reward") != int(attempt["result"] == "success"):
                        raise ValueError("Attempt result reward mismatch")
                    accepted = []
                    for request in requests:
                        sent = copy.deepcopy(request.get("parts_request", {}))
                        context = request["context"]
                        k = context["observation_policy_tick"]
                        arm_state = sent.get("arms", {}).get(arm, {})
                        active_decision = (
                            arm_state.get("attempt_id") == attempt["attempt_id"]
                            and sent.get("active_arm") == arm
                            and start <= k < stop
                        )
                        pre_entry_decision = (
                            k < start
                            and arm_state.get("attempt_id") is None
                            and arm_state.get("phase") == "READY"
                            and sent.get("active_arm") is None
                        )
                        if (
                            context["epoch"] != epoch
                            or not (active_decision or pre_entry_decision)
                            or request.get("discarded")
                            or request.get("error")
                            or request.get("parts_error")
                        ):
                            continue
                        wire.request(sent, declaration)
                        if sent["context"] != context or sent["mode"] != "collect":
                            raise ValueError("Recorded request context/mode mismatch")
                        used = []
                        for row in sequence:
                            part = row.get("parts") or {}
                            selection = part.get("selection") or {}
                            original = selection.get("request") or {}
                            if (
                                part.get("attempt_id") == attempt["attempt_id"]
                                and original.get("epoch") == epoch
                                and original.get("request_id") == context["request_id"]
                                and part.get("candidate_ref") is not None
                            ):
                                used.append(row)
                        if not used:
                            continue
                        verify_observation(root, request, rows)
                        group = file[request["hdf5_group"]]
                        scheduler = sent.setdefault("scheduler", {})
                        for name in ("targets", "valid_mask", "committed_mask"):
                            if f"scheduler/{name}" not in group:
                                raise ValueError("Missing queue state array")
                            scheduler[name] = np.asarray(group[f"scheduler/{name}"])
                        if "features/z" not in group:
                            raise ValueError("Request lacks resolved visual feature array")
                        z = np.asarray(group["features/z"], dtype=np.float32)
                        if feature_dim is None:
                            feature_dim = z.size
                        if (
                            z.shape != (feature_dim,)
                            or request.get("features", {}).get("feature_schema_id") != declaration["feature_schema_id"]
                        ):
                            raise ValueError("Visual feature schema mismatch")
                        if digest_feature(z) != request["features"].get("sha256"):
                            raise ValueError("Visual feature hash mismatch")
                        base = np.asarray(group["actions_native"])
                        u = np.asarray(group[f"{arm}/u"])
                        bounds = np.asarray(group[f"{arm}/B_rad"])
                        editable = np.asarray(group[f"{arm}/editable_mask"])
                        if (
                            u.shape != (50, 6)
                            or not np.isfinite(u).all()
                            or np.any(abs(u) > 1)
                            or editable.shape != (50,)
                            or editable.dtype != np.bool_
                            or not np.array_equal(bounds, declaration["per_arm"][arm]["B_rad"])
                        ):
                            raise ValueError("Request candidate contract mismatch")
                        prefix = (
                            np.asarray(group["committed_prefix"]) if "committed_prefix" in group else np.zeros((0, 14))
                        )
                        if prefix.ndim != 2 or prefix.shape[1] != 14 or np.any(editable[: len(prefix)]):
                            raise ValueError("Editable committed prefix")
                        if not np.array_equal(base[: len(prefix)], prefix):
                            raise ValueError("RTC prefix changed")
                        state = encode_state(z, request["observation_state"], base, sent, arm, feature_dim=feature_dim)
                        accepted.append((k, request, state, u, editable, bounds, used))
                    accepted.sort(key=lambda item: item[0])
                    if not accepted or len({item[0] for item in accepted}) != len(accepted):
                        raise ValueError("No unique accepted active decisions")
                    pending = []
                    for index, (k, _request, state, u, editable, bounds, used) in enumerate(accepted):
                        following = accepted[index + 1] if index + 1 < len(accepted) else None
                        next_tick = following[0] if following is not None else stop
                        if next_tick <= k:
                            raise ValueError("Invalid decision duration")
                        interval = [rows[(epoch, tick)] for tick in range(k, next_tick + 1)]
                        if any(b["time"] <= a["time"] for a, b in itertools.pairwise(interval)):
                            raise ValueError("Decision interval clock/gap invalid")
                        mask = np.broadcast_to(editable[:, None], (50, 6)) & (bounds > 0)[None]
                        executed = np.zeros((50, 6), np.bool_)
                        for row in used:
                            part = row["parts"]
                            selection = part["selection"]
                            model_index = selection["model_index"]
                            candidate_ref = part["candidate_ref"]
                            if (
                                candidate_ref.get("request_epoch") != epoch
                                or candidate_ref.get("request_id") != _request["context"]["request_id"]
                                or candidate_ref.get("model_index") != model_index
                                or candidate_ref.get("arm") != arm
                                or candidate_ref.get("behavior_snapshot_id") != _request["behavior_snapshot_id"]
                                or candidate_ref.get("actor_snapshot_id")
                                != _request["candidate_metadata"][arm]["actor_snapshot_id"]
                            ):
                                raise ValueError("Executed candidate snapshot/source mismatch")
                            if selection["target_tick"] != k + model_index or row["tick"] != selection["target_tick"]:
                                raise ValueError("Applied candidate target/source mismatch")
                            if not 0 <= model_index < 50 or not editable[model_index]:
                                raise ValueError("Applied candidate was not editable")
                            submitted = np.asarray(row["submitted_action"])
                            physical = np.asarray(part["physical_residual_rad"])
                            if (
                                submitted.shape != (14,)
                                or physical.shape != (14,)
                                or not np.allclose(
                                    submitted, np.asarray(part["base_target"]) + physical, atol=1e-7, rtol=0
                                )
                                or np.any(np.delete(physical, wire.INDICES[arm]) != 0)
                            ):
                                raise ValueError("Final submitted residual/source mismatch")
                            if k <= row["tick"] < next_tick:
                                executed[model_index] = bounds > 0
                        rewards = []
                        for tick in range(k + 1, next_tick + 1):
                            if tick <= start:
                                # Feedback before the first active command has no RL reward.
                                rewards.append(0.0)
                                continue
                            part = rows[(epoch, tick)].get("parts") or {}
                            height = part.get("arms", {}).get(arm, {})
                            if (
                                height.get("height_valid") is not True
                                or height.get("error_m") is None
                                or part.get("recording_error")
                            ):
                                raise ValueError("Height/reward observation gap")
                            grasp = int(tick == stop and attempt["result"] == "success")
                            previous_part = rows[(epoch, tick - 1)].get("parts") or {}
                            phase = previous_part.get("phase") or previous_part.get("arms", {}).get(arm, {}).get(
                                "phase"
                            )
                            rewards.append(
                                wire.height_reward(recipe, height["error_m"], grasp, phase=phase)["total_reward"]
                            )
                        next_mask = (
                            np.zeros((50, 6), np.bool_)
                            if following is None
                            else (np.broadcast_to(following[4][:, None], (50, 6)) & (following[5] > 0)[None])
                        )
                        pending.append(
                            {
                                "state": state,
                                "next_state": np.zeros_like(state) if following is None else following[2],
                                "action": u.ravel(),
                                "action_mask": mask.ravel(),
                                "executed_mask": executed.ravel(),
                                "next_action_mask": next_mask.ravel(),
                                "reward": discounted_return(rewards, gamma),
                                "elapsed_steps": next_tick - k,
                                "bootstrap": following is not None,
                                "success": attempt["result"] == "success",
                                "attempt_id": run["run_id"] + ":" + attempt["attempt_id"],
                                "group_id": group_id,
                            }
                        )
                    examples[arm].extend(pending)
                    attempt_labels[arm].append(
                        {
                            "attempt_id": run["run_id"] + ":" + attempt["attempt_id"],
                            "result": attempt["result"],
                            "split_role": "train",
                            "mock": False,
                        }
                    )
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    audit.append(
                        {"run_id": run["run_id"], "attempt_id": attempt.get("attempt_id"), "excluded": str(exc)}
                    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(
        json.dumps({"sources": sources, "excluded": audit}, ensure_ascii=False, indent=2) + "\n"
    )
    ready = {}
    for arm, records in examples.items():
        if not records:
            continue
        directory = output / arm
        directory.mkdir()
        arrays = {key: np.asarray([record[key] for record in records]) for key in records[0]}
        arrays["state"] = arrays["state"].astype(np.float32)
        arrays["next_state"] = arrays["next_state"].astype(np.float32)
        arrays["action"] = arrays["action"].astype(np.float32)
        shard_path = directory / "transitions.npz"
        np.savez_compressed(shard_path, **arrays)
        metadata = {
            "schema": "yam_parts_replay_v1",
            "status": "READY",
            "arm": arm,
            "mock": False,
            "split_role": "train",
            "state_schema": STATE_SCHEMA,
            "feature_dim": feature_dim,
            "state_dim": feature_dim + NON_VISUAL_DIM,
            "feature_schema_id": declaration["feature_schema_id"],
            "contract_sha": declaration["contract_sha"],
            "action_semantics": "accepted_candidate_plan_v1",
            "gamma": gamma,
            "reward_recipe": recipe,
            "sources": sources,
            "holdout_groups": sorted(holdout_groups),
            "attempts": attempt_labels[arm],
            "terminal_next_state": "unused_zero_placeholder_no_bootstrap",
            "shards": [{"path": shard_path.name, "sha256": digest(shard_path)}],
        }
        (directory / "READY.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        ready[arm] = len(records)
    return {
        "output": str(output),
        "ready_rows": ready,
        "excluded_attempts": len(audit),
        "status": "replay_prepared" if ready else "no_trainable_decisions",
    }


def digest_feature(z):
    return hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
