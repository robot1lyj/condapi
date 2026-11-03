#!/usr/bin/env python3
"""Resolve reviewed assets into a native LeRobot config; never run training.

Standard library only. --check-template is a static preparation check, not a
LeRobot import, model load, GPU test, or data-quality certification.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/native/fineart-tshirt-10h.json"
PREPARATION = ROOT / "configs/native/fineart-tshirt-10h.preparation.json"
JOINT_NAMES = [
    *[f"left_j{i}" for i in range(1, 7)],
    "left_gripper",
    *[f"right_j{i}" for i in range(1, 7)],
    "right_gripper",
]
RENAME = {
    "observation.images.top_rgb": "observation.images.cam_high",
    "observation.images.left_rgb": "observation.images.cam_left_wrist",
    "observation.images.right_rgb": "observation.images.cam_right_wrist",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_template(config, preparation):
    p = config["policy"]
    require(preparation["status"] == "PREPARATION_NOT_TRAINING_READY", "Preparation state changed")
    require(not preparation["automatic_training_start"], "Automatic launch is forbidden")
    require(p["type"] == "fineart_vla", "Use native FineART, not the OpenPI backend")
    require(p["pretrained_revision"] == preparation["checkpoint"]["revision"], "Checkpoint pin mismatch")
    require(p["use_relative_actions"] is False, "This baseline preserves absolute actions")
    require(
        p["normalization_mapping"] == {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
        "Wrong normalization",
    )
    require(config["rename_map"] == RENAME, "Wrong YAM camera mapping")
    require((p["chunk_size"], p["n_action_steps"]) == (50, 50), "H50 required")
    require(
        (p["fast_skip_tokens"], p["max_action_tokens"], p["tokenizer_max_length"]) == (128, 384, 128),
        "Checkpoint token layout changed",
    )
    require(p["auto_fit_fast_tokenizer"] is False, "Tokenizer refit requires a separately reviewed version")
    require(p["action_tokenizer_name"] == "${CHECKPOINT_SNAPSHOT}/action_tokenizer", "Use bundled tokenizer")
    require(p["text_tokenizer_name"] == "${TEXT_TOKENIZER_SNAPSHOT}", "Use pinned text tokenizer snapshot")
    require(
        p["input_features"] is None and p["output_features"] is None and p["action_feature_names"] is None,
        "Derive YAM features from released metadata",
    )
    require(
        not p["use_peft"] and not p["freeze_vision_encoder"] and not p["train_expert_only"], "Full fine-tuning required"
    )
    require(p["recipe_path"] is None, "Recipe must be embedded, not resolved through a mutable path")
    blend = p["recipe"]["blend"]
    require(set(blend) == {"high_level_subtask", "low_level_execution"}, "Unexpected recipe branch")
    require(
        (blend["high_level_subtask"]["weight"], blend["low_level_execution"]["weight"]) == (0.3, 0.7),
        "Recipe sampling changed",
    )
    require(
        config["dataset"]["eval_split"] == 0 and config["eval_steps"] == 0,
        "Independent dev is not a random native eval_split",
    )
    require(not config["resume"] and not p["push_to_hub"], "Preparation must not resume or publish")


def class_fields(source_root, relative, name):
    tree = ast.parse((source_root / relative).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}


def check_native_schema(config, source_root):
    """Check names against pinned upstream dataclasses without importing ML code."""
    groups = [
        (config, "src/lerobot/configs/train.py", "TrainPipelineConfig"),
        (config["dataset"], "src/lerobot/configs/default.py", "DatasetConfig"),
        (config["trackio"], "src/lerobot/configs/default.py", "TrackioConfig"),
    ]
    for payload, relative, name in groups:
        unknown = set(payload) - class_fields(source_root, relative, name)
        require(not unknown, f"Unknown upstream {name} fields: {sorted(unknown)}")
    policy_fields = {"type"}
    for relative, name in [
        ("src/lerobot/configs/policies.py", "PreTrainedConfig"),
        ("src/lerobot/policies/pi05/configuration_pi05.py", "PI05Config"),
        ("src/lerobot/policies/fineart_vla/configuration_fineart_vla.py", "FineARTVLAConfig"),
    ]:
        policy_fields |= class_fields(source_root, relative, name)
    require(
        not (set(config["policy"]) - policy_fields),
        f"Unknown policy fields: {sorted(set(config['policy']) - policy_fields)}",
    )
    fineart = source_root / "src/lerobot/policies/fineart_vla/configuration_fineart_vla.py"
    tree = ast.parse(fineart.read_text(encoding="utf-8"))
    recipe_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_fineart_vla_default_recipe")
    native_recipe = ast.literal_eval(next(n.value for n in recipe_fn.body if isinstance(n, ast.Return)))
    require(config["policy"]["recipe"] == native_recipe, "Recipe differs from pinned native subtask recipe")


def pinned_checkout(source_root, preparation):
    actual = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    require(actual == preparation["code"]["commit"], f"Wrong upstream commit: {actual}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    require(not dirty.strip(), "Pinned upstream checkout has tracked modifications")


def receipt(release_file, entry):
    path = Path(entry["path"])
    if not path.is_absolute():
        path = release_file.parent / path
    require(sha256(path) == entry["sha256"], f"Receipt hash mismatch: {path}")
    data = read_json(path)
    require(data.get("status") == "PASS", f"Audit did not pass: {path}")
    return data


def dataset_metadata(entry):
    root = Path(entry["root"])
    require(root.is_absolute() and root.is_dir(), "Released dataset root must exist and be absolute")
    info = read_json(root / "meta/info.json")
    require(info.get("codebase_version", "").startswith("v3"), "Release must be native LeRobot v3")
    require(info["fps"] == 30, "This dataset recipe uses the original 30 FPS")
    features = info["features"]
    for key in ("action", "observation.state"):
        require(features[key]["shape"] == [14], f"Wrong shape: {key}")
        require(features[key].get("names") == JOINT_NAMES, f"Unconfirmed YAM order: {key}")
    require(all(key in features for key in RENAME), "Release canonical YAM RGB keys before processor rename")
    require("language_persistent" in features, "Missing persistent language feature")
    ids = entry["source_episode_indices"]
    require(
        ids and len(ids) == len(set(ids)) and all(type(i) is int and i >= 0 for i in ids),
        "Invalid source episode mapping",
    )
    require(info["total_episodes"] == len(ids), "Source mapping does not cover released episodes")
    return root, info, set(ids)


def quantile_stats(train_root):
    stats = read_json(train_root / "meta/stats.json")
    for key in ("action", "observation.state"):
        for quantile in ("q01", "q99"):
            values = stats[key][quantile]
            require(
                len(values) == 14 and all(type(v) in (int, float) and math.isfinite(v) for v in values),
                f"Invalid {key} {quantile}",
            )
        require(
            all(a <= b for a, b in zip(stats[key]["q01"], stats[key]["q99"], strict=True)), f"Reversed quantiles: {key}"
        )
    return sha256(train_root / "meta/stats.json")


def check_language_audit(release, audit, datasets):
    """Require real materialization evidence before allowing per-anchor fallback.

    The standard-library preparer verifies the scan receipt and all referenced
    data hashes; the all-row Parquet scan is performed by the data audit tool.
    A sidecar timeline alone is deliberately insufficient.
    """
    mode = release.get("language_supervision_mode", "strict-parent")
    require(
        all(audit[k] == 0 for k in ("unresolved_overlaps", "empty_instructions", "same_timestamp_conflicts")),
        "Known language timeline remains ambiguous",
    )
    if mode == "strict-parent":
        require(audit["coverage"] == {"train": 1.0, "dev": 1.0}, "Continuous language coverage not complete")
        return {
            "mode": mode,
            "effective_language_coverage": audit["coverage"],
            "action_coverage": {"train": 1.0, "dev": 1.0},
        }
    require(
        mode == "native-empty-anchor-fallback" and audit.get("mode") == mode,
        "Unsupported / unrecorded language supervision mode",
    )
    require(
        audit.get("materialization_status") == "MATERIALIZED_AUDITED",
        "Empty-anchor fallback is NOT_MATERIALIZED; real v3 Parquet audit required",
    )
    require(audit["action_coverage"] == {"train": 1.0, "dev": 1.0}, "Fallback must retain complete action coverage")
    require(
        all("language_events" in info["features"] for _, info, _ in datasets.values()),
        "Both language columns must be declared for every anchor",
    )
    expected = {(pool, eid) for pool, (_, _, ids) in datasets.items() for eid in ids}
    observed, lengths, unknown = set(), {"train": 0, "dev": 0}, {"train": 0, "dev": 0}
    for parent in audit["parents"]:
        pool, eid = parent["pool"], parent["source_episode_index"]
        require((pool, eid) in expected and (pool, eid) not in observed, "Fallback parent mapping differs from release")
        observed.add((pool, eid))
        length = parent["length"]
        require(type(length) is int and length > 0, "Invalid fallback parent length")
        previous, run, maximum, count = -1, 0, 0, 0
        for interval in parent["empty_language_anchor_intervals"]:
            start, end = interval["from_frame"], interval["to_frame"]
            require(
                interval.get("language_persistent") == [] and interval.get("language_events") == [],
                "Empty anchors require two explicit empty lists, never null rows",
            )
            require(
                type(start) is int and type(end) is int and 0 <= start < end <= length and start >= previous,
                "Invalid / overlapping empty-anchor intervals",
            )
            run = run + end - start if start == previous else end - start
            maximum, count, previous = max(maximum, run), count + end - start, end
        require(
            count / length <= 0.02 and maximum <= 30,
            "Fallback exceeds initial engineering limits (2% per parent /30 contiguous frames)",
        )
        lengths[pool] += length
        unknown[pool] += count
    require(observed == expected, "Fallback audit must cover every released parent")
    scan = audit["parquet_scan"]
    require(
        all(
            scan.get(k) is True
            for k in (
                "checked_all_rows",
                "empty_columns_exactly_at_unknown_anchors",
                "normal_anchors_keep_full_timeline",
                "only_language_columns_changed",
            )
        ),
        "Actual Parquet empty-anchor scan incomplete",
    )
    require(scan["unknown_anchor_frames"] == unknown, "Actual Parquet empty-row count differs from recorded intervals")
    expected_files = {
        str(path.resolve()) for root, _, _ in datasets.values() for path in (root / "data").rglob("*.parquet")
    }
    require(expected_files, "No materialized v3 data Parquet files")
    actual_files = set()
    for item in scan["files"]:
        path = Path(item["path"]).resolve()
        require(str(path) in expected_files and str(path) not in actual_files, "Unexpected / duplicate scanned Parquet")
        require(sha256(path) == item["sha256"], "Materialized Parquet changed after empty-anchor scan")
        actual_files.add(str(path))
    require(actual_files == expected_files, "Scan does not cover all train/dev data shards")
    effective = {}
    for pool, (_, info, _) in datasets.items():
        require(lengths[pool] == info["total_frames"], "Fallback parent frame counts differ from metadata")
        effective[pool] = 1 - unknown[pool] / lengths[pool]
        require(
            math.isclose(audit["effective_language_coverage"][pool], effective[pool], abs_tol=1e-8),
            "Effective language coverage must exclude empty anchors",
        )
        if "coverage" in audit:
            require(
                math.isclose(audit["coverage"][pool], effective[pool], abs_tol=1e-8),
                "Legacy coverage field falsely reports complete language",
            )
    return {
        "mode": mode,
        "effective_language_coverage": effective,
        "action_coverage": audit["action_coverage"],
        "unknown_anchor_frames": unknown,
        "materialization_status": audit["materialization_status"],
    }


def check_release(release_file, preparation):
    release = read_json(release_file)
    require(release.get("schema_version") == 1 and release.get("status") == "QC_RELEASED", "Data is not QC_RELEASED")
    require(release["source_revision"] == preparation["source"]["revision"], "Wrong source revision")
    require(release["task"] == preparation["source"]["task"], "Wrong source task")
    contract = release["contract"]
    require(
        contract["action_semantics"] == "absolute" and contract["state_action_names"] == JOINT_NAMES,
        "Wrong raw numeric contract",
    )
    require(
        all(contract.get(k) is True for k in ("raw_numeric_identity_verified", "episode_boundaries_preserved")),
        "Numeric identity / episode preservation not verified",
    )
    require(
        all(isinstance(contract.get(k), str) and contract[k].strip() for k in ("joint_unit", "gripper_unit")),
        "Physical units need audit evidence",
    )
    train_root, train_info, train_ids = dataset_metadata(release["datasets"]["train"])
    dev_root, dev_info, dev_ids = dataset_metadata(release["datasets"]["dev"])
    require(train_root != dev_root and not train_ids & dev_ids, "Train/dev episode leakage")
    candidate_dir = ROOT / preparation["source"]["candidate_directory"]
    pools = {}
    for pool in ("train", "dev", "reserve"):
        rows = read_json(candidate_dir / preparation["source"][f"{pool}_candidates"])
        pools[pool] = {row["episode_index"] for row in rows}
    require(train_ids <= pools["train"] | pools["reserve"], "Train includes non-candidate / dev episodes")
    require(dev_ids <= pools["dev"] | pools["reserve"], "Dev includes non-candidate / train episodes")
    stats_hash = quantile_stats(train_root)
    audits = {
        key: receipt(release_file, release["audits"][key])
        for key in (
            "numeric_identity",
            "language_timeline",
            "split_leakage",
            "normalization",
            "fast_codec",
            "text_length",
        )
    }
    numeric = audits["numeric_identity"]
    require(numeric["raw_numeric_values_unchanged"] is True, "Raw numeric identity did not pass")
    require(
        numeric["joint_unit"] == contract["joint_unit"] and numeric["gripper_unit"] == contract["gripper_unit"],
        "Unit evidence differs from declared contract",
    )
    require(
        numeric["rows_checked"] == train_info["total_frames"] + dev_info["total_frames"],
        "Numeric identity did not cover every released row",
    )
    require(set(numeric["source_episode_indices"]) == train_ids | dev_ids, "Numeric identity episode coverage differs")
    lang = audits["language_timeline"]
    release["_validated_language_audit"] = check_language_audit(
        release, lang, {"train": (train_root, train_info, train_ids), "dev": (dev_root, dev_info, dev_ids)}
    )
    norm = audits["normalization"]
    require(
        norm["fit_split"] == "train" and norm["action_semantics"] == "absolute" and norm["stats_sha256"] == stats_hash,
        "Wrong norm provenance",
    )
    require(set(norm["source_episode_indices"]) == train_ids, "Norm fit episode set differs from released train")
    fast = audits["fast_codec"]
    require(
        fast["stats_sha256"] == stats_hash and fast["audit_split"] == "train",
        "FAST audit used different normalization / split",
    )
    require(
        fast["tokenizer_sha256"]
        == preparation["checkpoint"]["sha256"]["action_tokenizer/bpe_tokenizer/tokenizer.json"],
        "FAST audit used a different tokenizer",
    )
    require((fast["fast_skip_tokens"], fast["max_action_tokens"]) == (128, 384), "FAST audit layout mismatch")
    require(fast["truncated_chunks"] == 0 and fast["validation_samples"] >= 256, "FAST coverage / length audit failed")
    require(fast["length_audit_scope"] == "all_train_anchors", "FAST length coverage is incomplete")
    require(
        0 <= fast["reconstruction_rmse"] <= 0.1 and 0 <= fast["max_dim_rmse"] <= 0.2,
        "Bundled FAST reconstruction failed; review refit separately",
    )
    text = audits["text_length"]
    require(text["max_length"] == 128 and text["truncated_samples"] == 0, "Text truncation not cleared")
    require(
        text["tokenizer_sha256"] == preparation["text_tokenizer"]["sha256"]["tokenizer.json"],
        "Text length audit used a different tokenizer",
    )
    return release, train_info, dev_info, stats_hash


def resolve(args, config, preparation):
    for key in (
        "lerobot_root",
        "checkpoint_snapshot",
        "text_tokenizer_snapshot",
        "release_manifest",
        "train_output_dir",
        "config_output",
    ):
        require(getattr(args, key) is not None, f"--{key.replace('_', '-')} is required for --resolve")
    source = args.lerobot_root.resolve()
    pinned_checkout(source, preparation)
    snapshot = args.checkpoint_snapshot.resolve()
    for relative, expected in preparation["checkpoint"]["sha256"].items():
        require(sha256(snapshot / relative) == expected, f"Checkpoint asset hash mismatch: {relative}")
    text_snapshot = args.text_tokenizer_snapshot.resolve()
    for relative, expected in preparation["text_tokenizer"]["sha256"].items():
        require(sha256(text_snapshot / relative) == expected, f"Text tokenizer asset hash mismatch: {relative}")
    for relative, expected in preparation["text_tokenizer"]["git_blob_sha1"].items():
        content = (text_snapshot / relative).read_bytes()
        actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        require(actual == expected, f"Text tokenizer Git blob mismatch: {relative}")
    release_file = args.release_manifest.resolve()
    release, train_info, dev_info, stats_hash = check_release(release_file, preparation)
    # Start with the actual checkpoint config, then apply the reviewed draft.
    # Clear ALOHA features/names so native make_policy derives target YAM metadata.
    base = read_json(snapshot / "config.json")
    base.update(copy.deepcopy(config["policy"]))
    base["pretrained_path"] = str(snapshot)
    base["action_tokenizer_name"] = str(snapshot / "action_tokenizer")
    base["text_tokenizer_name"] = str(text_snapshot)
    config["policy"] = base
    config["dataset"]["root"] = release["datasets"]["train"]["root"]
    config["dataset"]["repo_id"] = release["datasets"]["train"]["repo_id"]
    output_dir = args.train_output_dir.resolve()
    require(not output_dir.exists(), "Native training output directory already exists")
    config["output_dir"] = str(output_dir)
    check_native_schema(config, source)
    serialized = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    # ${task} and ${subtask} are native recipe bindings, not unresolved paths.
    require(
        all(
            token not in serialized
            for token in (
                "${TRAIN_DATASET_ROOT}",
                "${CHECKPOINT_SNAPSHOT}",
                "${TEXT_TOKENIZER_SNAPSHOT}",
                "${TRAIN_OUTPUT_DIR}",
            )
        ),
        "Unresolved path token",
    )
    destination = args.config_output.resolve()
    lock_path = destination.with_suffix(".lock.json")
    require(not destination.exists() and not lock_path.exists(), "Refusing to overwrite config / provenance lock")
    lock = {
        "schema_version": 1,
        "status": "NATIVE_CONFIG_RESOLVED_NOT_EXECUTION_VALIDATED",
        "training_started": False,
        "code": preparation["code"],
        "checkpoint": preparation["checkpoint"],
        "text_tokenizer": preparation["text_tokenizer"],
        "template_sha256": sha256(TEMPLATE),
        "preparation_sha256": sha256(PREPARATION),
        "release_manifest": str(release_file),
        "release_manifest_sha256": sha256(release_file),
        "train_stats_sha256": stats_hash,
        "train_frames": train_info["total_frames"],
        "language_supervision": release["_validated_language_audit"],
        "dev_frames": dev_info["total_frames"],
        "dev_root": release["datasets"]["dev"]["root"],
        "dev_in_native_training": False,
        "server_resource_validation": "PENDING",
        "resolved_config_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "native_argv": ["<FINEART_ENV_PYTHON>", "-m", "lerobot.scripts.lerobot_train", f"--config_path={destination}"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return {"status": lock["status"], "config": str(destination), "lock": str(lock_path), "training_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-template", action="store_true")
    mode.add_argument("--resolve", action="store_true")
    parser.add_argument("--lerobot-root", type=Path)
    parser.add_argument("--checkpoint-snapshot", type=Path)
    parser.add_argument("--text-tokenizer-snapshot", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--train-output-dir", type=Path)
    parser.add_argument("--config-output", type=Path)
    args = parser.parse_args()
    try:
        config, preparation = read_json(TEMPLATE), read_json(PREPARATION)
        check_template(config, preparation)
        if args.check_template:
            if args.lerobot_root:
                pinned_checkout(args.lerobot_root.resolve(), preparation)
                check_native_schema(config, args.lerobot_root.resolve())
            result = {
                "status": preparation["status"],
                "static_template_check": "PASS",
                "native_schema_checked": args.lerobot_root is not None,
                "training_started": False,
                "blockers": preparation["blockers"],
            }
        else:
            result = resolve(args, config, preparation)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, TypeError, KeyError, OSError, StopIteration, subprocess.CalledProcessError) as error:
        print(
            json.dumps({"status": "BLOCKED", "error": str(error), "training_started": False}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
