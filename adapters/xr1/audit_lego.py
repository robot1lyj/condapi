"""Audit every derived frame against the immutable LeRobot row and nominal YAM FK.

This audits the documented training contract, not independent raw-to-port units
or physical robot calibration. Those limitations remain explicit in the report.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

import av
import numpy as np

from adapters.xr1.prepare_lego import IMAGE_KEYS
from adapters.xr1.prepare_lego import ForwardKinematics
from adapters.xr1.prepare_lego import _episode_metadata
from adapters.xr1.prepare_lego import _path
from adapters.xr1.prepare_lego import _row_data
from adapters.xr1.prepare_lego import digest
from adapters.xr1.prepare_lego import verify_fk_bundle


def audit(manifest_path, stats_path, output):
    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["source_repo"])
    source_path = root / "conversion_manifest.json"
    source = json.loads(source_path.read_text())
    contract = source["contract"]
    if (manifest["split"] != "train" or digest(source_path) != manifest["source_manifest_sha256"]
            or contract["joint_unit"] != "radian" or contract["action_mode"] != "absolute"
            or contract["gripper_unit"] != "normalized_0_closed_1_open" or not contract.get("sources")):
        raise ValueError("Missing or changed documented YAM units/action contract")
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != 30 or digest(root / "meta/info.json") != manifest["source_info_sha256"]:
        raise ValueError("Source metadata changed")
    xml = Path(manifest["fk_model"])
    if digest(xml) != manifest["fk_model_sha256"] or verify_fk_bundle(xml) != manifest["fk_bundle_sha256"]:
        raise ValueError("FK identity changed")
    stats = json.loads(stats_path.read_text())
    expected = {e["json"]: e["json_sha256"] for e in manifest["episodes"]}
    if stats["train_manifest_sha256"] != digest(manifest_path) or stats["train_json_sha256"] != expected:
        raise ValueError("Norm is not bound to this exact training split")
    metadata = _episode_metadata(root, info)
    fk = ForwardKinematics(xml)
    errors = {"fk_position_m": 0.0, "fk_rotation_matrix": 0.0, "source_value": 0.0}
    total = 0

    def compare(actual, wanted, label, tolerance=1e-9):
        actual = np.asarray(actual, dtype=np.float64)
        if actual.shape != wanted.shape or not np.isfinite(actual).all():
            raise ValueError(f"Invalid {label} shape or values")
        error = float(np.max(np.abs(actual - wanted)))
        errors[label] = max(errors[label], error)
        if error > tolerance:
            raise ValueError(f"{label} differs from the source: {error}")

    for index, entry in enumerate(manifest["episodes"]):
        path = Path(entry["json"])
        if digest(path) != entry["json_sha256"]:
            raise ValueError(f"Derived JSON changed: {path}")
        episode = json.loads(path.read_text())
        provenance = json.loads((path.parent / "provenance.json").read_text())
        episode_id = entry["episode_index"]
        row = metadata[episode_id]
        parquet, timestamps, state, action = _row_data(root, info, row, episode_id)
        if (episode["num_frames"] != entry["frames"] or entry["frames"] != len(state)
                or provenance["source_parquet"] != str(parquet)
                or provenance["source_episode_index"] != episode_id
                or provenance["fk_model_sha256"] != manifest["fk_model_sha256"]):
            raise ValueError("Derived/source episode identity differs")
        compare(provenance["source_timestamps"], timestamps, "source_value")
        for role, key in IMAGE_KEYS.items():
            prefix = f"videos/{key}"
            start = round(row[f"{prefix}/from_timestamp"] * info["fps"])
            original = _path(root, info["video_path"], row[f"{prefix}/chunk_index"], row[f"{prefix}/file_index"], key)
            view = {"top": "ego", "left": "wrist_left", "right": "wrist_right"}[role]
            video_entry = episode["observations"][view]
            video = path.parent / f"{role}.mp4"
            if (provenance["video_start"][role] != start or entry["video_start"][role] != start
                    or provenance["source_video"][role] != str(original)
                    or video_entry != [{"path": str(video), "start": 0, "crop_bbox": None}]
                    or digest(video) != provenance["derived_video_sha256"][role]):
                raise ValueError("Video identity/range changed")
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                if stream.frames != len(state) or float(stream.average_rate) != 30:
                    raise ValueError("Derived video frame count/rate differs")
        for group, joints in (("proprios", state), ("actions", action)):
            # Published radians/normalized gripper must be numerically plausible;
            # this consistency check is not an independent raw unit measurement.
            if np.max(np.abs(joints[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]])) > 4 * np.pi:
                raise ValueError("Joint values inconsistent with documented radians")
            if np.min(joints[:, [6, 13]]) < -0.05 or np.max(joints[:, [6, 13]]) > 1.05:
                raise ValueError("Gripper values inconsistent with documented normalization")
            poses = fk.poses(joints)
            for arm, name in enumerate(("left", "right")):
                compare(episode[group][f"{name}_ee_pos"], poses[:, arm, :3, 3], "fk_position_m")
                compare(episode[group][f"{name}_ee_rotm"], poses[:, arm, :3, :3].reshape(len(joints), 9), "fk_rotation_matrix")
                compare(episode[group][f"{name}_gripper_pos"], joints[:, arm * 7 + 6, None], "source_value")
                if group == "proprios":
                    compare(episode[group][f"{name}_arm_joint"], joints[:, arm * 7:arm * 7 + 6], "source_value")
        total += len(state)
        if (index + 1) % 25 == 0:
            print(json.dumps({"audited_episodes": index + 1, "frames": total}), flush=True)
    report = {
        "schema_version": 1, "source_contract": "yam-bimanual-v1", "source_dataset": str(root),
        "source_revision": manifest["source_manifest_sha256"], "train_episode_ids": manifest["episode_ids"],
        "fk_model": str(xml), "fk_model_sha256": manifest["fk_model_sha256"],
        "train_json_sha256": expected, "frames_and_units_verified": True, "target_alignment_verified": True,
        "verification_scope": "nominal_training_contract: documented units, exact source-row preservation and FK; not physical calibration",
        "raw_to_port_unit_preservation": "publisher/documented-port inference; not independently measured",
        "robot_calibration_verified": False, "raw_command_capture_timing_verified": False,
        "unit_contract": contract, "frames": total, "episodes": len(expected), "max_errors": errors,
        "target_alignment_scope": "action FK at the same LeRobot row; no shift, rescaling or next-state substitution",
        "video_scope": "all derived hashes/frame counts and source-offset metadata; full source pixel equivalence not claimed",
        "auditor_sha256": digest(Path(__file__)), "stats_sha256": digest(stats_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, output)  # exclusive publication; never overwrite an audit
    finally:
        temporary.unlink()
    print(json.dumps({"status": "nominal_training_audit_complete", "output": str(output), "frames": total}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.stats, args.output)


if __name__ == "__main__":
    main()
