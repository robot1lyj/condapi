"""Geometry/command audit of continuous collection on recorded contexts (no training)."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.block_contract import observation_vector
from yam_pld_rl.block_contract import window_context
from yam_pld_rl.collector import center_latent
from yam_pld_rl.data import sha256
from yam_pld_rl.trajectory import model_fk
from yam_pld_rl.trajectory import planning_context
from yam_pld_rl.trajectory import trajectory_decode


def main():
    import torch  # noqa: PLC0415
    from yam_pld_rl.sac import SquashedActor  # noqa: PLC0415

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=16)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    m = json.loads((args.bundle / "manifest.json").read_text())
    c = m["contract"]
    ac = m["actor_contract"]
    torch.set_num_threads(2)
    actor = SquashedActor(m["observation_dim"], m["config"]).eval()
    actor.load_state_dict(torch.load(args.bundle / "actor.pt", map_location="cpu", weights_only=True))
    poses = []
    with args.records.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("receipt", {}).get("status") == "rejected":
                continue
            x = r["proposal"]
            if x["gate"].get("apply_residual"):
                poses.append(x)
    actor_inputs = []
    actor_contexts = []
    for row in poses:
        base = np.asarray(row["physical_base"])
        actx = planning_context(base, row["previous_command"], row["state"], row["gate"],
                                row["arm"], row["delay_steps"], ac) if ac.get("trajectory") else None
        ap, an = feasible_bounds(base, row["previous_command"], row["arm"], row["delay_steps"], ac, context=actx)
        height = row["base_z_m"][row["arm"]]
        actor_inputs.append(observation_vector(np.asarray(row["x"])[:1536], row["state"], base,
                            row["previous_command"], row["arm"], row["delay_steps"], row["elapsed_seconds"],
                            ap, m["normalization"], negative_scale=an,
                            context=window_context(row["gate"], row["start_tick"], height, ac["action_dt_s"])))
        actor_contexts.append((ap, an))
    with torch.inference_mode():
        means = np.concatenate(
            [
                actor(torch.tensor(np.asarray(actor_inputs[i : i + 128])), deterministic=True)[
                    0
                ].numpy()
                for i in range(0, len(poses), 128)
            ]
        )
    rng = np.random.default_rng(138)
    reasons = Counter()
    projection = Counter()
    stats = []
    for row, mean, (ap, an) in zip(poses, means, actor_contexts, strict=True):
        arm, d = row["arm"], row["delay_steps"]
        off = 0 if arm == "left" else 7
        b = np.asarray(row["physical_base"])
        previous = row["previous_command"]
        ctx = planning_context(b, previous, row["state"], row["gate"], arm, d, c)
        reasons[str(ctx["reason"])] += 1
        if ctx["reason"] is not None:
            continue
        pos, neg = feasible_bounds(b, previous, arm, d, c, context=ctx)
        radius = np.asarray(c["radius"][off : off + 7])
        center = map_action(mean, ap, an) * np.asarray(ac["radius"][off : off + 7])
        center = np.clip(center, -radius * neg, radius * pos)
        mu = center_latent(center, radius, pos, neg)
        xyz0 = model_fk().positions(b[:, off : off + 6])
        for _ in range(args.samples):
            u = np.tanh(mu + rng.normal(size=7) * np.asarray(m["collector"]["latent_std"]))
            action = map_action(u, pos, neg)
            q, info = trajectory_decode(b, action, arm, c, ctx)
            projection[info["reason"]] += 1
            xyz = model_fk().positions(q[:, off : off + 6])
            delta = q - b
            if np.any(np.abs(delta) > np.asarray(c["radius"])+1e-7):
                raise ValueError("radius violated")
            if np.any(delta > np.asarray(c.get("positive_radius", c["radius"]))+1e-7):
                raise ValueError("positive directional radius violated")
            if np.any(-delta > np.asarray(c.get("negative_radius", c["radius"]))+1e-7):
                raise ValueError("negative directional radius violated")
            if not np.array_equal(q[:d], b[:d]):
                raise ValueError("prefix changed")
            if not np.array_equal(q[ctx["end_index"] :], b[ctx["end_index"] :]):
                raise ValueError("correction after scope exit")
            np.testing.assert_array_equal(q[:, 7-off:14-off], b[:, 7-off:14-off])
            editable = (np.abs(np.asarray(ctx["bias"])).sum(1)+np.abs(ctx["weight"]))>1e-12
            check_step = editable | np.r_[False, editable[:-1]]
            active = slice(off, off+7)
            if np.any(abs(np.diff(np.vstack((previous, q)), axis=0)[check_step, active]) > np.asarray(c["max_target_step"])[active]+2e-7):
                raise ValueError("target step violated")
            check_acc = editable[2:] | editable[1:-1] | editable[:-2]
            if np.any(abs(np.diff(q, n=2, axis=0)[check_acc, active]) > np.asarray(c["trajectory"]["max_target_second_difference"])[active]+2e-7):
                raise ValueError("target second difference violated")
            dist = np.linalg.norm(xyz - xyz0, axis=-1) * 1000
            stats.append(
                {
                    "arm": arm,
                    "phase": "closed" if row["gate"].get("close_tick") is not None else "preclosure",
                    "extra_close": float(np.maximum(-delta[:, off+6], 0).max()),
                    "first_10_extra_close": float(np.maximum(-delta[d:d+10, off+6], 0).max()),
                    "extra_open": float(np.maximum(delta[:, off+6], 0).max()),
                    "minimum_gripper_target": float(q[d:ctx["end_index"], off+6].min()),
                    "peak_mm": float(dist.max()),
                    "first_10_editable_peak_mm": float(dist[d : d + 10].max()),
                    "joint_peak_rad": float(abs(delta[:, off : off + 6]).max()),
                    "second_difference_rms_rad": float(
                        np.sqrt(np.mean(np.diff(q[:, off : off + 6], n=2, axis=0) ** 2))
                    ),
                    "source_second_difference_rms_rad": float(
                        np.sqrt(np.mean(np.diff(b[:, off : off + 6], n=2, axis=0) ** 2))
                    ),
                }
            )
    result = {
        "snapshot_sha256": sha256(args.bundle / "manifest.json"),
        "records_sha256": sha256(args.records),
        "contexts": len(poses),
        "eligible_contexts": reasons["None"],
        "context_reasons": dict(reasons),
        "samples_per_context": args.samples,
        "projection": dict(projection),
        "training": False,
        "independent_recorded_context_geometry_only": True,
        "temporal_sequence_or_closed_loop_tested": False,
        "groups": {},
    }
    for group in ("all", "left", "right", "preclosure", "closed"):
        selected = [v for v in stats if group == "all" or v["arm"] == group or v["phase"] == group]
        if not selected:
            result["groups"][group] = {"count": 0}
            continue
        result["groups"][group] = {
            k: np.quantile([v[k] for v in selected], [0, 0.5, 0.95, 0.99, 1]).tolist()
            for k in (
                "peak_mm",
                "first_10_editable_peak_mm",
                "joint_peak_rad",
                "second_difference_rms_rad",
                "source_second_difference_rms_rad",
                "extra_close", "first_10_extra_close", "extra_open", "minimum_gripper_target",
            )
        }
        result["groups"][group]["count"] = len(selected)
        result["groups"][group]["fraction_ge_10mm"] = float(
            np.mean([v["first_10_editable_peak_mm"] >= 10 for v in selected])
        )
        for threshold in (.05, .10, .15):
            result["groups"][group][f"fraction_extra_close_ge_{threshold}"] = float(
                np.mean([v["extra_close"] >= threshold for v in selected]))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
