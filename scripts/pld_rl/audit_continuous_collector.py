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
    with torch.inference_mode():
        means = np.concatenate(
            [
                actor(torch.tensor([x["x"] for x in poses[i : i + 128]], dtype=torch.float32), deterministic=True)[
                    0
                ].numpy()
                for i in range(0, len(poses), 128)
            ]
        )
    rng = np.random.default_rng(138)
    reasons = Counter()
    projection = Counter()
    stats = []
    for row, mean in zip(poses, means, strict=True):
        arm, d = row["arm"], row["delay_steps"]
        off = 0 if arm == "left" else 7
        b = np.asarray(row["physical_base"])
        previous = row["previous_command"]
        ctx = planning_context(b, previous, row["state"], row["gate"], arm, d, c)
        reasons[str(ctx["reason"])] += 1
        if ctx["reason"] is not None:
            continue
        pos, neg = feasible_bounds(b, previous, arm, d, c, context=ctx)
        ap, an = feasible_bounds(b, previous, arm, d, ac)
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
            if not np.array_equal(q[:d], b[:d]):
                raise ValueError("prefix changed")
            if not np.array_equal(q[ctx["end_index"] :], b[ctx["end_index"] :]):
                raise ValueError("correction after scope exit")
            dist = np.linalg.norm(xyz - xyz0, axis=-1) * 1000
            stats.append(
                {
                    "arm": arm,
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
    for arm in ("all", "left", "right"):
        selected = [v for v in stats if arm == "all" or v["arm"] == arm]
        result["groups"][arm] = {
            k: np.quantile([v[k] for v in selected], [0, 0.5, 0.95, 0.99, 1]).tolist()
            for k in (
                "peak_mm",
                "first_10_editable_peak_mm",
                "joint_peak_rad",
                "second_difference_rms_rad",
                "source_second_difference_rms_rad",
            )
        }
        result["groups"][arm]["fraction_ge_10mm"] = float(
            np.mean([v["first_10_editable_peak_mm"] >= 10 for v in selected])
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
