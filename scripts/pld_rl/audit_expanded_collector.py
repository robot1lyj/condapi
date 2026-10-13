"""Recorded-pose geometry with the frozen learned actor; inference only, no training."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages/pld-rl/src"))
from yam_pld_rl.block_contract import decode
from yam_pld_rl.block_contract import envelope_weights
from yam_pld_rl.block_contract import feasible_bounds
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.block_contract import map_action
from yam_pld_rl.collector import center_latent
from yam_pld_rl.collector import validate_collector
from yam_pld_rl.data import sha256
from yam_pld_rl.kinematics import YamKinematics


def main():
    import torch  # noqa: PLC0415 -- run actor inference on Thor, never local training
    from yam_pld_rl.sac import SquashedActor  # noqa: PLC0415

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--learning-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=64)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    m = json.loads((args.bundle/"manifest.json").read_text())
    validate_collector(m)
    if sha256(args.bundle/"actor.pt") != m["actor_sha256"]:
        raise ValueError("actor hash mismatch")
    c, ac = m["contract"], m["actor_contract"]
    z = np.load(args.learning_replay, allow_pickle=False)
    wanted = set(zip(z["attempt"], z["decision"], strict=True))
    poses = []
    with args.records.open() as stream:
        for line in stream:
            r = json.loads(line)["proposal"]
            key = "/".join((r["run_id"], r["episode_id"], r["attempt_id"]))
            if (key, r["decision_id"]) not in wanted:
                continue
            if r["contract_sha256"] != fingerprint(ac) or not r["gate"]["apply_residual"]:
                raise ValueError("source is not controllable parent coordinates")
            poses.append(r)
    if len(poses) != len(wanted):
        raise ValueError("missing or duplicate learning decisions")
    torch.set_num_threads(2)
    actor = SquashedActor(m["observation_dim"], m["config"]).eval()
    actor.load_state_dict(torch.load(args.bundle/"actor.pt", map_location="cpu", weights_only=True))
    x = np.asarray([r["x"] for r in poses], np.float32)
    with torch.inference_mode():
        means = np.concatenate([actor(torch.from_numpy(x[i:i+256]), deterministic=True)[0].numpy()
                                for i in range(0, len(x), 256)])
    fk = YamKinematics(args.bundle/"fk.xml")
    rng = np.random.default_rng(138)
    groups = {name: [] for name in ("old", "expanded")}
    scales, center_error, joint_rms = [], [], []
    for r, mean in zip(poses, means, strict=True):
        arm, d = r["arm"], r["delay_steps"]
        offset = 0 if arm == "left" else 7
        base = np.asarray(r.get("physical_base", r["base"]))
        ap, an = feasible_bounds(base, r["previous_command"], arm, d, ac)
        pos, neg = feasible_bounds(base, r["previous_command"], arm, d, c)
        physical = map_action(mean, ap, an)*np.asarray(ac["radius"][offset:offset+7])
        latent = center_latent(physical, c["radius"][offset:offset+7], pos, neg)
        old = decode(base, map_action(mean, ap, an), arm, d, ac)
        zero_noise = decode(base, map_action(np.tanh(latent), pos, neg), arm, d, c)
        center_error.append(float(np.abs(old-zero_noise).max()))
        if center_error[-1] > 1e-7:
            raise ValueError("parent physical center changed")
        scales.append(np.minimum(pos, neg)[:6])
        noise = rng.normal(size=(args.samples, 7))
        for name, contract, mu, std, positive, negative in (
            ("old", ac, np.arctanh(np.clip(mean, -.999999, .999999)), [.2]*7, ap, an),
            ("expanded", c, latent, m["collector"]["latent_std"], pos, neg),
        ):
            u = np.tanh(mu+noise*np.asarray(std))
            radius = np.asarray(contract["radius"][offset:offset+7])
            residual = u*np.where(u<0, negative, positive)*radius
            w = envelope_weights(d, contract)
            nonzero = w>0
            q = base[nonzero, offset:offset+6]
            targets = q[None]+w[nonzero, None]*residual[:, None, :6]
            delta = fk.positions(targets.reshape(-1, 6)).reshape(args.samples, len(q), 3)-fk.positions(q)
            dist = np.linalg.norm(delta, axis=-1)
            peak = np.argmax(dist, axis=1)
            xyz = delta[np.arange(args.samples), peak]*1000
            groups[name].append((arm, r["gate"].get("close_tick") is not None, xyz))
            if name == "expanded":
                joint_rms.append(np.mean(residual[:, :6]**2))
    report = {"snapshot_sha256": sha256(args.bundle/"manifest.json"), "actor_sha256": m["actor_sha256"],
              "records_sha256": sha256(args.records), "learning_replay_sha256": sha256(args.learning_replay),
              "contexts": len(poses), "episodes": len({r["episode_id"] for r in poses}),
              "attempts": len({(r["episode_id"], r["attempt_id"]) for r in poses}),
              "samples_per_context": args.samples, "seed": 138, "training": False,
              "max_parent_zero_noise_decoder_error": max(center_error),
              "fraction_contexts_any_joint_support_reduced": float(np.mean(np.any(np.asarray(scales)<1-1e-6, axis=1))),
              "minimum_joint_scale": float(np.min(scales)), "expanded_peak_joint_rms_rad": float(np.sqrt(np.mean(joint_rms))),
              "limits": "Recorded target FK only; no counterfactual measured trajectory, contact or success-rate evidence.",
              "groups": {}}
    for name, entries in groups.items():
        report["groups"][name] = {}
        for label in ("all", "left", "right", "before_close", "after_close"):
            chosen = [v for arm, closed, v in entries if label in {"all", arm, "after_close" if closed else "before_close"}]
            xyz = np.concatenate(chosen)
            dist = np.linalg.norm(xyz, axis=1)
            report["groups"][name][label] = {
                "contexts": len(chosen), "target_peak_mm_p50_p95_p99_max": np.quantile(dist, [.5,.95,.99,1]).tolist(),
                "horizontal_mm_p50_p95_max": np.quantile(np.linalg.norm(xyz[:, :2], axis=1), [.5,.95,1]).tolist(),
                "signed_vertical_mm_p05_p50_p95": np.quantile(xyz[:, 2], [.05,.5,.95]).tolist(),
                "fraction_ge_10mm": float(np.mean(dist>=10)), "fraction_ge_40mm": float(np.mean(dist>=40)),
                "fraction_ge_50mm": float(np.mean(dist>=50))}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
