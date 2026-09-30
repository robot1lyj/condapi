# ruff: noqa: E402, PLC0415
# ML imports remain behind the server execution guard.
"""Train audited left/right PARTS residual replay on an approved CUDA server."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adapters.parts.common import load_recipe
from adapters.parts.common import require_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-on-server", action="store_true")
    parser.add_argument("--resume", type=Path, help="matching saved training directory")
    parser.add_argument("--fresh-retrain", action="store_true", help="new networks; all successes plus rho failures")
    parser.add_argument("--check-only", action="store_true", help="stdlib recipe checks; no ML imports or training")
    args = parser.parse_args(argv)
    if args.resume and (args.resume / "LATEST.json").is_file():
        latest = json.loads((args.resume / "LATEST.json").read_text())
        snapshot = (args.resume / latest["path"]).resolve()
        if not snapshot.is_relative_to(args.resume.resolve()):
            parser.error("unsafe resume snapshot path")
        if hashlib.sha256((snapshot / "training.json").read_bytes()).hexdigest() != latest["training_sha256"]:
            parser.error("resume training manifest hash mismatch")
        args.resume = snapshot
    recipe = load_recipe(args.recipe)
    if args.check_only:
        print(json.dumps({"status": "recipe_checked_not_training_validated", "algorithm": recipe["algorithm"]}))
        return
    require_server(recipe, args.execute_on_server)
    if args.resume and args.fresh_retrain:
        parser.error("resume and fresh-retrain are separate operations")
    if args.output.exists():
        parser.error("output must be a new directory")
    sys.path.insert(0, str(ROOT / "packages/parts-rl/src"))
    import numpy as np
    from parts_rl.data import Replay
    from parts_rl.data import curate_attempts
    from parts_rl.data import digest
    from parts_rl.learner import Learner
    import torch

    if not torch.cuda.is_available():
        parser.error("server CUDA unavailable")
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    rng = np.random.default_rng(recipe["seed"])
    replays, learners, curated = {}, {}, {}
    prior = json.loads((args.resume / "training.json").read_text()) if args.resume else None
    if prior:
        curated = prior.get("curated_attempts", {}).copy()
    for arm in ("left", "right"):
        ready = Path(recipe["replay"][arm]).resolve()
        metadata = json.loads(ready.read_text())
        if metadata.get("gamma") != recipe["gamma"]:
            parser.error("discounted replay gamma must match recipe gamma")
        chosen = None
        if prior:
            chosen = curated.get(arm)
        if args.fresh_retrain:
            chosen = curate_attempts(metadata["attempts"], recipe["failure_fraction"], recipe["seed"])
            curated[arm] = chosen
        replay = Replay(ready, arm, selected_attempts=chosen)
        if replays and any(
            replay.metadata[key] != replays["left"].metadata[key]
            for key in ("contract_sha", "feature_dim", "feature_schema_id", "reward_recipe", "holdout_groups")
        ):
            parser.error("left/right replay must share a physical/reward/feature contract")
        mean, std = replay.normalization()
        learner = Learner(replay.state_dim, replay.metadata["feature_dim"], recipe, mean, std, device="cuda")
        if args.resume:
            comparable = {
                key: value
                for key, value in recipe.items()
                if key not in ("replay", "updates", "allowed_training_hosts")
            }
            if prior.get("recipe_core") != comparable or prior["contract_sha"] != replay.metadata["contract_sha"]:
                parser.error("resume recipe/contract mismatch")
            if digest(ready) != prior["replays"][arm]["sha256"]:
                parser.error("resume replay changed; use a fresh run for a new dataset")
            if digest(args.resume / f"{arm}_learner.pt") != prior["learners"][arm]["sha256"]:
                parser.error("resume learner hash mismatch")
            learner.restore(torch.load(args.resume / f"{arm}_learner.pt", map_location="cuda", weights_only=True))
        replays[arm], learners[arm] = replay, learner
    if args.resume:
        prior = json.loads((args.resume / "training.json").read_text())
        rng.bit_generator.state = prior["numpy_rng"]
        checkpoint = torch.load(args.resume / "right_learner.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["cuda_rng"]])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        code_sha = None
    if learners["left"].step != learners["right"].step:
        parser.error("left/right resume steps differ")
    core = {key: value for key, value in recipe.items() if key not in ("replay", "updates", "allowed_training_hosts")}
    with (args.output / "metrics.jsonl").open("w") as metrics:
        for update in range(1, recipe["updates"] + 1):
            for arm, learner in learners.items():
                result = learner.update(replays[arm].batch(recipe["batch_size"], rng))
                metrics.write(json.dumps({"step": learner.step, "arm": arm, **result}, allow_nan=False) + "\n")
            if update % recipe["save_every"] == 0 or update == recipe["updates"]:
                metrics.flush()
                save_checkpoint(args.output, recipe, core, learners, replays, curated, rng, code_sha, digest)


def save_checkpoint(output, recipe, core, learners, replays, curated, rng, code_sha, digest):
    import torch

    # Save portable FP32 actor bundles; original Pi weights are never loaded or changed.
    run_output = output
    output = run_output / "snapshots" / f"step_{learners['left'].step:08d}"
    output.mkdir(parents=True, exist_ok=False)
    actor_members, learner_members = {}, {}
    for arm, learner in learners.items():
        checkpoint_tmp = output / f".{arm}_learner.pt.tmp"
        torch.save(learner.checkpoint(), checkpoint_tmp)
        checkpoint_tmp.replace(output / f"{arm}_learner.pt")
        learner_members[arm] = {"path": f"{arm}_learner.pt", "sha256": digest(output / f"{arm}_learner.pt")}
        actor = {
            "schema": "yam_parts_actor_v1",
            "arm": arm,
            "contract_sha": replays[arm].metadata["contract_sha"],
            "state_schema": replays[arm].metadata["state_schema"],
            "feature_schema_id": replays[arm].metadata["feature_schema_id"],
            "feature_dim": replays[arm].metadata["feature_dim"],
            "state_dim": replays[arm].state_dim,
            "hidden_dims": recipe["hidden_dims"],
            "weights": {key: value.detach().cpu() for key, value in learner.actor.state_dict().items()},
            "mean": learner.mean.detach().cpu(),
            "std": learner.std.detach().cpu(),
        }
        actor_tmp = output / f".{arm}_actor.pt.tmp"
        torch.save(actor, actor_tmp)
        actor_tmp.replace(output / f"{arm}_actor.pt")
        actor_members[arm] = {
            "path": f"{arm}_actor.pt",
            "sha256": digest(output / f"{arm}_actor.pt"),
            "step": learner.step,
        }
    record = {
        "schema": "yam_parts_training_v1",
        "status": "trained_not_task_evaluated",
        "contract_sha": replays["left"].metadata["contract_sha"],
        "recipe_core": core,
        "state_schema": replays["left"].metadata["state_schema"],
        "feature_schema_id": replays["left"].metadata["feature_schema_id"],
        "feature_dim": replays["left"].metadata["feature_dim"],
        "reward_recipe": replays["left"].metadata["reward_recipe"],
        "code_sha": code_sha,
        "numpy_rng": rng.bit_generator.state,
        "curated_attempts": curated,
        "actors": actor_members,
        "learners": learner_members,
        "replays": {
            arm: {
                "path": recipe["replay"][arm],
                "sha256": digest(recipe["replay"][arm]),
                "rows": len(replay),
                "sources": replay.origins,
            }
            for arm, replay in replays.items()
        },
    }
    temp = output / ".training.json.tmp"
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(output / "training.json")
    latest = run_output / ".LATEST.json.tmp"
    latest.write_text(
        json.dumps(
            {"path": output.relative_to(run_output).as_posix(), "training_sha256": digest(output / "training.json")}
        )
        + "\n"
    )
    latest.replace(run_output / "LATEST.json")


if __name__ == "__main__":
    main()
