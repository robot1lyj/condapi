# ruff: noqa: E402, PLC0415
"""Server-only training of RLinf's native token head on cached frozen prefixes.

This is not a VLA trainer: only the native token encoder/decoder is loaded.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "packages/parts-rl/src", ROOT / "packages/vla-platform/src"):
    sys.path.insert(0, str(path))

from parts_rl.rlt_contract import FEATURE_EXTRACTOR
from parts_rl.rlt_contract import PREFIX_SCHEMA
from parts_rl.rlt_contract import UPSTREAM_COMMIT
from parts_rl.rlt_contract import load_token_manifest
from parts_rl.rlt_contract import token_identity
from vla_platform import parts as wire

from adapters.parts.common import require_server
from adapters.rlt.common import load_token_recipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="exact saved token snapshot directory")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--execute-on-server", action="store_true")
    args = parser.parse_args(argv)
    recipe = load_token_recipe(args.recipe)
    if args.check_only:
        print(json.dumps({"status": "recipe_checked_not_training_validated", "algorithm": recipe["algorithm"]}))
        return
    require_server(recipe, args.execute_on_server)
    if args.output.exists():
        parser.error("Output must be a new directory")
    import numpy as np
    from parts_rl.rlt_data import PrefixReplay
    from parts_rl.vendor.rlinf.rlt_token_transformer import RLTTokenTransformer
    import torch

    if not torch.cuda.is_available():
        parser.error("Server CUDA required")
    cache = PrefixReplay(recipe["prefix_ready"], prefix_seq_len=recipe["architecture"]["prefix_seq_len"])
    cache_sha = wire.sha256(recipe["prefix_ready"])
    core = {k: v for k, v in recipe.items() if k not in ("updates", "allowed_training_hosts")}
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(recipe["seed"])
    token = RLTTokenTransformer(**recipe["architecture"]).to(device="cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(token.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    start = 0
    if args.resume:
        prior = json.loads((args.resume / "training.json").read_text())
        if (
            prior.get("schema") != "yam_rlt_token_training_v1"
            or prior.get("recipe_core") != core
            or prior.get("prefix_ready_sha256") != cache_sha
            or prior.get("upstream_commit") != UPSTREAM_COMMIT
        ):
            parser.error("Resume token recipe/cache mismatch")
        manifest_path = args.resume / "token.json"
        if wire.sha256(manifest_path) != prior["token_manifest_sha256"]:
            parser.error("Resume token manifest hash mismatch")
        load_token_manifest(manifest_path)
        state_path = args.resume / "learner.pt"
        if wire.sha256(state_path) != prior["learner_sha256"]:
            parser.error("Token learner hash mismatch")
        state = torch.load(state_path, map_location="cuda", weights_only=True)
        token.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        rng.bit_generator.state = prior["numpy_rng"]
        start = prior["step"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with (args.output / "metrics.jsonl").open("w") as metrics:
        for offset in range(1, recipe["updates"] + 1):
            prefix, mask = cache.batch(recipe["batch_size"], rng)
            loss, _ = token.loss(torch.as_tensor(prefix, device="cuda"), torch.as_tensor(mask, device="cuda"))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite token reconstruction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(token.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            step = start + offset
            metrics.write(json.dumps({"step": step, "masked_reconstruction_mse": loss.item()}, allow_nan=False) + "\n")
            if offset % recipe["save_every"] == 0 or offset == recipe["updates"]:
                metrics.flush()
                snapshot = args.output / "snapshots" / f"step_{step:08d}"
                snapshot.mkdir(parents=True, exist_ok=False)
                weights = {k: t.detach().cpu() for k, t in token.state_dict().items()}
                if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in weights.values()):
                    raise ValueError("Finite FP32 token weights required")
                torch.save(
                    {
                        "schema": "yam_rlt_token_weights_v1",
                        "upstream_commit": UPSTREAM_COMMIT,
                        "architecture": recipe["architecture"],
                        "weights": weights,
                    },
                    snapshot / "token.pt",
                )
                torch.save(
                    {
                        "weights": weights,
                        "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    },
                    snapshot / "learner.pt",
                )
                manifest = {
                    "schema": "yam_rlt_token_v1",
                    "status": "trained_not_task_evaluated",
                    "upstream_commit": UPSTREAM_COMMIT,
                    "architecture": recipe["architecture"],
                    "base_identity": cache.metadata["base_identity"],
                    "prefix_schema": PREFIX_SCHEMA,
                    "feature_extractor": FEATURE_EXTRACTOR,
                    "prefix_ready_sha256": cache_sha,
                    "data_split": {k: cache.metadata[k] for k in ("train_groups", "holdout_groups")},
                    "weights": {"path": "token.pt", "sha256": wire.sha256(snapshot / "token.pt")},
                }
                manifest["feature_schema_id"] = token_identity(manifest)
                (snapshot / "token.json").write_text(json.dumps(manifest, indent=2) + "\n")
                record = {
                    "schema": "yam_rlt_token_training_v1",
                    "upstream_commit": UPSTREAM_COMMIT,
                    "recipe_core": core,
                    "step": step,
                    "prefix_ready_sha256": cache_sha,
                    "code_sha": code_sha,
                    "numpy_rng": rng.bit_generator.state,
                    "learner_sha256": wire.sha256(snapshot / "learner.pt"),
                    "token_manifest_sha256": wire.sha256(snapshot / "token.json"),
                }
                (snapshot / "training.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                pointer = args.output / ".LATEST.json.tmp"
                pointer.write_text(
                    json.dumps(
                        {
                            "path": snapshot.relative_to(args.output).as_posix(),
                            "training_sha256": wire.sha256(snapshot / "training.json"),
                        }
                    )
                    + "\n"
                )
                pointer.replace(args.output / "LATEST.json")


if __name__ == "__main__":
    main()
