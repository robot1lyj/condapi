# ruff: noqa: E402, PLC0415
"""Export immutable training observations from published client video/frame refs.

CPU data conversion only. No policy inference or optimizer is constructed.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/parts-rl/src"))
sys.path.insert(0, str(ROOT / "packages/vla-platform/src"))

import numpy as np
from parts_rl.prepare import jsonl
from parts_rl.prepare import tick_rows
from parts_rl.prepare import verify_observation
from parts_rl.rlt_data import training_groups
from vla_platform import parts as wire


def video_frame(path, index):
    import av

    with av.open(str(path)) as stream:
        for i, frame in enumerate(stream.decode(video=0)):
            if i == index:
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"Missing video frame {index}: {path}")


def prepare(run_paths, output, split, *, prompt):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Observations output must be a new directory")
    wire.text(prompt, "original policy task prompt")
    train, holdout = training_groups(split)
    observations, sources, excluded, seen = [], [], [], set()
    output.mkdir(parents=True, exist_ok=False)
    for run_path in run_paths:
        root = Path(run_path).resolve()
        audited = wire.audit_publication(root)
        run = json.loads((root / "run.json").read_text())
        group = run.get("layout_group_id")
        if group in holdout:
            excluded.append({"root": str(root), "reason": "holdout"})
            continue
        if group not in train or run.get("split_role") != "train" or run.get("mode") == "eval":
            raise ValueError("Raw publication training split mismatch")
        rows = tick_rows(root)
        sources.append({"root": str(root), "publication_sha256": audited["publication_sha256"]})
        for request in jsonl(root / "requests.jsonl"):
            context = request["context"]
            key = json.dumps(
                [run["run_id"], context["session_id"], context["epoch"], context["observation_id"]],
                separators=(",", ":"),
            )
            if key in seen:
                continue
            verify_observation(root, request, rows)
            refs = request["video_refs"]
            episode = (root / refs["episode_path"]).resolve()
            data = {"observation.state": np.asarray(request["observation_state"]), "prompt": np.asarray(prompt)}
            for role in ("top", "left", "right"):
                path = (episode / refs["segment"] / (role + ".mp4")).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Unsafe video path")
                data[f"observation.images.{role}_rgb"] = video_frame(path, refs["frame_indices"][role])
            relative = f"observation_{len(observations):08d}.npz"
            np.savez_compressed(output / relative, **data)
            observations.append(
                {
                    "path": relative,
                    "sha256": wire.sha256(output / relative),
                    "observation_key": key,
                    "group_id": group,
                    "context": context,
                    "video_refs": refs,
                }
            )
            seen.add(key)
    if not observations:
        raise ValueError("No real train observations exported; output is not READY")
    value = {
        "schema": "yam_rlt_observations_v1",
        "split_role": "train",
        "mock": False,
        "train_groups": split["train_groups"],
        "holdout_groups": split["holdout_groups"],
        "prompt": prompt,
        "sources": sources,
        "excluded": excluded,
        "observations": observations,
    }
    path = output / "observations.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"path": str(path), "observations": len(observations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--prompt", required=True, help="exact original policy task prompt")
    args = parser.parse_args()
    print(json.dumps(prepare(args.run, args.output, json.loads(args.split.read_text()), prompt=args.prompt)))


if __name__ == "__main__":
    main()
