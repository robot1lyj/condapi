"""Build fixed wrist crops and a visually reviewed reference bank; CPU only."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--reuse", type=Path)
    args = p.parse_args()
    root = args.dataset
    out = root / "fixed-crops"
    out.mkdir(exist_ok=False)
    refs = json.loads(args.references.read_text())
    if not all(q.get("reviewed") and q["label"] in ["held_at_snapshot", "empty_at_snapshot"] for q in refs):
        raise ValueError("Reference labels require explicit visual review")
    weights = np.zeros((16, 16), np.float32)
    weights[3:16, 4:12] = 1
    np.savez_compressed(out / "weights.npz", weights=weights)
    old_jobs = {}
    old_images = {}
    if args.reuse:
        old_jobs = {q["id"]: q for q in json.loads((args.reuse / "feature-jobs.json").read_text())}
        old_images = {q["id"]: q["sha256"] for q in json.loads((args.reuse / "image-jobs.json").read_text())}
    inputs = [
        {**q, "role": "query", "arm": q["view"]}
        for q in json.loads((root / "image-jobs.json").read_text())
        if q["view"] != "top"
    ]
    inputs.extend({**q, "id": f"REF{q['number']:03d}", "role": "reference"} for q in refs)
    jobs = []
    for q in inputs:
        source = Path(q["image"]) if q["role"] == "reference" else root / q["image"]
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        name = f"fixed-crops/{q['id']}.png"
        cached = old_jobs.get(q["id"])
        cached_source_sha = cached.get("source_sha256", old_images.get(q["id"])) if cached else None
        if cached and cached_source_sha == source_sha:
            path = args.reuse / cached["crop"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != cached["crop_sha256"]:
                raise ValueError("Cached crop identity mismatch")
            shutil.copyfile(path, root / name)
        else:
            Image.open(source).convert("RGB").crop((160, 180, 480, 480)).resize((256, 256)).save(root / name)
        jobs.append(
            {
                "id": q["id"],
                "episode_number": q["episode_number"],
                "arm": q["arm"],
                "role": q["role"],
                "crop": name,
                "crop_sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                "source_sha256": source_sha,
                "patch_weights": "fixed-crops/weights.npz",
                **({"label": q["label"]} if q["role"] == "reference" else {}),
            }
        )
    (root / "feature-jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    (root / "reference-labels.json").write_text(json.dumps(refs, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "references": len(refs)}))


if __name__ == "__main__":
    main()
