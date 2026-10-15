"""Stage frozen-model evidence for a curator release; no labels or training."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/pld-rl/src"))
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json


def stage(curator, references, output):
    manifest = json.loads((curator / "manifest.json").read_text())
    for name, digest in manifest["outputs_sha256"].items():
        if sha256(curator / name) != digest:
            raise ValueError("curator changed: " + name)
    candidates = json.loads((curator / "candidates.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    vision, intent = output / "vision-v1", output / "intent-v1"
    for directory in (vision, intent):
        (directory / "images").mkdir(parents=True)
    (vision / "montages").mkdir()
    (vision / "reference-images").mkdir()
    prepared, intent_candidates, deferred, jobs = [], [], [], {}

    def copy_images(candidate, directory):
        for frame in candidate["frames"]:
            for ref in frame["views"].values():
                source, target = curator / ref["image"], directory / ref["image"]
                if not target.exists():
                    shutil.copyfile(source, target)

    for candidate in candidates:
        proof = candidate.get("outcome")
        if not proof or not proof["kinematic_complete"] or not proof["fresh_rows"]:
            deferred.append({"candidate_id": candidate["candidate_id"], "reason": "missing_complete_endpoint"})
            continue
        # Do not borrow a later Pi recovery or the next grasp as outcome evidence.
        scoped = {**candidate, "frames": [f for f in candidate["frames"]
                                         if f["tick"] <= proof["terminal_tick"]]}
        if not scoped["frames"]:
            continue
        copy_images(scoped, intent)
        scoped["image_sha256"] = {r["image"]: sha256(intent / r["image"])
                                  for f in scoped["frames"] for r in f["views"].values()}
        intent_candidates.append(scoped)
        lifted = [f for f in scoped["frames"] if f["row"] in proof["lifted_rows"]
                  and candidate["arm"] in f["views"]]
        unique = {(f["views"][candidate["arm"]]["source_video"],
                   f["views"][candidate["arm"]]["video_frame_index"]) for f in lifted}
        if not lifted or lifted[-1]["row"] != proof["fresh_rows"][-1] or len(unique) < 2:
            deferred.append({"candidate_id": candidate["candidate_id"], "reason": "intent_or_nonlift_review"})
            continue
        context = [f for f in scoped["frames"] if f["tick"] <= candidate["close_tick"]
                   and candidate["arm"] in f["views"]]
        selected = ([context[0]] if context else []) + lifted
        rec = {**candidate, "frames": selected,
               "model_evidence_scope": "preclosure_context_and_postdrain_lifted_endpoint",
               "original_candidate_sha256": candidate["candidate_sha256"], "approved": False}
        copy_images(rec, vision)
        for frame in selected:
            ref = frame["views"][candidate["arm"]]
            jobs[ref["key"]] = {"id": ref["key"], "image": ref["image"],
                                "episode_number": candidate["episode_number"],
                                "episode_id": candidate["episode_id"], "view": candidate["arm"],
                                "sha256": sha256(vision / ref["image"])}
        shutil.copyfile(curator / candidate["montage"], vision / candidate["montage"])
        prepared.append(rec)

    refs_path = references / "reference-labels-staged.json"
    refs = json.loads(refs_path.read_text())
    for ref in refs:
        if ref.get("dataset_namespace") != "old_unmodified_pi14_20261011":
            raise ValueError("reference namespace must be independent of this collect batch")
        name = f"reference-images/REF{ref['number']:03d}.jpg"
        source = references / name
        if sha256(source) != ref["source_sha256"]:
            raise ValueError("reference source changed")
        shutil.copyfile(source, vision / name)
        ref["image"] = str(vision / name)
    write_json(vision / "reference-labels-staged.json", refs)
    write_json(vision / "image-jobs.json", list(jobs.values()))
    write_json(vision / "candidates-prepared.json", prepared)
    write_json(intent / "intent-prepared.json", intent_candidates)
    write_json(vision / "preparation.json", {
        "curator_manifest_sha256": sha256(curator / "manifest.json"),
        "reference_labels_sha256": sha256(refs_path), "references": len(refs),
        "vlm_candidates": len(prepared), "intent_candidates": len(intent_candidates),
        "deferred": deferred, "automatic_tracking": False, "training": False,
        "approved_rewards": 0, "producer_sha256": sha256(Path(__file__))})
    print(json.dumps({"lifted": len(prepared), "intent": len(intent_candidates),
                      "deferred": dict(Counter(d["reason"] for d in deferred))}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curator", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    stage(args.curator.resolve(), args.references.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
