"""Thor-only frozen Qwen3-VL review of chronological grasp evidence, resumable."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

PROMPT = """These images show one robot pick in chronological order, from just before closing to lift/carry before release. The two BLACK TRIANGULAR FINGERS belong to the robot. The first image is a wrist view; later images zoom around the fingers.
Did the robot pick up a LEGO brick? Answer success if a small solid brick is visibly held between the fingers during lift/carry while the distant scene changes. Normal release over a bin after carrying is still success. Answer failure if the fingers lift empty or the brick slips away before being carried. Large blue bins and loose distant bricks are background. Do not infer success from closed fingers or height alone.
Give a best binary decision and lower confidence if evidence is obscured. Return ONLY compact JSON, with at most THREE evidence frame letters and a reason under 20 words:
{"outcome":"success/failure", "confidence":"high/medium/low", "color":"red/blue/yellow/other/none", "evidence_frames":["B","C"], "reason":"visible evidence"}."""


def evidence_frames(candidate):
    frames = candidate["frames"]
    if not frames:
        raise ValueError("No images for this attempt")
    post = [f for f in frames if f["since_close_s"] >= 0.4]
    chosen = [frames[0]]
    if post:
        chosen += [
            post[0],
            min(post, key=lambda f: abs(f["since_close_s"] - 1.1)),
            max(post, key=lambda f: f["lift_from_closure_m"]),
            post[-1],
        ]
    else:
        chosen += frames[-2:]
    return sorted({f["row"]: f for f in chosen}.values(), key=lambda f: f["row"])


SNAPSHOT_PROMPT = """Classify what is physically BETWEEN the two black triangular robot finger pads in each QUERY image. First see labelled reference examples. A held brick is a distinct small foreground object between the inward finger faces. An empty gap can show a distant blue bin or loose bricks; these background objects are NOT held. Ignore colour alone. Judge each query separately, not task completion.
Return ONLY JSON: {"states":["held" or "empty" for each query in order], "confidence":"high/medium/low", "reason":"under 15 words"}."""


def raised_frames(candidate):
    post = [f for f in candidate["frames"] if f["since_close_s"] >= 0.45 and f["openness"] < 0.5]
    if not post:
        post = candidate["frames"][-3:]
    peak = max(post, key=lambda f: f["lift_from_closure_m"])
    # Inspect the peak and up to two preceding snapshots, before the releasing fingers.
    before = [f for f in post if f["row"] <= peak["row"]]
    return before[-3:]


def parse(text):
    start = text.find("{")
    end = text.rfind("}")
    data = json.loads(text[start : end + 1])
    data["event_type"] = "grasp"
    # Four/five valid frame references violate a formatting preference, not the
    # binary outcome schema. Keep them and record the deviation explicitly.
    assert len(data["evidence_frames"]) <= 5
    assert all(x in "ABCDE" and len(x) == 1 for x in data["evidence_frames"])
    assert data["outcome"] in ["success", "failure"]
    assert data["confidence"] in ["high", "medium", "low"]
    data["format_audit_flags"] = ["more_than_three_evidence_frames"] if len(data["evidence_frames"]) > 3 else []
    return data


def needs_inference(candidate, path, prompt):
    """Resume only matching, valid evidence; never silently reuse a stale decision."""
    if not path.exists():
        return True
    saved = json.loads(path.read_text())
    if (
        saved.get("candidate_id") != candidate["candidate_id"]
        or saved.get("montage_sha256") != candidate["montage_sha256"]
        or saved.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
        or saved.get("parse_error")
        or not saved.get("decision")
    ):
        raise ValueError("Existing VLM evidence is stale or invalid; use a new output name: " + str(path))
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-pixels", type=int, default=500000)
    p.add_argument("--output-name", default="vlm-primary")
    p.add_argument("--snapshot-check", action="store_true")
    p.add_argument("--terminal-hold", action="store_true", help="judge held at the local post-drain endpoint")
    args = p.parse_args()
    if args.snapshot_check and args.terminal_hold:
        p.error("choose snapshot classification or terminal holding")
    if platform.machine() != "aarch64":
        p.error("Neural inference must run on Thor")
    from PIL import Image  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import AutoProcessor  # noqa: PLC0415
    from transformers import Qwen3VLForConditionalGeneration  # noqa: PLC0415

    torch.set_num_threads(4)
    out = args.dataset / args.output_name
    out.mkdir(exist_ok=True)
    candidates = json.loads((args.dataset / "candidates-prepared.json").read_text())
    candidates = candidates[: args.limit] if args.limit else candidates
    prompt = SNAPSHOT_PROMPT if args.snapshot_check else PROMPT
    if args.terminal_hold:
        prompt = prompt.replace(
            "Normal release over a bin after carrying is still success.",
            "This is a local grasp-and-hold goal. A success requires the brick to STILL be held in the LAST "
            "lifted image. Earlier holding followed by an empty last image is failure. All lifted images "
            "are after queued corrections drained; do not credit later Pi recovery.")
    pending = [c for c in candidates if needs_inference(c, out / (c["candidate_id"] + ".json"), prompt)]
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
        )
        .eval()
        .cuda()
    )
    model.requires_grad_(False)  # noqa: FBT003
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, min_pixels=256 * 28 * 28, max_pixels=args.max_pixels
    )
    processor.tokenizer.padding_side = "left"
    refs = json.loads((args.dataset / "reference-labels.json").read_text()) if args.snapshot_check else []
    started = time.monotonic()
    with torch.inference_mode():
        for begin in range(0, len(pending), args.batch_size):
            group = pending[begin : begin + args.batch_size]
            images = []
            texts = []
            for c in group:
                path = args.dataset / c["montage"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == c["montage_sha256"]
                content = [{"type": "text", "text": prompt}]
                if args.snapshot_check:
                    for label in ["held_at_snapshot", "empty_at_snapshot"]:
                        options = [
                            r for r in refs if r["label"] == label and r["episode_number"] != c["episode_number"]
                        ]
                        for ref in options[:2]:
                            images.append(
                                Image.open(args.dataset / f"fixed-crops/REF{ref['number']:03d}.png").convert("RGB")
                            )
                            content += [
                                {
                                    "type": "text",
                                    "text": "REFERENCE " + ("held brick" if label.startswith("held") else "empty gap"),
                                },
                                {"type": "image"},
                            ]
                selected = raised_frames(c) if args.snapshot_check else evidence_frames(c)
                for k, f in enumerate(selected):
                    ref = f["views"][c["arm"]]
                    im = Image.open(args.dataset / ref["image"]).convert("RGB")
                    if args.snapshot_check:
                        im = im.crop((160, 180, 480, 480))
                    elif k:
                        im = im.crop((140, 140, 500, 480))
                    images.append(im)
                    content += [
                        {
                            "type": "text",
                            "text": f"{'QUERY' if args.snapshot_check else 'Frame'} {chr(65 + k)}, t={f['since_close_s']:+.2f}s:",
                        },
                        {"type": "image"},
                    ]
                texts.append(
                    processor.apply_chat_template(
                        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
                    )
                )
            inp = processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
            tick = time.monotonic()
            generated = model.generate(**inp, max_new_tokens=130, do_sample=False)
            decoded = processor.batch_decode(generated[:, inp["input_ids"].shape[-1] :], skip_special_tokens=True)
            for c, raw in zip(group, decoded, strict=True):
                try:
                    decision = (
                        json.loads(raw[raw.find("{") : raw.rfind("}") + 1]) if args.snapshot_check else parse(raw)
                    )
                    if args.snapshot_check:
                        assert len(decision["states"]) == len(raised_frames(c))
                        assert all(x in ["held", "empty"] for x in decision["states"])
                    error = None
                except (ValueError, KeyError, AssertionError) as e:
                    decision = None
                    error = str(e)
                record = {
                    "candidate_id": c["candidate_id"],
                    "decision": decision,
                    "raw_response": raw,
                    "parse_error": error,
                    "montage_sha256": c["montage_sha256"],
                    "evidence_image_rows": [
                        f["row"] for f in (raised_frames(c) if args.snapshot_check else evidence_frames(c))
                    ],
                    "model_id": "Qwen/Qwen3-VL-4B-Instruct",
                    "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                    "prompt_sha256": hashlib.sha256(
                        (SNAPSHOT_PROMPT if args.snapshot_check else PROMPT).encode()
                    ).hexdigest(),
                    "neural_training": False,
                    "user_review_label": None,
                }
                (out / (c["candidate_id"] + ".json")).write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n"
                )
            print(
                json.dumps(
                    {
                        "completed": begin + len(group),
                        "pending_total": len(pending),
                        "batch_seconds": time.monotonic() - tick,
                        "max_cuda_allocated": torch.cuda.max_memory_allocated(),
                        "ids": [c["candidate_id"] for c in group],
                    }
                ),
                flush=True,
            )
    (out / "run.json").write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3-VL-4B-Instruct",
                "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                "elapsed_s": time.monotonic() - started,
                "candidates": len(candidates),
                "processed_this_run": len(pending),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "prompt": prompt,
                "tracking": False,
                "training_executed": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
