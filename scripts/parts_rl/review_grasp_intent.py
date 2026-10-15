"""Thor-only frozen visual intent evidence for kinematic grasp failures.

This does not infer holding from height or publish rewards. Physics determines
which goal-failure status is possible; the model checks intent/loss in images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

PROMPT = """Inspect this chronological robot wrist/top image sequence for ATTEMPT versus OUTCOME separately.
grasp_intent asks whether the robot TRIES to pick up a loose LEGO brick. Approaching/alignment/descent and closing the two black triangular finger pads near a loose brick is a grasp ATTEMPT EVEN WHEN THE FINGERS END EMPTY. An empty or unsuccessful grasp is NOT evidence for grasp_intent=false. Return true for a visible successful OR failed pickup attempt. Return false ONLY with positive evidence of a different activity such as moving a large container, idling, or carrying an already-held object without attempting a new pickup. If the target or motion is unclear, use \"uncertain\", never false merely because no object was captured.
held_then_lost is a DIFFERENT question: was a small brick clearly held in an earlier image and clearly lost before the last image? Mere occlusion/blur is not loss; use \"uncertain\" in that case. Background objects through the gap do not count as held. Do not infer holding from closed fingers or height. Confidence describes the visual evidence, not whether the attempt succeeded.
Return ONLY compact JSON: {\"grasp_intent\":true/false/\"uncertain\",\"held_then_lost\":true/false/\"uncertain\",\"confidence\":\"high/medium/low\",\"reason\":\"under 20 words\"}."""


def selected_frames(candidate):
    frames = candidate["frames"]
    fresh = set(candidate.get("outcome", {}).get("fresh_rows", []))
    end = [f for f in frames if f["row"] in fresh]
    if not end:
        end = frames[-2:]
    choices = [frames[0], min(frames, key=lambda f: abs(f["since_close_s"])),
               max(frames, key=lambda f: f["lift_from_closure_m"]), end[len(end)//2], end[-1]]
    return sorted({f["row"]: f for f in choices}.values(), key=lambda f: f["row"])


def parse_decision(raw):
    decision = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    flags = []
    for key in ("grasp_intent", "held_then_lost"):
        value = decision[key]
        if isinstance(value, str) and value in ("true", "false"):
            decision[key] = value == "true"
            flags.append(key + "_string_boolean_normalized")
        elif type(value) is not bool and value != "uncertain":
            raise ValueError("invalid categorical decision")
    if decision["confidence"] not in ("high", "medium", "low"):
        raise ValueError("invalid confidence")
    decision["format_audit_flags"] = flags
    return decision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-name", default="vlm-intent")
    p.add_argument("--candidate-ids", nargs="*")
    args = p.parse_args()
    if platform.machine() != "aarch64":
        p.error("model inference runs on Thor")
    from PIL import Image  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import AutoProcessor  # noqa: PLC0415
    from transformers import Qwen3VLForConditionalGeneration  # noqa: PLC0415

    torch.set_num_threads(4)
    candidates = json.loads((args.dataset/"intent-prepared.json").read_text())
    if args.candidate_ids:
        wanted = set(args.candidate_ids)
        if wanted - {c["candidate_id"] for c in candidates}:
            p.error("unknown candidate identity")
        candidates = [c for c in candidates if c["candidate_id"] in wanted]
    out = args.dataset/args.output_name
    out.mkdir(exist_ok=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).eval().cuda()
    model.requires_grad_(False)  # noqa: FBT003
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
                                             min_pixels=256*28*28, max_pixels=320000)
    processor.tokenizer.padding_side = "left"
    started = time.monotonic()
    with torch.inference_mode():
        for begin in range(0, len(candidates), args.batch_size):
            group = candidates[begin:begin+args.batch_size]
            texts, images = [], []
            for c in group:
                content = [{"type": "text", "text": PROMPT}]
                frames = selected_frames(c)
                close_row = min(frames, key=lambda f: abs(f["since_close_s"]))["row"]
                for i, f in enumerate(frames):
                    for view in ([c["arm"], "top"] if i == 0 or f["row"] == close_row else [c["arm"]]):
                        if view not in f["views"]:
                            continue
                        path = args.dataset/f["views"][view]["image"]
                        if hashlib.sha256(path.read_bytes()).hexdigest() != c["image_sha256"][f["views"][view]["image"]]:
                            raise ValueError("intent image identity changed")
                        im = Image.open(path).convert("RGB")
                        if view == c["arm"] and i:
                            im = im.crop((140, 140, 500, 480))
                        images.append(im)
                        content += [{"type": "text", "text": f"Frame {i}, {view}, t={f['since_close_s']:+.2f}s"},
                                    {"type": "image"}]
                texts.append(processor.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
            inp = processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
            generated = model.generate(**inp, max_new_tokens=120, do_sample=False)
            decoded = processor.batch_decode(generated[:, inp["input_ids"].shape[-1]:], skip_special_tokens=True)
            for c, raw in zip(group, decoded, strict=True):
                try:
                    decision = parse_decision(raw)
                    error = None
                except (ValueError, KeyError) as exc:
                    decision, error = None, str(exc)
                (out/(c["candidate_id"]+".json")).write_text(json.dumps({
                    "candidate_id": c["candidate_id"], "candidate_sha256": c["candidate_sha256"],
                    "decision": decision, "raw_response": raw, "parse_error": error,
                    "evidence_image_rows": [f["row"] for f in selected_frames(c)],
                    "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                    "model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                    "neural_training": False}, ensure_ascii=False, indent=2)+"\n")
            print(json.dumps({"completed": begin+len(group), "total": len(candidates),
                              "elapsed_s": time.monotonic()-started}), flush=True)
    (out/"run.json").write_text(json.dumps({"prompt": PROMPT, "candidates": len(candidates),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_s": time.monotonic()-started, "training": False, "tracking": False}, indent=2)+"\n")


if __name__ == "__main__":
    main()
