"""Publish binary proposals with chronological evidence and review export; CPU only."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from PIL import ImageDraw


def audit_sheets(root, candidates):
    out = root / "audit-sheets"
    out.mkdir(exist_ok=True)
    manifest = []
    for begin in range(0, len(candidates), 15):
        canvas = Image.new("RGB", (1920, 1000), "white")
        draw = ImageDraw.Draw(canvas)
        selected = candidates[begin : begin + 15]
        for k, c in enumerate(selected):
            x, y = k % 3 * 640, k // 3 * 200
            frames = c["frames"]
            post = [f for f in frames if f["since_close_s"] >= 0.35]
            show = [frames[0]] + (
                [post[0], post[len(post) // 2], max(post, key=lambda f: f["lift_from_closure_m"]), post[-1]]
                if post
                else frames[-4:]
            )
            draw.text((x + 5, y + 2), c["candidate_id"], fill="black")
            for i, f in enumerate(show):
                im = (
                    Image.open(root / f["views"][c["arm"]]["image"])
                    .convert("RGB")
                    .crop((160, 180, 480, 480))
                    .resize((126, 158))
                )
                canvas.paste(im, (x + i * 128, y + 36))
                draw.text((x + i * 128, y + 18), f"{f['row']} {f['since_close_s']:.1f}s", fill="black")
        name = f"audit-sheets/page-{begin // 15 + 1:02d}.jpg"
        canvas.save(root / name, quality=95)
        manifest.append({"sheet": name, "ids": [c["candidate_id"] for c in selected]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--audit-sheets-only", action="store_true")
    args = p.parse_args()
    root = args.dataset
    candidates = json.loads(
        (root / ("candidates-prepared.json" if args.audit_sheets_only else "candidates.json")).read_text()
    )
    if args.audit_sheets_only:
        audit_sheets(root, candidates)
        return
    clips = {q["candidate_id"]: q for q in json.loads((root / "clip-manifest.json").read_text())}
    for c in candidates:
        c["clip"] = clips[c["candidate_id"]]
    summary = json.loads((root / "summary.json").read_text())
    identity = hashlib.sha256((root / "candidates.json").read_bytes()).hexdigest()
    data = {"summary": summary, "identity": identity, "cards": candidates}
    # Data is text in a script tag, so prevent source evidence from terminating it.
    encoded = (
        json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    template = Path(__file__).with_name("revised_review_template.html").read_text()
    (root / "index.html").write_text(template.replace("__DATA__", encoded))
    (root / "page-identity.json").write_text(
        json.dumps({"dataset_id": root.name, "candidates_sha256": identity}, indent=2) + "\n"
    )
    print(json.dumps({"cards": len(candidates), "page": str(root / "index.html")}))


if __name__ == "__main__":
    main()
