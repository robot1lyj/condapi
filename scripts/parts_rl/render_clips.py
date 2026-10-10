"""Render synchronized top/left/right review clips, using recorded video indices."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import h5py
from PIL import Image
from PIL import ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    candidates = json.loads((args.dataset / "candidates-unscored.json").read_text())
    episodes = json.loads((args.dataset / "episodes.json").read_text())
    outputs = []
    for episode in episodes:
        number = episode["number"]
        selected = [c for c in candidates if c["episode_number"] == number]
        writers = {}
        for candidate in selected:
            name = f"clips/{candidate['candidate_id']}.mp4"
            path = args.dataset / name
            if path.exists():
                raise ValueError(f"write a new clip: {path}")
            container = av.open(str(path), "w")
            stream = container.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 960, 288, "yuv420p"
            stream.codec_context.thread_count = 1
            # Nominal 10 Hz must not quantize irregular source times to 100 ms.
            stream.codec_context.time_base = Fraction(1, 1000)
            stream.time_base = Fraction(1, 1000)
            stream.options = {"crf": "25", "preset": "veryfast", "bf": "0"}
            writers[candidate["candidate_id"]] = {"container": container, "stream": stream, "frames": 0,
                                                   "first_time": None, "last_time": None, "path": name}
        folder = args.source / "raw" / episode["source_path"]
        manifest = json.loads((folder / "manifest.json").read_text())
        offset = 0
        for segment in manifest["segments"]:
            directory = folder / segment["path"]
            with h5py.File(directory / "samples.h5") as handle:
                times, ticks = handle["time"][:], handle["tick"][:]
                references, valid = handle["video_indices"][:], handle["observation_valid"][:]
            containers = [av.open(str(directory / f"{view}.mp4")) for view in ("top", "left", "right")]
            for container in containers:
                container.streams.video[0].codec_context.thread_count = 1
            decoders = [iter(container.decode(video=0)) for container in containers]
            current = [(-1, None)] * 3
            for row in range(len(times)):
                if (offset + row) % 3:
                    continue
                active = [c for c in selected if c["entry_time"] - 0.6 <= times[row] <= c["end_time"] + 0.6]
                if not active:
                    continue
                images = []
                for column, target in enumerate(references[row]):
                    index, frame = current[column]
                    if target < index:
                        raise ValueError("nonmonotone video references require explicit handling")
                    while index < target:
                        frame, index = next(decoders[column]), index + 1
                    current[column] = (index, frame)
                    images.append(frame.to_image().convert("RGB").resize((320, 240)))
                canvas = Image.new("RGB", (960, 288), "#111827")
                draw = ImageDraw.Draw(canvas)
                for column, (view, image) in enumerate(zip(("TOP", "LEFT", "RIGHT"), images, strict=True)):
                    canvas.paste(image, (column * 320, 48))
                    draw.text((column * 320 + 5, 30), f"{view} video {references[row, column]}", fill="white")
                relative = float(times[row] - episode["start_time"])
                draw.text((5, 6), f"E{number:02d} t={relative:.2f}s row={offset + row} tick={ticks[row]}",
                          fill="white" if valid[row] else "#ff6666")
                for candidate in active:
                    writer = writers[candidate["candidate_id"]]
                    if writer["first_time"] is None:
                        writer["first_time"] = float(times[row])
                    writer["last_time"] = float(times[row])
                    frame = av.VideoFrame.from_image(canvas)
                    # Preserve source elapsed time, including gaps; do not compress waits.
                    frame.pts = round((times[row] - writer["first_time"]) * 1000)
                    frame.time_base = Fraction(1, 1000)
                    for packet in writer["stream"].encode(frame):
                        writer["container"].mux(packet)
                    writer["frames"] += 1
            for container in containers:
                container.close()
            offset += len(times)
        for candidate in selected:
            writer = writers[candidate["candidate_id"]]
            for packet in writer["stream"].encode():
                writer["container"].mux(packet)
            writer["container"].close()
            if not writer["frames"]:
                raise RuntimeError("candidate has no rendered review frames")
            path = args.dataset / writer["path"]
            outputs.append({"candidate_id": candidate["candidate_id"], "path": writer["path"],
                            "frames": writer["frames"], "source_start_time": writer["first_time"],
                            "source_end_time": writer["last_time"], "rate_hz": 10,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps({"episode": number, "clips": len(selected)}), flush=True)
    (args.dataset / "clip-manifest.json").write_text(json.dumps(outputs, indent=2) + "\n")


if __name__ == "__main__":
    main()
