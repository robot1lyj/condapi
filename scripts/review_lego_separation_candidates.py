"""Render exact source frames around pose candidates, without modifying data.

Rows are top/left/right; columns follow review-index.json's episode-local
frame IDs. Preview images are evidence, not a video-completeness audit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np


def source_frames(reference, requested, fps):
    path = Path(reference["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (reference["size"], reference["mtime_ns"]):
        raise ValueError("source video changed since pose scan")
    targets = {reference["start_frame"] + f for f in requested}
    result = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(int(min(targets) / fps / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("source frame missing PTS")
            actual = float(frame.pts * frame.time_base * fps)
            index = round(actual)
            if abs(actual - index) > 0.05:
                raise ValueError("source PTS does not land on 30fps frame index")
            if index in targets:
                result[index - reference["start_frame"]] = frame.to_ndarray(format="rgb24")
            if index >= max(targets):
                break
    if set(result) != set(requested):
        raise ValueError(f"missing requested source frames: {set(requested) - set(result)}")
    return result


def write_jpeg(path, pixels):
    with av.open(str(path), "w", format="image2") as writer:
        stream = writer.add_stream("mjpeg", rate=1)
        stream.width, stream.height = pixels.shape[1], pixels.shape[0]
        stream.pix_fmt = "yuvj420p"
        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
            writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    candidates_path = args.scan / "candidates.json"
    candidates = json.loads(candidates_path.read_text())[args.offset : args.offset + args.limit]
    index = []
    for rank, candidate in enumerate(candidates, args.offset + 1):
        event = candidate["events"][0]
        baseline, start, end, peak = (
            event[k] for k in ("baseline_frame", "start_frame", "end_frame_exclusive", "peak_frame")
        )
        requested = sorted(
            {
                max(0, baseline - 30),
                baseline,
                (baseline + start) // 2,
                start,
                (start + end) // 2,
                peak,
                end - 1,
                min(candidate["length"] - 1, end + 30),
            }
        )
        views = {name: source_frames(candidate["videos"][name], requested, 30) for name in ("top", "left", "right")}
        sheet = np.concatenate(
            [np.concatenate([views[name][frame] for frame in requested], axis=1) for name in ("top", "left", "right")],
            axis=0,
        )
        path = args.output / f"{rank:03d}-episode-{candidate['episode_index']:04d}.jpg"
        write_jpeg(path, sheet)
        index.append(
            {
                "rank": rank,
                "episode_index": candidate["episode_index"],
                "source_episode_index": candidate["source_episode_index"],
                "frames": requested,
                "event": event,
                "sheet": path.name,
                "view_order": ["top", "left", "right"],
                "visual_label": None,
                "separation_success": None,
            }
        )
        if rank % 10 == 0:
            print(f"REVIEW_RENDER rank={rank}", flush=True)
    (args.output / "review-index.json").write_text(
        json.dumps(
            {
                "schema": "yam_lego_separation_visual_review_v1",
                "candidates_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
                "items": index,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
