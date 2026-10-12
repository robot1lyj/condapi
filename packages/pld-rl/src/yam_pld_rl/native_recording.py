"""Read immutable native HIL files and their session audit, without robot I/O."""

import json
from pathlib import Path

from yam_pld_rl.data import sha256


def event_episode(event):
    reply = event.get("details", {}).get("policy_reply") or {}
    # The original request wins over a malformed/foreign response identity.
    request = reply.get("pld_request") or {}
    trace = reply.get("pld") or {}
    return event.get("episode_id") or request.get("episode_id") or trace.get("episode_id")


def read_episode(folder):
    import h5py  # noqa: PLC0415 -- optional offline data dependency

    folder = Path(folder).resolve()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episode_id = manifest.get("episode_id")
    if not isinstance(episode_id, str) or not 0 < len(episode_id) <= 160:
        raise ValueError("native manifest episode_id required; folder name is not an identity")
    if manifest.get("outcome") == "recording":
        raise ValueError("copy/finalize the episode before offline processing")
    rows, events, hashes = [], [], {str(manifest_path): sha256(manifest_path)}
    required = ("time", "tick", "epoch", "measured_state", "observation_state", "observation_valid", "submitted_action",
                "is_intervention", "video_indices", "camera_host_received_at")
    for path in sorted(folder.glob("segment_*/samples.h5")):
        hashes[str(path)] = sha256(path)
        with h5py.File(path) as f:
            n = int(f["committed_rows"][()])
            if n < 0 or any(len(f[k]) < n for k in (*required, "details")):
                raise ValueError("invalid committed row count")
            data = {k: f[k][:n] for k in required}
            masks = {k: f[k + "__valid"][:n] for k in required if k + "__valid" in f}
            for i, raw in enumerate(f["details"][:n]):
                clock_valid = all(bool(masks[k][i]) for k in ("time", "tick", "epoch"))
                rows.append({"time": float(data["time"][i]), "tick": int(data["tick"][i]), "epoch": int(data["epoch"][i]),
                             "episode_id": episode_id, "measured_state": data["measured_state"][i].tolist(),
                             "measured_state_valid": clock_valid and bool(masks["measured_state"][i]),
                             "observation_state": data["observation_state"][i].tolist(),
                             "observation_valid": bool(data["observation_valid"][i]) and bool(masks["observation_valid"][i])
                             and bool(masks["observation_state"][i]),
                             "submitted_action": data["submitted_action"][i].tolist(),
                             "submitted_valid": clock_valid and bool(masks["submitted_action"][i]),
                             "is_intervention": bool(data["is_intervention"][i]),
                             "intervention_valid": bool(masks["is_intervention"][i]),
                             "camera_times": data["camera_host_received_at"][i].tolist(),
                             "camera_valid": masks["camera_host_received_at"][i].tolist(),
                             "video_indices": data["video_indices"][i].tolist(), "details": json.loads(raw),
                             "raw_ref": f"{folder.parent.name}/{folder.name}/{path.parent.name}/samples.h5#row={i}",
                             "videos": {v: str(path.parent / f"{v}.mp4") for v in ("top", "left", "right")}})
    if not rows:
        raise ValueError(f"no committed native HIL rows: {folder}")
    sidecar = (folder / manifest.get("policy_events", "../policy_events.jsonl")).resolve()
    if sidecar.parent != folder.parent:
        raise ValueError("policy_events must refer to this episode's session")
    if sidecar.is_file():
        hashes[str(sidecar)] = sha256(sidecar)
        before = sidecar.stat()
        with sidecar.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if event_episode(event) == episode_id:
                    events.append({**event, "audit_ref": f"{sidecar}#line={number}"})
        after = sidecar.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("session audit changed during reading; use a completed copied snapshot")
    elif (manifest.get("pld") or {}).get("mode", "off") != "off":
        raise ValueError("PLD recording requires the complete session policy_events.jsonl")
    for video in sorted({p for row in rows for p in row["videos"].values()}):
        if not Path(video).is_file() or not Path(video).stat().st_size:
            raise ValueError(f"missing source video: {video}")
        hashes[video] = sha256(video)
    return rows, events, hashes
