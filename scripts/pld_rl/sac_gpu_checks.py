"""Slurm-only numerical and CLI round-trip tests on synthetic fixtures."""

# ruff: noqa: E402

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packages/pld-rl/src"), str(ROOT / "scripts/pld_rl")]
from block_fixtures import contract
from block_fixtures import inference_case
from block_fixtures import records
from block_fixtures import stats
from curator_checks import native_fixture
from train_sac import export_bundle
from yam_pld_rl.block_contract import FEATURE_DIM
from yam_pld_rl.block_replay import compile_replay
from yam_pld_rl.data import sha256
from yam_pld_rl.data import write_json
from yam_pld_rl.runtime import BlockRuntime
from yam_pld_rl.sac import SacLearner


def native_hil_roundtrip(output, snapshot, bundle):
    import av  # noqa: PLC0415
    import h5py  # noqa: PLC0415

    _, rows, _, _ = native_fixture(snapshot)
    folder = output / "raw/ep/segment_000000"
    folder.mkdir(parents=True)
    with h5py.File(folder / "samples.h5", "w") as f:
        f["committed_rows"] = len(rows)
        columns = {"time": "time", "tick": "tick", "epoch": "epoch", "measured_state": "measured_state",
                   "observation_state": "measured_state", "submitted_action": "submitted_action",
                   "is_intervention": "is_intervention", "camera_host_received_at": "camera_times", "video_indices": "video_indices"}
        for name, key in columns.items():
            f[name] = np.asarray([r[key] for r in rows])
            if name != "video_indices":
                f[name + "__valid"] = np.ones((len(rows), 3) if name == "camera_host_received_at" else len(rows), dtype=bool)
        f.create_dataset("details", data=[json.dumps(r["details"]) for r in rows], dtype=h5py.string_dtype())
    for view in ("top", "left", "right"):
        with av.open(str(folder / f"{view}.mp4"), "w") as video:
            stream = video.add_stream("mpeg4", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
            for index in range(len(rows)):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), index, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    video.mux(packet)
            for packet in stream.encode():
                video.mux(packet)
    command = [sys.executable, str(ROOT / "scripts/pld_rl/curate_recording.py"), "--episodes", str(folder.parent)]
    subprocess.run([*command, "--output", str(output / "native-candidates"), "--export-evidence"], check=True)
    candidates = json.loads((output / "native-candidates/candidates.json").read_text())
    assert len(candidates) == 1
    assert candidates[0]["evidence_eligible"]
    c = candidates[0]
    reviewed = [{"candidate_id": c["candidate_id"], "candidate_sha256": c["candidate_sha256"], "status": "held",
                 "approved": True, "reviewer": "synthetic fixture oracle; NOT actual VLM", "evidence_rows":
                 [f["row"] for f in c["frames"] if f["lift_from_closure_m"] >= .05]}]
    write_json(output / "native-reviewed.json", reviewed)
    subprocess.run([*command, "--rewards", str(output / "native-reviewed.json"), "--output", str(output / "native-curator")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/pld_rl/prepare_blocks.py"), "--curator", str(output / "native-curator"),
                    "--bundles", str(bundle), "--output", str(output / "native-replay")], check=True)
    return json.loads((output / "native-replay/manifest.json").read_text())["audit"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("allocated server GPU required; no local training smoke")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(441)
    cfg = json.loads((ROOT / "configs/pld_rl/sac-block-v2.json").read_text())
    cfg.update(steps=40, critic_warmup_steps=4, save_interval=20, log_interval=10)
    c = contract()
    learner = SacLearner(FEATURE_DIM, cfg)
    export_bundle(args.output / "bootstrap", learner, c, stats(), sha256(args.encoder), cfg, {"fixture": True})
    manifest = json.loads((args.output / "bootstrap/manifest.json").read_text())
    snapshot = sha256(args.output / "bootstrap/manifest.json")
    recs, labels = records(c, snapshot)
    data, audit = compile_replay(recs, labels, c, {snapshot: manifest})
    b = {k: torch.as_tensor(data[k], device="cuda", dtype=torch.float32)
         for k in ("x", "nx", "scale", "next_scale", "action", "reward", "terminal", "discount")}
    u, _ = learner.actor(b["x"], deterministic=True)
    assert torch.count_nonzero(u) == 0
    terminal = b["terminal"] == 1
    poisoned = {**b, "nx": b["nx"].clone()}
    poisoned["nx"][terminal] = float("nan")
    target = learner.backup(poisoned)
    assert torch.equal(target[terminal], b["reward"][terminal])
    assert torch.isfinite(target).all()
    for _ in range(8):
        learner.update(b)
    saved = copy.deepcopy(learner.state_dict())
    clone = SacLearner(FEATURE_DIM, cfg)
    clone.load_state_dict(saved)
    torch.manual_seed(111)
    learner.update(b)
    torch.manual_seed(111)
    clone.update(b)
    for name in ("actor", "critic", "target"):
        for a, other in zip(getattr(learner, name).parameters(), getattr(clone, name).parameters(), strict=True):
            assert torch.equal(a, other), "optimizer/target resume mismatch"
    assert torch.equal(learner.log_alpha, clone.log_alpha)
    runtime = BlockRuntime(args.output / "bootstrap", args.encoder, base_weights_sha256="fixture-only")
    obs, rtc, request, output = inference_case(runtime, mode="eval")
    base = output["actions"]
    zero = runtime.new_session()(obs, rtc, request, output)
    assert zero["actions"].tobytes() == base.tobytes()
    request["mode"] = "collect"
    result = runtime.new_session()(obs, rtc, request, output)
    assert result["pld"]["arm"] == "left"
    assert "attempt_id" not in result["pld"]
    assert result["actions"][:3].tobytes() == base[:3].tobytes()
    assert result["actions"][:, 7:].tobytes() == base[:, 7:].tobytes()
    assert np.any(result["actions"] != base)
    for name, value in (("config", cfg), ("contract", c), ("normalization", stats()), ("rewards", labels)):
        write_json(args.output / f"{name}.json", value)
    (args.output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    subprocess.run([sys.executable, str(ROOT / "scripts/pld_rl/prepare_blocks.py"), "--records", str(args.output / "records.jsonl"),
                    "--rewards", str(args.output / "rewards.json"), "--bundles", str(args.output / "bootstrap"),
                    "--output", str(args.output / "replay")], check=True)
    native_audit = native_hil_roundtrip(args.output, snapshot, args.output / "bootstrap")
    command = [sys.executable, str(ROOT / "scripts/pld_rl/train_sac.py"), "--config", str(args.output / "config.json"),
               "--contract", str(args.output / "contract.json"), "--normalization", str(args.output / "normalization.json"),
               "--encoder", str(args.encoder), "--replay", str(args.output / "native-replay"), "--output", str(args.output / "run")]
    subprocess.run(command, check=True)
    restored = BlockRuntime(args.output / "run/bundle-000040", args.encoder, base_weights_sha256="fixture-only")
    request.update(restored.metadata)
    request["mode"] = "eval"
    evaluated = restored.new_session()(obs, rtc, request, {"actions": base})
    assert np.isfinite(evaluated["actions"]).all()
    assert evaluated["actions"][:3].tobytes() == base[:3].tobytes()
    write_json(args.output / "result.json", {"status": "passed_synthetic_only", "job": os.environ["SLURM_JOB_ID"],
               "gpu": torch.cuda.get_device_name(), "replay_audit": audit, "native_recording_audit": native_audit,
               "checks": ["zero equals Pi bitwise", "bounded nonzero collection", "FP64 RTC prefix exact",
                          "other arm unchanged", "terminal target ignores poisoned next observation", "exact optimizer resume",
                          "delayed reward compilation", "CLI 40 SAC updates", "export reload and infer"],
               "robot_deployed": False, "robot_success_rate": None})
    print((args.output / "result.json").read_text())


if __name__ == "__main__":
    main()
