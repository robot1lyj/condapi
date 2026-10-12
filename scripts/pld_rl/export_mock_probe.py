"""Export native client mock recordings against the real NumPy probe runtime.

No VLA forward, learning, hardware connection or modification of the YAM tree.
All recordings are explicitly synthetic and must never enter robot metrics.
"""

# ruff: noqa: PLC0415 -- load the explicitly selected read-only client only after CLI parsing

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/pld-rl/src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yam-root", type=Path, required=True, help="read-only client checkout")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.yam_root.resolve()))
    from yam_abc_reproduce.camera.mock_camera import MockCamera
    from yam_abc_reproduce.camera.worker import CameraWorker
    from yam_abc_reproduce.config import StationConfig
    from yam_abc_reproduce.hil.policy import PolicyWorker
    from yam_abc_reproduce.hil.recording import RecordingSession
    from yam_abc_reproduce.hil.run import Runtime
    from yam_abc_reproduce.hil.station import StationIO
    from yam_abc_reproduce.runtime import build_arm_units
    from yam_pld_rl.probe_runtime import ProbeRuntime

    probe = ProbeRuntime(ROOT / "third_party/yam-fk-model/yam_linear4310.xml", base_weights_sha256="fixture",
                         norm_sha256="fixture", engine_sha256="fixture", source_revision="fixture")

    class MockPi:
        rtc = True

        def __init__(self, arm):
            self.arm, self.session = arm, probe.new_session()
            self.metadata = {"rtc_mode": "trained", "rtc_max_delay_steps": 10, "action_horizon": 50,
                             "action_dim": 14, "action_dt_s": 1 / 30, "pld": probe.metadata}

        def infer_rtc(self, obs, *, target_start_tick, committed_actions, pld):
            time.sleep(.02)
            base = np.tile(obs["observation.state"], (50, 1))
            base[9:, 6 if self.arm == "left" else 13] = .35
            base[:len(committed_actions)] = committed_actions
            rtc = {"observation_policy_tick": target_start_tick, "delay_steps": len(committed_actions)}
            return self.session(obs, rtc, pld, {"actions": base, "server_timing": {"rtc_used": True}})

    reports = []
    for arm in ("left", "right"):
        io = StationIO(build_arm_units(StationConfig(), mock=True), mock=True)
        # Mock-only initial open state, never an SDK write.
        for unit in io.units:
            initial = unit.robot.get_joint_pos()
            initial[6] = .9
            unit.robot.command_joint_pos(initial)
        cameras = [CameraWorker(MockCamera(v, v, width=32, height=32)) for v in ("top", "left", "right")]
        worker = PolicyWorker(MockPi(arm))
        recorder = RecordingSession(args.output / arm, video_backend="libx264", metadata={"source_kind": "synthetic_mock_probe"})
        runtime = Runtime(io, cameras, worker, recorder, mode="inference",
                          settings={"policy_fusion": "rtc", "pld": {"mode": "probe"}, "tick_timeout": 2})
        try:
            for camera in cameras:
                camera.start()
            status = runtime.run(duration=1.8, auto_start=True)
            if status.get("error") or runtime.recording_error:
                raise RuntimeError(str(status))
        finally:
            worker.close()
            recorder.close()
            for camera in cameras:
                camera.stop()
            io.close()
        reports.append({"arm": arm, "episode": str(recorder.path / "episode_000001"), "native_episode_id": recorder.episode_id})
    (args.output / "fixture.json").write_text(json.dumps({"source_kind": "synthetic_mock_probe", "hardware_io": False,
                                                       "training": False, "episodes": reports}, indent=2) + "\n")
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
