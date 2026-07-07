#!/usr/bin/env python3
"""Convert OpenArm data to the HQ policy data contract.

HQ contract:
- task prompt: "Fold the T-shirt properly"
- arm joint state/action: degrees
- gripper state/action: HQ-style motor degrees, 0 deg open and -66 deg closed

Examples:
    python scripts/convert_openarm_hq_dataset.py from-hdf5 \
      --src /storage1t/ipc \
      --dst /storage1t/datasets/openarm_site_align_v1_deg \
      --dataset-id openarm_site_align_v1_deg \
      --overwrite

    python scripts/convert_openarm_hq_dataset.py from-lerobot \
      --src /share/home/linyongjia/datasets/openarm_site_align_v1 \
      --dst /share/home/linyongjia/datasets/openarm_site_align_v1_deg \
      --dataset-id openarm_site_align_v1_deg \
      --overwrite
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_TASK = "Fold the T-shirt properly"
DEFAULT_DATASET_ID = "openarm_site_align_v1_deg"
DEFAULT_GRIPPER_CLOSED_NORM = 0.0
DEFAULT_GRIPPER_OPEN_NORM = 0.84


def _add_hq_contract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--policy-joint-unit", choices=("degrees", "radians"), default="degrees")
    parser.add_argument("--policy-gripper-unit", choices=("dataset_degrees", "normalized"), default="dataset_degrees")
    parser.add_argument("--gripper-closed-norm", type=float, default=DEFAULT_GRIPPER_CLOSED_NORM)
    parser.add_argument("--gripper-open-norm", type=float, default=DEFAULT_GRIPPER_OPEN_NORM)


def _add_common_copy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")


def _run_from_hdf5(args: argparse.Namespace) -> None:
    from scripts import convert_openarm_site_hdf5_to_lerobot_v21  # noqa: PLC0415

    src_paths = args.src or [pathlib.Path("/storage1t/ipc")]
    convert_openarm_site_hdf5_to_lerobot_v21.convert_dataset(
        src_paths,
        args.dst,
        dataset_id=args.dataset_id,
        task=args.task,
        val_count=args.val_count,
        fps=args.fps,
        chunks_size=args.chunks_size,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        episodes=args.episodes,
        max_episodes=args.max_episodes,
        min_frames=args.min_frames,
        max_abs_state_action=args.max_abs_state_action,
        timestamp_mode=args.timestamp_mode,
        gripper_action_fallback=args.gripper_action_fallback,
        policy_joint_unit=args.policy_joint_unit,
        policy_gripper_unit=args.policy_gripper_unit,
        gripper_closed_norm=args.gripper_closed_norm,
        gripper_open_norm=args.gripper_open_norm,
        verify_video_frames=args.verify_video_frames,
        skip_invalid=args.skip_invalid,
        site_repeat=args.site_repeat,
    )


def _run_from_lerobot(args: argparse.Namespace) -> None:
    from scripts import convert_openarm_lerobot_units  # noqa: PLC0415

    convert_openarm_lerobot_units.convert_dataset(
        args.src,
        args.dst,
        dataset_id=args.dataset_id,
        policy_joint_unit=args.policy_joint_unit,
        policy_gripper_unit=args.policy_gripper_unit,
        task=args.task,
        gripper_closed_norm=args.gripper_closed_norm,
        gripper_open_norm=args.gripper_open_norm,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hdf5 = subparsers.add_parser("from-hdf5", help="Convert raw OpenArm HDF5 episodes to LeRobot v2.1.")
    hdf5.add_argument("--src", type=pathlib.Path, action="append", default=[])
    hdf5.add_argument("--dst", type=pathlib.Path, required=True)
    _add_hq_contract_args(hdf5)
    _add_common_copy_args(hdf5)
    hdf5.add_argument("--episodes", default=None)
    hdf5.add_argument("--max-episodes", type=int, default=None)
    hdf5.add_argument("--val-count", type=int, default=10)
    hdf5.add_argument("--fps", type=int, default=30)
    hdf5.add_argument("--chunks-size", type=int, default=1000)
    hdf5.add_argument("--timestamp-mode", choices=("raw_zeroed", "fps"), default="fps")
    hdf5.add_argument("--gripper-action-fallback", choices=("state", "zero", "error"), default="state")
    hdf5.add_argument("--min-frames", type=int, default=2)
    hdf5.add_argument("--max-abs-state-action", type=float, default=None)
    hdf5.add_argument("--verify-video-frames", action="store_true")
    hdf5.add_argument("--skip-invalid", action="store_true")
    hdf5.add_argument("--site-repeat", type=int, default=5)
    hdf5.add_argument("--dry-run", action="store_true")
    hdf5.set_defaults(func=_run_from_hdf5)

    lerobot = subparsers.add_parser("from-lerobot", help="Rewrite an existing OpenArm LeRobot v2.1 dataset.")
    lerobot.add_argument("--src", type=pathlib.Path, required=True)
    lerobot.add_argument("--dst", type=pathlib.Path, required=True)
    _add_hq_contract_args(lerobot)
    _add_common_copy_args(lerobot)
    lerobot.set_defaults(func=_run_from_lerobot)

    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
