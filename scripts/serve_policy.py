import dataclasses
import enum
import json
import logging
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


class RtcMode(enum.Enum):
    """RTC handling mode for websocket inference."""

    OFF = "off"
    AUTO = "auto"
    ONLY = "only"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # RTC handling mode: off (ignore), auto (use if provided), only (require rtc payload).
    rtc_mode: RtcMode = RtcMode.OFF
    # Optional metadata JSON file to send during handshake (merged into policy.metadata).
    rtc_metadata: str | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _load_metadata(path: str | None) -> dict:
    if path is None:
        return {}
    metadata_path = pathlib.Path(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Metadata JSON must be an object, got: {type(data).__name__}")
    return data


def _populate_action_metadata(policy: _policy.Policy, metadata: dict) -> None:
    model = getattr(policy, "_model", None)
    if model is None:
        return
    action_horizon = getattr(model, "action_horizon", None)
    action_dim = getattr(model, "action_dim", None)
    if action_horizon is None or action_dim is None:
        config = getattr(model, "config", None)
        if action_horizon is None:
            action_horizon = getattr(config, "action_horizon", None)
        if action_dim is None:
            action_dim = getattr(config, "action_dim", None)
    if action_horizon is not None:
        metadata.setdefault("action_horizon", int(action_horizon))
    if action_dim is not None:
        metadata.setdefault("action_dim", int(action_dim))


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = dict(policy.metadata)
    _populate_action_metadata(policy, policy_metadata)
    rtc_metadata = _load_metadata(args.rtc_metadata)
    if rtc_metadata:
        policy_metadata.update(rtc_metadata)
    policy_metadata["rtc_mode"] = args.rtc_mode.value

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        rtc_mode=args.rtc_mode.value,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
