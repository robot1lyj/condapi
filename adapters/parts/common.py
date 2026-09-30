"""Recipe validation is standard library only; never imports torch."""

import json
import math
from pathlib import Path
import socket


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    required = {
        "schema_version",
        "algorithm",
        "allowed_training_hosts",
        "replay",
        "hidden_dims",
        "actor_lr",
        "critic_lr",
        "gamma",
        "tau",
        "target_noise",
        "target_noise_clip",
        "policy_frequency",
        "q_weight",
        "success_bc_weight",
        "failure_anchor_weight",
        "reference_dropout",
        "batch_size",
        "updates",
        "save_every",
        "seed",
        "failure_fraction",
    }
    if set(recipe) != required or recipe["schema_version"] != 1 or recipe["algorithm"] != "parts_td3_bc_v1":
        raise ValueError("Unknown/incomplete PARTS recipe")
    if not isinstance(recipe["allowed_training_hosts"], list) or not all(
        isinstance(value, str) and value for value in recipe["allowed_training_hosts"]
    ):
        raise ValueError("allowed_training_hosts must explicitly list server hostnames")
    if not isinstance(recipe["replay"], dict) or set(recipe["replay"]) != {"left", "right"}:
        raise ValueError("Provide left/right audited READY paths")
    if not all(isinstance(value, str) and value for value in recipe["replay"].values()):
        raise ValueError("READY paths are not configured")
    widths = recipe["hidden_dims"]
    if not isinstance(widths, list) or not widths or any(type(v) is not int or v <= 0 for v in widths):
        raise ValueError("hidden_dims must be positive widths")
    for key in ("batch_size", "updates", "save_every", "policy_frequency"):
        if type(recipe[key]) is not int or recipe[key] <= 0:
            raise ValueError(f"{key} must be positive integer")
    if type(recipe["seed"]) is not int or recipe["seed"] < 0:
        raise ValueError("seed must be nonnegative integer")
    for key in required - {
        "schema_version",
        "algorithm",
        "allowed_training_hosts",
        "replay",
        "hidden_dims",
        "batch_size",
        "updates",
        "save_every",
        "policy_frequency",
        "seed",
    }:
        value = recipe[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be explicitly finite")
    for key in ("actor_lr", "critic_lr", "q_weight", "success_bc_weight"):
        if recipe[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("target_noise", "target_noise_clip", "failure_anchor_weight"):
        if recipe[key] < 0:
            raise ValueError(f"{key} cannot be negative")
    if not 0 < recipe["gamma"] <= 1 or not 0 < recipe["tau"] <= 1:
        raise ValueError("Invalid gamma/tau")
    if not 0 <= recipe["reference_dropout"] < 1 or not 0 <= recipe["failure_fraction"] <= 1:
        raise ValueError("Invalid dropout/failure fraction")
    return recipe


def require_server(recipe, execute_on_server):
    if not execute_on_server or socket.gethostname() not in recipe["allowed_training_hosts"]:
        raise ValueError("Training allowed only with --execute-on-server on a listed GPU server")
