"""Validate a checkpoint norm against the original RTC training contract.

Orbax's asset callback can remove exactly the final newline when saving JSON.
That changes the byte SHA without changing a single JSON value. Accept only
that proven serialization difference, never an arbitrary semantic mismatch.
"""

import hashlib


def checkpoint_norm_identity(path, training_sha256):
    data = path.read_bytes()
    saved_sha = hashlib.sha256(data).hexdigest()
    if saved_sha == training_sha256:
        relation = "byte_identical"
    elif not data.endswith(b"\n") and hashlib.sha256(data + b"\n").hexdigest() == training_sha256:
        relation = "checkpoint_omitted_one_final_newline"
    else:
        raise ValueError("Checkpoint YAM norm differs from RTC training contract")
    return {
        "checkpoint_norm_sha256": saved_sha,
        "training_norm_sha256": training_sha256,
        "identity_relation": relation,
    }
