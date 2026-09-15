"""Bind immutable real replay observations to a new checkpoint's training norm."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source, checkpoint, output):
    source, checkpoint, output = map(Path, (source, checkpoint, output))
    suite = json.loads((source / "suite.json").read_text())
    norm = checkpoint / "assets/yam/norm_stats.json"
    norm_hash = digest(norm)
    metadata_hash = digest(checkpoint / "params/_METADATA")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(norm, output / "norm_stats.json")
    for entry in suite["samples"]:
        for key in ("sample", "provenance"):
            if Path(entry[key]).name != entry[key]:
                raise ValueError("Expected flat, local replay filenames")
        sample = source / entry["sample"]
        provenance = json.loads((source / entry["provenance"]).read_text())
        if digest(sample) != provenance["sample_sha256"]:
            raise ValueError("Source observation hash mismatch")
        shutil.copy2(sample, output / entry["sample"])
        provenance.update(
            norm_stats_sha256=norm_hash,
            parent_provenance_sha256=digest(source / entry["provenance"]),
            normalization_scope="checkpoint_training_norm",
            checkpoint_metadata_sha256=metadata_hash,
        )
        (output / entry["provenance"]).write_text(json.dumps(provenance, indent=2))
    suite.update(
        parent_suite_sha256=digest(source / "suite.json"),
        norm_stats_sha256=norm_hash,
        normalization_scope="checkpoint_training_norm",
        normalization_method="unaltered checkpoint assets/yam/norm_stats.json",
        base_model_only=False,
        checkpoint_metadata_sha256=metadata_hash,
        validation_scope="real training replay; conversion parity and latency, not holdout task success",
    )
    (output / "suite.json").write_text(json.dumps(suite, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.source, args.checkpoint, args.output)
