"""Gate a trained Pi0.5 RTC Orbax checkpoint before FP32 conversion.

RTC changes the training/inference flow-time contract, not the weight mapping.
The existing audited mapper is reused only after the immutable training
contract, complete parameter metadata and checkpoint-specific norm agree.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from rtc_provenance import source_params_sha256
from rtc_norm_identity import checkpoint_norm_identity


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(checkpoint, contract):
    if (checkpoint / "model.safetensors").exists():
        raise ValueError("Expected original JAX/Orbax checkpoint, not converted weights")
    if not (checkpoint / "params" / "_METADATA").is_file():
        raise ValueError("Incomplete Orbax params: missing _METADATA")
    norm = checkpoint / "assets" / "yam" / "norm_stats.json"
    if not norm.is_file():
        raise ValueError("Checkpoint's YAM norm is missing")
    checkpoint_norm_identity(norm, contract.get("norm_sha256"))
    model = contract.get("model", {})
    delay = model.get("rtc_training_max_delay")
    if (
        model.get("pi05") is not True
        or model.get("action_horizon") != 50
        or model.get("action_dim") != 32
        or isinstance(delay, bool)
        or not isinstance(delay, int)
        or not 0 < delay < 50
    ):
        raise ValueError("Contract is not trained Pi0.5 YAM RTC H50/32D")
    return delay, norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, help="Audited JAX-to-PyTorch converter; defaults to repo or sibling")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new immutable conversion directory")
    contract = json.loads(args.training_contract.read_text())
    delay, norm = validate(args.checkpoint, contract)
    script = Path(__file__).resolve()
    repo_mapper = (
        script.parents[2] / "adapters/openpi/convert_jax_model_to_pytorch.py"
        if len(script.parents) > 2 else None
    )
    mapper = args.converter or (
        repo_mapper if repo_mapper and repo_mapper.is_file() else script.with_name("convert_jax_model_to_pytorch.py")
    )
    if not mapper.is_file():
        parser.error("The audited converter is not mounted in this container")
    subprocess.run(
        [sys.executable, str(mapper), "--checkpoint_dir", str(args.checkpoint), "--config_name", "pi05_yam",
         "--output_path", str(args.output), "--precision", "float32"],
        check=True,
    )
    audit = json.loads((args.output / "conversion_audit.json").read_text())
    if audit.get("output_precision") != "float32":
        raise RuntimeError("RTC conversion did not preserve FP32 weights")
    manifest = {
        "route": "pi05_yam_trained_rtc",
        "status": "fp32_converted_not_accuracy_or_engine_validated",
        "max_delay_steps": delay,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata_sha256": sha256(args.checkpoint / "params" / "_METADATA"),
        "source_params_files_sha256": source_params_sha256(args.checkpoint),
        "training_contract_sha256": sha256(args.training_contract),
        "norm_stats_sha256": sha256(norm),
        "norm_identity": checkpoint_norm_identity(norm, contract["norm_sha256"]),
        "conversion_audit_sha256": sha256(args.output / "conversion_audit.json"),
        "model_weights_sha256": sha256(args.output / "model.safetensors"),
        "precision": "FP32 preserved; BF16 compute is a later separately compared candidate",
        "rtc_semantics": "clean committed prefix at flow time zero; postfix denoised; no guidance fallback",
    }
    (args.output / "rtc_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
