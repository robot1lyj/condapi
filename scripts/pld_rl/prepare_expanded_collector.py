"""Copy a learned actor unchanged and pin a separate wider collection contract."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"packages/pld-rl/src"))
from yam_pld_rl.block_contract import fingerprint
from yam_pld_rl.collector import expanded_manifest
from yam_pld_rl.collector import smooth_manifest
from yam_pld_rl.data import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=("expanded", "continuous"), default="expanded")
    args = p.parse_args()
    parent = json.loads((args.parent/"manifest.json").read_text())
    if sha256(args.parent/"actor.pt") != parent["actor_sha256"]:
        raise ValueError("parent weights differ from manifest")
    builder = smooth_manifest if args.profile == "continuous" else expanded_manifest
    manifest = builder(parent, parent_snapshot=sha256(args.parent/"manifest.json"))
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("actor.pt", "fk.xml"):
        shutil.copy2(args.parent/name, args.output/name)
        if sha256(args.output/name) != sha256(args.parent/name):
            raise ValueError("bundle copy changed")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"snapshot_sha256": sha256(args.output/"manifest.json"),
                      "contract_sha256": fingerprint(manifest["contract"]),
                      "actor_sha256": manifest["actor_sha256"], "weights_unchanged": True}))


if __name__ == "__main__":
    main()
