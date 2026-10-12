"""Archive comparable offline diagnostics; no inferred robot success rate."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    args = p.parse_args()
    result = {"phase": "offline critic initialization", "robot_success_rate_measured": False, "runs": []}
    for name in ["calql-success-s0", "calql-success-s1", "calql-mixed-s0", "td-success-s0"]:
        directory = args.run_root / name
        item = {"id": name}
        for file in ["run.json", "status.json", "latest-evaluation.json"]:
            path = directory / file
            item[file] = json.loads(path.read_text()) if path.exists() else None
        item["checkpoint_sha256"] = {}
        for path in sorted(directory.glob("*.pt")):
            with path.open("rb") as stream:
                item["checkpoint_sha256"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        result["runs"].append(item)
    (args.run_root / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
