"""vla: stdlib-only project entrypoint. Plans are read-only; run is explicit."""

import argparse
import json
from pathlib import Path
import subprocess

from vla_platform.artifacts import seal_bundle
from vla_platform.artifacts import validate_bundle
from vla_platform.inventory import create_inventory
from vla_platform.project import Project
from vla_platform.project import tree_digest
from vla_platform.runtime import audit_environment
from vla_platform.runtime import execute
from vla_platform.splits import create_split
from vla_platform.splits import inspect_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("models")
    sub.add_parser("backends")
    hash_tree = sub.add_parser("hash-tree")
    hash_tree.add_argument("directory", type=Path)
    for name in ("datasets", "algorithms"):
        sub.add_parser(name)
    inventory = sub.add_parser("inventory")
    inventory_sub = inventory.add_subparsers(dest="operation", required=True)
    inventory_create = inventory_sub.add_parser("create")
    inventory_create.add_argument("--source-root", type=Path, required=True)
    inventory_create.add_argument("--listing", type=Path, required=True)
    inventory_create.add_argument("--output", type=Path, required=True)
    split = sub.add_parser("split")
    split_sub = split.add_subparsers(dest="operation", required=True)
    split_create = split_sub.add_parser("create")
    split_create.add_argument("recipe", type=Path)
    split_create.add_argument("--output", type=Path, required=True)
    split_inspect = split_sub.add_parser("inspect")
    split_inspect.add_argument("manifest", type=Path)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("experiment", type=Path)
        p.add_argument("operation", choices=("train", "infer", "benchmark", "export"))
        p.add_argument("--run-id", required=True)
    env = sub.add_parser("env")
    env.add_argument("operation", choices=("plan", "create", "audit"))
    env.add_argument("profile", help="Project-relative Conda profile")
    bundle = sub.add_parser("bundle")
    bundle.add_argument("operation", choices=("seal", "check"))
    bundle.add_argument("path", type=Path)
    bundle.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "bundle":
            if args.operation == "seal":
                if args.output is None:
                    parser.error("bundle seal requires --output")
                result = seal_bundle(args.path, args.output)
            else:
                result = validate_bundle(args.path)
        else:
            project = Project(args.root)
            if args.action in ("models", "backends"):
                registry = project.models() if args.action == "models" else project.backends()
                result = [
                    {
                        "id": key,
                        "status": value[1]["status"],
                        "backend": value[1].get("backend", key),
                        "operations": sorted(value[1].get("operations", {})),
                        "note": value[1].get("note", ""),
                    }
                    for key, value in registry.items()
                ]
            elif args.action == "hash-tree":
                result = {"directory": str(args.directory.resolve()), "sha256": tree_digest(args.directory)}
            elif args.action in ("datasets", "algorithms"):
                from vla_platform.project import read_toml  # noqa: PLC0415

                directory = project.root / "configs" / args.action
                result = [
                    {"id": read_toml(path).get("id"), "path": str(path.relative_to(project.root))}
                    for path in sorted(directory.glob("*.toml"))
                ]
            elif args.action == "split":
                if args.operation == "create":
                    output = args.output if args.output.is_absolute() else project.root / args.output
                    result = create_split(project.root, args.recipe, output)
                else:
                    path = args.manifest if args.manifest.is_absolute() else project.root / args.manifest
                    manifest, _, _ = inspect_split(project.root, path)
                    result = {
                        "status": "source_and_membership_verified",
                        "dataset_id": manifest["dataset_id"],
                        "split_id": manifest["split_id"],
                        "counts": {name: len(ids) for name, ids in manifest["partitions"].items()},
                    }
            elif args.action == "inventory":
                result = create_inventory(args.source_root, args.listing, args.output)
            elif args.action in ("plan", "run"):
                experiment = args.experiment if args.experiment.is_absolute() else project.root / args.experiment
                plan = project.plan(experiment, args.operation, args.run_id)
                if args.action == "run":
                    return execute(plan)
                result = plan.to_dict()
            elif args.operation == "audit":
                from vla_platform.project import inside  # noqa: PLC0415
                from vla_platform.project import read_toml  # noqa: PLC0415

                result = audit_environment(read_toml(inside(project.root, args.profile))["prefix"])
            else:
                result = project.environment_plan(args.profile)
                if args.operation == "create":
                    return subprocess.run(result["command"], cwd=project.root, check=False).returncode
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"vla: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
