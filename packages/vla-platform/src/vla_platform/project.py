"""Model selection and shared backend command planning, without model imports."""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
import os
from pathlib import Path
import re
import string
import tomllib

from vla_platform.contracts import require
from vla_platform.contracts import validate_contract


def read_toml(path):
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def digest(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"File changed while hashing: {path}")
    return value


def tree_digest(root):
    """Content identity for a local checkpoint directory, independent of mtimes."""
    root = Path(root).resolve()
    require(root.is_dir(), "Expected a checkpoint directory")
    entries = sorted(root.rglob("*"))
    require(all(not path.is_symlink() for path in entries), "Checkpoint symlink is not allowed")
    files = [path for path in entries if path.is_file()]
    require(bool(files), "Checkpoint directory is empty")
    hashes = {}
    for path in files:
        require(path.resolve().is_relative_to(root), "Checkpoint file escapes its root")
        hashes[str(path.relative_to(root))] = digest(path)
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value), "Invalid identifier")
    return value


def inside(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Expected project-relative path")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Path escapes project root")
    return path


def render(template, values):
    require(isinstance(template, str), "Command arguments must be strings")
    for _, field, spec, conversion in string.Formatter().parse(template):
        if field is not None:
            require(field in values and not spec and not conversion, f"Unknown or unsafe placeholder: {field}")
    return template.format_map(values)


@dataclass(frozen=True)
class Plan:
    schema_version: int
    model: str
    operation: str
    run_id: str
    cwd: str
    prefix: str
    command: list[str]
    output: str
    contract_id: str
    source_hashes: dict[str, str]
    target: str
    implementation: str
    acceptance: str = "planned_not_executed"
    components: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.settings_path = self.root / "vla.toml"
        self.settings = read_toml(self.settings_path)
        require(self.settings.get("schema_version") == 2, "Unsupported project version")

    def models(self):
        directory = inside(self.root, self.settings["models"])
        found = {}
        for path in sorted(directory.glob("*.toml")):
            model = read_toml(path)
            name = identifier(model["id"])
            require(name not in found, f"Duplicate model: {name}")
            require(model.get("schema_version") == 1, f"Invalid model schema: {name}")
            require(model.get("status") in ("implemented", "planned"), f"Invalid model status: {name}")
            require(model.get("backend") in self.backends(), f"Unknown backend for model: {name}")
            require(isinstance(model.get("operations", []), list), "Model operations must be a capability list")
            found[name] = (path, model)
        return found

    def backends(self):
        found = {}
        for path in sorted(inside(self.root, self.settings["backends"]).glob("*/backend.toml")):
            backend = read_toml(path)
            name = identifier(backend["id"])
            require(name not in found, f"Duplicate backend: {name}")
            require(backend.get("schema_version") == 1, f"Invalid backend schema: {name}")
            require(backend.get("status") == "implemented", f"Backend is not implemented: {name}")
            found[name] = (path, backend)
        return found

    def plan(self, experiment_path, operation, run_id):
        identifier(run_id)
        experiment_path = Path(experiment_path).resolve()
        experiment = read_toml(experiment_path)
        extra_sources = []
        components = {}
        if experiment.get("schema_version") == 2:
            from vla_platform.composition import resolve  # noqa: PLC0415

            experiment, extra_sources, components = resolve(self, experiment, operation)
        require(experiment.get("schema_version") == 1, "Unsupported experiment version")
        name = identifier(experiment["model"])
        require(name in self.models(), f"Unknown model: {name}")
        model_path, model = self.models()[name]
        require(model["status"] == "implemented", f"{name}: planned, not integrated")
        require(operation in model.get("operations", []), f"{name} does not implement {operation}")
        backend_path, backend = self.backends()[model["backend"]]
        operations = backend.get("operations", {})
        require(operation in operations, f"Backend {model['backend']} does not implement {operation}")
        op = operations[operation]
        profile_path = inside(self.root, experiment["environment"])
        profile = read_toml(profile_path)
        require(profile.get("manager") == "conda", "Only Conda runtime profiles are supported")
        require(profile.get("model") == name, "Environment belongs to another model family")
        require(profile.get("target") in ("workstation", "server", "thor"), "Unsupported execution target")
        require(profile.get("target") in op["targets"], "Operation cannot run on this target")
        require(profile.get("schema_version") == 1, "Unsupported environment profile version")
        prefix = Path(profile["prefix"])
        require(prefix.is_absolute() and len(prefix.parts) >= 4, "Use a dedicated absolute Conda prefix")
        require(prefix.name not in ("base", "miniconda3", "anaconda3"), "Do not use a base environment")
        contract_path = inside(self.root, experiment["contract"])
        contract = validate_contract(read_toml(contract_path))
        require(contract["id"] in model["contracts"], "Model has no adapter for this robot contract")
        values = experiment.get("parameters", {})
        require(isinstance(values, dict), "parameters must be a table")
        require(all(type(v) in (str, int, float, bool) for v in values.values()), "Parameters must be scalar")
        reserved = {"root", "output", "run_id", "contract", "policy_type", "split_manifest"}
        require(not reserved.intersection(values), "Parameters overwrite platform fields")
        modular = bool(components)
        required = set(op.get("modular_required", op.get("required", [])) if modular else op.get("required", []))
        require(required.issubset(values), f"Missing parameters: {sorted(required - values.keys())}")
        for key, allowed in op.get("choices", {}).items():
            if key in values:
                require(values[key] in allowed, f"Unsupported {key}: {values[key]}")
        # Native configuration remains upstream JSON; record its hash, not a lossy translation.
        file_sources = []
        values = dict(values)
        files = (
            op.get("modular_file_parameters", op.get("file_parameters", []))
            if modular else op.get("file_parameters", [])
        )
        for key in files:
            path = Path(values[key]).expanduser()
            path = path.resolve() if path.is_absolute() else (self.root / path).resolve()
            require(path.is_file(), f"Missing native configuration: {path}")
            values[key] = str(path)
            file_sources.append(path)
        output = inside(self.root, self.settings["runs"]) / run_id
        values = {
            **values,
            "root": str(self.root),
            "output": str(output / "artifacts"),
            "run_id": run_id,
            "contract": str(contract_path),
            "policy_type": model.get("policy_type", ""),
            "split_manifest": components.get("split_manifest", ""),
        }
        template = op.get("modular_command", op["command"]) if modular else op["command"]
        args = [render(token, values) for token in template]
        require(bool(args) and args[0] == "python", "Model entrypoint must run Python inside Conda")
        command = ["conda", "run", "--no-capture-output", "--prefix", str(prefix), *args]
        sources = [self.settings_path, experiment_path, model_path, backend_path, profile_path, contract_path]
        sources.extend(file_sources)
        sources.extend(extra_sources)
        sources.extend(inside(self.root, path) for path in backend.get("sources", []))
        sources.append(inside(self.root, profile["spec"]))
        if profile.get("lock"):
            sources.append(inside(self.root, profile["lock"]))
        if profile["target"] == "thor" and operation in ("infer", "benchmark"):
            sources.append(self.root / "scripts/thor/maxn_session.py")
        return Plan(
            2,
            name,
            operation,
            run_id,
            str(self.root),
            str(prefix),
            command,
            str(output),
            contract["id"],
            {str(p): digest(p) for p in sources},
            profile["target"],
            model["backend"],
            components=components,
        )

    def environment_plan(self, profile_path):
        path = inside(self.root, profile_path)
        profile = read_toml(path)
        require(profile.get("manager") == "conda", "Only Conda environments are supported")
        require(profile.get("schema_version") == 1, "Unsupported environment profile version")
        require(profile.get("target") in ("workstation", "server", "thor"), "Unsupported execution target")
        require(profile.get("model") in self.models(), "Unknown environment model")
        spec = inside(self.root, profile["spec"])
        prefix = Path(profile["prefix"])
        require(prefix.is_absolute() and len(prefix.parts) >= 4, "Use a dedicated absolute Conda prefix")
        require(prefix.name not in ("base", "miniconda3", "anaconda3"), "Do not use a base environment")
        require(not prefix.exists(), "Environment already exists; creation will not update it")
        return {
            "profile": str(path),
            "spec_sha256": digest(spec),
            "dependency_lock_sha256": digest(inside(self.root, profile["lock"])) if profile.get("lock") else None,
            "status": "bootstrap_not_model_ready",
            "command": ["conda", "env", "create", "--prefix", str(prefix), "--file", str(spec)],
        }


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
