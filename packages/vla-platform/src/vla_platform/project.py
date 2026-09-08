"""Declarative plugin registry and dependency-free command planning."""

from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
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
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    plugin: str
    operation: str
    run_id: str
    cwd: str
    prefix: str
    command: list[str]
    output: str
    contract_id: str
    source_hashes: dict[str, str]
    target: str
    acceptance: str = "planned_not_executed"

    def to_dict(self):
        return asdict(self)


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.settings_path = self.root / "vla.toml"
        self.settings = read_toml(self.settings_path)
        require(self.settings.get("schema_version") == 1, "Unsupported project version")

    def plugins(self):
        directory = inside(self.root, self.settings["plugins"])
        found = {}
        for path in sorted(directory.glob("*/plugin.toml")):
            plugin = read_toml(path)
            name = identifier(plugin["id"])
            require(name not in found, f"Duplicate plugin: {name}")
            require(plugin.get("schema_version") == 1, f"Invalid plugin schema: {name}")
            require(plugin.get("status") in ("implemented", "planned"), f"Invalid plugin status: {name}")
            found[name] = (path, plugin)
        return found

    def plan(self, experiment_path, operation, run_id):
        identifier(run_id)
        experiment_path = Path(experiment_path).resolve()
        experiment = read_toml(experiment_path)
        require(experiment.get("schema_version") == 1, "Unsupported experiment version")
        name = identifier(experiment["plugin"])
        require(name in self.plugins(), f"Unknown plugin: {name}")
        plugin_path, plugin = self.plugins()[name]
        require(plugin["status"] == "implemented", f"{name}: planned, not integrated")
        operations = plugin.get("operations", {})
        require(operation in operations, f"{name} does not implement {operation}")
        op = operations[operation]
        profile_path = inside(self.root, experiment["environment"])
        profile = read_toml(profile_path)
        require(profile.get("manager") == "conda", "Only Conda runtime profiles are supported")
        require(profile.get("plugin") == name, "Environment belongs to another model family")
        require(profile.get("target") in ("workstation", "server", "thor"), "Unsupported execution target")
        require(profile.get("target") in op["targets"], "Operation cannot run on this target")
        require(profile.get("schema_version") == 1, "Unsupported environment profile version")
        prefix = Path(profile["prefix"])
        require(prefix.is_absolute() and len(prefix.parts) >= 4, "Use a dedicated absolute Conda prefix")
        require(prefix.name not in ("base", "miniconda3", "anaconda3"), "Do not use a base environment")
        contract_path = inside(self.root, experiment["contract"])
        contract = validate_contract(read_toml(contract_path))
        require(contract["id"] in plugin["contracts"], "Model has no adapter for this robot contract")
        values = experiment.get("parameters", {})
        require(isinstance(values, dict), "parameters must be a table")
        require(all(type(v) in (str, int, float, bool) for v in values.values()), "Parameters must be scalar")
        reserved = {"root", "output", "run_id", "contract"}
        require(not reserved.intersection(values), "Parameters overwrite platform fields")
        required = set(op.get("required", []))
        require(required.issubset(values), f"Missing parameters: {sorted(required - values.keys())}")
        for key, allowed in op.get("choices", {}).items():
            require(values.get(key) in allowed, f"Unsupported {key}: {values.get(key)}")
        output = inside(self.root, self.settings["runs"]) / run_id
        values = {
            **values,
            "root": str(self.root),
            "output": str(output / "artifacts"),
            "run_id": run_id,
            "contract": str(contract_path),
        }
        args = [render(token, values) for token in op["command"]]
        require(bool(args) and args[0] == "python", "Plugin entrypoint must run Python inside Conda")
        command = ["conda", "run", "--no-capture-output", "--prefix", str(prefix), *args]
        sources = [self.settings_path, experiment_path, plugin_path, profile_path, contract_path]
        sources.extend(inside(self.root, path) for path in plugin.get("sources", []))
        if profile["target"] == "thor" and operation in ("infer", "benchmark"):
            sources.append(self.root / "scripts/thor/maxn_session.py")
        return Plan(
            1,
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
        )

    def environment_plan(self, profile_path):
        path = inside(self.root, profile_path)
        profile = read_toml(path)
        require(profile.get("manager") == "conda", "Only Conda environments are supported")
        require(profile.get("schema_version") == 1, "Unsupported environment profile version")
        require(profile.get("target") in ("workstation", "server", "thor"), "Unsupported execution target")
        require(profile.get("plugin") in self.plugins(), "Unknown environment plugin")
        spec = inside(self.root, profile["spec"])
        prefix = Path(profile["prefix"])
        require(prefix.is_absolute() and len(prefix.parts) >= 4, "Use a dedicated absolute Conda prefix")
        require(prefix.name not in ("base", "miniconda3", "anaconda3"), "Do not use a base environment")
        require(not prefix.exists(), "Environment already exists; creation will not update it")
        return {
            "profile": str(path),
            "spec_sha256": digest(spec),
            "status": "bootstrap_not_model_ready",
            "command": ["conda", "env", "create", "--prefix", str(prefix), "--file", str(spec)],
        }


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
