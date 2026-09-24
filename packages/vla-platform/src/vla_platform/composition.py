"""Compile versioned model, data and algorithm components into a native backend plan."""

import json
import math
from pathlib import Path

from vla_platform.artifacts import validate_bundle
from vla_platform.contracts import require
from vla_platform.contracts import validate_contract
from vla_platform.project import identifier
from vla_platform.project import inside
from vla_platform.project import read_toml
from vla_platform.project import tree_digest
from vla_platform.splits import inspect_split


def _algorithm(root, relative):
    path = inside(root, relative)
    algorithm = read_toml(path)
    require(algorithm.get("schema_version") == 1, "Unsupported algorithm component version")
    allowed = {"schema_version", "id", "backend", "operation", "method", "state_space",
               "action_space", "overridable", "parameters"}
    require(set(algorithm).issubset(allowed), "Unknown algorithm component fields")
    identifier(algorithm.get("id"))
    for key in ("backend", "operation", "method", "state_space", "action_space"):
        require(isinstance(algorithm.get(key), str) and algorithm[key], f"Missing algorithm {key}")
    require(algorithm["operation"] in ("train", "infer", "benchmark", "export"), "Invalid algorithm operation")
    parameters = algorithm.get("parameters", {})
    require(isinstance(parameters, dict), "Algorithm parameters must be a table")
    require(all(type(value) in (str, int, float, bool)
                and (type(value) is not float or math.isfinite(value)) for value in parameters.values()),
            "Algorithm parameters must be scalar")
    overridable = algorithm.get("overridable", [])
    require(isinstance(overridable, list) and all(isinstance(key, str) for key in overridable)
            and len(overridable) == len(set(overridable)),
            "Invalid algorithm override list")
    require(set(overridable).issubset(parameters), "Overridable keys need defaults")
    return path, algorithm


def _native_split_binding(root, backend, operation, parameters, selected, episodes, dataset):
    binding = backend["operations"][operation].get("split_binding")
    require(isinstance(binding, dict), "Backend has no split-bound training interface")
    parameter = binding.get("parameter")
    cursor = binding.get("json_path")
    field = binding.get("episode_field")
    require(parameter in parameters and isinstance(parameters[parameter], str), "Missing split-bound native config")
    native = Path(parameters[parameter]).expanduser()
    native = native.resolve() if native.is_absolute() else inside(root, str(native))
    require(native.is_file(), f"Missing split-bound native config: {native}")
    current = json.loads(native.read_text())
    if binding.get("type") == "platform_manifest":
        require(isinstance(current, dict), "Native Pi recipe must be a JSON object")
        require(Path(current.get("dataset", "")).resolve() == Path(dataset["source_root"]).resolve(),
                "Native recipe dataset differs from split source")
        return native
    require(isinstance(cursor, list) and cursor and all(isinstance(key, str) for key in cursor),
            "Invalid backend split JSON path")
    require(field in ("episode_id", "asset_uri"), "Invalid backend split episode field")
    for key in cursor:
        require(isinstance(current, dict) and key in current, f"Missing native split field: {key}")
        current = current[key]
    require(isinstance(current, list), "Native training selection must be a list")
    actual = [str(value) for value in current]
    require(len(actual) == len(set(actual)), "Duplicate native training episode")
    desired = [str(episodes[episode_id].get(field)) for episode_id in selected]
    require(all(value != "None" for value in desired), f"Episode inventory lacks {field}")
    require(sorted(actual) == sorted(desired), "Native training selection differs from frozen train split")
    if backend["id"] == "openwam":
        loader = json.loads(native.read_text()).get("dataloader", {})
        require(Path(loader.get("dataset_dir", "")).resolve() == Path(dataset["source_root"]).resolve(),
                "OpenWAM native dataset differs from split source")
        require(loader.get("action_mode") == "joint" and loader.get("action_semantics") == "absolute"
                and loader.get("camera_layout") == dataset["cameras"],
                "OpenWAM native data semantics differ from dataset component")
    return native


def resolve(project, experiment, operation):
    """Return a legacy-compatible experiment plus every extra provenance source."""
    root = project.root
    require(experiment.get("schema_version") == 2, "Unsupported modular experiment version")
    allowed = {"schema_version", "operation", "model", "algorithm", "environment", "contract",
               "dataset", "split_manifest", "bundle_manifest", "overrides"}
    require(set(experiment).issubset(allowed), "Unknown modular experiment fields")
    require(operation == experiment.get("operation"), "Experiment operation mismatch")
    name = identifier(experiment.get("model"))
    models = project.models()
    require(name in models, f"Unknown model: {name}")
    _, model = models[name]
    require(model["status"] == "implemented" and operation in model.get("operations", []),
            "Model operation is not implemented")
    backend = project.backends()[model["backend"]][1]
    algorithm_path, algorithm = _algorithm(root, experiment["algorithm"])
    require(algorithm["backend"] == model["backend"], "Algorithm/backend mismatch")
    require(algorithm["operation"] == operation, "Algorithm/operation mismatch")
    io = model.get("io", {}).get(operation)
    require(isinstance(io, dict), "Model lacks an explicit modular IO contract")
    require(isinstance(io.get("methods"), list), "Model IO methods must be a list")
    for field in ("state_space", "action_space"):
        require(io.get(field) == algorithm[field], f"Model/algorithm {field} mismatch")
    require(algorithm["method"] in io.get("methods", []), "Model does not support this algorithm method")
    contract_path = inside(root, experiment["contract"])
    contract = validate_contract(read_toml(contract_path))
    require(contract["id"] in model["contracts"], "Model/robot contract mismatch")
    require(io.get("contract_id") == contract["id"], "Modular IO/robot contract mismatch")
    require(io.get("cameras") == contract["cameras"], "Model/robot camera order mismatch")
    overrides = experiment.get("overrides", {})
    require(isinstance(overrides, dict) and set(overrides).issubset(algorithm.get("overridable", [])),
            "Experiment overrides undeclared algorithm parameters")
    parameters = dict(algorithm.get("parameters", {}))
    for key, value in overrides.items():
        require(type(value) is type(parameters[key]), f"Override type changed for {key}")
        parameters[key] = value
    sources = [algorithm_path]
    receipt = {"algorithm": algorithm["id"], "method": algorithm["method"]}
    if operation == "train":
        require("bundle_manifest" not in experiment, "Training consumes a split, not an inference bundle")
        require("dataset" in experiment and "split_manifest" in experiment,
                "Training requires a dataset component and frozen split")
        split_path = Path(experiment["split_manifest"]).expanduser()
        split_path = split_path.resolve() if split_path.is_absolute() else inside(root, str(split_path))
        split, dataset, episodes = inspect_split(root, split_path)
        dataset_path = inside(root, split["dataset_config"])
        require(inside(root, experiment["dataset"]) == dataset_path, "Experiment/split dataset mismatch")
        require(dataset["contract_id"] == contract["id"], "Dataset/robot contract mismatch")
        for field in ("state_space", "action_space"):
            require(dataset[field] == io[field], f"Dataset/model {field} mismatch")
        require(dataset["cameras"] == io["cameras"], "Dataset/model camera order mismatch")
        native = _native_split_binding(root, backend, operation, parameters,
                                       split["partitions"]["train"], episodes, dataset)
        sources.extend((dataset_path, inside(root, split["recipe"]), split_path,
                        Path(dataset["episode_manifest"]), native))
        if model["backend"] == "openpi":
            pi_recipe = json.loads(native.read_text())
            assets = Path(pi_recipe["norm_assets"]).resolve() / "yam"
            sources.extend((assets / "norm_stats.json", assets / "provenance.json"))
            params = Path(pi_recipe["init_params"]).resolve()
            expected = pi_recipe["init_params_tree_sha256"]
            require(tree_digest(params) == expected, "Pi base checkpoint tree SHA256 mismatch")
            receipt.update({"init_params": str(params), "init_params_tree_sha256": expected})
        if model["backend"] == "openwam":
            loader = json.loads(native.read_text())["dataloader"]
            stats = Path(loader["normalization_json"]).resolve()
            require(stats.is_file(), "Missing OpenWAM train-only normalization statistics")
            payload = json.loads(stats.read_text())
            require([str(value) for value in payload.get("episodes", [])]
                    == [str(value) for value in loader["episodes"]],
                    "OpenWAM normalization statistics differ from selected episodes")
            sources.append(stats)
        receipt.update({"dataset": dataset["id"], "split": split["split_id"],
                        "split_manifest": str(split_path),
                        "train_episodes": len(split["partitions"]["train"])})
    else:
        require("dataset" not in experiment and "split_manifest" not in experiment,
                "Inference/export must use checkpoint assets, not a training split")
        require(isinstance(experiment.get("bundle_manifest"), str) and experiment["bundle_manifest"],
                "Inference/export requires a sealed bundle manifest")
        bundle_path = Path(experiment["bundle_manifest"]).expanduser()
        bundle_path = bundle_path.resolve() if bundle_path.is_absolute() else inside(root, str(bundle_path))
        bundle = validate_bundle(bundle_path)
        manifest = json.loads(bundle_path.read_text())
        require(bundle["model"] == name and manifest["contract_id"] == contract["id"],
                "Inference bundle model/contract mismatch")
        if "model_version" in parameters:
            require(parameters["model_version"] == bundle["model_version"],
                    "Inference model version differs from bundle")
        checkpoint = Path(parameters.get("checkpoint", "")).expanduser()
        require(checkpoint.is_absolute() and checkpoint.resolve().is_relative_to(bundle_path.parent),
                "Inference checkpoint must be inside its sealed bundle")
        weights = [Path(item["path"]) for item in manifest["files"] if item["role"] == "weights"]
        require(any((bundle_path.parent / weight).resolve().is_relative_to(checkpoint.resolve())
                    for weight in weights), "Inference checkpoint excludes sealed weights")
        sources.append(bundle_path)
        receipt.update({"bundle_manifest": str(bundle_path), "bundle_sha256": bundle["manifest_sha256"]})
    legacy = {
        "schema_version": 1,
        "model": name,
        "environment": experiment["environment"],
        "contract": experiment["contract"],
        "parameters": parameters,
    }
    return legacy, sources, receipt
