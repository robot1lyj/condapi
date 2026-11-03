"""Execute pinned native protocol logic with stdlib-only dependency stubs.

This is not a model/tokenizer/numerical-loss test or a full dataset load. Native
recipe/render/collate/formatter function bodies are unchanged AST extractions.
Only package imports, base registry/types and non-language tensor collation are
replaced so no Torch/JAX/Transformers/datasets/pyarrow/numpy is imported.
"""

import __future__

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import urllib.request

PIN = "f5c9bc02137c331b009eccd44ece6f11b62187a8"
SOURCE_SHA256 = {
    "annotations/steerable_pipeline/writer.py": "855dabe6b1697c8e8f551d33efc8159089ea2b639601d6cb87ca73ded243350d",
    "datasets/dataset_reader.py": "0cb7190b9f6eccfa174bf3e89ad7b4868c32b85aa3db1ebf0dfd740682561b8f",
    "datasets/io_utils.py": "5179ceeb98cc9a1e5859fbcd4fc36e28cd9573a5a9aba26122fb1a86f2b59272",
    "datasets/language.py": "251f09d150c7058c166b2caa7c8fb0983978ce1188fa955e397a69fe87750587",
    "datasets/language_render.py": "60f3c72939a6edce42f2a45c185ebe850740cd1d87f949092eb18f1dfa5e5224",
    "datasets/recipe.py": "05bf54bd3678cef59698af5ee70c3606b84d48acc42b7f66f5a8925358f146b6",
    "policies/fineart_vla/configuration_fineart_vla.py": "9d664802b713c7bb7fa3d62c50a8f3fc8041ff553db6b21d299f426fa7edfdba",
    "policies/fineart_vla/modeling_fineart_vla.py": "60158c1e1ffb3b40fc89d71f3977ce287341e7e4a1631d3317aa2d9d7e148c95",
    "policies/fineart_vla/processor_fineart_vla.py": "5d3a3440c03756926d55a3ed20bd51fe8f67c92a2c52a7e9deb56fce66ddab6f",
    "processor/render_messages_processor.py": "ead5c7ccdcdc256bf23f76369415670015b659b7a52db94c5e2ae79a0dba62f2",
    "utils/collate.py": "6247666ba1e0beff90b25f41a606c84195ab15f1537f8331c00988b31f3ea9dc",
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--source-root",
    type=Path,
    required=True,
    help="Pinned checkout src/lerobot directory or a new source snapshot directory",
)
parser.add_argument(
    "--download-source",
    action="store_true",
    help="Explicitly fetch only the11 hash-pinned source files; does not install packages",
)
parser.add_argument("--output", type=Path, help="Optional new JSON evidence file; existing files are not overwritten")
args = parser.parse_args()
SOURCE = args.source_root.resolve()
for relative, expected in SOURCE_SHA256.items():
    path = SOURCE / relative
    if not path.exists() and args.download_source:
        url = f"https://raw.githubusercontent.com/huggingface/lerobot/{PIN}/src/lerobot/{relative}"
        data = urllib.request.urlopen(url, timeout=30).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Downloaded pinned source hash mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(
            f"Missing or different pinned source: {relative}; use the fixed checkout or explicit --download-source"
        )
CONSTANTS = {
    "LANGUAGE_PERSISTENT": "language_persistent",
    "LANGUAGE_EVENTS": "language_events",
    "MESSAGES_RENDERED": "messages_rendered",
    "QUERY_KIND": "query_kind",
    "QUERY_TEXT": "query_text",
}


def source_ast(relative):
    return ast.parse((SOURCE / relative).read_text())


def protocol_module(name, relative, env, only=None):
    tree = source_ast(relative)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("lerobot")):
            continue
        if isinstance(node, ast.Import) and any(
            a.name.split(".")[0] in {"numpy", "torch", "datasets", "pyarrow"} for a in node.names
        ):
            continue
        if (
            only is not None
            and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
            and getattr(node, "name", None) not in only
        ):
            continue
        nodes.append(node)
    module = types.ModuleType("native_protocol_" + name)
    sys.modules[module.__name__] = module
    module.__dict__.update(env)
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(SOURCE / relative),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        module.__dict__,
    )
    return module


language = protocol_module("language", "datasets/language.py", CONSTANTS)
recipe = protocol_module("recipe", "datasets/recipe.py", CONSTANTS)
render = protocol_module(
    "render",
    "datasets/language_render.py",
    {
        **CONSTANTS,
        "unwrap_scalar": lambda x: x,
        "column_for_style": language.column_for_style,
        "DEFAULT_BINDINGS": recipe.DEFAULT_BINDINGS,
        "TrainingRecipe": recipe.TrainingRecipe,
        "render_message_turns": recipe.render_message_turns,
    },
)


class Registry:
    @staticmethod
    def register(name):
        return lambda cls: cls


class Keys:
    COMPLEMENTARY_DATA = "complementary_data"
    OBSERVATION = "observation"
    ACTION = "action"


processor = protocol_module(
    "processor",
    "processor/render_messages_processor.py",
    {
        **CONSTANTS,
        "unwrap_scalar": lambda x: x,
        "TransitionKey": Keys,
        "ProcessorStepRegistry": Registry,
        "ProcessorStep": object,
        "ComplementaryDataProcessorStep": object,
    },
)
# Wire the pinned native processor's private rendering hook for this protocol test.
processor._render_sample = render.render_sample  # noqa: SLF001
formatter = protocol_module("formatter", "policies/fineart_vla/processor_fineart_vla.py", {}, only={"_format_messages"})
writer = protocol_module(
    "writer_check", "annotations/steerable_pipeline/writer.py", {}, only={"_validate_atom_invariants"}
)
collate = protocol_module(
    "collate",
    "utils/collate.py",
    {
        **CONSTANTS,
        "LANGUAGE_COLUMNS": language.LANGUAGE_COLUMNS,
        "default_collate": lambda rows: {k: [r[k] for r in rows] for k in rows[0]},
    },
    only={"lerobot_collate_fn"},
)
# Match the pinned native collator's private list-preservation configuration.
collate._PYTHON_LIST_KEYS = {  # noqa: SLF001
    "messages_rendered",
    "message_streams",
    "target_message_indices",
    *language.LANGUAGE_COLUMNS,
}
io = protocol_module(
    "language_io",
    "datasets/io_utils.py",
    {"LANGUAGE_COLUMNS": language.LANGUAGE_COLUMNS},
    only={"hf_transform_to_torch"},
)
native_cfg = source_ast("policies/fineart_vla/configuration_fineart_vla.py")
default_fn = next(
    n for n in native_cfg.body if isinstance(n, ast.FunctionDef) and n.name == "_fineart_vla_default_recipe"
)
raw_recipe = ast.literal_eval(next(n.value for n in default_fn.body if isinstance(n, ast.Return)))
training_recipe = recipe.TrainingRecipe.from_dict(raw_recipe)
step = processor.RenderTrainingMessagesStep(recipe=training_recipe)
task = "fold and stack the t-shirts"
timeline = [
    {
        "role": "assistant",
        "style": "subtask",
        "content": "Turn the shirt lengthwise.",
        "timestamp": 29.5,
        "camera": None,
        "tool_calls": None,
    },
    {
        "role": "assistant",
        "style": "subtask",
        "content": "Fold the near end toward the middle.",
        "timestamp": 35.5,
        "camera": None,
        "tool_calls": None,
    },
]
tests = []


def transition(persistent, index, timestamp=35.0):
    return {
        "action": [[0.0] * 14 for _ in range(50)],
        "complementary_data": {
            "task": task,
            "index": index,
            "timestamp": timestamp,
            "language_persistent": persistent,
            "language_events": [],
        },
    }


branches = {}
for i in range(1000):
    branch = render._select_recipe(training_recipe, i)  # noqa: SLF001 - Exercise the pinned native recipe selector.
    branches.setdefault(branch.weight, i)
    value = step(transition([], i))
    data = value["complementary_data"]
    assert data["messages_rendered"] == [{"role": "user", "content": task}]
    assert data["message_streams"] == ["low_level"]
    assert data["target_message_indices"] == []
    assert value["action"] == transition([], i)["action"]
assert set(branches) == {0.3, 0.7}
tests.append("1000 empty-language single anchors: task-to-action fallback, no text targets, H50 unchanged")

high, low = branches[0.3], branches[0.7]
single_messages = {}
for weight, idx in branches.items():
    known = step(transition(timeline, idx, 35.5))["complementary_data"]
    assert bool(known["target_message_indices"]) == (weight == 0.3)
    assert known["message_streams"] == (["high_level", "high_level"] if weight == 0.3 else ["low_level"])
    single_messages[str(weight)] = known
tests.append("known anchors retain native deterministic30/70 routing")

batch = {
    "action": ["known-high-actions", "unknown-high-actions", "known-low-actions", "unknown-low-actions"],
    "complementary_data": {
        "task": [task] * 4,
        "index": [high, high, low, low],
        "timestamp": [35.5] * 4,
        "language_persistent": [timeline, [], timeline, []],
        "language_events": [[], [], [], []],
    },
}
out = step(batch)
assert out["action"] == batch["action"]
assert out["complementary_data"]["target_message_indices"] == [[1], [], [], []]
assert out["complementary_data"]["message_streams"] == [
    ["high_level", "high_level"],
    ["low_level"],
    ["low_level"],
    ["low_level"],
]
tests.append("mixed known/empty batch keeps all4 samples, including both recipe-index branches at empty anchors")

raw_samples = [
    {"task": task, "language_persistent": timeline, "language_events": []},
    {"task": task, "language_persistent": [], "language_events": []},
]
collated = collate.lerobot_collate_fn(raw_samples)
assert collated["language_persistent"] == [timeline, []]
assert io.hf_transform_to_torch({"language_persistent": [timeline, []], "language_events": [[], []]})[
    "language_persistent"
] == [timeline, []]
tests.append("native language collate and HF transform preserve mixed per-anchor[] without cross-frame broadcast check")

before = step(transition(timeline, high, 34.96666666666667))["complementary_data"]
gap = step(transition([], high, 35.0))["complementary_data"]
after = step(transition(timeline, high, 35.5))["complementary_data"]
assert before["messages_rendered"][1]["content"] == "Turn the shirt lengthwise."
assert gap["target_message_indices"] == []
assert gap["messages_rendered"][0]["content"] == task
assert after["messages_rendered"][1]["content"] == "Fold the near end toward the middle."
tests.append("known->empty UNKNOWN anchor->known does not carry the previous subtask through the gap")

null_row = {
    "role": "assistant",
    "style": "subtask",
    "content": None,
    "timestamp": 35.0,
    "camera": None,
    "tool_calls": None,
}
assert render.active_at(35.0, persistent=[null_row], style="subtask")["content"] is None
null_high = step(transition([null_row], high))["complementary_data"]
null_low = step(transition([null_row], low))["complementary_data"]
assert null_high["target_message_indices"] == [1]
assert null_high["messages_rendered"][1]["content"] == ""
assert null_low["messages_rendered"] == [{"role": "user", "content": ""}]
prompt, spans = formatter._format_messages(  # noqa: SLF001 - Exercise the pinned native formatter.
    null_high["messages_rendered"], [1], "<eos>"
)
assert prompt[spans[1][0] : spans[1][1]] == "<eos>"
try:
    writer._validate_atom_invariants(null_row)  # noqa: SLF001 - Exercise the pinned native writer invariant.
except ValueError:
    pass
else:
    raise AssertionError("Official writer unexpectedly accepted null content and null tools")
tests.append(
    "null-content subtask is NOT absence: empty target+EOS or empty low-level instruction; official writer rejects null/null"
)

for empty_index in (high, low):
    data = step(transition([], empty_index))["complementary_data"]
    assert data["target_message_indices"] == []
    _, target_spans = formatter._format_messages(  # noqa: SLF001 - Exercise the pinned native formatter.
        data["messages_rendered"], [], "<eos>"
    )
    assert not [target_spans[i] for i in data["target_message_indices"]]
tests.append("fallback formatter has no supervised payload span; native tokenizer initializes all labels to-100")

assert not any(name in sys.modules for name in ("torch", "jax", "transformers", "datasets", "pyarrow", "numpy"))
sources = {}
for name in sorted(SOURCE_SHA256):
    path = SOURCE / name
    relative = f"src/lerobot/{name}"
    sources[relative] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "url": f"https://github.com/huggingface/lerobot/blob/{PIN}/{relative}",
    }
report = {
    "status": "PURE_PROTOCOL_CHECK_PASS",
    "commit": PIN,
    "training_run": False,
    "model_or_tensor_library_imported": False,
    "tests": tests,
    "source_snapshots": sources,
    "scope_limit": "Native AST recipe/render/collate/formatter logic with dependency stubs, not a real Arrow/HF Dataset load, tokenizer or numerical loss execution",
    "native_text_loss_evidence": "processor labels initialize-100, only target spans replace them; _lin_ce_small uses ignore_index=-100/reduction=sum and n_valid.clamp(min=1), so no subtask target contributes CE/z-loss",
}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
print(
    json.dumps(
        {
            "status": report["status"],
            "protocol_checks": len(tests),
            "model_imported": False,
            "commit": PIN,
            "result": str(args.output) if args.output else None,
        },
        indent=2,
    )
)
