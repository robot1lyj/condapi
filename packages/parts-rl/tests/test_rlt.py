# ruff: noqa: PT011, PT012, PT018, PLC0415
"""RLT protocol/assets/static checks only: never import Torch or run a network."""

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from adapters.rlt.common import bind_token
from adapters.rlt.common import load_recipe
from adapters.rlt.common import load_token_recipe
import numpy as np
from parts_policy import PartsPolicyExtension
import pytest
from rlt_policy import verify_manifest
from scripts.parts.build_behavior_manifest import build
from test_contract import Base
from test_contract import Features
from test_contract import declaration
from test_contract import manifest
from test_contract import sent
from vla_platform import parts as wire

from parts_rl.rlt_contract import ALGORITHM
from parts_rl.rlt_contract import FEATURE_EXTRACTOR
from parts_rl.rlt_contract import PREFIX_SCHEMA
from parts_rl.rlt_contract import UPSTREAM_COMMIT
from parts_rl.rlt_contract import load_token_manifest
from parts_rl.rlt_contract import token_identity
from parts_rl.rlt_data import PrefixReplay
from parts_rl.rlt_features import FrozenRltFeatures

ROOT = Path(__file__).resolve().parents[3]
ARCH = {
    "input_dim": 2048,
    "embed_dim": 2048,
    "prefix_seq_len": 768,
    "num_layers": 2,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "dropout_rate": 0.0,
}
IDENTITY = {"checkpoint_weights_sha256": "a" * 64, "norm_stats_sha256": "b" * 64}
SPLIT = {"train_groups": ["train"], "holdout_groups": ["held"]}


def write(path, value):
    path.write_text(json.dumps(value))


def token_fixture(root):
    root.mkdir()
    (root / "token.pt").write_bytes(b"inert asset fixture; never load as a model")
    value = {
        "schema": "yam_rlt_token_v1",
        "status": "trained_not_task_evaluated",
        "upstream_commit": UPSTREAM_COMMIT,
        "architecture": ARCH,
        "base_identity": IDENTITY,
        "prefix_schema": PREFIX_SCHEMA,
        "feature_extractor": FEATURE_EXTRACTOR,
        "prefix_ready_sha256": "c" * 64,
        "data_split": SPLIT,
        "weights": {"path": "token.pt", "sha256": wire.sha256(root / "token.pt")},
    }
    value["feature_schema_id"] = token_identity(value)
    write(root / "token.json", value)
    return root / "token.json", value


def test_pinned_native_source_and_causal_reconstruction():
    root = ROOT / "packages/parts-rl/src/parts_rl/vendor/rlinf"
    source = json.loads((root / "SOURCE.json").read_text())
    path = root / "rlt_token_transformer.py"
    assert source["commit"] == UPSTREAM_COMMIT
    assert source["sha256"] == wire.sha256(path)
    assert source["changes"] == [] and "Apache License" in (root / "LICENSE").read_text()
    tree = ast.parse(path.read_text())
    decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLTTokenDecoder")
    code = ast.unparse(decoder)
    assert "target_embeddings.detach()" in code and "frozen_targets[:, :-1]" in code
    assert "torch.triu(" in code and "diagonal=1" in code and "attn_mask=causal_mask" in code


@pytest.mark.parametrize("kind", ["token", "actor"])
def test_recipe_and_host_guards_never_import_torch(tmp_path, kind):
    path = tmp_path / "recipe.json"
    template = "token_recipe" if kind == "token" else "train_recipe"
    value = json.loads((ROOT / f"configs/rlt/{template}.example.json").read_text())
    if kind == "token":
        value["prefix_ready"] = "/not_loaded/PREFIX_READY.json"
    else:
        value.update(
            token_manifest="/not_loaded/token.json", replay=dict.fromkeys(wire.INDICES, "/not_loaded/READY.json")
        )
    write(path, value)
    loader = load_token_recipe if kind == "token" else load_recipe
    assert loader(path)["allowed_training_hosts"] == []
    module = "adapters.rlt.train_token" if kind == "token" else "adapters.parts.train"
    suffix = "" if kind == "token" else ', backend="rlt"'
    snippet = (
        f"from {module} import main; import sys; "
        f"main(['--recipe', {str(path)!r}, '--output', {str(tmp_path / 'out')!r}, '--check-only']{suffix}); "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    snippet = (
        f"from {module} import main; import sys;\ntry:\n"
        f" main(['--recipe', {str(path)!r}, '--output', {str(tmp_path / 'out')!r}, '--execute-on-server']{suffix})\n"
        "except ValueError as e:\n assert 'listed GPU server' in str(e)\n"
        "else:\n raise AssertionError('guard did not reject')\nassert 'torch' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["source", "weights", "schema", "path", "untrained", "holdout"])
def test_token_contract_rejects_incompatible_assets(tmp_path, change):
    path, value = token_fixture(tmp_path / "token")
    assert load_token_manifest(path)[0] == value
    if change == "source":
        value["upstream_commit"] = "0" * 40
    elif change == "weights":
        (path.parent / "token.pt").write_bytes(b"changed")
    elif change == "schema":
        value["feature_schema_id"] = "old-pooled-feature"
    elif change == "path":
        value["weights"]["path"] = "../token/token.pt"
    elif change == "untrained":
        value["status"] = "random_initialization"
    else:
        value["data_split"] = {"train_groups": ["held"], "holdout_groups": ["held"]}
    write(path, value)
    with pytest.raises(ValueError):
        load_token_manifest(path)


def test_actor_replay_token_and_holdout_binding(tmp_path):
    path, value = token_fixture(tmp_path / "token")
    replay = SimpleNamespace(
        metadata={"feature_schema_id": value["feature_schema_id"], "feature_dim": 2048, "holdout_groups": ["held"]}
    )
    assert bind_token({"token_manifest": path}, replay) == value
    replay.metadata["feature_schema_id"] = "pooled-prefix"
    with pytest.raises(ValueError, match="pooled"):
        bind_token({"token_manifest": path}, replay)
    replay.metadata["feature_schema_id"] = value["feature_schema_id"]
    replay.metadata["holdout_groups"] = ["different"]
    with pytest.raises(ValueError, match="holdout"):
        bind_token({"token_manifest": path}, replay)


def test_behavior_bundle_is_portable_and_checks_base_and_hash(tmp_path):
    token_path, token = token_fixture(tmp_path / "original")
    contract = declaration(["off", "shadow", "collect"])
    contract["feature_schema_id"] = token["feature_schema_id"]
    result = build(tmp_path / "behavior", contract, IDENTITY, token_manifest=token_path, exploration_std=0.1, seed=7)
    loaded, path = verify_manifest(result["path"], IDENTITY)
    assert loaded["exploration_space"] == "pre_tanh_gaussian"
    assert path.parent == tmp_path / "behavior/token"
    with pytest.raises(ValueError, match="binding"):
        verify_manifest(result["path"], {**IDENTITY, "norm_stats_sha256": "d" * 64})
    (path.parent / "token.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        verify_manifest(result["path"], IDENTITY)


@pytest.mark.parametrize("problem", [None, "std", "token", "algorithm", "schema"])
def test_learned_behavior_snapshot_binding_without_model_execution(tmp_path, problem):
    token_path, token = token_fixture(tmp_path / "token")
    contract = declaration(["off", "collect", "eval"])
    contract["feature_schema_id"] = token["feature_schema_id"]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    actors = {}
    for arm in wire.INDICES:
        asset = snapshot / f"{arm}_actor.pt"
        asset.write_bytes(b"inert snapshot fixture, no model")
        actors[arm] = {"path": asset.name, "sha256": wire.sha256(asset)}
    trained = {
        "schema": "yam_rlt_training_v1",
        "state_schema": wire.STATE_SCHEMA,
        "contract_sha": contract["contract_sha"],
        "feature_schema_id": token["feature_schema_id"],
        "feature_dim": 2048,
        "actors": actors,
        "token_manifest": {"sha256": wire.sha256(token_path)},
        "recipe_core": {"fixed_std": 0.1, "algorithm": ALGORITHM},
    }
    if problem == "token":
        trained["token_manifest"]["sha256"] = "0" * 64
    elif problem == "algorithm":
        trained["recipe_core"]["algorithm"] = "parts_td3_bc_v1"
    elif problem == "schema":
        trained["schema"] = "yam_parts_training_v1"
    write(snapshot / "training.json", trained)
    kwargs = {
        "snapshot": snapshot,
        "token_manifest": token_path,
        "exploration_std": 0.2 if problem == "std" else 0.1,
        "seed": 0,
    }
    if problem:
        with pytest.raises(ValueError):
            build(tmp_path / "behavior", contract, IDENTITY, **kwargs)
        assert not (tmp_path / "behavior").exists()
    else:
        result = build(tmp_path / "behavior", contract, IDENTITY, **kwargs)
        assert verify_manifest(result["path"], IDENTITY)[0]["actors"]["left"]["sha256"] == actors["left"]["sha256"]


class TensorFixture:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape

    def __getitem__(self, key):
        return TensorFixture(self.value[key])

    def detach(self):
        return self


class ExpertFixture:
    def forward(self, *, inputs_embeds):
        if inputs_embeds[0] is None:
            return [None, "suffix"], "cache"
        return [TensorFixture(np.full((1, 8, 2048), 7, np.float32)), None], "cache"


class PiFixture:
    def __init__(self):
        self.paligemma_with_expert = ExpertFixture()

    def embed_prefix(self, images, masks, tokens, token_masks):
        return TensorFixture(np.ones((1, 8, 2048))), TensorFixture(np.ones((1, 8), bool)), "attention"

    def infer(self):
        prefix, _, _ = self.embed_prefix([1, 2, 3], None, TensorFixture(np.ones((1, 2))), None)
        result = self.paligemma_with_expert.forward(inputs_embeds=[prefix, None])
        self.paligemma_with_expert.forward(inputs_embeds=[None, "suffix"])
        return result


def test_final_prefix_capture_restores_methods_and_ignores_suffix():
    pi = PiFixture()
    seen = []

    def readout(prefix, mask):
        seen.append((prefix.value.copy(), mask.value.copy()))
        return np.array([prefix.value.mean()], np.float32)

    capture = FrozenRltFeatures(pi, readout)
    with capture.capture():
        result = pi.infer()
        assert result[1] == "cache" and capture.z.tolist() == [7]
    assert seen[0][0].shape == (1, 6, 2048)
    assert "embed_prefix" not in pi.__dict__ and "forward" not in pi.paligemma_with_expert.__dict__
    with pytest.raises(RuntimeError), capture.capture():
        assert capture.z is None
        raise RuntimeError("inference failure")
    assert "embed_prefix" not in pi.__dict__


def test_capture_rejects_repeated_or_missing_prefix():
    pi = PiFixture()
    capture = FrozenRltFeatures(pi)
    with pytest.raises(ValueError, match="Ambiguous"), capture.capture():
        pi.infer()
        pi.infer()
    with pytest.raises(ValueError, match="stale"), capture.capture():
        pi.paligemma_with_expert.forward(inputs_embeds=["uncaptured input", None])


def prefix_fixture(root):
    root.mkdir()
    np.savez(root / "prefix.npz", prefix=np.ones((6, 2048), np.float32), mask=np.ones(6, bool))
    value = {
        "schema": "yam_rlt_prefixes_v1",
        "status": "READY",
        "prefix_schema": PREFIX_SCHEMA,
        "base_identity": IDENTITY,
        "mock": False,
        "split_role": "train",
        **SPLIT,
        "observations_manifest_sha256": "c" * 64,
        "members": [
            {
                "path": "prefix.npz",
                "sha256": wire.sha256(root / "prefix.npz"),
                "observation_key": "observation-1",
                "group_id": "train",
            }
        ],
    }
    write(root / "PREFIX_READY.json", value)
    return root / "PREFIX_READY.json", value


def test_prefix_cache_sampling_and_hash_rejection(tmp_path):
    path, _ = prefix_fixture(tmp_path / "cache")
    replay = PrefixReplay(path, prefix_seq_len=768)
    prefix, mask = replay.batch(2, np.random.default_rng(0))
    assert prefix.shape == (2, 6, 2048) and mask.shape == (2, 6) and mask.dtype == np.bool_
    (path.parent / "prefix.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        replay.batch(2, np.random.default_rng(0))


@pytest.mark.parametrize("problem", ["holdout", "mock", "duplicate", "nonfinite", "no_mask", "length"])
def test_prefix_cache_rejects_invalid_training_input(tmp_path, problem):
    path, value = prefix_fixture(tmp_path / "cache")
    if problem == "holdout":
        value["members"][0]["group_id"] = "held"
    elif problem == "mock":
        value["mock"] = True
    elif problem == "duplicate":
        value["members"].append(copy.deepcopy(value["members"][0]))
    elif problem in ("nonfinite", "no_mask"):
        np.savez(
            path.parent / "prefix.npz",
            prefix=np.full((6, 2048), np.nan if problem == "nonfinite" else 1, np.float32),
            mask=np.full(6, problem != "no_mask", bool),
        )
        value["members"][0]["sha256"] = wire.sha256(path.parent / "prefix.npz")
    write(path, value)
    with pytest.raises(ValueError):
        PrefixReplay(path, prefix_seq_len=5 if problem == "length" else 768)


def test_rlt_collect_samples_once_preserves_rtc_and_base():
    class ActorFixture:
        calls = 0

        def __call__(self, state):
            raise AssertionError("collect should not run deterministic actor too")

        def sample(self, state, rng):
            self.calls += 1
            return np.tanh(rng.normal(0, 0.1, (50, 6))).astype(np.float32)

    declaration_value = manifest(["off", "collect"])
    declaration_value["exploration_space"] = "pre_tanh_gaussian"
    actors = {arm: ActorFixture() for arm in wire.INDICES}
    extension = PartsPolicyExtension(declaration_value, actors=actors, features=Features())
    base = Base()
    output = extension.infer(
        {"observation.state": np.zeros(14)},
        {"delay_steps": 3, "observation_policy_tick": 0},
        sent("collect"),
        lambda: (base.infer({}), True, [], None),
    )[0]
    assert base.calls == 1 and output["actions"] is base.actions
    for arm, actor in actors.items():
        candidate = output["parts"]["candidates"][arm]
        assert actor.calls == 1 and np.all(candidate["u"][:3] == 0)
        assert np.any(candidate["u"][3:] != 0) and np.all(np.abs(candidate["u"]) <= 1)
        assert candidate["editable_mask"].tolist() == [False] * 3 + [True] * 47
    duplicate = PartsPolicyExtension(declaration_value, actors=actors, features=Features())
    again = duplicate.infer(
        {"observation.state": np.zeros(14)},
        {"delay_steps": 3, "observation_policy_tick": 0},
        sent("collect"),
        lambda: (base.infer({}), True, [], None),
    )[0]
    np.testing.assert_array_equal(output["parts"]["candidates"]["left"]["u"], again["parts"]["candidates"]["left"]["u"])


def test_raw_video_observation_export_uses_existing_publication(tmp_path, monkeypatch):
    from scripts.rlt.prepare_observations import prepare
    from test_replay import SPLIT as RAW_SPLIT
    from test_replay import package

    root = tmp_path / "raw"
    package(root)
    monkeypatch.setattr(
        "scripts.rlt.prepare_observations.video_frame", lambda path, index: np.full((4, 4, 3), index, np.uint8)
    )
    result = prepare([root], tmp_path / "observations", RAW_SPLIT, prompt="Sort the blocks")
    value = json.loads(Path(result["path"]).read_text())
    assert result["observations"] == 2 and value["mock"] is False
    with np.load(Path(result["path"]).parent / value["observations"][1]["path"]) as sample:
        assert sample["observation.state"].shape == (14,)
        assert sample["observation.images.top_rgb"][0, 0].tolist() == [2, 2, 2]
        assert str(sample["prompt"]) == "Sort the blocks"


def test_video_decoder_uses_frame_index(tmp_path):
    av = pytest.importorskip("av")
    from scripts.rlt.prepare_observations import video_frame

    path = tmp_path / "frames.mkv"
    with av.open(str(path), mode="w") as file:
        stream = file.add_stream("ffv1", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "bgr0"
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 50, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                file.mux(packet)
        for packet in stream.encode():
            file.mux(packet)
    assert video_frame(path, 2).shape == (16, 16, 3)
    assert np.all(video_frame(path, 1) == 50)
    with pytest.raises(ValueError, match="Missing video frame"):
        video_frame(path, 3)
    assert "torch" not in sys.modules
