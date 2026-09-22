"""Small real data/config tests only: no model construction, optimizer, or training loop."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/openwam"))
from vla_platform.project import Project  # noqa: E402

from adapters.openwam.common import CAMERAS  # noqa: E402
from adapters.openwam.common import validate_data_config  # noqa: E402
from adapters.openwam.common import verify_source  # noqa: E402
from adapters.openwam.metrics import metrics_hook  # noqa: E402


@pytest.fixture
def data_modules():
    pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    pytest.importorskip("torch")
    from adapters.openwam import data  # noqa: PLC0415

    return np, data


@pytest.fixture
def dataset(tmp_path, data_modules):
    np, data = data_modules
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"shape": [14]},
            "observation.state": {"shape": [14]},
            **{key: {"dtype": "video"} for key in CAMERAS},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 9}) + "\n")
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "sort lego"}) + "\n")
    for episode in (0, 1):
        # Episode 1 is a held-out outlier; it must not leak into train statistics.
        actions = np.arange(9 * 14, dtype=np.float32).reshape(9, 14) + episode * 10000
        values = {
            "episode_index": [episode] * 9,
            "frame_index": list(range(9)),
            "timestamp": (np.arange(9) / 30).tolist(),
            "task_index": [0] * 9,
            "action": actions.tolist(),
            "observation.state": (actions + 100).tolist(),
        }
        pq.write_table(pa.table(values), root / f"data/episode_{episode:06d}.parquet")
    stats_path = tmp_path / "stats.json"
    data.prepare_stats(root, [0], stats_path)
    config = json.loads((ROOT / "configs/native/openwam-yam.example.json").read_text())["dataloader"]
    config.update(
        dataset_dir=str(root),
        episodes=[0],
        normalization_json=str(stats_path),
        normalization_stats_path=str(tmp_path / "normalization_stats.npy"),
        num_frames=9,
        video_stride=2,
        height=96,
        width=64,
        fps=30,
    )
    return root, config


def test_source_snapshot_and_platform_planning():
    assert verify_source() == "7c5861e45cfe1339a0323f0e0b03a3316c37971c"
    project = Project(ROOT)
    plan = project.plan(ROOT / "configs/experiments/openwam-yam.toml", "train", "openwam-contract-test")
    assert plan.implementation == "openwam"
    assert plan.target == "server"
    assert "third_party/openwam/UPSTREAM.json" in " ".join(plan.source_hashes)
    assert not Path(plan.output).exists()
    plan = project.plan(ROOT / "configs/experiments/openwam-reference.toml", "infer", "openwam-infer-test")
    assert plan.operation == "infer"


@pytest.mark.parametrize(
    "patch",
    [
        {"unify_action": True, "unify_action_map": [0] * 14},
        {"camera_layout": CAMERAS[::-1]},
        {"action_semantics": "delta"},
        {"episodes": [1, 1]},
        {"num_frames": 10},
        {"binary_action_dims": [6, 13]},
    ],
)
def test_rejects_ambiguous_contract(patch):
    recipe = json.loads((ROOT / "configs/native/openwam-yam.example.json").read_text())
    recipe["dataloader"].update(patch)
    with pytest.raises(ValueError, match=r"YAM|window|episode|absolute"):
        validate_data_config(recipe["dataloader"], recipe["model"]["architecture"])


def test_real_windows_masks_stats_and_native_inverse(dataset, data_modules, monkeypatch):
    np, data = data_modules
    root, config = dataset
    from PIL import Image  # noqa: PLC0415

    seen = []

    def decode(path, indices, height, width):
        seen.append(indices)
        camera = next(i for i, name in enumerate(CAMERAS) if name in path)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        return [Image.new("RGB", (width, height), colors[camera]) for _ in indices]

    monkeypatch.setattr(data, "decode_video_frames", decode)
    ds = data.YamDataset(config)
    assert len(ds) == 8
    first = ds[0]
    assert first["action"].shape == (8, 14)
    assert first["proprio"].shape == (1, 14)
    assert first["prompt"] == "sort lego"
    assert first["video"][0].getpixel((0, 0)) == (255, 0, 0)
    assert first["video"][0].getpixel((0, 95)) == (0, 255, 0)
    assert first["video"][0].getpixel((63, 95)) == (0, 0, 255)
    raw = data.read_episode(root, ds.info, 0)
    np.testing.assert_allclose(ds.action_norm.unnormalize(first["action"].numpy()), raw["action"][:8], atol=1e-5)
    # Independent state statistics; state +100 must still map to the same normalized first vector.
    np.testing.assert_allclose(first["proprio"].numpy(), first["action"][:1].numpy(), atol=1e-5)
    last = ds[7]
    assert last["action_mask"].sum().item() == 2 * 14
    assert last["video_mask"].tolist() == [True, False, False, False, False]
    assert seen[-1] == [7, 8, 8, 8, 8]
    assert last["action"][2:].count_nonzero() == 0
    with pytest.raises(IndexError):
        ds[len(ds)]
    assert max(json.loads(Path(config["normalization_json"]).read_text())["stats"]["joint"]["max"]) < 10000


def test_explicit_80d_mapping_and_masks(dataset, data_modules, monkeypatch):
    np, data = data_modules
    _, config = dataset
    from PIL import Image  # noqa: PLC0415

    monkeypatch.setattr(data, "decode_video_frames", lambda path, ids, h, w: [Image.new("RGB", (w, h)) for _ in ids])
    slots = list(range(14, 28))
    config.update(unify_action=True, unify_action_map=slots)
    validate_data_config(config, {"action_dim": 80, "state_dim": 80, "use_proprioception": True})
    ds = data.YamDataset(config)
    sample = ds[7]
    assert sample["action"].shape == (8, 80)
    assert sample["action_mask"].sum().item() == 28
    assert not sample["action_mask"][:, :14].any()
    assert sample["proprio_mask"].sum().item() == 14
    np.testing.assert_allclose(
        sample["action"][:2, slots].numpy(),
        ds.action_norm.normalize(data.read_episode(ds.root, ds.info, 0)["action"][7:]),
    )


def test_stale_stats_and_source_protection(dataset, data_modules):
    _, data = data_modules
    root, config = dataset
    with pytest.raises(ValueError, match="outside"):
        data.prepare_stats(root, [0], root / "new.json")
    with pytest.raises(ValueError, match="new path"):
        data.prepare_stats(root, [0], Path(config["normalization_json"]))
    path = root / "data/episode_000000.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        data.YamDataset(config)


def test_composition_and_checkpoint_head_guard(dataset, tmp_path):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf  # noqa: PLC0415

    from adapters.openwam.train import compose_config  # noqa: PLC0415

    _, dl = dataset
    recipe = json.loads((ROOT / "configs/native/openwam-yam.example.json").read_text())
    recipe["dataloader"] = dl
    recipe["training"]["finetune_ckpt_path"] = None
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    cfg, workers = compose_config(path, tmp_path / "out")
    assert workers == 1
    assert cfg.dataloader.fps == 30
    assert cfg.training.save_full_states_for_resume
    assert cfg.model.architecture.action_dim == 14
    source = tmp_path / "checkpoint"
    source.mkdir()
    OmegaConf.save(cfg, source / "config.yaml")
    (source / "checkpoint_step_10.safetensors").touch()  # no weights loaded in these tests
    recipe["training"]["finetune_ckpt_path"] = str(source)
    path.write_text(json.dumps(recipe))
    compose_config(path, tmp_path / "new")
    recipe["model"]["architecture"].update(action_dim=80, state_dim=80)
    recipe["dataloader"].update(unify_action=True, unify_action_map=list(range(14)))
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="action_dim mismatch"):
        compose_config(path, tmp_path / "new")
    recipe["training"]["resume_ckpt_path"] = str(source)
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="mutually exclusive"):
        compose_config(path, tmp_path / "new")


def test_resume_requires_same_contract_and_full_state(dataset, tmp_path):
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf  # noqa: PLC0415

    from adapters.openwam.train import compose_config  # noqa: PLC0415

    _, dl = dataset
    recipe = json.loads((ROOT / "configs/native/openwam-yam.example.json").read_text())
    recipe["dataloader"] = dl
    recipe["training"]["finetune_ckpt_path"] = None
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    cfg, _ = compose_config(path, tmp_path / "old")
    source = tmp_path / "source"
    source.mkdir()
    OmegaConf.save(cfg, source / "config.yaml")
    (source / "checkpoint_step_10.safetensors").touch()
    resume = {
        "schema_version": 1,
        "nproc_per_node": 1,
        "training": {"finetune_ckpt_path": None, "resume_ckpt_path": str(source)},
    }
    path.write_text(json.dumps(resume))
    with pytest.raises(ValueError, match="full-state"):
        compose_config(path, tmp_path / "new")
    (source / "accel_state_step_10").mkdir()
    compose_config(path, tmp_path / "new")
    changed = copy.deepcopy(resume)
    changed["training"]["learning_rate"] = 0.5
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Resume changed training"):
        compose_config(path, tmp_path / "new")


def test_metrics_rank_and_microstep_semantics(tmp_path):
    calls = []
    hook = metrics_hook(lambda self, **kw: calls.append(kw))
    values = {
        "metrics": {"loss_total": 2, "loss_action": 1, "loss_video": 1, "grad_norm": float("nan")},
        "opt_step": 3,
        "global_step": 12,
        "lr": 0.0001,
        "epoch": 0,
        "steps_per_sec": 2,
        "output_path": str(tmp_path),
    }
    hook(SimpleNamespace(accelerator=SimpleNamespace(is_main_process=False)), **values)
    assert not (tmp_path / "metrics.jsonl").exists()
    hook(SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True)), **values)
    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert record["step"] == 12
    assert record["metrics"]["optimizer_step"] == 3
    assert record["metrics"]["nonfinite_metric_count"] == 1
    assert len(calls) == 2


def test_actual_mp4_decode_and_camera_composition(dataset, data_modules):
    av = pytest.importorskip("av")
    np, data = data_modules
    root, config = dataset
    for camera_index, camera in enumerate(CAMERAS):
        path = root / f"videos/{camera}/episode_000000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 96, "yuv420p"
            for index in range(9):
                array = np.zeros((96, 64, 3), dtype=np.uint8)
                array[..., camera_index] = 80 + index * 10
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    ds = data.YamDataset(config)
    sample = ds[7]
    assert len(sample["video"]) == 5
    assert sample["video"][0].getpixel((10, 10))[0] > 140
    assert sample["video"][0].getpixel((10, 80))[1] > 140
    assert sample["video"][0].getpixel((50, 80))[2] > 140
    assert sample["video"][1].tobytes() == sample["video"][-1].tobytes()
    # Missing wrists must fail, never silently render a black tile or skip to another episode.
    (root / f"videos/{CAMERAS[1]}/episode_000000.mp4").unlink()
    with pytest.raises((OSError, RuntimeError)):
        ds[0]


@pytest.mark.parametrize("unified", [False, True])
def test_native_checkpoint_normalizer_roundtrip(dataset, data_modules, tmp_path, monkeypatch, unified):
    """Use native checkpoint code, stubbing only its unused model type import (no large model deps)."""
    import importlib.util  # noqa: PLC0415

    from omegaconf import OmegaConf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    np, data = data_modules
    _, config = dataset
    if unified:
        config.update(unify_action=True, unify_action_map=list(range(20, 34)))
    monkeypatch.setattr(data, "decode_video_frames", lambda path, ids, h, w: [Image.new("RGB", (w, h)) for _ in ids])
    ds = data.YamDataset(config)
    monkeypatch.setitem(sys.modules, "openwam.model.architectures.base", SimpleNamespace(BaseWAMArchitecture=object))
    spec = importlib.util.spec_from_file_location(
        "test_native_checkpoint_loader", ROOT / "third_party/openwam/openwam/deploy/model_loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    np.save(tmp_path / "normalization_stats.npy", data.load_stats(config["normalization_json"])["stats"])
    normalizer = module._build_normalizer(OmegaConf.create({"dataloader": config}), str(tmp_path))  # noqa: SLF001
    sample = ds[0]
    raw = data.read_episode(ds.root, ds.info, 0)
    np.testing.assert_allclose(normalizer.unnormalize(sample["action"].numpy()), raw["action"][:8], atol=1e-5)
    np.testing.assert_allclose(normalizer.normalize(raw["observation.state"][:1]), sample["proprio"].numpy(), atol=1e-5)


def test_training_entry_rejects_unallocated_local_execution(tmp_path, monkeypatch):
    from adapters.openwam.train import main  # noqa: PLC0415

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit, match="2"):
        main(["--config", str(tmp_path / "missing.json"), "--output", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="2"):
        main(["--config", str(tmp_path / "missing.json"), "--worker", "--check-only"])
    assert not (tmp_path / "out").exists()
