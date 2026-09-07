import dataclasses
from types import SimpleNamespace

import pytest

from openpi.policies import policy_config


@pytest.mark.parametrize(("precision", "compile_model"), [("float32", False), ("bfloat16", True)])
def test_policy_selects_precision_before_loading(tmp_path, monkeypatch, precision, compile_model):
    calls = {}

    @dataclasses.dataclass(frozen=True)
    class Model:
        dtype: str = "bfloat16"

        def load_pytorch(self, train_config, weight_path, *, compile_model):
            calls["load_dtype"] = train_config.model.dtype
            calls["compile"] = compile_model
            backbone = SimpleNamespace(to_bfloat16_for_selected_params=lambda dtype: calls.update(cast=dtype))
            return SimpleNamespace(paligemma_with_expert=backbone)

    data = SimpleNamespace(
        create=lambda *args: SimpleNamespace(
            data_transforms=SimpleNamespace(inputs=[], outputs=[]),
            model_transforms=SimpleNamespace(inputs=[], outputs=[]),
            use_quantile_norm=True,
        )
    )

    @dataclasses.dataclass(frozen=True)
    class Train:
        model: Model
        data: object
        assets_dirs: tuple = ()
        policy_metadata: object = None

    (tmp_path / "model.safetensors").touch()
    monkeypatch.setattr(policy_config._policy, "Policy", lambda *args, **kwargs: kwargs)  # noqa: SLF001
    original = Train(Model(), data)
    policy_config.create_trained_policy(
        original,
        tmp_path,
        norm_stats={},
        pytorch_device="cpu",
        pytorch_precision=precision,
        pytorch_compile=compile_model,
    )
    assert calls == {"load_dtype": precision, "compile": compile_model, "cast": precision}
    assert original.model.dtype == "bfloat16"
