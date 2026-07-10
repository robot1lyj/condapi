import numpy as np

from scripts import evaluate_openarm_checkpoint_sweep as sweep


def test_prompt_override_for_awbc_evaluation() -> None:
    item = {
        "observation.state": np.zeros(16, dtype=np.float32),
        "task_index": np.asarray(0, dtype=np.int64),
    }

    default = sweep._build_observation(item, {0: "Fold the T-shirt properly"}, None)  # noqa: SLF001
    positive = sweep._build_observation(  # noqa: SLF001
        item,
        {0: "Fold the T-shirt properly"},
        "Fold the T-shirt properly, Advantage: positive",
    )

    assert default["prompt"] == "Fold the T-shirt properly"
    assert positive["prompt"] == "Fold the T-shirt properly, Advantage: positive"
