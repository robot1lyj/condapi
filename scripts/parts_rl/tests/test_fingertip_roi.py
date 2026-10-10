"""Image geometry and counterexamples, without model imports or training."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("fingertip_roi", Path(__file__).parents[1] / "analyze_fingertip_roi.py")
roi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roi)


def test_blue_background_is_a_counterexample_to_grasp_inference():
    rgb = np.full((100, 120, 3), [10, 35, 170], dtype=np.uint8)
    bridge, _, _ = roi.bridge_mask(rgb.shape[:2], [30, 50], [90, 50])
    # An entirely empty gap against a blue bin exceeds 80% with both definitions.
    assert roi.color_pixels(rgb)[bridge].mean() == 1
    assert roi.color_pixels(rgb, shadow_tolerant=True)[bridge].mean() == 1


def test_white_object_cannot_be_distinguished_from_white_table_by_color():
    rgb = np.full((30, 40, 3), 230, dtype=np.uint8)
    assert not roi.color_pixels(rgb).any()
    assert not roi.color_pixels(rgb, shadow_tolerant=True).any()


def test_tip_uses_seed_component_instead_of_disconnected_upper_noise():
    mask = np.zeros((100, 100), bool)
    mask[30:100, 10:40] = True
    mask[:10, 60:100] = True
    tip, component = roi.finger_tip(mask, [20, 90])
    assert 30 <= tip[1] <= 31
    assert not component[:10].any()


def test_tiny_gap_is_unknown_instead_of_small_denominator_success():
    with pytest.raises(ValueError, match="tiny"):
        roi.bridge_mask((100, 100), [50, 50], [53, 50])
