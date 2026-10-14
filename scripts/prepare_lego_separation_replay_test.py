"""Replay selection tests use synthetic inventories, without training imports."""

from collections import Counter

from scripts.prepare_lego_separation_replay import balanced_replay


def test_replay_is_deterministic_unique_stratified_and_duration_balanced():
    metadata = {i: {"length": 1000 + i % 10 * 10} for i in range(200)}
    selected = balanced_replay(set(metadata), metadata, 103000, 100, 42)
    assert selected == balanced_replay(set(metadata), metadata, 103000, 100, 42)
    assert len(selected) == len(set(selected)) == 100
    assert Counter(i // 20 for i in selected) == dict.fromkeys(range(10), 10)
    assert abs(sum(metadata[i]["length"] for i in selected) - 103000) <= 10


def test_duration_matching_does_not_leave_base_pool():
    base = set(range(1000, 1200))
    metadata = {i: {"length": i} for i in base}
    selected = balanced_replay(base, metadata, 110000, 100, 42)
    assert set(selected) <= base
    assert abs(sum(metadata[i]["length"] for i in selected) - 110000) <= 1
