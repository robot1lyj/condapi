import json

import pytest
from select_lego_rtc_episodes import select_episodes
from yam_episode_exclusions import find_excluded_episodes
from yam_episode_exclusions import require_no_excluded_episodes


def registry(tmp_path, **changes):
    item = {
        "source_repo": "lerobot/test",
        "source_revision": "revision-a",
        "split": "train",
        "source_episode_index": 19,
        "identity_verified": True,
        "flags": ["video_frozen_run_corrected"],
    }
    item.update(changes)
    path = tmp_path / "exclusions.json"
    path.write_text(json.dumps({"schema_version": 1, "episodes": [item]}))
    return path


def source_row():
    return {
        "episode_index": 7,
        "source_repo": "lerobot/test",
        "source_revision": "revision-a",
        "source_episode_index": 19,
        "length": 300,
    }


def test_exact_versioned_source_identity_and_corrected_flags(tmp_path):
    row = source_row()
    path = registry(tmp_path)
    assert find_excluded_episodes([row], registry=path)[0]["episode_index"] == 7
    for field, other in (("source_repo", "other/test"), ("source_revision", "revision-b"), ("source_episode_index", 7)):
        assert not find_excluded_episodes([{**row, field: other}], registry=path)
    assert not find_excluded_episodes([row], split="val", registry=path)
    assert not find_excluded_episodes([{"episode_index": 19}], registry=path)


@pytest.mark.parametrize(
    "changes",
    [
        {"identity_verified": False},
        {"identity_verified": 1},
        {"flags": []},
        {"flags": "video_frozen_run_corrected"},
        {"source_episode_index": True},
    ],
)
def test_reject_unverified_or_incomplete_registry(tmp_path, changes):
    with pytest.raises(ValueError, match="verified source identity"):
        find_excluded_episodes([source_row()], registry=registry(tmp_path, **changes))


def test_preflight_checks_only_selected_converted_ids_and_preserves_source(tmp_path):
    path = registry(tmp_path)
    repo = tmp_path / "train"
    repo.mkdir()
    manifest = repo / "conversion_manifest.json"
    raw = json.dumps(
        {"split": "train", "episodes": [source_row(), {**source_row(), "episode_index": 8, "source_episode_index": 20}]}
    )
    manifest.write_text(raw)
    require_no_excluded_episodes(repo, [8], registry=path)
    for selection in (None, [7], [7, 8]):
        with pytest.raises(ValueError, match=r"QC-excluded episodes selected: \[7\]"):
            require_no_excluded_episodes(repo, selection, registry=path)
    assert manifest.read_text() == raw


def test_default_registry_filters_known_bad_origin_without_rewriting_ids():
    row = {
        "episode_index": 3490,
        "source_repo": "lerobot/abc_130k_v3_train",
        "source_revision": "68651e4929d9fb00f798937b2d62617cab5c771d",
        "source_episode_index": 101616,
        "length": 90000,
    }
    assert find_excluded_episodes([row])[0]["episode_index"] == 3490
    healthy = {**row, "episode_index": 3491, "source_episode_index": 101617}
    assert select_episodes([row, healthy], fps=30, hours=0.5, seed=42) == [healthy]
