from collections.abc import Callable
from pathlib import Path
import random

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class AdvantageLeRobotDataset(LeRobotDataset):
    """LeRobot dataset wrapper for Stage Advantage progress supervision.

    Each sample is paired with a random frame from the same episode. The label is
    current stage progress minus the sampled frame's stage progress.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
    ):
        super().__init__(
            repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )
        self._ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(episodes or [])}

    def _episode_array_index(self, episode_index: int) -> int:
        if self.episodes is not None:
            return self._ep_idx_to_arr_idx[episode_index]
        return episode_index

    def _get_sample_with_images(self, index: int) -> dict:
        item = self.hf_dataset[index]
        episode_index = int(item["episode_index"].item())

        query_indices = None
        if self.delta_indices is not None:
            query_indices, _ = self._get_query_indices(index, self._episode_array_index(episode_index))

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            item = {**self._query_videos(query_timestamps, episode_index), **item}

        if self.image_transforms is not None:
            for camera in self.meta.camera_keys:
                item[camera] = self.image_transforms(item[camera])

        return item

    def _add_action_sequence(self, item: dict, index: int, episode_index: int) -> dict:
        if self.delta_indices is None:
            return item

        query_indices, padding = self._get_query_indices(index, self._episode_array_index(episode_index))
        query_result = self._query_hf_dataset(query_indices)
        return {**item, **padding, **query_result}

    def _sample_same_episode_item(self, index: int, episode_index: int) -> dict:
        arr_idx = self._episode_array_index(episode_index)
        episode_start = int(self.episode_data_index["from"][arr_idx].item())
        episode_end = int(self.episode_data_index["to"][arr_idx].item())

        if episode_end - episode_start <= 1:
            raise ValueError(f"Episode {episode_index} has fewer than two frames; cannot build progress pair.")

        while True:
            random_index = random.randint(episode_start, episode_end - 1)
            if random_index != index:
                break
        random_item = self._get_sample_with_images(random_index)

        if int(random_item["episode_index"].item()) != episode_index:
            raise ValueError(
                f"Sampled frame from a different episode: expected {episode_index}, "
                f"got {int(random_item['episode_index'].item())}."
            )
        return random_item

    def __getitem__(self, index: int) -> dict:
        item = self._get_sample_with_images(index)
        episode_index = int(item["episode_index"].item())
        arr_idx = self._episode_array_index(episode_index)

        task_idx = int(item["task_index"].item())
        episode_start = int(self.episode_data_index["from"][arr_idx].item())
        episode_end = int(self.episode_data_index["to"][arr_idx].item())
        episode_length = episode_end - episode_start

        item = self._add_action_sequence(item, index, episode_index)
        item["episode_length"] = episode_length
        item["task"] = self.meta.tasks[task_idx]

        random_item = self._sample_same_episode_item(index, episode_index)
        item.update({f"his_-100_{key}": value for key, value in random_item.items()})

        current_progress = float(item["stage_progress_gt"].item())
        history_progress = float(item["his_-100_stage_progress_gt"].item())
        item["progress"] = current_progress - history_progress
        return item
