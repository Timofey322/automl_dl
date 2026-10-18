from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MultiTimeframeDataset(Dataset[tuple[dict[str, Tensor], Tensor]]):
    """Leakage-safe windows whose rows were indexed by their availability time."""

    def __init__(
        self,
        *,
        frames: Mapping[str, pd.DataFrame],
        targets: pd.DataFrame,
        context_lengths: Mapping[str, int],
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
        normalize_windows: bool = True,
    ) -> None:
        if set(frames) != set(context_lengths):
            raise ValueError("frames and context_lengths must have identical timeframes")
        if not targets.index.is_monotonic_increasing:
            raise ValueError("targets index must be sorted")

        self.frames = {key: value.sort_index() for key, value in frames.items()}
        self.targets = targets
        self.context_lengths = dict(context_lengths)
        self.feature_columns = tuple(feature_columns)
        self.target_columns = tuple(target_columns)
        self.normalize_windows = normalize_windows

        for timeframe, frame in self.frames.items():
            missing = set(self.feature_columns).difference(frame.columns)
            if missing:
                raise ValueError(f"{timeframe} is missing features: {sorted(missing)}")
        missing_targets = set(self.target_columns).difference(targets.columns)
        if missing_targets:
            raise ValueError(f"Missing targets: {sorted(missing_targets)}")

        self.decision_times = [
            decision_time
            for decision_time in targets.dropna(subset=list(self.target_columns)).index
            if all(
                frame.index.searchsorted(decision_time, side="right")
                >= self.context_lengths[timeframe]
                for timeframe, frame in self.frames.items()
            )
        ]

    def __len__(self) -> int:
        return len(self.decision_times)

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor]:
        decision_time = self.decision_times[index]
        windows: dict[str, Tensor] = {}
        for timeframe, frame in self.frames.items():
            end = int(frame.index.searchsorted(decision_time, side="right"))
            start = end - self.context_lengths[timeframe]
            values = torch.as_tensor(
                frame.iloc[start:end][list(self.feature_columns)].to_numpy(),
                dtype=torch.float32,
            )
            if self.normalize_windows:
                mean = values.mean(dim=0, keepdim=True)
                scale = values.std(dim=0, keepdim=True).clamp_min(1e-6)
                values = (values - mean) / scale
            windows[timeframe] = values

        target = torch.as_tensor(
            self.targets.loc[decision_time, list(self.target_columns)].to_numpy(dtype=float),
            dtype=torch.float32,
        )
        return windows, target
