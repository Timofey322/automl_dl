from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    n_splits: int = 5
    test_size: int = 1_000
    label_horizon: int = 24
    embargo: int = 0
    min_train_size: int = 5_000

    def split(self, n_samples: int) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        if self.n_splits <= 0 or self.test_size <= 0:
            raise ValueError("n_splits and test_size must be positive")
        first_validation = n_samples - self.n_splits * self.test_size
        if first_validation - self.label_horizon - self.embargo < self.min_train_size:
            raise ValueError("Not enough observations for the requested walk-forward split")

        gap = self.label_horizon + self.embargo
        for split_index in range(self.n_splits):
            validation_start = first_validation + split_index * self.test_size
            validation_end = validation_start + self.test_size
            train_end = validation_start - gap
            yield (
                np.arange(0, train_end, dtype=int),
                np.arange(validation_start, validation_end, dtype=int),
            )
