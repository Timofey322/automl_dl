from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score, log_loss, mean_pinball_loss


def forecast_metrics(
    *,
    direction_true: NDArray[np.int_],
    direction_predicted: NDArray[np.int_],
    direction_probabilities: NDArray[np.float64],
    return_true: NDArray[np.float64],
    return_quantiles: NDArray[np.float64],
    quantiles: Sequence[float],
) -> dict[str, float]:
    if return_quantiles.shape[1] != len(quantiles):
        raise ValueError("Return prediction columns must match quantiles")

    result = {
        "balanced_accuracy": float(balanced_accuracy_score(direction_true, direction_predicted)),
        "log_loss": float(
            log_loss(
                direction_true,
                direction_probabilities,
                labels=[-1, 0, 1],
            )
        ),
        "quantile_crossing_rate": float(
            np.mean(np.any(np.diff(return_quantiles, axis=1) < 0.0, axis=1))
        ),
    }
    for index, quantile in enumerate(quantiles):
        result[f"pinball_q{quantile:g}"] = float(
            mean_pinball_loss(
                return_true,
                return_quantiles[:, index],
                alpha=quantile,
            )
        )
    return result
