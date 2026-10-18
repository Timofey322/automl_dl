from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from automl_dl.evaluation import forecast_metrics
from automl_dl.models import GradientBoostingBaseline
from automl_dl.validation import PurgedWalkForwardSplit


@dataclass(frozen=True)
class BaselineResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]


def run_baseline_walk_forward(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    return_column: str,
    splitter: PurgedWalkForwardSplit,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> BaselineResult:
    features = table[feature_columns].to_numpy(dtype=float)
    direction = table["label"].to_numpy(dtype=int)
    future_return = table[return_column].to_numpy(dtype=float)
    predictions: list[pd.DataFrame] = []

    for fold, (train_indices, validation_indices) in enumerate(splitter.split(len(table))):
        model = GradientBoostingBaseline(quantiles=quantiles)
        model.fit(
            features[train_indices],
            direction[train_indices],
            future_return[train_indices],
        )
        probabilities = model.predict_probabilities(features[validation_indices])
        return_quantiles = model.predict_quantiles(features[validation_indices])
        fold_predictions = pd.DataFrame(
            {
                "actual_direction": direction[validation_indices],
                "predicted_direction": model.predict_direction(features[validation_indices]),
                "probability_short": probabilities[:, 0],
                "probability_flat": probabilities[:, 1],
                "probability_long": probabilities[:, 2],
                "actual_return": future_return[validation_indices],
                **{
                    f"return_q{quantile:g}": return_quantiles[:, quantile_index]
                    for quantile_index, quantile in enumerate(quantiles)
                },
                "fold": fold,
            },
            index=table.index[validation_indices],
        )
        predictions.append(fold_predictions)

    combined = pd.concat(predictions).sort_index()
    metrics = forecast_metrics(
        direction_true=combined["actual_direction"].to_numpy(dtype=int),
        direction_predicted=combined["predicted_direction"].to_numpy(dtype=int),
        direction_probabilities=combined[
            ["probability_short", "probability_flat", "probability_long"]
        ].to_numpy(dtype=float),
        return_true=combined["actual_return"].to_numpy(dtype=float),
        return_quantiles=np.column_stack(
            [combined[f"return_q{quantile:g}"] for quantile in quantiles]
        ),
        quantiles=quantiles,
    )
    return BaselineResult(combined, metrics)
