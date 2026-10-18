from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from automl_dl.data import mark_candle_availability
from automl_dl.features import add_market_features
from automl_dl.labels import forward_return_targets, triple_barrier_labels

DEFAULT_FEATURES = (
    "log_return",
    "range_pct",
    "body_pct",
    "log_volume",
    "volume_change",
    "volatility",
    "atr_pct",
)


def build_baseline_table(
    frames: Mapping[str, pd.DataFrame],
    *,
    decision_timeframe: str,
    horizon: int,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    volatility_window: int = 24,
) -> pd.DataFrame:
    """Align latest fully observable features and targets at each decision time."""
    if decision_timeframe not in frames:
        raise ValueError(f"Missing decision timeframe: {decision_timeframe}")

    available_features: dict[str, pd.DataFrame] = {}
    decision_featured: pd.DataFrame | None = None
    for timeframe, frame in frames.items():
        featured = add_market_features(frame, volatility_window=volatility_window)
        missing = set(feature_columns).difference(featured.columns)
        if missing:
            raise ValueError(f"{timeframe} is missing features: {sorted(missing)}")
        available_features[timeframe] = mark_candle_availability(
            featured[list(feature_columns)],
            timeframe,
        ).add_prefix(f"{timeframe}_")
        if timeframe == decision_timeframe:
            decision_featured = featured

    if decision_featured is None:
        raise RuntimeError("Decision timeframe was not processed")

    decision_available = available_features[decision_timeframe]
    table = pd.DataFrame(index=decision_available.index)
    for featured in available_features.values():
        table = pd.merge_asof(
            table.sort_index(),
            featured.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
            allow_exact_matches=True,
        )

    labels = triple_barrier_labels(decision_featured, horizon=horizon)
    returns = forward_return_targets(decision_featured, (horizon,))
    targets = mark_candle_availability(
        pd.concat([labels, returns], axis=1),
        decision_timeframe,
    )
    return table.join(targets, how="inner").dropna()
