from __future__ import annotations

import pandas as pd

from automl_dl.labels import LABEL_FLAT, LABEL_LONG, triple_barrier_labels


def test_triple_barrier_detects_first_upper_hit() -> None:
    frame = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "high": [100.5, 101.5, 103.0, 103.5],
            "low": [99.5, 100.5, 101.0, 102.5],
            "atr": [1.0, 1.0, 1.0, 1.0],
        }
    )

    labels = triple_barrier_labels(
        frame,
        horizon=2,
        take_profit_atr=2.0,
        stop_loss_atr=1.0,
    )

    assert labels.loc[0, "label"] == LABEL_LONG
    assert labels.loc[0, "exit_offset"] == 2


def test_triple_barrier_is_flat_when_both_hit_same_candle() -> None:
    frame = pd.DataFrame(
        {
            "close": [100.0, 100.0],
            "high": [100.0, 103.0],
            "low": [100.0, 98.0],
            "atr": [1.0, 1.0],
        }
    )

    labels = triple_barrier_labels(frame, horizon=1)

    assert labels.loc[0, "label"] == LABEL_FLAT
