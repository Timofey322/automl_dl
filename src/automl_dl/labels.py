from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_SHORT = -1
LABEL_FLAT = 0
LABEL_LONG = 1


def triple_barrier_labels(
    frame: pd.DataFrame,
    *,
    horizon: int,
    take_profit_atr: float = 2.0,
    stop_loss_atr: float = 1.0,
    atr_column: str = "atr",
) -> pd.DataFrame:
    """Label which volatility-scaled price barrier is reached first.

    If both barriers are touched within the same OHLC candle, the observation is
    labelled flat because intrabar ordering is unknown.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required = {"close", "high", "low", atr_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for triple barrier labels: {sorted(missing)}")

    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    atr = frame[atr_column].to_numpy(dtype=float)
    size = len(frame)

    labels = np.full(size, np.nan)
    exit_offsets = np.full(size, np.nan)
    mfe = np.full(size, np.nan)
    mae = np.full(size, np.nan)

    for index in range(size - horizon):
        if not np.isfinite(atr[index]) or atr[index] <= 0:
            continue
        upper = close[index] + take_profit_atr * atr[index]
        lower = close[index] - stop_loss_atr * atr[index]
        future_high = high[index + 1 : index + horizon + 1]
        future_low = low[index + 1 : index + horizon + 1]
        mfe[index] = np.max(future_high / close[index] - 1.0)
        mae[index] = np.min(future_low / close[index] - 1.0)

        label = LABEL_FLAT
        offset = horizon
        for step, (bar_high, bar_low) in enumerate(zip(future_high, future_low, strict=True), 1):
            hit_upper = bar_high >= upper
            hit_lower = bar_low <= lower
            if hit_upper and hit_lower:
                label = LABEL_FLAT
                offset = step
                break
            if hit_upper:
                label = LABEL_LONG
                offset = step
                break
            if hit_lower:
                label = LABEL_SHORT
                offset = step
                break

        labels[index] = label
        exit_offsets[index] = offset

    return pd.DataFrame(
        {
            "label": labels,
            "exit_offset": exit_offsets,
            "mfe": mfe,
            "mae": mae,
        },
        index=frame.index,
    )
