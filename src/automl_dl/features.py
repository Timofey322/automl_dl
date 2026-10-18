from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_features(frame: pd.DataFrame, volatility_window: int = 24) -> pd.DataFrame:
    """Create scale-stable features using past and current closed candles only."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    result = frame.copy()
    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["log_return"] = np.log(result["close"]).diff()
    result["range_pct"] = (result["high"] - result["low"]) / previous_close
    result["body_pct"] = (result["close"] - result["open"]) / result["open"]
    result["log_volume"] = np.log1p(result["volume"])
    result["volume_change"] = result["log_volume"].diff()
    result["volatility"] = result["log_return"].rolling(volatility_window).std()
    result["atr"] = true_range.rolling(volatility_window).mean()
    result["atr_pct"] = result["atr"] / result["close"]
    return result.replace([np.inf, -np.inf], np.nan)


def completed_features(frame: pd.DataFrame, volatility_window: int = 24) -> pd.DataFrame:
    return add_market_features(frame, volatility_window).dropna()
