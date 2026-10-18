from __future__ import annotations

import pandas as pd

from automl_dl.data.market_data import resample_ohlcv


def test_resample_keeps_only_complete_candles() -> None:
    index = pd.date_range("2026-01-01", periods=7, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": range(7),
            "high": [value + 2 for value in range(7)],
            "low": [value - 1 for value in range(7)],
            "close": [value + 1 for value in range(7)],
            "volume": [1.0] * 7,
        },
        index=index,
    )

    result = resample_ohlcv(frame, "5m")

    assert len(result) == 1
    assert result.iloc[0].to_dict() == {
        "open": 0.0,
        "high": 6.0,
        "low": -1.0,
        "close": 5.0,
        "volume": 5.0,
    }
