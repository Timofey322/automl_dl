from __future__ import annotations

import pandas as pd
import pytest

from automl_dl.trading import backtest_signals, trading_metrics


def test_backtest_uses_stop_when_both_barriers_hit() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    candles = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 103.0, 100.0],
            "low": [100.0, 98.0, 100.0],
            "close": [100.0, 100.0, 100.0],
        },
        index=index,
    )
    signals = pd.DataFrame(
        {
            "side": [1, 0, 0],
            "take_profit_return": [0.02, 0.0, 0.0],
            "stop_loss_return": [0.01, 0.0, 0.0],
            "max_holding_bars": [2, 0, 0],
        },
        index=index,
    )

    trades, equity = backtest_signals(
        candles,
        signals,
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    metrics = trading_metrics(trades, equity, periods_per_year=365 * 24 * 4)

    assert trades.iloc[0]["exit_reason"] == "stop"
    assert trades.iloc[0]["net_return"] == pytest.approx(-0.01)
    assert metrics["trade_count"] == 1.0
