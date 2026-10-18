from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeSignal:
    side: int
    confidence: float
    take_profit_return: float
    stop_loss_return: float
    max_holding_bars: int


@dataclass(frozen=True)
class RiskParameters:
    min_confidence: float = 0.55
    min_edge_after_costs: float = 0.001
    round_trip_cost: float = 0.002
    minimum_reward_risk: float = 1.5
    max_holding_bars: int = 24


class RiskEngine:
    """Convert calibrated model outputs into conservative executable signals."""

    def __init__(self, parameters: RiskParameters | None = None) -> None:
        self.parameters = parameters or RiskParameters()

    def decide(
        self,
        *,
        direction_probabilities: np.ndarray,
        return_quantiles: np.ndarray,
        atr_return: float,
    ) -> TradeSignal:
        if direction_probabilities.shape != (3,):
            raise ValueError("direction_probabilities order must be [short, flat, long]")
        if return_quantiles.shape != (3,):
            raise ValueError("return_quantiles must contain [q10, q50, q90]")

        short_probability, _, long_probability = direction_probabilities
        q10, q50, q90 = return_quantiles
        threshold = self.parameters.round_trip_cost + self.parameters.min_edge_after_costs

        if long_probability >= self.parameters.min_confidence and q50 > threshold:
            stop = max(abs(min(q10, 0.0)), atr_return)
            take = max(q90, stop * self.parameters.minimum_reward_risk)
            return TradeSignal(
                1,
                float(long_probability),
                float(take),
                float(stop),
                self.parameters.max_holding_bars,
            )

        if short_probability >= self.parameters.min_confidence and q50 < -threshold:
            stop = max(max(q90, 0.0), atr_return)
            take = max(abs(q10), stop * self.parameters.minimum_reward_risk)
            return TradeSignal(
                -1,
                float(short_probability),
                float(take),
                float(stop),
                self.parameters.max_holding_bars,
            )

        return TradeSignal(0, float(max(short_probability, long_probability)), 0.0, 0.0, 0)


def backtest_signals(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    initial_equity: float = 1.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0002,
) -> tuple[pd.DataFrame, pd.Series]:
    """Backtest one position at a time; signals at close execute at next open."""
    required_candles = {"open", "high", "low", "close"}
    required_signals = {"side", "take_profit_return", "stop_loss_return", "max_holding_bars"}
    if missing := required_candles.difference(candles.columns):
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    if missing := required_signals.difference(signals.columns):
        raise ValueError(f"Missing signal columns: {sorted(missing)}")
    if not candles.index.equals(signals.index):
        raise ValueError("candles and signals must have the same index")

    equity = initial_equity
    equity_values = pd.Series(index=candles.index, dtype=float)
    equity_values.iloc[0] = equity
    trades: list[dict[str, object]] = []
    index = 0

    while index < len(candles) - 1:
        equity_values.iloc[index] = equity
        signal = signals.iloc[index]
        side = int(signal["side"])
        if side == 0:
            index += 1
            continue

        entry_index = index + 1
        entry_price = float(candles.iloc[entry_index]["open"])
        entry_price *= 1.0 + side * slippage_rate
        take_return = float(signal["take_profit_return"])
        stop_return = float(signal["stop_loss_return"])
        holding = max(1, int(signal["max_holding_bars"]))
        take_price = entry_price * (1.0 + side * take_return)
        stop_price = entry_price * (1.0 - side * stop_return)
        final_index = min(entry_index + holding - 1, len(candles) - 1)
        exit_index = final_index
        exit_price = float(candles.iloc[final_index]["close"])
        reason = "time"

        for cursor in range(entry_index, final_index + 1):
            high = float(candles.iloc[cursor]["high"])
            low = float(candles.iloc[cursor]["low"])
            hit_take = high >= take_price if side == 1 else low <= take_price
            hit_stop = low <= stop_price if side == 1 else high >= stop_price
            if hit_stop:
                exit_index, exit_price, reason = cursor, stop_price, "stop"
                break
            if hit_take:
                exit_index, exit_price, reason = cursor, take_price, "take"
                break

        exit_price *= 1.0 - side * slippage_rate
        gross_return = side * (exit_price / entry_price - 1.0)
        net_return = gross_return - 2.0 * fee_rate
        equity *= 1.0 + net_return
        equity_values.iloc[index : exit_index + 1] = equity
        trades.append(
            {
                "signal_time": candles.index[index],
                "entry_time": candles.index[entry_index],
                "exit_time": candles.index[exit_index],
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "net_return": net_return,
                "exit_reason": reason,
                "equity": equity,
            }
        )
        index = exit_index + 1

    equity_values = equity_values.ffill().fillna(initial_equity)
    return pd.DataFrame(trades), equity_values


def trading_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    *,
    periods_per_year: int,
) -> dict[str, float]:
    returns = equity.pct_change().fillna(0.0)
    volatility = float(returns.std())
    sharpe = 0.0
    if volatility > 0.0:
        sharpe = float(returns.mean() / volatility * np.sqrt(periods_per_year))

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    if trades.empty:
        return {
            "trade_count": 0.0,
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            "sharpe": sharpe,
            "max_drawdown": float(drawdown.min()),
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    trade_returns = trades["net_return"].astype(float)
    gains = float(trade_returns[trade_returns > 0.0].sum())
    losses = abs(float(trade_returns[trade_returns < 0.0].sum()))
    return {
        "trade_count": float(len(trades)),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((trade_returns > 0.0).mean()),
        "profit_factor": gains / losses if losses > 0.0 else 0.0,
    }
