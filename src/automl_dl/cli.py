from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from automl_dl.config import SUPPORTED_SYMBOLS
from automl_dl.data import BinanceOHLCVStore, mark_candle_availability
from automl_dl.experiments import run_baseline_walk_forward
from automl_dl.features import add_market_features
from automl_dl.labels import triple_barrier_labels
from automl_dl.pipeline import build_baseline_table
from automl_dl.trading import (
    RiskEngine,
    RiskParameters,
    backtest_signals,
    trading_metrics,
)
from automl_dl.validation import PurgedWalkForwardSplit

app = typer.Typer(no_args_is_help=True)


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@app.command()
def download(
    symbol: Annotated[str, typer.Option(help="CCXT symbol, for example BTC/USDT")],
    since: Annotated[str, typer.Option(help="UTC ISO date or datetime")],
    until: Annotated[str | None, typer.Option(help="Optional UTC ISO end")] = None,
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    """Incrementally download closed Binance one-minute candles."""
    if symbol not in SUPPORTED_SYMBOLS:
        raise typer.BadParameter(f"Supported symbols: {', '.join(SUPPORTED_SYMBOLS)}")
    store = BinanceOHLCVStore(data_dir)
    frame = store.update(
        symbol,
        timeframe="1m",
        start=_utc_datetime(since),
        until=_utc_datetime(until) if until else None,
    )
    typer.echo(f"Stored {len(frame):,} closed candles for {symbol}")


@app.command()
def download_all(
    since: Annotated[str, typer.Option(help="UTC ISO date or datetime")],
    until: Annotated[str | None, typer.Option(help="Optional UTC ISO end")] = None,
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    """Incrementally download all configured Binance symbols."""
    store = BinanceOHLCVStore(data_dir)
    for symbol in SUPPORTED_SYMBOLS:
        frame = store.update(
            symbol,
            timeframe="1m",
            start=_utc_datetime(since),
            until=_utc_datetime(until) if until else None,
        )
        typer.echo(f"Stored {len(frame):,} closed candles for {symbol}")


@app.command()
def resample(
    symbol: Annotated[str, typer.Option(help="CCXT symbol, for example BTC/USDT")],
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    """Build higher timeframes from locally stored one-minute candles."""
    paths = BinanceOHLCVStore(data_dir).build_timeframes(symbol)
    for timeframe, path in paths.items():
        typer.echo(f"{timeframe}: {path}")


@app.command()
def prepare(
    symbol: Annotated[str, typer.Option(help="CCXT symbol, for example BTC/USDT")],
    timeframe: Annotated[str, typer.Option()] = "1h",
    horizon: Annotated[int, typer.Option(min=1)] = 24,
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
) -> None:
    """Create causal market features and triple-barrier training targets."""
    store = BinanceOHLCVStore(data_dir)
    candles = store.load(symbol, timeframe)
    if candles.empty:
        raise typer.BadParameter(f"No local data for {symbol} {timeframe}")

    featured = add_market_features(candles)
    labels = triple_barrier_labels(featured, horizon=horizon)
    dataset = pd.concat([featured, labels], axis=1).dropna()
    safe_symbol = symbol.replace("/", "-")
    output = data_dir / "processed" / safe_symbol / f"{timeframe}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output)
    typer.echo(f"Stored {len(dataset):,} labelled rows in {output}")


@app.command()
def baseline(
    symbol: Annotated[str, typer.Option(help="CCXT symbol, for example BTC/USDT")],
    decision_timeframe: Annotated[str, typer.Option()] = "15m",
    horizon: Annotated[int, typer.Option(min=1)] = 16,
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
    output_dir: Annotated[Path, typer.Option()] = Path("artifacts/baseline"),
) -> None:
    """Run a purged walk-forward gradient boosting baseline."""
    store = BinanceOHLCVStore(data_dir)
    timeframes = ("5m", "15m", "1h", "4h")
    frames = {timeframe: store.load(symbol, timeframe) for timeframe in timeframes}
    missing = [timeframe for timeframe, frame in frames.items() if frame.empty]
    if missing:
        raise typer.BadParameter(f"Missing local timeframes: {', '.join(missing)}")

    table = build_baseline_table(
        frames,
        decision_timeframe=decision_timeframe,
        horizon=horizon,
    )
    if len(table) < 500:
        raise typer.BadParameter(f"Only {len(table)} complete rows; at least 500 are required")

    test_size = max(50, len(table) // 10)
    splitter = PurgedWalkForwardSplit(
        n_splits=3,
        test_size=test_size,
        label_horizon=horizon,
        embargo=horizon,
        min_train_size=max(100, len(table) // 2),
    )
    feature_columns = [
        column
        for column in table.columns
        if any(column.startswith(f"{timeframe}_") for timeframe in timeframes)
    ]
    result = run_baseline_walk_forward(
        table,
        feature_columns=feature_columns,
        return_column=f"return_h{horizon}",
        splitter=splitter,
    )

    risk_engine = RiskEngine(
        RiskParameters(
            round_trip_cost=0.0024,
            max_holding_bars=horizon,
        )
    )
    signal_rows = []
    atr_values = (
        table[f"{decision_timeframe}_atr_pct"]
        .reindex(result.predictions.index)
        .to_numpy(dtype=float)
    )
    for (_, prediction), atr_return in zip(
        result.predictions.iterrows(),
        atr_values,
        strict=True,
    ):
        signal = risk_engine.decide(
            direction_probabilities=prediction[
                ["probability_short", "probability_flat", "probability_long"]
            ].to_numpy(dtype=float),
            return_quantiles=prediction[["return_q0.1", "return_q0.5", "return_q0.9"]].to_numpy(
                dtype=float
            ),
            atr_return=float(atr_return),
        )
        signal_rows.append(
            {
                "side": signal.side,
                "take_profit_return": signal.take_profit_return,
                "stop_loss_return": signal.stop_loss_return,
                "max_holding_bars": signal.max_holding_bars,
            }
        )
    signals = pd.DataFrame(signal_rows, index=result.predictions.index)
    available_candles = mark_candle_availability(
        frames[decision_timeframe][["open", "high", "low", "close"]],
        decision_timeframe,
    ).loc[result.predictions.index]
    trades, equity = backtest_signals(available_candles, signals)
    periods_per_year = {
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "1h": 365 * 24,
        "4h": 365 * 6,
        "1d": 365,
    }[decision_timeframe]
    metrics = {
        **result.metrics,
        **{
            f"trading_{name}": value
            for name, value in trading_metrics(
                trades,
                equity,
                periods_per_year=periods_per_year,
            ).items()
        },
    }

    safe_symbol = symbol.replace("/", "-")
    destination = output_dir / safe_symbol / decision_timeframe
    destination.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(destination / "predictions.parquet")
    trades.to_parquet(destination / "trades.parquet")
    equity.rename("equity").to_frame().to_parquet(destination / "equity.parquet")
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    typer.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
