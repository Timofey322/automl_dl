from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from automl_dl.config import SUPPORTED_SYMBOLS
from automl_dl.data import BinanceOHLCVStore
from automl_dl.features import add_market_features
from automl_dl.labels import triple_barrier_labels

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


if __name__ == "__main__":
    app()
