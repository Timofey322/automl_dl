from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
PANDAS_RULES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W-MON",
    "1M": "1MS",
}


def mark_candle_availability(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Move candle-open timestamps to the moment the candle becomes observable."""
    if timeframe not in PANDAS_RULES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    result = frame.copy()
    result.index = result.index + pd.tseries.frequencies.to_offset(PANDAS_RULES[timeframe])
    result.index.name = "available_at"
    return result


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS).set_index("timestamp")


def _to_frame(rows: list[list[float]]) -> pd.DataFrame:
    if not rows:
        return _empty_ohlcv()
    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return (
        frame.set_index("timestamp")
        .sort_index()
        .loc[lambda value: ~value.index.duplicated(keep="last")]
        .astype(float)
    )


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate closed lower-timeframe candles into closed higher-timeframe candles."""
    if timeframe not in PANDAS_RULES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if frame.empty:
        return frame.copy()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("OHLCV index must be a timezone-aware DatetimeIndex")

    rule = PANDAS_RULES[timeframe]
    result = frame.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    result = result.dropna(subset=["open", "high", "low", "close"])

    source_step = frame.index.to_series().diff().median()
    if pd.isna(source_step):
        return result.iloc[0:0]
    source_end = frame.index[-1] + source_step
    bucket_starts = pd.DatetimeIndex(result.index)
    bucket_ends = bucket_starts + pd.tseries.frequencies.to_offset(rule)
    complete = pd.Series(bucket_ends <= source_end, index=result.index)
    return result[complete].copy()


@dataclass
class BinanceOHLCVStore:
    root: Path
    exchange: Any = None
    page_limit: int = 1000

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.exchange is None:
            self.exchange = ccxt.binance({"enableRateLimit": True})

    def path_for(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "-")
        # Windows paths are case-insensitive, so 1m and 1M need distinct slugs.
        timeframe_slug = "1mo" if timeframe == "1M" else timeframe
        return self.root / "raw" / safe_symbol / f"{timeframe_slug}.parquet"

    def load(self, symbol: str, timeframe: str = "1m") -> pd.DataFrame:
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            return _empty_ohlcv()
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame.sort_index()

    def fetch(
        self,
        symbol: str,
        *,
        timeframe: str = "1m",
        since: datetime,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch pages and exclude the current, potentially incomplete candle."""
        if since.tzinfo is None:
            raise ValueError("since must be timezone-aware")
        until = until or datetime.now(UTC)
        if until.tzinfo is None:
            raise ValueError("until must be timezone-aware")

        cursor_ms = int(since.timestamp() * 1000)
        until_ms = int(until.timestamp() * 1000)
        candle_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)
        rows: list[list[float]] = []

        while cursor_ms < until_ms:
            page = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=cursor_ms,
                limit=self.page_limit,
            )
            if not page:
                break
            closed = [row for row in page if int(row[0]) + candle_ms <= until_ms]
            rows.extend(closed)
            next_cursor = int(page[-1][0]) + 1
            if next_cursor <= cursor_ms:
                raise RuntimeError("Exchange returned a non-advancing OHLCV page")
            cursor_ms = next_cursor
            if len(page) < self.page_limit:
                break

        return _to_frame(rows)

    def update(
        self,
        symbol: str,
        *,
        timeframe: str = "1m",
        start: datetime,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        existing = self.load(symbol, timeframe)
        fetch_from = start
        if not existing.empty:
            candle_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)
            fetch_from = (existing.index[-1] + pd.Timedelta(milliseconds=candle_ms)).to_pydatetime()

        fresh = self.fetch(symbol, timeframe=timeframe, since=fetch_from, until=until)
        combined = pd.concat([existing, fresh]).sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="last")]

        path = self.path_for(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path)
        return combined

    def build_timeframes(
        self,
        symbol: str,
        *,
        source_timeframe: str = "1m",
        targets: tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d", "1w", "1M"),
    ) -> dict[str, Path]:
        source = self.load(symbol, source_timeframe)
        if source.empty:
            raise ValueError(f"No source data for {symbol} {source_timeframe}")

        outputs: dict[str, Path] = {}
        for timeframe in targets:
            aggregated = resample_ohlcv(source, timeframe)
            path = self.path_for(symbol, timeframe)
            path.parent.mkdir(parents=True, exist_ok=True)
            aggregated.to_parquet(path)
            outputs[timeframe] = path
        return outputs
