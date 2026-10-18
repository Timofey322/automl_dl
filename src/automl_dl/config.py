from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SYMBOLS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT")
SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M")


@dataclass(frozen=True)
class DataConfig:
    exchange: str = "binance"
    symbols: tuple[str, ...] = SUPPORTED_SYMBOLS
    base_timeframe: str = "1m"
    timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES
    data_dir: Path = Path("data")
    page_limit: int = 1000


@dataclass(frozen=True)
class BarrierConfig:
    horizon: int = 24
    take_profit_atr: float = 2.0
    stop_loss_atr: float = 1.0
    atr_window: int = 14


@dataclass(frozen=True)
class ModelConfig:
    input_features: int = 8
    context_lengths: dict[str, int] = field(
        default_factory=lambda: {"1m": 256, "5m": 128, "15m": 96}
    )
    patch_lengths: dict[str, int] = field(default_factory=lambda: {"1m": 16, "5m": 8, "15m": 8})
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    horizons: tuple[int, ...] = (5, 15, 60)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
