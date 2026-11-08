# AutoML для временных рядов с использованием глубокого обучения

Годовой проект магистерской программы «Искусственный интеллект» НИУ ВШЭ.

## Информация о проекте

- **Тема № 17:** Предсказание стоимости акций
- **Автор:** Курлянд Тимофей Александрович
- **Куратор:** Дмитрий Янышев

## Цель

Разработать и исследовать AutoML-подход к прогнозированию стоимости криптовалют и ценных бумаг по временным рядам с применением моделей глубокого обучения.

## Поддерживаемые тикеры

Источником данных служит Binance через CCXT. Система работает с торговыми парами:

- **BTC/USDT** — Bitcoin;
- **ETH/USDT** — Ethereum;
- **BNB/USDT** — BNB;
- **SOL/USDT** — Solana.

## Архитектура

Система строится как исследовательский конвейер:

1. Инкрементальная загрузка закрытых минутных свечей OHLCV.
2. Построение таймфреймов `5m`, `15m`, `1h`, `4h`, `1d`, `1w`, `1M`.
3. Расчёт причинных признаков и разметки triple barrier.
4. Мультитаймфреймный Patch Transformer с отдельным энкодером каждого масштаба.
5. Прогноз направления, квантилей доходности, волатильности и ценовых экстремумов.
6. Risk Engine, формирующий `LONG`, `SHORT` или `NO TRADE`, а также `TP` и `SL`.
7. Walk-forward валидация и бэктест с комиссиями и проскальзыванием.

Pretrained-модели Kronos, Chronos-2 и Granite TTM планируется сравнивать с основной
обучаемой архитектурой на одинаковых временных разбиениях.

## Быстрый старт

Требуются Python 3.11 и [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra dev
uv run automl-dl download --symbol BTC/USDT --since 2025-01-01
uv run automl-dl backfill-all --since 2021-01-01
uv run automl-dl resample --symbol BTC/USDT
uv run automl-dl prepare --symbol BTC/USDT --timeframe 1h --horizon 24
uv run automl-dl baseline --symbol BTC/USDT --decision-timeframe 15m --horizon 16
```

Данные сохраняются локально в Parquet и не добавляются в Git. Результаты
walk-forward экспериментов сохраняются в `artifacts/baseline/`.

## Чекпойнт 2: EDA

Файл чекпойнта: [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb).

Он включает:
- статистики, пропуски, ошибочные значения и выбросы;
- баланс классов triple-barrier и связи признаков с таргетом;
- корреляции, `pairplot`, сводные таблицы;
- стационарность ADF/KPSS и ACF;
- простой HGB-baseline с permutation importance;
- расширенный causal feature store (>1000 признаков) и исследование важности;
- разбор аугментаций и практическую применимость задачи.

```powershell
uv sync --extra dev --extra eda
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb
```

## Проверка

```powershell
uv run ruff check .
uv run mypy
uv run pytest
```
