"""Backtest af den daglige LSTM-strategi over et år (default kalenderåret 2025).

Simuleringen går én handelsdag ad gangen: på beslutningsdag ``t`` bruges kun data til
og med ``t`` til at forudsige hver akties næste dags open→close-afkast. Den stærkeste
ticker købes ved næste open og sælges ved næste close (all-in, som live-traderen i
``trader.rotate_to_symbol``). Det realiserede afkast tilskrives porteføljen, og
simuleringen rykker til næste dag.

Ingen look-ahead: feature-vinduet slutter på ``t``; handlen sker på ``t+1``. Forudsigelse
og facit deler index via ``targets_next_day_open_to_close_pct`` (alignet på ``t``).

Kør:
    python -m stock_predictor.backtest
    python -m stock_predictor.main --backtest

Resultat: pop op-vindue med equity-kurve (start 100.000) + equal-weight buy & hold-benchmark,
samt CSV-log i output/backtest_<år>.csv.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from alpaca.data.historical import StockHistoricalDataClient

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stock_predictor import config  # noqa: E402
from stock_predictor.data_fetcher import _fetch_symbol_range, fetch_daily_bars  # noqa: E402
from stock_predictor.feature_engineer import (  # noqa: E402
    engineer_features,
    targets_next_day_open_to_close_pct,
)
from stock_predictor.predict import _load_bundle  # noqa: E402

logger = logging.getLogger(__name__)

START_EQUITY = 100_000.0
# Antal sekvenser pr. model-forward (CPU-venligt; begrænser RAM).
INFER_BATCH = 256
# Ekstra kalenderdage bag beslutningsvinduet, så 600-dages vindue + warmup er dækket.
_LOOKBACK_CALENDAR_DAYS = 5 * 366
# Buy & hold-benchmark: SPY hentes live via Alpaca, kun til backtest (caches ikke).
_BENCHMARK_SYMBOL = "SPY"


@dataclass
class BacktestResult:
    daily_log: pd.DataFrame  # trade_date, chosen_symbol, predicted_pct, actual_pct, equity
    equity: pd.Series        # strategi-equity indekseret på handelsdato
    benchmark: pd.Series | None  # SPY buy & hold (samme index), None hvis ikke hentet
    final_equity: float
    total_return_pct: float


def _predict_symbol(
    model,
    scaler,
    device,
    n_features: int,
    seq_len: int,
    ohlcv: pd.DataFrame,
    year: int,
) -> tuple[pd.Series, pd.Series] | None:
    """Returnér (pred, actual) indekseret på handelsdato (t+1) i ``year`` for ét symbol.

    pred: modellens forudsagte open→close % for handelsdagen.
    actual: realiseret open→close % samme dag (facit fra targets_next_day_open_to_close_pct).
    """
    try:
        feats = engineer_features(ohlcv)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Feature engineering fejlede: %s", exc)
        return None
    if len(feats) < seq_len + 1:
        return None

    vals = feats.to_numpy(dtype=np.float64)
    if vals.shape[1] != n_features:
        return None
    dates = feats.index
    y_actual = targets_next_day_open_to_close_pct(ohlcv, feats.index)

    windows: list[np.ndarray] = []
    trade_dates: list[pd.Timestamp] = []
    actuals: list[float] = []

    # Beslutningsindeks i: vindue slutter på dates[i]; handel på dates[i+1].
    for i in range(seq_len - 1, len(feats) - 1):
        trade_date = dates[i + 1]
        if trade_date.year != year:
            continue
        window = vals[i - seq_len + 1 : i + 1]
        tgt = float(y_actual.iloc[i])
        if np.any(np.isnan(window)) or np.isnan(tgt):
            continue
        windows.append(window)
        trade_dates.append(trade_date)
        actuals.append(tgt)

    if not windows:
        return None

    preds: list[float] = []
    with torch.no_grad():
        for start in range(0, len(windows), INFER_BATCH):
            batch = windows[start : start + INFER_BATCH]
            flat = scaler.transform(np.concatenate(batch, axis=0))
            xt = torch.from_numpy(
                flat.reshape(len(batch), seq_len, n_features).astype(np.float32)
            ).to(device)
            out = model(xt).squeeze(-1).detach().cpu().numpy().reshape(-1)
            preds.extend(float(v) for v in out)

    idx = pd.DatetimeIndex(trade_dates)
    return (
        pd.Series(preds, index=idx),
        pd.Series(actuals, index=idx),
    )


def _fetch_spy_benchmark(year: int, trade_index: pd.DatetimeIndex) -> pd.Series | None:
    """Hent SPY-dagsbarer live via Alpaca som buy & hold-benchmark.

    Kun til backtesten: SPY skrives ikke til cache og indgår ikke i watchlisten. Returnér
    en equity-kurve (start START_EQUITY) alignet på strategiens handelsdage, eller None
    hvis nøgler mangler / download fejler (så vises grafen uden benchmark).
    """
    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        logger.warning(
            "Springer SPY-benchmark over: ingen Alpaca-nøgler (sæt ALPACA_API_KEY/"
            "ALPACA_SECRET_KEY i .env)."
        )
        return None
    try:
        client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke oprette Alpaca-klient til SPY-benchmark: %s", exc)
        return None

    # Alpaca behandler `end` som eksklusiv → hent en dag ekstra for at få 31. dec med.
    spy = _fetch_symbol_range(client, _BENCHMARK_SYMBOL, date(year, 1, 1), date(year + 1, 1, 2))
    if spy is None or spy.empty:
        logger.warning("Ingen SPY-data fra Alpaca — viser graf uden benchmark.")
        return None

    close = spy["close"].astype(float).sort_index()
    close = close[close.index.year == year]
    if close.empty:
        logger.warning("SPY-data dækkede ikke %s — viser graf uden benchmark.", year)
        return None

    norm = close / float(close.iloc[0]) * START_EQUITY
    logger.info("SPY-benchmark hentet live fra Alpaca (%s barer i %s).", len(close), year)
    return norm.reindex(trade_index).ffill().bfill()


def run_backtest(
    year: int = 2025,
    *,
    show_plot: bool = True,
    output_path: Path | None = None,
) -> BacktestResult:
    """Kør hele backtesten for kalenderåret ``year`` og (valgfrit) vis pop op-graf."""
    model, scaler, device, n_features, seq_len = _load_bundle()
    logger.info("Model indlæst (seq_len=%s, n_features=%s, device=%s).", seq_len, n_features, device)

    end = date(year, 12, 31)
    fetch_result = fetch_daily_bars(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        config.WATCHLIST,
        end=end,
        lookback_calendar_days=_LOOKBACK_CALENDAR_DAYS,
        extra_buffer_days=0,
        prefer_cache_only=True,
    )
    bars = fetch_result.bars
    if not bars:
        raise RuntimeError(
            "Ingen barer fra cache — kør evt. import_watchlist_csv_to_cache eller --train først."
        )
    logger.info("Indlæste %s symboler fra cache.", len(bars))

    pred_cols: Dict[str, pd.Series] = {}
    actual_cols: Dict[str, pd.Series] = {}

    for sym in config.WATCHLIST:
        ohlcv = bars.get(sym)
        if ohlcv is None or ohlcv.empty:
            continue
        res = _predict_symbol(model, scaler, device, n_features, seq_len, ohlcv, year)
        if res is None:
            continue
        pred_cols[sym], actual_cols[sym] = res

    if not pred_cols:
        raise RuntimeError("Ingen symboler gav forudsigelser for året — tjek cache-dækning.")

    pred_df = pd.DataFrame(pred_cols).sort_index()
    actual_df = pd.DataFrame(actual_cols).reindex(pred_df.index)
    logger.info(
        "Forudsigelser: %s handelsdage × %s symboler.", pred_df.shape[0], pred_df.shape[1]
    )

    # --- Day-by-day simulering ---
    equity = START_EQUITY
    rows = []
    equity_points = []
    for trade_date, pred_row in pred_df.iterrows():
        valid = pred_row.dropna()
        if valid.empty:
            continue
        best = valid.idxmax()
        realized = actual_df.loc[trade_date, best]
        if pd.isna(realized):
            continue
        equity *= 1.0 + float(realized) / 100.0
        rows.append(
            {
                "trade_date": trade_date.date().isoformat(),
                "chosen_symbol": best,
                "predicted_pct": round(float(valid[best]), 4),
                "actual_pct": round(float(realized), 4),
                "equity": round(equity, 2),
            }
        )
        equity_points.append((trade_date, equity))

    if not rows:
        raise RuntimeError("Simuleringen producerede ingen handler.")

    daily_log = pd.DataFrame(rows)
    equity_series = pd.Series(
        [p[1] for p in equity_points], index=pd.DatetimeIndex([p[0] for p in equity_points])
    )

    # --- Buy & hold-benchmark: SPY hentet live via Alpaca (kun til backtest) ---
    benchmark = _fetch_spy_benchmark(year, equity_series.index)

    final_equity = float(equity_series.iloc[-1])
    total_return_pct = (final_equity / START_EQUITY - 1.0) * 100.0

    # --- Gem CSV ---
    if output_path is None:
        output_path = _ROOT / "output" / f"backtest_{year}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_log.to_csv(output_path, index=False)
    logger.info("Daglig log gemt til %s (%s handelsdage).", output_path, len(daily_log))

    if benchmark is not None:
        bench_final = float(benchmark.iloc[-1])
        bench_return = (bench_final / START_EQUITY - 1.0) * 100.0
        logger.info(
            "Strategi: %.2f USD (%+.2f%%) | SPY buy & hold: %.2f USD (%+.2f%%)",
            final_equity,
            total_return_pct,
            bench_final,
            bench_return,
        )
    else:
        logger.info("Strategi: %.2f USD (%+.2f%%)", final_equity, total_return_pct)

    result = BacktestResult(
        daily_log=daily_log,
        equity=equity_series,
        benchmark=benchmark,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
    )

    if show_plot:
        _plot(result, year)

    return result


def plot_saved_backtest(
    year: int = 2025,
    *,
    output_path: Path | None = None,
    show_plot: bool = True,
) -> BacktestResult:
    """Genåbn grafen for en tidligere gemt backtest uden at køre simuleringen igen.

    Equity-kurven læses fra ``output/backtest_<år>.csv`` (kolonnerne trade_date + equity),
    og SPY buy & hold-benchmark hentes live via Alpaca på ny (samme som under selve
    backtesten). Praktisk til at se en gammel backtest med SPY + strategi igen.
    """
    if output_path is None:
        output_path = _ROOT / "output" / f"backtest_{year}.csv"
    if not output_path.exists():
        raise FileNotFoundError(
            f"Ingen gemt backtest fundet: {output_path}. Kør --backtest for at lave en."
        )

    daily_log = pd.read_csv(output_path)
    if "trade_date" not in daily_log.columns or "equity" not in daily_log.columns:
        raise ValueError(
            f"{output_path} mangler kolonnerne 'trade_date'/'equity' — er det en backtest-log?"
        )

    idx = pd.DatetimeIndex(pd.to_datetime(daily_log["trade_date"]))
    equity_series = pd.Series(daily_log["equity"].astype(float).to_numpy(), index=idx)
    if equity_series.empty:
        raise ValueError(f"{output_path} indeholder ingen handelsdage.")

    benchmark = _fetch_spy_benchmark(year, equity_series.index)
    final_equity = float(equity_series.iloc[-1])
    total_return_pct = (final_equity / START_EQUITY - 1.0) * 100.0

    result = BacktestResult(
        daily_log=daily_log,
        equity=equity_series,
        benchmark=benchmark,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
    )
    logger.info(
        "Genindlæst backtest fra %s (%s handelsdage, slut %.2f USD, %+.2f%%).",
        output_path,
        len(equity_series),
        final_equity,
        total_return_pct,
    )
    if show_plot:
        _plot(result, year)
    return result


def _plot(result: BacktestResult, year: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        result.equity.index,
        result.equity.values,
        label="LSTM-strategi (all-in bedste ticker)",
        color="#1f77b4",
        linewidth=1.8,
    )
    if result.benchmark is not None:
        ax.plot(
            result.benchmark.index,
            result.benchmark.values,
            label="SPY buy & hold",
            color="#888888",
            linewidth=1.4,
            linestyle="--",
        )
    ax.axhline(START_EQUITY, color="#cccccc", linewidth=1.0, zorder=0)

    if result.benchmark is not None:
        bench_return = (float(result.benchmark.iloc[-1]) / START_EQUITY - 1.0) * 100.0
        bench_line = (
            f"   |   SPY buy & hold: {float(result.benchmark.iloc[-1]):,.0f} USD "
            f"({bench_return:+.2f}%)"
        )
    else:
        bench_line = ""
    ax.set_title(
        f"Backtest {year} — start {START_EQUITY:,.0f} USD\n"
        f"Strategi: {result.final_equity:,.0f} USD ({result.total_return_pct:+.2f}%){bench_line}"
    )
    ax.set_xlabel("Handelsdato")
    ax.set_ylabel("Porteføljeværdi (USD)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    last_date = result.equity.index[-1]
    ax.annotate(
        f"{result.final_equity:,.0f} USD\n{result.total_return_pct:+.2f}%",
        xy=(last_date, result.final_equity),
        xytext=(-90, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color="#1f77b4",
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_backtest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
