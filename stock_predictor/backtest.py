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

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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
from stock_predictor import metrics as _metrics  # noqa: E402
from stock_predictor.data_fetcher import (  # noqa: E402
    _cache_path,
    _fetch_symbol_range,
    _read_cache_parquet,
    fetch_daily_bars,
)
from stock_predictor.feature_engineer import (  # noqa: E402
    engineer_features,
    targets_next_day_open_to_close_pct,
)
from stock_predictor.predict import _load_bundle  # noqa: E402

logger = logging.getLogger(__name__)

START_EQUITY = 100_000.0
# Mappe hvor hver backtest gemmes som sin egen tidsstemplede CSV + JSON-sidecar (overskrives ikke).
_BACKTEST_DIR = _ROOT / "output" / "backtests"
# Antal sekvenser pr. model-forward (CPU-venligt; begrænser RAM).
INFER_BATCH = 256
# Kalenderdage bag beslutningsvinduet: nok til at dække SEQ_LEN + feature-warmup.
# Skalerer med SEQ_LEN (samme udregning som live-inferens), så backtesten ikke
# mister de første år når lookback-vinduet vokser (fx SEQ_LEN=2000 ≈ 8 års vindue).
_LOOKBACK_CALENDAR_DAYS = config.INFERENCE_FETCH_CALENDAR_DAYS
# Buy & hold-benchmark: SPY læses fra lokal cache (cache/SPY.parquet, bygget af
# build_spy_cache fra yfinance og dækker 2015→nu), med live Alpaca-IEX som fallback.
# Alpacas gratis IEX-feed har kun SPY fra ~2021, så cachen er nødvendig før det.
_BENCHMARK_SYMBOL = "SPY"


def _spy_cached_close() -> pd.Series | None:
    """Justeret SPY-lukkekurs fra den lokale cache (cache/SPY.parquet), eller None.

    Byg/refresh cachen med ``python -m stock_predictor.build_spy_cache``.
    """
    df = _read_cache_parquet(_cache_path(config.CACHE_DIR, _BENCHMARK_SYMBOL))
    if df is None or df.empty or "close" not in df.columns:
        return None
    close = df["close"].astype(float)
    close.index = pd.to_datetime(close.index)
    return close.sort_index()


# Visningsnavne + plot-stil for de fire strategi-kurver (rækkefølge bevares).
_STRATEGY_LABELS: dict[str, str] = {
    "best": "Bedste ticker",
    "second": "Næstbedste ticker",
    "third": "Tredjebedste ticker",
    "avg": "Snit af top 3 (1/3 hver)",
}
_STRATEGY_STYLE: dict[str, dict] = {
    "best": {"color": "#1f77b4", "linewidth": 1.8, "linestyle": "-"},
    "second": {"color": "#2ca02c", "linewidth": 1.2, "linestyle": "-"},
    "third": {"color": "#9467bd", "linewidth": 1.2, "linestyle": "-"},
    "avg": {"color": "#d62728", "linewidth": 1.8, "linestyle": "-"},
}

# Navngivne markedsregimer til regime-backtesten: (label, start, slut, er_stress).
# Vinduer der ikke overlapper det valgte interval springes automatisk over (for få dage).
_REGIMES: list[tuple[str, str, str, bool]] = [
    ("Dot-com crash", "2000-03-01", "2002-10-09", True),
    ("Mid-2000s bull", "2003-01-01", "2007-09-30", False),
    ("GFC 2008", "2007-10-01", "2009-03-09", True),
    ("Post-GFC recovery", "2009-03-10", "2011-06-30", False),
    ("2011 EU debt crisis", "2011-07-01", "2011-12-31", True),
    ("QE bull", "2012-01-01", "2014-12-31", False),
    ("China/oil selloff", "2015-06-01", "2016-02-29", True),
    ("Low-vol 2017", "2017-01-01", "2017-12-31", False),
    ("2018 Q4 selloff", "2018-10-01", "2018-12-31", True),
    ("Pre-COVID bull", "2019-01-01", "2020-02-18", False),
    ("COVID crash", "2020-02-19", "2020-03-23", True),
    ("COVID recovery", "2020-03-24", "2021-12-31", False),
    ("2022 bear", "2022-01-01", "2022-10-12", True),
    ("2023-24 bull", "2022-10-13", "2024-12-31", False),
    ("2025", "2025-01-01", "2025-12-31", False),
]

# Forbehold der skrives i JSON og på grafen, så resultaterne ikke overfortolkes.
_REGIME_LIMITATIONS = [
    "Survivorship bias: watchlisten er nutidens vindere; afnoterede tabere mangler.",
    "news_sentiment er 0 (inert) før ~2015 — let input-skift i de ældre regimer.",
    "Universet vokser over tid; tidlige år vælger top-1 fra et lille tværsnit.",
    "Samme model over hele historikken (ingen genoptræning) — ingen omkostninger.",
]


@dataclass
class BacktestResult:
    daily_log: pd.DataFrame  # se _simulate for kolonner (best/second/third/avg + equities)
    equities: Dict[str, pd.Series]  # nøgler "best"/"second"/"third"/"avg" → equity på handelsdato
    benchmark: pd.Series | None  # SPY buy & hold (samme index), None hvis ikke hentet
    final_equity: float          # slut-equity for "best" (bagudkompatibel hovedtal)
    total_return_pct: float      # samlet afkast for "best"
    params: dict | None = None   # hyperparametre der lavede modellen (lookback/lag/neuroner/...)
    run_id: str | None = None    # tidsstempel-id for kørslen (None for ældre/legacy logs)

    @property
    def equity(self) -> pd.Series:
        """Bedste-ticker-kurven (bagudkompatibel med tidligere enkelt-equity-felt)."""
        return self.equities["best"]


def _predict_symbol(
    model,
    scaler,
    device,
    n_features: int,
    seq_len: int,
    ohlcv: pd.DataFrame,
    year: int,
    end_year: int | None = None,
) -> tuple[pd.Series, pd.Series] | None:
    """Returnér (pred, actual) indekseret på handelsdato (t+1) for ét symbol.

    Med kun ``year`` dækkes det ene kalenderår (uændret adfærd). Gives ``end_year`` dækkes
    hele intervallet [year, end_year] — brugt af den lange regime-backtest.

    pred: modellens forudsagte open→close %.
    actual: realiseret open→close % samme dag (facit fra targets_next_day_open_to_close_pct).
    """
    y0 = year
    y1 = end_year if end_year is not None else year
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
        if not (y0 <= trade_date.year <= y1):
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
            out = model(xt).detach().cpu().numpy()
            preds.extend(float(v) for v in out.reshape(-1))

    idx = pd.DatetimeIndex(trade_dates)
    return (
        pd.Series(preds, index=idx),
        pd.Series(actuals, index=idx),
    )


def _fetch_spy_benchmark(year: int, trade_index: pd.DatetimeIndex) -> pd.Series | None:
    """Hent SPY-dagsbarer live via Alpaca som buy & hold-benchmark.

    Returnér en equity-kurve (start START_EQUITY) alignet på strategiens handelsdage,
    eller None hvis hverken cache eller live-data dækker året.

    Læser først den lokale SPY-cache (cache/SPY.parquet); falder tilbage til live Alpaca-IEX
    hvis cachen mangler/ikke dækker året (IEX har dog kun data fra ~2021).
    """
    cached = _spy_cached_close()
    if cached is not None:
        close = cached[cached.index.year == year]
        if not close.empty:
            norm = close / float(close.iloc[0]) * START_EQUITY
            logger.info("SPY-benchmark fra cache (%s barer i %s).", len(close), year)
            return norm.reindex(trade_index).ffill().bfill()

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


def _spy_buy_hold(
    start: date, end: date, trade_index: pd.DatetimeIndex
) -> tuple[pd.Series | None, pd.Timestamp | None, pd.Timestamp | None]:
    """SPY buy & hold over [start, end] som benchmark til regime-backtesten.

    Læser SPY fra den lokale cache (cache/SPY.parquet, dækker 2015→nu); falder tilbage til
    live Alpaca-IEX hvis cachen mangler. Normaliserer close til START_EQUITY og aligner på
    strategiens handelsdage. Returnér (equity, dækket_start, dækket_slut) — equity er None
    hvis hverken cache eller live-data dækker perioden. Cachen dækker fra 2015; ved endnu
    tidligere ``start`` dækker SPY kun en del, og de faktiske dæknings-datoer returneres med.
    """
    cached = _spy_cached_close()
    if cached is not None:
        close = cached[(cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))]
        if not close.empty:
            covered_start, covered_end = close.index[0], close.index[-1]
            norm = close / float(close.iloc[0]) * START_EQUITY
            equity = norm.reindex(trade_index).ffill()  # ingen bfill: pre-dæknings-dage = NaN
            return equity, covered_start, covered_end

    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        logger.warning("Springer SPY-benchmark over: ingen Alpaca-nøgler i .env.")
        return None, None, None
    try:
        client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        # Alpaca behandler `end` eksklusivt → hent en dag ekstra.
        spy = _fetch_symbol_range(client, _BENCHMARK_SYMBOL, start, end + timedelta(days=1))
    except Exception as exc:  # noqa: BLE001
        logger.warning("SPY-benchmark fejlede (%s) — fortsætter uden.", exc)
        return None, None, None
    if spy is None or spy.empty:
        logger.warning("Ingen SPY-data fra Alpaca for perioden — fortsætter uden benchmark.")
        return None, None, None

    close = spy["close"].astype(float).sort_index()
    close.index = pd.to_datetime(close.index)
    close = close[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
    if close.empty:
        return None, None, None
    covered_start, covered_end = close.index[0], close.index[-1]
    norm = close / float(close.iloc[0]) * START_EQUITY
    equity = norm.reindex(trade_index).ffill()  # ingen bfill: lad pre-dæknings-dage være NaN
    return equity, covered_start, covered_end


def _simulate(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Kør dag-for-dag-simuleringen for top-3 strategierne.

    For hver handelsdag rangeres dagens forudsigelser faldende, og op til tre picks
    med gyldigt facit (ikke-NaN actual) udvælges. Fire equity-kurver kompounderes fra
    START_EQUITY: bedste/næstbedste/tredjebedste pick samt et ligevægtet snit af de
    tilgængelige top-3. Mangler 2./3. pick en dag, føres den pågældende kurve uændret
    videre (ingen handel). Returnér (daily_log, equities).
    """
    equity = {"best": START_EQUITY, "second": START_EQUITY, "third": START_EQUITY, "avg": START_EQUITY}
    rows: list[dict] = []
    points: dict[str, list[float]] = {k: [] for k in equity}
    index: list[pd.Timestamp] = []

    for trade_date, pred_row in pred_df.iterrows():
        valid = pred_row.dropna().sort_values(ascending=False)
        if valid.empty:
            continue

        # Saml op til tre picks med gyldigt facit, i rangordnet rækkefølge.
        picks: list[tuple[str, float, float]] = []
        for sym, pred_val in valid.items():
            realized = actual_df.loc[trade_date, sym]
            if pd.isna(realized):
                continue
            picks.append((str(sym), float(pred_val), float(realized)))
            if len(picks) == 3:
                break
        if not picks:
            continue

        # Kompoundér hver kurve (2./3. føres videre hvis pick mangler).
        equity["best"] *= 1.0 + picks[0][2] / 100.0
        if len(picks) >= 2:
            equity["second"] *= 1.0 + picks[1][2] / 100.0
        if len(picks) >= 3:
            equity["third"] *= 1.0 + picks[2][2] / 100.0
        avg_realized = float(np.mean([p[2] for p in picks]))
        equity["avg"] *= 1.0 + avg_realized / 100.0

        def _slot(i: int) -> tuple[str, float, float]:
            return picks[i] if len(picks) > i else ("", float("nan"), float("nan"))

        (b_sym, b_pred, b_act) = _slot(0)
        (s_sym, s_pred, s_act) = _slot(1)
        (t_sym, t_pred, t_act) = _slot(2)
        rows.append(
            {
                "trade_date": trade_date.date().isoformat(),
                "best_symbol": b_sym,
                "best_pred": round(b_pred, 4),
                "best_actual": round(b_act, 4),
                "second_symbol": s_sym,
                "second_pred": round(s_pred, 4) if np.isfinite(s_pred) else s_pred,
                "second_actual": round(s_act, 4) if np.isfinite(s_act) else s_act,
                "third_symbol": t_sym,
                "third_pred": round(t_pred, 4) if np.isfinite(t_pred) else t_pred,
                "third_actual": round(t_act, 4) if np.isfinite(t_act) else t_act,
                "avg_actual": round(avg_realized, 4),
                "equity_best": round(equity["best"], 2),
                "equity_second": round(equity["second"], 2),
                "equity_third": round(equity["third"], 2),
                "equity_avg": round(equity["avg"], 2),
            }
        )
        index.append(trade_date)
        for k in equity:
            points[k].append(equity[k])

    daily_log = pd.DataFrame(rows)
    idx = pd.DatetimeIndex(index)
    equities = {k: pd.Series(points[k], index=idx) for k in equity}
    return daily_log, equities


def _read_model_params() -> dict:
    """Saml hyperparametrene der lavede modellen.

    Arkitekturen (lookback/lag/neuroner/dropout/n_features) læses fra selve checkpointet —
    det er facit for den trænede model. Trænings-only-parametre (LR, batch, …) snapshottes
    fra nuværende ``config`` (de gemmes ikke i checkpointet). Fejler checkpoint-læsning,
    falder vi tilbage til config-værdierne for arkitekturen også.
    """
    arch = {
        "lookback_days": config.SEQ_LEN,
        "layers": config.LSTM_LAYERS,
        "neurons": config.LSTM_HIDDEN,
        "dropout": config.DROPOUT,
        "n_features": config.N_FEATURES,
    }
    try:
        ckpt = torch.load(config.MODEL_PATH, map_location="cpu")
        arch.update(
            {
                "lookback_days": int(ckpt.get("seq_len", config.SEQ_LEN)),
                "layers": int(ckpt["layers"]),
                "neurons": int(ckpt["hidden"]),
                "dropout": float(ckpt["dropout"]),
                "n_features": int(ckpt["n_features"]),
            }
        )
    except (OSError, KeyError, RuntimeError) as exc:  # pragma: no cover - defensivt
        logger.warning("Kunne ikke læse arkitektur fra checkpoint (%s) — bruger config.", exc)

    # Trænings-only-parametre fra nuværende config (jf. brugervalg "read current config only").
    arch.update(
        {
            "lr": config.LR,
            "batch_size": config.BATCH_SIZE,
            "weight_decay": config.WEIGHT_DECAY,
            "huber_delta": config.HUBER_DELTA,
            "training_years": config.TRAINING_YEARS,
            "val_ratio": config.VAL_RATIO,
        }
    )
    return arch


def _summary_metrics(result: BacktestResult) -> dict:
    """Opsummér slut-afkast pr. strategi + retnings-træf til JSON-sidecar/visning."""
    metrics: dict = {}
    for key, series in result.equities.items():
        if series is None or series.empty:
            continue
        metrics[f"final_return_{key}_pct"] = round(
            (float(series.iloc[-1]) / START_EQUITY - 1.0) * 100.0, 2
        )
    stats = _direction_stats(result.daily_log)
    if stats is not None:
        metrics["direction_accuracy_pct"] = round(stats["dir_accuracy"], 2)
    return metrics


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
        lookback_calendar_days=366 + _LOOKBACK_CALENDAR_DAYS,
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
    daily_log, equities = _simulate(pred_df, actual_df)
    if daily_log.empty:
        raise RuntimeError("Simuleringen producerede ingen handler.")

    equity_series = equities["best"]

    # --- Buy & hold-benchmark: SPY hentet live via Alpaca (kun til backtest) ---
    benchmark = _fetch_spy_benchmark(year, equity_series.index)

    final_equity = float(equity_series.iloc[-1])
    total_return_pct = (final_equity / START_EQUITY - 1.0) * 100.0

    # --- Tidsstempel + hyperparametre for denne kørsel ---
    now = datetime.now()
    run_id = f"{year}_{now:%Y%m%d_%H%M%S}"
    timestamp_display = now.strftime("%d/%m/%y %H:%M")
    params = _read_model_params()

    # --- Gem CSV (hver kørsel som sin egen tidsstemplede fil — overskrives ikke) ---
    if output_path is None:
        output_path = _BACKTEST_DIR / f"backtest_{run_id}.csv"
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
        equities=equities,
        benchmark=benchmark,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        params=params,
        run_id=run_id,
    )

    # --- Gem JSON-sidecar med parametre + nøgletal ved siden af CSV'en ---
    meta = {
        "run_id": run_id,
        "timestamp_display": timestamp_display,
        "timestamp_iso": now.isoformat(timespec="seconds"),
        "year": year,
        "params": params,
        "metrics": _summary_metrics(result),
    }
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Parametre + nøgletal gemt til %s.", json_path)

    if show_plot:
        _plot(result, year, params=params)

    return result


def list_saved_backtests(year: int | None = None) -> list[dict]:
    """Find alle gemte backtests (nyeste først) til udvælgelse i --show-backtest.

    Scanner ``output/backtests/*.csv`` (hver med valgfri ``.json``-sidecar med parametre +
    nøgletal) samt evt. den ældre ``output/backtest_<år>.csv`` (uden parametre). Returnerer
    en liste af dicts: ``{run_id, timestamp_display, params, metrics, csv_path, json_path}``.
    Filtrér på ``year`` hvis angivet.
    """
    runs: list[dict] = []

    if _BACKTEST_DIR.is_dir():
        for csv_path in _BACKTEST_DIR.glob("backtest_*.csv"):
            json_path = csv_path.with_suffix(".json")
            meta: dict = {}
            if json_path.is_file():
                try:
                    meta = json.loads(json_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    meta = {}
            run_year = meta.get("year")
            if run_year is None:
                # Udled år fra filnavnet backtest_<år>_<stamp>.csv.
                parts = csv_path.stem.split("_")
                run_year = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            if year is not None and run_year != year:
                continue
            runs.append(
                {
                    "run_id": meta.get("run_id", csv_path.stem),
                    "timestamp_display": meta.get("timestamp_display", ""),
                    "year": run_year,
                    "params": meta.get("params"),
                    "metrics": meta.get("metrics"),
                    "csv_path": csv_path,
                    "json_path": json_path if json_path.is_file() else None,
                    "mtime": csv_path.stat().st_mtime,
                }
            )

    # Ældre enkelt-fil-logs uden parametre (bagudkompatibelt).
    legacy_years = [year] if year is not None else [2025]
    for y in legacy_years:
        legacy = _ROOT / "output" / f"backtest_{y}.csv"
        if legacy.is_file():
            runs.append(
                {
                    "run_id": legacy.stem,
                    "timestamp_display": "(legacy, ingen parametre)",
                    "year": y,
                    "params": None,
                    "metrics": None,
                    "csv_path": legacy,
                    "json_path": None,
                    "mtime": legacy.stat().st_mtime,
                }
            )

    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def plot_saved_backtest(
    year: int = 2025,
    *,
    output_path: Path | None = None,
    show_plot: bool = True,
) -> BacktestResult:
    """Genåbn grafen for en tidligere gemt backtest uden at køre simuleringen igen.

    Equity-kurven læses fra ``output_path`` (eller ældre ``output/backtest_<år>.csv``),
    parametrene fra en evt. ``.json``-sidecar ved siden af, og SPY buy & hold-benchmark
    hentes live via Alpaca på ny. Praktisk til at se en gammel backtest med dens parametre.
    """
    if output_path is None:
        output_path = _ROOT / "output" / f"backtest_{year}.csv"
    if not output_path.exists():
        raise FileNotFoundError(
            f"Ingen gemt backtest fundet: {output_path}. Kør --backtest for at lave en."
        )

    daily_log = pd.read_csv(output_path)
    if "trade_date" not in daily_log.columns:
        raise ValueError(
            f"{output_path} mangler kolonnen 'trade_date' — er det en backtest-log?"
        )

    idx = pd.DatetimeIndex(pd.to_datetime(daily_log["trade_date"]))
    # Nye logs har equity_best/second/third/avg; gamle logs har kun "equity" (= bedste).
    col_map = {
        "best": "equity_best",
        "second": "equity_second",
        "third": "equity_third",
        "avg": "equity_avg",
    }
    if "equity_best" in daily_log.columns:
        equities = {
            key: pd.Series(daily_log[col].astype(float).to_numpy(), index=idx)
            for key, col in col_map.items()
            if col in daily_log.columns
        }
    elif "equity" in daily_log.columns:
        equities = {"best": pd.Series(daily_log["equity"].astype(float).to_numpy(), index=idx)}
    else:
        raise ValueError(
            f"{output_path} mangler equity-kolonner (equity_best/.../equity) — er det en backtest-log?"
        )

    equity_series = equities["best"]
    if equity_series.empty:
        raise ValueError(f"{output_path} indeholder ingen handelsdage.")

    benchmark = _fetch_spy_benchmark(year, equity_series.index)
    final_equity = float(equity_series.iloc[-1])
    total_return_pct = (final_equity / START_EQUITY - 1.0) * 100.0

    # Parametre fra JSON-sidecar ved siden af CSV'en (hvis den findes).
    params: dict | None = None
    run_id: str | None = None
    json_path = output_path.with_suffix(".json")
    if json_path.is_file():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            params = meta.get("params")
            run_id = meta.get("run_id")
        except (OSError, ValueError):
            params = None

    result = BacktestResult(
        daily_log=daily_log,
        equities=equities,
        benchmark=benchmark,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        params=params,
        run_id=run_id,
    )
    logger.info(
        "Genindlæst backtest fra %s (%s handelsdage, slut %.2f USD, %+.2f%%).",
        output_path,
        len(equity_series),
        final_equity,
        total_return_pct,
    )
    if show_plot:
        _plot(result, year, params=params)
    return result


# Korte labels + facit-kolonne pr. strategi til stats-boksen.
_STRATEGY_STATS_COLS: dict[str, tuple[str, str]] = {
    "best": ("Bedste", "best_actual"),
    "second": ("Næstbedste", "second_actual"),
    "third": ("Tredjebedste", "third_actual"),
    "avg": ("Snit top 3", "avg_actual"),
}


def _direction_stats(daily_log: pd.DataFrame) -> dict | None:
    """Opsummér op-dag-chance og dagligt snit-afkast pr. strategi fra daily_log.

    For hver strategi (bedste/næstbedste/tredjebedste/snit af top 3) beregnes op-dag-
    chance (andel dage med positivt facit) og snit pr. dag i pct, kun over dage hvor
    strategien faktisk handlede (ikke-NaN facit). Returnerer None hvis loggen er tom
    eller mangler best_actual/best_pred. ``dir_accuracy`` er retnings-træf for bedste pick.
    """
    if daily_log is None or daily_log.empty:
        return None
    if "best_actual" not in daily_log.columns or "best_pred" not in daily_log.columns:
        return None

    per_strategy: list[dict] = []
    for key, (label, col) in _STRATEGY_STATS_COLS.items():
        if col not in daily_log.columns:
            continue
        series = pd.to_numeric(daily_log[col], errors="coerce").dropna()
        if series.empty:
            continue
        per_strategy.append(
            {
                "label": label,
                "win_rate": float((series > 0).mean()) * 100.0,  # op-dag-chance
                "avg_day": float(series.mean()),                 # snit pr. dag (handlede dage)
                "n": int(len(series)),
            }
        )

    # Retnings-træf for det bedste pick: andel dage hvor fortegnet på pred matchede facit.
    # Kontant-dage (best_actual=NaN ved konfidens-gate) udelades, så de ikke skævvrider tallet.
    best_actual = pd.to_numeric(daily_log["best_actual"], errors="coerce")
    best_pred = pd.to_numeric(daily_log["best_pred"], errors="coerce")
    mask = best_actual.notna() & best_pred.notna()
    if mask.any():
        dir_accuracy = float(((best_pred[mask] > 0) == (best_actual[mask] > 0)).mean()) * 100.0
    else:
        dir_accuracy = float("nan")
    return {
        "n": int(len(daily_log)),
        "dir_accuracy": dir_accuracy,
        "per_strategy": per_strategy,
    }


def _plot(result: BacktestResult, year: int, params: dict | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))

    title_parts: list[str] = []
    for key, label in _STRATEGY_LABELS.items():
        series = result.equities.get(key)
        if series is None or series.empty:
            continue
        ax.plot(series.index, series.values, label=label, **_STRATEGY_STYLE[key])
        final = float(series.iloc[-1])
        ret = (final / START_EQUITY - 1.0) * 100.0
        title_parts.append(f"{label}: {final:,.0f} USD ({ret:+.2f}%)")

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
        title_parts.append(
            f"SPY buy & hold: {float(result.benchmark.iloc[-1]):,.0f} USD ({bench_return:+.2f}%)"
        )
    title_prefix = f"Backtest {year}"
    if result.run_id:
        title_prefix += f" — kørsel {result.run_id}"
    ax.set_title(
        f"{title_prefix} — start {START_EQUITY:,.0f} USD\n" + "   |   ".join(title_parts),
        fontsize=9,
    )
    ax.set_xlabel("Handelsdato")
    ax.set_ylabel("Porteføljeværdi (USD)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    stats = _direction_stats(result.daily_log)
    if stats is not None:
        header = f"{'Strategi':<13}{'Op-dag':>8}{'Snit/dag':>10}"
        lines = [header, "-" * len(header)]
        for s in stats["per_strategy"]:
            lines.append(
                f"{s['label']:<13}{s['win_rate']:>7.1f}%{s['avg_day']:>+9.2f}%"
            )
        lines.append("")
        lines.append(f"Retnings-træf (bedste): {stats['dir_accuracy']:.1f}%")
        lines.append(f"Handelsdage: {stats['n']}")
        stats_text = "\n".join(lines)
        ax.text(
            0.015,
            0.985,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            zorder=5,
        )

    if params:
        param_lines = [
            "Parametre",
            "-" * 18,
            f"{'Lookback':<11}{params.get('lookback_days', '?'):>7}",
            f"{'Lag':<11}{params.get('layers', '?'):>7}",
            f"{'Neuroner':<11}{params.get('neurons', '?'):>7}",
            f"{'Dropout':<11}{params.get('dropout', '?'):>7}",
            f"{'LR':<11}{params.get('lr', '?'):>7}",
            f"{'Batch':<11}{params.get('batch_size', '?'):>7}",
        ]
        ax.text(
            0.985,
            0.985,
            "\n".join(param_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
            zorder=5,
        )

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


# Strategier der rapporteres i regime-tabellen (best = top-1 all-in; avg = ligevægt top-3).
_REGIME_STRATEGIES = ("best", "avg")
# Symbol-kolonne pr. strategi til turnover (avg er en kurv → ingen enkelt-kolonne).
_REGIME_SYMBOL_COLS = {"best": "best_symbol", "avg": None}


def _dated_symbols(daily_log: pd.DataFrame, col: str) -> pd.Series | None:
    """Symbol-serie indekseret på handelsdato (til turnover pr. regime)."""
    if col not in daily_log.columns or "trade_date" not in daily_log.columns:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(daily_log["trade_date"]))
    return pd.Series(daily_log[col].to_numpy(), index=idx)


def _regime_report(
    equities: Dict[str, pd.Series],
    daily_log: pd.DataFrame,
    start_year: int,
    end_year: int,
    variant: str = "baseline",
    strategies: tuple[str, ...] = _REGIME_STRATEGIES,
) -> list[dict]:
    """Beregn nøgletal pr. markedsregime OG pr. kalenderår for hver rapporteret strategi.

    For hvert vindue skæres equity-kurven til [start, slut], og ``metrics.summarize`` giver
    Sharpe/Sortino/Calmar/MaxDD/afkast/hit-rate (+ turnover for best). Vinduer med < 5
    handelsdage i kurven springes over (fx regimer uden for det valgte interval). ``variant``
    mærker rækkerne, og ``strategies`` vælger hvilke equity-nøgler der rapporteres.
    """
    windows: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    # Hele perioden først (window_type="overall") → samlet afkast/Sharpe/MaxDD pr. strategi.
    windows.append(("overall", "Full period", pd.Timestamp(f"{start_year}-01-01"),
                    pd.Timestamp(f"{end_year}-12-31")))
    for label, s, e, _stress in _REGIMES:
        windows.append(("regime", label, pd.Timestamp(s), pd.Timestamp(e)))
    for y in range(start_year, end_year + 1):
        windows.append(("year", str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")))

    records: list[dict] = []
    for kind, label, s, e in windows:
        for strat in strategies:
            eq = equities.get(strat)
            if eq is None or eq.empty:
                continue
            eq_slice = eq.loc[(eq.index >= s) & (eq.index <= e)]
            if len(eq_slice) < 5:
                continue
            symbols = None
            col = _REGIME_SYMBOL_COLS.get(strat)
            if col:
                ds = _dated_symbols(daily_log, col)
                if ds is not None:
                    symbols = ds.loc[(ds.index >= s) & (ds.index <= e)]
            summ = _metrics.summarize(eq_slice, symbols=symbols)
            records.append(
                {
                    "variant": variant,
                    "window_type": kind,
                    "window": label,
                    "strategy": strat,
                    "start": str(eq_slice.index[0].date()),
                    "end": str(eq_slice.index[-1].date()),
                    "n_days": int(len(eq_slice)),
                    **summ,
                }
            )
    return records


def run_regime_backtest(
    start_year: int = 2006,
    end_year: int = 2025,
    *,
    show_plot: bool = True,
    output_path: Path | None = None,
) -> dict:
    """Ét gennemløb af den NUVÆRENDE model over hele historikken → nøgletal pr. regime.

    Ingen genoptræning og ingen omkostninger: modellen indlæses én gang og scorer hver dag i
    [start_year, end_year] fra disk-cachen. ``_simulate`` kompounderer equity-kurverne, og
    ``_regime_report`` opsummerer Sharpe/MaxDD/afkast pr. markedsregime + pr. kalenderår.
    Hurtig (~2-4 t CPU) robusthedslæsning — se forbeholdene i ``_REGIME_LIMITATIONS``.
    """
    if end_year < start_year:
        raise ValueError("end_year skal være ≥ start_year.")
    model, scaler, device, n_features, seq_len = _load_bundle()
    logger.info(
        "Regime-backtest %s-%s (model seq_len=%s, n_features=%s).",
        start_year, end_year, seq_len, n_features,
    )

    # Hent fra cache med nok opvarmning til at dække start_year. +1 år fordi span-leddet
    # lander ved slutningen af start_year; _LOOKBACK_CALENDAR_DAYS dækker SEQ_LEN + warmup.
    lookback = (end_year - start_year + 1) * 366 + _LOOKBACK_CALENDAR_DAYS
    fetch_result = fetch_daily_bars(
        config.ALPACA_API_KEY,
        config.ALPACA_SECRET_KEY,
        config.WATCHLIST,
        end=date(end_year, 12, 31),
        lookback_calendar_days=lookback,
        extra_buffer_days=0,
        prefer_cache_only=True,
    )
    bars = fetch_result.bars
    if not bars:
        raise RuntimeError("Ingen barer fra cache — kør import/--train først.")
    logger.info("Indlæste %s symboler fra cache.", len(bars))

    pred_cols: Dict[str, pd.Series] = {}
    actual_cols: Dict[str, pd.Series] = {}
    for sym in config.WATCHLIST:
        ohlcv = bars.get(sym)
        if ohlcv is None or ohlcv.empty:
            continue
        res = _predict_symbol(model, scaler, device, n_features, seq_len, ohlcv, start_year, end_year)
        if res is None:
            continue
        pred_cols[sym], actual_cols[sym] = res
    if not pred_cols:
        raise RuntimeError("Ingen symboler gav forudsigelser — tjek cache-dækning for perioden.")

    pred_df = pd.DataFrame(pred_cols).sort_index()
    actual_df = pd.DataFrame(actual_cols).reindex(pred_df.index)

    # Min-symboler-pr-dag: drop dage hvor for få navne har gyldig forudsigelse (tyndt univers
    # i de tidlige år ville ellers give degenererede all-in-dage).
    min_syms = int(getattr(config, "MIN_SYMBOLS_PER_DAY", 10))
    valid_counts = pred_df.notna().sum(axis=1)
    keep = valid_counts >= min_syms
    dropped = int((~keep).sum())
    pred_df = pred_df.loc[keep]
    actual_df = actual_df.loc[keep]
    if pred_df.empty:
        raise RuntimeError(f"Ingen dage med ≥{min_syms} symboler — sænk MIN_SYMBOLS_PER_DAY eller start senere.")
    logger.info(
        "Forudsigelser: %s handelsdage × %s symboler (%s dage droppet pga <%s symboler).",
        pred_df.shape[0], pred_df.shape[1], dropped, min_syms,
    )

    daily_log, equities = _simulate(pred_df, actual_df)
    if daily_log.empty:
        raise RuntimeError("Simuleringen producerede ingen handler.")

    records = _regime_report(equities, daily_log, start_year, end_year)
    params = _read_model_params()

    # --- Hele perioden: samlet afkast for best/avg og SPY buy & hold ---
    def _overall_ret(strat: str):
        r = next((x for x in records if x["window_type"] == "overall"
                  and x["strategy"] == strat), None)
        return r["total_return_pct"] if r else None

    eq_index = equities["best"].index
    period_start, period_end = eq_index[0].date(), eq_index[-1].date()
    spy_equity, spy_cov_start, spy_cov_end = _spy_buy_hold(period_start, period_end, eq_index)
    spy_total = None
    if spy_equity is not None:
        sv = spy_equity.dropna()
        if len(sv) >= 2:
            spy_total = round((float(sv.iloc[-1]) / float(sv.iloc[0]) - 1.0) * 100.0, 2)

    overall = {
        "period_start": str(period_start),
        "period_end": str(period_end),
        "best_total_return_pct": _overall_ret("best"),
        "avg_total_return_pct": _overall_ret("avg"),
        "spy_total_return_pct": spy_total,
        "spy_covered_start": str(spy_cov_start.date()) if spy_cov_start is not None else None,
        "spy_covered_end": str(spy_cov_end.date()) if spy_cov_end is not None else None,
    }

    now = datetime.now()
    run_id = f"regime_{start_year}_{end_year}_{now:%Y%m%d_%H%M%S}"
    if output_path is None:
        output_path = _BACKTEST_DIR / f"{run_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    json_path = output_path.with_suffix(".json")
    out = {
        "run_id": run_id,
        "timestamp_iso": now.isoformat(timespec="seconds"),
        "start_year": start_year,
        "end_year": end_year,
        "min_symbols_per_day": min_syms,
        "overall": overall,
        "params": params,
        "limitations": _REGIME_LIMITATIONS,
        "regimes": records,
    }
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Regime-tabel gemt til %s og %s.", output_path, json_path)

    # --- Hovedtal for hele perioden ---
    def _fmt(v):
        return f"{v:+.1f}%" if isinstance(v, (int, float)) else "n/a"

    logger.info("=== Hele perioden %s..%s ===", period_start, period_end)
    logger.info("  Bedste (top-1):  %8s", _fmt(overall["best_total_return_pct"]))
    logger.info("  Snit top-3:      %8s", _fmt(overall["avg_total_return_pct"]))
    if spy_total is not None:
        cov = ""
        if spy_cov_start is not None and (spy_cov_start.date() > period_start or spy_cov_end.date() < period_end):
            cov = f"  (SPY dækker kun {spy_cov_start.date()}..{spy_cov_end.date()})"
        logger.info("  SPY buy & hold:  %8s%s", _fmt(spy_total), cov)
    else:
        logger.info("  SPY buy & hold:  %8s  (ingen Alpaca-data/nøgler)", "n/a")

    # Kort log-oversigt pr. regime (strategi=best): ret/Sharpe/MaxDD.
    _by = {r["window"]: r for r in records
           if r["window_type"] == "regime" and r["strategy"] == "best"}
    logger.info("--- Regime-nøgletal (best): ret/Sharpe/MaxDD ---")
    for label, _s, _e, _stress in _REGIMES:
        b = _by.get(label)
        if not b:
            continue
        logger.info(
            "%-22s %+6.1f / %5.2f / %6.1f",
            label, b["total_return_pct"], b["sharpe"], b["max_drawdown_pct"],
        )

    result = {"run_id": run_id, "equities": equities, "records": records, "params": params,
              "overall": overall, "benchmark": spy_equity,
              "csv_path": output_path, "json_path": json_path}
    if show_plot:
        _plot_regime(equities, records, start_year, end_year, run_id,
                     benchmark=spy_equity, overall=overall)
    return result


def _plot_regime(
    equities: Dict[str, pd.Series],
    records: list[dict],
    start_year: int,
    end_year: int,
    run_id: str,
    benchmark: pd.Series | None = None,
    overall: dict | None = None,
) -> None:
    """Log-skala equity over hele historikken med skraverede stress-regimer + nøgletals-boks.

    ``benchmark`` (SPY buy & hold) tegnes som grå stiplet linje, og ``overall`` (hovedtal for
    hele perioden) vises i titlen, så best/avg/SPY ses side om side.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 7))
    for key, label in _STRATEGY_LABELS.items():
        if key not in _REGIME_STRATEGIES:
            continue
        series = equities.get(key)
        if series is None or series.empty:
            continue
        ax.plot(series.index, series.values, label=label, **_STRATEGY_STYLE[key])

    # SPY buy & hold-benchmark (grå stiplet), hvis hentet.
    if benchmark is not None:
        bm = benchmark.dropna()
        if not bm.empty:
            ax.plot(bm.index, bm.values, label="SPY buy & hold", color="#888888",
                    linewidth=1.4, linestyle="--")

    # Skravér stress-regimer (kriser) der overlapper det viste interval.
    lo = pd.Timestamp(f"{start_year}-01-01")
    hi = pd.Timestamp(f"{end_year}-12-31")
    for label, s, e, stress in _REGIMES:
        if not stress:
            continue
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        if e_ts < lo or s_ts > hi:
            continue
        ax.axvspan(max(s_ts, lo), min(e_ts, hi), color="#d62728", alpha=0.10, zorder=0)

    ax.set_yscale("log")
    subtitle = f"kørsel {run_id}"
    totals = ""
    if overall:
        def _t(v):
            return f"{v:+.0f}%" if isinstance(v, (int, float)) else "n/a"
        totals = (f"\nHele perioden — Bedste: {_t(overall.get('best_total_return_pct'))}   "
                  f"Snit top-3: {_t(overall.get('avg_total_return_pct'))}   "
                  f"SPY: {_t(overall.get('spy_total_return_pct'))}")
    ax.set_title(
        f"Regime-backtest {start_year}-{end_year} — samme model, ingen genoptræning (log-skala)\n"
        f"{subtitle}{totals}",
        fontsize=10,
    )
    ax.set_xlabel("Handelsdato")
    ax.set_ylabel("Porteføljeværdi (USD, log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

    # Nøgletals-boks: best pr. regime — Sharpe/MaxDD.
    base = {r["window"]: r for r in records
            if r["window_type"] == "regime" and r["strategy"] == "best"}
    header = f"{'Regime':<20}{'Sharpe':>8}{'MaxDD':>8}"
    lines = [header, "-" * len(header)]
    for label, _s, _e, _stress in _REGIMES:
        b = base.get(label)
        if not b:
            continue
        lines.append(f"{label[:20]:<20}{b['sharpe']:>8.2f}{b['max_drawdown_pct']:>7.1f}%")
    ax.text(
        0.995, 0.02, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.0, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9), zorder=5,
    )
    # Forbehold nederst-venstre.
    ax.text(
        0.005, 0.02, "Forbehold:\n" + "\n".join("• " + l for l in _REGIME_LIMITATIONS),
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="#555555",
        bbox=dict(boxstyle="round", facecolor="#fff8e1", edgecolor="#e0c060", alpha=0.9), zorder=5,
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
