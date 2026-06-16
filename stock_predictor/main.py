"""Orchestrér daglig træning (--train) eller forudsigelse + paper-handel (--run).

`--run` kalder inferens, som henter OHLCV via fetch_daily_bars for watchlisten
(inkl. Watchlist-metriker + vol_annual_pct). Cache-first: hvis disk-cache dækker til
sidste afsluttede handelsdag, bruges kun cache; ellers inkrementel tail/backfill fra Alpaca.

Lang historik fra CSV: kør `python -m stock_predictor.tools.import_watchlist_csv_to_cache`
før træning for at fylde cache fra output/Watchlist/*.csv.

Efter ændring af træningsmål (fx open→close-label): kør altid `--train` før `--run`,
så checkpoint og scaler matcher det nye mål.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stock_predictor import config  # noqa: E402
from stock_predictor.backtest import (  # noqa: E402
    list_saved_backtests,
    plot_saved_backtest,
    run_backtest,
    run_regime_backtest,
)
from stock_predictor.predict import predict_rankings  # noqa: E402
from stock_predictor.train import train_model  # noqa: E402
from stock_predictor.trader import rotate_to_symbol  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s %(message)s")


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Alpaca daglig predictor + paper trading.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--train",
        action="store_true",
        help=(
            "Træn LSTM-modellen (gem scaler + checkpoints). Bruger disk-cache når den "
            "er opdateret til sidste handelsdag — offline uden API-nøgler. Ellers hentes "
            "manglende symboler fra Alpaca. Valgfrit: importér CSV med "
            "python -m stock_predictor.tools.import_watchlist_csv_to_cache først."
        ),
    )
    grp.add_argument(
        "--run",
        action="store_true",
        help=(
            "Inferens på nyeste vindue + paper-handel. Henter barrer pr. symbol "
            "(med disk-cache: inkrementel tail/backfill mod Alpaca, ikke fuld re-download). "
            "Lukker gammel papirposition og køber stærkeste ticker."
        ),
    )
    grp.add_argument(
        "--backtest",
        action="store_true",
        help=(
            "Backtest strategien over kalenderåret 2025 fra disk-cache (offline). "
            "Simulerer dag-for-dag, gemmer output/backtest_2025.csv og viser en pop op-graf "
            "med equity-kurve (start 100.000) + buy & hold-benchmark."
        ),
    )
    grp.add_argument(
        "--regime-backtest",
        type=int,
        metavar="START_ÅR",
        nargs="?",
        const=2006,
        help=(
            "Kør den NUVÆRENDE model én gang over hele historikken (default fra 2006) og "
            "rapportér Sharpe/MaxDD/afkast pr. markedsregime (GFC, COVID-krak, 2022-bjørn, …) "
            "+ pr. kalenderår. Ingen genoptræning, fra disk-cache (~2-4 t CPU). Gemmer "
            "output/backtests/regime_*.csv/.json og viser en log-skala graf. Angiv evt. et "
            "tidligere START_ÅR (ned til ~1994; køretid stiger med historikken)."
        ),
    )
    grp.add_argument(
        "--show-backtest",
        type=int,
        metavar="ÅR",
        nargs="?",
        const=-1,
        help=(
            "Vis en liste over gemte backtests (output/backtests/) med deres parametre, "
            "vælg én, og genåbn dens graf uden at køre simuleringen igen. SPY buy & hold-"
            "benchmark hentes live på ny. Angiv valgfrit et ÅR for kun at vise det års kørsler."
        ),
    )
    grp.add_argument(
        "--walk-forward",
        action="store_true",
        help=(
            "Walk-forward genoptræning (Option 4C): genoptræn modellen pr. år på et "
            "ekspanderende vindue og scor hvert år out-of-sample, stitch til én equity-kurve. "
            "Kræver WALK_FORWARD_ENABLED=True. TUNG (~mange træninger) — kør helst på GPU. "
            "Gemmer output/backtests/walkforward_*.csv/.json og viser graf."
        ),
    )
    grp.add_argument(
        "--update-sentiment",
        action="store_true",
        help=(
            "Backfill/vedligehold nyheds-sentiment (finBERT på Alpaca News) i cachen. "
            "Tung éngangs-historik; normal --run/--train opdaterer selv nyere sentiment. "
            "Videresender øvrige argumenter til tools.update_news_sentiment "
            "(fx --max-tickers, --since, --rescore)."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Udvid debug-log.")
    return p.parse_known_args(argv)


def _format_run_row(idx: int, run: dict) -> str:
    """Én linje i valg-listen: nummer, tidsstempel, nøgleparametre + bedste afkast."""
    params = run.get("params") or {}
    metrics = run.get("metrics") or {}
    if params:
        p = (
            f"lookback={params.get('lookback_days', '?')} "
            f"lag={params.get('layers', '?')} "
            f"neuroner={params.get('neurons', '?')} "
            f"dropout={params.get('dropout', '?')}"
        )
    else:
        p = "(ingen parametre gemt)"
    ret = metrics.get("final_return_best_pct")
    ret_str = f"{ret:+.2f}%" if isinstance(ret, (int, float)) else "?"
    ts = run.get("timestamp_display") or run.get("run_id", "")
    return f"  [{idx}] {ts:<18} {p}   bedste: {ret_str}"


def _show_backtest(year_filter: int | None) -> int:
    """List gemte backtests, lad brugeren vælge én, og vis dens graf med parametre."""
    runs = list_saved_backtests(year_filter)
    if not runs:
        scope = f" for {year_filter}" if year_filter is not None else ""
        print(f"Ingen gemte backtests fundet{scope}. Kør --backtest for at lave en.")
        return 1

    if len(runs) == 1:
        chosen = runs[0]
    else:
        print("Gemte backtests (nyeste først):")
        for i, run in enumerate(runs, start=1):
            print(_format_run_row(i, run))
        try:
            raw = input(f"Vælg backtest [1-{len(runs)}]: ").strip()
        except EOFError:
            raw = ""
        if not raw.isdigit() or not (1 <= int(raw) <= len(runs)):
            print("Ugyldigt valg.")
            return 1
        chosen = runs[int(raw) - 1]

    plot_saved_backtest(
        year=chosen.get("year") or 2025,
        output_path=chosen["csv_path"],
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args, extra = _parse_args(argv)
    _configure_logging(args.verbose)

    if args.update_sentiment:
        from stock_predictor.tools.update_news_sentiment import main as update_sentiment_main

        forwarded = list(extra)
        if args.verbose and "-v" not in forwarded and "--verbose" not in forwarded:
            forwarded.append("-v")
        return update_sentiment_main(forwarded)

    if args.train:
        train_model()
        return 0

    if args.walk_forward:
        from stock_predictor.train import train_model_walk_forward

        train_model_walk_forward()
        return 0

    if args.backtest:
        run_backtest()
        return 0

    if args.regime_backtest is not None:
        run_regime_backtest(start_year=args.regime_backtest)
        return 0

    if args.show_backtest is not None:
        year_filter = None if args.show_backtest == -1 else args.show_backtest
        return _show_backtest(year_filter)

    logger = logging.getLogger(__name__)

    # Long/short markeds-neutral live-bog (Option 2) når slået til; ellers enkelt-symbol-rotation.
    if bool(getattr(config, "LONG_SHORT_LIVE_ENABLED", False)):
        from stock_predictor.config import LONG_SHORT_BOTTOM_K, LONG_SHORT_TOP_K
        from stock_predictor.predict import predict_rankings_detailed
        from stock_predictor.trader import rebalance_long_short

        ranked = predict_rankings_detailed()  # [(sym, score, band)] faldende
        if len(ranked) < LONG_SHORT_TOP_K + LONG_SHORT_BOTTOM_K:
            logger.error(
                "For få symboler (%s) til long/short (kræver top-%s + bottom-%s).",
                len(ranked), LONG_SHORT_TOP_K, LONG_SHORT_BOTTOM_K,
            )
            return 1
        longs = [(s, sc) for s, sc, _b in ranked[:LONG_SHORT_TOP_K]]
        shorts = [(s, sc) for s, sc, _b in ranked[-LONG_SHORT_BOTTOM_K:]]
        logger.info(
            "Long/short: long %s | short %s",
            [s for s, _ in longs], [s for s, _ in shorts],
        )
        rebalance_long_short(longs, shorts)
        return 0

    # Retningsbestemt enkelt-navn (bedste |bevægelse|: long ved op, short ved ned) når slået til.
    if bool(getattr(config, "DIRECTIONAL_LIVE_ENABLED", False)):
        from stock_predictor.predict import predict_best_directional

        best_sym, score, _band, side = predict_best_directional()
        logger.info(
            "Retningsbestemt valg: %s %s (forudsagt open→close %+f pct, største |bevægelse|).",
            best_sym,
            side.upper(),
            score,
        )
        rotate_to_symbol(best_sym, float(score), side=side)
        return 0

    best_sym, score, ranking = predict_rankings()
    logger.info(
        "Bedste ticker: %s (forudsagt open→close %+f pct)",
        best_sym,
        score,
    )
    for sym, s in ranking:
        logger.debug("  %s %+f", sym, s)
    rotate_to_symbol(best_sym, float(score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
