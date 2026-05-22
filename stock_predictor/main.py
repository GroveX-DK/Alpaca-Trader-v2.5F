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

from stock_predictor.predict import predict_rankings  # noqa: E402
from stock_predictor.train import train_model  # noqa: E402
from stock_predictor.trader import rotate_to_symbol  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s %(message)s")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
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
    p.add_argument("--verbose", "-v", action="store_true", help="Udvid debug-log.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    if args.train:
        train_model()
        return 0

    best_sym, score, ranking = predict_rankings()
    logger = logging.getLogger(__name__)
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
