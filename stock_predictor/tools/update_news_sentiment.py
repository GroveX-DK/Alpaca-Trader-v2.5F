"""Backfill/vedligehold nyheds-sentiment i Parquet-cachen via finBERT på Alpaca News.

Henter historiske nyheder pr. ticker (Alpaca, tilbage til config.NEWS_SENTIMENT_HISTORY_START),
scorer dem med finBERT (ProsusAI/finbert) og materialiserer kolonnen ``news_sentiment`` i
OHLCV-cachen. Råe artikler arkiveres i cache/news/<TICKER>.parquet, så de kan gen-scores uden
re-fetch (--rescore). Kald er inkrementelle/resumable (henter kun siden sidste arkiverede dato).

Bemærk: den daglige drift (--run/--train) opdaterer selv nyere sentiment automatisk; dette
værktøj er til den tunge éngangs-backfill og bulk-gen-scoring.

Kør:
    python -m stock_predictor.tools.update_news_sentiment
    python -m stock_predictor.tools.update_news_sentiment --symbols-only AAPL MSFT
    python -m stock_predictor.tools.update_news_sentiment --max-tickers 3 -v   # smoke
    python -m stock_predictor.tools.update_news_sentiment --since 2020-01-01
    python -m stock_predictor.tools.update_news_sentiment --rescore            # ingen fetch
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # Windows-konsol (cp1252) kan ikke alle tegn; tving UTF-8.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from stock_predictor import config  # noqa: E402
from stock_predictor.data_fetcher import _cache_path  # noqa: E402
from stock_predictor import news_sentiment as ns  # noqa: E402

logger = logging.getLogger(__name__)


def _cache_start(path: Path, floor: date) -> date:
    """Tidligste dato vi vil hente nyheder fra: max(historik-gulv, cachens første bar)."""
    try:
        df = pd.read_parquet(path, columns=None)
    except Exception:  # noqa: BLE001
        return floor
    if df.empty:
        return floor
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index)
    cmin = pd.to_datetime(df.index).min()
    if pd.isna(cmin):
        return floor
    return max(floor, cmin.date())


def _classified_count(sym: str) -> int | None:
    """Antal scorede artikler hvis tickeren allerede er fuldt klassificeret, ellers None.

    Arkivet skrives kun efter at finBERT har scoret alt i én batch, så et ikke-tomt arkiv
    uden u-scorede rækker betyder "færdig". Tom/manglende arkiv eller u-scorede rækker → None
    (skal behandles).
    """
    a = ns.read_archive(sym)
    if a.empty:
        return None
    if a["score"].isna().any():
        return None
    return int(a["score"].notna().sum())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill nyheds-sentiment i Parquet-cachen.")
    p.add_argument("--symbols-only", nargs="*", help="Kun disse tickere (default: hele watchlisten).")
    p.add_argument("--max-tickers", type=int, default=0, help="Begræns antal (smoke-test).")
    p.add_argument("--since", default=None, help="Tving historik-start (YYYY-MM-DD).")
    p.add_argument("--rescore", action="store_true", help="Gen-scor arkiver med finBERT uden at hente nyheder.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Behandl alle tickere igen, også dem der allerede er klassificeret (gap-fill).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    # Dæmp støjende tredjeparts-loggere (HTTP-kald, model-download) så fremdrift kan læses.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.symbols_only:
        symbols = [s.strip().upper() for s in args.symbols_only if s.strip()]
    else:
        symbols = [s.strip().upper() for s in config.WATCHLIST]
    if args.max_tickers and args.max_tickers > 0:
        symbols = symbols[: args.max_tickers]

    floor = config.NEWS_SENTIMENT_HISTORY_START
    if args.since:
        try:
            floor = datetime.strptime(args.since, "%Y-%m-%d").date()
        except ValueError:
            print(f"Ugyldig --since dato: {args.since!r} (forventer YYYY-MM-DD)", flush=True)
            return 2

    end = date.today()

    # Nøgler kræves kun ved fetch (ikke ved --rescore); fejl tidligt hvis de mangler.
    if not args.rescore and not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        print("Manglende ALPACA_API_KEY/ALPACA_SECRET_KEY i .env — kan ikke hente nyheder.", flush=True)
        return 2

    skip_classified = not (args.rescore or args.force)
    mode = "GEN-SCORING (uden fetch)" if args.rescore else f"backfill fra {floor} til {end}"
    extra = " (springer allerede klassificerede over)" if skip_classified else ""
    print(f"Nyheds-sentiment: {mode} for {len(symbols)} ticker(e){extra}.", flush=True)

    client = None  # NewsClient oprettes dovent ved første ticker der skal hentes

    ok = 0
    skipped = 0
    failed: list[str] = []
    for i, sym in enumerate(symbols, 1):
        ohlcv_path = _cache_path(config.CACHE_DIR, sym)
        try:
            if skip_classified:
                done = _classified_count(sym)
                if done is not None:
                    skipped += 1
                    print(f"[{i}/{len(symbols)}] SKIP {sym}: allerede klassificeret ({done} artikler)", flush=True)
                    continue
            if args.rescore:
                series = ns.rescore_archive(sym, ohlcv_path)
            else:
                if client is None:
                    from alpaca.data.historical.news import NewsClient

                    client = NewsClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
                start = _cache_start(ohlcv_path, floor)
                series = ns.ensure_sentiment_current(client, sym, ohlcv_path, start=start, end=end)
            n_days = 0 if series is None else int(series.notna().sum())
            ok += 1
            print(f"[{i}/{len(symbols)}] OK {sym}: {n_days} dage med sentiment", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(sym)
            print(f"[{i}/{len(symbols)}] FEJL {sym}: {exc}", flush=True)

    print(f"\nFærdig: {ok} behandlet, {skipped} sprunget over, {len(failed)} fejlede (af {len(symbols)}).", flush=True)
    if failed:
        print(f"Mislykkedes ({len(failed)}): {', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
