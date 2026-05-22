"""
market_data_fetcher.py
──────────────────────
Henter fuld historisk OHLCV-data for:
  • Alle NYSE-aktier  (hentet via NASDAQ FTP-liste)
  • Alle DAX 40-aktier
  • Råvarer: Olie (Brent + WTI), Guld, Sølv, Kobber

For hver række beregnes:
  • daily_return       – dagligt log-afkast
  • rolling_vol_20d    – 20-dages rullende annualiseret volatilitet (252 handelsdage)
  • rolling_vol_60d    – 60-dages rullende annualiseret volatilitet
  • parkinson_vol      – Parkinson range-baseret volatilitet (High/Low)
  • atr_14             – Average True Range over 14 dage

Output-struktur (én CSV pr. ticker):
  output/
    NYSE/
      AAPL.csv
      MSFT.csv
      ...
    DAX40/
      SAP.csv
      SIE.csv
      ...
    Commodities/
      Gold.csv
      Silver.csv
      ...

Allerede downloadede filer springes automatisk over ved genkørsel.

Kørsel:
  pip install yfinance pandas numpy pyarrow requests tqdm
  python market_data_fetcher.py
"""

import math
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from pathlib import Path

# ──────────────────────────────────────────────
# KONFIGURATION
# ──────────────────────────────────────────────
OUTPUT_DIR    = Path("output")   # undermapper: NYSE/, DAX40/, Commodities/
SKIP_EXISTING = True             # spring over hvis filen allerede findes

START_DATE    = "1980-01-01"     # yfinance returnerer kun det der findes
BATCH_SIZE    = 50               # antal tickers pr. yf.download()-kald
SLEEP_BETWEEN = 1.0              # sekunder mellem batches (undgå rate-limit)
MAX_RETRIES   = 3
ROLLING_SHORT = 20               # handelsdage
ROLLING_LONG  = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# TICKER-LISTER
# ──────────────────────────────────────────────

WATCHLIST = [
    "AAPL", "NVDA", "TSLA", "MSFT", "AMD", "META", "GOOGL", "AMZN",
    "BAC", "COST", "CRM", "CVX", "GS", "HD", "INTC", "JNJ", "JPM",
    "KO", "LMT", "MA", "NEE", "PFE", "PG", "UNH", "V", "WMT", "XOM",
    "ABBV", "AMT", "ABBNY", "AMX", "ASML", "AZN", "BABA", "BP", "BHP",
    "BUD", "CHKP", "CNI", "EQNR", "ERIC", "FMX", "HDB", "HSBC", "INFY",
    "ITUB", "MUFG", "NICE", "NOK", "NVO", "NVS", "PBR", "RIO", "RY",
    "SAN", "SAP", "SHEL", "SHOP", "SONY", "SU", "TD", "TM", "UL",
    "VALE", "WIT", "TSM", "SQM", "TCEHY", "SFTBY", "ATLKY", "CABGY",
    "AMKBY", "VLVLY", "WEGZY", "DBSDY", "SINGY", "SGAPY",
]







# ──────────────────────────────────────────────
# VOLATILITETS-BEREGNING
# ──────────────────────────────────────────────

def add_volatility_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tilføjer volatilitets- og momentum-kolonner til et OHLCV-DataFrame
    med MultiIndex-kolonner (metric, ticker) fra yf.download().
    Returnerer et 'langt' DataFrame med én række pr. (dato, ticker).
    """
    # Lav om til langt format: én række pr. (Date, Ticker)
    df = df.stack(level=1, future_stack=True).rename_axis(["Date", "Ticker"]).reset_index()

    # Sørg for korrekte kolonnenavne (yfinance kan variere)
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # Dagligt log-afkast pr. ticker
    df = df.sort_values(["Ticker", "Date"])
    df["daily_return"] = (
        df.groupby("Ticker")["Close"]
          .transform(lambda s: np.log(s / s.shift(1)))
    )

    # Rullende annualiseret volatilitet (std af log-afkast × √252)
    for window, col in [(ROLLING_SHORT, f"rolling_vol_{ROLLING_SHORT}d"),
                        (ROLLING_LONG,  f"rolling_vol_{ROLLING_LONG}d")]:
        df[col] = (
            df.groupby("Ticker")["daily_return"]
              .transform(lambda s, w=window: s.rolling(w, min_periods=max(2, w//2)).std() * math.sqrt(252))
        )

    # Parkinson volatilitet: bruger High/Low – mere præcis end close-to-close
    # σ² = 1/(4·n·ln2) · Σ(ln(H/L))²
    df["_log_hl_sq"] = np.log(df["High"] / df["Low"]) ** 2
    df["parkinson_vol"] = (
        df.groupby("Ticker")["_log_hl_sq"]
          .transform(lambda s: np.sqrt(s.rolling(ROLLING_SHORT, min_periods=5).mean() / (4 * math.log(2))) * math.sqrt(252))
    )
    df.drop(columns=["_log_hl_sq"], inplace=True)

    df["_prev_close"] = df.groupby("Ticker")["Close"].shift(1)
    df["_tr"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["_prev_close"]).abs(),
        (df["Low"]  - df["_prev_close"]).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = (
        df.groupby("Ticker")["_tr"]
          .transform(lambda s: s.rolling(14, min_periods=1).mean())
    )
    df.drop(columns=["_prev_close", "_tr"], inplace=True)

    return df


# ──────────────────────────────────────────────
# DATA-HENTNING
# ──────────────────────────────────────────────

def fetch_batch(tickers: list[str], label: str) -> pd.DataFrame | None:
    """Henter OHLCV for en liste af tickers med retry-logik."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                tickers,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                log.warning("[%s] Tom respons – springer over", label)
                return None
            return raw
        except Exception as e:
            log.warning("[%s] Forsøg %d/%d fejlede: %s", label, attempt, MAX_RETRIES, e)
            time.sleep(2 ** attempt)
    return None


def safe_filename(ticker: str) -> str:
    """Gør ticker-symbol sikkert som filnavn (fjerner tegn Windows ikke kan lide)."""
    return ticker.replace("/", "-").replace("=", "_").replace("*", "_")


def save_ticker_csv(df: pd.DataFrame, ticker: str, group_dir: Path) -> Path:
    """Gemmer én tickers data som CSV i group_dir/TICKER.csv"""
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / f"{safe_filename(ticker)}.csv"
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


def fetch_and_save_group(
    tickers: list[str],
    group_name: str,
    ticker_name_map: dict[str, str] | None = None,
) -> dict[str, int]:
    """
    Henter data for en gruppe i batches og gemmer én CSV pr. ticker.
    ticker_name_map bruges til at omdøbe yfinance-symboler (fx råvarer).
    Returnerer stats: {"saved": n, "skipped": n, "empty": n}
    """
    group_dir = OUTPUT_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    stats = {"saved": 0, "skipped": 0, "empty": 0}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    log.info("━━━  %s  –  %d tickers  –  %d batches  ━━━",
             group_name, len(tickers), len(batches))

    for batch in tqdm(batches, desc=group_name, unit="batch"):

        # Spring over hvis ALLE tickers i batchen allerede er downloadet
        if SKIP_EXISTING:
            pending = [
                t for t in batch
                if not (group_dir / f"{safe_filename((ticker_name_map or {}).get(t, t))}.csv").exists()
            ]
            if not pending:
                stats["skipped"] += len(batch)
                continue
        else:
            pending = batch

        raw = fetch_batch(pending, group_name)
        if raw is None:
            stats["empty"] += len(pending)
            time.sleep(SLEEP_BETWEEN)
            continue

        # Sikrer MultiIndex selv ved enkelt ticker
        if not isinstance(raw.columns, pd.MultiIndex):
            raw.columns = pd.MultiIndex.from_product([raw.columns, pending[:1]])

        processed = add_volatility_columns(raw)
        processed["Date"] = pd.to_datetime(processed["Date"])

        # Gem én fil pr. ticker
        for ticker, ticker_df in processed.groupby("Ticker"):
            # Omdøb symbol til læsbart navn hvis map er givet (fx "GC=F" → "Gold")
            display_name = ticker_name_map.get(ticker, ticker) if ticker_name_map else ticker
            ticker_df = ticker_df.drop(columns=["Ticker"])
            ticker_df.insert(0, "Ticker", display_name)

            if SKIP_EXISTING:
                out_path = group_dir / f"{safe_filename(display_name)}.csv"
                if out_path.exists():
                    stats["skipped"] += 1
                    continue

            save_ticker_csv(ticker_df, display_name, group_dir)
            stats["saved"] += 1

        time.sleep(SLEEP_BETWEEN)

    return stats


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    total_stats: dict[str, dict] = {}

    # ── Watchlist ─────────────────────────────
    total_stats["Watchlist"] = fetch_and_save_group(WATCHLIST, "Watchlist")

    # ── Opsummering ───────────────────────────
    print("\n" + "="*55)
    print("  RESULTAT")
    print("="*55)
    print(f"  {'Gruppe':<15} {'Gemt':>8} {'Sprunget over':>14} {'Tom':>6}")
    print("  " + "-"*51)
    for grp, s in total_stats.items():
        print(f"  {grp:<15} {s['saved']:>8} {s['skipped']:>14} {s['empty']:>6}")
    print("="*55)

    # Vis mappestruktur
    print("\n  Output-struktur:")
    for grp_dir in sorted(OUTPUT_DIR.iterdir()):
        if grp_dir.is_dir():
            n = len(list(grp_dir.glob("*.csv")))
            print(f"    output/{grp_dir.name}/   ({n} CSV-filer)")
    print()
    log.info("✓ Færdig!")


if __name__ == "__main__":
    main()