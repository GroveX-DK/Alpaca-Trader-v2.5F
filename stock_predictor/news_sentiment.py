"""Nyheds-sentiment pr. aktie-pr. dag via finBERT på Alpaca News.

Pipeline:
  Alpaca News (headline + summary, tilbage til 2015)  →  finBERT (ProsusAI/finbert,
  positiv/neutral/negativ)  →  score = p_pos − p_neg ∈ [-1, 1]  →  daglig middelværdi
  pr. ticker  →  kolonnen ``news_sentiment`` i OHLCV-cachen (samme mønster som vix_close).

To lag persistens:
  * Råt artikel-arkiv ``cache/news/<TICKER>.parquet`` (id, date, headline, summary, score)
    så værdier kan *gen-scores* uden at hente nyheder igen.
  * Den aggregerede dagsværdi materialiseres ind i OHLCV-cachen som ``news_sentiment``.

Dage uden nyheder/ukendt historik fyldes neutralt (0.0) i feature-laget (se
``feature_engineer._feature_frame``), så rækker aldrig droppes pga. manglende sentiment.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from stock_predictor import config
from stock_predictor.data_fetcher import _atomic_save_parquet, _cache_path
from stock_predictor.feature_engineer import FEATURE_COLUMNS, build_dataset_frame

logger = logging.getLogger(__name__)

_ARCHIVE_COLUMNS = ("id", "date", "headline", "summary", "score")
# Feature-kolonner der genberegnes i build_dataset_frame (bæres ikke med som rå base).
_DERIVED_FEATURE_COLUMNS = set(FEATURE_COLUMNS) - {"vix_close", "news_sentiment"}

# --- finBERT (lazy-loaded) --------------------------------------------------
_FINBERT = None  # (tokenizer, model, device, label_idx) cache


def _is_model_cached(name: str) -> bool:
    """True hvis HF-modellen allerede ligger i den lokale hub-cache (ren filtjek, ingen import)."""
    import os
    from pathlib import Path

    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snap = hub / ("models--" + name.replace("/", "--")) / "snapshots"
    return snap.is_dir() and any(snap.iterdir())


def _load_finbert():
    """Indlæs ProsusAI/finbert én gang. Importerer transformers dovent."""
    global _FINBERT
    if _FINBERT is not None:
        return _FINBERT

    import os

    name = config.FINBERT_MODEL_NAME
    cached = _is_model_cached(name)
    if cached:
        # Tving fuldt offline FØR transformers/huggingface_hub importeres, så der ikke laves
        # netværks-round-trips (ETag-validering, safetensors-probe) ved load fra cache.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        logger.info("Indlæser finBERT (%s) fra lokal cache (offline).", name)
    else:
        logger.info("finBERT (%s) ikke i cache — downloader ~440MB (engangs).", name)

    import torch  # lokal import — kun nødvendig når der faktisk scores
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from stock_predictor.torch_device import resolve_device

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    device = resolve_device(str(getattr(config, "FINBERT_DEVICE", "auto")))
    model.to(device)
    model.eval()

    # Map model-labels → index for positiv/negativ uafhængigt af rækkefølge.
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    label_idx = {v: k for k, v in id2label.items()}
    if "positive" not in label_idx or "negative" not in label_idx:
        raise RuntimeError(f"Uventede finBERT-labels: {id2label}")

    _FINBERT = (tokenizer, model, device, label_idx)
    return _FINBERT


def finbert_scores(
    texts: list[str],
    *,
    progress_label: str | None = None,
    progress_every: int = 2000,
) -> list[float]:
    """Returnér p_pos − p_neg ∈ [-1, 1] pr. tekst (batchet finBERT-inferens).

    Med progress_label logges en heartbeat ("scoret X/Y") for hver progress_every tekster,
    så lange tickere (fx AAPL ~30k artikler) ses som aktive frem for "frosne".
    """
    if not texts:
        return []

    import torch

    tokenizer, model, device, label_idx = _load_finbert()
    pos_i, neg_i = label_idx["positive"], label_idx["negative"]
    batch = int(getattr(config, "FINBERT_BATCH_SIZE", 32))
    total = len(texts)
    next_mark = progress_every
    out: list[float] = []
    for start in range(0, total, batch):
        chunk = [t if isinstance(t, str) and t.strip() else " " for t in texts[start : start + batch]]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        out.extend((probs[:, pos_i] - probs[:, neg_i]).astype(float).tolist())
        if progress_label and len(out) >= next_mark:
            logger.info("%s: scoret %d/%d artikler", progress_label, len(out), total)
            next_mark += progress_every
    return out


# --- Alpaca News-hentning ---------------------------------------------------
def _to_dt(d: date | datetime | pd.Timestamp) -> datetime:
    ts = pd.Timestamp(d)
    return datetime(ts.year, ts.month, ts.day)


def _article_date(created_at) -> pd.Timestamp:
    """Artiklens handelsdag: UTC-tidsstempel → US/Eastern kalenderdag (normaliseret)."""
    ts = pd.Timestamp(created_at)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/New_York").normalize().tz_localize(None)


# Alpaca-py auto-paginerer ét get_news-kald (når limit udelades), så vi henter i
# afgrænsede dato-bidder for at holde hukommelse/varighed nede ved lange backfills.
_FETCH_CHUNK_DAYS = 90


def fetch_news(client, symbol: str, start: date, end: date) -> pd.DataFrame:
    """Hent Alpaca-nyheder for ét symbol i [start, end]. Kolonner: id,date,headline,summary.

    Itererer i 90-dages bidder; alpaca-py auto-paginerer internt inden for hver bid.
    """
    from alpaca.data.requests import NewsRequest

    rows: list[dict] = []
    chunk_start = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    step = pd.Timedelta(days=_FETCH_CHUNK_DAYS)
    logger.info("%s: henter nyheder %s..%s", symbol, start, end)
    chunk_i = 0

    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + step, end_ts)
        req = NewsRequest(
            symbols=symbol,
            start=_to_dt(chunk_start),
            end=_to_dt(chunk_end) + timedelta(days=1),  # medtag slutdagen
            include_content=False,
            exclude_contentless=True,
        )
        try:
            news_set = client.get_news(req)
            articles = news_set.data.get("news", []) if hasattr(news_set, "data") else []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Alpaca get_news fejlede for %s [%s..%s]: %s",
                symbol, chunk_start.date(), chunk_end.date(), exc,
            )
            articles = []

        for art in articles:
            rows.append(
                {
                    "id": int(getattr(art, "id", 0) or 0),
                    "date": _article_date(getattr(art, "created_at")),
                    "headline": str(getattr(art, "headline", "") or ""),
                    "summary": str(getattr(art, "summary", "") or ""),
                }
            )
        chunk_i += 1
        if chunk_i % 10 == 0:
            logger.info("%s: hentet %d artikler (gennem %s)", symbol, len(rows), chunk_end.date())
        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not rows:
        logger.info("%s: ingen nyheder fundet i intervallet.", symbol)
        return pd.DataFrame(columns=["id", "date", "headline", "summary"])
    df = pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)
    logger.info("%s: hentet %d artikler i alt (%d unikke).", symbol, len(rows), len(df))
    return df


# --- Artikel-arkiv ----------------------------------------------------------
def _archive_path(symbol: str) -> Path:
    return Path(config.NEWS_CACHE_DIR) / f"{symbol.upper()}.parquet"


def read_archive(symbol: str) -> pd.DataFrame:
    path = _archive_path(symbol)
    if not path.is_file():
        return pd.DataFrame(columns=list(_ARCHIVE_COLUMNS))
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke læse nyheds-arkiv %s: %s", path, exc)
        return pd.DataFrame(columns=list(_ARCHIVE_COLUMNS))
    for col in _ARCHIVE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[list(_ARCHIVE_COLUMNS)]


def write_archive(symbol: str, df: pd.DataFrame) -> None:
    _atomic_save_parquet(df[list(_ARCHIVE_COLUMNS)].reset_index(drop=True), _archive_path(symbol))


def score_pending(symbol: str, archive: pd.DataFrame) -> pd.DataFrame:
    """Kør finBERT på rækker uden score (in-place på en kopi); returnér opdateret arkiv."""
    if archive.empty:
        return archive
    pending = archive["score"].isna()
    if not pending.any():
        return archive
    texts = (
        archive.loc[pending, "headline"].fillna("")
        + ". "
        + archive.loc[pending, "summary"].fillna("")
    ).tolist()
    logger.info("%s: scorer %d nye artikler med finBERT…", symbol, len(texts))
    scores = finbert_scores(texts, progress_label=symbol)
    archive.loc[pending, "score"] = scores
    return archive


def daily_sentiment(archive: pd.DataFrame) -> pd.Series:
    """Daglig middel-score (kun scorede artikler), indekseret på normaliseret dato."""
    if archive.empty:
        return pd.Series(dtype=float)
    scored = archive.dropna(subset=["score"])
    if scored.empty:
        return pd.Series(dtype=float)
    s = scored.groupby("date")["score"].mean().sort_index()
    s.index = pd.to_datetime(s.index)
    return s


# --- Materialisering ind i OHLCV-cachen ------------------------------------
def _merge_into_ohlcv_cache(ohlcv_path: Path, sentiment: pd.Series) -> None:
    """Skriv/opdatér news_sentiment i en OHLCV-Parquet og genmaterialisér features."""
    if not ohlcv_path.is_file() or sentiment.empty:
        return
    try:
        raw = pd.read_parquet(ohlcv_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke læse OHLCV-cache %s: %s", ohlcv_path, exc)
        return
    if raw.empty:
        return
    if not isinstance(raw.index, pd.DatetimeIndex):
        if "date" in raw.columns:
            raw = raw.set_index("date")
        else:
            raw.index = pd.to_datetime(raw.index)
    raw.index = pd.to_datetime(raw.index).normalize().tz_localize(None)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    if not {"open", "high", "low", "close", "volume"}.issubset(raw.columns):
        return

    new_vals = sentiment.reindex(raw.index)
    existing = raw["news_sentiment"] if "news_sentiment" in raw.columns else None
    merged = new_vals.combine_first(existing) if existing is not None else new_vals

    vix = pd.to_numeric(raw["vix_close"], errors="coerce") if "vix_close" in raw.columns else None
    base = raw.drop(columns=[c for c in raw.columns if c in _DERIVED_FEATURE_COLUMNS], errors="ignore")
    base["news_sentiment"] = merged
    full = build_dataset_frame(base, vix)
    _atomic_save_parquet(full, ohlcv_path)


def ensure_sentiment_current(
    client,
    symbol: str,
    ohlcv_path: Path,
    *,
    start: date,
    end: date,
    persist: bool = True,
) -> pd.Series:
    """Inkrementelt: hent gap-nyheder → arkivér + score → daglig sentiment (+ materialisér).

    Henter kun [max(start, sidste_arkiverede+1) .. end] fra Alpaca, så kald er resumable.
    Returnerer hele den daglige sentiment-serie fra arkivet (også uden ny hentning).
    """
    archive = read_archive(symbol)

    fetch_start = start
    if not archive.empty:
        last_dt = archive["date"].max()
        if pd.notna(last_dt):
            cand = (pd.Timestamp(last_dt) + pd.Timedelta(days=1)).date()
            fetch_start = max(start, cand)

    changed = False
    if client is not None and fetch_start <= end:
        fresh = fetch_news(client, symbol, fetch_start, end)
        if not fresh.empty:
            fresh["score"] = np.nan
            fresh = fresh[list(_ARCHIVE_COLUMNS)]
            combined = (
                fresh
                if archive.empty
                else pd.concat([archive, fresh], ignore_index=True)
            )
            combined = combined.drop_duplicates("id", keep="first").reset_index(drop=True)
            if len(combined) != len(archive):
                archive = combined
                changed = True

    # Score evt. nye (eller tidligere u-scorede) artikler.
    pre_scored = archive["score"].notna().sum() if not archive.empty else 0
    archive = score_pending(symbol, archive)
    if not archive.empty and archive["score"].notna().sum() != pre_scored:
        changed = True

    if changed:
        write_archive(symbol, archive)

    series = daily_sentiment(archive)

    if persist and changed:
        _merge_into_ohlcv_cache(ohlcv_path, series)

    return series


def rescore_archive(symbol: str, ohlcv_path: Path) -> pd.Series:
    """Gen-scor HELE arkivet med finBERT (uden Alpaca-kald) og materialisér på ny.

    Bruges når finBERT-modellen eller aggregeringen ændres.
    """
    archive = read_archive(symbol)
    if archive.empty:
        return pd.Series(dtype=float)
    archive["score"] = np.nan
    archive = score_pending(symbol, archive)
    write_archive(symbol, archive)
    series = daily_sentiment(archive)
    _merge_into_ohlcv_cache(ohlcv_path, series)
    return series


def refresh_watchlist(
    api_key: str,
    secret_key: str,
    symbols: list[str],
    bars: Optional[dict] = None,
    *,
    end: Optional[date] = None,
    lookback_days: Optional[int] = None,
) -> Optional[dict]:
    """Auto-opdatér nyere sentiment for watchlisten under --run/--train.

    Henter kun et lille, nyligt vindue (NEWS_AUTO_REFRESH_LOOKBACK_DAYS) pr. symbol og
    skriver til både arkiv og OHLCV-cache. Når ``bars`` gives, opdateres news_sentiment-
    kolonnen også in-memory, så engineer_features ser den friske værdi uden gen-læsning.
    Springes over (uden fejl) hvis deaktiveret eller uden API-nøgler.
    """
    if not getattr(config, "NEWS_AUTO_REFRESH_ENABLED", True):
        logger.debug("News auto-refresh deaktiveret.")
        return bars
    if not (api_key and secret_key):
        logger.info("Springer news-sentiment over (manglende API-nøgler).")
        return bars

    end = end or date.today()
    lookback = int(lookback_days if lookback_days is not None else config.NEWS_AUTO_REFRESH_LOOKBACK_DAYS)
    start = end - timedelta(days=lookback)

    try:
        from alpaca.data.historical.news import NewsClient

        client = NewsClient(api_key, secret_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke oprette Alpaca NewsClient: %s", exc)
        return bars

    syms = sorted({s.strip().upper() for s in symbols if s and s.strip()})
    logger.info("Opdaterer news-sentiment for %s symboler (%s..%s).", len(syms), start, end)
    updated = 0
    for sym in syms:
        try:
            series = ensure_sentiment_current(
                client, sym, _cache_path(config.CACHE_DIR, sym), start=start, end=end
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("News-sentiment fejlede for %s: %s", sym, exc)
            continue
        if bars is not None and sym in bars and series is not None and not series.empty:
            df = bars[sym]
            reindexed = series.reindex(df.index)
            existing = df["news_sentiment"] if "news_sentiment" in df.columns else None
            df["news_sentiment"] = (
                reindexed.combine_first(existing) if existing is not None else reindexed
            )
            updated += 1
    logger.info("News-sentiment opdateret (%s symboler med friske bars-værdier).", updated)
    return bars


__all__ = [
    "finbert_scores",
    "fetch_news",
    "read_archive",
    "write_archive",
    "score_pending",
    "daily_sentiment",
    "ensure_sentiment_current",
    "rescore_archive",
    "refresh_watchlist",
]
