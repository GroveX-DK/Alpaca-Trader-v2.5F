"""Konfiguration: API-nøgler fra .env, watchlist og hyperparametre."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Indlæs .env fra projektrod (stock_predictors forældremappe) eller cwd
_PKG_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_ROOT.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Alpaca/IEX:    kun US-noteringer (NYSE/NASDAQ). Lokale tickere (fx VOLV-B, 005930, PETR4)
# understøttes ikke — erstattet hvor US-ticker er entydig; ellers fjernet.
WATCHLIST = [
    "AAPL",
    "NVDA",
    "TSLA",
    "MSFT",
    "AMD",
    "META",
    "GOOGL",
    "AMZN",
    "BAC",
    "COST",
    "CRM",
    "CVX",
    "GS",
    "HD",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "LMT",
    "MA",
    "NEE",
    "PFE",
    "PG",
    "UNH",
    "V",
    "WMT",
    "XOM",
    "ABBV",
    "AMT",
    "AMX",
    "ASML",
    "AZN",
    "BABA",
    "BP",
    "BHP",
    "BUD",
    "CHKP",
    "CNI",
    "EQNR",
    "ERIC",
    "FMX",
    "HDB",
    "HSBC",
    "INFY",
    "ITUB",
    "MUFG",
    "NICE",
    "NOK",
    "NVO",
    "NVS",
    "PBR",
    "RIO",
    "RY",
    "SAN",
    "SAP",
    "SHEL",
    "SHOP",
    "SONY",
    "SU",
    "TD",
    "TM",
    "UL",
    "VALE",
    "WIT",
    "TSM",
    "SQM",
    "TCEHY",
    "SFTBY",
    "ATLKY",
    "CABGY",
    "AMKBY",
    "VLVLY",
    "WEGZY",
    "DBSDY",
    "SINGY",
    "SGAPY",
]

# LSTM-sekvenslængde i handelsdage (skal matche checkpoint seq_len; gen-træn ved ændring)
LOOKBACK_DAYS = 600
SEQ_LEN = LOOKBACK_DAYS

# Ekstra kalenderdages buffer ved API/cache-hentning
FETCH_EXTRA_DAYS = 60
# Handelsdage tabt ved engineer_features dropna (MACD, vol, m.m.)
FEATURE_WARMUP_TRADING_DAYS = 35
_TRADING_DAYS_PER_YEAR = 252
_CALENDAR_PER_TRADING = 365.25 / _TRADING_DAYS_PER_YEAR


def _trading_to_calendar_days(trading_days: int) -> int:
    return int(trading_days * _CALENDAR_PER_TRADING)


# Kalenderdage at hente ved inference (ikke det samme som SEQ_LEN)
INFERENCE_FETCH_CALENDAR_DAYS = (
    _trading_to_calendar_days(SEQ_LEN + FEATURE_WARMUP_TRADING_DAYS) + FETCH_EXTRA_DAYS
)

# Træningsvinduer
TRAINING_YEARS = 5
# Antal kolonner fra engineer_features; ændres ved nye features — gen-træn og kassér gamle lstm_stock.pt + feature_scaler.joblib.
N_FEATURES = 15

# Watchlist-CSV (OHLCV + ekstra kolonner) til cache-import
WATCHLIST_CSV_DIR = _PROJECT_ROOT / "output" / "Watchlist"

# Model (lettere / hurtigere CPU: fx LSTM_HIDDEN=128, LSTM_LAYERS=2)
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
DROPOUT = 0.2
LR = 1e-4   
EPOCHS = 100
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.25
EARLY_STOP_PATIENCE = 5
SAVE_MODEL_ONLY_IF_BETTER_THAN_DISK = True
CACHE_DIR = _PKG_ROOT / "cache"
OHLCV_CACHE_ENABLED = True
VOLATILITY_ROLLING_RETURNS = 21
VOLATILITY_TRADING_DAYS_PER_YEAR = 252
MODEL_DIR = _PKG_ROOT / "models"
MODEL_PATH = MODEL_DIR / "lstm_stock.pt"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
TRADE_LOG_PATH = MODEL_DIR / "trade_log.csv"
# Hver vellykket åbningskøb: tid, symbol, beløb (USD), forventet næste dags open→close %
TRADE_ACTIVITY_LOG_PATH = MODEL_DIR / "trade_activity_log.csv"

RANDOM_SEED = 42

# --- PyTorch: enhed, AMP, DataLoader, compile ---
# "auto": cuda hvis muligt, ellers mps (Apple), ellers cpu
TRAIN_DEVICE = "auto"
# Mixed precision (kun når træning ender på CUDA og TRAIN_AMP er True)
TRAIN_AMP = True
# DataLoader; på CUDA (træning fra host-RAM) bruges værdien; ellers 0 hvis mps
TRAIN_NUM_WORKERS = 0
TORCH_COMPILE_TRAIN = False
# default | reduce-overhead | max-autotune (sidste to mest til GPU)
TORCH_COMPILE_MODE = "default"

# Inference: "cpu" giver enklest drift og reproducerbarhed; "auto" bruger gpu hvis muligt
INFERENCE_DEVICE = "cpu"

# ReduceLROnPlateau styret af val_mse (kun når val-split findes)
LR_SCHEDULER_ENABLED = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-6
