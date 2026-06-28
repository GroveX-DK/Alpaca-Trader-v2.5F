"""Konfiguration: API-nøgler fra .env, watchlist og hyperparametre."""

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

_PKG_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_ROOT.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

WATCHLIST = [
    # === INFORMATION TECHNOLOGY (35) ===
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # Nvidia
    "AVGO",   # Broadcom
    "ORCL",   # Oracle
    "CRM",    # Salesforce
    "ADBE",   # Adobe
    "AMD",    # Advanced Micro Devices
    "QCOM",   # Qualcomm
    "TXN",    # Texas Instruments
    "AMAT",   # Applied Materials
    "MU",     # Micron Technology
    "INTC",   # Intel
    "LRCX",   # Lam Research
    "KLAC",   # KLA Corporation
    "ADI",    # Analog Devices
    "NOW",    # ServiceNow
    "INTU",   # Intuit
    "PANW",   # Palo Alto Networks
    "FTNT",   # Fortinet
    "CDNS",   # Cadence Design Systems
    "SNPS",   # Synopsys
    "ACN",    # Accenture
    "IBM",    # IBM
    "DELL",   # Dell Technologies
    "HPQ",    # HP Inc
    "HPE",    # Hewlett Packard Enterprise
    "MRVL",   # Marvell Technology
    "PLTR",   # Palantir
    "CRWD",   # CrowdStrike
    "NET",    # Cloudflare
    "WDAY",   # Workday
    "SNOW",   # Snowflake
    "TEAM",   # Atlassian
    "ZS",     # Zscaler

    # === FINANCIALS (35) ===
    "BRK.B",  # Berkshire Hathaway
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "WFC",    # Wells Fargo
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "C",      # Citigroup
    "BLK",    # BlackRock
    "SCHW",   # Charles Schwab
    "AXP",    # American Express
    "V",      # Visa
    "MA",     # Mastercard
    "SPGI",   # S&P Global
    "MCO",    # Moody's
    "CB",     # Chubb
    "PGR",    # Progressive
    "MET",    # MetLife
    "PRU",    # Prudential Financial
    "ALL",    # Allstate
    "TRV",    # Travelers
    "USB",    # US Bancorp
    "PNC",    # PNC Financial
    "TFC",    # Truist Financial
    "COF",    # Capital One
    "ICE",    # Intercontinental Exchange
    "CME",    # CME Group
    "MSCI",   # MSCI Inc
    "FIS",    # Fidelity National Info
    "FISV",   # Fiserv
    "PYPL",   # PayPal
    "XYZ",    # Block (tidl. SQ; omdøbt 2025)
    "AIG",    # AIG
    "HIG",    # Hartford Financial
    "AFL",    # Aflac
    "RJF",    # Raymond James

    # === HEALTH CARE (30) ===
    "LLY",    # Eli Lilly
    "UNH",    # UnitedHealth
    "JNJ",    # Johnson & Johnson
    "ABBV",   # AbbVie
    "MRK",    # Merck
    "TMO",    # Thermo Fisher Scientific
    "ABT",    # Abbott Laboratories
    "DHR",    # Danaher
    "PFE",    # Pfizer
    "AMGN",   # Amgen
    "GILD",   # Gilead Sciences
    "BMY",    # Bristol-Myers Squibb
    "CVS",    # CVS Health
    "CI",     # Cigna
    "ELV",    # Elevance Health
    "HUM",    # Humana
    "MDT",    # Medtronic
    "BSX",    # Boston Scientific
    "SYK",    # Stryker
    "ZTS",    # Zoetis
    "ISRG",   # Intuitive Surgical
    "REGN",   # Regeneron
    "VRTX",   # Vertex Pharmaceuticals
    "BIIB",   # Biogen
    "IQV",    # IQVIA
    "BDX",    # Becton Dickinson
    "IDXX",   # IDEXX Laboratories
    "RMD",    # ResMed
    "DXCM",   # Dexcom
    "A",      # Agilent Technologies

    # === COMMUNICATION SERVICES (19) ===
    "META",   # Meta Platforms
    "GOOGL",  # Alphabet (Google)
    "NFLX",   # Netflix
    "DIS",    # Walt Disney
    "CMCSA",  # Comcast
    "T",      # AT&T
    "VZ",     # Verizon
    "TMUS",   # T-Mobile
    "CHTR",   # Charter Communications
    "SNAP",   # Snap
    "PINS",   # Pinterest
    "WBD",    # Warner Bros. Discovery
    "PSKY",   # Paramount Skydance (tidl. PARA; fusioneret 2025)
    "LYV",    # Live Nation
    "EA",     # Electronic Arts
    "TTWO",   # Take-Two Interactive
    "SPOT",   # Spotify
    "OMC",    # Omnicom Group (overtog IPG 2025)
    "FOXA",   # Fox Corporation

    # === ENERGY (19) ===
    "XOM",    # ExxonMobil
    "CVX",    # Chevron
    "COP",    # ConocoPhillips
    "EOG",    # EOG Resources
    "SLB",    # SLB (Schlumberger)
    "MPC",    # Marathon Petroleum
    "PSX",    # Phillips 66
    "VLO",    # Valero Energy
    "OXY",    # Occidental Petroleum
    "HAL",    # Halliburton
    "BKR",    # Baker Hughes
    "DVN",    # Devon Energy
    "FANG",   # Diamondback Energy
    "KMI",    # Kinder Morgan
    "WMB",    # Williams Companies
    "OKE",    # ONEOK
    "LNG",    # Cheniere Energy
    "TRGP",   # Targa Resources
    "APA",    # APA Corporation

    # === CONSUMER STAPLES (15) ===
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
    "PEP",    # PepsiCo
    "PM",     # Philip Morris
    "MO",     # Altria Group
    "COST",   # Costco
    "WMT",    # Walmart
    "MDLZ",   # Mondelez
    "KHC",    # Kraft Heinz
    "GIS",    # General Mills
    "CL",     # Colgate-Palmolive
    "KMB",    # Kimberly-Clark
    "EL",     # Estée Lauder
    "TSN",    # Tyson Foods
    "SYY",    # Sysco

    # === CONSUMER DISCRETIONARY (20) ===
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "MCD",    # McDonald's
    "NKE",    # Nike
    "SBUX",   # Starbucks
    "HD",     # Home Depot
    "LOW",    # Lowe's
    "TJX",    # TJX Companies
    "BKNG",   # Booking Holdings
    "MAR",    # Marriott International
    "HLT",    # Hilton Worldwide
    "GM",     # General Motors
    "F",      # Ford Motor
    "ABNB",   # Airbnb
    "RCL",    # Royal Caribbean
    "CMG",    # Chipotle
    "PHM",    # PulteGroup
    "DHI",    # D.R. Horton
    "ORLY",   # O'Reilly Auto Parts
    "AZO",    # AutoZone

    # === INDUSTRIALS (15) ===
    "GE",     # GE Aerospace
    "RTX",    # RTX Corporation (Raytheon)
    "HON",    # Honeywell
    "LMT",    # Lockheed Martin
    "NOC",    # Northrop Grumman
    "BA",     # Boeing
    "CAT",    # Caterpillar
    "DE",     # John Deere
    "UPS",    # UPS
    "FDX",    # FedEx
    "UNP",    # Union Pacific
    "ETN",    # Eaton
    "EMR",    # Emerson Electric
    "ITW",    # Illinois Tool Works
    # === MATERIALS (5) ===
    "LIN",    # Linde
    "SHW",    # Sherwin-Williams
    "FCX",    # Freeport-McMoRan
    "NEM",    # Newmont
    "ECL",    # Ecolab

    # === UTILITIES (5) ===
    "NEE",    # NextEra Energy
    "SO",     # Southern Company
    "DUK",    # Duke Energy
    "D",      # Dominion Energy
    "EXC",    # Exelon
]
LOOKBACK_DAYS = 1000
SEQ_LEN = LOOKBACK_DAYS
FETCH_EXTRA_DAYS = 60
FEATURE_WARMUP_TRADING_DAYS = 35
_TRADING_DAYS_PER_YEAR = 252
_CALENDAR_PER_TRADING = 365.25 / _TRADING_DAYS_PER_YEAR

def _trading_to_calendar_days(trading_days: int) -> int:
    return int(trading_days * _CALENDAR_PER_TRADING)

INFERENCE_FETCH_CALENDAR_DAYS = (
    _trading_to_calendar_days(SEQ_LEN + FEATURE_WARMUP_TRADING_DAYS) + FETCH_EXTRA_DAYS
)

TRAINING_YEARS = 10
N_FEATURES = 22
NEWS_SENTIMENT_HISTORY_START = date(2015, 1, 1)
NEWS_CACHE_DIR = _PKG_ROOT / "cache" / "news"
NEWS_AUTO_REFRESH_ENABLED = True
NEWS_AUTO_REFRESH_LOOKBACK_DAYS = 7
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32
FINBERT_DEVICE = "auto"
WATCHLIST_CSV_DIR = _PROJECT_ROOT / "output" / "Watchlist"
LSTM_HIDDEN = 128
LSTM_LAYERS = 3
DROPOUT = 0.2
LR = 1e-5
# Tidligere 1: modellen nåede reelt aldrig at træne (én epoch → nær-tilfældige outputs →
# all-in på et nær-tilfældigt pick ≈ univers-snit ≈ SPY). Med kort SEQ_LEN er mange epochs
# billige; early stopping (patience) afgør hvornår der reelt stoppes.
EPOCHS = 1
BATCH_SIZE = 32
# Lidt stærkere L2 (var 1e-5) for at dæmpe den hurtige overfit på overlappende vinduer.
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 1.0
VAL_RATIO = 0.25
# Tidligere 1 (stoppede ved første ikke-forbedring → reelt 1 epoch). 10 giver modellen et
# reelt budget til at finde et bedre stoppunkt end epoch 1.
EARLY_STOP_PATIENCE = 1
# Valgfri trænings-stride (kun træningssættet): behold kun hvert N'te vindue PR. SYMBOL for at
# skære i de næsten-identiske, overlappende vinduer. 1 = brug alle vinduer (val/inferens altid 1).
# Hæv til 2-3 hvis val stadig divergerer tidligt trods kort SEQ_LEN.
TRAIN_WINDOW_STRIDE = 1
SAVE_MODEL_ONLY_IF_BETTER_THAN_DISK = True
# Mindste antal symboler med gyldig forudsigelse en dag skal have i regime-backtesten,
# så det tynde tidlige univers ikke giver degenererede all-in-dage.
MIN_SYMBOLS_PER_DAY = 10
CACHE_DIR = _PKG_ROOT / "cache"
OHLCV_CACHE_ENABLED = True
# OHLCV-prisjustering: "all" (split+udbytte, total-return) | "split" | "raw".
# "raw" = gammel adfærd (inkrementel cache-splice bevares ved revert).
OHLCV_ADJUSTMENT = "all"
VOLATILITY_ROLLING_RETURNS = 21
VOLATILITY_TRADING_DAYS_PER_YEAR = 252
MODEL_DIR = _PKG_ROOT / "models"
MODEL_PATH = MODEL_DIR / "lstm_stock.pt"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
TRADE_LOG_PATH = MODEL_DIR / "trade_log.csv"
TRADE_ACTIVITY_LOG_PATH = MODEL_DIR / "trade_activity_log.csv"
RANDOM_SEED = 42
TRAIN_DEVICE = "auto"
TRAIN_AMP = True
TRAIN_NUM_WORKERS = 0
TORCH_COMPILE_TRAIN = False
TORCH_COMPILE_MODE = "default"
INFERENCE_DEVICE = "cpu"
LR_SCHEDULER_ENABLED = True
LR_SCHEDULER_FACTOR = 0.5
# Reager hurtigere på val-plateau (var 5) så LR skæres ned før modellen begynder at divergere.
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-6
# CPCV (Combinatorial Purged Cross-Validation) — holdbarheds-backtest med genoptræning pr.
# fold-kombination. N grupper, k test-grupper → C(N,k) genoptræninger, φ=C(N,k)·k/N stier.
# N=6,k=2 = 15 genoptræninger / 5 stier (López de Prado-default). DYRT: kør på GPU-desktop.
CPCV_N_GROUPS = 6
CPCV_K_TEST = 2
# Embargo (kalenderdage) efter hver test-blok oven i SEQ_LEN-purge — dæmper seriel label-leakage.
CPCV_EMBARGO_DAYS = 10
