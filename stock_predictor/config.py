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
# Sekvenslængde til LSTM'en. Tidligere 2000 (~8 år): et urealistisk langt vindue til at
# forudsige ÉN næste-dags bevægelse — gradienter forsvinder over så mange skridt, signalet
# (de seneste dage/uger) drukner, og hver epoch tog ~7 t på CPU, hvilket tvang EPOCHS=1.
# ~40 handelsdage (~2 mdr) giver en lærbar horisont, ~10× hurtigere epochs og langt mindre
# vindues-overlap (mindre overfit pr. epoch). OBS: ændring kræver fuld genoptræning
# (checkpoint gemmer seq_len og inferens validerer mod det).
LOOKBACK_DAYS = 40
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
# Basis-feature-antal (FEATURE_COLUMNS uden makro). Det effektive N_FEATURES sættes
# længere nede afhængigt af MACRO_FEATURES_ENABLED (krise-robusthed-sektionen).
_BASE_N_FEATURES = 22
N_FEATURES = _BASE_N_FEATURES
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
LR = 1e-4
# Tidligere 1: modellen nåede reelt aldrig at træne (én epoch → nær-tilfældige outputs →
# all-in på et nær-tilfældigt pick ≈ univers-snit ≈ SPY). Med kort SEQ_LEN er mange epochs
# billige; early stopping (patience) afgør hvornår der reelt stoppes.
EPOCHS = 80
BATCH_SIZE = 32
# Lidt stærkere L2 (var 1e-5) for at dæmpe den hurtige overfit på overlappende vinduer.
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 1.0
VAL_RATIO = 0.25
# Tidligere 1 (stoppede ved første ikke-forbedring → reelt 1 epoch). 10 giver modellen et
# reelt budget til at finde et bedre stoppunkt end epoch 1.
EARLY_STOP_PATIENCE = 10
# Valgfri trænings-stride (kun træningssættet): behold kun hvert N'te vindue PR. SYMBOL for at
# skære i de næsten-identiske, overlappende vinduer. 1 = brug alle vinduer (val/inferens altid 1).
# Hæv til 2-3 hvis val stadig divergerer tidligt trods kort SEQ_LEN.
TRAIN_WINDOW_STRIDE = 1
SAVE_MODEL_ONLY_IF_BETTER_THAN_DISK = True
# Mindste antal symboler med gyldig forudsigelse en dag skal have i regime-backtesten,
# så det tynde tidlige univers ikke giver degenererede all-in-dage.
MIN_SYMBOLS_PER_DAY = 10
# Risk-off-filter i regime-backtesten: handl kun når VIX (forrige dags close) er UNDER denne
# tærskel; ellers stå i kontanter (0 % den dag). 30 ≈ klassisk "stress"-niveau.
VIX_RISK_OFF_THRESHOLD = 30.0
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
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_MIN_LR = 1e-6

# ============================================================================
# Krise-robusthed (branch: crisis-robustness)
# ALLE flag herunder defaulter til den NUVÆRENDE adfærd, så pipelinen er uændret
# indtil et flag slås til. Det er den "bløde" revert-sti: sæt alt til False og
# systemet opfører sig som før. Fuld ændringslog + revert-kommandoer:
# docs/CRISIS_ROBUSTNESS_CHANGES.md.
# ============================================================================

# --- Option 3: markeds-brede krise-signal-features (VIX-termstruktur, breadth,
#     tværsnits-korrelation, kreditspænd, bond-vol/put-call). Slået fra => det
#     nuværende feature-sæt (FEATURE_COLUMNS). Når True forventes N_FEATURES at
#     matche len(FEATURE_COLUMNS) inkl. de nye makro-kolonner.
MACRO_FEATURES_ENABLED = True
# Cache-fil for den samlede markeds-brede makro-frame (date -> kolonner).
MACRO_CACHE_PATH = CACHE_DIR / "macro" / "macro_features.parquet"
# Rullende vindue til breadth (% over glidende gennemsnit) og tværsnits-korrelation.
MACRO_BREADTH_MA_DAYS = 200
MACRO_CORR_WINDOW_DAYS = 21
# De markeds-brede makro-kolonner (samme for alle symboler, gemt som vix_close).
# Rækkefølgen er autoritativ og appendes til FEATURE_COLUMNS når flag er til.
# Neutralværdi pr. kolonne bruges når kilden mangler (ingen drop af rækker).
MACRO_FEATURE_COLUMNS = (
    "vix_ts_slope",       # ^VIX / ^VIX3M (>1 = backwardation/panik)
    "vvix_level",         # ^VVIX / 100 (vol-of-vol)
    "breadth_pct",        # andel af watchlist over 200d MA (0..1)
    "xsec_corr",          # middel parvis korrelation af 21d-afkast (0..1)
    "credit_ratio_chg",   # 5d pct-ændring i HYG/LQD (negativ i stress)
    "move_chg",           # 5d pct-ændring i ^MOVE (positiv i bond-stress)
    "oil_log_ret",        # ln(WTI_t / WTI_{t-1}) (CL=F; markeds-bred olie-puls, alle tickere)
    "oil_vol_annual_pct", # annualiseret log-vol af WTI (samme form som vol_annual_pct)
)
MACRO_FEATURE_NEUTRAL = {
    "vix_ts_slope": 1.0,
    "vvix_level": 0.9,
    "breadth_pct": 0.5,
    "xsec_corr": 0.3,
    "credit_ratio_chg": 0.0,
    "move_chg": 0.0,
    "oil_log_ret": 0.0,         # ingen ændring når oliekilde mangler
    "oil_vol_annual_pct": 35.0, # ≈ typisk WTI-annualiseret vol (kun fyld ved manglende kilde)
}

# --- Option 5: dagens-open-feature. Live-pipelinen kører LIGE EFTER markedsåbning, så
#     dagens open er kendt. next_open_gap = ln(open_{t+1} / close_t) føjer dagens åbnings-
#     gap til vinduets sidste række — samme tidsalignment som target (open→close næste dag),
#     så ingen leakage. Slået fra => uændret feature-sæt (uden next_open_gap).
#     OBS: ændrer N_FEATURES og kræver fuld genoptræning (checkpoint gemmer n_features).
OPEN_FEATURE_ENABLED = True

# Effektivt feature-antal: basis + dagens-open + makro når slået til. Checkpoint gemmer
# dette tal, og inferens validerer mod det — alle flag fra => uændret 22-feature-model.
N_FEATURES = (
    _BASE_N_FEATURES
    + (1 if OPEN_FEATURE_ENABLED else 0)
    + (len(MACRO_FEATURE_COLUMNS) if MACRO_FEATURES_ENABLED else 0)
)

# --- Option 4A: krise-oversampling — vægt træningsdage efter VIX, så modellen ser
#     krak-regimer oftere (WeightedRandomSampler). VIX_REF er niveauet hvor vægt ~1.
CRISIS_OVERSAMPLE_ENABLED = True
CRISIS_OVERSAMPLE_VIX_REF = 20.0
# Var 5.0: kraftig oversampling skævvrider train-fordelingen ift. det (uvægtede) seneste
# val-vindue og bidrager til at val stiger tidligt. 2.5 beholder krise-fokus mere nænsomt.
CRISIS_OVERSAMPLE_MAX_WEIGHT = 2.5

# --- Option 4B: usikkerheds-head — modellen forudsiger kvantiler (10/50/90) frem
#     for ét punkt-estimat. Score = median (q50); konfidensbånd = q90 - q10.
#     Slået fra => head = Linear(hidden, 1) + HuberLoss (uændret).
UNCERTAINTY_HEAD_ENABLED = True
UNCERTAINTY_QUANTILES = (0.1, 0.5, 0.9)

# --- Option 4C: walk-forward genoptræning (TUNG — ~hundredvis af CPU-timer;
#     anbefales kun på GPU). Default fra.
WALK_FORWARD_ENABLED = False
WALK_FORWARD_TRAIN_MIN_YEARS = 8   # mindste træningsvindue før første OOS-år
WALK_FORWARD_STEP_YEARS = 1        # antal år pr. OOS-skridt

# --- Option 2: long/short markeds-neutral i backtesten. Long top-k, short bottom-k,
#     dollar-neutral; brutto-eksponering skaleres af vol-target og (når
#     usikkerheds-head er til) konfidensbåndet.
LONG_SHORT_ENABLED = True
LONG_SHORT_TOP_K = 5
LONG_SHORT_BOTTOM_K = 5
LONG_SHORT_TARGET_VOL_PCT = 15.0   # årlig vol-target for daglige strategi-afkast
LONG_SHORT_MAX_GROSS = 1.0         # loft på brutto (long+short) som andel af equity
# --- Option 2 (live): long/short i paper-traderen. Default fra => --run bevarer den
#     nuværende enkelt-symbol-rotation (rotate_to_symbol).
LONG_SHORT_LIVE_ENABLED = False

# ============================================================================
# Retningsbestemt enkelt-navn-strategi ("bedste aktie uanset hvad", men long hvis
# den forudsagte bevægelse er op og short hvis den er ned) + ærlige handelsomkostninger.
# ============================================================================

# --- Retningsbestemt udvælgelse i backtesten (_simulate): vælg navnet med den STØRSTE
#     forudsagte ABSOLUTTE bevægelse; long hvis pred>0, short hvis pred<0 (short-afkast =
#     -realiseret). Fra => gammel adfærd (højeste pred, kun long).
DIRECTIONAL_ENABLED = True
# Konfidens-gate: handl kun når |pred| (forudsagt %) er mindst dette. 0.0 = handl altid
# ("bedste aktie uanset hvad"). Hæv (fx 0.3) for at stå i kontanter på lav-konfidens-dage.
DIRECTIONAL_MIN_ABS_PCT = 0.0
# --- Retningsbestemt enkelt-navn LIVE i paper-traderen (rotate_to_symbol med side).
#     Default fra => --run bevarer ren long-rotation indtil du slår dette til.
DIRECTIONAL_LIVE_ENABLED = False

# --- Handelsomkostninger i backtesten (spread+slippage+kurtage) pr. ben i basispoint.
#     Strategien roterer ~100 %/dag, så omkostninger er afgørende for et ærligt facit.
#     Lægges på hver entry og exit (round-trip = 2×). Sæt 0.0 for det gamle brutto-tal.
#     5 bp/ben ≈ 0,10 % round-trip ~ realistisk for likvide mega-caps på Alpaca.
BACKTEST_COST_BPS_PER_SIDE = 5.0