"""Paper trading og handelslog."""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alpaca.data.historical import StockHistoricalDataClient  # noqa: E402
from alpaca.data.requests import StockBarsRequest  # noqa: E402
from alpaca.data.timeframe import TimeFrame  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import OrderSide, TimeInForce  # noqa: E402
from alpaca.trading.requests import MarketOrderRequest  # noqa: E402

from stock_predictor import config  # noqa: E402

logger = logging.getLogger(__name__)

STATE_PATH = config.MODEL_DIR / "open_trade_state.json"
_AGENT_DEBUG_LOG = _ROOT / "debug-bde6e6.log"
_DEBUG_SESSION_LOG = _ROOT / "debug-9f6248.log"
_DEBUG_SESSION_ID = "9f6248"
_POST_CLOSE_CASH_POLL_SEC = 1.0
_POST_CLOSE_CASH_MAX_WAIT_SEC = 45.0
_POST_CLOSE_MIN_CASH_USD = 50.0


def _session_dbg(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    run_id: str = "pre-fix",
) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_SESSION_LOG.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # endregion


def _agent_dbg(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    run_id: str = "pre-fix",
) -> None:
    # region agent log
    try:
        payload = {
            "sessionId": "bde6e6",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _AGENT_DEBUG_LOG.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # endregion


_TRADE_ACTIVITY_FIELDNAMES = [
    "logged_at",
    "event",
    "symbol",
    "amount_usd",
    "qty",
    "predicted_gain_pct",
    "cash_before",
    "buying_power_before",
    "daytrading_buying_power_before",
    "regt_buying_power_before",
    "non_marginable_buying_power_before",
    "notes",
]


def _acct_money(acct, attr: str) -> float:
    """Alpaca returnerer typisk strenge; normalisér til ikke-negativ float."""
    raw = getattr(acct, attr, None)
    if raw is None:
        return 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _trade_budget_usd(
    acct,
    *,
    post_rotation_fallback: bool = False,
) -> tuple[float, dict[str, float]]:
    """
    Konservativt USD-loft før reserve: min(cash, buying_power); med DTBP>0 også min(DTBP).
    Ved DTBP=0: kun ekstra loft fra Reg T / NMBP hvis feltet er >0 (NMBP=0 er almindeligt på
    margin og må ikke nulstille loftet).

    post_rotation_fallback: efter lukning af positioner kan Alpaca vise cash=0 mens equity/regt
    reflekterer kontoværdi (~ikke fuld margin-BP). Brug min(regt, equity, bp, dtbp) kun da.
    """
    cash = _acct_money(acct, "cash")
    bp = _acct_money(acct, "buying_power")
    dtbp = _acct_money(acct, "daytrading_buying_power")
    regt = _acct_money(acct, "regt_buying_power")
    nmbp = _acct_money(acct, "non_marginable_buying_power")
    equity = _acct_money(acct, "equity")
    portfolio_value = _acct_money(acct, "portfolio_value")

    ceiling = min(cash, bp)
    branch = "min_cash_bp"
    if dtbp > 0:
        ceiling = min(ceiling, dtbp)
        branch = "also_dtbp"
    else:
        # NMBP kan være 0 på margin (alle instrumenter marginérbare) — må ikke tolkes som "0 USD".
        extra = [x for x in (regt, nmbp) if x > 0]
        if extra:
            ceiling = min(ceiling, min(extra))
            branch = "dtbp0_extra_caps"
        else:
            branch = "dtbp0_no_extra_caps"

    if ceiling <= 0 and post_rotation_fallback and cash <= 0:
        spendable = [x for x in (regt, equity, portfolio_value) if x > 0]
        if spendable and bp > 0:
            base = min(spendable)
            ceiling = min(bp, dtbp if dtbp > 0 else bp, base)
            branch = "post_rotation_regt_equity"

    metrics = {
        "cash": cash,
        "equity": equity,
        "portfolio_value": portfolio_value,
        "buying_power": bp,
        "daytrading_buying_power": dtbp,
        "regt_buying_power": regt,
        "non_marginable_buying_power": nmbp,
    }
    # region agent log
    _agent_dbg(
        "H1",
        "trader.py:_trade_budget_usd",
        "ceiling_after_caps",
        {
            "ceiling": ceiling,
            "branch": branch,
            "cash": cash,
            "bp": bp,
            "dtbp": dtbp,
            "regt": regt,
            "nmbp": nmbp,
            "equity": equity,
            "portfolio_value": portfolio_value,
            "extra_caps_used": [x for x in (regt, nmbp) if x > 0],
        },
    )
    # endregion
    _session_dbg(
        "H2",
        "trader.py:_trade_budget_usd",
        "budget_branch",
        {
            "branch": branch,
            "ceiling": ceiling,
            "post_rotation_fallback": post_rotation_fallback,
            "cash": cash,
            "regt": regt,
            "equity": equity,
        },
    )
    return ceiling, metrics


def _wait_positions_closed(tc: TradingClient, max_wait_sec: float) -> int:
    """Vent til ingen åbne positioner (efter close_all). Returnér antal positioner til sidst."""
    deadline = time.time() + max_wait_sec
    remaining = -1
    while time.time() < deadline:
        try:
            remaining = len(tc.get_all_positions())
        except Exception:  # noqa: BLE001
            remaining = -1
        if remaining == 0:
            return 0
        time.sleep(_POST_CLOSE_CASH_POLL_SEC)
    try:
        return len(tc.get_all_positions())
    except Exception:  # noqa: BLE001
        return remaining if remaining >= 0 else -1


def _poll_cash_after_close(
    tc: TradingClient,
    min_cash: float,
    max_wait_sec: float,
    run_id: str,
) -> tuple[object, int, float]:
    """Poll get_account indtil cash >= min_cash eller timeout. Returnér (acct, attempts, cash)."""
    deadline = time.time() + max_wait_sec
    attempts = 0
    acct = tc.get_account()
    cash = _acct_money(acct, "cash")
    while cash < min_cash and time.time() < deadline:
        attempts += 1
        time.sleep(_POST_CLOSE_CASH_POLL_SEC)
        acct = tc.get_account()
        cash = _acct_money(acct, "cash")
        _session_dbg(
            "H1",
            "trader.py:_poll_cash_after_close",
            "poll_tick",
            {
                "attempt": attempts,
                "cash": cash,
                "equity": _acct_money(acct, "equity"),
                "regt": _acct_money(acct, "regt_buying_power"),
            },
            run_id=run_id,
        )
    return acct, attempts, cash


def _ensure_trade_activity_schema() -> None:
    """Gammelt CSV-skema uden købekraft-kolonner: arkivér én gang så nyt header skrives."""
    path = config.TRADE_ACTIVITY_LOG_PATH
    if not path.is_file():
        return
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except OSError:
        return
    if "buying_power_before" in first:
        return
    legacy = path.with_name(path.stem + "_legacy.csv")
    n = 0
    while legacy.is_file():
        n += 1
        legacy = path.with_name(f"{path.stem}_legacy_{n}.csv")
    path.rename(legacy)
    logger.warning(
        "trade_activity_log.csv havde gammelt skema — omdøbt til %s (ny fil får udvidet header).",
        legacy.name,
    )


def _append_trade_activity_csv(
    row: dict[str, Optional[float] | Optional[str]],
) -> None:
    """Log hver åbningshandel: ca.-beløb (USD) og modellets forventede open→close %."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_trade_activity_schema()
    fieldnames = _TRADE_ACTIVITY_FIELDNAMES
    append_header = not config.TRADE_ACTIVITY_LOG_PATH.is_file()
    with config.TRADE_ACTIVITY_LOG_PATH.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        if append_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _append_trade_csv(
    row: dict[str, Optional[float] | Optional[str]],
) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    append_header = not config.TRADE_LOG_PATH.is_file()
    fieldnames = [
        "logged_date",
        "symbol",
        "predicted_gain_pct",
        "actual_gain_pct",
        "notes",
    ]
    with config.TRADE_LOG_PATH.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        if append_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _read_state() -> Optional[dict]:
    if not STATE_PATH.is_file():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke læse state-fil (%s): %s", STATE_PATH, exc)
        return None


def _write_state(payload: Optional[dict]) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not payload:
        if STATE_PATH.is_file():
            STATE_PATH.unlink(missing_ok=True)
        return
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _latest_daily_close(symbol: str) -> Optional[float]:
    """Seneste lukkekurs fra daglig bar (Alpaca) som konservativt P/L-mark."""

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    try:
        end = date.today()
        start = end - timedelta(days=14)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed="iex",
        )
        bs = client.get_stock_bars(req)
        bars = bs.data.get(symbol)
        if not bars:
            return None
        last_bar = bars[-1]
        return float(last_bar.close)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke hente seneste luk for %s: %s", symbol, exc)
        return None


def _finalize_open_trade(predicted_pct: Optional[float]) -> None:
    """Før rotation: log faktisk %-afkast for aktuelle Alpaca-papirposition(er)."""
    try:
        t = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
        positions = t.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke hente åbne positioner før log (%s)", exc)
        return

    if not positions:
        state = _read_state()
        if state:
            _write_state(None)
        return

    for pos in positions:
        symbol = str(pos.symbol)
        avg = float(pos.avg_entry_price)
        close_px = _latest_daily_close(symbol)
        if close_px is None:
            logger.warning(
                "Ingen lukkekurs til beregn-afkast — springer CSV-linje for %s.", symbol,
            )
            continue
        actual_pct = (close_px - avg) / max(avg, 1e-9) * 100.0
        predicted = predicted_pct if predicted_pct is not None else float("nan")
        _append_trade_csv(
            {
                "logged_date": date.today().isoformat(),
                "symbol": symbol,
                "predicted_gain_pct": predicted,
                "actual_gain_pct": actual_pct,
                "notes": "marked med seneste daglige luk vs gns. entry (paper)",
            }
        )


def rotate_to_symbol(symbol: str, predicted_gain_pct: float) -> None:
    """Luk eksisterende positioner, hedging log, derefter åbn alle midler i ét symbol."""

    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        logger.error("Manglende API-nøgler — ingen handel.")
        raise RuntimeError("Manglende Alpaca-nøgler.")

    state = _read_state()
    predicted_from_prev = (
        float(state["predicted_gain_pct"]) if state and "predicted_gain_pct" in state else None
    )
    try:
        _finalize_open_trade(predicted_from_prev)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Kunne ikke færdiggøre sidste trades log (non-fatal): %s", exc,
        )

    tc = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)

    try:
        positions_snapshot = tc.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke hente positioner før luk: %s", exc)
        positions_snapshot = []
    had_open_positions = len(positions_snapshot) > 0
    acct_pre = tc.get_account()
    _session_dbg(
        "H4",
        "trader.py:rotate_to_symbol",
        "before_close",
        {
            "had_open_positions": had_open_positions,
            "cash": _acct_money(acct_pre, "cash"),
            "equity": _acct_money(acct_pre, "equity"),
            "open_symbols": [p.symbol for p in positions_snapshot],
        },
    )

    # Luk alle åbne positioner
    try:
        tc.close_all_positions(cancel_orders=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("close_all_positions fejlede (fortsætter): %s", exc)
        for p in tc.get_all_positions():
            try:
                tc.close_position(p.symbol)
            except Exception as exc2:  # noqa: BLE001
                logger.warning("Kunne ikke lukke %s: %s", p.symbol, exc2)

    positions_left = _wait_positions_closed(tc, _POST_CLOSE_CASH_MAX_WAIT_SEC)
    use_rotation_fallback = had_open_positions and positions_left == 0

    acct_after_close = tc.get_account()
    _session_dbg(
        "H3",
        "trader.py:rotate_to_symbol",
        "after_close_before_poll",
        {
            "positions_left": positions_left,
            "cash": _acct_money(acct_after_close, "cash"),
            "equity": _acct_money(acct_after_close, "equity"),
            "regt": _acct_money(acct_after_close, "regt_buying_power"),
            "use_rotation_fallback": use_rotation_fallback,
        },
    )

    if use_rotation_fallback and _acct_money(acct_after_close, "cash") < _POST_CLOSE_MIN_CASH_USD:
        acct, poll_attempts, cash_polled = _poll_cash_after_close(
            tc,
            _POST_CLOSE_MIN_CASH_USD,
            _POST_CLOSE_CASH_MAX_WAIT_SEC,
            run_id="pre-fix",
        )
        _session_dbg(
            "H1",
            "trader.py:rotate_to_symbol",
            "after_poll",
            {"poll_attempts": poll_attempts, "cash": cash_polled},
        )
    else:
        acct = acct_after_close

    ceiling_usd, am = _trade_budget_usd(
        acct,
        post_rotation_fallback=use_rotation_fallback,
    )
    cash_balance = am["cash"]
    reserve = max(50.0, ceiling_usd * 0.005)
    notional = ceiling_usd - reserve
    logger.info(
        "Konto (USD): cash=%.2f equity=%.2f portfolio_value=%.2f buying_power=%.2f "
        "daytrading_buying_power=%.2f regt_buying_power=%.2f non_marginable_buying_power=%.2f "
        "-> loft=%.2f (reserve=%.2f planlagt_køb=%.2f; post_rotation_fallback=%s)",
        am["cash"],
        am.get("equity", 0.0),
        am.get("portfolio_value", 0.0),
        am["buying_power"],
        am["daytrading_buying_power"],
        am["regt_buying_power"],
        am["non_marginable_buying_power"],
        ceiling_usd,
        reserve,
        notional,
        use_rotation_fallback,
    )
    if notional <= 0:
        equity_v = am.get("equity", 0.0)
        portfolio_v = am.get("portfolio_value", 0.0)
        hint = ""
        if cash_balance <= 0 and max(equity_v, portfolio_v) > reserve:
            hint = (
                f" cash=0 men equity/portfolio≈{max(equity_v, portfolio_v):.2f} — "
                "køb kræver fri cash (vent på afregning efter salg eller luk positioner)."
            )
        logger.error(
            "Ikke tilstrækkelig købekraft efter broker-loft (loft=%.2f USD; cash=%.2f; "
            "equity=%.2f; portfolio_value=%.2f; buying_power=%.2f; dtbp=%.2f).%s",
            ceiling_usd,
            cash_balance,
            equity_v,
            portfolio_v,
            am["buying_power"],
            am["daytrading_buying_power"],
            hint,
        )
        raise RuntimeError(
            "Ikke tilstrækkelig købekraft (cash-only: min af cash, buying_power og evt. DTBP)."
            + hint,
        )

    dc = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    last_px: Optional[float] = None
    try:
        end = date.today()
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=end - timedelta(days=21),
            end=end,
            feed="iex",
        )
        bs = dc.get_stock_bars(req)
        bars_seq = bs.data.get(symbol)
        if bars_seq:
            last_px = float(bars_seq[-1].close)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kunne ikke hente prisbenchmark for %s: %s — bruger kun notional", symbol, exc)

    acct_order = tc.get_account()
    dtbp_submit = _acct_money(acct_order, "daytrading_buying_power")
    pdt = bool(getattr(acct_order, "pattern_day_trader", False))
    # Alpaca 403 når DTBP=0: både samme-dags rotation OG flad PDT-konto (logs: had_open_positions=false, pdt=true).
    block_buy_dtbp = dtbp_submit <= 0 and (had_open_positions or pdt)
    # region agent log
    _agent_dbg(
        "H4",
        "trader.py:rotate_to_symbol",
        "pre_submit_account",
        {
            "dtbp": dtbp_submit,
            "had_open_positions": had_open_positions,
            "pattern_day_trader": pdt,
            "block_buy_dtbp": block_buy_dtbp,
            "notional_plan": round(notional, 2),
            "symbol": symbol,
        },
        run_id="dtbp-gate-v2",
    )
    # endregion
    if block_buy_dtbp:
        logger.error(
            "Spring købsordre over: daytrading_buying_power er %.2f (pattern_day_trader=%s, lukkede_position=%s). "
            "Alpaca returnerer 403 insufficient DTBP for denne markeds-køb — vent til DTBP genopfyldes eller næste session.",
            dtbp_submit,
            pdt,
            had_open_positions,
        )
        # region agent log
        _agent_dbg(
            "H5",
            "trader.py:rotate_to_symbol",
            "blocked_dtbp_zero",
            {
                "skipped_submit": True,
                "pdt": pdt,
                "had_open_positions": had_open_positions,
            },
            run_id="dtbp-gate-v2",
        )
        # endregion
        raise RuntimeError(
            "daytrading_buying_power er 0 — Alpaca afviser køb (PDT eller samme-dags rotation). "
            "Prøv senere eller næste handelsdag.",
        )

    try:
        if last_px and last_px > 0:
            # Sidste luk kan være lavere end markedsordrens udfyldelse → buffer mod utilstrækkelig kontant ved udfyldelse.
            conservative_px = last_px * 1.03
            qty = notional / conservative_px * 0.998
            qty = math.floor(qty * 1000) / 1000
            if qty > 0:
                mr = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                mr = MarketOrderRequest(
                    symbol=symbol,
                    notional=round(notional, 2),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
        else:
            mr = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        tc.submit_order(order_data=mr)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ordre fejlede for %s: %s", symbol, exc)
        raise

    qty_val = getattr(mr, "qty", None)
    _append_trade_activity_csv(
        {
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "event": "OPEN_BUY",
            "symbol": symbol,
            "amount_usd": round(notional, 2),
            "qty": float(qty_val) if qty_val is not None else "",
            "predicted_gain_pct": predicted_gain_pct,
            "cash_before": round(cash_balance, 2),
            "buying_power_before": round(am["buying_power"], 2),
            "daytrading_buying_power_before": round(am["daytrading_buying_power"], 2),
            "regt_buying_power_before": round(am["regt_buying_power"], 2),
            "non_marginable_buying_power_before": round(am["non_marginable_buying_power"], 2),
            "notes": (
                "amount_usd=planlagt købsbudget (broker-loft − reserve); loft=min(cash,BP[,DTBP eller RegT+NMBP])"
            ),
        }
    )

    _write_state(
        {
            "symbol": symbol,
            "predicted_gain_pct": predicted_gain_pct,
            "opened_on": date.today().isoformat(),
        }
    )
    logger.info(
        "Kører nu paper-position i %s (forudsagt open→close %.4f pct). Cash %.2f BP %.2f",
        symbol,
        predicted_gain_pct,
        cash_balance,
        am["buying_power"],
    )


__all__ = ["rotate_to_symbol"]
