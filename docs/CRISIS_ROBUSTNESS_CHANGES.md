# Crisis-robustness overhaul — change log & revert guide

Branch: **`crisis-robustness`** (off `main`). Goal: stop the strategy losing ~70% in crashes
(COVID), via long/short construction, crisis-signal features, and smarter training. Plan:
`~/.claude/plans/this-is-a-pytorch-synthetic-planet.md`.

This file is the single source of truth for **what changed** so it can all be undone.

## How to revert

- **Soft (keep code, restore behavior):** set every flag in the "Krise-robusthed" section of
  `stock_predictor/config.py` to `False`. The pipeline then behaves exactly as `main`.
- **Model only:** copy the baseline back over the live checkpoint:
  - `cp stock_predictor/models/baseline_backup/lstm_stock.pt stock_predictor/models/lstm_stock.pt`
  - `cp stock_predictor/models/baseline_backup/feature_scaler.joblib stock_predictor/models/feature_scaler.joblib`
- **Hard (everything, incl. cache):** `git checkout main` — restores code, the per-symbol
  parquet cache, and the committed model as they were before this branch.

## New config flags (all default to current behavior)

In `stock_predictor/config.py`, section "Krise-robusthed":

| Flag | Default | Effect when True |
|------|---------|------------------|
| `MACRO_FEATURES_ENABLED` | `False` | Adds market-wide crisis-signal features (Option 3) |
| `CRISIS_OVERSAMPLE_ENABLED` | `False` | Weights training days by VIX (Option 4A) |
| `UNCERTAINTY_HEAD_ENABLED` | `False` | Quantile head + pinball loss (Option 4B) |
| `WALK_FORWARD_ENABLED` | `False` | Walk-forward retrain mode (Option 4C, heavy) |
| `LONG_SHORT_ENABLED` | `False` | Long/short market-neutral backtest (Option 2) |
| `LONG_SHORT_LIVE_ENABLED` | `False` | Long/short in the live paper trader (Option 2) |

Tuning knobs alongside each (VIX refs, top/bottom-k, vol target, quantiles, etc.).

## Change log (chronological)

### Stage 0 — Safety scaffolding (done)
- Created branch `crisis-robustness` off `main`.
- Backed up model + scaler → `stock_predictor/models/baseline_backup/` (lstm_stock.pt,
  feature_scaler.joblib). (Note: repo already had `models/backup_pre_yfinance_features/` and
  `*.feat22.bak` from earlier work — left untouched.)
- `config.py`: appended the "Krise-robusthed" flag section (all default False).
- Added this changelog (`docs/CRISIS_ROBUSTNESS_CHANGES.md`).

### Stage 1 — Crisis-signal features (Option 3) — done (code); full cache backfill deferred to Stage 4
- **New file `stock_predictor/macro_features.py`** — builds one market-wide frame (date →
  cols) and caches it to `config.MACRO_CACHE_PATH` (`cache/macro/macro_features.parquet`):
  `vix_ts_slope` (^VIX/^VIX3M), `vvix_level` (^VVIX/100), `breadth_pct` (% of watchlist >200d
  MA, from cache), `xsec_corr` (mean pairwise corr of 21d returns, implied-corr estimator,
  from cache), `credit_ratio_chg` (5d Δ HYG/LQD), `move_chg` (5d Δ ^MOVE). VIX-family/MOVE/
  HYG/LQD via yfinance. **Put/call dropped** — no free daily source; ^MOVE covers bond-vol.
- **New file `stock_predictor/tools/backfill_macro_features.py`** — builds/loads the frame and
  materializes the macro columns into each per-symbol parquet (mirrors `rebuild_cache_features`).
- `config.py`: added `MACRO_FEATURE_COLUMNS` (6), `MACRO_FEATURE_NEUTRAL`, `MACRO_CACHE_PATH`,
  `MACRO_BREADTH_MA_DAYS=200`, `MACRO_CORR_WINDOW_DAYS=21`; `N_FEATURES` now computed
  = `_BASE_N_FEATURES(22) + (6 if MACRO_FEATURES_ENABLED else 0)`.
- `feature_engineer.py`: `FEATURE_COLUMNS` = base 22 + macro (only when flag on); macro
  passthrough in `_feature_frame` (neutral-fill, never drops rows); `build_dataset_frame` gained
  a `macro` param.
- `data_fetcher.py`: `_read_cache_parquet` now carries the macro columns; `_dataset_for_cache`
  injects the cached macro frame on incremental merges (lazy `_macro_frame_for_cache`, gated).
- **Verified:** flag OFF → 22 features, byte-identical behavior. Flag ON → 28 features, no NaN,
  no row loss; COVID window shows breadth→0.05 / xsec_corr→0.78 / vix_ts_slope→1.34 /
  move_chg→0.57 / credit_ratio_chg→−0.09 (all crisis signals fire).
- **TODO before retrain (Stage 4):** run the full backfill across all cache files:
  `python -m stock_predictor.tools.backfill_macro_features --use-cached-frame`
  (smoke-tested on AAPL/MSFT only so far). Rewrites ~233 parquets — git branch is the backup.

### Stage 2 — Training methodology (Option 4 A+B+C) — done (code); needs retrain (Stage 4)
- **`model.py`**: `DailyLSTM` gained `n_outputs` (1 = point/Huber; >1 = quantile head). Head is
  `Linear(hidden, n_outputs)`; forward returns `(batch, n_outputs)`.
- **`train.py`**:
  - **A (oversampling):** `_crisis_sample_weights` (w = clip(VIX/`VIX_REF`, 1, `MAX_WEIGHT`) from
    raw vix_close) → `WeightedRandomSampler` on the train loader when `CRISIS_OVERSAMPLE_ENABLED`.
  - **B (uncertainty):** `_PinballLoss` + `_make_criterion` (pinball when `UNCERTAINTY_HEAD_ENABLED`,
    else HuberLoss); `_model_n_outputs`; model built with `n_outputs`; checkpoint now stores
    `n_outputs` + `quantiles`; baseline-reset also triggers on `n_outputs` mismatch.
  - **C (walk-forward):** `_train_fold` (compact per-fold trainer) + `train_model_walk_forward`
    (expanding-window retrain, OOS-scores each year via `backtest._predict_symbol`, stitches one
    equity curve, saves `output/backtests/walkforward_*.csv/.json` + plot). Gated by
    `WALK_FORWARD_ENABLED`; heavy — GPU recommended.
- **`predict.py`**: `_load_bundle` builds model with `n_outputs`, attaches `_quantiles`; new
  `reduce_outputs` (median score + q90−q10 band) and `quantile_indices`; split scoring into
  `_score_watchlist`; added `predict_rankings_detailed` (returns `(sym, score, band)`) for the
  live long/short. `predict_rankings` unchanged externally (median score).
- **`backtest.py`**: `_predict_symbol` now returns `(pred, actual, band)` via `reduce_outputs`;
  both callers capture `band_cols`/`band_df` (regime path filters band with the symbol-count mask).
- **`main.py`**: added `--walk-forward` (routes to `train_model_walk_forward`).
- **Verified:** all modules import; CLI parses; unit checks for multi-output forward, pinball loss,
  `reduce_outputs` (median+band), and a synthetic `_train_fold` run (pinball + weighted sampler) pass.

### Stage 3 — Long/short (Option 2) — done (code); live path paper-only, retrain for sizing
- **`backtest.py`**: `_simulate_long_short(pred_df, actual_df, band_df)` — long top-k / short
  bottom-k, dollar-neutral; daily raw % = mean(long actual) − mean(short actual); exposure
  vol-targeted to `LONG_SHORT_TARGET_VOL_PCT` (prev-day realized vol, 63d window, no
  look-ahead) and shrunk by the uncertainty band, capped at `LONG_SHORT_MAX_GROSS`.
  `_regime_report` gained a `strategies` param; `run_regime_backtest` adds a `long_short`
  variant + overlays it on `_plot_regime` when `LONG_SHORT_ENABLED`.
- **`trader.py`**: `rebalance_long_short(longs, shorts, exposure=)` — closes all, splits budget
  equally across legs (dollar-neutral), BUY longs (notional) / SELL-short bottom names (whole
  shares), skips non-shortable (`_is_shortable`), logs OPEN_LONG/OPEN_SHORT. Paper only.
  Helpers `_last_prices`, `_is_shortable`. Reuses `_finalize_open_trade`/`_trade_budget_usd`.
- **`main.py`**: `--run` routes to `rebalance_long_short` (top-k/bottom-k from
  `predict_rankings_detailed`) when `LONG_SHORT_LIVE_ENABLED`; else unchanged single-symbol path.
- **Verified:** imports OK; synthetic `_simulate_long_short` runs, finite equity, vol +
  confidence sizing paths exercised (wide crash-window bands cut exposure).
### Stage 4 — Retrain + evaluate — data prep DONE; retrain/eval are heavy user-run jobs

Done in-session:
- **Flag-off regression PASS:** `run_backtest(2025)` → +66.27% / 250 days through the updated
  `_predict_symbol` 3-tuple path = identical to pre-change behavior.
- **Full macro backfill RAN:** `backfill_macro_features --use-cached-frame` materialized the 6
  macro columns into **199/199** cache parquets (built+cached `cache/macro/macro_features.parquet`
  first). Non-null counts track each source's history (VIX3M/VVIX ~2006-07, HYG/LQD credit ~2007,
  MOVE ~2002; shorter-history tickers capped at their own length). Cache is retrain-ready.

To activate and evaluate the crisis-robustness model, run these on the `crisis-robustness` branch:

1. ~~Backfill macro columns~~ — **already done** (re-run `backfill_macro_features` without
   `--use-cached-frame` only if you want to refresh the yfinance frame first).
2. **Flip the flags** in `config.py` "Krise-robusthed" section:
   `MACRO_FEATURES_ENABLED=True`, `UNCERTAINTY_HEAD_ENABLED=True`,
   `CRISIS_OVERSAMPLE_ENABLED=True`, `LONG_SHORT_ENABLED=True`
   (leave `WALK_FORWARD_ENABLED`/`LONG_SHORT_LIVE_ENABLED` off until validated). `N_FEATURES`
   auto-becomes 28.
3. **Back up the live model** (already in `models/baseline_backup/`), then **retrain**:
   ```
   python -m stock_predictor.main --train
   ```
   Produces a 28-feature quantile-head checkpoint (stores `n_outputs`/`quantiles`). Heavy on CPU.
4. **Evaluate** against regimes (esp. the COVID-crash row) — compare to a baseline run:
   ```
   python -m stock_predictor.main --regime-backtest 2015
   ```
   The plot/CSV now include the `long_short` curve. Success = COVID max-drawdown materially
   reduced vs the −70% baseline without wrecking calm-market returns.
5. **(Optional, GPU)** walk-forward rigor: set `WALK_FORWARD_ENABLED=True`, then
   `python -m stock_predictor.main --walk-forward`.
6. **(Optional, paper)** live long/short: set `LONG_SHORT_LIVE_ENABLED=True`, then
   `python -m stock_predictor.main --run` (paper account; verify both legs open/close).

If results disappoint: revert via the "How to revert" section above (flags off → model backup
→ or `git checkout main`).

**Regression check (flags off):** single-year `run_backtest(2025)` runs end-to-end through the
updated `_predict_symbol` 3-tuple path — confirms no behavior change with flags off.

### Stage 5 — Adjusted close (split + dividend) input data — done (code); cache rebuild + retrain operational

Previously OHLCV was fetched from Alpaca with **no `adjustment`** → `Adjustment.RAW`. Raw prices
inject fake returns: splits look like ~-75% crashes, dividend ex-dates leave ~0.5–1% gaps — pure
noise to a return/momentum model. Switched the model-feeding fetches to **`Adjustment.ALL`**
(split + dividend = total-return series). Intraday **open→close %** (the training target,
`targets_next_day_open_to_close_pct`) is **unchanged** — adjustment is a constant per-day factor —
so only cross-day features move.

- `config.py`: new flag `OHLCV_ADJUSTMENT = "all"` (next to `OHLCV_CACHE_ENABLED`).
  Values: `"all"` | `"split"` | `"raw"`.
- `data_fetcher.py`:
  - imports `Adjustment`; both `StockBarsRequest` sites (`_fetch_symbol_range`,
    `_fetch_all_symbols_batch`) now pass `adjustment=Adjustment(config.OHLCV_ADJUSTMENT)`.
  - `_merge_trim_save_symbol`: when adjustment ≠ RAW, **re-fetch the whole window** and overwrite
    the cache instead of splicing `backfill + old cache + tail`. Adjusted prices rescale
    retroactively on new corporate actions, so a splice would create a silent scale
    discontinuity. RAW keeps the old incremental splice (clean revert).
- **`trader.py` deliberately left RAW** — its three `StockBarsRequest` sites size orders
  (dollars→shares) and mark P&L at actual tradable prices; adjusting them would corrupt qty/P&L.

**Operational (one-time):** delete the raw per-symbol cache so it rebuilds adjusted, then retrain:
```
rm stock_predictor/cache/*.parquet     # keep cache/news/ and cache/macro/
python -m stock_predictor.main --train # features changed → retrain before --run/backtest
```
Commit the regenerated parquet files (repo tracks them).

**Revert:** set `OHLCV_ADJUSTMENT = "raw"` and rebuild the cache (the incremental splice path is
restored automatically for RAW).

### Stage 5 — "matches SPY + crashes" fixes: trainable model, directional single-name, honest costs (done)

Diagnosis: the strategy matched SPY because the model was effectively **untrained**
(`EPOCHS=1`, `EARLY_STOP_PATIENCE=1`) on a **2000-day** LSTM window — windows overlapped 99.95%
(stride-1 `_build_index`) so it overfit inside one epoch (the "val rises after epoch 1" symptom),
and the cost-free backtest flattered a ~zero edge. Crashes came from **all-in, long-only** on one
name. There is **no look-ahead bias** — `targets_next_day_open_to_close_pct` uses `shift(-1)`, so
day-`t` features (incl. day-`t` close) predict day-`t+1`'s open→close, traded on day `t+1`.

**Training health** (`config.py`, `train.py`):
- `SEQ_LEN` (`LOOKBACK_DAYS`) `2000 → 40`; `EPOCHS` `1 → 80`; `EARLY_STOP_PATIENCE` `1 → 10`;
  `WEIGHT_DECAY` `1e-5 → 1e-4`; `LR_SCHEDULER_PATIENCE` `5 → 3`; `CRISIS_OVERSAMPLE_MAX_WEIGHT`
  `5 → 2.5`.
- New `TRAIN_WINDOW_STRIDE` (default `1`): optional per-symbol training-window stride
  (`_subsample_train_stride`) to cut overlap; val/inference stay stride-1.
- `train.py` now logs **`val_dir_acc`** and **`val_ic`** (Pearson) per epoch so "accuracy" is
  measurable, not just loss.

**Directional single-name strategy** (the requested rule: "best stock, long if up / short if down"):
- `_simulate` picks the largest **|pred|** name; **long if pred>0, short if pred<0** (short P&L =
  −realized). New columns `best_side`, `best_signed_actual`. Long-only top-2/3/avg kept as
  references. `predict.py` adds `predict_best_directional()`; `trader.py` `rotate_to_symbol(...,
  side=)` can short (shortable check, whole shares); `main.py` routes to it behind a flag.

**Honest costs** (`_simulate`, `_simulate_long_short`): subtract a round-trip cost
(`BACKTEST_COST_BPS_PER_SIDE × 2`) on every trading day.

New flags (default = current intended behavior on this branch):

| Flag | Default | Effect |
|------|---------|--------|
| `DIRECTIONAL_ENABLED` | `True` | `_simulate` "best" = largest-|pred| directional single name |
| `DIRECTIONAL_MIN_ABS_PCT` | `0.0` | Confidence gate: sit in cash when best `|pred|` below this |
| `DIRECTIONAL_LIVE_ENABLED` | `False` | Route live `--run` through the directional (short-capable) path |
| `BACKTEST_COST_BPS_PER_SIDE` | `5.0` | Per-side cost (bps) in both backtest sims; `0` = old gross |

**Operational:** `SEQ_LEN` changed ⇒ the old checkpoint is incompatible (inference guards on it).
**Retrain before backtest/run:** `python -m stock_predictor.main --train` (GPU). Then
`--backtest <year>` and `--regime-backtest <start>`; check val improves for many epochs, the
net-of-cost curve vs SPY, and that COVID/2022 drawdown shrinks (shorts fire on big predicted drops).

**Revert:** `DIRECTIONAL_ENABLED=False` restores top-long selection; `BACKTEST_COST_BPS_PER_SIDE=0`
restores gross numbers; restore `SEQ_LEN=2000`/`EPOCHS`/`EARLY_STOP_PATIENCE` and retrain for the
old training regime.

### Stage 6 — Two new inputs: today's open gap + oil price (done — code); cache backfill + retrain operational

Rationale: the live pipeline runs **right after the open**, so today's open is known — but the model
never saw it (`required_last_bar_date` ends the fetch at yesterday's complete bar, and the target is
the *next* day's open→close). And oil moves the whole market (esp. the ~19 energy names) yet had no
input. Both added as scale-invariant features, flag-gated. **`N_FEATURES` 28 → 31.**

**1) Today's-open gap (`next_open_gap`)** — `ln(open_{t+1} / close_t)`, the overnight gap into the
prediction day's open. Same time-alignment as the target (open→close of day t+1), so **no leakage**:
the gap uses only the prediction day's open, which is known at trade time.
- `config.py`: new flag `OPEN_FEATURE_ENABLED = True` (Option 5); `N_FEATURES` now
  `= _BASE_N_FEATURES(22) + (1 if OPEN_FEATURE_ENABLED) + (len(MACRO_FEATURE_COLUMNS) if MACRO_FEATURES_ENABLED)`.
- `feature_engineer.py`: `_OPEN_FEATURE_COLUMNS = ("next_open_gap",)` folded into `FEATURE_COLUMNS`
  between base and macro (only when flag on); `_feature_frame` computes
  `out["next_open_gap"] = np.log(open_.shift(-1) / close_nz)`. The last history row is NaN (no t+1)
  and is dropped by `engineer_features` — it was never a training sample (target also NaN there).
  Recomputed from OHLCV every time, so **no cache backfill needed** for this column.
- `data_fetcher.py`: new `fetch_todays_open(...)` (live daily-bar fetch that passes
  `required_last_bar_date`, returns `{sym: today_open}`, `{}` before open / no keys) and
  `append_todays_open_row(...)` (appends a phantom today-row carrying only the open, so the last
  complete bar gets `next_open_gap`; the phantom row is dropped by `dropna`; missing open ⇒ gap 0).
- `predict.py`: `_score_watchlist` fetches today's open once for the watchlist and injects it per
  symbol before `engineer_features` (gated by `OPEN_FEATURE_ENABLED`). Training/backtest need no
  change — full history already has each row's real next-day open.

**2) Oil price (`oil_log_ret`, `oil_vol_annual_pct`)** — WTI front-month **`CL=F` via yfinance**
(same source as the other macro features). Two market-wide columns, identical value per date for
every ticker, mirroring the stock `log_ret` / `vol_annual_pct`:
- `config.py`: appended `oil_log_ret`, `oil_vol_annual_pct` to `MACRO_FEATURE_COLUMNS` (now 8) and
  `MACRO_FEATURE_NEUTRAL` (`0.0` / `35.0`). Gated by the existing `MACRO_FEATURES_ENABLED`.
- `macro_features.py`: `build_macro_frame` fetches `CL=F` and adds
  `oil_log_ret = ln(c/c.shift(1))`, `oil_vol_annual_pct = rolling_annualized_log_vol_pct(c)`
  (reused from `feature_engineer`). Flows through the existing cache + per-ticker materialization.

New flag (default = active on this branch):

| Flag | Default | Effect |
|------|---------|--------|
| `OPEN_FEATURE_ENABLED` | `True` | Adds `next_open_gap` (today's open→prior-close gap) to the feature set |

**Operational — `--train` is one command:** oil is materialized into the cache **automatically** at the
start of training. `train_model()` calls `macro_features.ensure_macro_oil_cache()`, which rebuilds the
market-wide frame (incl. `CL=F`) when stale, sets the in-process macro-frame cache
(`data_fetcher.set_macro_frame_cache`), and backfills the oil columns into any per-symbol parquet whose
schema is missing them (cheap pyarrow schema gate → first run rewrites ~199 parquets, later runs skip).
```
python -m stock_predictor.main --train                    # auto-refreshes oil, then 31-feature checkpoint (GPU desktop)
```
- `next_open_gap` needs no backfill (derived from cached OHLCV). The old 28-feature checkpoint is
  auto-rejected by the existing `n_features` guard. Commit the regenerated parquets (as in Stage 5).
- The standalone tool still works as an explicit/manual path (e.g. to refresh the cache without training):
  `python -m stock_predictor.tools.backfill_macro_features`.
- Refresh failures (e.g. offline) never block training — the feature layer falls back to neutral oil.
- Not wired into `--backtest` / `--regime-backtest` / `--run` / walk-forward: they read whatever the
  cache holds, so run `--train` (or the tool) first to populate oil.

**Revert:** set `OPEN_FEATURE_ENABLED = False` **and** remove `oil_log_ret`/`oil_vol_annual_pct` from
`MACRO_FEATURE_COLUMNS` (+ their neutral entries) → back to the 28-feature set; retrain (or restore
the model backup).
