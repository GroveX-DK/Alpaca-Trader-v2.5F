# Analysis: Would a Transformer beat the current LSTM for prediction?

> Outcome: **analysis only** — no implementation. This memo captures the assessment and
> the preferred design direction if revisited later.

## Context
The predictor currently uses `DailyLSTM` (`stock_predictor/model.py`):
a 3-layer unidirectional LSTM (hidden 128) whose **final hidden state** feeds a
`Linear(128→1)`. Input is `(batch, 1000, 22)` — 1000 trading days of 22 stationary
features. Target is next-day **open→close return %** (single noisy scalar). Training:
LR `1e-5`, `1` epoch, early-stop patience `1`, heavy weight decay, stride-1 overlapping
windows. See `stock_predictor/config.py`.

## Verdict
A transformer **can** help but is **unlikely to be the highest-leverage change**, and a
naive swap could perform *worse*. Architecture is probably not the bottleneck here.

### Why architecture likely isn't the bottleneck
1. **Low signal-to-noise target.** Next-day intraday return is near the noise floor; no
   architecture fixes an information-limited problem.
2. **Instant overfitting.** LR `1e-5` + 1 epoch + patience 1 implies the model overfits
   almost immediately. A transformer has more capacity → overfits faster → needs even
   stronger regularization just to match the LSTM.
3. **Inflated data via overlap.** Stride-1 windows are highly correlated; effective
   sample size is far below the nominal ~100–250k. Transformers are data-hungry.

### Where a transformer genuinely could help
1. **Better sequence aggregation.** The LSTM collapses 1000 steps into one final vector.
   Attention pooling can selectively weight relevant days (earnings cycles, regime
   shifts) — the most plausible real upside.
2. **Long-range dependencies.** Attention reaches any of the 1000 steps directly; LSTMs
   struggle to carry signal that far.

### Higher-leverage levers (orthogonal to LSTM-vs-transformer)
- Shorter / smarter sequence length (1000 steps is huge for a 1-day target).
- Reducing window overlap (larger train stride) to lower correlated-sample inflation.
- Target definition (horizon, classification vs regression, vol-normalized target).
- Regularization and ensembling.

## Preferred design (if implemented later)
**Transformer encoder + attention pooling** — closest drop-in to the current LSTM:
learned positional encoding over the sequence, a few encoder layers, attention pooling
into the existing regression head. Recommended as an **A/B option** behind a config flag,
benchmarked head-to-head against `DailyLSTM` using the existing per-file backtest /
`--regime-backtest` harness. Keep the LSTM as default until the transformer demonstrably
wins on backtest metrics. Consider pairing with a shorter SEQ_LEN to cut overfitting and
the O(seq_len²) attention cost.

## Next step
None — analysis only. Revisit this memo to scope an A/B implementation if desired.
