# Alpaca Trader (stock_predictor)

Daglig LSTM der rangerer en watchlist og kører paper-handel via Alpaca.

## Modelmål

Modellen forudsiger **næste handelsdags intradag-afkast** i procent:

`(close_{t+1} / open_{t+1} - 1) × 100`

Features slutter på dag `t`; score er estimatet for sessionen på dag `t+1`.

## Kørsel

```bash
# Efter ændring af label, features eller hyperparametre — altid træn først:
python -m stock_predictor.main --train

# Inferens + paper-rotation til stærkeste ticker:
python -m stock_predictor.main --run
```

Valgfri lang historik: `python -m stock_predictor.tools.import_watchlist_csv_to_cache` før `--train`.

Kopiér `.env.example` til `.env` og udfyld Alpaca-nøgler.
