#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
echo "Running paper trading (inference + paper rotation)..."
echo
python3 -m stock_predictor.main --run
status=$?
echo
echo "Done (exit code $status). Press Enter to close."
read -r _
