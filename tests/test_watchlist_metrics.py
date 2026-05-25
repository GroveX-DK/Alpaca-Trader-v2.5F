"""Smoke-tests for Watchlist-metriker mod reference-CSV."""

from __future__ import annotations

import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestWatchlistMetrics(unittest.TestCase):
    def test_aapl_matches_watchlist_csv(self) -> None:
        from stock_predictor.watchlist_metrics import assert_watchlist_metrics_match_csv

        csv_path = _PROJECT_ROOT / "output" / "Watchlist" / "AAPL.csv"
        self.assertTrue(csv_path.is_file(), f"Mangler {csv_path}")
        assert_watchlist_metrics_match_csv(str(csv_path), n_rows=11_400, rtol=1e-12, atol=1e-9)


class TestFeaturePipeline(unittest.TestCase):
    def test_engineer_features_has_21_columns(self) -> None:
        from pathlib import Path

        import pandas as pd

        from stock_predictor import config
        from stock_predictor.feature_engineer import engineer_features

        pq = Path(__file__).resolve().parent.parent / "stock_predictor" / "cache" / "AAPL.parquet"
        if not pq.is_file():
            self.skipTest("Kør import_watchlist_csv_to_cache for AAPL først")
        df = pd.read_parquet(pq)
        feats = engineer_features(df)
        self.assertEqual(len(feats.columns), int(config.N_FEATURES))
        self.assertEqual(int(config.N_FEATURES), 21)

    def test_lstm_accepts_n_features(self) -> None:
        import torch

        from stock_predictor.model import DailyLSTM

        m = DailyLSTM(n_features=15, hidden_size=32, num_layers=2, dropout=0.1)
        y = m(torch.randn(2, 64, 15))
        self.assertEqual(tuple(y.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
