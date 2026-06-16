"""PyTorch LSTM til regression af næste handelsdags open→close-afkast (%)."""

from __future__ import annotations

import torch
from torch import nn


class DailyLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_outputs: int = 1,
    ) -> None:
        """``n_outputs``: 1 = punkt-estimat (Huber); >1 = kvantiler (usikkerheds-head).

        Med kvantiler forudsiger hovedet fx 10/50/90-percentilen af næste dags open→close.
        Score til ranking = median (q50); konfidensbånd = q90 − q10 (bredt => usikkert).
        """
        super().__init__()
        self.n_outputs = int(n_outputs)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        return: (batch, n_outputs)  — (batch, 1) i punkt-estimat-tilfældet.
        """
        _, (hn, _) = self.lstm(x)
        last = hn[-1]
        return self.head(last)
