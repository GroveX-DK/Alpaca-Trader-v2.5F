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
    ) -> None:
        """Punkt-estimat af næste dags open→close-afkast (%) trænet med Huber-loss."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        return: (batch, 1)  — punkt-estimat.
        """
        _, (hn, _) = self.lstm(x)
        last = hn[-1]
        return self.head(last)
