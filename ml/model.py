import torch
from torch import nn


class ActivityLSTM(nn.Module):
    """Stacked LSTM over a 2.56 s window, classifying from the final hidden state."""

    def __init__(
        self,
        n_channels: int = 6,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 6,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_channels, hidden, num_layers=layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(self.dropout(sequence[:, -1]))


class Probabilities(nn.Module):
    """What ships: the app shows confidence, so softmax belongs inside the exported graph."""

    def __init__(self, model: ActivityLSTM) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=-1)
