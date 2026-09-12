import torch
import torch.nn as nn


class GlossModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        scores = self.attention(output)

        weights = torch.softmax(
            scores,
            dim=1
        )

        output = torch.sum(
            output * weights,
            dim=1
        )

        output = self.dropout(output)

        return self.fc(output)