import torch.nn as nn


class ModelDigits(nn.Module):
    def __init__(
        self,
        input_dim=64,
        out_dim=10,
        hidden_dim=200,
        dropout=0.2,
        batch_norm_flag=True,
    ):
        super().__init__()

        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_dim)

        self.relu = nn.ReLU()
        self.dp = nn.Dropout(dropout)

        self.batch_norm_flag = batch_norm_flag

    def forward(self, x):
        x = self.lin_in(x)
        if self.batch_norm_flag:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.lin_out(x)
        return x
