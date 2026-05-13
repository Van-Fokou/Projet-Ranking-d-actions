import time
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader





# ---------------------------
# Dataset
# ---------------------------
class ListNetDataset(Dataset):
    """
    Each item is a (X, y) pair for a given date
    X: numpy array shape (n_items, n_features)
    y: numpy array shape (n_items,)
    """
    def __init__(self, df, feature_cols, target_col):
        self.groups = []
        # Keep chronological ordering
        for date, group in df.groupby('date'):
            # group is in arbitrary order — keep deterministic order by ticker to be stable
            group_sorted = group.sort_values('ticker')
            X = group_sorted[feature_cols].values.astype(np.float32)
            y = group_sorted[target_col].values.astype(np.float32)
            tickers = group_sorted['ticker'].values.tolist()
            self.groups.append((X, y, tickers, date))
        # sort groups by date ascending (chronological)
        self.groups.sort(key=lambda tup: pd.to_datetime(tup[3]))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        X, y, tickers, date = self.groups[idx]
        return torch.from_numpy(X), torch.from_numpy(y), tickers, str(date)

# collate fn not needed because batch_size=1 and variable sizes handled per sample



# ---------------------------
# Model
# ---------------------------
class ListNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # output a single score per item
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (n_items, n_features)
        scores = self.net(x).squeeze(-1)  # (n_items,)
        return scores


# ---------------------------
# Residual MLP Block
# ---------------------------
class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        return self.norm(x + h)   # skip connection


# ---------------------------
# Self-Attention intra-groupe (optionnelle)
# ---------------------------
class GroupSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        att_out, _ = self.att(x, x, x)
        return self.norm(x + att_out)


# ---------------------------
# ListNet Attention Model
# ---------------------------
class ListNetModelAttention(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.1, use_attention=True, num_layers=3, att_heads=4):
        super().__init__()

        self.use_attention = use_attention

        # Build MLP
        layers = []
        last_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(last_dim, hdim))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            last_dim = hdim

        self.mlp = nn.Sequential(*layers)

        # projection
        self.linear_in = nn.Linear(input_dim, last_dim)
        self.norm_in = nn.LayerNorm(last_dim)

        # Attention
        if self.use_attention:
            self.att = GroupSelfAttention(last_dim, num_heads=att_heads)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(last_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.out = nn.Linear(last_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x: (batch_size, n_assets, n_features)
        returns: (batch_size, n_assets)
        """
        #h = self.mlp(x)
        h = F.silu(self.norm_in(self.linear_in(x)))

        if self.use_attention:
            h = self.att(h)

        # residual MLP layers
        for block in self.blocks:
            h = block(h)
            
        out = self.out(h).squeeze(-1)
        return out