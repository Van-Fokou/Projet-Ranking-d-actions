import os
import random
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

# Optional: spearman from scipy if available, else fallback
try:
    from scipy.stats import spearmanr
    _has_scipy = True
except Exception:
    _has_scipy = False



# ---------------------------
# Utility metrics
# ---------------------------
def ndcg_at_k(y_true, y_pred, k=5):
    """
    Compute NDCG@k for a single list
    y_true, y_pred: 1D numpy arrays or lists (same length)
    """
    # convert to numpy
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)

    # get ordering of predictions
    order_pred = np.argsort(-y_pred)
    ranked_true = y_true[order_pred][:k]

    # DCG
    gains = (2 ** ranked_true - 1)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)

    # ideal DCG
    order_ideal = np.argsort(-y_true)
    ideal_true = y_true[order_ideal][:k]
    ideal_gains = (2 ** ideal_true - 1)
    idcg = np.sum(ideal_gains / discounts)
    return dcg / idcg if idcg > 0 else 0.0

def top_k_accuracy_tensors(scores, targets, k=5):
    """
    scores, targets: 1D torch tensors
    returns fraction of top-k true recovered in top-k pred
    """
    _, pred_idx = torch.topk(scores, k=min(k, scores.size(0)))
    _, true_idx = torch.topk(targets, k=min(k, targets.size(0)))
    set_pred = set(pred_idx.cpu().numpy().tolist())
    set_true = set(true_idx.cpu().numpy().tolist())
    return len(set_pred & set_true) / float(min(k, scores.size(0)))

def spearman_corr(y_true, y_pred):
    """
    robust spearman: try scipy, fallback to pandas ranks + Pearson
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return 0.0
    if _has_scipy:
        coef, _ = spearmanr(y_true, y_pred)
        if np.isnan(coef):
            return 0.0
        return float(coef)
    else:
        # fallback
        r_true = pd.Series(y_true).rank().values
        r_pred = pd.Series(y_pred).rank().values
        # Pearson on ranks
        num = np.cov(r_true, r_pred, ddof=0)[0, 1]
        den = np.std(r_true) * np.std(r_pred)
        return float(num / den) if den > 0 else 0.0