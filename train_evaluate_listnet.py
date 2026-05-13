import os
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Optional: spearman from scipy if available, else fallback
try:
    from scipy.stats import spearmanr
    _has_scipy = True
except Exception:
    _has_scipy = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8
CLIP_GRAD = 5.0

NDCG_K = 5
TOP_K = 5
OUT_DIR = "listnet_results"

BEST_MODEL_PATH = os.path.join(OUT_DIR, "listnet_best.pth")
RESULTS_PICKLE = os.path.join(OUT_DIR, "results.pkl")






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







# ---------------------------
# ListNet loss
# ---------------------------
def listnet_loss(scores, targets):
    """
    ListNet: cross-entropy between softmax(targets) and softmax(scores)
    scores, targets: torch tensors (n_items,)
    """
    # If targets are all equal or negative/pos mix, softmax still defined
    targets_exp = torch.exp(targets)

    P_y = torch.softmax(targets_exp, dim=0)
    P_s = torch.log_softmax(scores, dim=0)
    loss = -torch.sum(P_y * P_s)
    return loss


def listnet_loss_2(y_pred, y_true):
    # scale to avoid exploding exp
    y_scaled = 10 * y_true

    # exponentiate
    y_true_trans = torch.exp(y_scaled)

    # avoid infinite values
    y_true_trans = torch.clamp(y_true_trans, max=50)

    P_true = torch.softmax(y_true_trans, dim=0)
    P_pred = torch.softmax(y_pred, dim=0)

    return - torch.sum(P_true * torch.log(P_pred + 1e-12))





def listnet_loss_ranked(y_pred, y_true, tau=2.0):
    # Compute ranks (descending order)
    ranks = torch.argsort(torch.argsort(-y_true))
    ranks = ranks.float()

    # Temperature scaling
    P_true = torch.softmax(tau * ranks, dim=0)

    P_pred = torch.softmax(y_pred, dim=0)

    return -torch.sum(P_true * torch.log(P_pred + 1e-12))




# ---------------------------
# Train / Eval loops
# ---------------------------
def evaluate_model(model, optimizer, dataloader, device, k_ndcg=NDCG_K, topk=TOP_K):
    model.eval()
    ndcg_list = []
    topk_list = []
    spearman_list = []
    losses_val = []

    all_scores = []
    all_true = []
    all_dates = []
    all_tickers = []
    all_features = []


    
    with torch.no_grad():
        for X, y, tickers, date in dataloader:

            X_np = X.squeeze(0).numpy()
            y_np = y.squeeze(0).numpy()
            
            X = X.squeeze(0).to(device)
            y = y.squeeze(0).to(device)
            scores = model(X)
            scores_np = scores.cpu().numpy()
            y_np = y.cpu().numpy()

            loss_val = listnet_loss_2(scores, y)

            #total_loss_val += float(loss_val.item())
            #n_groups += 1
            #pbar.set_postfix({'loss': f"{total_loss_val / n_groups:.6f}"})

            losses_val.append(float(loss_val.item()))

            ndcg_val = ndcg_at_k(y_np, scores_np, k=k_ndcg)
            topk_val = top_k_accuracy_tensors(scores, y, k=topk)
            spear = spearman_corr(y_np, scores_np)

            ndcg_list.append(ndcg_val)
            topk_list.append(topk_val)
            spearman_list.append(spear)
            

            all_scores.append(scores_np)
            all_true.append(y_np)
            all_dates.append(str(date))
            all_tickers.append(tickers)
            all_features.append(X_np)


    metrics = {
        'ndcg_mean': float(np.nanmean(ndcg_list)) if ndcg_list else 0.0,
        'ndcg_std': float(np.nanstd(ndcg_list)) if ndcg_list else 0.0,
        'topk_mean': float(np.nanmean(topk_list)) if topk_list else 0.0,
        'topk_std': float(np.nanstd(topk_list)) if topk_list else 0.0,
        'spearman_mean': float(np.nanmean(spearman_list)) if spearman_list else 0.0,
        'spearman_std': float(np.nanstd(spearman_list)) if spearman_list else 0.0,
        'n_groups': len(ndcg_list)
    }

    collected = {
        'all_scores': all_scores,
        'all_true': all_true,
        'all_dates': all_dates,
        'all_tickers': all_tickers,
        'all_features': all_features,
        'ndcg_list': ndcg_list,
        'topk_list': topk_list,
        'losses_val': losses_val,
        'spearman_list': spearman_list
    }


    return metrics, collected
    



def train_loop(model, optimizer, scheduler, train_loader, val_loader, device,
               epochs=EPOCHS, patience=PATIENCE, clip_grad=CLIP_GRAD):
    best_ndcg = -np.inf
    best_epoch = -1
    epochs_no_improve = 0
    start_time = time.time()

    train_losses = []
    train_ndcgs = []
    val_ndcgs = []
    val_topks = []
    val_spearmans = []
    k_ndcg = NDCG_K


    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_groups = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (train)", leave=False)
        for X, y, tickers, date in pbar:
            X = X.squeeze(0).to(DEVICE)
            y = y.squeeze(0).to(DEVICE)

            X_np = X.squeeze(0).numpy()
            y_np = y.squeeze(0).numpy()

            scores = model(X)
            loss = listnet_loss_2(scores, y)

            #scores_np = scores.cpu().numpy()
            #y_np = y.cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()


            optimizer.zero_grad()
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss += float(loss.item())
            n_groups += 1
            pbar.set_postfix({'loss': f"{total_loss / n_groups:.6f}"})

            epoch_loss = total_loss / max(1, n_groups)
            train_losses.append(epoch_loss)
            train_ndcgs.append(ndcg_at_k(y_np, scores_np, k=k_ndcg))

        # scheduler step based on epoch (if provided)
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        # Validation
        val_metrics, collect = evaluate_model(model, optimizer, val_loader, device, k_ndcg=NDCG_K, topk=TOP_K)
        #val_ndcg = val_metrics['ndcg_mean']


        val_ndcgs.append(val_metrics['ndcg_mean'])
        val_topks.append(val_metrics['topk_mean'])
        val_spearmans.append(val_metrics['spearman_mean'])


        PRINT_EVERY_EPOCH = True
        

        if PRINT_EVERY_EPOCH:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} | Train loss avg: {total_loss / max(1, n_groups):.6f} | Val NDCG@{NDCG_K}: {val_metrics['ndcg_mean']:.6f} | groups val: {val_metrics['n_groups']} | elapsed: {elapsed:.1f}s")

        # Early stopping & checkpoint
        if val_metrics['ndcg_mean'] > best_ndcg + 1e-8:
            best_ndcg = val_metrics['ndcg_mean']
            best_epoch = epoch
            epochs_no_improve = 0
            # save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ndcg': best_ndcg
            }, BEST_MODEL_PATH)
            if PRINT_EVERY_EPOCH:
                print(f"  -> New best model saved (Val NDCG {best_ndcg:.6f}) at epoch {epoch}")
        else:
            epochs_no_improve += 1
            if PRINT_EVERY_EPOCH:
                print(f"  No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs (best epoch: {best_epoch}, best val NDCG: {best_ndcg:.6f})")
            break

        history = {
        'train_losses': train_losses,
        'train_ndcgs': train_ndcgs,
        'val_losses': collect['losses_val'],
        'val_ndcgs': val_ndcgs,
        'val_topks': val_topks,
        'val_spearmans': val_spearmans,
        'best_epoch': best_epoch,
        'best_ndcg': best_ndcg
     }

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s. Best val NDCG@{NDCG_K}: {best_ndcg:.6f} at epoch {best_epoch}")
    return history    




