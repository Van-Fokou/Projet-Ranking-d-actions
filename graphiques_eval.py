import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_evaluate_listnet import *

from scipy.stats import spearmanr

# Style pour les graphiques


plt.style.use('seaborn-v0_8')
plt.rcParams.update({
'figure.dpi': 150,
'axes.titlesize': 12,
'axes.labelsize': 10,
'legend.fontsize': 9,
'font.family': 'serif',
'font.serif': 'Times New Roman',
'axes.titleweight': 'bold',
'xtick.labelsize': 9,
'ytick.labelsize': 9,
})



# ---------------------------
# Utilities for collection & plots
# ---------------------------
# Plotting functions adapted from user's module

# Metric settings
NDCG_K = 5
TOP_K = 5

OUT_DIR = "listnet_results"

def plot_loss_curves(train_losses, val_losses): 
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses[:len(val_losses)], label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_ndcg_by_epoch(ndcg_train, ndcg_val): 
    plt.figure(figsize=(8, 4))
    plt.plot(ndcg_train[:len(ndcg_val)], label="Train NDCG")
    plt.plot(ndcg_val, label="Val NDCG")
    plt.xlabel("Epoch")
    plt.ylabel(f"NDCG@{NDCG_K}")
    plt.title("NDCG@K Evolution per Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_distribution_true_pred(y_true, y_pred):    
    plt.figure(figsize=(8, 4))
    sns.kdeplot(np.concatenate(y_true), label="True Returns")
    sns.kdeplot(np.concatenate(y_pred), label="Predicted Scores")
    plt.title("Distribution: True vs Predicted")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_true_vs_pred_scatter(y_true, y_pred):  
    y_true_flat = np.concatenate(y_true)
    y_pred_flat = np.concatenate(y_pred)
    plt.figure(figsize=(8, 4))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=8)
    plt.xlabel("True Returns")
    plt.ylabel("Predicted Scores")
    plt.title("Scatter: True vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_rank_correlation(y_true, y_pred):  
    y_true_flat = np.concatenate(y_true)
    y_pred_flat = np.concatenate(y_pred)
    rank_true = np.argsort(np.argsort(-y_true_flat))
    rank_pred = np.argsort(np.argsort(-y_pred_flat))
    plt.figure(figsize=(8, 4))
    plt.scatter(rank_true, rank_pred, alpha=0.4, s=6)
    plt.xlabel("True Ranks")
    plt.ylabel("Predicted Ranks")
    plt.title("Rank Correlation Scatter")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_rank_correlation_2(y_true, y_pred):  
    rank_true = []
    rank_pred = []
    for item in y_true:
        sort_true = np.argsort(np.argsort(-item))
        rank_true.append(sort_true)
    for item in y_pred:
        sort_pred = np.argsort(np.argsort(-item))
        rank_pred.append(sort_pred)
        
    #y_true_flat = np.concatenate(y_true)
    #y_pred_flat = np.concatenate(y_pred)
    rank_true = np.concatenate(rank_true)
    rank_pred = np.concatenate(rank_pred)
    plt.figure(figsize=(8, 4))
    plt.scatter(rank_true, rank_pred, alpha=0.4, s=6)
    plt.xlabel("True Ranks")
    plt.ylabel("Predicted Ranks")
    plt.title("Rank Correlation Scatter")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_topk_accuracy_over_time(topk_values, dates):   
    plt.figure(figsize=(8, 4))
    plt.plot(dates, topk_values)
    plt.title("Top-K Accuracy Over Time")
    plt.xlabel("Date")
    plt.ylabel("Top-K Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_spearman_over_time(spearman_values, dates):  

    plt.figure(figsize=(8, 4))
    plt.plot(dates, spearman_values)
    plt.xlabel("Date")
    plt.ylabel("Spearman ρ")
    plt.title("Spearman correlation over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_returns_of_top_models(group_table, k=5):  
    df = group_table.copy()
    df_true = df.groupby("date").apply(lambda x: x.sort_values("true_rank", ascending=True)["true_return"].iloc[:k].mean())
    df_pred = df.groupby("date").apply(lambda x: x.sort_values("predicted_rank", ascending=True)["true_return"].iloc[:k].mean())
    plt.figure(figsize=(8, 4))
    plt.plot(df_true.index, df_true.values, label="True Top-k Mean Return")
    plt.plot(df_pred.index, df_pred.values, label="Predicted Top-k Mean Return")
    plt.grid()
    plt.title(f"Mean True Returns of Top-{k} Ranked Assets")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    plt.show()



def correlation_matrix_scores_returns(group_table): 
    subset = group_table[["true_return", "predicted_score", "true_rank", "predicted_rank"]]
    corr = subset.corr()
    plt.figure(figsize=(8, 4))
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()



def plot_rank_histograms(group_table):  
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(group_table["true_rank"], bins=20, alpha=0.7)
    plt.title("Distribution of True Ranks")
    plt.subplot(1, 2, 2)
    plt.hist(group_table["predicted_rank"], bins=20, alpha=0.7)
    plt.title("Distribution of Predicted Ranks")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def permutation_importance_listnet(
    model, optimizer, dataloader, feature_cols, device, metric="ndcg", k=5, n_repeats=5
):
    metrics, _ = evaluate_model(model, optimizer, dataloader, device, k_ndcg=k)
    baseline = metrics["ndcg_mean"]

    importances = {f: [] for f in feature_cols}

    model.eval()
    for f_idx, feat in enumerate(feature_cols):
        for _ in range(n_repeats):
            scores_drop = []

            for X, y, tickers, date in dataloader:
                X = X.squeeze(0).to(device)
                y = y.squeeze(0).to(device)

                # Permutation intra-groupe
                perm = torch.randperm(X.size(0))
                X_perm = X.clone()
                X_perm[:, f_idx] = X[perm, f_idx]

                with torch.no_grad():
                    scores = model(X_perm)

                ndcg_val = ndcg_at_k(y.cpu().numpy(), scores.cpu().numpy(), k=k)
                scores_drop.append(ndcg_val)

            importances[feat].append(baseline - np.mean(scores_drop))

    # Aggregate
    imp_mean = {k: np.mean(v) for k, v in importances.items()}
    imp_std = {k: np.std(v) for k, v in importances.items()}

    return pd.DataFrame({
        "feature": imp_mean.keys(),
        "importance_mean": imp_mean.values(),
        "importance_std": imp_std.values()
    }).sort_values("importance_mean", ascending=False)


def plot_feature_importance(df):
    plt.figure(figsize=(8, 4))
    plt.barh(df["feature"], df["importance_mean"], xerr=df["importance_std"])
    plt.xlabel("Δ NDCG@K (Permutation Importance)")
    plt.title("ListNet Feature Importance")
    plt.gca().invert_yaxis()
    plt.grid(axis="x")
    plt.show()