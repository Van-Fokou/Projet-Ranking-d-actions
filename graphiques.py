import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. Tableaux principaux (DataFrames)
# ---------------------------------------------------------------------------

def summary_table(results):
    """
    results = dict contenant:
        - ndcg_mean
        - ndcg_std
        - topk_mean
        - topk_std
        - spearman_mean
        - spearman_std
        - mae
        - rmse
        - accuracy
    Retourne un DataFrame propre
    """
    df = pd.DataFrame({
        "Metric": ["NDCG@K", "Top-K Accuracy", "Spearman ρ", "MAE", "RMSE", "Accuracy"],
        "Mean": [
            results["ndcg_mean"],
            results["topk_mean"],
            results["spearman_mean"],
            results["mae"],
            results["rmse"],
            results["accuracy"]
        ],
        "Std": [
            results["ndcg_std"],
            results["topk_std"],
            results["spearman_std"],
            None,
            None,
            None
        ]
    })

    return df


def create_group_level_table(all_scores, all_true, all_dates, all_tickers):
    """
    Create a detailed per-group table:
        date, ticker, true_return, predicted_score, rank_true, rank_pred
    """
    all_rows = []

    for scores, y_true, date, tickers in zip(all_scores, all_true, all_dates, all_tickers):

        # Ranks
        rank_true = np.argsort(np.argsort(-y_true))
        rank_pred = np.argsort(np.argsort(-scores))

        for t, yt, yp, r_t, r_p in zip(tickers, y_true, scores, rank_true, rank_pred):
            all_rows.append({
                "date": date,
                "ticker": t,
                "true_return": yt,
                "predicted_score": yp,
                "true_rank": r_t,
                "predicted_rank": r_p
            })

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# 2. Graphiques individuels
# ---------------------------------------------------------------------------

def plot_loss_curves(train_losses, val_losses):
    """
    Courbe d'entraînement et validation
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_ndcg_by_epoch(ndcg_train, ndcg_val):
    """
    NDCG par epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ndcg_train, label="Train NDCG")
    plt.plot(ndcg_val, label="Val NDCG")
    plt.xlabel("Epoch")
    plt.ylabel("NDCG@K")
    plt.title("NDCG@K Evolution per Epoch")
    plt.grid()
    plt.legend()
    plt.show()


def plot_distribution_true_pred(y_true, y_pred):
    """
    Distribution comparée des rendements et scores prédits
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_true, label="True Returns")
    sns.kdeplot(y_pred, label="Predicted Scores")
    plt.title("Distribution: True vs Predicted")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()


def plot_true_vs_pred_scatter(y_true, y_pred):
    """
    Scatter true vs predicted
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Returns")
    plt.ylabel("Predicted Scores")
    plt.title("Scatter: True vs Predicted")
    plt.grid()
    plt.show()


def plot_rank_correlation(y_true, y_pred):
    """
    Scatter des ranks
    """
    rank_true = np.argsort(np.argsort(-y_true))
    rank_pred = np.argsort(np.argsort(-y_pred))

    plt.figure(figsize=(8, 8))
    plt.scatter(rank_true, rank_pred, alpha=0.6)
    plt.xlabel("True Ranks")
    plt.ylabel("Predicted Ranks")
    plt.title("Rank Correlation Scatter")
    plt.grid()
    plt.show()


def plot_topk_accuracy_over_time(topk_values, dates):
    """
    Top-k accuracy au fil du temps
    """
    plt.figure(figsize=(12, 5))
    plt.plot(dates, topk_values)
    plt.title("Top-K Accuracy Over Time")
    plt.xlabel("Date")
    plt.ylabel("Top-K Accuracy")
    plt.grid()
    plt.show()


def plot_spearman_over_time(spearman_values, dates):
    """
    Spearman par date
    """
    plt.figure(figsize=(12, 5))
    plt.plot(dates, spearman_values)
    plt.xlabel("Date")
    plt.ylabel("Spearman ρ")
    plt.title("Spearman correlation over time")
    plt.grid()
    plt.show()


def plot_returns_of_top_models(group_table, k=5):
    """
    Rendements moyens des k meilleurs actifs prédits vs vrais
    """
    df = group_table.copy()

    df_true = df.groupby("date").apply(lambda x: 
                                       x.sort_values("true_rank", ascending=False)["true_return"].iloc[:k].mean())

    df_pred = df.groupby("date").apply(lambda x: 
                                       x.sort_values("predicted_rank", ascending=False)["true_return"].iloc[:k].mean())

    plt.figure(figsize=(12, 5))
    plt.plot(df_true.index, df_true.values, label="True Top-k Mean Return")
    plt.plot(df_pred.index, df_pred.values, label="Predicted Top-k Mean Return")
    plt.grid()
    plt.title(f"Mean True Returns of Top-{k} Ranked Assets")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Heatmaps & correlation
# ---------------------------------------------------------------------------

def plot_rank_difference_heatmap(group_table, date):
    """
    Heatmap differences de rangs pour une date
    """
    df = group_table[group_table["date"] == date]
    diff = df["predicted_rank"].values - df["true_rank"].values

    plt.figure(figsize=(10, 2))
    sns.heatmap([diff], cmap="coolwarm", center=0, annot=False, cbar=True)
    plt.title(f"Rank Prediction Error - {date}")
    plt.yticks([])
    plt.xlabel("Assets")
    plt.show()


def correlation_matrix_scores_returns(group_table):
    """
    Matrice de corrélation:
        true_return, predicted_score, true_rank, predicted_rank
    """
    subset = group_table[["true_return", "predicted_score", "true_rank", "predicted_rank"]]
    corr = subset.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    return corr


# ---------------------------------------------------------------------------
# 4. Ordre global – Distribution des ranks
# ---------------------------------------------------------------------------

def plot_rank_histograms(group_table):
    """
    Histogrammes
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(group_table["true_rank"], bins=20, alpha=0.7)
    plt.title("Distribution of True Ranks")

    plt.subplot(1, 2, 2)
    plt.hist(group_table["predicted_rank"], bins=20, alpha=0.7)
    plt.title("Distribution of Predicted Ranks")

    plt.show()
