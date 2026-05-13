import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler





# ---------------------------
# Data loading
# ---------------------------
def load_and_split(data_path, FEATURE_COLS, TARGET_COL):
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Keep only required cols
    df = df[['date', 'ticker'] + FEATURE_COLS + [TARGET_COL]]
    # Drop rows without target (safety)
    df = df.dropna(subset=[TARGET_COL])
    # Chronological partitioning as described:
    train_period = (df['date'] >= '2010-01-01') & (df['date'] <= '2018-12-31')  # train
    val_period = (df['date'] >= '2019-01-01') & (df['date'] <= '2019-12-31')    # validation
    test_period = (df['date'] >= '2020-01-01') & (df['date'] <= '2024-12-31')   # test

    train_df = df.loc[train_period].copy()
    val_df = df.loc[val_period].copy()
    test_df = df.loc[test_period].copy()

    # sanity
    if train_df.empty:
        raise ValueError("Train dataframe empty — check date ranges and data file")
    if test_df.empty:
        print("Warning: test dataframe empty — check date ranges and data file")

    return train_df, val_df, test_df

# ---------------------------
# Normalization utilities
# ---------------------------
class GroupScaler:
    """
    Fit scaler on train groups concatenated (standard scaler on features),
    then transform each group's matrix.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, list_of_group_dfs):
        # list_of_group_dfs: list of DataFrames or numpy arrays (n_items x n_features)
        concatenated = np.vstack([g for g in list_of_group_dfs if len(g) > 0])
        self.scaler.fit(concatenated)
        self.fitted = True

    def transform(self, X):
        # X: numpy array (n_items x n_features)
        if not self.fitted:
            raise RuntimeError("Scaler not fitted")
        return self.scaler.transform(X)