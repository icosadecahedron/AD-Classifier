"""Preprocessing: log transform, variance filtering, standardization."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_expression(expr_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply log transform + variance filtering to a raw expression matrix.

    Expects expr_df to have a 'sample_id' column plus one column per gene.
    Returns a new dataframe with the same 'sample_id' column preserved.
    """
    pre_cfg = config["preprocessing"]
    sample_ids = expr_df["sample_id"]
    gene_data = expr_df.drop(columns=["sample_id"]).astype(float)

    if pre_cfg.get("log_transform", True):
        # Guard against negative values from synthetic data before log2
        gene_data = np.log2(gene_data - gene_data.min().min() + 1)

    top_n = pre_cfg.get("variance_filter_top_n")
    if top_n and top_n < gene_data.shape[1]:
        top_genes = gene_data.var(axis=0).sort_values(ascending=False).index[:top_n]
        gene_data = gene_data[top_genes]

    result = gene_data.copy()
    result.insert(0, "sample_id", sample_ids.values)
    return result


def standardize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on train only, apply to both train and test."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler
