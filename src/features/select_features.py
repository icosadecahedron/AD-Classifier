"""
Differential expression-based feature selection.

Uses Welch's t-test per gene (AD vs control) + Benjamini-Hochberg FDR
correction as a lightweight, dependency-free stand-in for R's `limma`.
For a real limma run, export this matrix and labels to R and use
`limma::lmFit` + `eBayes` — the statistical logic here (t-test + BH-FDR)
approximates the same "significant, effect-size-filtered gene list" output
that limma would produce, so downstream code is unaffected either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover - fallback for envs without statsmodels
    _HAS_STATSMODELS = False


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Manual Benjamini-Hochberg FDR correction (numpy-only fallback).

    Used automatically when statsmodels isn't installed; produces the same
    result as statsmodels.stats.multitest.multipletests(method='fdr_bh').
    """
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    fdr = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity (cumulative minimum from the largest p-value down)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.clip(fdr, 0, 1)
    out = np.empty(n)
    out[order] = fdr
    return out


def differential_expression(
    expr_df: pd.DataFrame, labels: pd.Series, config: dict
) -> pd.DataFrame:
    """Run per-gene Welch's t-test between AD and control groups.

    Returns a dataframe of gene, log2fc, p_value, fdr, sorted by fdr.
    """
    gene_cols = [c for c in expr_df.columns if c != "sample_id"]
    ad_mask = labels.values == 1
    ctrl_mask = labels.values == 0

    results = []
    for gene in gene_cols:
        vals = expr_df[gene].values
        ad_vals = vals[ad_mask]
        ctrl_vals = vals[ctrl_mask]

        t_stat, p_val = stats.ttest_ind(ad_vals, ctrl_vals, equal_var=False)
        log2fc = ad_vals.mean() - ctrl_vals.mean()  # already log-scale from preprocessing

        results.append({"gene": gene, "log2fc": log2fc, "p_value": p_val, "t_stat": t_stat})

    de_df = pd.DataFrame(results)
    de_df["p_value"] = de_df["p_value"].fillna(1.0)
    if _HAS_STATSMODELS:
        _, fdr_vals, _, _ = multipletests(de_df["p_value"], method="fdr_bh")
    else:
        fdr_vals = _benjamini_hochberg(de_df["p_value"].values)
    de_df["fdr"] = fdr_vals
    de_df = de_df.sort_values("fdr").reset_index(drop=True)
    return de_df


def select_significant_genes(de_df: pd.DataFrame, config: dict) -> list[str]:
    """Filter DE results by FDR + effect size threshold, cap at max_features."""
    fs_cfg = config["feature_selection"]
    filtered = de_df[
        (de_df["fdr"] < fs_cfg["fdr_threshold"])
        & (de_df["log2fc"].abs() >= fs_cfg["min_abs_log2fc"])
    ]
    filtered = filtered.sort_values("fdr").head(fs_cfg["max_features"])
    return filtered["gene"].tolist()
