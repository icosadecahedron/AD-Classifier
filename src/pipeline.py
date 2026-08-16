"""
Main pipeline entry point: data -> preprocessing -> feature selection ->
model training/tracking -> persisted artifacts for the serving API.

Run with:  python -m src.pipeline
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from sklearn.model_selection import train_test_split
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_expression
from src.features.select_features import differential_expression, select_significant_genes
from src.models.train import run_experiments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    logger.info("Step 1/5: Loading dataset...")
    expr_df, labels = load_dataset(config)
    logger.info("Loaded %d samples x %d genes", expr_df.shape[0], expr_df.shape[1])

    logger.info("Step 2/5: Splitting train/test data...")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        expr_df,
        labels,
        test_size=0.20,
        stratify=labels,
        random_state=42,
    )

    logger.info(
        "Train samples: %d | Test samples: %d",
        len(y_train),
        len(y_test),
    )

    logger.info("Step 3/5: Preprocessing...")

    X_train_processed = preprocess_expression(X_train_raw, config)
    X_test_processed = preprocess_expression(X_test_raw, config)

    logger.info("Step 4/5: Differential expression feature selection on TRAIN data only...")

    de_results = differential_expression(
        X_train_processed,
        y_train,
        config,
    )

    selected_genes = select_significant_genes(
        de_results,
        config,
    )

    logger.info(
        "Selected %d significant genes (FDR < %s)",
        len(selected_genes),
        config["feature_selection"]["fdr_threshold"],
    )

    output_cfg = config["output"]

    Path(output_cfg["model_dir"]).mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_cfg["feature_list_path"], "w") as f:
        json.dump(selected_genes, f, indent=2)

    X_train = X_train_processed[selected_genes]
    X_test = X_test_processed[selected_genes]

    logger.info("Step 5/5: Training + benchmarking models...")

    best_model_name, best_metrics, all_results = run_experiments(
        X_train,
        y_train,
        config,
    )