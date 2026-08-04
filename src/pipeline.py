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

    logger.info("Step 1/4: Loading dataset...")
    expr_df, labels = load_dataset(config)
    logger.info("Loaded %d samples x %d genes", expr_df.shape[0], expr_df.shape[1] - 1)

    logger.info("Step 2/4: Preprocessing (log transform + variance filter)...")
    expr_processed = preprocess_expression(expr_df, config)

    logger.info("Step 3/4: Differential expression feature selection...")
    de_results = differential_expression(expr_processed, labels, config)
    selected_genes = select_significant_genes(de_results, config)
    logger.info("Selected %d significant genes (FDR < %s)", len(selected_genes),
                config["feature_selection"]["fdr_threshold"])

    output_cfg = config["output"]
    Path(output_cfg["model_dir"]).mkdir(parents=True, exist_ok=True)
    with open(output_cfg["feature_list_path"], "w") as f:
        json.dump(selected_genes, f, indent=2)

    X = expr_processed[selected_genes]

    logger.info("Step 4/4: Training + benchmarking models with MLflow tracking...")
    best_model_name, best_metrics, all_results = run_experiments(X, labels, config)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Best model: %s | CV ROC-AUC: %.4f", best_model_name, best_metrics["cv_auc"])
    for r in all_results:
        logger.info(
            "  %-20s AUC=%.4f  F1=%.4f  Acc=%.4f",
            r["model"], r["cv_mean_roc_auc"], r["cv_mean_f1"], r["cv_mean_accuracy"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
