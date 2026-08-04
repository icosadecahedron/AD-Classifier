"""
Model training: benchmarks multiple classifiers with stratified CV,
logs every run to MLflow (if installed), and persists the best model +
selected gene list.

xgboost and mlflow are optional at import time so this pipeline still runs
in restricted/offline environments (falls back to GradientBoostingClassifier
and plain logging respectively). Install both from requirements.txt for the
full intended experience -- MLflow experiment tracking and XGBoost as one of
the benchmarked models.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

logger = logging.getLogger(__name__)

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    _HAS_MLFLOW = False
    logger.warning("mlflow not installed - falling back to plain logging for experiment tracking.")

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    _HAS_XGBOOST = False
    logger.warning("xgboost not installed - substituting GradientBoostingClassifier for 'xgboost' config entry.")


@contextmanager
def _tracking_run(run_name: str):
    """MLflow run context if available, else a no-op context."""
    if _HAS_MLFLOW:
        with mlflow.start_run(run_name=run_name):
            yield
    else:
        yield


def _log_params(params: dict):
    if _HAS_MLFLOW:
        mlflow.log_params(params)
    else:
        logger.info("  params: %s", params)


def _log_metric(name: str, value: float):
    if _HAS_MLFLOW:
        mlflow.log_metric(name, value)


def _build_model(name: str, params: dict):
    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "random_forest":
        return RandomForestClassifier(**params)
    if name == "xgboost":
        if _HAS_XGBOOST:
            return XGBClassifier(**params, eval_metric="logloss")
        # Fallback: closest sklearn equivalent, same hyperparameter names where possible
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.05),
        )
    raise ValueError(f"Unknown model: {name}")


def run_experiments(
    X: pd.DataFrame, y: pd.Series, config: dict
) -> tuple[str, dict, list[dict]]:
    """Train + cross-validate every model in config.models, logging each run.

    Returns (best_model_name, best_model_metrics, all_results).
    """
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    if _HAS_MLFLOW:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_cfg["test_size"], stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=train_cfg["cv_folds"], shuffle=True, random_state=42)
    all_results = []
    best_score = -np.inf
    best_model_name = None
    best_model = None

    for model_cfg in config["models"]:
        name = model_cfg["name"]
        params = model_cfg["params"]

        with _tracking_run(name):
            _log_params(params)
            model = _build_model(name, params)

            cv_scores = cross_validate(
                model, X_train, y_train, cv=cv, scoring=train_cfg["scoring"]
            )
            mean_auc = cv_scores["test_roc_auc"].mean()
            mean_f1 = cv_scores["test_f1"].mean()
            mean_acc = cv_scores["test_accuracy"].mean()

            _log_metric("cv_mean_roc_auc", mean_auc)
            _log_metric("cv_mean_f1", mean_f1)
            _log_metric("cv_mean_accuracy", mean_acc)

            model.fit(X_train, y_train)
            held_out_acc = model.score(X_test, y_test)
            _log_metric("held_out_accuracy", held_out_acc)

            result = {
                "model": name,
                "cv_mean_roc_auc": mean_auc,
                "cv_mean_f1": mean_f1,
                "cv_mean_accuracy": mean_acc,
                "held_out_accuracy": held_out_acc,
            }
            all_results.append(result)
            logger.info(
                "Model %-20s CV AUC=%.4f  F1=%.4f  Acc=%.4f  HeldOutAcc=%.4f",
                name, mean_auc, mean_f1, mean_acc, held_out_acc,
            )

            if mean_auc > best_score:
                best_score = mean_auc
                best_model_name = name
                best_model = model

    output_cfg = config["output"]
    Path(output_cfg["model_dir"]).mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_cfg["best_model_path"])
    joblib.dump(list(X.columns), output_cfg["model_dir"] + "/feature_columns.joblib")

    logger.info(
        "Best model: %s (CV AUC=%.4f), saved to %s",
        best_model_name, best_score, output_cfg["best_model_path"],
    )
    return best_model_name, {"cv_auc": best_score}, all_results
