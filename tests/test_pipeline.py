
import numpy as np
import pandas as pd
import pytest

from src.data.load_data import _generate_synthetic_expression_data, load_dataset
from src.data.preprocess import preprocess_expression, standardize_features
from src.features.select_features import differential_expression, select_significant_genes

TEST_CONFIG = {
    "data": {"source": "synthetic", "n_samples": 100, "n_genes": 200, "random_seed": 1},
    "preprocessing": {"log_transform": True, "standardize": True, "variance_filter_top_n": 150},
    "feature_selection": {"fdr_threshold": 0.05, "min_abs_log2fc": 0.1, "max_features": 50},
}


class TestDataLoading:
    def test_synthetic_shape(self):
        expr_df, labels = _generate_synthetic_expression_data(n_samples=100, n_genes=200, seed=1)
        assert expr_df.shape == (100, 201)  # + sample_id column
        assert len(labels) == 100

    def test_labels_are_binary(self):
        _, labels = _generate_synthetic_expression_data(n_samples=100, n_genes=200, seed=1)
        assert set(labels.unique()).issubset({0, 1})

    def test_no_missing_values(self):
        expr_df, _ = _generate_synthetic_expression_data(n_samples=50, n_genes=50, seed=1)
        assert expr_df.drop(columns=["sample_id"]).isnull().sum().sum() == 0

    def test_load_dataset_dispatches_to_synthetic(self):
        expr_df, labels = load_dataset(TEST_CONFIG)
        assert expr_df.shape[0] == 100
        assert len(labels) == 100

    def test_load_dataset_raises_on_unknown_source(self):
        bad_config = {"data": {**TEST_CONFIG["data"], "source": "not_a_real_source"}}
        with pytest.raises(ValueError):
            load_dataset(bad_config)


class TestPreprocessing:
    def test_preprocess_preserves_sample_id(self):
        expr_df, _ = _generate_synthetic_expression_data(n_samples=50, n_genes=100, seed=1)
        processed = preprocess_expression(expr_df, TEST_CONFIG)
        assert "sample_id" in processed.columns

    def test_variance_filter_reduces_columns(self):
        expr_df, _ = _generate_synthetic_expression_data(n_samples=50, n_genes=200, seed=1)
        processed = preprocess_expression(expr_df, TEST_CONFIG)
        # +1 for sample_id column
        assert processed.shape[1] == TEST_CONFIG["preprocessing"]["variance_filter_top_n"] + 1

    def test_standardize_features_zero_mean(self):
        X_train = pd.DataFrame(np.random.randn(80, 10))
        X_test = pd.DataFrame(np.random.randn(20, 10))
        X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-6)


class TestFeatureSelection:
    def test_de_returns_all_genes(self):
        expr_df, labels = _generate_synthetic_expression_data(n_samples=100, n_genes=100, seed=1)
        processed = preprocess_expression(expr_df, TEST_CONFIG)
        de_df = differential_expression(processed, labels, TEST_CONFIG)
        n_genes = processed.shape[1] - 1
        assert len(de_df) == n_genes

    def test_de_fdr_between_0_and_1(self):
        expr_df, labels = _generate_synthetic_expression_data(n_samples=100, n_genes=100, seed=1)
        processed = preprocess_expression(expr_df, TEST_CONFIG)
        de_df = differential_expression(processed, labels, TEST_CONFIG)
        assert (de_df["fdr"] >= 0).all() and (de_df["fdr"] <= 1).all()

    def test_select_significant_genes_respects_max_features(self):
        expr_df, labels = _generate_synthetic_expression_data(n_samples=200, n_genes=500, seed=1)
        processed = preprocess_expression(expr_df, TEST_CONFIG)
        de_df = differential_expression(processed, labels, TEST_CONFIG)
        selected = select_significant_genes(de_df, TEST_CONFIG)
        assert len(selected) <= TEST_CONFIG["feature_selection"]["max_features"]

    def test_select_significant_genes_finds_injected_signal(self):
        expr_df, labels = _generate_synthetic_expression_data(n_samples=300, n_genes=1000, seed=1)
        processed = preprocess_expression(expr_df, TEST_CONFIG)
        de_df = differential_expression(processed, labels, TEST_CONFIG)
        selected = select_significant_genes(de_df, TEST_CONFIG)
        known_ad_genes = {"APOE", "BIN1", "CLU", "PICALM", "TREM2", "MAPT", "ABCA7", "SORL1"}
        assert len(set(selected) & known_ad_genes) > 0
