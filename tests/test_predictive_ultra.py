"""
Tests for predictive_ultra.py
"""
import os
import sys
import tempfile
import numpy as np
import pytest

# Add parent to path so we can import the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import predictive_ultra as pu

TEST_DATA = os.path.join(SCRIPT_DIR, "..", "test_data.csv")
HAS_DATA = os.path.exists(TEST_DATA)


@pytest.fixture
def sample_df():
    if not HAS_DATA:
        pytest.skip("test_data.csv not found")
    return pu.load_data(TEST_DATA, "usertext", "chainid")


def test_imports():
    assert pu is not None


def test_load_data(sample_df):
    assert len(sample_df) > 0
    assert "text" in sample_df.columns
    assert "label" in sample_df.columns


def test_preflight(sample_df):
    passed, issues = pu.run_preflight(sample_df, "chainid")
    assert passed is True
    assert len(issues["blockers"]) == 0


def test_make_model_xgboost():
    model, actual_type, note = pu.make_model("xgboost")
    assert actual_type == "xgboost" or note == "xgboost_fallback"


def test_make_model_logreg():
    model, actual_type, note = pu.make_model("logreg")
    assert actual_type == "logistic_regression"


def test_compute_metrics_binary(sample_df):
    from sklearn.preprocessing import LabelEncoder
    y = sample_df["label"].values[:100]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # Flip ~30% of labels to simulate a realistic prediction
    y_pred_enc = y_enc.copy()
    rng = np.random.RandomState(42)
    flip_idx = rng.choice(len(y_enc), size=30, replace=False)
    y_pred_enc[flip_idx] = 1 - y_pred_enc[flip_idx]
    metrics = pu.compute_metrics(y_enc, y_pred_enc, n_classes=2)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] >= 0.5


def test_tfidf_features(sample_df):
    texts = sample_df["text"].tolist()[:50]
    fe = pu.build_features(texts, texts, "tfidf", max_features=1000)
    assert fe["X_train"] is not None
    assert fe["X_test"] is not None
    assert fe["vectorizer"] is not None


def test_transformer_model_map():
    assert "distilbert" in pu.TRANSFORMER_MODEL_MAP
    assert "roberta" in pu.TRANSFORMER_MODEL_MAP
    assert "mpnet" in pu.TRANSFORMER_MODEL_MAP


@pytest.mark.slow
def test_full_pipeline_xgboost_tfidf(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = pu.train_classifier(
                sample_df.iloc[:50],
                model_type="xgboost",
                feature_mode="tfidf",
                test_size=0.3,
                max_features=500,
                n_bootstrap=0,
                explain_mode="none",
                output_dir=tmpdir,
            )
            assert result["metrics"]["macro_f1"] > 0
        except Exception as e:
            if "xgboost" in str(e).lower():
                pytest.skip("XGBoost not installed")
            raise


def test_shared_infrastructure():
    try:
        from ultra_shared.logging import setup_logging
        from ultra_shared.config import load_config
        from ultra_shared.cache import EmbeddingCache
        assert setup_logging is not None
    except ImportError:
        pytest.skip("ultra_shared not installed")
