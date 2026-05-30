#!/usr/bin/env python3
"""
Predictive Modeling ULTRA v1.0
==============================
Standalone Python script for text classification with XGBoost, logistic regression,
SVM, random forest, and other models. Full evaluation pipeline with SHAP
explainability, cross-validation, calibration, and error analysis.

Architecture:
  preflight → features → model training → evaluation → explainability → report

Evidence base:
  predictive.py (corpus_topic_lab v0.58) — gated pipeline architecture
  SHAP (Lundberg & Lee 2017) — Tree SHAP for gradient boosted models
  imbalanced-learn (Lemaître et al. 2017) — SMOTE for class imbalance
  sklearn (Pedregosa et al. 2011) — Vectorizers, models, metrics, CV
  Optuna (Akiba et al. 2019) — Hyperparameter optimization

Usage:
  python predictive_ultra.py data.csv text_col label_col                        # train + evaluate
  python predictive_ultra.py data.csv text_col label_col --model xgboost       # XGBoost (default)
  python predictive_ultra.py data.csv text_col label_col --model logreg        # logistic regression
  python predictive_ultra.py data.csv text_col label_col --model svm           # SVM
  python predictive_ultra.py data.csv text_col label_col --model rf            # random forest
  python predictive_ultra.py data.csv text_col label_col --model nb            # Complement Naive Bayes
  python predictive_ultra.py data.csv text_col label_col --model ensemble      # soft voting ensemble
  python predictive_ultra.py data.csv text_col label_col --split temporal      # temporal split
  python predictive_ultra.py data.csv text_col label_col --group user          # group k-fold
  python predictive_ultra.py data.csv text_col label_col --features sbert      # SBERT embeddings
  python predictive_ultra.py data.csv text_col label_col --tune                # Optuna tuning
  python predictive_ultra.py data.csv text_col label_col --resample smote      # SMOTE oversampling
  python predictive_ultra.py data.csv text_col label_col --calibrate           # probability calibration
  python predictive_ultra.py data.csv text_col label_col --explain shap        # SHAP explanations
  python predictive_ultra.py data.csv text_col label_col --cleanlab            # label quality audit
  python predictive_ultra.py data.csv text_col label_col --all                 # everything
  python predictive_ultra.py data.csv text_col label_col --date date_col       # temporal split
  python predictive_ultra.py data.csv text_col label_col --test data2.csv      # predict on new data

Author: Peter Pang (2026)
License: MIT
"""

import argparse
import base64
import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Optional imports ─────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, \
        GroupKFold, train_test_split, cross_val_score, cross_val_predict
    from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, \
        classification_report, confusion_matrix, roc_auc_score, roc_curve, \
        precision_recall_curve, cohen_kappa_score, matthews_corrcoef, \
        brier_score_loss, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
    from sklearn.base import BaseEstimator, TransformerMixin, clone as skclone
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
    from sklearn.naive_bayes import ComplementNB
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import cleanlab
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import torch
    import transformers
    from transformers import (
        AutoTokenizer, AutoModel,
        DistilBertTokenizer, DistilBertForSequenceClassification,
        DistilBertModel, Trainer, TrainingArguments
    )
    from torch.utils.data import Dataset
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING & PREFLIGHT
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path, text_col, label_col, group_col=None, date_col=None,
              encoding="utf-8"):
    """Load CSV/TSV data for classification."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding=encoding)
    elif p.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t", encoding=encoding)
    elif p.suffix.lower() == ".json":
        df = pd.read_json(path, encoding=encoding)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    for col in [text_col, label_col]:
        if col and col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    df["text"] = df[text_col].astype(str)

    if label_col:
        df = df.dropna(subset=[label_col]).reset_index(drop=True)
        df["label"] = df[label_col].astype(str)

    if group_col and group_col in df.columns:
        df["group"] = df[group_col].astype(str)
    else:
        df["group"] = df.index.astype(str)

    if date_col and date_col in df.columns:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    return df


def run_preflight(df, label_col):
    """Preflight validation before training."""
    print(f"\n{'='*60}")
    print("PREFLIGHT GATE")
    print(f"{'='*60}")

    issues = {"blockers": [], "warnings": []}

    labels = df["label"]
    classes = labels.unique()
    n_classes = len(classes)
    class_counts = labels.value_counts()

    print(f"  Classes: {n_classes}")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count} ({count/len(df)*100:.1f}%)")

    if n_classes < 2:
        issues["blockers"].append("Need at least 2 classes")
    if class_counts.min() < 2:
        issues["blockers"].append("Each class needs at least 2 samples")

    min_pct = class_counts.min() / len(df) * 100
    if min_pct < 5:
        issues["warnings"].append(
            f"Severe class imbalance: minority class {min_pct:.1f}%")
    elif min_pct < 10:
        issues["warnings"].append(
            f"Class imbalance: minority class {min_pct:.1f}%")

    print(f"\n  Blockers: {len(issues['blockers'])}")
    for b in issues["blockers"]:
        print(f"    ✗ {b}")
    print(f"  Warnings: {len(issues['warnings'])}")
    for w in issues["warnings"]:
        print(f"    ⚠ {w}")

    if issues["blockers"]:
        print("\n  ✗ PREFLIGHT BLOCKED — cannot proceed")
        return False, issues

    print("\n  ✓ PREFLIGHT PASSED")
    return True, issues


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

TRANSFORMER_MODEL_MAP = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "deberta": "microsoft/deberta-v3-base",
}


def build_features(train_texts, test_texts, feature_mode="tfidf",
                   max_features=12000, ngram_range=(1, 2),
                   embedding_model="all-MiniLM-L6-v2",
                   train_meta=None, test_meta=None):
    """Build feature matrix from text data.

    Modes: tfidf, sbert, tfidf+sbert, distilbert, roberta, mpnet, deberta,
           tfidf+{model} (e.g. tfidf+roberta)
    """
    t0 = time.time()
    vectorizer = None
    embedding_model_obj = None
    scaler = None

    feature_parts = []
    feature_names = []

    if "tfidf" in feature_mode or feature_mode == "tfidf":
        print(f"  Building TF-IDF features ({max_features} max, {ngram_range} n-grams)...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english",
            min_df=2,
            max_df=0.85,
        )
        X_tfidf_train = vectorizer.fit_transform(train_texts)
        X_tfidf_test = vectorizer.transform(test_texts)
        feature_parts.append(X_tfidf_train)
        feature_parts.append(X_tfidf_test)
        feature_names.extend([f"tfidf__{n}" for n in vectorizer.get_feature_names_out()])
        print(f"    TF-IDF dims: {X_tfidf_train.shape}")

    # Generalized transformer embedding extraction (distilbert, roberta, mpnet, deberta)
    active_transformer = next((t for t in TRANSFORMER_MODEL_MAP if t in feature_mode), None)
    if active_transformer and HAS_TORCH_TRANSFORMERS:
        model_name = TRANSFORMER_MODEL_MAP[active_transformer]
        print(f"  Computing {active_transformer} embeddings ({model_name})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()

        def _encode(texts, batch_size=32):
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embs.append(cls_emb)
            return np.vstack(all_embs)

        train_emb2 = _encode(train_texts)
        test_emb2 = _encode(test_texts)
        scaler2 = StandardScaler()
        train_emb2 = scaler2.fit_transform(train_emb2)
        test_emb2 = scaler2.transform(test_emb2)

        from scipy.sparse import csr_matrix
        feature_parts.append(csr_matrix(train_emb2))
        feature_parts.append(csr_matrix(test_emb2))
        feature_names.extend([f"{active_transformer}_dim_{i}" for i in range(train_emb2.shape[1])])
        print(f"    {active_transformer} dims: {train_emb2.shape}")
    elif active_transformer and not HAS_TORCH_TRANSFORMERS:
        print(f"    [!] transformers/torch not installed. Skipping {active_transformer} features.")

    if "sbert" in feature_mode:
        print(f"  Computing SBERT embeddings ({embedding_model})...")
        if HAS_SBERT:
            embedding_model_obj = SentenceTransformer(embedding_model)
            train_emb = embedding_model_obj.encode(train_texts, show_progress_bar=False)
            test_emb = embedding_model_obj.encode(test_texts, show_progress_bar=False)
            scaler = StandardScaler()
            train_emb = scaler.fit_transform(train_emb)
            test_emb = scaler.transform(test_emb)

            from scipy.sparse import csr_matrix
            feature_parts.append(csr_matrix(train_emb))
            feature_parts.append(csr_matrix(test_emb))
            feature_names.extend([f"sbert_dim_{i}" for i in range(train_emb.shape[1])])
            print(f"    SBERT dims: {train_emb.shape}")
        else:
            print("    [!] sentence-transformers not installed. Skipping SBERT features.")

    # Combine features
    if len(feature_parts) >= 2:
        from scipy.sparse import hstack
        X_train = hstack(feature_parts[0::2])
        X_test = hstack(feature_parts[1::2])
    elif len(feature_parts) == 2:
        from scipy.sparse import hstack
        X_train = feature_parts[0]
        X_test = feature_parts[1]
    else:
        X_train = feature_parts[0]
        X_test = feature_parts[1]

    elapsed = time.time() - t0
    print(f"  Feature engineering done in {elapsed:.1f}s. "
          f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    fe = {
        "X_train": X_train,
        "X_test": X_test,
        "feature_names": feature_names,
        "vectorizer": vectorizer,
        "embedding_model": embedding_model_obj,
        "scaler": scaler,
    }
    return fe


def select_top_features(X_train, y_train, feature_names, k=5000, method="chi2"):
    """Feature selection via chi2 or mutual information."""
    if k >= X_train.shape[1]:
        return X_train, feature_names, None

    print(f"  Selecting top {k} features via {method}...")
    if method == "chi2":
        selector = SelectKBest(chi2, k=min(k, X_train.shape[1]))
    else:
        selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))

    X_selected = selector.fit_transform(X_train, y_train)
    selected_mask = selector.get_support()
    selected_names = [n for n, s in zip(feature_names, selected_mask) if s]
    print(f"    {X_train.shape[1]} → {X_selected.shape[1]} features")
    return X_selected, selected_names, selector


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def make_model(model_type="xgboost_preferred", random_state=42,
               class_weight_mode=None, n_classes=2,
               n_estimators=100, max_iter=1000,
               scale_pos_weight=None, use_gpu=False):
    """Create a classifier based on model type."""
    sk = random_state

    if model_type in ("xgboost_preferred", "xgb", "xgboost"):
        if HAS_XGB:
            params = dict(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.6,
                min_child_weight=3,
                gamma=0.1,
                reg_lambda=2,
                reg_alpha=0.1,
                random_state=sk,
                eval_metric="logloss",
                tree_method="hist",
                verbosity=0,
            )
            if n_classes == 2 and scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight
            if use_gpu:
                params["device"] = "cuda"
            model = xgb.XGBClassifier(**params)
            return model, "xgboost", None
        else:
            model = LogisticRegression(
                max_iter=max_iter, random_state=sk,
                class_weight=class_weight_mode,
                solver="lbfgs", n_jobs=-1,
            )
            return model, "logistic_regression", "xgboost_fallback"

    elif model_type in ("logreg", "logistic_regression"):
        model = LogisticRegression(
            max_iter=max_iter, random_state=sk,
            class_weight=class_weight_mode,
            solver="lbfgs", n_jobs=-1,
        )
        return model, "logistic_regression", None

    elif model_type in ("svm", "svc", "linear_svc"):
        model = SVC(kernel="linear", probability=True,
                    random_state=sk, class_weight=class_weight_mode)
        return model, "svm", None

    elif model_type in ("rf", "random_forest"):
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=sk,
            class_weight=class_weight_mode, n_jobs=-1,
        )
        return model, "random_forest", None

    elif model_type in ("nb", "cnb", "complement_nb"):
        model = ComplementNB()
        return model, "complement_nb", None

    elif model_type in ("sgd", "sgd_log"):
        model = SGDClassifier(loss="log_loss", max_iter=max_iter,
                              random_state=sk, class_weight=class_weight_mode)
        return model, "sgd_log_loss", None

    elif model_type in ("ensemble", "voting", "soft_voting"):
        models = []
        if HAS_XGB:
            models.append(("xgb", xgb.XGBClassifier(
                n_estimators=50, random_state=sk, verbosity=0)))
        models.append(("lr", LogisticRegression(max_iter=1000, random_state=sk)))
        models.append(("rf", RandomForestClassifier(
            n_estimators=50, random_state=sk, n_jobs=-1)))
        model = VotingClassifier(estimators=models, voting="soft")
        return model, "soft_voting_ensemble", None

    # Default
    model = LogisticRegression(max_iter=1000, random_state=sk,
                                class_weight=class_weight_mode)
    return model, "logistic_regression", "default_fallback"


def make_resampling_pipeline(model, method="smote", random_state=42):
    """Wrap model in resampling pipeline."""
    if not HAS_IMBLEARN:
        return model, "no_resampling", "imblearn_not_available"

    if method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "adasyn":
        sampler = ADASYN(random_state=random_state)
    elif method == "oversample":
        sampler = RandomOverSampler(random_state=random_state)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        return model, "no_resampling", None

    pipeline = ImbPipeline([
        ("sampler", sampler),
        ("classifier", model),
    ])
    return pipeline, f"resampling_{method}", None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob=None, n_classes=2):
    """Compute all classification metrics."""
    metrics = {}

    metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    metrics["per_class"] = {}
    for i in range(n_classes):
        metrics["per_class"][str(i)] = {
            "precision": round(p[i], 4), "recall": round(r[i], 4),
            "f1": round(f1[i], 4), "support": int(s[i]),
        }

    metrics["macro_f1"] = round(f1_score(y_true, y_pred, average="macro"), 4)
    metrics["weighted_f1"] = round(f1_score(y_true, y_pred, average="weighted"), 4)
    metrics["micro_f1"] = round(f1_score(y_true, y_pred, average="micro"), 4)
    metrics["kappa"] = round(cohen_kappa_score(y_true, y_pred), 4)
    metrics["mcc"] = round(matthews_corrcoef(y_true, y_pred), 4)

    if y_prob is not None:
        if n_classes == 2:
            try:
                metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob[:, 1]), 4)
            except Exception:
                metrics["roc_auc"] = None
            try:
                metrics["brier"] = round(brier_score_loss(y_true, y_prob[:, 1]), 4)
            except Exception:
                metrics["brier"] = None
        else:
            try:
                metrics["roc_auc"] = round(
                    roc_auc_score(y_true, y_prob, multi_class="ovr"), 4)
            except Exception:
                metrics["roc_auc"] = None
            metrics["brier"] = None

    return metrics


def bootstrap_ci(y_true, y_pred, n_iterations=1000, metric_fn=None):
    """Bootstrap confidence interval for a metric."""
    if metric_fn is None:
        metric_fn = lambda t, p: f1_score(t, p, average="macro")

    n = len(y_true)
    scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_iterations):
        idx = rng.randint(0, n, n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    scores = sorted(scores)
    return {
        "mean": round(np.mean(scores), 4),
        "ci_low": round(scores[int(0.025 * n_iterations)], 4),
        "ci_high": round(scores[int(0.975 * n_iterations)], 4),
    }


def run_cross_validation(X, y, model, n_splits=5, groups=None, parallel=False):
    """Run stratified k-fold CV with optional parallel fold processing."""
    print(f"\n  Running {n_splits}-fold stratified cross-validation...")
    t0 = time.time()

    if groups is not None:
        cv = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if parallel:
        from joblib import Parallel, delayed
        def _fit_fold(fold, train_idx, test_idx):
            X_fold_train, X_fold_test = X[train_idx], X[test_idx]
            y_fold_train, y_fold_test = y[train_idx], y[test_idx]
            model_clone = skclone(model)
            if hasattr(model_clone, "set_params"):
                model_clone.set_params(early_stopping_rounds=0)
            if hasattr(model_clone, "steps"):
                final_step = model_clone.steps[-1]
                if hasattr(final_step[1], "set_params"):
                    final_step[1].set_params(random_state=fold, early_stopping_rounds=0)
            elif hasattr(model_clone, "random_state"):
                model_clone.set_params(random_state=fold)
            model_clone.fit(X_fold_train, y_fold_train)
            y_fold_pred = model_clone.predict(X_fold_test)
            try:
                y_fold_prob = model_clone.predict_proba(X_fold_test)
            except Exception:
                y_fold_prob = None
            fm = compute_metrics(y_fold_test, y_fold_pred, y_fold_prob, len(set(y)))
            fm["fold"] = fold
            return fm
        fold_metrics = Parallel(n_jobs=-1)(
            delayed(_fit_fold)(fold, ti, ti_val)
            for fold, (ti, ti_val) in enumerate(cv.split(X, y, groups), 1)
        )
        # Fix fold numbers
        for i, fm in enumerate(fold_metrics):
            fm["fold"] = i + 1
    else:
        fold_metrics = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
            X_fold_train, X_fold_test = X[train_idx], X[test_idx]
            y_fold_train, y_fold_test = y[train_idx], y[test_idx]
            model_clone = skclone(model)
            if hasattr(model_clone, "steps"):
                final_step = model_clone.steps[-1]
                if hasattr(final_step[1], "set_params"):
                    final_step[1].set_params(random_state=fold)
            elif hasattr(model_clone, "random_state"):
                model_clone.set_params(random_state=fold)
            model_clone.fit(X_fold_train, y_fold_train)
            y_fold_pred = model_clone.predict(X_fold_test)
            try:
                y_fold_prob = model_clone.predict_proba(X_fold_test)
            except Exception:
                y_fold_prob = None
            fm = compute_metrics(y_fold_test, y_fold_pred, y_fold_prob, len(set(y)))
            fm["fold"] = fold
            fold_metrics.append(fm)

    elapsed = time.time() - t0
    df_cv = pd.DataFrame(fold_metrics)
    print(f"  CV done in {elapsed:.1f}s")
    print(f"  {'Metric':<20} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    for col in ["accuracy", "macro_f1", "weighted_f1", "kappa", "mcc", "roc_auc"]:
        if col in df_cv.columns:
            vals = df_cv[col].dropna()
            if len(vals) > 0:
                print(f"  {col:<20} {vals.mean():>8.4f} {vals.std():>8.4f}")

    return df_cv


def run_threshold_optimization(y_true, y_prob):
    """Find optimal probability threshold for binary classification."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob[:, 1])
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    results = {"threshold": thresholds.tolist(),
               "precision": precisions.tolist()[:-1],
               "recall": recalls.tolist()[:-1],
               "f1": f1_scores.tolist()[:-1],
               "best_threshold": round(best_threshold, 4),
               "best_f1": round(f1_scores[best_idx], 4)}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def explain_model(model, X_test, feature_names, explain_mode="shap",
                  max_features=30):
    """Generate model explanations."""
    print(f"\n{'='*60}")
    print("EXPLAINABILITY")
    print(f"{'='*60}")

    result = {"mode": explain_mode}

    if explain_mode == "shap" and HAS_SHAP:
        print("  Computing SHAP values...")
        t0 = time.time()
        try:
            # Extract inner model if wrapped in Pipeline or CalibratedClassifierCV
            inner_model = model
            if hasattr(model, "steps"):
                inner_model = model.steps[-1][1]
            if "CalibratedClassifierCV" in type(inner_model).__name__:
                inner_model = inner_model.estimator
            if hasattr(inner_model, "steps"):
                inner_model = inner_model.steps[-1][1]

            # Memory guard: cap at 200 * 20000 cells
            from scipy.sparse import issparse
            max_cells = 200 * 20000
            n_samples = X_test.shape[0]
            n_feats = X_test.shape[1]
            if n_samples * n_feats > max_cells:
                n_samples = min(n_samples, max_cells // n_feats)
                n_samples = max(n_samples, 100)
                print(f"    Memory guard: subsampling to {n_samples} docs")
            X_shap = X_test[:n_samples]
            if issparse(X_shap):
                X_shap = X_shap.toarray()

            if hasattr(inner_model, "get_booster") or "XGB" in type(inner_model).__name__:
                explainer = shap.TreeExplainer(inner_model)
                shap_values = explainer.shap_values(X_shap)
                result["explainer_type"] = "TreeExplainer"
            elif isinstance(inner_model, LogisticRegression) or \
                    "LogisticRegression" in type(inner_model).__name__:
                explainer = shap.LinearExplainer(inner_model, X_shap)
                shap_values = explainer.shap_values(X_shap)
                result["explainer_type"] = "LinearExplainer"
            else:
                n_bg = min(50, X_shap.shape[0])
                explainer = shap.KernelExplainer(model.predict_proba, X_shap[:n_bg])
                shap_values = explainer.shap_values(X_shap[:n_bg])
                result["explainer_type"] = "KernelExplainer"

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Global importance
            importances = np.abs(shap_values).mean(axis=0)
            # Adaptive top-K: show features covering 90% of total |SHAP|
            total_abs = np.sum(importances)
            sorted_idx = np.argsort(importances)[::-1]
            cumulative = np.cumsum(importances[sorted_idx]) / total_abs
            n_show = int(np.searchsorted(cumulative, 0.90) + 1)
            n_show = min(n_show, max_features)

            top_indices = sorted_idx[:n_show]

            top_features = []
            for idx in top_indices:
                name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                top_features.append({
                    "feature": name,
                    "importance": round(float(importances[idx]), 4),
                })

            result["global_importance"] = top_features
            elapsed = time.time() - t0
            print(f"  SHAP done in {elapsed:.1f}s")
            print(f"  Top 10 features (adaptive, {n_show} cover 90% of importance):")
            for f in top_features[:10]:
                print(f"    {f['feature']:<40} {f['importance']:.4f}")

        except Exception as e:
            print(f"  [!] SHAP failed: {e}. Falling back to permutation importance.")
            result["mode"] = "permutation_fallback"
            explain_mode = "permutation"

    if explain_mode in ("permutation", "permutation_fallback"):
        print("  Computing permutation importance...")
        t0 = time.time()
        from sklearn.inspection import permutation_importance
        try:
            X_perm = X_test
            from scipy.sparse import issparse
            if issparse(X_perm):
                X_perm = X_perm.toarray()
            perm = permutation_importance(
                model, X_perm, y_test,
                n_repeats=10, random_state=42, n_jobs=-1,
            )
            top_indices = np.argsort(perm.importances_mean)[-max_features:][::-1]
            top_features = []
            for idx in top_indices:
                name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                top_features.append({
                    "feature": name,
                    "importance": round(float(perm.importances_mean[idx]), 4),
                    "std": round(float(perm.importances_std[idx]), 4),
                })
            result["global_importance"] = top_features
            elapsed = time.time() - t0
            print(f"  Permutation importance done in {elapsed:.1f}s")
        except Exception as e:
            print(f"  [!] Permutation importance failed: {e}")

    # Also get built-in feature importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-max_features:][::-1]
        if not result.get("global_importance"):
            result["global_importance"] = []
            for idx in top_indices:
                name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                result["global_importance"].append({
                    "feature": name,
                    "importance": round(float(importances[idx]), 4),
                })

    # Feature family importance
    if result.get("global_importance"):
        families = defaultdict(float)
        for ft in result["global_importance"]:
            name = ft["feature"]
            if name.startswith("tfidf__"):
                families["tfidf"] += ft["importance"]
            elif name.startswith("sbert_dim_"):
                families["sbert_embeddings"] += ft["importance"]
            elif name.startswith("meta_"):
                families["metadata"] += ft["importance"]
            else:
                families["other"] += ft["importance"]
        result["feature_families"] = dict(families)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def train_classifier(df, text_col="text", label_col="label",
                     model_type="xgboost_preferred",
                     feature_mode="tfidf",
                     split_mode="random_stratified",
                     test_size=0.2, random_state=42,
                     n_estimators=100, max_features=12000,
                     class_weight_mode=None,
                     resampling=None,
                     calibrate=False,
                     auto_tune=False, optuna_trials=20,
                     cleanlab_audit=False,
                     explain_mode="shap",
                     validation_mode="cv", cv_folds=5,
                     n_bootstrap=1000,
                     group_col=None, date_col=None,
                     embedding_model="all-MiniLM-L6-v2",
                     output_dir="output"):
    """Full training pipeline.

    Returns dict with all results.
    """
    print(f"\n{'='*60}")
    print("TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"  Model: {model_type}")
    print(f"  Features: {feature_mode}")
    print(f"  Split: {split_mode}")
    print(f"  Test size: {test_size}")
    print(f"  Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Encode labels
    le = LabelEncoder()
    y_all = le.fit_transform(df["label"])
    n_classes = len(le.classes_)
    class_names = le.classes_.tolist()
    print(f"  Label encoding: {n_classes} classes → {class_names}")

    # Split
    print(f"\n  Splitting data ({split_mode})...")
    if split_mode == "temporal" and date_col and "date" in df.columns:
        df_sorted = df.sort_values("date").reset_index(drop=False)
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_texts = df_sorted["text"].iloc[:split_idx].tolist()
        test_texts = df_sorted["text"].iloc[split_idx:].tolist()
        y_sorted = y_all[df_sorted["index"].values]
        y_train = y_sorted[:split_idx]
        y_test = y_sorted[split_idx:]
        split_info = {"mode": "temporal", "train": split_idx, "test": len(df_sorted) - split_idx}
    else:
        train_texts, test_texts, y_train, y_test = train_test_split(
            df["text"].tolist(), y_all,
            test_size=test_size, random_state=random_state,
            stratify=y_all,
        )
        split_info = {"mode": "random_stratified",
                      "train": len(train_texts), "test": len(test_texts)}

    print(f"    Train: {len(train_texts)} docs")
    print(f"    Test:  {len(test_texts)} docs")

    # DistilBERT fine-tuning path (bypasses sklearn feature pipeline)
    if model_type == "distilbert":
        print(f"\n  Using DistilBERT fine-tuning path...")
        db_result = train_distilbert(
            train_texts, y_train, test_texts, y_test,
            class_names, output_dir=output_dir)
        if db_result is None:
            print("  [!] DistilBERT not available. Falling back to XGBoost.")
            model_type = "xgboost_preferred"
        else:
            y_pred = db_result["y_pred"]
            y_prob = db_result["y_prob"]
            metrics = compute_metrics(y_test, y_pred, y_prob, n_classes)
            train_time = db_result["train_time"]

            print(f"\n  Test Metrics:")
            print(f"    Accuracy:     {metrics['accuracy']:.4f}")
            print(f"    Macro F1:     {metrics['macro_f1']:.4f}")
            print(f"    Weighted F1:  {metrics['weighted_f1']:.4f}")
            print(f"    ROC-AUC:      {metrics.get('roc_auc', 0):.4f}")
            print(f"    Training:     {train_time:.1f}s")
            print(f"\n  {classification_report(y_test, y_pred, target_names=class_names)}")

            result = {
                "model": db_result["model"],
                "tokenizer": db_result.get("tokenizer"),
                "actual_type": "distilbert",
                "class_names": class_names,
                "n_classes": n_classes,
                "split_info": split_info,
                "metrics": metrics,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "training_time": train_time,
                "feature_engineering": {"X_test": None, "feature_names": []},
                "label_encoder": le,
            }
            save_artifacts(result, output_dir)
            return result

    # Features
    fe = build_features(train_texts, test_texts, feature_mode,
                        max_features, (1, 2), embedding_model)
    X_train, X_test = fe["X_train"], fe["X_test"]
    feature_names = fe["feature_names"]

    # Feature selection for large feature spaces
    selector = None
    if X_train.shape[1] > 5000 and feature_mode == "tfidf":
        X_train, feature_names, selector = select_top_features(
            X_train, y_train, feature_names, k=5000, method="chi2")
        X_test = selector.transform(X_test)

    # GPU detection
    use_gpu = False
    if model_type in ("xgboost_preferred", "xgb", "xgboost"):
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5)
            use_gpu = result.returncode == 0
        except Exception:
            use_gpu = False
        if use_gpu:
            print(f"  GPU detected — XGBoost will use CUDA")

    # Scale pos weight for imbalanced binary
    scale_pos_weight_val = None
    if n_classes == 2 and class_weight_mode == "balanced":
        class_counts = np.bincount(y_train)
        if class_counts[0] > 0 and class_counts[1] > 0:
            scale_pos_weight_val = class_counts[0] / class_counts[1]
            print(f"  scale_pos_weight: {scale_pos_weight_val:.2f} "
                  f"(neg={class_counts[0]}, pos={class_counts[1]})")

    # Split train into train/val for early stopping (on feature matrix indices)
    X_val, y_val = None, None
    if model_type in ("xgboost_preferred", "xgb", "xgboost") and len(train_texts) >= 100:
        n_val = max(1, int(len(train_texts) * 0.15))
        rng = np.random.RandomState(random_state)
        val_idx = rng.choice(len(train_texts), n_val, replace=False)
        train_idx_inner = np.array([i for i in range(len(train_texts)) if i not in val_idx])
        print(f"  Early stopping split: train={len(train_idx_inner)}, val={n_val}")
    else:
        train_idx_inner = None

    # Model (raw, unwrapped)
    model, actual_type, fallback_note = make_model(
        model_type, random_state, class_weight_mode, n_classes,
        n_estimators, scale_pos_weight=scale_pos_weight_val, use_gpu=use_gpu,
    )
    print(f"\n  Model: {actual_type}")
    if fallback_note:
        print(f"    Note: {fallback_note}")

    # Optuna tuning (on raw model before pipeline wrapping)
    tuning_report = None
    if auto_tune and HAS_OPTUNA:
        print(f"\n  Hyperparameter tuning with Optuna ({optuna_trials} trials)...")
        tuning_report = run_optuna_tuning(
            X_train, y_train, model_type, optuna_trials, random_state)
        if tuning_report and tuning_report.get("best_params"):
            model.set_params(**tuning_report["best_params"])

    # Resampling (wraps the tuned model)
    if resampling and HAS_IMBLEARN:
        model, resample_type, resample_note = make_resampling_pipeline(
            model, resampling, random_state)
        print(f"  Resampling: {resampling}")

    # Early stopping: split X_train into train/val on the feature matrix
    if train_idx_inner is not None:
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        X_train_fit = X_train[train_idx_inner]
        y_train_fit = y_train[train_idx_inner]
        print(f"  Training with {len(train_idx_inner)} docs, val set has {len(val_idx)} docs")
    else:
        X_train_fit, y_train_fit = X_train, y_train

    # Train
    print(f"\n  Training...")
    t0 = time.time()
    fit_kwargs = {}
    if actual_type == "xgboost" and X_val is not None and train_idx_inner is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
        if hasattr(model, "set_params"):
            try:
                model.set_params(early_stopping_rounds=50)
            except Exception:
                pass
    model.fit(X_train_fit, y_train_fit, **fit_kwargs)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.2f}s")
    # Disable early stopping for downstream CV/calibration
    if hasattr(model, "set_params"):
        try:
            model.set_params(early_stopping_rounds=0)
        except Exception:
            pass

    # Predict
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        y_prob = None

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_prob, n_classes)
    print(f"\n  Test Metrics:")
    print(f"    Accuracy:     {metrics['accuracy']:.4f}")
    print(f"    Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"    Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"    Kappa:        {metrics['kappa']:.4f}")
    print(f"    MCC:          {metrics['mcc']:.4f}")
    if metrics.get("roc_auc"):
        print(f"    ROC-AUC:      {metrics['roc_auc']:.4f}")
    if metrics.get("brier"):
        print(f"    Brier score:  {metrics['brier']:.4f}")

    # Per-class metrics
    print(f"\n  Per-Class Metrics:")
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for cls, cm in metrics["per_class"].items():
        class_name = class_names[int(cls)] if int(cls) < len(class_names) else cls
        print(f"  {class_name:<15} {cm['precision']:>10.4f} {cm['recall']:>10.4f} "
              f"{cm['f1']:>8.4f} {cm['support']:>8}")

    # Classification report
    print(f"\n  {classification_report(y_test, y_pred, target_names=class_names)}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>12}", end="")
    for name in class_names:
        print(f" {name:>10}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"  {name:<10}", end="")
        for j in range(len(class_names)):
            print(f" {cm[i, j]:>10}", end="")
        print()

    # Bootstrap CI
    if n_bootstrap > 0:
        ci = bootstrap_ci(y_test, y_pred, n_bootstrap)
        print(f"\n  Bootstrap 95% CI (macro-F1): "
              f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] (mean={ci['mean']:.4f})")
        metrics["bootstrap_ci"] = ci

    # Cross-validation
    cv_report = None
    if validation_mode == "cv" and cv_folds > 0:
        cv_report = run_cross_validation(
            X_train, y_train, model, cv_folds,
            groups=None, parallel=True,
        )

    # Threshold optimization (binary only)
    threshold_report = None
    if n_classes == 2 and y_prob is not None:
        threshold_report = run_threshold_optimization(y_test, y_prob)
        print(f"\n  Optimal threshold: {threshold_report['best_threshold']:.4f} "
              f"(best F1: {threshold_report['best_f1']:.4f})")

    # Calibration
    calibration_report = None
    if calibrate and n_classes == 2:
        from sklearn.calibration import calibration_curve
        print(f"\n  Calibration: wrapping model in CalibratedClassifierCV...")
        try:
            # Disable early stopping on clone so CV folds don't fail
            model_for_cal = skclone(model)
            if hasattr(model_for_cal, "set_params"):
                model_for_cal.set_params(early_stopping_rounds=0)
            cal_model = CalibratedClassifierCV(
                model_for_cal, method="sigmoid", cv=3)
            cal_model.fit(X_train, y_train)
            cal_y_prob = cal_model.predict_proba(X_test)
            prob_true, prob_pred = calibration_curve(
                y_test, cal_y_prob[:, 1], n_bins=10)
            ece = np.mean(np.abs(prob_true - prob_pred)).item()
            print(f"    Calibrated ECE: {ece:.4f}")
            model = cal_model
            y_prob = cal_y_prob
            y_pred = cal_model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred, y_prob, n_classes)
        except Exception as e:
            print(f"    [!] Calibration failed: {e}. Using uncalibrated model.")
            prob_true, prob_pred = calibration_curve(
                y_test, y_prob[:, 1], n_bins=10)
            ece = np.mean(np.abs(prob_true - prob_pred)).item()
        calibration_report = {
            "ece": round(ece, 4),
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        print(f"    Expected Calibration Error: {ece:.4f}")
        if ece > 0.10:
            print(f"    ⚠ Calibration error > 0.10 — calibration may need more data")

    # Cleanlab label audit
    cleanlab_report = None
    if cleanlab_audit and HAS_CLEANLAB:
        print(f"\n  Cleanlab label quality audit...")
        try:
            from cleanlab.filter import find_label_issues
            from cleanlab.rank import get_label_quality_scores

            oof_pred_probs = cross_val_predict(model, X_train, y_train,
                                               cv=3, method="predict_proba")
            label_issues = find_label_issues(
                y_train, oof_pred_probs, return_indices_ranked_by="self_confidence")
            quality_scores = get_label_quality_scores(y_train, oof_pred_probs)
            issue_examples = []
            for idx in label_issues[:20]:
                issue_examples.append({
                    "index": int(idx),
                    "text": train_texts[idx][:200],
                    "current_label": class_names[y_train[idx]],
                    "quality_score": round(float(quality_scores[idx]), 4),
                })
            cleanlab_report = {
                "n_label_issues": len(label_issues),
                "n_total": len(y_train),
                "issue_pct": round(len(label_issues) / len(y_train) * 100, 2),
                "examples": issue_examples,
            }
            print(f"    Label issues found: {cleanlab_report['n_label_issues']} "
                  f"({cleanlab_report['issue_pct']:.1f}%)")
        except Exception as e:
            print(f"    [!] Cleanlab audit failed: {e}")

    # Explainability
    explanation = None
    if explain_mode != "none":
        explanation = explain_model(
            model, X_test[:500], feature_names, explain_mode, 30)

    # Build result
    result = {
        "model": model,
        "actual_type": actual_type,
        "fallback_note": fallback_note,
        "class_names": class_names,
        "n_classes": n_classes,
        "split_info": split_info,
        "metrics": metrics,
        "cv_report": cv_report,
        "threshold_report": threshold_report,
        "calibration_report": calibration_report,
        "cleanlab_report": cleanlab_report,
        "explanation": explanation,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_engineering": fe,
        "label_encoder": le,
        "selector": selector,
        "training_time": train_time,
    }

    # Save artifacts
    save_artifacts(result, output_dir)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6.5: DISTILBERT FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_distilbert(train_texts, train_labels, test_texts, test_labels,
                     class_names, n_epochs=5, batch_size=16,
                     learning_rate=2e-5, output_dir="output"):
    """Fine-tune DistilBERT for text classification."""
    if not HAS_TORCH_TRANSFORMERS:
        return None

    print(f"\n  Fine-tuning DistilBERT...")
    t0 = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def _tokenize(texts):
        return tokenizer(texts, padding=True, truncation=True,
                         max_length=512, return_tensors="pt")

    train_enc = _tokenize(train_texts)
    test_enc = _tokenize(test_texts)

    n_classes = len(set(train_labels))
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=n_classes)
    model.to(device)

    train_dataset = TextDataset(train_enc, train_labels.tolist())
    test_dataset = TextDataset(test_enc, test_labels.tolist())

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, "distilbert_checkpoints"),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=os.path.join(output_dir, "distilbert_logs"),
        logging_steps=50,
        report_to="none",
        fp16=False,
        dataloader_pin_memory=False,
    )

    def compute_metrics_fn(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    train_time = time.time() - t0
    print(f"  DistilBERT training done in {train_time:.1f}s")

    # Predict
    model.eval()
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in test_enc.items()}
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()
        y_pred = np.argmax(logits, axis=-1)
        y_prob = torch.nn.functional.softmax(
            torch.from_numpy(logits), dim=-1).numpy()

    result = {
        "model": model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "train_time": train_time,
    }
    return result


def run_comparison(df, output_dir="output", feature_mode="tfidf+sbert",
                   cv_folds=3, n_bootstrap=500):
    """Run multiple model/feature combinations and compare."""
    MODES = [
        ("XGBoost+TF-IDF", "xgboost", "tfidf"),
        ("XGBoost+SBERT", "xgboost", "sbert"),
        ("XGBoost+DistilBERT", "xgboost", "distilbert"),
        ("XGBoost+RoBERTa", "xgboost", "roberta"),
        ("XGBoost+MPNet", "xgboost", "mpnet"),
        ("XGBoost+DeBERTa", "xgboost", "deberta"),
        ("XGBoost+TF-IDF+DistilBERT", "xgboost", "tfidf+distilbert"),
        ("XGBoost+TF-IDF+RoBERTa", "xgboost", "tfidf+roberta"),
        ("XGBoost+TF-IDF+MPNet", "xgboost", "tfidf+mpnet"),
        ("XGBoost+TF-IDF+DeBERTa", "xgboost", "tfidf+deberta"),
    ]
    if HAS_TORCH_TRANSFORMERS:
        MODES.append(("DistilBERT FT", "distilbert", "tfidf"))

    results = {}
    print(f"\n{'='*60}")
    print("COMPARISON MODE")
    print(f"{'='*60}")

    for name, model_type, feats in MODES:
        print(f"\n  {'─'*50}")
        print(f"  [{name}] model={model_type}, features={feats}")

        out_sub = os.path.join(output_dir, name.replace("+", "_").replace(" ", "_"))
        try:
            r = train_classifier(
                df, text_col="text", label_col="label",
                model_type=model_type, feature_mode=feats,
                split_mode="random_stratified", test_size=0.2,
                max_features=12000, output_dir=out_sub,
                cv_folds=cv_folds, n_bootstrap=n_bootstrap,
                validation_mode="cv", explain_mode="none",
                calibrate=False,
            )
            results[name] = {
                "macro_f1": r["metrics"]["macro_f1"],
                "accuracy": r["metrics"]["accuracy"],
                "roc_auc": r["metrics"].get("roc_auc"),
                "kappa": r["metrics"]["kappa"],
                "mcc": r["metrics"]["mcc"],
                "weighted_f1": r["metrics"]["weighted_f1"],
                "training_time": r["training_time"],
                "class_reports": r["metrics"].get("per_class"),
            }
            if r.get("bootstrap_ci"):
                results[name]["bootstrap_ci"] = r["bootstrap_ci"]
            if r.get("cv_report") is not None and isinstance(r["cv_report"], dict):
                cv_f1s = [f["macro_f1"] for f in r["cv_report"]["fold_metrics"]]
                results[name]["cv_mean_f1"] = round(float(np.mean(cv_f1s)), 4)
                results[name]["cv_std_f1"] = round(float(np.std(cv_f1s)), 4)
        except Exception as e:
            import traceback
            results[name] = {"macro_f1": None, "error": str(e)}
            print(f"    [!] Failed: {e}")
            traceback.print_exc()

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    header = f"{'Model':<30} {'MacroF1':<8} {'Acc':<8} {'AUC':<8} {'Kappa':<8} {'Time':<8} {'CV F1':<8}"
    print(header)
    print(f"{'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    sorted_names = sorted(results.keys(),
                          key=lambda k: results[k].get("macro_f1") or 0,
                          reverse=True)
    for name in sorted_names:
        r = results[name]
        mf1 = f"{r['macro_f1']:.4f}" if r.get("macro_f1") is not None else "  FAIL"
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "   —"
        auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") is not None else "   —"
        kap = f"{r['kappa']:.4f}" if r.get("kappa") is not None else "   —"
        tm = f"{r['training_time']:.1f}s" if r.get("training_time") is not None else "   —"
        cv = f"{r.get('cv_mean_f1', 0):.4f}" if r.get("cv_mean_f1") else "   —"
        print(f"{name:<30} {mf1:<8} {acc:<8} {auc:<8} {kap:<8} {tm:<8} {cv:<8}")

    # Overfitting check
    print(f"\n  Overfitting Check:")
    print(f"  {'Model':<30} {'Test F1':<10} {'CV F1':<10} {'Delta':<10} {'Verdict':<12}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")
    for name in sorted_names:
        r = results[name]
        mf1 = r.get("macro_f1")
        cv = r.get("cv_mean_f1")
        if mf1 is not None and cv is not None:
            delta = mf1 - cv
            verdict = "OK" if abs(delta) < 0.05 else ("OVERFIT ⚠" if delta < -0.05 else "SUSPICIOUS ⚠")
            print(f"{name:<30} {mf1:<10.4f} {cv:<10.4f} {delta:<+10.4f} {verdict:<12}")

    # Save comparison
    comp_df = pd.DataFrame([
        {"Model": k, **{kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}}
        for k, v in results.items()
    ])
    comp_path = os.path.join(output_dir, "comparison.csv")
    comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
    print(f"\n  Comparison saved to {comp_path}")

    return results


def run_optuna_tuning(X, y, model_type, n_trials=20, random_state=42):
    """Hyperparameter tuning with Optuna."""
    def objective(trial):
        if model_type in ("xgb", "xgboost", "xgboost_preferred"):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 0.5),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10, log=True),
            }
            model = xgb.XGBClassifier(**params, random_state=random_state,
                                       eval_metric="logloss", tree_method="hist",
                                       verbosity=0)
        elif model_type in ("logreg", "logistic_regression"):
            params = {
                "C": trial.suggest_float("C", 0.001, 100, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            }
            model = LogisticRegression(**params, max_iter=1000, random_state=random_state)
        elif model_type in ("rf", "random_forest"):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            model = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
        else:
            return 0.0

        scores = cross_val_score(model, X, y, cv=3, scoring="f1_macro")
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    report = {
        "best_params": study.best_params,
        "best_value": round(study.best_value, 4),
        "n_trials": n_trials,
    }
    print(f"    Best macro-F1: {report['best_value']:.4f}")
    print(f"    Best params: {report['best_params']}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_artifacts(result, output_dir):
    """Save model, vectorizers, predictions, and reports."""
    os.makedirs(output_dir, exist_ok=True)

    # Predictions CSV
    y_test = result["y_test"]
    y_pred = result["y_pred"]
    pred_df = pd.DataFrame({
        "true_label": [result["class_names"][i] for i in y_test],
        "pred_label": [result["class_names"][i] for i in y_pred],
    })
    if result["y_prob"] is not None:
        for i, name in enumerate(result["class_names"]):
            pred_df[f"prob_{name}"] = result["y_prob"][:, i]
    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"),
                   index=False, encoding="utf-8-sig")

    # Feature engineering artifacts
    fe = result["feature_engineering"]
    if HAS_JOBLIB:
        if fe["vectorizer"] is not None:
            joblib.dump(fe["vectorizer"],
                        os.path.join(output_dir, "vectorizer.joblib"))
        joblib.dump(result["label_encoder"],
                    os.path.join(output_dir, "label_encoder.joblib"))
        if fe["scaler"] is not None:
            joblib.dump(fe["scaler"],
                        os.path.join(output_dir, "scaler.joblib"))
        if result["selector"] is not None:
            joblib.dump(result["selector"],
                        os.path.join(output_dir, "feature_selector.joblib"))

    # Model
    if HAS_JOBLIB:
        joblib.dump(result["model"],
                    os.path.join(output_dir, "model.joblib"))

    print(f"\n  Saved artifacts to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: PREDICT ON NEW DATA
# ═══════════════════════════════════════════════════════════════════════════════

def predict_on_new(df_new, output_dir="output"):
    """Load saved model and predict on new data."""
    print(f"\n{'='*60}")
    print("PREDICT ON NEW DATA")
    print(f"{'='*60}")

    if not HAS_JOBLIB:
        print("  [!] joblib required for model loading")
        return

    model_path = os.path.join(output_dir, "model.joblib")
    vec_path = os.path.join(output_dir, "vectorizer.joblib")
    le_path = os.path.join(output_dir, "label_encoder.joblib")

    if not all(os.path.exists(p) for p in [model_path, vec_path, le_path]):
        print("  [!] No saved model found. Train first.")
        return

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    le = joblib.load(le_path)

    texts = df_new["text"].tolist()
    X = vectorizer.transform(texts)

    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)
    except Exception:
        y_prob = None

    results = pd.DataFrame({
        "text": texts,
        "pred_label": le.inverse_transform(y_pred),
    })
    if y_prob is not None:
        for i, name in enumerate(le.classes_):
            results[f"prob_{name}"] = y_prob[:, i]

    out_path = os.path.join(output_dir, "predictions_new.csv")
    results.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Predicted {len(results)} documents")
    print(f"  Saved to {out_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, class_names, output_dir="output"):
    """Plot confusion matrix."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix to {path}")


def plot_feature_importance(result, output_dir="output", top_n=20):
    """Plot feature importance bar chart."""
    if not HAS_MATPLOTLIB:
        return
    if not result.get("explanation") or not result["explanation"].get("global_importance"):
        return

    features = result["explanation"]["global_importance"][:top_n]
    names = [f["feature"][:50] for f in features]
    scores = [f["importance"] for f in features]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), scores, color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + max(scores) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature importance to {path}")


def plot_calibration_curve(result, output_dir="output"):
    """Plot calibration curve."""
    if not HAS_MATPLOTLIB:
        return
    if not result.get("calibration_report"):
        return

    cr = result["calibration_report"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)
    ax.plot(cr["prob_pred"], cr["prob_true"], "o-", label=f"Model (ECE={cr['ece']:.4f})")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved calibration curve to {path}")


def plot_threshold_curve(result, output_dir="output"):
    """Plot threshold vs F1 curve."""
    if not HAS_MATPLOTLIB:
        return
    if not result.get("threshold_report"):
        return

    tr = result["threshold_report"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(tr["threshold"], tr["f1"], label="F1", linewidth=2)
    ax.plot(tr["threshold"], tr["precision"], label="Precision", alpha=0.7)
    ax.plot(tr["threshold"], tr["recall"], label="Recall", alpha=0.7)
    ax.axvline(tr["best_threshold"], color="red", linestyle="--",
               label=f"Best threshold={tr['best_threshold']:.3f} (F1={tr['best_f1']:.3f})")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Optimization", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved threshold curve to {path}")


def plot_cv_result(cv_report, output_dir="output"):
    """Plot cross-validation fold results."""
    if not HAS_MATPLOTLIB or cv_report is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_plot = ["accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(cv_report))
    width = 0.25

    for i, metric in enumerate(metrics_to_plot):
        if metric in cv_report.columns:
            vals = cv_report[metric].dropna()
            if len(vals) > 0:
                ax.bar(x + i * width, vals, width, label=metric)

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Cross-Validation Results", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(len(cv_report))])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "cv_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved CV results to {path}")


def plot_shap_summary(result, X_test, feature_names, class_names, output_dir="output"):
    """Plot SHAP beeswarm summary."""
    if not HAS_MATPLOTLIB or not HAS_SHAP:
        return
    if not result.get("explanation"):
        return
    if result["explanation"].get("explainer_type") not in ("TreeExplainer", "LinearExplainer"):
        return

    model = result["model"]
    inner_model = model
    if hasattr(model, "steps"):
        inner_model = model.steps[-1][1]
    if "CalibratedClassifierCV" in type(inner_model).__name__:
        inner_model = inner_model.estimator
    if hasattr(inner_model, "steps"):
        inner_model = inner_model.steps[-1][1]
    try:
        if result["explanation"]["explainer_type"] == "TreeExplainer":
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X_test[:200])
        elif result["explanation"]["explainer_type"] == "LinearExplainer":
            explainer = shap.LinearExplainer(inner_model, X_test[:200])
            shap_values = explainer.shap_values(X_test[:200])
        else:
            return

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        n_feat = shap_values.shape[1]
        shap_feature_names = feature_names[:n_feat]

        # Beeswarm plot (replaces deprecated summary_plot)
        shap.summary_plot(shap_values, X_test[:200],
                          feature_names=shap_feature_names,
                          show=False, max_display=20, plot_type="dot")
        plt.tight_layout()
        path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP summary to {path}")
    except Exception as e:
        print(f"  [!] SHAP plot failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(result, output_dir="output"):
    """Generate comprehensive HTML report."""
    def img_to_b64(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    metrics = result["metrics"]
    class_names = result["class_names"]

    cm = confusion_matrix(result["y_test"], result["y_pred"])

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictive Modeling ULTRA — Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa;
         color: #333; line-height: 1.6; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #2980b9; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 12px; }
  h3 { color: #555; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
  th { background: #3498db; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #ddd; }
  tr:hover { background: #f0f7ff; }
  .metric-box { display: inline-block; background: #ecf0f1; border-radius: 6px;
                padding: 12px 20px; margin: 6px; text-align: center; min-width: 100px; }
  .metric-box .value { font-size: 22px; font-weight: bold; color: #2c3e50; }
  .metric-box .label { font-size: 11px; color: #7f8c8d; margin-top: 4px; }
  .section { background: white; border-radius: 8px; padding: 20px;
             margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .summary { background: #eaf2f8; border-left: 4px solid #3498db;
             padding: 12px 16px; border-radius: 4px; margin: 15px 0; }
  img { max-width: 100%; border-radius: 6px; margin: 10px 0; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 13px; }
</style>
</head>
<body>
<h1>Predictive Modeling ULTRA — Classification Report</h1>
<p style="color:#777;">Model: """ + result["actual_type"] + """ · """ + str(result["n_classes"]) + """ classes · """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
"""

    # Summary metrics
    html += '<div class="section"><h2>Overall Metrics</h2>\n'
    html += '<div class="summary">\n'
    for key, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"),
                        ("weighted_f1", "Weighted F1"), ("kappa", "Cohen's κ"),
                        ("mcc", "MCC"), ("roc_auc", "ROC-AUC")]:
        val = metrics.get(key)
        if val is not None:
            html += f'<div class="metric-box"><div class="value">{val:.4f}</div><div class="label">{label}</div></div>\n'
    html += '</div></div>\n'

    # Per-class
    html += f'<div class="section"><h2>Per-Class Metrics</h2>\n'
    html += '<table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>\n'
    for cls, cm_data in metrics["per_class"].items():
        name = class_names[int(cls)] if int(cls) < len(class_names) else cls
        html += f'<tr><td>{name}</td><td>{cm_data["precision"]:.4f}</td>'
        html += f'<td>{cm_data["recall"]:.4f}</td><td>{cm_data["f1"]:.4f}</td>'
        html += f'<td>{cm_data["support"]}</td></tr>\n'
    html += '</table></div>\n'

    # Confusion matrix table
    html += f'<div class="section"><h2>Confusion Matrix</h2>\n'
    html += '<table><tr><th>True ↓ / Pred →</th>'
    for name in class_names:
        html += f'<th>{name}</th>'
    html += '</tr>\n'
    for i, name in enumerate(class_names):
        html += f'<tr><td><strong>{name}</strong></td>'
        for j in range(len(class_names)):
            html += f'<td>{cm[i, j]}</td>'
        html += '</tr>\n'
    html += '</table></div>\n'

    # Images
    for name, title in [("confusion_matrix", "Confusion Matrix"),
                         ("feature_importance", "Feature Importance"),
                         ("calibration_curve", "Calibration Curve"),
                         ("threshold_curve", "Threshold Optimization"),
                         ("shap_summary", "SHAP Summary")]:
        b64 = img_to_b64(os.path.join(output_dir, f"{name}.png"))
        if b64:
            html += f'<div class="section"><h2>{title}</h2>\n'
            html += f'<img src="data:image/png;base64,{b64}" alt="{title}">\n'
            html += '</div>\n'

    # Feature importance table
    if result.get("explanation") and result["explanation"].get("global_importance"):
        html += f'<div class="section"><h2>Top Features</h2>\n'
        html += '<table><tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>\n'
        for i, ft in enumerate(result["explanation"]["global_importance"][:30], 1):
            html += f'<tr><td>{i}</td><td><code>{ft["feature"]}</code></td>'
            html += f'<td>{ft["importance"]:.4f}</td></tr>\n'
        html += '</table></div>\n'

    html += """
<div class="footer" style="text-align:center;color:#aaa;font-size:12px;margin-top:40px;">
  Predictive Modeling ULTRA v1.0 — Powered by sklearn, xgboost, shap, optuna
</div>
</body>
</html>"""

    path = os.path.join(output_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML report saved to {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Predictive Modeling ULTRA — Text Classification Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predictive_ultra.py data.csv text_col label_col
  python predictive_ultra.py data.csv text_col label_col --all
  python predictive_ultra.py data.csv text_col label_col --model xgboost --features sbert
  python predictive_ultra.py data.csv text_col label_col --model distilbert
  python predictive_ultra.py data.csv text_col label_col --features tfidf+distilbert
  python predictive_ultra.py data.csv text_col label_col --compare
  python predictive_ultra.py data.csv text_col label_col --split temporal --date date_col
  python predictive_ultra.py data.csv text_col label_col --group user --model rf
  python predictive_ultra.py data.csv text_col label_col --tune --calibrate --explain shap
  python predictive_ultra.py data.csv text_col label_col --resample smote --cleanlab
  python predictive_ultra.py data.csv text_col label_col --test new_data.csv
        """,
    )
    parser.add_argument("corpus", help="Path to CSV/TSV corpus file")
    parser.add_argument("text_col", help="Name of text column")
    parser.add_argument("label_col", help="Name of label column")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--encoding", default="utf-8", help="File encoding")
    parser.add_argument("--model", default="xgboost_preferred",
                        help="Model: xgboost, logreg, svm, rf, nb, sgd, ensemble, distilbert")
    parser.add_argument("--features", default="tfidf",
                        help="Features: tfidf, sbert, distilbert, roberta, mpnet, deberta, tfidf+{model}")
    parser.add_argument("--split", default="random_stratified",
                        help="Split mode: random_stratified, temporal, group")
    parser.add_argument("--group", help="Group column for group split")
    parser.add_argument("--date", help="Date column for temporal split")
    parser.add_argument("--test", help="Path to new data for prediction")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--n-estimators", type=int, default=100)

    parser.add_argument("--tune", action="store_true", help="Optuna hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trial count")
    parser.add_argument("--calibrate", action="store_true", help="Calibration analysis")
    parser.add_argument("--resample",
                        help="Resampling: smote, adasyn, oversample, undersample")
    parser.add_argument("--explain", default="shap",
                        help="Explainability: shap, permutation, none")
    parser.add_argument("--cleanlab", action="store_true",
                        help="Cleanlab label quality audit")
    parser.add_argument("--class-weight", help="Class weight: balanced, None")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap iterations for CI (0 to skip)")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--report", action="store_true", help="HTML report")
    parser.add_argument("--all", action="store_true", help="Run with all options")
    parser.add_argument("--compare", action="store_true",
                        help="Run multiple model/feature combos and compare")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("PREDICTIVE MODELING ULTRA v1.0")
    print("=" * 60)
    print(f"  Corpus: {args.corpus}")
    print(f"  Text: {args.text_col}, Label: {args.label_col}")
    print(f"  Model: {args.model}")
    print(f"  Features: {args.features}")
    print(f"  Output: {args.output}")

    # Check dependencies
    missing = []
    if not HAS_SKLEARN: missing.append("scikit-learn")
    if not HAS_XGB: missing.append("xgboost")
    if not HAS_MODELS: missing.append("sklearn models")
    if not HAS_SHAP: missing.append("shap")
    if not HAS_IMBLEARN: missing.append("imbalanced-learn")
    if not HAS_OPTUNA: missing.append("optuna")
    if not HAS_CLEANLAB: missing.append("cleanlab")
    if not HAS_SBERT: missing.append("sentence-transformers")
    if not HAS_TORCH_TRANSFORMERS: missing.append("transformers/torch")
    if not HAS_MATPLOTLIB: missing.append("matplotlib")
    if not HAS_JOBLIB: missing.append("joblib")
    if missing:
        print(f"  [!] Missing: {', '.join(missing)}")
    else:
        print("  All dependencies available.")

    # Load data
    print("\n  Loading data...")
    df = load_data(args.corpus, args.text_col, args.label_col,
                   args.group, args.date, args.encoding)
    print(f"  Loaded {len(df)} documents ({df['label'].nunique()} classes)")

    # Preflight
    passed, issues = run_preflight(df, args.label_col)
    if not passed:
        print("\n  ✗ Cannot proceed. Fix blockers and retry.")
        return

    # Determine flags from --all
    run_all = args.all
    if run_all:
        args.calibrate = True
        args.tune = True
        args.cleanlab = True
        args.resample = args.resample or "smote"
        args.explain = "shap"
        args.plots = True
        args.report = True
        if not args.group and not args.date:
            pass

    # Comparison mode — runs multiple model/feature combos
    if args.compare:
        run_comparison(df, args.output, args.features,
                       cv_folds=args.cv_folds, n_bootstrap=args.bootstrap)
        return

    # Train
    result = train_classifier(
        df,
        text_col="text",
        label_col="label",
        model_type=args.model,
        feature_mode=args.features,
        split_mode=args.split,
        test_size=args.test_size,
        max_features=args.max_features,
        n_estimators=args.n_estimators,
        class_weight_mode=args.class_weight,
        resampling=args.resample,
        calibrate=args.calibrate,
        auto_tune=args.tune,
        optuna_trials=args.trials,
        cleanlab_audit=args.cleanlab,
        explain_mode=args.explain,
        cv_folds=args.cv_folds,
        n_bootstrap=args.bootstrap,
        group_col=args.group,
        date_col=args.date,
        output_dir=args.output,
    )

    # Predict on new data
    if args.test:
        df_new = load_data(args.test, args.text_col, args.label_col,
                           args.encoding)
        predict_on_new(df_new, args.output)

    # Plots
    if run_all or args.plots:
        print(f"\n{'='*60}")
        print("GENERATING PLOTS")
        print(f"{'='*60}")
        plot_confusion_matrix(
            confusion_matrix(result["y_test"], result["y_pred"]),
            result["class_names"], args.output)
        plot_feature_importance(result, args.output)
        plot_calibration_curve(result, args.output)
        plot_threshold_curve(result, args.output)
        if result.get("cv_report") is not None:
            plot_cv_result(result["cv_report"], args.output)
        if result["feature_engineering"]:
            plot_shap_summary(
                result,
                result["feature_engineering"]["X_test"][:200],
                result["feature_engineering"]["feature_names"],
                result["class_names"],
                args.output)

    # HTML report
    if run_all or args.report:
        generate_html_report(result, args.output)

    # Summary
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"  Model: {result['actual_type']}")
    print(f"  Macro F1: {result['metrics']['macro_f1']:.4f}")
    print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"  Output: {os.path.abspath(args.output)}")
    print(f"  Files:")
    for f in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            s = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
            print(f"    {f:<40} {s:>10}")

    return result


if __name__ == "__main__":
    main()
