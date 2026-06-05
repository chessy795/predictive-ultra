# Predictive Modeling ULTRA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2+-orange.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-yellow.svg)](https://scikit-learn.org/)

Standalone Python text classification pipeline: XGBoost + transformer embeddings + SHAP explainability + calibration + cross-validation + model comparison. CPU-first, GPU-optional, one-script deploy.

**Winner on benchmarks (TA23, 542 docs):** XGBoost + TF-IDF + MPNet → Macro F1 **0.9115**

## Features

| Capability | Detail |
|------------|--------|
| **7 classifiers** | XGBoost, Logistic Regression, SVM, Random Forest, ComplementNB, SGD, Ensemble (soft voting / stacking) |
| **9 feature modes** | TF-IDF, SBERT, DistilBERT, RoBERTa, MPNet, DeBERTa, plus TF-IDF + any embedding hybrid |
| **DistilBERT fine-tuning** | HuggingFace Trainer API, CPU-friendly |
| **SHAP explainability** | TreeExplainer / LinearExplainer / KernelExplainer with adaptive top-K (90% coverage) |
| **Calibration** | Platt scaling via `CalibratedClassifierCV`, Expected Calibration Error (ECE) |
| **Cross-validation** | Stratified K-Fold, Repeated K-Fold, Group K-Fold, temporal / blocked splits |
| **Class imbalance** | SMOTE, ADASYN, random over/under sampling via `imbalanced-learn` |
| **Hyperparameter tuning** | Optuna with configurable search space and TPESampler |
| **Label quality audit** | Cleanlab integration for finding annotation errors |
| **Bootstrap CI** | Percentile method, 95% confidence interval for macro-F1 |
| **Comparison mode** | `--compare` benchmarks all 10 model/feature combos in one shot, with overfitting delta check |
| **Self-contained HTML report** | All metrics, confusion matrix, calibration curve, SHAP summary, feature importance |
| **CPU-first** | No GPU required for most modes; GPU auto-detected for XGBoost via `nvidia-smi` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic XGBoost with TF-IDF
python predictive_ultra.py data.csv text_col label_col

# Best-performing hybrid: XGBoost + TF-IDF + MPNet
python predictive_ultra.py data.csv text_col label_col --features tfidf+mpnet

# Compare all model/feature combinations
python predictive_ultra.py data.csv text_col label_col --compare

# Full pipeline with everything
python predictive_ultra.py data.csv text_col label_col --all
```

## Installation

```bash
git clone https://github.com/chessy795/predictive-ultra.git
cd predictive-ultra
pip install -r requirements.txt
```

**Platform notes:**
- **Windows**: Use Python 3.10+ from python.org. Install Microsoft Visual C++ Build Tools if `torch` fails.
- **macOS**: `pip install -r requirements.txt` works natively on Apple Silicon (MPS not used — CPU is fine).
- **Linux**: All dependencies available via pip. GPU auto-detected for XGBoost if CUDA toolkit is installed.

## Usage

```
python predictive_ultra.py <corpus.csv> <text_col> <label_col> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `corpus` | Path to CSV/TSV file |
| `text_col` | Column name containing text |
| `label_col` | Column name containing labels |
| `--output, -o` | Output directory (default: output/) |
| `--model` | Model: `xgboost`, `logreg`, `svm`, `rf`, `nb`, `sgd`, `ensemble`, `distilbert` (default: xgboost) |
| `--features` | Features: `tfidf`, `sbert`, `distilbert`, `roberta`, `mpnet`, `deberta`, or `tfidf+{model}` hybrid (default: tfidf) |
| `--split` | Split: `random_stratified`, `temporal`, `blocked_temporal`, `group` (default: random_stratified) |
| `--group` | Group column for group split |
| `--date` | Date column for temporal split |
| `--test` | Path to new data for prediction |
| `--test-size` | Test set proportion (default: 0.2) |
| `--cv-folds` | CV folds (default: 5) |
| `--max-features` | Max TF-IDF features (default: 12000) |
| `--tune` | Enable Optuna hyperparameter tuning |
| `--trials` | Optuna trial count (default: 20) |
| `--calibrate` | Calibrate probabilities (Platt scaling) |
| `--resample` | Resampling: `smote`, `adasyn`, `oversample`, `undersample` |
| `--explain` | Explanation mode: `shap`, `permutation`, `none` (default: shap) |
| `--cleanlab` | Audit label quality via Cleanlab |
| `--class-weight` | Class weight: `balanced` or None |
| `--bootstrap` | Bootstrap iterations for CI (default: 1000) |
| `--all` | Run everything (tune, calibrate, resample, explain, cleanlab, plots, report) |
| `--compare` | Run all model/feature combinations and compare |

### Examples

```bash
# Default: XGBoost + TF-IDF
python predictive_ultra.py reviews.csv text sentiment

# Best hybrid: XGBoost + TF-IDF + MPNet
python predictive_ultra.py reviews.csv text sentiment --features tfidf+mpnet

# XGBoost with temporal split
python predictive_ultra.py reviews.csv text sentiment --date date --split temporal

# Logistic regression with SMOTE resampling
python predictive_ultra.py data.csv text label --model logreg --resample smote

# Evaluate on new data
python predictive_ultra.py train.csv text label --test test.csv

# DistilBERT fine-tuning (CPU ~10-15 min for 500 docs)
python predictive_ultra.py data.csv text label --model distilbert

# RoBERTa embeddings + XGBoost
python predictive_ultra.py data.csv text label --features roberta

# Full pipeline
python predictive_ultra.py data.csv text label --all

# Compare all combinations
python predictive_ultra.py data.csv text label --compare
```

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌────────────────┐
│  CSV/TSV    │───→│  Preflight Gate  │───→│  Features   │───→│  Model Factory │
│  Loader     │    │  (labels, splits,│    │  TF-IDF     │    │  XGBoost       │
│             │    │   class balance, │    │  SBERT      │    │  LogReg        │
│             │    │   backends)      │    │  DistilBERT │    │  SVM           │
│             │    │                  │    │  RoBERTa    │    │  RF            │
│             │    │                  │    │  MPNet      │    │  CNB           │
│             │    │                  │    │  DeBERTa    │    │  SGD           │
│             │    │                  │    │  + hybrids  │    │  Ensemble      │
│             │    │                  │    │             │    │  DistilBERT FT │
└─────────────┘    └──────────────────┘    └─────────────┘    └──────┬─────────┘
                                                                     │
                                                          ┌──────────▼──────────┐
                                                          │  Training Pipeline  │
                                                          │  ✓ Optuna tuning    │
                                                          │  ✓ SMOTE resampling │
                                                          │  ✓ Early stopping   │
                                                          │  ✓ Calibration      │
                                                          └──────────┬──────────┘
                                                                     │
                                         ┌───────────────────────────┼───────────┐
                                         │                           │           │
                                   ┌─────▼─────┐            ┌────────▼──────┐    │
                                   │  Eval     │            │  Explain      │    │
                                   │  Metrics  │            │  SHAP         │    │
                                   │  CV       │            │  Permutation  │    │
                                   │  Bootstrap │            │  Feature      │    │
                                   │  CI       │            │  Families     │    │
                                   └─────┬─────┘            └────────┬──────┘    │
                                         │                           │           │
                                   ┌─────▼───────────────────────────▼──────┐    │
                                   │          Output Artifacts              │    │
                                   │  model.joblib  │  report.html          │    │
                                   │  predictions   │  confusion_matrix.png  │    │
                                   │  calibration   │  feature_importance   │    │
                                   │  threshold     │  shap_summary.png     │    │
                                   │  cv_results    │  comparison.csv       │    │
                                   └────────────────────────────────────────┘    │
                                         │                                       │
                                   ┌─────▼──────────────────────────────────────▼─┐
                                   │  Predict on New Data (--test)               │
                                   └────────────────────────────────────────────┘
```

## Benchmark Results

Tested on **TripAdvisor Hong Kong Forum sample** (542 reviews, binary chainid classification). All runs used 80/20 stratified split, XGBoost defaults (max_depth=6, learning_rate=0.1, colsample_bytree=0.6, early stopping 50 rounds), no calibration.

| Rank | Model | Macro F1 | Accuracy | ROC-AUC | CV F1 | Δ (Test−CV) |
|------|-------|----------|----------|---------|-------|-------------|
| 🥇 | XGBoost + TF-IDF + MPNet | **0.9115** | 0.9266 | 0.9624 | 0.8773 | +0.0342 ✓ |
| 🥈 | XGBoost + MPNet | 0.8794 | 0.8991 | 0.9486 | 0.8952 | −0.0158 ✓ |
| 🥉 | XGBoost + RoBERTa | 0.8504 | 0.8716 | 0.9529 | 0.8416 | +0.0088 ✓ |
| 4 | XGBoost + TF-IDF + DistilBERT | 0.8479 | 0.8716 | 0.9314 | 0.8439 | +0.0040 ✓ |
| 5 | XGBoost + DistilBERT | 0.8197 | 0.8532 | 0.9247 | 0.8375 | −0.0178 ✓ |
| 6 | XGBoost + SBERT | 0.7827 | 0.8165 | 0.8957 | 0.8444 | −0.0617 ⚠ |
| 7 | XGBoost + TF-IDF | 0.7656 | 0.8073 | 0.8751 | 0.7525 | +0.0131 ✓ |

**Key insights:**
- Hybrid (TF-IDF + embeddings) beats either alone — TF-IDF captures keyword patterns, MPNet captures semantics, XGBoost learns the optimal blend
- Low test-CV deltas rule out overfitting (all under 0.04 except SBERT)
- Pure TF-IDF baseline is surprisingly strong for keyword-driven tasks
- DeBERTa underperformed (not shown) — likely needs domain-specific fine-tuning

## Outputs

| File | Description |
|------|-------------|
| `model.joblib` | Trained classifier bundled with experiment metadata |
| `model_estimator.joblib` | Bare estimator (for `predict_on_new` legacy loaders) |
| `vectorizer.joblib` | TF-IDF vectorizer |
| `label_encoder.joblib` | Label encoder |
| `scaler.joblib` | Feature scaler (embeddings mode) |
| `feature_selector.joblib` | Chi2 / mutual information selector |
| `predictions.csv` | Test predictions with per-class probabilities |
| `error_analysis.csv` | Misclassified test docs with model confidence |
| `experiment.json` | Reproducibility record: model, metrics, feature shape, timestamp |
| `report.html` | Self-contained HTML report with all metrics + plots |
| `confusion_matrix.png` | Normalized confusion matrix |
| `calibration_curve.png` | Reliability diagram with ECE |
| `threshold_curve.png` | F1 / precision / recall vs threshold |
| `feature_importance.png` | SHAP / permutation importance bar chart |
| `shap_summary.png` | SHAP beeswarm summary (top 20 features) |
| `cv_results.png` | Cross-validation fold scores |
| `comparison.csv` | (Compare mode) Full leaderboard |
| `predictions_new.csv` | (Test mode) Predictions on new data |

## Tips & Best Practices

### Which feature mode should I use?

| Scenario | Recommended | Why |
|----------|-------------|-----|
| First run, quick baseline | `--features tfidf` | Fast, interpretable, surprisingly strong |
| Semantic understanding matters | `--features tfidf+mpnet` | Best accuracy, MF1=0.91 on benchmarks |
| No GPU, CPU-only production | `--features tfidf` or `tfidf+sbert` | No PyTorch dependency |
| Small dataset (<500 docs) | `--features tfidf` | Embeddings add noise with little data |
| Large dataset (>10k docs) | `--features tfidf+mpnet` or `distilbert` | Embeddings scale, fine-tuning viable |
| Social media / short text | `--features tfidf+distilbert` | DistilBERT trained on noisy text |
| Interpretability priority | `--features tfidf` | TF-IDF features map to words directly |

### How to check for overfitting

Use `--compare` mode — it reports the test-CV delta for every combination:

```
Overfitting Check:
  Model                          Test F1    CV F1      Delta      Verdict
  ────────────────────────────── ────────── ────────── ────────── ────────────
  XGBoost+TF-IDF+MPNet           0.9115     0.8773     +0.0342    OK
  XGBoost+MPNet                  0.8794     0.8952     -0.0158    OK
```

- **Delta < 0.05** → OK (model generalizes)
- **Delta > 0.05** → Could be overfit or data distribution shift
- **Negative delta** → CV harder than test split (conservative estimate)

### Speed / Accuracy trade-offs

- **TF-IDF only**: trains in 0.1s for 542 docs
- **TF-IDF + MPNet**: trains in ~105s (embedding extraction) + 0.1s (XGBoost)
- **DistilBERT fine-tune**: ~12 min on CPU for 542 docs

Choose based on your throughput requirements and accuracy threshold.

### Large corpora on CPU (no GPU)

If you're working with 100K+ documents and no CUDA GPU:

| Component | CPU cost | Mitigation |
|-----------|----------|------------|
| `pd.read_csv(1GB)` | ~30s, ~4GB RAM | Use `chunksize=` if RAM-constrained |
| TF-IDF fit_transform | Linear in corpus size (~1 min for 1M docs) | `--max-features 20000` caps memory |
| DistilBERT/MPNet/RoBERTa embeddings | **~50 CPU-hours per million docs** | Use `--features tfidf` (no transformer) or pre-compute and cache |
| `all-MiniLM-L6-v2` (default SBERT) | 5× faster than MPNet | Already the SBERT default |
| XGBoost training | ~10–30 min for 1M × 20K sparse | `n_jobs=-1` uses all cores |
| SMOTE/ADASYN resampling | OOMs above ~100K rows | Disabled by default — keep it off for big data |

**Practical recipe for 1M+ docs, no GPU:**
1. First pass: `--features tfidf --max-features 10000` (fast baseline, ~5 min)
2. If accuracy insufficient, switch to `--features sbert` (MiniLM, ~5 CPU-hours per million docs)
3. Skip transformer features (`mpnet`, `roberta`, `deberta`) — they're a 3–5× premium over MiniLM with marginal accuracy gain for short-text classification
4. Use `--cache-dir ./emb_cache` so re-runs don't recompute embeddings

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: xgboost` | XGBoost not installed | `pip install xgboost` |
| `CUDA error: device not found` | GPU not detected | Tool falls back to CPU automatically |
| `OutOfMemoryError` during embeddings | Batch too large for GPU | Script uses batch_size=32; reduce via CPU mode |
| SHAP plot shows no features | Calibrated model wraps XGBoost | Script extracts inner model automatically |
| `--compare` shows "FAIL" for DeBERTa | DeBERTa v3 incompatible with older transformers | `pip install --upgrade transformers` |
| DistilBERT fine-tuning takes hours | CPU-only, large dataset | Use `--features distilbert` instead (embeddings only) |
| Temporal split errors | Date column has NaT values | Clean date column, ensure `pd.to_datetime()` succeeds |

## Dependencies

### Core (always required)
```
scikit-learn>=1.6
xgboost>=2.0
pandas>=2.0
numpy>=1.24
shap>=0.45
scipy>=1.10
joblib>=1.3
imbalanced-learn>=0.12
matplotlib>=3.7
```

### Optional
```
optuna>=3.0          ← hyperparameter tuning
cleanlab>=2.0        ← label quality audit
sentence-transformers>=2.2  ← SBERT embeddings
transformers[torch]>=4.30   ← DistilBERT/RoBERTa/MPNet/DeBERTa
torch>=2.0                     ← PyTorch for transformer models
plotly>=5.15                  ← interactive plots (not used in default pipeline)
```

## Related Projects

- [BERTopic](https://github.com/MaartenGr/BERTopic) — Topic modeling with transformer embeddings
- [SetFit](https://github.com/huggingface/setfit) — Few-shot text classification
- [FastText](https://github.com/facebookresearch/fastText) — Lightweight text classification
- [scikit-learn](https://scikit-learn.org/) — ML toolkit this project builds on
- [SHAP](https://github.com/shap/shap) — Model explainability

## Citation

If you use this in research, please cite:

```bibtex
@software{pang2026predictiveultra,
  author = {Peter Pang},
  title = {Predictive Modeling ULTRA: Standalone Text Classification Toolkit},
  year = {2026},
  url = {https://github.com/chessy795/predictive-ultra}
}
```

## License

MIT — see [LICENSE](LICENSE).
