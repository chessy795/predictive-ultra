# Predictive Modeling ULTRA

Standalone Python text classification pipeline with XGBoost, hybrid transformer embeddings, SHAP explainability, cross-validation, calibration, and model comparison.

## Features

- **7 classifier types**: XGBoost, Logistic Regression, SVM, Random Forest, ComplementNB, SGD, Ensemble (soft voting / stacking)
- **8 feature modes**: TF-IDF, SBERT, DistilBERT, RoBERTa, MPNet, DeBERTa, and TF-IDF + any embedding hybrid
- **DistilBERT fine-tuning** via HuggingFace Trainer API
- **SHAP explainability**: beeswarm, waterfall, force plots with adaptive top-K selection
- **Calibration**: Platt scaling via `CalibratedClassifierCV` with ECE reporting
- **Cross-validation**: Stratified K-Fold, Repeated K-Fold, Group K-Fold, temporal/blocked splits
- **Class imbalance**: SMOTE, ADASYN, random over/under sampling via `imbalanced-learn`
- **Hyperparameter tuning**: Optuna with configurable search space
- **Label quality audit**: Cleanlab integration for finding annotation errors
- **Robust bootstrap confidence intervals** (percentile method)
- **`--compare` mode**: benchmarks all model/feature combos in one shot
- **Self-contained HTML report** with all metrics, plots, and explanation
- **CPU-first**: no GPU required for most modes; GPU auto-detected for XGBoost

## Quick Start

```bash
pip install -r requirements.txt

# Basic XGBoost with TF-IDF
python predictive_ultra.py data.csv text_col label_col

# Best-performing hybrid: XGBoost + TF-IDF + MPNet embeddings
python predictive_ultra.py data.csv text_col label_col --features tfidf+mpnet

# Compare all model/feature combinations
python predictive_ultra.py data.csv text_col label_col --compare

# Full pipeline with everything
python predictive_ultra.py data.csv text_col label_col --all
```

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
| `--model` | Model type: `xgboost`, `logreg`, `svm`, `rf`, `nb`, `sgd`, `ensemble`, `distilbert` (default: xgboost) |
| `--features` | Feature mode: `tfidf`, `sbert`, `distilbert`, `roberta`, `mpnet`, `deberta`, `tfidf+sbert`, `tfidf+distilbert`, `tfidf+roberta`, `tfidf+mpnet`, `tfidf+deberta` (default: tfidf) |
| `--split` | Split mode: `random_stratified`, `temporal`, `blocked_temporal`, `group` (default: random_stratified) |
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

# TF-IDF + MPNet hybrid (best score on benchmarks)
python predictive_ultra.py data.csv text label --features tfidf+mpnet
```

## Benchmark Results

Tested on TripAdvisor Hong Kong Forum sample (542 reviews, binary chainid classification).

| Rank | Model | Macro F1 | Accuracy | ROC-AUC | CV F1 |
|------|-------|----------|----------|---------|-------|
| 1 | XGBoost + TF-IDF + MPNet | **0.9115** | 0.9266 | 0.9624 | 0.8773 |
| 2 | XGBoost + MPNet | 0.8794 | 0.8991 | 0.9486 | 0.8952 |
| 3 | XGBoost + RoBERTa | 0.8504 | 0.8716 | 0.9529 | 0.8416 |
| 4 | XGBoost + TF-IDF + DistilBERT | 0.8479 | 0.8716 | 0.9314 | 0.8439 |
| 5 | XGBoost + DistilBERT | 0.8197 | 0.8532 | 0.9247 | 0.8375 |
| 6 | XGBoost + SBERT | 0.7827 | 0.8165 | 0.8957 | 0.8444 |
| 7 | XGBoost + TF-IDF | 0.7656 | 0.8073 | 0.8751 | 0.7525 |

**Key insight**: XGBoost + TF-IDF + MPNet embeddings achieves competitive results with fine-tuned transformers at a fraction of the training time, using CPU inference.

## Outputs

| File | Description |
|------|-------------|
| `model.joblib` | Trained classifier |
| `vectorizer.joblib` | TF-IDF vectorizer |
| `label_encoder.joblib` | Label encoder |
| `predictions.csv` | Test predictions with probabilities |
| `report.html` | Self-contained HTML report |
| `confusion_matrix.png` | Normalized confusion matrix |
| `calibration_curve.png` | Reliability diagram |
| `threshold_curve.png` | F1 vs threshold curve |
| `feature_importance.png` | SHAP / permutation importance |
| `shap_summary.png` | SHAP beeswarm summary |
| `cv_results.png` | Cross-validation fold scores |
| `comparison.csv` | (Compare mode) Full leaderboard |

## Dependencies

Core: `scikit-learn`, `xgboost`, `pandas`, `numpy`, `imbalanced-learn`, `shap`

Optional: `optuna` (tuning), `cleanlab` (label audit), `sentence-transformers` (SBERT embeddings), `transformers[torch]` (DistilBERT/RoBERTa/MPNet), `plotly` (interactive plots), `matplotlib` (static plots)

## License

MIT
