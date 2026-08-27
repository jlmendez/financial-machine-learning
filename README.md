# Financial Machine Learning

Applied machine-learning workflows for synthetic credit-risk scenarios spanning supervised modeling, ensemble methods, clustering, and anomaly detection.

## Highlights

- Reproducible synthetic financial data
- Logistic Regression and Random Forest comparison
- ROC-AUC and classification diagnostics
- K-Means and DBSCAN segmentation
- Isolation Forest anomaly detection
- Clear separation between supervised and unsupervised workflows

## Tech stack

Python · pandas · NumPy · scikit-learn

## Repository structure

- `src/financial_ml_demo.py` — reproducible supervised and unsupervised ML workflow
- `requirements.txt` — Python dependencies
- `.gitignore` — excludes environments and generated artifacts

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/financial_ml_demo.py
```

The project uses synthetic data, so no private financial dataset is required.

## Portfolio context

This repository condenses a broader set of experiments with classical models, ensembles, clustering, dimensionality reduction, and anomaly detection into a compact reproducible Python project.
