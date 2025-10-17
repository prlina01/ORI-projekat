# Predictive Pricing for Used Cars in Serbia

This project trains two models—Gradient Boosting and a TabTransformer—to predict used car prices in Serbia. Both models share an enhanced preprocessing pipeline featuring:

- Isolation Forest filtering with LOESS-based soft clipping for robust outlier handling.
- Feature engineering with vehicle age (`Starost`) and a custom depreciation index.
- Robust scaling and appropriate categorical encodings for each model.
- Spatial-temporal cross-validation that rotates geographic regions and recent vehicles across folds.

## Requirements

Install dependencies with [uv](https://github.com/astral-sh/uv) (recommended) or any Python environment manager:

```bash
uv pip install -r requirements.txt
```

## Usage

Run the training and evaluation script via `uv run` to isolate dependencies and execute the models end-to-end:

```bash
uv run python main.py kola_skup_podataka.tsv
```

The script prints the cross-validated RMSE for both Gradient Boosting and the TabTransformer.

## Outputs

- `GradientBoostingRegressor RMSE`: Cross-validated RMSE (lower is better).
- `TabTransformer RMSE`: Cross-validated RMSE from the transformer-based model.

These metrics help compare traditional ensemble methods with transformer architectures on structured automotive data.
