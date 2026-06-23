# Farmer Income Prediction

Predict an Indian farmer's **annual total income** from personal, location, and climate features.
Scored by **MAPE**; evaluated on a single **85/15** train/CV split.

## Result
| Model | CV MAPE | CV Accuracy | CV R² | CV Adj R² |
|---|---|---|---|---|
| Baseline (lean XGBoost) | 21.2% | 78.8% | — | — |
| **Final (Trylitics_final.ipynb)** | **18.5%** | **81.5%** | **0.94** | **0.94** |

## How it works (one-liner)
Model `log1p(agri income)` where `agri = total − Non_Agriculture_Income`, then reconstruct
`total = k · expm1(pred) + Non_Agriculture_Income`. The biggest lever is **keeping
`Non_Agriculture_Income` as a feature** (corr 0.935 with total) — it took CV MAPE 21.2% → 18.5%.
Model = **LightGBM + XGBoost + CatBoost** ensemble (MAE objective) on ~62 leakage-free features
(hierarchical district/village/mandi/state target encodings, year-trend slopes + latest-year levels,
land interactions; NaNs kept for the trees).

## Files
| File | Purpose |
|---|---|
| `Trylitics_final.ipynb` | **Final pipeline** — one purpose per cell. Run top-to-bottom. |
| `Model_pipeline.ipynb` | Lean 41-feature baseline (≈21.2%). |
| `Pipeline.md` | Technical end-to-end write-up. |
| `Project_Report.md` | Formal project report (objective → results). |
| `LTF Challenge data with dictionary.xlsx` | Data (`TrainData`, `TestData`, `Dictionary`). |

## Run
Open **`Trylitics_final.ipynb`** and **Run All** — it prints the train/CV **MAPE / Accuracy / R²** table.

```
Dependencies: numpy pandas scikit-learn lightgbm xgboost catboost openpyxl
```

## Honest note on the ceiling
~18.5% is close to the legitimate floor for this dataset. The model can *rank* high vs low earners
(classifier AUC ≈ 0.89), but **no feature predicts the income magnitude within the high-earner group**
(the data has no crop-type / acreage / yield / price columns). Going meaningfully lower would need such
external features, not more modelling on the current data.
