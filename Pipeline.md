# Farmer-Income Prediction — Pipeline (end to end)

Predict a farmer's **annual total income** from personal, location, and climate features.
Scored by **MAPE** (mean absolute percentage error). Evaluated on a single **85/15** train/CV split.

- **Baseline pipeline:** `Model_pipeline.ipynb` — lean 41-feature XGBoost. **CV MAPE ≈ 21.2%**.
- **Advanced pipeline:** `Model_advanced.ipynb` / `model_advanced.py` — the model documented here. **CV MAPE ≈ 18.5%**.

---

## 1. Problem & metric

- **Target column:** `Target_Variable/Total Income`.
- **Range:** ≈ ₹30,000 → ₹7.8 million, heavily right-skewed (a few very high earners).
- **Metric:** MAPE `= mean(|true − pred| / true) × 100`. It is a **relative** error, so it weights every
  farmer by `1/income` — low-income farmers dominate the score, high-income farmers barely count.

---

## 2. Data

- `LTF Challenge data with dictionary.xlsx` — `TrainData` (47,970 × 105), `TestData` (9,986 × 105, target blank), `Dictionary`.
- Mix of: personal (sex, marital status, ownership), location (state/district/village/city/mandi),
  socio-economic village indices, soil/agro-zone/water-body (per season-year), rainfall & temperature
  (per season-year), land areas, loan/bureau fields, and **`Non_Agriculture_Income`**.

---

## 3. Target formulation (income decomposition)

We do **not** model total income directly. We split it:

```
agri_income = Total income − Non_Agriculture_Income          # what the model predicts
target      = log1p(agri_income)            (clipped ≥ 0)     # log stabilises the heavy tail
prediction  = expm1(model_output)                            # back to rupees (agri income)
total_pred  = prediction + Non_Agriculture_Income            # reconstruct total
```

**Why this works:** `Non_Agriculture_Income` is a **known input** (present and fully populated in both
train and test). Adding it back exactly means the model only has to predict the *agricultural* part —
no model error is spent on the part we already know. `log1p`/`expm1` are an exact inverse pair (base *e*),
which compresses the right tail so the regressor isn't dominated by a few huge incomes.

> **Key insight (the single biggest lever):** besides using `Non_Agriculture_Income` for the
> decomposition, we **also keep it as a model feature**. It correlates **0.935** with total income, so it
> is hugely informative — yet the earlier baseline discarded it as a predictor. Re-introducing it is what
> moved CV MAPE **21.2% → 18.5%**; everything else combined adds ~0.1%.

---

## 4. Feature engineering

All per-row transforms are deterministic (no target) and live in `base_features()`.

| Group | What | Why |
|---|---|---|
| **Loan** | `Total_outstanding_loan = Avg_Disbursement × No_of_Active_Loan`; keep both raw cols too | proxy for creditworthiness / scale |
| **Water bodies** | `has_river`, `has_water` from the 2022 water-body text | cheap irrigation-access flags |
| **Temperature** | parse `"min/max"` strings → per-season **mean of the max** (`kharif_temp_max`, `rabi_temp_max`) | min/max are ~0.93 collinear; one number per season suffices |
| **Year series** | for each multi-year metric (cropping density, ag score, irrigated area, groundwater): **least-squares slope/year** *and* the **latest-year level (2022)** | slope = trend; level = current state (more predictive of current income) |
| **Rainfall** | `rabi_rain_slope` over 2020–22 | the only rainfall signal worth keeping |
| **Land** | `log_land`, `Total_Land_missing` flag | land is the strongest physical driver of agri income |
| **Categoricals (low-card)** | one-hot: `SEX`, `REGION`, `MARITAL_STATUS`, `R022 village agri category` | safe, small dimensionality |
| **Interactions** | `land × DISTRICT_te`, `land × VILLAGE_te` | scale × locality |
| **NaN handling** | **kept as NaN** (not filled with 0) | LightGBM/XGBoost/CatBoost split on missing natively; 0-fill injects fake values |

Dropped (raw text / redundant / geo handled elsewhere): soil type, agro-eco zone, raw water-body and
rainfall and temperature columns, total geographical area, socio-economic *category* strings, address
type, ownership, location, zipcode.

---

## 5. Leakage-free target encoding (`te_oof`)

High-cardinality location keys are replaced by their **smoothed mean agricultural income**:

- **DISTRICT_te** → smoothed toward the global mean.
- **VILLAGE_te** → smoothed toward **its own district mean** (hierarchical empirical-Bayes; villages are
  sparse, ~8 rows each, so they lean on their district).
- **MANDI_te**, **STATE_te** → smoothed toward the global mean.

Leakage discipline:
- **Train rows** get **out-of-fold** values: each row's encoding is computed from the *other* K folds, so
  the model never sees a row's own target baked into its features.
- **CV / test rows** get the **full-train** map (computed only from train labels).
- Smoothing `(mean·count + prior·m)/(count + m)` shrinks rare categories toward the prior to avoid
  overfitting tiny groups.

> Note: in the advanced model these granular encodings turned out to add ~0 beyond `DISTRICT_te` once
> `Non_Agriculture_Income` is a feature — they are kept because they are harmless and principled, but they
> are **not** the source of the gain.

---

## 6. Model

**Ensemble of three gradient-boosted trees, all with an MAE objective**, averaged in income space:

| Model | Objective | Notes |
|---|---|---|
| LightGBM | `regression_l1` (MAE) | 3000 trees, lr 0.015, 255 leaves |
| XGBoost | `reg:absoluteerror` (MAE) | 2500 trees, lr 0.015, depth 10, `tree_method=hist` |
| CatBoost | `MAE` | 2500 iters, depth 9, lr 0.02 |

**Why MAE (L1), not MSE?** MAPE rewards the **median**, not the mean. L1 in log-space seeks the conditional
median → MAPE-aligned. (An MSE/`squarederror` NN scored ~31% MAPE for this reason; switching it to L1 closed
most of the gap.) **Why trees?** On tabular data with dozens of mid-signal features and no spatial/sequential
structure, gradient-boosted trees beat neural nets (a fair L1 torch MLP reached ~23% vs XGBoost ~21%).

---

## 7. Bias correction `k`

The log→exp round-trip and right-skew make raw predictions slightly biased. After ensembling we apply a
**single multiplicative scalar** to the agricultural prediction:

```
total_pred = k · agri_pred + Non_Agriculture_Income
```

`k` is chosen on the **train** split to minimise total-income MAPE (swept over 0.80–1.20). In the advanced
model it lands ≈ 0.995 (near no-op) because the ensemble is already well-calibrated; it is kept for safety.

> Earlier experiments confirmed a *level-dependent* multiplier (per-bin / linear / isotonic, keyed on the
> prediction) does **not** help MAPE — the bias on the usable (predicted-income) axis is flat, so only a
> constant correction is justified. A *rising* multiplier improves the high tail but worsens overall MAPE.

---

## 8. Validation

- **Single 85/15 split** (`random_state=42`): 40,774 train / 7,196 CV.
- **Leakage control:** every target-dependent step (the four target encodings, the bias `k`) is fit on the
  85% train only; CV rows are transformed with train-derived maps. Per-row features use no target.
- **Reported metrics** are on **reconstructed total income** (`expm1(pred)·k + non_agri`) vs actual total —
  i.e. the real competition quantity, not the agri sub-problem.

---

## 9. Results

| Pipeline | train MAPE | **CV MAPE** | CV Accuracy (100−MAPE) |
|---|---|---|---|
| Baseline (`Non_Agriculture_Income` discarded) | 15.1% | 21.2% | 78.8% |
| **Advanced (this pipeline)** | 12.7% | **18.5%** | **81.5%** |

Verified leakage-free: `Non_Agriculture_Income` is a given input (not derived from the target) and is fully
populated in the test set, so the gain generalises.

---

## 10. What did **not** help (measured)

- Granular **VILLAGE/MANDI/STATE** target encodings — flat once `nai` is a feature.
- **Latest-year levels** vs slopes, **NaN-keep** vs 0-fill — each ~0.0–0.1%.
- **Direct total-income** modelling (instead of decompose) — slightly worse (18.8%).
- **Heavier tuning / bigger ensemble** — ~0.1%.
- **Sample weights ∝ 1/y**, **Pseudo-Huber**, **reduced regularization** — all *worse* on MAPE.
- **Calibration / second-stage residual models** (GAM+NN, per-bin/linear/curve multipliers) — at best ~0.3%,
  usually flat; they cannot recover signal the trees already extracted.

---

## 11. Limitations & honest notes

- **The high tail is feature-limited, not model-limited.** Residuals fan out positive as income grows — the
  model under-predicts genuine high earners. Only `Total_Land` separates them (AUC ≈ 0.71); every other
  feature is ≈ 0.50. No model class or calibration fixes this without **new features** (crop type, cultivated
  acreage, yield, prices).
- **Chasing the tail hurts MAPE.** Because MAPE weights high earners ~0, boosting them over-predicts the
  many typical farmers and raises the score. The MAPE-optimal behaviour *is* to under-predict the tail.
- **~18.5% appears to be the legitimate floor** on this feature set. Lower (e.g. 15%) was not reachable
  without leaking the target, which this pipeline deliberately avoids.

---

## 12. Files & how to run

| File | Purpose |
|---|---|
| `Model_advanced.ipynb` | This pipeline, one purpose per cell (run top→bottom). |
| `model_advanced.py` | Same pipeline as a script; prints train/CV MAPE and saves `residual_advanced.png`. |
| `Model_pipeline.ipynb` | The lean 41-feature baseline (≈21.2%). |
| `Pipeline.md` | This document. |

```bash
python model_advanced.py          # -> prints CV MAPE ≈ 18.5%, writes residual_advanced.png
# or open Model_advanced.ipynb and Run All
```

Dependencies: `numpy pandas scikit-learn lightgbm xgboost catboost matplotlib openpyxl` (+ `torch` only for
the optional neural-net comparison in `Model_pipeline.ipynb`).
