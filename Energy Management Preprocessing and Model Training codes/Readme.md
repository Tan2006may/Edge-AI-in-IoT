# Edge AI in IoT – Smart Grid ML Benchmark Suite

This repository contains a single, self-contained benchmarking script for **Industrial IoT (IIoT) Smart Grid** data.  
It trains and evaluates multiple models for:

- **Power Consumption Regression** → `Power_Consumption_kWh`
- **Energy Efficiency Regression** → `Energy_Efficiency_Score`
- **User Type Classification** → `User_Type`

The script automatically measures:

- Cross-validated performance (RMSE / MAE / R² / Macro-F1 / Accuracy)
- Training time
- Peak memory usage (Python + optional RSS)
- Inference latency (seconds per sample)
- Approximate **minimum data points required** (learning-curve style heuristic)

All results and best models are saved under `model_outputs/`.

---

##  Files

- `run_models_final_benchmarked.py` – main script (this file)
- `iiot_smart_grid_dataset.csv` – input dataset (must be placed in the same directory)
- `model_outputs/` – auto-created folder containing benchmark results, curves and saved models

---

##  Expected Dataset Schema

The script expects a CSV file named:

```text
iiot_smart_grid_dataset.csv
````

with at least the following columns:

* `Timestamp` – time of measurement (string/datetime)
* `Power_Consumption_kWh` – **target** for power regression
* `Energy_Efficiency_Score` – **target** for efficiency regression
* `User_Type` – **classification label** (e.g. Residential / Commercial / Industrial)

All other columns are treated as **features**.
The script automatically:

* Parses `Timestamp` to datetime
* Derives time features:

  * `Hour`
  * `DayOfWeek`
  * `Month`

If a column `Normalized_Consumption` exists, it is **dropped** to avoid leakage.

---

##  Dependencies

Install Python packages (Python 3.8+ recommended):

```bash
pip install numpy pandas scikit-learn xgboost lightgbm psutil
```

> `xgboost`, `lightgbm`, and `psutil` are optional – the script will **auto-skip** XGBoost / LightGBM models or RSS tracking if not installed.

---

##  How to Run

From the folder containing `run_models_final_benchmarked.py` and `iiot_smart_grid_dataset.csv`:

```bash
python run_models_final_benchmarked.py
```

On success, you will see console logs like:

* Dataset size and quick summaries
* Per-model progress (name, RMSE / R² or Macro-F1 / Accuracy, training time)
* Final summary tables for each task
* Best model winners and data-point estimates

---

##  What the Script Does (High-Level Flow)

1. **Load & preprocess dataset**

   * Reads `iiot_smart_grid_dataset.csv`
   * Parses `Timestamp`
   * Adds `Hour`, `DayOfWeek`, `Month`
   * Creates three supervised learning tasks:

     * **Power Regression**: predict `Power_Consumption_kWh`
     * **Efficiency Regression**: predict `Energy_Efficiency_Score`
     * **User Classification**: predict `User_Type`

2. **Feature handling**

   * Splits features into **numeric** and **categorical**:

     * numeric → `np.number`
     * categorical → `object` / `category`
   * Builds task-specific preprocessing pipelines:

     * `SimpleImputer(median)` for numeric
     * `SimpleImputer(constant='missing')` + `OneHotEncoder` for categorical
     * Optional `StandardScaler` for models that need scaling

3. **Model families**

   ### Regression (Power & Efficiency)

   * `LinearRegression`
   * `Ridge`
   * `Lasso`
   * `RandomForestRegressor`
   * `GradientBoostingRegressor`
   * `ExtraTreesRegressor`
   * `SVR` (RBF kernel)
   * `XGBRegressor` (if available)
   * `LGBMRegressor` (if available)

   ### Classification (User_Type)

   * `LogisticRegression` (multinomial, class_weight='balanced')
   * `KNeighborsClassifier`
   * `DecisionTreeClassifier`
   * `RandomForestClassifier` (class_weight='balanced')
   * `GradientBoostingClassifier`
   * `SVC` (RBF kernel)
   * `XGBClassifier` (if available)
   * `LGBMClassifier` (if available)

4. **Hyperparameter tuning**

   * Uses `GridSearchCV` with:

     * 5-fold `KFold` for regression
     * 5-fold `StratifiedKFold` for classification
   * Scoring:

     * Regression → `neg_root_mean_squared_error`
     * Classification → `f1_macro`

5. **Benchmark metrics**

   * For regression (per model):

     * RMSE (CV)
     * MAE (CV)
     * R² (CV)
   * For classification (per model):

     * Macro-F1 (CV)
     * Accuracy (CV)
   * Resource metrics:

     * Training time (seconds)
     * Peak Python heap from `tracemalloc`
     * Optional RSS peak (if `psutil` available)
     * Inference latency → average seconds per sample on a batch (≤256 samples)

6. **Minimum datapoint estimation**

   * For each best model (per task), performs a **subsampling heuristic**:

     * Trains on fractions of data: `[1%, 2%, 5%, 10%, 20%, 40%, 80%, 100%]`
     * Measures performance (Macro-F1 for classification, negative RMSE for regression)
     * Estimates **smallest `n_samples` achieving 98% of max performance**
   * Saves learning-curve-style CSVs for later plotting.

7. **Saving best models**

   * Re-fits the best model for each task on **all available data**
   * Saves them as:

     * `model_outputs/best_power_<ModelName>.pkl`
     * `model_outputs/best_eff_<ModelName>.pkl`
     * `model_outputs/best_user_<ModelName>.pkl`

---

##  Outputs (in `model_outputs/`)

The script automatically creates a folder:

```text
model_outputs/
```

and writes:

### Benchmark tables

* `power_benchmarks.csv`
  → Results for power regression models

* `eff_benchmarks.csv`
  → Results for efficiency regression models

* `user_benchmarks.csv`
  → Results for user classification models

Each row includes:

* Model name
* Best hyperparameters
* Performance metrics (RMSE / MAE / R² OR Macro-F1 / Accuracy)
* Training time
* Memory usage
* Inference latency

### Learning curves & min datapoints

* `power_min_datapoints_curve.csv`
* `eff_min_datapoints_curve.csv`
* `user_min_datapoints_curve.csv`

Each contains:

* `n_samples` – number of training samples used

* `score` – performance metric for that subset
  (Macro-F1 for classification, negative RMSE for regression)

* `min_datapoints_summary.txt`
  → Human-readable summary:

  ```text
  Power best: <ModelName> min_required=<N>
  Eff best:   <ModelName> min_required=<N>
  User best:  <ModelName> min_required=<N>
  ```

### Best-model summary

* `final_winners_summary.csv`

Contains for each task:

* Task name
* Best model
* Primary score summary (RMSE/R² or Macro-F1/Accuracy)
* Best hyperparameter JSON

### Saved models

* `best_power_<ModelName>.pkl`
* `best_eff_<ModelName>.pkl`
* `best_user_<ModelName>.pkl`

These can be loaded later with `pickle.load(...)` for deployment or further experiments.

---

##  Reproducibility

* All key randomness is controlled via `RANDOM_STATE = 42`.
* CV splits (`KFold`/`StratifiedKFold`) are seeded with this random state.
* Subsampling heuristic uses the same seed for comparability.

---

##  Typical Workflow

1. Place your dataset as `iiot_smart_grid_dataset.csv` in the project root.

2. Install dependencies.

3. Run:

   ```bash
   python run_models_final_benchmarked.py
   ```

4. Inspect:

   * Console output for quick comparisons
   * CSVs in `model_outputs/` for detailed analysis
   * `final_winners_summary.csv` to see which models to use going forward
   * `_min_datapoints_curve.csv` files to understand data requirements

---

##  Notes / Customization

* You can adjust:

  * Model lists and hyperparameter grids
  * CV schemes (e.g., more/less folds)
  * Metrics (e.g., switch to `r2` or `roc_auc` if needed)
* If your dataset uses different column names, update:

  * `DATA_FILE`
  * Column references in `load_dataset()` and the three target definitions.

---
