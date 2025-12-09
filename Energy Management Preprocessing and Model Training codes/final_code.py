# run_models_final_benchmarked.py
# Extended from your original script to measure time, memory, inference latency,
# and estimate minimum datapoints (learning-curve heuristic). Saves CSV outputs.

import sys
import os
import time
import json
import tracemalloc
import math
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.utils.multiclass import type_of_target
from sklearn.impute import SimpleImputer

# Classical ML - Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

# Classical ML - Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Optional: XGBoost & LightGBM (auto-skip if not installed)
HAVE_XGB, HAVE_LGBM = True, True
try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception:
    HAVE_XGB = False
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    HAVE_LGBM = False

# psutil optional for RSS memory
try:
    import psutil
except Exception:
    psutil = None

RANDOM_STATE = 42
DATA_FILE = "iiot_smart_grid_dataset.csv"
OUT_DIR = "model_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Utilities ----------------

def ohe_compat():
    """Return OneHotEncoder compatible with old/new sklearn (sparse vs sparse_output)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def split_cols(X):
    """Robust numeric/categorical split."""
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(include=['object', 'category']).columns.tolist()
    # Optionally treat low-cardinality numeric as categorical (uncomment if needed)
    # for c in num.copy():
    #     if X[c].nunique() <= 10:
    #         num.remove(c); cat.append(c)
    return num, cat

def reg_preprocessor(num_cols, cat_cols, scale_num=True):
    num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler() if scale_num else 'passthrough')])
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', ohe_compat())])
    return ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')

def cls_preprocessor(num_cols, cat_cols, scale_num=False):
    num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler() if scale_num else 'passthrough')])
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', ohe_compat())])
    return ColumnTransformer(transformers=[('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], remainder='drop')

def validate_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def print_header(title):
    line = "=" * len(title)
    print(f"\n{title}\n{line}")

def to_pretty(df):
    pd.set_option('display.max_colwidth', 120)
    pd.set_option('display.width', 140)
    return df

def human_size(x):
    if x is None:
        return "0B"
    x = float(x)
    for unit in ['B','KB','MB','GB']:
        if x < 1024.0:
            return f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{x:.1f}TB"

# ---------------- Dataset loading ----------------

def load_dataset():
    df = pd.read_csv(DATA_FILE)
    if "Timestamp" not in df.columns:
        raise ValueError("Expected a 'Timestamp' column in the dataset.")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    # Time features (leak-safe)
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Month"] = df["Timestamp"].dt.month
    validate_columns(df, ["Power_Consumption_kWh", "Energy_Efficiency_Score", "User_Type"])
    return df

# ---------------- Benchmark helpers ----------------

def measure_fit(gs, X, y):
    """Fits GridSearchCV object while measuring time + tracemalloc + RSS (if available)."""
    proc = psutil.Process() if psutil is not None else None
    rss_before = proc.memory_info().rss if proc else 0
    tracemalloc.start()
    t0 = time.time()
    gs.fit(X, y)
    fit_time = time.time() - t0
    current, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = proc.memory_info().rss if proc else 0
    rss_peak = max(rss_before, rss_after) if proc else 0
    return fit_time, int(tracemalloc_peak), int(rss_peak)

def measure_inference_latency(estimator, X_sample, n_repeat=30):
    """Return average seconds per sample for estimator.predict(X_sample)."""
    # warm up
    try:
        _ = estimator.predict(X_sample)
    except Exception:
        # safety: if estimator requires different input, return large latency
        return float('inf')
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = estimator.predict(X_sample)
        times.append(time.perf_counter() - t0)
    avg_batch = sum(times) / len(times)
    return avg_batch / max(1, X_sample.shape[0])

def estimate_min_datapoints(best_estimator, X, y, task_type='clf', sizes=[0.01,0.02,0.05,0.1,0.2,0.4,0.8,1.0]):
    """Subsampling heuristic: find smallest k reaching 98% of max perf (higher is better).
       For regression we use negative RMSE as metric (so higher -> better)."""
    n = X.shape[0]
    perf = []
    for frac in sizes:
        k = max(100, int(n * frac))
        Xs = X.sample(k, random_state=RANDOM_STATE)
        ys = y.loc[Xs.index]
        Xt, Xe, yt, ye = train_test_split(Xs, ys, test_size=0.3, random_state=RANDOM_STATE, stratify=ys if task_type=='clf' else None)
        est = best_estimator
        # train a fresh clone to avoid contamination
        try:
            est.fit(Xt, yt)
            yhat = est.predict(Xe)
            if task_type == 'clf':
                score = f1_score(ye, yhat, average='macro')
            else:
                score = -rmse(ye, yhat)
            perf.append((k, float(score)))
        except Exception:
            perf.append((k, float('-inf')))
    if not perf:
        return [], n
    max_perf = max(p for _, p in perf)
    threshold = 0.98 * max_perf
    min_k = next((k for k,p in perf if p >= threshold), perf[-1][0])
    return perf, int(min_k)

# ---------------- CV runners (patched) ----------------

def run_regression_search(name, pipe, grid, X, y, cv):
    # use built-in neg_root_mean_squared_error so sign handling is clear
    gs = GridSearchCV(pipe, grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, refit=True)
    fit_time, tracemalloc_peak, rss_peak = measure_fit(gs, X, y)
    rmse_cv = -gs.best_score_
    maes, r2s = [], []
    for tr, te in cv.split(X):
        yte = y.iloc[te]
        yhat = gs.best_estimator_.predict(X.iloc[te])
        maes.append(mean_absolute_error(yte, yhat))
        r2s.append(r2_score(yte, yhat))
    # inference latency
    n_sample = min(256, max(1, X.shape[0]))
    X_sample = X.sample(n_sample, random_state=RANDOM_STATE)
    latency = measure_inference_latency(gs.best_estimator_, X_sample, n_repeat=30)
    return {
        "Model": name,
        "Best_Params": gs.best_params_,
        "RMSE_CV": float(rmse_cv),
        "MAE_CV": float(np.mean(maes)),
        "R2_CV": float(np.mean(r2s)),
        "Train_Time_s": float(fit_time),
        "Tracemalloc_Peak_Bytes": int(tracemalloc_peak),
        "RSS_Peak_Bytes": int(rss_peak),
        "Inference_s_per_sample": float(latency)
    }

def run_classification_search(name, pipe, grid, X, y, cv):
    if type_of_target(y) not in ("binary", "multiclass"):
        raise ValueError("User_Type must be a categorical label (binary/multiclass).")
    gs = GridSearchCV(pipe, grid, scoring='f1_macro', cv=cv, n_jobs=-1, refit=True)
    fit_time, tracemalloc_peak, rss_peak = measure_fit(gs, X, y)
    accs, f1s = [], []
    for tr, te in cv.split(X, y):
        yte = y.iloc[te]
        yhat = gs.best_estimator_.predict(X.iloc[te])
        accs.append(accuracy_score(yte, yhat))
        f1s.append(f1_score(yte, yhat, average="macro"))
    # inference latency
    n_sample = min(256, max(1, X.shape[0]))
    X_sample = X.sample(n_sample, random_state=RANDOM_STATE)
    latency = measure_inference_latency(gs.best_estimator_, X_sample, n_repeat=30)
    return {
        "Model": name,
        "Best_Params": gs.best_params_,
        "MacroF1_CV": float(np.mean(f1s)),
        "Accuracy_CV": float(np.mean(accs)),
        "Train_Time_s": float(fit_time),
        "Tracemalloc_Peak_Bytes": int(tracemalloc_peak),
        "RSS_Peak_Bytes": int(rss_peak),
        "Inference_s_per_sample": float(latency)
    }

# ---------------- Main ----------------

def main():
    t0 = time.time()
    df = load_dataset()
    print_header("DATA QUICK CHECK")
    print("Rows,Cols:", df.shape)
    print("User_Type counts:\n", df["User_Type"].value_counts())
    print("Power summary:\n", df["Power_Consumption_kWh"].describe())
    print("Efficiency summary:\n", df["Energy_Efficiency_Score"].describe())
    print_header("BEGIN MODELLING & BENCHMARKS")

    # Leak-safe splits
    drop_norm = "Normalized_Consumption" in df.columns
    drop_list = ["Normalized_Consumption"] if drop_norm else []
    X_power = df.drop(columns=["Power_Consumption_kWh", "Energy_Efficiency_Score"] + drop_list).drop(columns=["Timestamp"], errors='ignore')
    y_power = df["Power_Consumption_kWh"].copy()

    X_eff = df.drop(columns=["Energy_Efficiency_Score", "Power_Consumption_kWh"] + drop_list).drop(columns=["Timestamp"], errors='ignore')
    y_eff = df["Energy_Efficiency_Score"].copy()

    X_user = df.drop(columns=["User_Type"]).drop(columns=["Timestamp"], errors='ignore')
    y_user = df["User_Type"].copy()

    num_p, cat_p = split_cols(X_power)
    num_e, cat_e = split_cols(X_eff)
    num_u, cat_u = split_cols(X_user)

    # CV scheme
    cv_reg = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---------------- POWER (Regression) ----------------
    prep_p_tree = reg_preprocessor(num_p, cat_p, scale_num=False)
    prep_p_lin  = reg_preprocessor(num_p, cat_p, scale_num=True)

    power_models = {}
    power_grids  = {}

    power_models["LinearRegression"] = Pipeline([("prep", prep_p_lin),  ("m", LinearRegression())])
    power_grids["LinearRegression"]  = {}

    power_models["Ridge"] = Pipeline([("prep", prep_p_lin), ("m", Ridge())])
    power_grids["Ridge"]  = {"m__alpha":[0.1, 1.0, 10.0]}

    power_models["Lasso"] = Pipeline([("prep", prep_p_lin), ("m", Lasso(max_iter=10000))])
    power_grids["Lasso"]  = {"m__alpha":[0.0005, 0.001, 0.01]}

    power_models["RandomForest"] = Pipeline([("prep", prep_p_tree), ("m", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    power_grids["RandomForest"]  = {"m__n_estimators":[200, 400], "m__max_depth":[None, 12, 20]}

    power_models["GradientBoosting"] = Pipeline([("prep", prep_p_tree), ("m", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    power_grids["GradientBoosting"]  = {"m__n_estimators":[200, 400], "m__learning_rate":[0.05, 0.1], "m__max_depth":[2, 3]}

    power_models["ExtraTrees"] = Pipeline([("prep", prep_p_tree), ("m", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    power_grids["ExtraTrees"]  = {"m__n_estimators":[400, 600], "m__max_depth":[None, 12, 20]}

    power_models["SVR_RBF"] = Pipeline([("prep", prep_p_lin), ("m", SVR(kernel="rbf"))])
    power_grids["SVR_RBF"]  = {"m__C":[0.5, 1.0, 2.0], "m__epsilon":[0.05, 0.1]}

    if HAVE_XGB:
        power_models["XGBRegressor"] = Pipeline([("prep", prep_p_tree), ("m", XGBRegressor(random_state=RANDOM_STATE, n_estimators=500, subsample=0.8, colsample_bytree=0.8, tree_method="hist"))])
        power_grids["XGBRegressor"]  = {"m__max_depth":[3, 6], "m__learning_rate":[0.05, 0.1]}

    if HAVE_LGBM:
        power_models["LGBMRegressor"] = Pipeline([("prep", prep_p_tree), ("m", LGBMRegressor(random_state=RANDOM_STATE, n_estimators=800))])
        power_grids["LGBMRegressor"]  = {"m__num_leaves":[31, 63], "m__learning_rate":[0.05, 0.1], "m__max_depth":[-1, 12]}

    power_rows = []
    print_header("Running Power Regression models")
    for n in power_models:
        try:
            rec = run_regression_search(n, power_models[n], power_grids.get(n, {}), X_power, y_power, cv_reg)
            power_rows.append(rec)
            print("Done:", n, "RMSE:", rec["RMSE_CV"], "R2:", rec["R2_CV"], "Time(s):", rec["Train_Time_s"])
        except Exception as e:
            print("Model failed:", n, e)

    df_power = pd.DataFrame(power_rows).sort_values("RMSE_CV", ascending=True).reset_index(drop=True)
    df_power.to_csv(os.path.join(OUT_DIR, "power_benchmarks.csv"), index=False)

    # ---------------- EFFICIENCY (Regression) ----------------
    prep_e_tree = reg_preprocessor(num_e, cat_e, scale_num=False)
    prep_e_lin  = reg_preprocessor(num_e, cat_e, scale_num=True)

    eff_models = {}
    eff_grids  = {}

    eff_models["LinearRegression"] = Pipeline([("prep", prep_e_lin), ("m", LinearRegression())])
    eff_grids["LinearRegression"]  = {}

    eff_models["Ridge"] = Pipeline([("prep", prep_e_lin), ("m", Ridge())])
    eff_grids["Ridge"]  = {"m__alpha":[0.1, 1.0, 10.0]}

    eff_models["Lasso"] = Pipeline([("prep", prep_e_lin), ("m", Lasso(max_iter=10000))])
    eff_grids["Lasso"]  = {"m__alpha":[0.0005, 0.001, 0.01]}

    eff_models["RandomForest"] = Pipeline([("prep", prep_e_tree), ("m", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    eff_grids["RandomForest"]  = {"m__n_estimators":[200, 400], "m__max_depth":[None, 12, 20]}

    eff_models["GradientBoosting"] = Pipeline([("prep", prep_e_tree), ("m", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    eff_grids["GradientBoosting"]  = {"m__n_estimators":[200, 400], "m__learning_rate":[0.05, 0.1], "m__max_depth":[2, 3]}

    eff_models["ExtraTrees"] = Pipeline([("prep", prep_e_tree), ("m", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
    eff_grids["ExtraTrees"]  = {"m__n_estimators":[400, 600], "m__max_depth":[None, 12, 20]}

    eff_models["SVR_RBF"] = Pipeline([("prep", prep_e_lin), ("m", SVR(kernel="rbf"))])
    eff_grids["SVR_RBF"]  = {"m__C":[0.5, 1.0, 2.0], "m__epsilon":[0.05, 0.1]}

    if HAVE_XGB:
        eff_models["XGBRegressor"] = Pipeline([("prep", prep_e_tree), ("m", XGBRegressor(random_state=RANDOM_STATE, n_estimators=500, subsample=0.8, colsample_bytree=0.8, tree_method="hist"))])
        eff_grids["XGBRegressor"]  = {"m__max_depth":[3, 6], "m__learning_rate":[0.05, 0.1]}

    if HAVE_LGBM:
        eff_models["LGBMRegressor"] = Pipeline([("prep", prep_e_tree), ("m", LGBMRegressor(random_state=RANDOM_STATE, n_estimators=800))])
        eff_grids["LGBMRegressor"]  = {"m__num_leaves":[31, 63], "m__learning_rate":[0.05, 0.1], "m__max_depth":[-1, 12]}

    eff_rows = []
    print_header("Running Efficiency Regression models")
    for n in eff_models:
        try:
            rec = run_regression_search(n, eff_models[n], eff_grids.get(n, {}), X_eff, y_eff, cv_reg)
            eff_rows.append(rec)
            print("Done:", n, "RMSE:", rec["RMSE_CV"], "R2:", rec["R2_CV"], "Time(s):", rec["Train_Time_s"])
        except Exception as e:
            print("Model failed:", n, e)

    df_eff = pd.DataFrame(eff_rows).sort_values("RMSE_CV", ascending=True).reset_index(drop=True)
    df_eff.to_csv(os.path.join(OUT_DIR, "eff_benchmarks.csv"), index=False)

    # ---------------- USER TYPE (Classification) ----------------
    prep_u_tree   = cls_preprocessor(num_u, cat_u, scale_num=False)
    prep_u_scaled = cls_preprocessor(num_u, cat_u, scale_num=True)

    user_models = {}
    user_grids  = {}

    user_models["LogisticRegression"] = Pipeline([("prep", prep_u_scaled), ("m", LogisticRegression(max_iter=1000, multi_class="multinomial", class_weight='balanced', solver='saga'))])
    user_grids["LogisticRegression"]  = {"m__C":[0.1, 1.0, 3.0]}

    user_models["KNN"] = Pipeline([("prep", prep_u_scaled), ("m", KNeighborsClassifier())])
    user_grids["KNN"]  = {"m__n_neighbors":[7, 11, 15]}

    user_models["DecisionTree"] = Pipeline([("prep", prep_u_tree), ("m", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    user_grids["DecisionTree"]  = {"m__max_depth":[6, 10, None], "m__min_samples_split":[2, 5]}

    user_models["RandomForest"] = Pipeline([("prep", prep_u_tree), ("m", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'))])
    user_grids["RandomForest"]  = {"m__n_estimators":[200, 400], "m__max_depth":[10, None], "m__min_samples_split":[2, 5]}

    user_models["GradientBoosting"] = Pipeline([("prep", prep_u_tree), ("m", GradientBoostingClassifier(random_state=RANDOM_STATE))])
    user_grids["GradientBoosting"]  = {"m__n_estimators":[200, 400], "m__learning_rate":[0.05, 0.1], "m__max_depth":[2, 3]}

    user_models["SVC_RBF"] = Pipeline([("prep", prep_u_scaled), ("m", SVC(kernel='rbf'))])
    user_grids["SVC_RBF"]  = {"m__C":[0.5, 1.0, 2.0], "m__gamma":["scale", 0.1]}

    if HAVE_XGB:
        user_models["XGBClassifier"] = Pipeline([("prep", prep_u_tree), ("m", XGBClassifier(random_state=RANDOM_STATE, n_estimators=500, subsample=0.8, colsample_bytree=0.8, tree_method="hist"))])
        user_grids["XGBClassifier"]  = {"m__max_depth":[3, 6], "m__learning_rate":[0.05, 0.1]}

    if HAVE_LGBM:
        user_models["LGBMClassifier"] = Pipeline([("prep", prep_u_tree), ("m", LGBMClassifier(random_state=RANDOM_STATE, n_estimators=800))])
        user_grids["LGBMClassifier"]  = {"m__num_leaves":[31, 63], "m__learning_rate":[0.05, 0.1], "m__max_depth":[-1, 12]}

    user_rows = []
    print_header("Running User Classification models")
    for n in user_models:
        try:
            rec = run_classification_search(n, user_models[n], user_grids.get(n, {}), X_user, y_user, cv_cls)
            user_rows.append(rec)
            print("Done:", n, "MacroF1:", rec.get("MacroF1_CV"), "Acc:", rec.get("Accuracy_CV"), "Time(s):", rec.get("Train_Time_s"))
        except Exception as e:
            print("Model failed:", n, e)

    df_user = pd.DataFrame(user_rows).sort_values(["MacroF1_CV","Accuracy_CV"], ascending=False).reset_index(drop=True)
    df_user.to_csv(os.path.join(OUT_DIR, "user_benchmarks.csv"), index=False)

    # ---------------- Save results & estimate min datapoints ----------------
    # Choose best properly
    best_power = df_power.loc[df_power['RMSE_CV'].idxmin()]
    best_eff   = df_eff.loc[df_eff['RMSE_CV'].idxmin()]
    best_user  = df_user.loc[df_user['MacroF1_CV'].idxmax()]

    # Retrain best estimators on full data and estimate min datapoints
    # Note: models dicts map names -> pipeline templates; clone by re-creating pipelines from those dicts
    best_power_name = best_power['Model']
    best_eff_name = best_eff['Model']
    best_user_name = best_user['Model']

    # retrain and save best models
    try:
        power_est = power_models[best_power_name]
        power_est.fit(X_power, y_power)
        pickle.dump(power_est, open(os.path.join(OUT_DIR, f"best_power_{best_power_name}.pkl"), "wb"))
    except Exception as e:
        print("Failed to retrain/save best power model:", e)
        power_est = None

    try:
        eff_est = eff_models[best_eff_name]
        eff_est.fit(X_eff, y_eff)
        pickle.dump(eff_est, open(os.path.join(OUT_DIR, f"best_eff_{best_eff_name}.pkl"), "wb"))
    except Exception as e:
        print("Failed to retrain/save best eff model:", e)
        eff_est = None

    try:
        user_est = user_models[best_user_name]
        user_est.fit(X_user, y_user)
        pickle.dump(user_est, open(os.path.join(OUT_DIR, f"best_user_{best_user_name}.pkl"), "wb"))
    except Exception as e:
        print("Failed to retrain/save best user model:", e)
        user_est = None

    # estimate min datapoints
    print_header("Estimating min datapoints (subsampling heuristic)")
    perf_p, minp = estimate_min_datapoints(power_est, X_power, y_power, task_type='reg')
    perf_e, mine = estimate_min_datapoints(eff_est, X_eff, y_eff, task_type='reg')
    perf_u, minu = estimate_min_datapoints(user_est, X_user, y_user, task_type='clf')

    # save curves
    pd.DataFrame(perf_p, columns=["n_samples","score"]).to_csv(os.path.join(OUT_DIR, "power_min_datapoints_curve.csv"), index=False)
    pd.DataFrame(perf_e, columns=["n_samples","score"]).to_csv(os.path.join(OUT_DIR, "eff_min_datapoints_curve.csv"), index=False)
    pd.DataFrame(perf_u, columns=["n_samples","score"]).to_csv(os.path.join(OUT_DIR, "user_min_datapoints_curve.csv"), index=False)

    with open(os.path.join(OUT_DIR, "min_datapoints_summary.txt"), "w") as f:
        f.write(f"Power best: {best_power_name} min_required={minp}\n")
        f.write(f"Eff best:   {best_eff_name} min_required={mine}\n")
        f.write(f"User best:  {best_user_name} min_required={minu}\n")

    # Save final summary winners (safe selection)
    winners = pd.DataFrame({
        "Task": ["Power Regression", "Efficiency Regression", "User Classification"],
        "Best Model": [best_power_name, best_eff_name, best_user_name],
        "Primary Score": [
            f"RMSE={best_power['RMSE_CV']:.4f}, R2={best_power['R2_CV']:.4f}",
            f"RMSE={best_eff['RMSE_CV']:.4f}, R2={best_eff['R2_CV']:.4f}",
            f"Macro-F1={best_user['MacroF1_CV']:.4f}, Acc={best_user['Accuracy_CV']:.4f}"
        ],
        "Best Params": [
            json.dumps(best_power["Best_Params"]),
            json.dumps(best_eff["Best_Params"]),
            json.dumps(best_user["Best_Params"])
        ]
    })
    winners.to_csv(os.path.join(OUT_DIR, "final_winners_summary.csv"), index=False)

    # print results to console
    print_header("RESULTS — Power Regression (5-fold CV, sorted by RMSE)")
    print(to_pretty(df_power.head(20)))
    print_header("RESULTS — Energy Efficiency Regression (5-fold CV, sorted by RMSE)")
    print(to_pretty(df_eff.head(20)))
    print_header("RESULTS — User Type Classification (5-fold CV, sorted by Macro-F1 then Accuracy)")
    print(to_pretty(df_user.head(20)))
    print_header("FINAL — Best Models Summary (saved to model_outputs/final_winners_summary.csv)")
    print(to_pretty(winners))

    # also save detailed per-task CSVs (already saved above), plus min datapoints summary
    print("\nSaved outputs to", OUT_DIR)
    print("power_benchmarks.csv, eff_benchmarks.csv, user_benchmarks.csv")
    print("power_min_datapoints_curve.csv, eff_min_datapoints_curve.csv, user_min_datapoints_curve.csv")
    print("min_datapoints_summary.txt, final_winners_summary.csv")
    dt = time.time() - t0
    print_header(f"Done in {dt:.1f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", e)
        sys.exit(1)
