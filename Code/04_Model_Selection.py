# =============================================================================
# 04_Model_Selection.py — Nested CV + OOF Evaluation + Model Selection
# =============================================================================
# 功能：从头执行完整的嵌套交叉验证（5×5×50 Optuna），收集所有模型的OOF
#       预测，用于模型筛选，并保存所有模型的最佳参数供后续使用。
# 口径说明：model_selection_metrics.csv 中 Sensitivity/Specificity/Accuracy/
#       F1/PPV/NPV 等阈值化指标所用 Optimal_Cutoff 是在同一 OOF 集合上按
#       Youden 指数重算的（cutoff derived from the same OOF set），属于
#       apparent threshold performance，解释时需注意轻度乐观偏倚。
#
# 输入：Derivation cohort.csv（训练集，来自 feature_selection/）
# 输出（均在 E:\new\Model Selection）：
#   - oof_predictions_all_models.csv
#   - model_selection_metrics.csv
#   - roc_curves_all_models.png / .pdf / .tif
#   - best_params_all_models.pkl
#
# 全局开关：
#   OUTER_FOLDS = 5, INNER_TRIALS = 50, INNER_FOLDS = 5, BOOTSTRAP_ITER = 1000
# =============================================================================

import os
import warnings
import gc
import time
import pickle
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix)
from sklearn.exceptions import ConvergenceWarning

import optuna
from optuna.samplers import TPESampler

# --- Monkey-patch for LightGBM/sklearn 1.8+ compatibility ---
# sklearn 1.8 renamed `force_all_finite` to `ensure_all_finite` in check_X_y
# and check_array. LightGBM <= 4.5 still uses the old parameter name.
import sklearn.utils.validation as _skval
_skval_check_X_y_orig = _skval.check_X_y
_skval_check_array_orig = _skval.check_array
def _patched_check_X_y(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _skval_check_X_y_orig(*args, **kwargs)
def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return _skval_check_array_orig(*args, **kwargs)
_skval.check_X_y = _patched_check_X_y
_skval.check_array = _patched_check_array

# Optional imports
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not installed — skipping LightGBM")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: catboost not installed — skipping CatBoost")

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("INFO: XGBoost loaded successfully")
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGBoost")

# Try tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("WARNING: tqdm not installed — progress bar disabled")

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Global Configuration
# =============================================================================
OUTER_FOLDS = 5
INNER_TRIALS = 50  # 多目标优化需要更多 trials 以覆盖帕累托前沿
INNER_FOLDS = 5
BOOTSTRAP_ITER = 1000

RANDOM_STATE = 123
N_JOBS = max(1, (os.cpu_count() or 1) // 2)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "feature_selection")
RESULT_DIR = os.path.join(BASE_DIR, "Model Selection")
os.makedirs(RESULT_DIR, exist_ok=True)

# Image settings
IMG_DPI = 600
FONT_TITLE = 14
FONT_LABEL = 12
FONT_LEGEND = 10
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": FONT_LABEL,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_LABEL,
    "legend.fontsize": FONT_LEGEND,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": IMG_DPI,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    # SCI 出版要求：PDF 嵌入 TrueType（Type 42）字体，避免字体替换
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

print(f"=== Configuration ===")
print(f"  OUTER_FOLDS: {OUTER_FOLDS}, INNER_FOLDS: {INNER_FOLDS}, INNER_TRIALS: {INNER_TRIALS}")
print(f"  BOOTSTRAP_ITER: {BOOTSTRAP_ITER}")
print(f"  RANDOM_STATE: {RANDOM_STATE}")
print(f"  Data dir: {DATA_DIR}")
print(f"  Result dir: {RESULT_DIR}")

# =============================================================================
# Data Loading (Training Set Only — No External Validation)
# =============================================================================
print("\n=== Loading Data ===")
train_df = pd.read_csv(os.path.join(DATA_DIR, "Derivation cohort.csv"))
print(f"  Derivation cohort: {train_df.shape}")

feature_cols = [c for c in train_df.columns if c not in ("ID", "label")]
X_train_full = train_df[feature_cols].values.astype(np.float64)
y_train_full = train_df["label"].values.astype(np.int32)

print(f"  Features ({len(feature_cols)}): {feature_cols}")
print(f"  Pos/Neg: {np.sum(y_train_full == 1)}/{np.sum(y_train_full == 0)}")

# =============================================================================
# Optuna Search Spaces
# =============================================================================
def suggest_params(trial, model_name):
    if model_name == "L2_LR":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    elif model_name == "MLP":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(10,), (20,), (10, 5), (50,), (100,), (100, 50)]),
            "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.0001, 0.1, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": trial.suggest_categorical("solver", ["adam", "lbfgs"])
        }
    elif model_name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
            "p": trial.suggest_int("p", 1, 2)
        }
    elif model_name == "DT":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 20)
        }
    elif model_name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)
        }
    elif model_name == "LightGBM":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
        }
    elif model_name == "CatBoost":
        return {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
            "border_count": trial.suggest_int("border_count", 32, 255)
        }
    elif model_name == "SVM":
        # Use separate parameter names for string and float gamma to avoid
        # Optuna "Cannot set different distribution kind" error.
        # Only one of gamma_str / gamma_float is registered per trial.
        use_str = trial.suggest_categorical("gamma_use_str", [True, False])
        if use_str:
            gamma = trial.suggest_categorical("gamma_str", ["scale", "auto"])
        else:
            gamma = trial.suggest_float("gamma_float", 1e-4, 1e1, log=True)
        return {
            "C": trial.suggest_float("C", 1e-1, 1e3, log=True),
            "gamma": gamma,
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "degree": trial.suggest_int("degree", 2, 5),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])
        }
    elif model_name == "GaussianNB":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-9, 1e-3, log=True)}
    elif model_name == "XGBoost":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _get_fixed_params(model_name):
    if model_name == "L2_LR":
        return {"max_iter": 5000, "random_state": RANDOM_STATE}
    elif model_name == "MLP":
        return {
            "max_iter": 2000,
            "early_stopping": True,
            "validation_fraction": 0.2,
            "n_iter_no_change": 15,
            "random_state": RANDOM_STATE,
            "verbose": False
        }
    elif model_name == "KNN":
        return {"n_jobs": N_JOBS}
    elif model_name == "DT":
        return {"random_state": RANDOM_STATE}
    elif model_name == "RF":
        return {"random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    elif model_name == "LightGBM":
        # 早停通过 fit 的 callbacks=[lgb.early_stopping(50)] 施加（见三处调用点）
        return {"n_estimators": 1000, "verbose": -1, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    elif model_name == "CatBoost":
        return {"iterations": 1000, "early_stopping_rounds": 50, "verbose": 0,
                "random_seed": RANDOM_STATE, "thread_count": max(1, os.cpu_count() or 1),
                "train_dir": RESULT_DIR}
    elif model_name == "SVM":
        # Default class_weight='balanced' as fallback if Optuna does not search
        return {"probability": True, "random_state": RANDOM_STATE, "class_weight": "balanced"}
    elif model_name == "GaussianNB":
        return {}
    elif model_name == "XGBoost":
        return {"n_estimators": 1000, "early_stopping_rounds": 50, "eval_metric": "auc",
                "verbosity": 0, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    return {}


def build_model(model_name, params):
    fixed = _get_fixed_params(model_name)
    merged = {**fixed, **params}
    if model_name == "L2_LR":
        # penalty now comes from Optuna search; only set default if not provided
        if "penalty" not in merged:
            merged["penalty"] = "l2"
        return LogisticRegression(**merged)
    elif model_name == "MLP":
        return MLPClassifier(**merged)
    elif model_name == "KNN":
        return KNeighborsClassifier(**merged)
    elif model_name == "DT":
        return DecisionTreeClassifier(**merged)
    elif model_name == "RF":
        return RandomForestClassifier(**merged)
    elif model_name == "LightGBM":
        if not HAS_LGBM:
            raise RuntimeError("LightGBM not installed")
        return LGBMClassifier(**merged)
    elif model_name == "CatBoost":
        if not HAS_CATBOOST:
            raise RuntimeError("CatBoost not installed")
        return CatBoostClassifier(**merged)
    elif model_name == "SVM":
        return SVC(**merged)
    elif model_name == "GaussianNB":
        return GaussianNB(**merged)
    elif model_name == "XGBoost":
        if not HAS_XGB:
            raise RuntimeError("XGBoost not installed")
        return XGBClassifier(**merged)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Bootstrap AUC CI — BCa Method
# =============================================================================
def bootstrap_auc_ci(y_true, y_prob, n_iter=BOOTSTRAP_ITER, alpha=0.05, random_state=RANDOM_STATE):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)

    try:
        theta_hat = roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan, np.nan, np.nan

    if np.isnan(theta_hat):
        return np.nan, np.nan, np.nan

    # Jackknife
    jack_estimates = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            jack_estimates[i] = roc_auc_score(y_true[mask], y_prob[mask])
        except ValueError:
            jack_estimates[i] = theta_hat

    jack_mean = np.mean(jack_estimates)
    diff = jack_mean - jack_estimates
    sum_cubed = np.sum(diff ** 3)
    sum_sq = np.sum(diff ** 2)
    a = 0.0 if sum_sq == 0 else sum_cubed / (6.0 * (sum_sq ** 1.5))

    # Bootstrap
    rng = np.random.RandomState(random_state)
    boot_estimates = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_estimates[b] = roc_auc_score(y_true[idx], y_prob[idx])
        except ValueError:
            boot_estimates[b] = theta_hat

    # Bias correction
    prop_less = np.clip(np.mean(boot_estimates < theta_hat), 0.001, 0.999)
    z0 = norm.ppf(prop_less)

    z_alpha_lo = norm.ppf(alpha / 2.0)
    z_alpha_hi = norm.ppf(1.0 - alpha / 2.0)

    num_lo = z0 + (z0 + z_alpha_lo) / (1.0 - a * (z0 + z_alpha_lo))
    alpha1 = norm.cdf(num_lo)
    num_hi = z0 + (z0 + z_alpha_hi) / (1.0 - a * (z0 + z_alpha_hi))
    alpha2 = norm.cdf(num_hi)

    alpha1 = np.clip(alpha1, 0.001, 0.999)
    alpha2 = np.clip(alpha2, 0.001, 0.999)

    lower = np.percentile(boot_estimates, 100.0 * alpha1)
    upper = np.percentile(boot_estimates, 100.0 * alpha2)
    boot_mean = np.mean(boot_estimates)

    return boot_mean, lower, upper


# =============================================================================
# Nested CV for a single model — collects OOF predictions
# =============================================================================
def nested_cv_single_model(model_name, X, y, outer_folds, inner_folds, inner_trials):
    """Run nested CV for one model. Returns (best_params, outer_aucs, outer_brier,
    outer_cutoffs, oof_df). oof_df has columns: model, fold, y_true, y_pred_prob."""
    outer_aucs = []
    outer_brier = []
    outer_cutoffs = []
    best_params_overall = None
    best_auc_overall = -np.inf
    oof_records = []

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(outer_cv.split(X, y))  # Pre-compute all splits outside loop

    fold_range = range(1, outer_folds + 1)
    if HAS_TQDM:
        fold_range = tqdm(fold_range, desc=f"  [{model_name}]", leave=False)

    for fold_idx in fold_range:
        train_idx, val_idx = cv_splits[fold_idx - 1]

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        X_tr_df = pd.DataFrame(X_tr)
        X_val_df = pd.DataFrame(X_val)

        def objective(trial):
            params = suggest_params(trial, model_name)
            inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE)
            aucs = []
            briers = []
            for tr_idx, ev_idx in inner_cv.split(X_tr, y_tr):
                try:
                    X_in_tr_df = pd.DataFrame(X_tr[tr_idx])
                    X_in_ev_df = pd.DataFrame(X_tr[ev_idx])
                    clf = build_model(model_name, params)
                    # ---- 树模型统一使用早停（与 tune_model_on_full_training 目标函数一致）----
                    # early_stopping_rounds=50 见 _get_fixed_params（CatBoost/XGBoost 构造参数）
                    if model_name in ("LightGBM", "XGBoost", "CatBoost"):
                        if model_name == "LightGBM":
                            import lightgbm as lgb
                            clf.fit(X_in_tr_df, y_tr[tr_idx],
                                    eval_set=[(X_in_ev_df, y_tr[ev_idx])],
                                    callbacks=[lgb.early_stopping(50, verbose=False)])
                        elif model_name == "XGBoost":
                            clf.fit(X_in_tr_df, y_tr[tr_idx],
                                    eval_set=[(X_in_ev_df, y_tr[ev_idx])],
                                    verbose=False)
                        elif model_name == "CatBoost":
                            clf.fit(X_in_tr_df, y_tr[tr_idx],
                                    eval_set=(X_in_ev_df, y_tr[ev_idx]),
                                    verbose=False)
                    else:
                        clf.fit(X_in_tr_df, y_tr[tr_idx])
                    proba = clf.predict_proba(X_in_ev_df)[:, 1]
                    aucs.append(roc_auc_score(y_tr[ev_idx], proba))
                    briers.append(brier_score_loss(y_tr[ev_idx], proba))
                    # Free per-fold variables explicitly to avoid accumulation
                    try:
                        del X_in_tr_df, X_in_ev_df, clf, proba
                    except NameError:
                        pass
                except Exception as e:
                    # 内层 trial 失败：先打印警告再返回惩罚值，避免静默吞错
                    print(f"WARNING: [{model_name}] inner trial failed: {type(e).__name__}: {e}")
                    return 0.0, 0.5
            mean_auc = np.mean(aucs) if aucs else 0.0
            mean_brier = np.mean(briers) if briers else 0.5
            return mean_auc, mean_brier  # AUC ↑, Brier ↓

        sampler = TPESampler(seed=RANDOM_STATE)
        # 注：不使用 MedianPruner —— 目标函数未 report 中间值，且 MedianPruner
        #     本身不支持多目标 study（directions 为二元时 pruner 不生效）
        study = optuna.create_study(
            directions=["maximize", "minimize"],  # AUC ↑, Brier ↓
            sampler=sampler
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=inner_trials, n_jobs=1, show_progress_bar=False)

        # Multi-objective: select trial maximizing AB_Score on the Pareto front
        # optuna 4.x: best_trials returns Pareto front for multi-objective studies.
        # Try the modern get_pareto_front_trials() first, fall back to best_trials.
        try:
            pareto_trials = study.get_pareto_front_trials()
        except AttributeError:
            # optuna 4.9.0 compatibility: best_trials is the correct API
            pareto_trials = study.best_trials
        if len(pareto_trials) == 0:
            # 多目标 study 不存在单一 best_trial（访问 study.best_trial 会抛
            # RuntimeError），改为从全部完成态 trial 中按 AB_Score 取最大者兜底
            print("    Warning: Pareto front empty, falling back to AB_Score max over completed trials.")
            fallback_trials = [t for t in study.trials if t.values is not None]
            if len(fallback_trials) == 0:
                raise RuntimeError(
                    f"[{model_name}] fold {fold_idx}: no completed Optuna trials with valid values")
            pareto_trials = [max(fallback_trials,
                                 key=lambda t: 0.5 * t.values[0] + 0.5 * (1 - t.values[1]))]
        print(f"    Pareto trials found: {len(pareto_trials)}")
        # Select trial with highest AB_Score = 0.5*AUC + 0.5*(1-Brier) on Pareto front,
        # consistent with the external model selection criterion.
        best_trial = max(pareto_trials, key=lambda t: 0.5 * t.values[0] + 0.5 * (1 - t.values[1]))
        best_params = best_trial.params.copy()
        # Strip Optuna-internal params (e.g. gamma_use_str, gamma_str, gamma_float)
        # before passing to build_model, to avoid SVC.__init__() unexpected keyword errors.
        _optuna_internal_keys = {"gamma_use_str", "gamma_str", "gamma_float"}
        for _k in _optuna_internal_keys:
            best_params.pop(_k, None)
        ab_score = 0.5 * best_trial.values[0] + 0.5 * (1 - best_trial.values[1])
        print(f"    Pareto-selected: AUC={best_trial.values[0]:.4f}, Brier={best_trial.values[1]:.4f}, AB_Score={ab_score:.4f}")
        best_clf = build_model(model_name, best_params)
        try:
            # ---- 对树模型使用早停 ----
            if model_name in ("LightGBM", "XGBoost", "CatBoost"):
                X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
                    X_tr_df, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
                )
                if model_name == "LightGBM":
                    import lightgbm as lgb
                    best_clf.fit(X_tr_sub, y_tr_sub,
                                 eval_set=[(X_val_sub, y_val_sub)],
                                 callbacks=[lgb.early_stopping(50, verbose=False)])
                elif model_name == "XGBoost":
                    best_clf.fit(X_tr_sub, y_tr_sub,
                                 eval_set=[(X_val_sub, y_val_sub)],
                                 verbose=False)
                elif model_name == "CatBoost":
                    best_clf.fit(X_tr_sub, y_tr_sub,
                                 eval_set=(X_val_sub, y_val_sub),
                                 verbose=False)
            else:
                best_clf.fit(X_tr_df, y_tr)
            y_val_prob = best_clf.predict_proba(X_val_df)[:, 1]
            auc_val = roc_auc_score(y_val, y_val_prob)
            brier_val = brier_score_loss(y_val, y_val_prob)
            outer_aucs.append(auc_val)
            outer_brier.append(brier_val)

            # Youden cutoff
            fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            if len(thresholds) > 0 and best_idx < len(thresholds):
                fold_cutoff = thresholds[best_idx]
            else:
                fold_cutoff = 0.5
                print(f"\nWARNING: [{model_name}] fold {fold_idx} cutoff fallback to 0.5")
            # roc_curve 返回的首个阈值恒为 inf，若 Youden 恰好选中该点，
            # 非有限 cutoff 无法用于阈值化，回退为 0.5
            if not np.isfinite(fold_cutoff):
                fold_cutoff = 0.5
                print(f"\nWARNING: [{model_name}] fold {fold_idx} cutoff non-finite "
                      f"(roc_curve first threshold=inf), fallback to 0.5")
            outer_cutoffs.append(fold_cutoff)

            # Collect OOF predictions
            for i in range(len(y_val)):
                oof_records.append({
                    'model': model_name,
                    'fold': fold_idx,
                    'y_true': int(y_val[i]),
                    'y_pred_prob': float(y_val_prob[i])
                })

            if auc_val > best_auc_overall:
                best_auc_overall = auc_val
                best_params_overall = best_params

        except Exception as e:
            print(f"\nFAILED [{model_name}] fold {fold_idx}: {e}")
            outer_aucs.append(np.nan)
            outer_brier.append(np.nan)
            # 三个列表必须保持等长，否则逐折指标对齐会错位
            outer_cutoffs.append(np.nan)

        del study, X_tr_df, X_val_df
        gc.collect()

    oof_df = pd.DataFrame(oof_records)
    # OOF 完整性校验：任一外层 fold 失败都会导致 OOF 行数少于样本数，
    # 此时 OOF 未覆盖全队列，下游基于 OOF 的指标口径需留意
    if len(oof_df) != len(y):
        print(f"\n*** WARNING: [{model_name}] OOF predictions incomplete: "
              f"{len(oof_df)}/{len(y)} rows (some outer folds failed) ***")
    # Note: best_params_overall is the params from the best-performing outer fold,
    # used only for reference. Final params are determined by tune_model_on_full_training().
    return best_params_overall, outer_aucs, outer_brier, outer_cutoffs, oof_df


def tune_model_on_full_training(model_name, X, y, inner_folds, inner_trials):
    def objective(trial):
        params = suggest_params(trial, model_name)
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        briers = []
        for tr_idx, ev_idx in inner_cv.split(X, y):
            try:
                X_in_tr_df = pd.DataFrame(X[tr_idx])
                X_in_ev_df = pd.DataFrame(X[ev_idx])
                clf = build_model(model_name, params)
                # ---- 对树模型使用早停 ----
                if model_name in ("LightGBM", "XGBoost", "CatBoost"):
                    if model_name == "LightGBM":
                        import lightgbm as lgb
                        clf.fit(X_in_tr_df, y[tr_idx],
                                eval_set=[(X_in_ev_df, y[ev_idx])],
                                callbacks=[lgb.early_stopping(50, verbose=False)])
                    elif model_name == "XGBoost":
                        clf.fit(X_in_tr_df, y[tr_idx],
                                eval_set=[(X_in_ev_df, y[ev_idx])],
                                verbose=False)
                    elif model_name == "CatBoost":
                        clf.fit(X_in_tr_df, y[tr_idx],
                                eval_set=(X_in_ev_df, y[ev_idx]),
                                verbose=False)
                else:
                    clf.fit(X_in_tr_df, y[tr_idx])
                proba = clf.predict_proba(X_in_ev_df)[:, 1]
                aucs.append(roc_auc_score(y[ev_idx], proba))
                briers.append(brier_score_loss(y[ev_idx], proba))
                # Free per-fold variables explicitly
                try:
                    del X_in_tr_df, X_in_ev_df, clf, proba
                except NameError:
                    pass
            except Exception as e:
                # 内层 trial 失败：先打印警告再返回惩罚值，避免静默吞错
                print(f"WARNING: [{model_name}] inner trial failed: {type(e).__name__}: {e}")
                return 0.0, 0.5
        mean_auc = np.mean(aucs) if aucs else 0.0
        mean_brier = np.mean(briers) if briers else 0.5
        return mean_auc, mean_brier  # AUC ↑, Brier ↓

    # 注：不使用 MedianPruner —— 目标函数未 report 中间值，且 MedianPruner
    #     本身不支持多目标 study（directions 为二元时 pruner 不生效）
    study = optuna.create_study(
        directions=["maximize", "minimize"],  # AUC ↑, Brier ↓
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=inner_trials, n_jobs=1, show_progress_bar=False)

    # Multi-objective: select trial maximizing AB_Score on the Pareto front.
    # Try get_pareto_front_trials() first, fall back to best_trials for optuna 4.9.
    try:
        pareto_trials = study.get_pareto_front_trials()
    except AttributeError:
        pareto_trials = study.best_trials
    if len(pareto_trials) == 0:
        # 多目标 study 不存在单一 best_trial（访问 study.best_trial 会抛
        # RuntimeError），改为从全部完成态 trial 中按 AB_Score 取最大者兜底
        print("    Warning: Full-train Pareto front empty, falling back to AB_Score max over completed trials.")
        fallback_trials = [t for t in study.trials if t.values is not None]
        if len(fallback_trials) == 0:
            raise RuntimeError(
                f"[{model_name}] full-train tuning: no completed Optuna trials with valid values")
        pareto_trials = [max(fallback_trials,
                             key=lambda t: 0.5 * t.values[0] + 0.5 * (1 - t.values[1]))]
    print(f"    Full-train Pareto trials found: {len(pareto_trials)}")
    # Select trial with highest AB_Score = 0.5*AUC + 0.5*(1-Brier) on Pareto front,
    # consistent with the external model selection criterion.
    best_trial = max(pareto_trials, key=lambda t: 0.5 * t.values[0] + 0.5 * (1 - t.values[1]))
    ab_score = 0.5 * best_trial.values[0] + 0.5 * (1 - best_trial.values[1])
    print(f"  Full-train Pareto-selected: AUC={best_trial.values[0]:.4f}, Brier={best_trial.values[1]:.4f}, AB_Score={ab_score:.4f}")
    # Strip Optuna-internal params before returning
    best_params = best_trial.params.copy()
    _optuna_internal_keys = {"gamma_use_str", "gamma_str", "gamma_float"}
    for _k in _optuna_internal_keys:
        best_params.pop(_k, None)
    return best_params


# =============================================================================
# OOF-Based Metrics Computation
# =============================================================================
def compute_oof_metrics(y_true, y_prob):
    """Compute all metrics from OOF predictions for a single model."""
    # AUC + BCa CI
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(y_true, y_prob)

    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # Optimal cutoff (Youden)
    # 口径说明：cutoff 在同一 OOF 集合上重算，后续 Sens/Spec 等阈值化指标
    # 属于 apparent threshold performance（见文件头注释与 metrics CSV 说明）
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    cutoff = thresholds[best_idx] if len(thresholds) > 0 and best_idx < len(thresholds) else 0.5
    # roc_curve 返回的首个阈值恒为 inf，若 Youden 恰好选中该点需防护，回退 0.5
    if not np.isfinite(cutoff):
        cutoff = 0.5

    # Binary predictions
    y_pred = (y_prob >= cutoff).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ppv = prec  # PPV = Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "AUC": auc_mean,
        "AUC_CI_Lower": auc_lo,
        "AUC_CI_Upper": auc_hi,
        "Brier": brier,
        "Optimal_Cutoff": cutoff,
        "Sensitivity": sens,
        "Specificity": spec,
        "Accuracy": acc,
        "F1": f1,
        "PPV": ppv,
        "NPV": npv,
    }


# =============================================================================
# Save Figure Helper
# =============================================================================
def save_figure(fig, base_name, result_dir=RESULT_DIR):
    """Save figure in PNG, PDF, TIF formats (DPI=600)."""
    for fmt in ["png", "pdf", "tif"]:
        fpath = os.path.join(result_dir, f"{base_name}.{fmt}")
        fig.savefig(fpath, dpi=IMG_DPI, format=fmt, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)


# =============================================================================
# Plot: All-ROC Curves
# =============================================================================
def save_roc_curves(oof_data_dict, result_dir):
    """Draw 10 ROC curves on one plot using per-model OOF predictions."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (model_name, entry) in enumerate(oof_data_dict.items()):
        y_true = entry["y_true"]
        y_prob = entry["y_prob"]
        auc_val = entry["metrics"]["AUC"]
        auc_lo = entry["metrics"]["AUC_CI_Lower"]
        auc_hi = entry["metrics"]["AUC_CI_Upper"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, color=colors[idx], lw=1.5,
                label=f"{model_name}: AUC = {auc_val:.3f} "
                      f"(95% CI {auc_lo:.3f}-{auc_hi:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("1 - Specificity", fontsize=FONT_LABEL)
    ax.set_ylabel("Sensitivity", fontsize=FONT_LABEL)
    ax.set_title("ROC Curves — All Models (OOF)", fontsize=FONT_TITLE)
    # 图例格式 "模型名: AUC = x.xxx (95% CI lo-hi)"（去掉 nested-CV 口径括注）
    ax.legend(loc="lower right", fontsize=FONT_LEGEND, frameon=True)
    ax.grid(True, alpha=0.3)

    save_figure(fig, "roc_curves_all_models", result_dir)
    print(f"  ROC curves saved to {result_dir}/roc_curves_all_models.*")


# =============================================================================
# Model Registry
# =============================================================================
MODEL_LIST = [
    "L2_LR",
    "MLP",
    "KNN",
    "DT",
    "RF",
    "LightGBM",
    "CatBoost",
    "SVM",
    "GaussianNB",
    "XGBoost",
]

# =============================================================================
# Main: Nested CV
# =============================================================================
print("\n" + "=" * 70)
print("=== Nested Cross-Validation (Model Selection) ===")
print("=" * 70)

results = {}
all_oof_dfs = []

for idx, model_name in enumerate(MODEL_LIST, 1):
    print(f"\n[{idx}/{len(MODEL_LIST)}] Processing: {model_name}")

    if model_name == "LightGBM" and not HAS_LGBM:
        print(f"  SKIPPED: lightgbm not installed")
        continue
    if model_name == "CatBoost" and not HAS_CATBOOST:
        print(f"  SKIPPED: catboost not installed")
        continue
    if model_name == "XGBoost" and not HAS_XGB:
        print(f"  SKIPPED: xgboost not installed")
        continue

    try:
        t0 = time.time()
        best_params, cv_aucs, cv_brier, cv_cutoffs, oof_df = nested_cv_single_model(
            model_name, X_train_full, y_train_full,
            OUTER_FOLDS, INNER_FOLDS, INNER_TRIALS
        )
        elapsed = time.time() - t0

        results[model_name] = {
            "best_params": best_params,
            "cv_aucs": cv_aucs,
            "cv_brier": cv_brier,
            "cv_cutoffs": cv_cutoffs,
            "oof_df": oof_df,
        }
        all_oof_dfs.append(oof_df)

        auc_mean = np.nanmean(cv_aucs)
        auc_std = np.nanstd(cv_aucs)
        print(f"  Done in {elapsed:.1f}s | CV AUC: {auc_mean:.4f} +/- {auc_std:.4f}")

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()

# =============================================================================
# Save OOF Predictions
# =============================================================================
print("\n" + "=" * 70)
print("=== Saving OOF Predictions ===")
print("=" * 70)

oof_all = pd.concat(all_oof_dfs, ignore_index=True)
oof_csv = os.path.join(RESULT_DIR, "oof_predictions_all_models.csv")
oof_all.to_csv(oof_csv, index=False, float_format='%.6f')
print(f"  Saved: {oof_csv} ({oof_all.shape[0]} rows)")

# =============================================================================
# OOF-Based Metrics & Model Selection
# =============================================================================
print("\n" + "=" * 70)
print("=== OOF-Based Model Selection Metrics ===")
print("=" * 70)

oof_data = {}  # For ROC plot
metrics_rows = []

for model_name in results:
    oof_model = results[model_name]["oof_df"]
    if len(oof_model) == 0 or "y_true" not in oof_model.columns:
        print(f"  {model_name}: SKIPPED (no valid OOF predictions)")
        continue
    y_true_all = oof_model["y_true"].values
    y_prob_all = oof_model["y_pred_prob"].values

    metrics = compute_oof_metrics(y_true_all, y_prob_all)
    metrics["Model"] = model_name

    # Add CV AUC summary (mean ± SD, per-fold values)
    cv_aucs = results[model_name]["cv_aucs"]
    cv_briers = results[model_name]["cv_brier"]
    metrics["CV_AUC_Mean"] = np.nanmean(cv_aucs)
    metrics["CV_AUC_SD"] = np.nanstd(cv_aucs)
    metrics["CV_Brier_Mean"] = np.nanmean(cv_briers)
    metrics["CV_Brier_SD"] = np.nanstd(cv_briers)
    # Per-fold AUCs as comma-separated string
    metrics["CV_AUC_PerFold"] = ", ".join([f"{a:.4f}" for a in cv_aucs])

    metrics_rows.append(metrics)

    oof_data[model_name] = {"y_true": y_true_all, "y_prob": y_prob_all, "metrics": metrics}

    auc_str = f"{metrics['AUC']:.4f} ({metrics['AUC_CI_Lower']:.4f}-{metrics['AUC_CI_Upper']:.4f})"
    print(f"  {model_name}: AUC={auc_str}, Cutoff={metrics['Optimal_Cutoff']:.4f}, "
          f"Sens={metrics['Sensitivity']:.4f}, Spec={metrics['Specificity']:.4f}, "
          f"F1={metrics['F1']:.4f}")

# Save metrics CSV
metrics_df = pd.DataFrame(metrics_rows)
col_order = ["Model", "AUC", "AUC_CI_Lower", "AUC_CI_Upper", "Brier",
             "CV_AUC_Mean", "CV_AUC_SD", "CV_Brier_Mean", "CV_Brier_SD",
             "CV_AUC_PerFold", "Optimal_Cutoff", "Sensitivity", "Specificity",
             "Accuracy", "F1", "PPV", "NPV"]
# Filter to columns that actually exist in metrics_df
col_order = [c for c in col_order if c in metrics_df.columns]
metrics_df = metrics_df[col_order]

# Add AB_Score = 0.5*AUC + 0.5*(1-Brier) before saving
metrics_df["AB_Score"] = 0.5 * metrics_df["AUC"] + 0.5 * (1 - metrics_df["Brier"])

# 口径说明：本 CSV 中 Sensitivity/Specificity/Accuracy/F1/PPV/NPV 等阈值化指标
# 所用 Optimal_Cutoff 是在同一 OOF 集合上按 Youden 指数重算的，即
# "cutoff derived from the same OOF set (apparent threshold performance)"，
# 并非独立外部阈值，解释时需注意轻度乐观偏倚（CSV 结构为保持下游 05 脚本
# 兼容不加额外标注列，在此以注释与日志说明）。
metrics_csv = os.path.join(RESULT_DIR, "model_selection_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False, float_format='%.6f')
print(f"\n  Metrics (with AB_Score) saved to: {metrics_csv}")
print("  NOTE: threshold-based metrics (Sens/Spec/Acc/F1/PPV/NPV) use a cutoff derived")
print("        from the same OOF set (apparent threshold performance).")

# Auto-recommend best model by AB_Score
best_auc_row = metrics_df.loc[metrics_df["AUC"].idxmax()]
best_ab_row = metrics_df.loc[metrics_df["AB_Score"].idxmax()]
best_model_name = best_ab_row["Model"]
print(f"\n  >>> BEST by AUC: {best_auc_row['Model']} "
      f"(AUC={best_auc_row['AUC']:.4f}, Brier={best_auc_row['Brier']:.4f})")
print(f"  >>> BEST by AB_Score: {best_model_name} "
      f"(AUC={best_ab_row['AUC']:.4f}, Brier={best_ab_row['Brier']:.4f}, "
      f"AB_Score={best_ab_row['AB_Score']:.4f})")

# =============================================================================
# ROC Curves
# =============================================================================
print("\n" + "=" * 70)
print("=== Plotting ROC Curves ===")
print("=" * 70)
save_roc_curves(oof_data, RESULT_DIR)

# =============================================================================
# Save Best Parameters
# =============================================================================
print("\n" + "=" * 70)
print("=== Saving Best Parameters ===")
print("=" * 70)

best_params_dict = {}
for model_name in results:
    final_params = tune_model_on_full_training(
        model_name, X_train_full, y_train_full, INNER_FOLDS, INNER_TRIALS
    )
    best_params_dict[model_name] = final_params
    print(f"  {model_name}: {best_params_dict[model_name]}")

pkl_path = os.path.join(RESULT_DIR, "best_params_all_models.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(best_params_dict, f)
print(f"\n  Best params saved to: {pkl_path}")

# =============================================================================
# Plot 1: Horizontal Lollipop Chart — AB_Score Ranking
# =============================================================================
print("\n" + "=" * 70)
print("=== Plotting AB_Score Lollipop Chart ===")
print("=" * 70)

df_lollipop = metrics_df.sort_values("AB_Score", ascending=True).reset_index(drop=True)
n_models = len(df_lollipop)

fig, ax = plt.subplots(figsize=(9, max(5, n_models * 0.45)))
y_pos = np.arange(n_models)
ab_scores = df_lollipop["AB_Score"].values

# Color gradient: deep blue, darker = higher score
norm_c = plt.Normalize(ab_scores.min(), ab_scores.max())
blue_cmap = plt.cm.Blues

for i in range(n_models):
    score = ab_scores[i]
    # Gray stem from x=0.70 to score
    ax.plot([0.70, score], [y_pos[i], y_pos[i]], "-", color="lightgray",
            lw=0.8, alpha=0.8, zorder=1)
    # Dot (champion is last index = n_models-1, highlighted separately below)
    ax.scatter(score, y_pos[i], s=80, c=[blue_cmap(norm_c(score))],
               zorder=3, edgecolors="white", linewidth=0.5)

# Highlight champion (last element = highest AB_Score)
champion_idx = n_models - 1
champion_score = ab_scores[champion_idx]
# Redraw champion with red color (larger dot)
ax.scatter(champion_score, y_pos[champion_idx], s=120,
           c="#E63946", zorder=5, edgecolors="darkred", linewidth=1.0)

# Value labels: only AB_Score at the right of each lollipop
label_offset = ab_scores.max() * 0.02  # dynamic offset proportional to score range
for i in range(n_models):
    champion_row = (i == champion_idx)
    label_color = "#E63946" if champion_row else "#333333"
    font_wt = "bold" if champion_row else "normal"
    ax.text(ab_scores[i] + label_offset, y_pos[i],
            f"AB={ab_scores[i]:.3f}",
            fontsize=7.5, color=label_color, va="center", fontfamily="monospace",
            fontweight=font_wt)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_lollipop["Model"].values, fontsize=9)
ax.set_xlabel("AB_Score = 0.5*AUC + 0.5*(1-Brier)", fontsize=11)
ax.set_title("Model Selection by AB_Score (Balance of AUC & Calibration)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0.70, 0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")
plt.tight_layout()
save_figure(fig, "Model_Selection_Lollipop", RESULT_DIR)
print("  Lollipop chart saved.")

# =============================================================================
# Plot 2: Performance Gap Waterfall — AUC vs Calibration Balance
# （原 Pareto Frontier 可视化已按需求删除；Pareto 前沿仍用于超参选择，仅不出图）
# =============================================================================
print("\n" + "=" * 70)
print("=== Plotting Performance Gap Waterfall ===")
print("=" * 70)

df_gap = metrics_df.sort_values("AB_Score", ascending=True).reset_index(drop=True)
n_gap = len(df_gap)
auc_gap = df_gap["AUC"].values
cal_gap = 1 - df_gap["Brier"].values
gap_vals = auc_gap - cal_gap  # positive = AUC better, negative = cal better

fig, ax = plt.subplots(figsize=(10, max(5.5, n_gap * 0.45)))
y_pos_gap = np.arange(n_gap)

# 2.1: Pre-compute axis limits with right padding
x_min_gap = min(cal_gap.min(), auc_gap.min()) - 0.015
x_max_gap = max(cal_gap.max(), auc_gap.max()) + 0.08
ax.set_xlim(x_min_gap, x_max_gap)

for i in range(n_gap):
    auc_val = auc_gap[i]
    cal_val = cal_gap[i]
    gap = gap_vals[i]
    champion = (i == n_gap - 1)  # Last = highest AB_Score

    # 2.2: Champion bar — sand-gold color, double line width
    bar_color = "#F4A460" if champion else "#C0C0C0"
    bar_lw = 10 if champion else 5
    ax.plot([cal_val, auc_val], [y_pos_gap[i], y_pos_gap[i]],
            "-", color=bar_color, lw=bar_lw, alpha=0.85 if champion else 0.6,
            solid_capstyle="butt", zorder=5 if champion else 2)

    # AUC endpoint: deep blue circle
    ax.scatter(auc_val, y_pos_gap[i], s=90, c="#1F77B4", zorder=6,
               edgecolors="white", linewidth=0.5)

    # Calibration endpoint: orange square
    ax.scatter(cal_val, y_pos_gap[i], s=90, c="#FF7F0E", marker="s",
               zorder=6, edgecolors="white", linewidth=0.5)

    # 2.3: Champion endpoints — gold glow border
    if champion:
        ax.scatter(auc_val, y_pos_gap[i], s=160, c="#1F77B4",
                   edgecolors="#FFD700", linewidth=3, zorder=7)
        ax.scatter(cal_val, y_pos_gap[i], s=160, c="#FF7F0E", marker="s",
                   edgecolors="#FFD700", linewidth=3, zorder=7)

    # Gap label
    mid_x = (auc_val + cal_val) / 2
    if champion:
        # 2.4: Champion — right-aligned single line with full info
        ax.text(x_max_gap - 0.005, y_pos_gap[i],
                f"★ {best_model_name}  |  Gap={gap:+.3f}  |  AB={df_gap['AB_Score'].iloc[i]:.3f}",
                fontsize=9, fontweight="bold", color="#E63946",
                ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#E63946", alpha=0.95))
    else:
        # 2.5: Non-champion — gray small font with white box
        ax.text(mid_x, y_pos_gap[i], f"Gap={gap:+.3f}",
                fontsize=6, color="#888888", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.6))

ax.set_yticks(y_pos_gap)
ax.set_yticklabels(df_gap["Model"].values, fontsize=9)
ax.set_xlabel("Score", fontsize=11)
ax.set_title("Performance Gap Waterfall: AUC vs Calibration Balance",
             fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")

# 2.6: Legend — simplified with champion highlight
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#F4A460", lw=6, label="★ AB_Score Champion"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#1F77B4",
           markersize=8, label="AUC", markeredgecolor="#1F77B4"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF7F0E",
           markersize=8, label="Calibration (1-Brier)", markeredgecolor="#FF7F0E"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)
plt.tight_layout()
save_figure(fig, "Model_Selection_Gap_Waterfall", RESULT_DIR)
print("  Gap waterfall plot saved.")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("=== Model Selection Complete ===")
print(f"  Recommended model: {best_model_name}")
print(f"  All outputs saved to: {RESULT_DIR}")
print("=" * 70)