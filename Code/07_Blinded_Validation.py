# =============================================================================
# 07_Blinded_Validation.py — Blinded Cohort External Validation
# =============================================================================
# 功能：对最终模型进行盲法（Blinded）外部验证，做法与 05 中对
#       Temporal / Geographical 队列的处理保持一致：
#   1. 用与 01→02→03 完全一致的预处理链，将原始盲法队列转换为模型输入特征
#      （raw → 派生比值 → log1p → 拟合好的插补器 → RobustScaler → one-hot）。
#   2. 模型仅在 Derivation 上训练（不并入任何外部数据）。
#   3. 盲法队列计算两遍（由 --temporal-mode 开关控制策略）：
#        [calibration，默认] (a) Raw          —— 原始概率；
#                            (b) Recalibrated —— 用 Temporal 队列拟合校准器后重算。
#        [merge]             (a) Before retraining —— Derivation-only 模型；
#                            (b) After retraining  —— 并入 Temporal 重训后的模型（model revision）。
#      用法：python 07_Blinded_Validation.py [--model X] [--temporal-mode calibration|merge]
#   4. 输出全部判别/校准/临床指标（Bootstrap 95% CI、Hosmer-Lemeshow、DCA）；
#      校准曲线与决策曲线沿用既有脚本（05 / 06）的绘图风格。
#
# 输入：
#   - blinded cohort.csv（原始，位于 E:\new）
#   - feature_selection\{Derivation,Temporal} cohort.csv（已标准化）
#   - Model Selection\best_params_all_models.pkl、model_selection_metrics.csv
#   - data_preprocessing\preprocessing_objects.joblib（预处理参数）
# 输出（E:\new\Blinded Validation\）：
#   - 表格：blinded_validation_summary.csv, blinded_table1_baseline.csv,
#           blinded_hosmer_lemeshow.csv, blinded_cutoff_sensitivity.csv,
#           blinded_predictions.csv,
#           blinded_auc_comparison.csv, blinded_calibrator.csv
#   - 图：ROC, 校准曲线, DCA, 混淆矩阵, PR曲线, 概率分布 (PNG+PDF+TIF, 600dpi)
#   - 报告：blinded_report.txt
# =============================================================================

import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, mannwhitneyu, chi2_contingency
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             brier_score_loss, accuracy_score, recall_score,
                             precision_score, f1_score, confusion_matrix,
                             average_precision_score)

warnings.filterwarnings("ignore")

# =============================================================================
# Global Configuration
# =============================================================================
RANDOM_STATE = 123
N_JOBS = max(1, (os.cpu_count() or 1) // 2)
BOOTSTRAP_ITER = 1000

# --- Model selection (same convention as 05) ---
SELECTED_MODEL = None  # None = auto recommended; override: --model X

# --- 训练集策略开关 (--temporal-mode) ---
#   calibration : 模型仅在 Derivation 上训练；Temporal 仅用于拟合重校准器（默认，原做法）
#   merge       : 将 Temporal 并入训练集后重训模型（model revision），
#                 用于对比“重训前 vs 重训后”的判别力（AUC / AUPRC）。
TEMPORAL_MODE = "calibration"
_argv = sys.argv[1:]
for _i, _a in enumerate(_argv):
    if _a == "--model" and _i + 1 < len(_argv):
        SELECTED_MODEL = _argv[_i + 1]
    elif _a == "--temporal-mode" and _i + 1 < len(_argv):
        TEMPORAL_MODE = _argv[_i + 1]
if TEMPORAL_MODE not in ("calibration", "merge"):
    raise SystemExit(f"Unknown --temporal-mode '{TEMPORAL_MODE}' (use 'calibration' or 'merge')")

# 两套结果的命名（随模式变化）
if TEMPORAL_MODE == "merge":
    SET1_NAME, SET2_NAME = "Before retraining (Derivation only)", "After retraining (Derivation + Temporal)"
    SET1_TAG,  SET2_TAG  = "Before retraining", "After retraining"
else:
    SET1_NAME, SET2_NAME = "Raw", "Recalibrated (Temporal)"
    SET1_TAG,  SET2_TAG  = "Without calibration", "With calibration"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "feature_selection")
SELECTION_DIR = os.path.join(BASE_DIR, "Model Selection")
PREPROC_DIR = os.path.join(BASE_DIR, "data_preprocessing")
RESULT_DIR = os.path.join(BASE_DIR, "Blinded Validation")
os.makedirs(RESULT_DIR, exist_ok=True)

BLINDED_RAW = os.path.join(BASE_DIR, "blinded cohort.csv")
DERIVATION_RAW = os.path.join(BASE_DIR, "Derivation cohort.csv")

# --- 绘图配色（沿用 05 的色盲友好配色方案） ---
COLOR_RAW = "#1f77b4"       # 原始概率（蓝）
COLOR_CAL = "#d62728"       # 时间集校准后（红）
COLOR_DER_REF = "#2ca02c"   # Derivation 参考曲线（绿）

# Image settings
IMG_DPI = 600
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": IMG_DPI,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def save_figure(fig, base_name):
    for ext in ["png", "pdf", "tif"]:
        fig.savefig(os.path.join(RESULT_DIR, f"{base_name}.{ext}"),
                    dpi=IMG_DPI, bbox_inches="tight", transparent=False)
    plt.close(fig)


# =============================================================================
# ScaledImputer stub (needed to unpickle preprocessing_objects.joblib)
# =============================================================================
def _get_scale_params(train_df, cols):
    means, stds = {}, {}
    for col in cols:
        vals = train_df[col].dropna()
        means[col] = vals.mean()
        s = vals.std()
        stds[col] = s if s > 0 else 1.0
    return means, stds


class ScaledImputer:
    def __init__(self, imputer, use_scaling=True, feature_names=None):
        self.imputer = imputer
        self.use_scaling = use_scaling
        self.feature_names = feature_names or []
        self.means_ = None
        self.stds_ = None

    def transform(self, X):
        if self.use_scaling:
            cols = self.feature_names if self.feature_names else list(range(X.shape[1]))
            df = pd.DataFrame(X, columns=cols)
            for c in cols:
                df[c] = (df[c] - self.means_[c]) / self.stds_[c]
            Xi = self.imputer.transform(df.values)
            dfi = pd.DataFrame(Xi, columns=cols)
            for c in cols:
                dfi[c] = dfi[c] * self.stds_[c] + self.means_[c]
            return dfi.values
        return self.imputer.transform(X)


def _r_make_names(s):
    """Emulate R make.names(): non-alphanumeric -> '.'."""
    return "".join(ch if (ch.isalnum() or ch == "_") else "." for ch in s)


def _logit(p, eps=1e-10):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


# =============================================================================
# Bootstrap CIs
# =============================================================================
def bootstrap_auc_ci(y_true, y_prob, n_iter=BOOTSTRAP_ITER, alpha=0.05, random_state=RANDOM_STATE):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    try:
        theta = roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(random_state)
    boots = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        try:
            boots[b] = roc_auc_score(y_true[idx], y_prob[idx])
        except ValueError:
            boots[b] = theta
    return theta, np.percentile(boots, 100 * alpha / 2), np.percentile(boots, 100 * (1 - alpha / 2))


def bootstrap_auprc_ci(y_true, y_prob, n_iter=BOOTSTRAP_ITER, alpha=0.05, random_state=RANDOM_STATE):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    try:
        theta = average_precision_score(y_true, y_prob)
    except ValueError:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(random_state)
    boots = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        try:
            boots[b] = average_precision_score(y_true[idx], y_prob[idx])
        except ValueError:
            boots[b] = theta
    return theta, np.percentile(boots, 100 * alpha / 2), np.percentile(boots, 100 * (1 - alpha / 2))


def _bootstrap_auc_diff_ci(y1, p1, y2, p2, n_boot=BOOTSTRAP_ITER):
    rng = np.random.RandomState(RANDOM_STATE)
    diffs = []
    n1, n2 = len(y1), len(y2)
    for _ in range(n_boot):
        i1 = rng.choice(n1, n1, replace=True)
        i2 = rng.choice(n2, n2, replace=True)
        try:
            diffs.append(roc_auc_score(y1[i1], p1[i1]) - roc_auc_score(y2[i2], p2[i2]))
        except ValueError:
            continue
    diffs = np.array(diffs)
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def _paired_diff_ci(y, p_new, p_old, metric="auc", n_boot=BOOTSTRAP_ITER):
    """同一队列上配对 Bootstrap 的 Δ(metric)（用于“重训前 vs 重训后”比较）。"""
    y = np.asarray(y)
    n = len(y)
    if metric == "auc":
        point = roc_auc_score(y, p_new) - roc_auc_score(y, p_old)
        fn = lambda yy, pp: roc_auc_score(yy, pp)
    else:
        point = average_precision_score(y, p_new) - average_precision_score(y, p_old)
        fn = lambda yy, pp: average_precision_score(yy, pp)
    rng = np.random.RandomState(RANDOM_STATE)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(fn(y[idx], p_new[idx]) - fn(y[idx], p_old[idx]))
        except ValueError:
            continue
    vals = np.array(vals)
    if vals.size == 0:
        return point, np.nan, np.nan
    return point, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# =============================================================================
# Metrics & Tests
# =============================================================================
def compute_metrics(y_true, y_prob, cutoff):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auc, auc_lo, auc_hi = bootstrap_auc_ci(y_true, y_prob)
    auprc, auprc_lo, auprc_hi = bootstrap_auprc_ci(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    try:
        cm = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    except (TypeError, ValueError):
        cm = LogisticRegression(C=1e12, solver="lbfgs", max_iter=5000)
    cm.fit(logit.reshape(-1, 1), y_true)
    calib_int = cm.intercept_[0]
    calib_slope = cm.coef_[0][0]

    # Bootstrap 95% CI for Calibration Slope（与 05 同口径：1000 次、seed=123、百分位法）
    rng = np.random.RandomState(RANDOM_STATE)
    bs_slopes = np.empty(BOOTSTRAP_ITER)
    for b in range(BOOTSTRAP_ITER):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            cm_bs = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
            cm_bs.fit(logit[idx].reshape(-1, 1), y_true[idx])
            bs_slopes[b] = cm_bs.coef_[0][0]
        except Exception:
            bs_slopes[b] = np.nan
    bs_slopes = bs_slopes[~np.isnan(bs_slopes)]
    slope_ci_lo = np.percentile(bs_slopes, 2.5) if len(bs_slopes) > 0 else np.nan
    slope_ci_hi = np.percentile(bs_slopes, 97.5) if len(bs_slopes) > 0 else np.nan

    return {
        "AUC": auc, "AUC_CI_Lower": auc_lo, "AUC_CI_Upper": auc_hi,
        "AUPRC": auprc, "AUPRC_CI_Lower": auprc_lo, "AUPRC_CI_Upper": auprc_hi,
        "Brier": brier,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "PPV": precision_score(y_true, y_pred, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Calib_Intercept": calib_int, "Calib_Slope": calib_slope,
        "Calib_Slope_CI_Lower": slope_ci_lo,
        "Calib_Slope_CI_Upper": slope_ci_hi,
        "Cutoff": cutoff,
    }


def hosmer_lemeshow_test(y_true, y_prob, g=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    try:
        df["bin"] = pd.qcut(df["p"], g, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["p"], g)
    grp = df.groupby("bin", observed=True).agg(
        obs=("y", "sum"), exp=("p", "sum"), n=("y", "count"))
    grp = grp[grp["n"] > 0]
    grp["obs_neg"] = grp["n"] - grp["obs"]
    grp["exp_neg"] = grp["n"] - grp["exp"]
    eps = 1e-10
    chi2_val = (((grp["obs"] - grp["exp"]) ** 2) / (grp["exp"] + eps)).sum() + \
               (((grp["obs_neg"] - grp["exp_neg"]) ** 2) / (grp["exp_neg"] + eps)).sum()
    dof = len(grp) - 2
    p_val = 1 - chi2.cdf(chi2_val, dof) if dof > 0 else np.nan
    return chi2_val, dof, p_val


def net_benefit_curve(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true)
    n = len(y_true)
    nb = np.empty(len(thresholds))
    for i, pt in enumerate(thresholds):
        pred = (y_prob >= pt).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        nb[i] = tp / n - (fp / n) * (pt / (1 - pt))
    return nb


# =============================================================================
# Model Builder (mirrors 04/05)
# =============================================================================
def _get_fixed_params(model_name):
    if model_name == "L2_LR":
        return {"solver": "lbfgs", "max_iter": 5000, "random_state": RANDOM_STATE}
    elif model_name == "KNN":
        return {"n_jobs": N_JOBS}
    elif model_name == "DT":
        return {"random_state": RANDOM_STATE}
    elif model_name == "RF":
        return {"random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    elif model_name == "LightGBM":
        # 与 04/05 一致：n_estimators=1000，早停通过 fit 的 callbacks 施加
        return {"n_estimators": 1000, "verbose": -1, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    elif model_name == "CatBoost":
        # 与 04/05 一致：iterations=1000 + early_stopping_rounds=50
        return {"iterations": 1000, "early_stopping_rounds": 50, "verbose": 0,
                "random_seed": RANDOM_STATE, "thread_count": max(1, os.cpu_count() or 1),
                "train_dir": RESULT_DIR}
    elif model_name == "SVM":
        # class_weight 与 04/05 的回退默认值保持一致（若 best_params 缺失该键，
        # 07 才不会退化为 None 而与 05 的模型不一致）
        return {"probability": True, "random_state": RANDOM_STATE,
                "class_weight": "balanced"}
    elif model_name == "GaussianNB":
        return {}
    elif model_name == "XGBoost":
        # 与 04/05 一致：n_estimators=1000，构造参数内携带 early_stopping_rounds
        return {"n_estimators": 1000, "early_stopping_rounds": 50, "eval_metric": "auc",
                "verbosity": 0, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    return {}


def build_model(model_name, params):
    merged = {**_get_fixed_params(model_name), **params}
    if model_name == "L2_LR":
        return LogisticRegression(penalty="l2", **merged)
    elif model_name == "KNN":
        return KNeighborsClassifier(**merged)
    elif model_name == "DT":
        return DecisionTreeClassifier(**merged)
    elif model_name == "RF":
        return RandomForestClassifier(**merged)
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**merged)
    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**merged)
    elif model_name == "SVM":
        return SVC(**merged)
    elif model_name == "GaussianNB":
        return GaussianNB(**merged)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**merged)
    raise ValueError(f"Unknown model: {model_name}")


def fit_predict(model_name, params, X_tr, y_tr, X_te):
    """训练并预测。

    树模型（LightGBM/XGBoost/CatBoost）与 04/05 对齐：在 train 内做分层 20% eval
    set + early_stopping_rounds=50；其余模型全量拟合。此前 07 未使用早停，导致
    calibration / merge 两套结果与 04/05 的模型不可比。
    """
    clf = build_model(model_name, params)
    cols = [f"f{i}" for i in range(X_tr.shape[1])]
    X_tr_df = pd.DataFrame(X_tr, columns=cols)
    X_te_df = pd.DataFrame(X_te, columns=cols)
    y_tr = np.asarray(y_tr)

    if model_name in ("LightGBM", "XGBoost", "CatBoost"):
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_tr_df, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
        )
        if model_name == "LightGBM":
            import lightgbm as lgb
            clf.fit(X_sub, y_sub,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
        elif model_name == "XGBoost":
            clf.fit(X_sub, y_sub, eval_set=[(X_val, y_val)], verbose=False)
        else:  # CatBoost
            clf.fit(X_sub, y_sub, eval_set=(X_val, y_val), verbose=False)
        prob = clf.predict_proba(X_te_df)[:, 1]
        return clf, prob

    try:
        clf.fit(X_tr_df, y_tr)
        prob = clf.predict_proba(X_te_df)[:, 1]
    except Exception:
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
    return clf, prob


# =============================================================================
# Preprocessing Reproduction: raw blinded -> model-ready scaled features
# =============================================================================
def transform_raw_cohort(raw_df, pp, model_feature_cols):
    numeric_cols = pp["numeric_cols"]
    categorical_cols = pp["categorical_cols"]
    log_cols = pp["transformed_log_cols"]
    srp = pp["scaler_robust_params"]
    imputer = pp["imputer"]

    df = raw_df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if "LDL-C/HDL-C" not in df.columns and {"LDL-C", "HDL-C"}.issubset(df.columns):
            df["LDL-C/HDL-C"] = df["LDL-C"] / df["HDL-C"]
        if "total cholesterol/HDL-C" not in df.columns and {"total cholesterol", "HDL-C"}.issubset(df.columns):
            df["total cholesterol/HDL-C"] = df["total cholesterol"] / df["HDL-C"]
        if "triglycerides/HDL-C" not in df.columns and {"triglycerides", "HDL-C"}.issubset(df.columns):
            df["triglycerides/HDL-C"] = df["triglycerides"] / df["HDL-C"]

    num = df[numeric_cols].copy()
    for c in log_cols:
        num[c] = np.log1p(num[c])
    imp = imputer.transform(num.values)
    scaled = pd.DataFrame(imp, columns=numeric_cols, index=df.index)
    for c in numeric_cols:
        med = srp[c]["median"]
        iqr = srp[c]["iqr"]
        scaled[c] = 0.0 if iqr == 0 else (scaled[c] - med) / iqr

    for c in categorical_cols:
        vals = pp["encoders"][c]["values"]
        for v in vals[1:]:
            scaled[f"{c}_{v}"] = (df[c] == v).astype(int)

    rename_map = {}
    for c in numeric_cols:
        rename_map[c] = _r_make_names(c)
    for c in categorical_cols:
        vals = pp["encoders"][c]["values"]
        for v in vals[1:]:
            # 与 R 端 make.names 完全一致：R 的 read.csv(check.names=TRUE) 不会把
            # gender_1 简化成 gender，二分类也必须保留 _1 后缀，否则特征名对不上。
            rename_map[f"{c}_{v}"] = _r_make_names(f"{c}_{v}")
    scaled = scaled.rename(columns=rename_map)

    missing = [c for c in model_feature_cols if c not in scaled.columns]
    if missing:
        raise KeyError(f"Transformed cohort missing model features: {missing}")
    return scaled[model_feature_cols]


# =============================================================================
# Reference-style plotting helpers (mirror 05 calibration / 07 DCA)
# =============================================================================
def plot_calibration_reference(curves, title_tag):
    """curves: list of dicts {label, y_true, y_prob, color, hl_p}. 参考 05 plot_calibration_curves。"""
    fig, ax = plt.subplots(figsize=(7, 7))
    for c in curves:
        y_true, y_prob = c["y_true"], c["y_prob"]
        n_bins = 8 if len(y_prob) < 100 else 10
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        brier_val = brier_score_loss(y_true, y_prob)
        hl_p = c.get("hl_p", np.nan)
        if hl_p is not None and not np.isnan(hl_p):
            label = f"{c['label']} (Brier={brier_val:.3f}, HL p={hl_p:.3f})"
        else:
            label = f"{c['label']} (Brier={brier_val:.3f})"
        ax.plot(prob_pred, prob_true, "o-", color=c["color"], markersize=6, lw=1.5,
                markerfacecolor=c["color"], markeredgewidth=1.0, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Ideal")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Observed Proportion", fontsize=12, fontweight="bold")
    ax.set_title(f"Calibration Curves — {title_tag}")
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def plot_dca_reference(model_curves, prevalence, title_tag, primary_idx=0):
    """参考 07 plot_dca_curves：阈值 0.01–0.80；Treat None / Treat All / Model + 临床价值区域。"""
    thresholds = np.arange(0.01, 0.81, 0.01)
    fig, ax = plt.subplots(figsize=(7.5, 6))

    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, np.zeros_like(thresholds), color="black", lw=1.3, ls=":", label="Treat None")
    ax.plot(thresholds, nb_all, color="gray", lw=1.5, ls="--",
            label=f"Treat All (prevalence={prevalence:.3f})")
    ax.plot(thresholds, np.full_like(thresholds, prevalence), color="darkred", lw=1.0, ls="-.",
            alpha=0.6, label="Ideal (perfect)")

    primary_nb = None
    model_nbs = []
    for i, mc in enumerate(model_curves):
        nb = net_benefit_curve(mc["y_true"], mc["y_prob"], thresholds)
        model_nbs.append(nb)
        ax.plot(thresholds, nb, color=mc["color"], lw=2.4, label=mc["label"])
        if i == primary_idx:
            primary_nb = nb

    if primary_nb is not None:
        gain = primary_nb - nb_all
        ax.fill_between(thresholds, nb_all, primary_nb, where=(gain > 0),
                        color=model_curves[primary_idx]["color"], alpha=0.12,
                        label="Clinical value region")

    ax.set_xlim([0.0, 0.80])
    # y 下限只按模型曲线自适应：Treat All 在高阈值必然陡降到 -1 以下，
    # 若纳入 ylim 会把模型曲线压成一条平线（与 06 DCA 的处理保持一致）。
    y_lo = min([-0.05] + [float(np.nanmin(nb)) - 0.02 for nb in model_nbs]) if model_nbs else -0.05
    ax.set_ylim([y_lo, max(0.55, prevalence + 0.05)])
    ax.set_xlabel("Threshold Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Net Benefit", fontsize=12, fontweight="bold")
    ax.set_title(f"Decision Curve Analysis — {title_tag}")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


# =============================================================================
# MAIN
# =============================================================================
print("=" * 70)
print("07_Blinded_Validation.py — Blinded Cohort External Validation")
print("=" * 70)

# --- best params & model selection ---
with open(os.path.join(SELECTION_DIR, "best_params_all_models.pkl"), "rb") as f:
    best_params_dict = pickle.load(f)
metrics_csv = os.path.join(SELECTION_DIR, "model_selection_metrics.csv")
sel_df = pd.read_csv(metrics_csv)
recommended = sel_df.loc[sel_df["AB_Score"].idxmax(), "Model"] if "AB_Score" in sel_df.columns \
    else sel_df.loc[sel_df["AUC"].idxmax(), "Model"]
model_name = SELECTED_MODEL if SELECTED_MODEL else recommended
print(f"Recommended model: {recommended} | Using: {model_name}")

best_params = best_params_dict[model_name]
crow = sel_df[sel_df["Model"] == model_name]
final_cutoff = float(crow["Optimal_Cutoff"].values[0]) if len(crow) else 0.5
print(f"Best params: {best_params}")
print(f"Optimal cutoff (Derivation, from 04): {final_cutoff:.4f}")

# --- load preprocessing objects ---
pp = joblib.load(os.path.join(PREPROC_DIR, "preprocessing_objects.joblib"))

# --- load Derivation (already scaled) ---
der_df = pd.read_csv(os.path.join(DATA_DIR, "Derivation cohort.csv"))
feature_cols = [c for c in der_df.columns if c not in ("ID", "label")]
X_der = der_df[feature_cols].values.astype(np.float64)
y_der = der_df["label"].values.astype(np.int32)
print(f"\nDerivation (training): {X_der.shape}, Pos/Neg: {np.sum(y_der==1)}/{np.sum(y_der==0)}")

# --- load Temporal (already scaled) — used ONLY as calibration set ---
temp_df = pd.read_csv(os.path.join(DATA_DIR, "Temporal external validation cohort.csv"))
X_temp = temp_df[feature_cols].values.astype(np.float64)
y_temp = temp_df["label"].values.astype(np.int32)
print(f"Temporal (calibration set): {X_temp.shape}, Pos/Neg: {np.sum(y_temp==1)}/{np.sum(y_temp==0)}")
print(f"Features ({len(feature_cols)}): {feature_cols}")

# --- load & transform blinded ---
blind_raw = pd.read_csv(BLINDED_RAW)
print(f"\nBlinded raw: {blind_raw.shape}")
y_blind = blind_raw["label"].values.astype(np.int32)
X_blind = transform_raw_cohort(blind_raw, pp, feature_cols).values.astype(np.float64)
print(f"Blinded transformed: {X_blind.shape}, Pos/Neg: {np.sum(y_blind==1)}/{np.sum(y_blind==0)}")

# --- train model on Derivation only ---
print(f"\n=== Training Final Model: {model_name} (Derivation only) ===")
clf, der_prob = fit_predict(model_name, best_params, X_der, y_der, X_der)
_, temp_prob = fit_predict(model_name, best_params, X_der, y_der, X_temp)
_, blind_prob_raw = fit_predict(model_name, best_params, X_der, y_der, X_blind)
print("  Training complete.")

# --- 第二套结果：calibration 模式=时间集重校准；merge 模式=并入 Temporal 后重训 ---
print(f"\n=== Temporal strategy: {TEMPORAL_MODE} ===")
if TEMPORAL_MODE == "merge":
    X_tr_m = np.vstack([X_der, X_temp])
    y_tr_m = np.concatenate([y_der, y_temp])
    print(f"  Retraining on Derivation + Temporal: {X_tr_m.shape}, "
          f"Pos/Neg: {np.sum(y_tr_m==1)}/{np.sum(y_tr_m==0)}")
    _, der_prob_m = fit_predict(model_name, best_params, X_tr_m, y_tr_m, X_der)
    _, temp_prob_m = fit_predict(model_name, best_params, X_tr_m, y_tr_m, X_temp)
    _, blind_prob_cal = fit_predict(model_name, best_params, X_tr_m, y_tr_m, X_blind)
    der_prob_cal = der_prob_m
    cal_a, cal_b = np.nan, np.nan
    # Temporal 已成为训练数据，故重训后模型不再做时间集重校准
    final_cutoff_cal = final_cutoff
    print("  Retraining complete (no post-hoc recalibration: Temporal is training data).")
else:
    print("\n=== Recalibration (calibrator fitted on Temporal) ===")
    logit_temp = _logit(temp_prob).reshape(-1, 1)
    calib = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    calib.fit(logit_temp, y_temp)
    cal_a = calib.coef_[0][0]
    cal_b = calib.intercept_[0]
    print(f"  Calibrator (logit scale): slope={cal_a:.4f}, intercept={cal_b:.4f}")

    blind_prob_cal = calib.predict_proba(_logit(blind_prob_raw).reshape(-1, 1))[:, 1]
    der_prob_cal = calib.predict_proba(_logit(der_prob).reshape(-1, 1))[:, 1]
    # 校准后的截断值：把 Derivation 最优截断值经校准器映射到校准概率空间
    final_cutoff_cal = calib.predict_proba(np.array([[_logit(final_cutoff)]]))[:, 1][0]
    print(f"  Applied cutoff: raw={final_cutoff:.4f} → recalibrated={final_cutoff_cal:.4f}")

# --- metrics (raw vs recalibrated) ---
print("\n=== Metrics ===")
m_der_der = compute_metrics(y_der, der_prob, final_cutoff)          # Derivation apparent
m_blind_raw = compute_metrics(y_blind, blind_prob_raw, final_cutoff)
m_blind_cal = compute_metrics(y_blind, blind_prob_cal, final_cutoff_cal)

def _fmt(m):
    return (f"AUC={m['AUC']:.3f} ({m['AUC_CI_Lower']:.3f}-{m['AUC_CI_Upper']:.3f}), "
            f"AUPRC={m['AUPRC']:.3f}, Brier={m['Brier']:.3f}")

print(f"  Derivation (apparent)  : {_fmt(m_der_der)}")
print(f"  Blinded [{SET1_NAME}] : {_fmt(m_blind_raw)}")
print(f"  Blinded [{SET2_NAME}] : {_fmt(m_blind_cal)}")
for tag, m in [(SET1_NAME, m_blind_raw), (SET2_NAME, m_blind_cal)]:
    print(f"    [{tag}] clinical: Acc={m['Accuracy']:.3f}, Sens={m['Sensitivity']:.3f}, "
          f"Spec={m['Specificity']:.3f}, PPV={m['PPV']:.3f}, NPV={m['NPV']:.3f}, F1={m['F1']:.3f}")
    print(f"    [{tag}] calib   : Intercept={m['Calib_Intercept']:.3f}, "
          f"Slope={m['Calib_Slope']:.3f} ({m['Calib_Slope_CI_Lower']:.3f}-{m['Calib_Slope_CI_Upper']:.3f})")

# HL tests
hl_chi_raw, hl_dof_raw, hl_p_raw = hosmer_lemeshow_test(y_blind, blind_prob_raw, g=10)
hl_chi_cal, hl_dof_cal, hl_p_cal = hosmer_lemeshow_test(y_blind, blind_prob_cal, g=10)
print(f"  Hosmer-Lemeshow (Blinded): [{SET1_NAME}] chi2={hl_chi_raw:.3f}, p={hl_p_raw:.4f} | "
      f"[{SET2_NAME}] chi2={hl_chi_cal:.3f}, p={hl_p_cal:.4f}")

# AUC comparison Derivation vs Blinded (AUC invariant to monotone recalibration)
diff, d_lo, d_hi = _bootstrap_auc_diff_ci(y_der, der_prob, y_blind, blind_prob_raw)
z = diff / max((d_hi - d_lo) / (2 * 1.96), 1e-9)
p_diff = 2 * (1 - norm.cdf(abs(z)))
print(f"  AUC diff (Der-apparent − Blinded) = {diff:.3f} ({d_lo:.3f}-{d_hi:.3f}), p={p_diff:.3f}")

# 重训前 vs 重训后：同队列配对 Bootstrap 的 ΔAUC / ΔAUPRC（merge 模式）
if TEMPORAL_MODE == "merge":
    auc_delta, ad_lo, ad_hi = _paired_diff_ci(y_blind, blind_prob_cal, blind_prob_raw, "auc")
    auprc_delta, pd_lo, pd_hi = _paired_diff_ci(y_blind, blind_prob_cal, blind_prob_raw, "auprc")
    print(f"  Delta AUC   (after - before) = {auc_delta:+.4f} ({ad_lo:+.4f}, {ad_hi:+.4f})")
    print(f"  Delta AUPRC (after - before) = {auprc_delta:+.4f} ({pd_lo:+.4f}, {pd_hi:+.4f})")
else:
    auc_delta = auprc_delta = ad_lo = ad_hi = pd_lo = pd_hi = np.nan

# --- Table 1: baseline characteristics ---
CONT_VARS = ["age", "mean arterial pressure", "pulse pressure index", "pulse pressure",
             "total cholesterol", "triglycerides", "LDL-C", "HDL-C", "uric acid",
             "plasma fibrinogen", "creatinine", "urea"]
BIN_VARS = ["gender", "hypertension", "diabetes", "smoking"]
der_raw = pd.read_csv(DERIVATION_RAW)
der_ids = set(der_df["ID"].tolist())
der_raw = der_raw[der_raw["ID"].isin(der_ids)]
print(f"\nTable 1: Derivation n={len(der_raw)}, Blinded n={len(blind_raw)}")

t1_rows = []
for v in CONT_VARS:
    a = der_raw[v].dropna().values
    b = blind_raw[v].dropna().values
    try:
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:
        p = np.nan
    t1_rows.append({"Variable": v, "Type": "Continuous",
                    "Derivation (mean±SD)": f"{a.mean():.2f} ± {a.std():.2f}",
                    "Blinded (mean±SD)": f"{b.mean():.2f} ± {b.std():.2f}",
                    "P_value": p})
for v in BIN_VARS:
    a = der_raw[v].dropna().astype(int).values
    b = blind_raw[v].dropna().astype(int).values
    ct = pd.crosstab(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
    try:
        p = chi2_contingency(ct)[1]
    except Exception:
        p = np.nan
    t1_rows.append({"Variable": v, "Type": "Binary",
                    "Derivation (n,%)": f"{int(a.sum())} ({100*a.mean():.1f}%)",
                    "Blinded (n,%)": f"{int(b.sum())} ({100*b.mean():.1f}%)",
                    "P_value": p})
pd.DataFrame(t1_rows).to_csv(os.path.join(RESULT_DIR, "blinded_table1_baseline.csv"),
                             index=False, float_format="%.4f")

# --- Cutoff sensitivity (on recalibrated probabilities) ---
fpr, tpr, thr = roc_curve(y_blind, blind_prob_cal)
youden = tpr - fpr
best_i = np.argmax(youden)
self_cut = thr[best_i] if np.isfinite(thr[best_i]) else 0.5
y_pred_self = (blind_prob_cal >= self_cut).astype(int)
tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_blind, y_pred_self, labels=[0, 1]).ravel()
sens_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0
spec_s = tn_s / (tn_s + fp_s) if (tn_s + fp_s) else 0
pd.DataFrame([
    {"Calibration": SET2_NAME, "Cutoff_Type": "Applied (Derivation optimal)",
     "Cutoff": final_cutoff_cal, "Sensitivity": m_blind_cal["Sensitivity"], "Specificity": m_blind_cal["Specificity"]},
    {"Calibration": SET2_NAME, "Cutoff_Type": "Self-optimal (Blinded Youden)",
     "Cutoff": self_cut, "Sensitivity": sens_s, "Specificity": spec_s},
]).to_csv(os.path.join(RESULT_DIR, "blinded_cutoff_sensitivity.csv"),
          index=False, float_format="%.6f")

# --- Summary table (仅盲法、第二套结果) ---
def _row(cohort, status, n, events, m, auc_est):
    r = {"Cohort": cohort, "Model": model_name, "Calibration_Status": status,
         "AUC_Estimation": auc_est, "N": n, "Events": events}
    r.update(m)
    return r
summary = pd.DataFrame([
    _row("Blinded", SET2_NAME, len(y_blind), int(np.sum(y_blind == 1)),
         m_blind_cal, "Derivation + Temporal training set" if TEMPORAL_MODE == "merge"
         else "Bootstrap mean (calibrated probs)"),
])
front = ["Cohort", "Model", "Calibration_Status", "AUC_Estimation", "N", "Events"]
summary = summary[front + [c for c in summary.columns if c not in front]]
summary.to_csv(os.path.join(RESULT_DIR, "blinded_validation_summary.csv"),
               index=False, float_format="%.4f")

# --- Calibrator table ---
pd.DataFrame([{"Temporal_Mode": TEMPORAL_MODE, "Calibration_Set": "Temporal" if TEMPORAL_MODE == "calibration" else "(none)",
               "Slope_logit": cal_a, "Intercept_logit": cal_b,
               "Cutoff_Raw": final_cutoff, "Cutoff_Recalibrated": final_cutoff_cal}]).to_csv(
    # merge 模式下校准器不存在（cal_a/cal_b 为 NaN），统一写空字段而非 "NaN"，
    # 避免 R 端 read.csv 把它读成字符串 "NaN" 后误当有效系数。
    os.path.join(RESULT_DIR, "blinded_calibrator.csv"),
    index=False, float_format="%.6f", na_rep="")

# --- HL table (仅第二套结果) ---
pd.DataFrame([
    {"Cohort": "Blinded", "Calibration_Status": SET2_NAME,
     "Chi2": hl_chi_cal, "df": hl_dof_cal, "P_value": hl_p_cal},
]).to_csv(os.path.join(RESULT_DIR, "blinded_hosmer_lemeshow.csv"), index=False, float_format="%.4f")

# --- AUC comparison table ---
pd.DataFrame([{"Comparison": "Derivation vs Blinded", "AUC_Diff": diff,
               "CI_Lower": d_lo, "CI_Upper": d_hi, "P_value": p_diff}]).to_csv(
    os.path.join(RESULT_DIR, "blinded_auc_comparison.csv"), index=False, float_format="%.4f")

# --- 重训前 vs 重训后 判别力对比表（merge 模式）---
if TEMPORAL_MODE == "merge":
    pd.DataFrame([
        {"Comparison": "After retraining vs Before retraining (Blinded)", "Metric": "AUC",
         "Before": m_blind_raw["AUC"], "After": m_blind_cal["AUC"],
         "Delta": auc_delta, "CI_Lower": ad_lo, "CI_Upper": ad_hi},
        {"Comparison": "After retraining vs Before retraining (Blinded)", "Metric": "AUPRC",
         "Before": m_blind_raw["AUPRC"], "After": m_blind_cal["AUPRC"],
         "Delta": auprc_delta, "CI_Lower": pd_lo, "CI_Upper": pd_hi},
    ]).to_csv(os.path.join(RESULT_DIR, "blinded_retrain_comparison.csv"),
              index=False, float_format="%.4f")

# --- predictions file ---
pred_cols = {
    "ID": blind_raw["ID"].values,
    "label": y_blind,
    "prob_blind_raw": blind_prob_raw,
    "prob_blind_recalibrated": blind_prob_cal,
    "predicted_class": (blind_prob_cal >= final_cutoff_cal).astype(int),
}
if TEMPORAL_MODE == "merge":
    pred_cols["prob_blind_retrained"] = blind_prob_cal   # 并入 Temporal 后重训模型的概率
pd.DataFrame(pred_cols).to_csv(os.path.join(RESULT_DIR, "blinded_predictions.csv"),
                               index=False, float_format="%.6f")

# --- 供 R 端 (07_Blinded_Validation.R) 使用的原始概率导出 ---
t_cols = {"label": y_temp, "prob_blind_raw": temp_prob}
d_cols = {"label": y_der, "prob_blind_raw": der_prob}
if TEMPORAL_MODE == "merge":
    t_cols["prob_blind_retrained"] = temp_prob_m
    d_cols["prob_blind_retrained"] = der_prob_m
pd.DataFrame(t_cols).to_csv(os.path.join(RESULT_DIR, "temporal_predictions.csv"),
                            index=False, float_format="%.6f")
pd.DataFrame(d_cols).to_csv(os.path.join(RESULT_DIR, "derivation_predictions.csv"),
                            index=False, float_format="%.6f")
pd.DataFrame({
    "label": y_blind,
    "prob_blind_raw": blind_prob_raw,
}).to_csv(os.path.join(RESULT_DIR, "blinded_predictions_raw.csv"), index=False, float_format="%.6f")

# --- 模式标记（供 R 端读取：以何种策略生成上述概率）---
with open(os.path.join(RESULT_DIR, "temporal_mode.txt"), "w", encoding="utf-8") as f:
    f.write(TEMPORAL_MODE + "\n")

# =============================================================================
# Figures
# =============================================================================
print("\n=== Generating Figures ===")

# 需要对比展示的两套概率（merge 模式下为“重训前 / 重训后”两条曲线）
fig_sets = [(SET1_NAME, blind_prob_raw, COLOR_RAW, m_blind_raw),
            (SET2_NAME, blind_prob_cal, COLOR_CAL, m_blind_cal)]
if TEMPORAL_MODE == "calibration":
    fig_sets = fig_sets[1:]   # 单调校准不改变排序，故只画第二套避免完全重叠

# (1) ROC — 盲法（merge 模式下同时画重训前后两条曲线）
fig, ax = plt.subplots(figsize=(6.5, 6))
for _name, _p, _col, _m in fig_sets:
    _fpr, _tpr, _ = roc_curve(y_blind, _p)
    ax.plot(_fpr, _tpr, color=_col, lw=2.4,
            label=f"{_name} AUC={_m['AUC']:.3f} ({_m['AUC_CI_Lower']:.3f}-{_m['AUC_CI_Upper']:.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":")
ax.set_xlabel("1 − Specificity", fontweight="bold")
ax.set_ylabel("Sensitivity", fontweight="bold")
ax.set_title(f"ROC Curve — Blinded Validation ({model_name})")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
save_figure(fig, "blind_roc")
print("  [1/6] ROC saved")

# (2) Calibration — 第二套结果（参考 05 风格）
cal_curves = [
    {"label": f"Blinded {SET2_NAME}", "y_true": y_blind, "y_prob": blind_prob_cal,
     "color": COLOR_CAL, "hl_p": hl_p_cal},
]
fig = plot_calibration_reference(cal_curves, f"Blinded ({model_name})")
save_figure(fig, "blind_calibration")
print("  [2/6] Calibration saved")

# (3) DCA — 第二套结果（参考 07 风格）
prev_blind = float(np.mean(y_blind))
dca_curves = [
    {"label": SET2_NAME, "y_true": y_blind, "y_prob": blind_prob_cal, "color": COLOR_CAL},
]
fig = plot_dca_reference(dca_curves, prev_blind, f"Blinded ({model_name})", primary_idx=0)
save_figure(fig, "blind_dca")
print("  [3/6] DCA saved")

# (4) Confusion matrix — 仅校准后
fig, ax = plt.subplots(figsize=(5.5, 5))
cm = confusion_matrix(y_blind, (blind_prob_cal >= final_cutoff_cal).astype(int), labels=[0, 1])
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Neg", "Pred Pos"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["True Neg", "True Pos"])
ax.set_title(f"Confusion Matrix — Blinded ({SET2_TAG})\n(cutoff={final_cutoff_cal:.3f})")
fig.colorbar(im, ax=ax, shrink=0.8)
save_figure(fig, "blind_confusion_matrix")
print("  [4/6] Confusion matrix saved")

# (5) PR curve（merge 模式下同时画重训前后两条曲线）
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for _name, _p, _col, _m in fig_sets:
    _prec, _rec, _ = precision_recall_curve(y_blind, _p)
    ax.plot(_rec, _prec, color=_col, lw=2.4,
            label=f"{_name} AUPRC={_m['AUPRC']:.3f} "
                  f"({_m['AUPRC_CI_Lower']:.3f}-{_m['AUPRC_CI_Upper']:.3f})")
ax.axhline(prev_blind, color="gray", ls="--", lw=1.2, label=f"Baseline (prevalence) = {prev_blind:.3f}")
ax.set_xlabel("Recall (Sensitivity)", fontweight="bold")
ax.set_ylabel("Precision (PPV)", fontweight="bold")
ax.set_title(f"Precision-Recall Curve — Blinded ({model_name})")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
save_figure(fig, "blind_pr_curve")
print("  [5/6] PR curve saved")

# (6) Probability distribution — 仅校准后
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.hist(blind_prob_cal[y_blind == 0], bins=20, alpha=0.6, color="#1f77b4", label="No Event", density=True)
ax.hist(blind_prob_cal[y_blind == 1], bins=20, alpha=0.6, color="#d62728", label="Event", density=True)
ax.axvline(final_cutoff_cal, color="black", ls="--", lw=1.5, label=f"Cutoff = {final_cutoff_cal:.3f}")
ax.set_xlabel("Predicted Probability", fontweight="bold")
ax.set_ylabel("Density", fontweight="bold")
ax.set_title(f"Predicted Probability Distribution — Blinded ({SET2_TAG})")
ax.legend(loc="upper center", fontsize=9)
ax.grid(alpha=0.3, axis="y")
save_figure(fig, "blind_probability_distribution")
print("  [6/6] Probability distribution saved")

# =============================================================================
# Text report
# =============================================================================
lines = []
lines.append("=" * 70)
lines.append("Blinded Cohort External Validation Report")
lines.append("=" * 70)
lines.append(f"Temporal strategy: {TEMPORAL_MODE}")
if TEMPORAL_MODE == "merge":
    lines.append(f"Model: {model_name} retrained on Derivation + Temporal (n={len(y_der)+len(y_temp)})")
    lines.append("Recalibrator: none (Temporal is part of the training set)")
    lines.append(f"Cutoff: {final_cutoff:.4f} (Derivation optimal, applied to BOTH models)")
    lines.append("  NOTE: the retrained model may have a different probability scale; "
                 "the same Derivation-optimal cutoff is applied to both models for "
                 "comparability and was NOT re-optimized on the retrained model.")
else:
    lines.append(f"Model: {model_name} (trained on Derivation only, n={len(y_der)})")
    lines.append(f"Recalibrator: Platt (logit) fitted on Temporal (n={len(y_temp)}); "
                 f"slope={cal_a:.4f}, intercept={cal_b:.4f}")
    lines.append(f"Cutoff: raw={final_cutoff:.4f} -> recalibrated={final_cutoff_cal:.4f}")
lines.append(f"Cutoff source: 04_Model_Selection.py 重复嵌套 CV 的重复级 Optimal_Cutoff "
             f"中位数（N_REPEATS 次重复）")
lines.append(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
lines.append("")
lines.append(f"[Cohorts] Derivation n={len(y_der)} (events={int(np.sum(y_der==1))}); "
             f"Temporal n={len(y_temp)} (events={int(np.sum(y_temp==1))}, "
             f"{'training' if TEMPORAL_MODE == 'merge' else 'calibration only'}); "
             f"Blinded n={len(y_blind)} (events={int(np.sum(y_blind==1))}, "
             f"prevalence={np.mean(y_blind):.3f})")
lines.append("")
lines.append("[Discrimination — Blinded]")
lines.append(f"  {SET1_NAME}: AUC={m_blind_raw['AUC']:.3f} ({m_blind_raw['AUC_CI_Lower']:.3f}-{m_blind_raw['AUC_CI_Upper']:.3f}), "
             f"AUPRC={m_blind_raw['AUPRC']:.3f} ({m_blind_raw['AUPRC_CI_Lower']:.3f}-{m_blind_raw['AUPRC_CI_Upper']:.3f})")
lines.append(f"  {SET2_NAME}: AUC={m_blind_cal['AUC']:.3f} ({m_blind_cal['AUC_CI_Lower']:.3f}-{m_blind_cal['AUC_CI_Upper']:.3f}), "
             f"AUPRC={m_blind_cal['AUPRC']:.3f} ({m_blind_cal['AUPRC_CI_Lower']:.3f}-{m_blind_cal['AUPRC_CI_Upper']:.3f})")
if TEMPORAL_MODE == "merge":
    lines.append(f"  Delta AUC   (after - before) = {auc_delta:+.4f} ({ad_lo:+.4f}, {ad_hi:+.4f})")
    lines.append(f"  Delta AUPRC (after - before) = {auprc_delta:+.4f} ({pd_lo:+.4f}, {pd_hi:+.4f})")
lines.append(f"  Derivation apparent vs Blinded (before): diff={diff:.3f} ({d_lo:.3f}-{d_hi:.3f}), p={p_diff:.4f}")
lines.append("")
lines.append(f"[Blinded — {SET2_NAME}]")
lines.append(f"  Brier={m_blind_cal['Brier']:.4f}; Calib Intercept={m_blind_cal['Calib_Intercept']:.3f}, "
             f"Slope={m_blind_cal['Calib_Slope']:.3f} "
             f"({m_blind_cal['Calib_Slope_CI_Lower']:.3f}-{m_blind_cal['Calib_Slope_CI_Upper']:.3f}); "
             f"HL chi2={hl_chi_cal:.3f}, p={hl_p_cal:.4f}")
lines.append(f"  Acc={m_blind_cal['Accuracy']:.3f}, Sens={m_blind_cal['Sensitivity']:.3f}, "
             f"Spec={m_blind_cal['Specificity']:.3f}, PPV={m_blind_cal['PPV']:.3f}, "
             f"NPV={m_blind_cal['NPV']:.3f}, F1={m_blind_cal['F1']:.3f}")
lines.append("")
lines.append("Outputs saved to: " + RESULT_DIR)
lines.append("=" * 70)
report = "\n".join(lines)
with open(os.path.join(RESULT_DIR, "blinded_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print("\n" + report)

print("\n=== Blinded Validation Complete ===")
