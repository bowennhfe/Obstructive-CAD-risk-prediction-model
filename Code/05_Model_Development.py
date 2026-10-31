# =============================================================================
# 05_Model_Development.py — Final Model Training & Full Validation
#                         + Subgroup Vulnerability Analysis
# =============================================================================
# 功能：
#   Part 1  读取最佳参数，在 Derivation 队列上训练，并在 Derivation 与两个外部
#           验证集上全面评估，生成论文所需的所有图表和统计检验。
#           校准策略：Derivation / Temporal 使用 Derivation 拟合的 Platt 校准器；
#           Geographical 使用「时间集校准」（Temporal 拟合的 Platt），
#           与 07 盲法验证完全一致；模型始终只在 Derivation 上训练。
#   Part 2  亚组脆弱性分析（精确反标准化年龄 + ESC SCORE2 三分组 +
#           Raw/Recalibrated 校准对比 + Brier 改善棒棒糖图），直接复用内存中的
#           模型、校准器与各队列预测概率，不二次读盘。
#
# 输入：
#   - Derivation / Temporal / Geographical cohort.csv（来自 feature_selection/）
#   - E:\new\Model Selection\best_params_all_models.pkl
# 输出：
#   - E:\new\Model Development（Part 1）：final_model_summary.csv,
#     predicted_probabilities_*.csv, auc_comparison_results.csv,
#     hosmer_lemeshow_results.csv,
#     以及 ROC/校准曲线/DCA/SHAP 等图（PNG+PDF+TIF, DPI=600）
#   - E:\new\Subgroup Analysis（Part 2）：Subgroup_Performance_{model}.csv,
#     Subgroup_Calibration_Comparison.csv, subgroup_brier_lollipop.*
# =============================================================================

import os
import sys
import warnings
import pickle
import traceback
import numpy as np
import pandas as pd
import sklearn as _sklearn
import optuna as _optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, mannwhitneyu
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, average_precision_score,
                             precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

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

# --- Required library check (不再运行时静默 pip install) ---
def _ensure_library(pip_name, import_name=None):
    """检测依赖包是否可导入；缺失时抛出带明确安装提示的 ImportError。"""
    if import_name is None:
        import_name = pip_name
    try:
        __import__(import_name)
    except ImportError as e:
        raise ImportError(
            f"缺少依赖包 '{pip_name}'，脚本无法继续运行。"
            f"请先手动安装: {sys.executable} -m pip install {pip_name}"
        ) from e

# Ensure required libraries
# lightgbm / catboost 改为按需导入：只有真正选用该模型时才要求安装，
# 否则选择 L2_LR 等模型时缺库不应中断整个脚本。
for lib in ["shap", "seaborn", "statsmodels"]:
    _ensure_library(lib)

import shap
import seaborn as sns
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("INFO: XGBoost loaded successfully")
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — XGBoost will be skipped")

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# ScaledImputer stub for unpickling preprocessing_objects.joblib
# =============================================================================
# 01_data_preprocessing.py 保存的 preprocessing_objects.joblib 内含自定义类
# ScaledImputer（imputer 变量）。pickle 反序列化按模块级属性查找该类，
# 因此必须定义为模块级（函数内定义会报
# "module '__main__' has no attribute 'ScaledImputer'"）。
# 与 02_vif_analysis.py 的 stub 保持一致。
class ScaledImputer:
    """Minimal stub for unpickling preprocessing_objects.joblib."""
    def __init__(self, imputer=None, use_scaling=True, feature_names=None):
        self.imputer = imputer
        self.use_scaling = use_scaling
        self.feature_names = feature_names


def _r_make_names(name):
    """模拟 R 的 make.names()：非法字符（空格、-/ 等）替换为 '.'。
    feature_selection CSV 列名由 R read.csv(check.names=TRUE) 生成，
    与原始列名（空格/连字符形式）的对应关系据此建立。
    例如 'mean arterial pressure' -> 'mean.arterial.pressure'。"""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_.]", ".", str(name))

# =============================================================================
# Global Configuration
# =============================================================================
RANDOM_STATE = 123
N_JOBS = max(1, (os.cpu_count() or 1) // 2)
BOOTSTRAP_ITER = 1000

# --- Colorblind-friendly (Okabe-Ito based) palette, consistent with 06_Clinical_Plots_R.R ---
# 蓝-橙-绿组合对红绿色盲友好；固定队列顺序颜色映射，保证跨脚本一致。
COLORBLIND_PALETTE = {
    "Derivation": "#1f77b4",   # blue
    "Temporal": "#ff7f0e",     # orange
    "Geographical": "#2ca02c", # green
}
COHORT_FALLBACK_COLORS = ["#d62728", "#9467bd", "#8c564b"]  # 仅当出现第4+队列时使用

# --- User-specified model ---
# Set to None to auto-use recommended model; or set to e.g. "L2_LR", "RF", etc.
# Command-line override: python 05_Model_Development.py --model SVM
SELECTED_MODEL = None  # <--- 设为 None 自动选择推荐模型，或手动指定模型名
if len(sys.argv) > 1 and sys.argv[1] == "--model" and len(sys.argv) > 2:
    SELECTED_MODEL = sys.argv[2]

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "feature_selection")
SELECTION_DIR = os.path.join(BASE_DIR, "Model Selection")
RESULT_DIR = os.path.join(BASE_DIR, "Model Development")
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
    # SCI 出版要求：PDF 嵌入 TrueType（Type 42）字体，避免字体替换（与 04 脚本一致）
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =============================================================================
# Model Building Functions
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
        # 与 04_Model_Selection.py 一致：n_estimators=1000，
        # 早停通过 fit 的 callbacks=[lgb.early_stopping(50)] 施加（见主流程调用点）
        return {"n_estimators": 1000, "verbose": -1, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    elif model_name == "CatBoost":
        # 与 04_Model_Selection.py 一致：iterations=1000 + early_stopping_rounds=50
        return {"iterations": 1000, "early_stopping_rounds": 50, "verbose": 0,
                "random_seed": RANDOM_STATE, "thread_count": max(1, os.cpu_count() or 1),
                "train_dir": RESULT_DIR}
    elif model_name == "SVM":
        return {"probability": True, "random_state": RANDOM_STATE, "class_weight": "balanced"}
    elif model_name == "GaussianNB":
        return {}
    elif model_name == "XGBoost":
        # 与 04_Model_Selection.py 一致：n_estimators=1000，构造参数内携带
        # early_stopping_rounds=50 与 eval_metric="auc"（早停基于 eval_set 的 AUC）
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
        # saga needs higher max_iter
        if merged.get("penalty") in ("l1", "elasticnet") and "solver" not in merged:
            merged["solver"] = "saga"
        if merged.get("penalty") in ("l1", "elasticnet") and merged.get("max_iter", 5000) < 10000:
            merged["max_iter"] = 10000
        return LogisticRegression(**merged)
    elif model_name == "KNN":
        return KNeighborsClassifier(**merged)
    elif model_name == "DT":
        return DecisionTreeClassifier(**merged)
    elif model_name == "RF":
        return RandomForestClassifier(**merged)
    elif model_name == "LightGBM":
        _ensure_library("lightgbm")
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**merged)
    elif model_name == "CatBoost":
        _ensure_library("catboost")
        from catboost import CatBoostClassifier
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
# BCa Bootstrap AUC CI
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

    rng = np.random.RandomState(random_state)
    boot_estimates = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_estimates[b] = roc_auc_score(y_true[idx], y_prob[idx])
        except ValueError:
            boot_estimates[b] = theta_hat

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


def bootstrap_auprc_ci(y_true, y_prob, n_iter=BOOTSTRAP_ITER, alpha=0.05, random_state=RANDOM_STATE):
    """Bootstrap percentile CI for AUPRC (Average Precision).
    Percentile method is preferred over BCa for AUPRC due to better stability
    with discrete metrics (Boyd et al., 2013)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)

    try:
        theta_hat = average_precision_score(y_true, y_prob)
    except ValueError:
        return np.nan, np.nan, np.nan

    rng = np.random.RandomState(random_state)
    boot_scores = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_scores[b] = average_precision_score(y_true[idx], y_prob[idx])
        except ValueError:
            boot_scores[b] = theta_hat

    lower = np.percentile(boot_scores, 100 * (alpha / 2))
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return np.mean(boot_scores), lower, upper


# =============================================================================
# Compute Clinical Metrics
# =============================================================================
def compute_metrics(y_true, y_prob, cutoff):
    """Compute all metrics for a given cutoff."""
    y_pred = (y_prob >= cutoff).astype(int)
    # 显式 labels=[0,1]，防止某一类缺失时 ravel() 返回维度不足而报错
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(y_true, y_prob)
    auprc_mean, auprc_lo, auprc_hi = bootstrap_auprc_ci(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    prob_clip = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit_prob = np.log(prob_clip / (1 - prob_clip))
    calib_intercept = sm.GLM(y_true, np.ones((len(y_true), 1)), family=sm.families.Binomial(), offset=logit_prob).fit().params[0]
    calib_slope = sm.GLM(y_true, sm.add_constant(logit_prob), family=sm.families.Binomial()).fit().params[1]

    # Bootstrap 95% CI for Calibration Slope
    rng = np.random.RandomState(123)
    n_bs = 1000
    bs_slopes = np.empty(n_bs)
    for b in range(n_bs):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            cal_bs = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000)
            cal_bs.fit(logit_prob[idx].reshape(-1, 1), y_true[idx])
            bs_slopes[b] = cal_bs.coef_[0][0]
        except Exception:
            bs_slopes[b] = np.nan
    bs_slopes = bs_slopes[~np.isnan(bs_slopes)]
    slope_ci_lo = np.percentile(bs_slopes, 2.5) if len(bs_slopes) > 0 else np.nan
    slope_ci_hi = np.percentile(bs_slopes, 97.5) if len(bs_slopes) > 0 else np.nan

    return {
        "AUC": auc_mean, "AUC_CI_Lower": auc_lo, "AUC_CI_Upper": auc_hi,
        "AUPRC": auprc_mean, "AUPRC_CI_Lower": auprc_lo, "AUPRC_CI_Upper": auprc_hi,
        "Brier": brier,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "PPV": precision_score(y_true, y_pred, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "Calib_Intercept": calib_intercept,
        "Calib_Slope": calib_slope,
        "Calib_Slope_CI_Lower": slope_ci_lo,
        "Calib_Slope_CI_Upper": slope_ci_hi,
        "Cutoff": cutoff,
    }


# =============================================================================
# Independent AUC Comparison via Bootstrap Z-Test
# =============================================================================
def auc_comparison_test(y1, prob1, y2, prob2, label1="Cohort1", label2="Cohort2"):
    """Compare two independent-sample AUCs using bootstrap z-test.
    Returns: (auc1, auc2, z_statistic, p_value, auc_diff, auc_diff_ci_lo, auc_diff_ci_hi)
    Note: label1/label2 are kept for API compatibility but no longer used internally.
    """
    return _bootstrap_auc_comparison(y1, prob1, y2, prob2)


def _bootstrap_auc_diff_ci(y1, prob1, y2, prob2, n_boot=BOOTSTRAP_ITER):
    """Bootstrap 95% CI for AUC difference (independent samples)."""
    rng = np.random.RandomState(RANDOM_STATE)
    diffs = []
    n1, n2 = len(y1), len(y2)
    for _ in range(n_boot):
        idx1 = rng.choice(n1, n1, replace=True)
        idx2 = rng.choice(n2, n2, replace=True)
        diffs.append(roc_auc_score(y1[idx1], prob1[idx1]) - roc_auc_score(y2[idx2], prob2[idx2]))
    return np.mean(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def _bootstrap_auc_comparison(y1, prob1, y2, prob2, n_boot=BOOTSTRAP_ITER):
    """Fallback: bootstrap-based independent z-test for AUC comparison.
    Returns: (auc1, auc2, z_stat, p_value, auc_diff, auc_diff_lo, auc_diff_hi)
    """
    auc1 = roc_auc_score(y1, prob1)
    auc2 = roc_auc_score(y2, prob2)
    n1, n2 = len(y1), len(y2)
    rng = np.random.RandomState(RANDOM_STATE)

    boot1 = []
    for _ in range(n_boot):
        idx = rng.choice(n1, n1, replace=True)
        boot1.append(roc_auc_score(y1[idx], prob1[idx]))
    boot2 = []
    for _ in range(n_boot):
        idx = rng.choice(n2, n2, replace=True)
        boot2.append(roc_auc_score(y2[idx], prob2[idx]))

    se_diff = np.sqrt(np.std(boot1) ** 2 + np.std(boot2) ** 2)
    auc_diff = auc1 - auc2
    z_stat = auc_diff / se_diff if se_diff > 0 else 0
    p_val = 2 * norm.sf(abs(z_stat))

    # Bootstrap AUC difference CI
    _, auc_diff_lo, auc_diff_hi = _bootstrap_auc_diff_ci(y1, prob1, y2, prob2)

    return auc1, auc2, z_stat, p_val, auc_diff, auc_diff_lo, auc_diff_hi


# =============================================================================
# Hosmer-Lemeshow Test
# =============================================================================
def hosmer_lemeshow_test(y_true, y_prob, g=10):
    """Hosmer-Lemeshow goodness-of-fit test (分位数分组).
    Returns: (chi2_statistic, p_value, df)
    """
    n = len(y_true)
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    obs = []
    exp = []
    for i in range(g):
        start = int(i * n / g)
        end = int((i + 1) * n / g)
        if i == g - 1:
            end = n
        obs.append(np.sum(y_true_sorted[start:end]))
        exp.append(np.sum(y_prob_sorted[start:end]))

    obs = np.array(obs, dtype=np.float64)
    exp = np.array(exp, dtype=np.float64)

    group_n = np.array([int((i+1)*n/g) - int(i*n/g) for i in range(g)])
    group_n[-1] = n - int((g-1)*n/g)

    # Merge groups with too few samples (<5)
    while np.min(group_n) < 5 and len(group_n) > 2:
        merge_idx = np.argmin(group_n)
        if merge_idx == 0:
            obs[1] += obs[0]
            exp[1] += exp[0]
            group_n[1] += group_n[0]
        else:
            obs[merge_idx - 1] += obs[merge_idx]
            exp[merge_idx - 1] += exp[merge_idx]
            group_n[merge_idx - 1] += group_n[merge_idx]
        obs = np.delete(obs, merge_idx)
        exp = np.delete(exp, merge_idx)
        group_n = np.delete(group_n, merge_idx)

    expected_non_events = group_n - exp
    chi2_stat = np.sum(((obs - exp) ** 2) / np.clip(exp, 1e-8, None) + (((group_n - obs) - expected_non_events) ** 2) / np.clip(expected_non_events, 1e-8, None))
    df = max(1, len(obs) - 2)
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value, df


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
# Plot 1: ROC Curves (Dynamic Cohorts)
# =============================================================================
def plot_roc_curves(preds_dict, model_name, result_dir, nested_cv_auc=None, nested_cv_ci_lower=None, nested_cv_ci_upper=None):
    """ROC curves for all available cohorts.

    图-表同源：Derivation 曲线使用 04 脚本 nested-CV OOF 原始概率绘制，
    图例标注的 nested-CV AUC 与 final_model_summary.csv 的
    Optimism-Corrected (Nested-CV) 行一致（= N_REPEATS 次重复的均值，
    CI 为 均值 ± 1.96×重复级 SE，非 bootstrap 区间）。
    注：04 导出的 OOF 文件为 repeat 1 的代表性 OOF，故曲线本身来自第 1 次重复，
    其 AUC 与图例的 10 次重复均值可能相差千分位——图标题已注明曲线来源。
    外部队列使用校准后概率，图例 AUC 由绘图概率直接计算（BCa bootstrap CI）。
    """
    # 固定队列顺序的色盲友好配色
    def _cohort_color(cohort, idx):
        return COLORBLIND_PALETTE.get(cohort, COHORT_FALLBACK_COLORS[idx % len(COHORT_FALLBACK_COLORS)])
    colors = {cohort: _cohort_color(cohort, i) for i, cohort in enumerate(preds_dict)}

    # Load nested-CV OOF predictions (raw scale) for Derivation figure-table consistency
    oof_data = None
    oof_path = os.path.join(SELECTION_DIR, "oof_predictions_all_models.csv")
    if os.path.exists(oof_path):
        try:
            oof_all = pd.read_csv(oof_path)
            oof_sub = oof_all[oof_all["model"] == model_name]
            if len(oof_sub) > 0:
                oof_data = (oof_sub["y_true"].values.astype(np.int32),
                            oof_sub["y_pred_prob"].values.astype(np.float64))
        except Exception as e:
            print(f"  WARNING: failed to load OOF for ROC ({e}), falling back to preds_dict.")

    fig, ax = plt.subplots(figsize=(7, 7))

    for i, (cohort, (y_true, y_prob)) in enumerate(preds_dict.items()):
        if cohort == "Derivation" and oof_data is not None:
            # 曲线与图例均基于 nested-CV OOF 原始概率（图-表同源）
            y_true_plot, y_prob_plot = oof_data
            auc_mean = roc_auc_score(y_true_plot, y_prob_plot)
            if nested_cv_auc is not None:
                auc_mean = nested_cv_auc  # 与表格报告的 nested-CV AUC 完全一致
            ci_low = nested_cv_ci_lower if nested_cv_ci_lower is not None else np.nan
            ci_high = nested_cv_ci_upper if nested_cv_ci_upper is not None else np.nan
            if not np.isnan(ci_low) and not np.isnan(ci_high):
                label = (f"{cohort} (Nested-CV AUC={auc_mean:.3f}, "
                         f"95% CI {ci_low:.3f}-{ci_high:.3f}; mean±1.96SE over repeats)")
            else:
                label = f"{cohort} (Nested-CV AUC={auc_mean:.3f})"
        else:
            y_true_plot, y_prob_plot = y_true, y_prob
            auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(y_true_plot, y_prob_plot)
            label = f"{cohort} (AUC={auc_mean:.3f}, 95% CI {auc_lo:.3f}-{auc_hi:.3f})"
        fpr, tpr, _ = roc_curve(y_true_plot, y_prob_plot)
        ax.plot(fpr, tpr, lw=1.8, color=colors[cohort], label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"ROC Curves — {model_name}\n"
                 f"(Derivation curve: repeat 1 OOF; AUC label: repeat-level mean)",
                 fontsize=10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)

    save_figure(fig, f"roc_curves_{model_name}", result_dir)
    print(f"  ROC curves saved.")


# =============================================================================
# Plot 1a: Derivation Optimism (Apparent vs Nested-CV AUC)
# =============================================================================
def plot_derivation_optimism(clf, X_train_scaled, y_train, oof_y_prob, oof_y_true, model_name, result_dir,
                             nested_cv_auc=None, nested_cv_ci_lower=None, nested_cv_ci_upper=None):
    """Visualize optimism in Derivation AUC (Apparent vs Nested-CV).

    图例中的 Nested-CV AUC 优先取 final_model_summary.csv 报告值（repeat-level
    均值 ± 1.96×重复级 SE）；曲线本身用 04 导出的 repeat 1 代表性 OOF 绘制。
    两者可能相差千分位，图标题已注明曲线来源。

    Args:
        clf: Trained final model.
        X_train_scaled: Scaled training features.
        y_train: Training labels (used for apparent AUC only).
        oof_y_prob: OOF predictions from 04 (same length as y_train).
        oof_y_true: OOF true labels from 04 (same length as oof_y_prob).
        model_name: Model name string.
        result_dir: Output directory.
        nested_cv_auc: Report-level nested-CV AUC (repeat mean); falls back to OOF AUC.
        nested_cv_ci_lower: Lower bound (mean - 1.96*repeat SE).
        nested_cv_ci_upper: Upper bound (mean + 1.96*repeat SE).
    """
    # Length consistency check
    if len(oof_y_prob) != len(oof_y_true):
        raise ValueError("OOF predictions and labels length mismatch")

    # Get apparent predictions (predict entire training set)
    X_df = pd.DataFrame(X_train_scaled)
    apparent_prob = clf.predict_proba(X_df)[:, 1]
    apparent_auc = roc_auc_score(y_train, apparent_prob)

    # OOF (Nested-CV) predictions — use oof_y_true to avoid label mismatch
    fpr_oof, tpr_oof, _ = roc_curve(oof_y_true, oof_y_prob)
    oof_auc = roc_auc_score(oof_y_true, oof_y_prob)
    fpr_app, tpr_app, _ = roc_curve(y_train, apparent_prob)

    # 图例数值优先用报告值（repeat-level 均值），与 final_model_summary.csv 一致；
    # 曲线本身仍来自 repeat 1 代表性 OOF。
    reported_auc = nested_cv_auc if nested_cv_auc is not None else oof_auc
    if nested_cv_ci_lower is not None and nested_cv_ci_upper is not None:
        oof_label = (f"Nested-CV AUC = {reported_auc:.3f}\n"
                     f"(95% CI {nested_cv_ci_lower:.3f}-{nested_cv_ci_upper:.3f}; "
                     f"mean±1.96SE over repeats)")
    else:
        oof_label = f"Nested-CV AUC = {reported_auc:.3f}"

    optimism = apparent_auc - reported_auc
    optimism_pct = optimism * 100

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    # Nested-CV ROC
    ax.plot(fpr_oof, tpr_oof, color="#1F77B4", lw=2.0, linestyle="-",
            label=oof_label)
    # Apparent ROC
    ax.plot(fpr_app, tpr_app, color="#D62728", lw=2.0, linestyle="--",
            label=f"Apparent AUC = {apparent_auc:.3f}")
    # Random classifier
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle="--", alpha=0.5,
            label="Random (AUC = 0.5)")

    # Optimism annotation box (top-left) — removed; reported in Table

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=11)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=11)
    ax.set_title(f"Apparent vs Nested-CV AUC — {model_name}\n"
                 f"(Derivation Set; Nested-CV curve = repeat 1 OOF)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout()
    save_figure(fig, f"derivation_optimism_roc_{model_name}", result_dir)
    print(f"  Derivation Optimism ROC saved.")


# =============================================================================
# Plot 1b: Precision-Recall (PR) Curves
# =============================================================================
def plot_pr_curves(preds_dict, metrics_dict, model_name, result_dir, preds_dict_raw=None):
    """PR curves for all available cohorts with AUPRC and 95% CI in legend.

    PR 曲线对概率校准非平移不变，因此 Derivation 曲线使用 raw apparent 概率
    绘制，与 metrics_dict 中图例 AUPRC 的 raw 口径保持一致；外部队列保持
    calibrated 概率不变。
    """
    # 固定队列顺序的色盲友好配色（与 COLORBLIND_PALETTE 一致）
    colors = {cohort: COLORBLIND_PALETTE.get(cohort, COHORT_FALLBACK_COLORS[i % len(COHORT_FALLBACK_COLORS)])
              for i, cohort in enumerate(preds_dict)}

    fig, ax = plt.subplots(figsize=(7, 7))

    for cohort, (y_true, y_prob) in preds_dict.items():
        # Derivation: 改用 raw apparent 概率（与图例 AUPRC 口径一致）
        if cohort == "Derivation" and preds_dict_raw is not None and cohort in preds_dict_raw:
            y_true, y_prob = preds_dict_raw[cohort]
        # Get AUPRC with CI from metrics_dict
        m = metrics_dict.get(cohort, {})
        auprc = m.get("AUPRC", average_precision_score(y_true, y_prob))
        auprc_lo = m.get("AUPRC_CI_Lower", np.nan)
        auprc_hi = m.get("AUPRC_CI_Upper", np.nan)
        # Derivation 曲线与 AUPRC 均为 raw apparent 口径（PR 曲线对概率平移不敏感，
        # 与 ROC 的 nested-CV 口径不同），图例需显式标注以免与 ROC 图混淆。
        tag = f"{cohort} (apparent)" if cohort == "Derivation" else cohort
        if not np.isnan(auprc_lo) and not np.isnan(auprc_hi):
            label = f"{tag} (AUPRC={auprc:.3f}, {auprc_lo:.3f}-{auprc_hi:.3f})"
        else:
            label = f"{tag} (AUPRC={auprc:.3f})"

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, lw=1.8, color=colors[cohort], label=label)

        # Random baseline = prevalence (positive rate)
        prevalence = np.mean(y_true)
        ax.axhline(y=prevalence, color=colors[cohort], linestyle=":", lw=0.8, alpha=0.4)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {model_name}\n"
                 f"Derivation PR drawn on apparent (in-sample) probabilities; "
                 f"see final_model_summary.csv for nested-CV AUPRC.", fontsize=10)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.3)

    save_figure(fig, f"pr_curves_{model_name}", result_dir)
    print(f"  PR curves saved.")


# =============================================================================
# Plot 2: Calibration Curves (Dynamic Cohorts)
# =============================================================================
def plot_calibration_curves(preds_dict, preds_dict_raw, model_name, hl_results, result_dir):
    """Calibration curves — Raw (dashed) + Recalibrated (solid) for external sets.

    Uses quantile-stratified bins (10 bins) with binomial 95% CI bands.
    For external validation sets, shows both raw (uncalibrated) and recalibrated
    curves. Derivation 主曲线使用 apparent raw 概率（与 HL 检验的 raw 口径一致），
    图例标注 "apparent"；外部队列主曲线保持 calibrated 概率不变。
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    # 固定队列顺序的色盲友好配色（与 COLORBLIND_PALETTE 一致）
    colors = {cohort: COLORBLIND_PALETTE.get(cohort, COHORT_FALLBACK_COLORS[i % len(COHORT_FALLBACK_COLORS)])
              for i, cohort in enumerate(preds_dict)}

    for cohort, (y_true, y_prob) in preds_dict.items():
        # Derivation 主曲线：改用 apparent raw 概率，避免 HL p（raw 口径）与
        # 曲线/Brier（calibrated 口径）混用
        if cohort == "Derivation" and cohort in preds_dict_raw:
            y_true, y_prob = preds_dict_raw[cohort]
        # Fixed 10 bins (quantile strategy); fallback to 8 if n < 100
        if len(y_prob) < 100:
            n_bins = 8
        else:
            n_bins = 10
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="quantile"
        )

        # --- Binomial 95% CI per bin ---
        # (Removed for cleaner visuals; CI shown in Loess supplementary plot)

        hl_p = hl_results.get(cohort, {}).get("p", np.nan)
        brier_val = brier_score_loss(y_true, y_prob)
        # Derivation 为 apparent raw 口径，图例明确标注
        cohort_tag = f"{cohort} (apparent)" if cohort == "Derivation" else cohort
        # 外部队列曲线用 calibrated 概率（Brier 同口径），而 HL 检验固定基于 raw 概率，
        # 图例必须标注 "(raw)"，避免读者误以为两者同一口径。
        hl_tag = "HL p" if cohort == "Derivation" else "HL p(raw)"
        label = (f"{cohort_tag} (Brier={brier_val:.3f}, {hl_tag}={hl_p:.3f})"
                 if not np.isnan(hl_p) else f"{cohort_tag} (Brier={brier_val:.3f})")

        # Recalibrated: solid line with filled markers
        ax.plot(prob_pred, prob_true, "o-", color=colors[cohort], markersize=6,
                lw=1.5, markerfacecolor=colors[cohort], markeredgewidth=1.0,
                label=label)

        # Observed proportion labels for Derivation
        if cohort == "Derivation":
            for i in range(n_bins):
                if not np.isnan(prob_true[i]):
                    ax.text(prob_pred[i], prob_true[i] + 0.02,
                            f'{prob_true[i]:.2f}', fontsize=7, ha='center',
                            color=colors[cohort])

        # Draw raw (uncalibrated) curve for external sets
        if cohort != "Derivation" and cohort in preds_dict_raw:
            y_true_raw, y_prob_raw = preds_dict_raw[cohort]
            prob_true_raw, prob_pred_raw = calibration_curve(
                y_true_raw, y_prob_raw, n_bins=n_bins, strategy="quantile"
            )
            brier_raw = brier_score_loss(y_true_raw, y_prob_raw)
            label_raw = f"{cohort} Raw (Brier={brier_raw:.3f})"
            ax.plot(prob_pred_raw, prob_true_raw, "s--", color=colors[cohort],
                    markersize=5, lw=1.0, markerfacecolor="white",
                    markeredgewidth=1.0, alpha=0.7, label=label_raw)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Ideal")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Observed Proportion", fontsize=12, fontweight="bold")
    ax.set_title(f"Calibration Curves — {model_name}")
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)

    save_figure(fig, f"calibration_curves_{model_name}", result_dir)
    print(f"  Calibration curves (quantile + CI) saved.")


# =============================================================================
# Plot 3: Geographical Calibration Improvement (Raw vs Temporal-Recalibrated)
# =============================================================================
def plot_calibration_improvement(cohort, y_true, y_prob_raw, y_prob_cal,
                                 m_raw, m_cal, hl_p_raw, hl_p_cal,
                                 model_name, result_dir):
    """地理集校准度改善对照图。

    (A) 校准曲线：Raw（虚线空心方块）vs Recalibrated（实线实心圆），
        并标注校准斜率/截距/Brier 的前后变化；
    (B) 预测概率分布迁移：时间集校准器对地理集概率分布的整体平移。
    """
    COL_RAW_LOCAL = "#1f77b4"   # 原始概率（蓝）
    COL_CAL_LOCAL = "#d62728"   # 时间集校准后（红）

    n = len(y_true)
    n_bins = 10 if n >= 100 else 8
    pt_raw, pp_raw = calibration_curve(y_true, y_prob_raw, n_bins=n_bins, strategy="quantile")
    pt_cal, pp_cal = calibration_curve(y_true, y_prob_cal, n_bins=n_bins, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- (A) 校准曲线 ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.55, label="Ideal")
    ax.plot(pp_raw, pt_raw, "s--", color=COL_RAW_LOCAL, markersize=6, lw=1.3,
            markerfacecolor="white", markeredgewidth=1.1,
            label=f"Raw (Brier={m_raw['Brier']:.3f}, HL p={hl_p_raw:.3f})")
    ax.plot(pp_cal, pt_cal, "o-", color=COL_CAL_LOCAL, markersize=6, lw=1.6,
            markerfacecolor=COL_CAL_LOCAL, markeredgewidth=1.0,
            label=f"Recalibrated (Brier={m_cal['Brier']:.3f}, HL p={hl_p_cal:.3f})")
    txt = (f"Calibration slope: {m_raw['Calib_Slope']:.2f} \u2192 {m_cal['Calib_Slope']:.2f}\n"
           f"Calibration intercept: {m_raw['Calib_Intercept']:.2f} \u2192 {m_cal['Calib_Intercept']:.2f}\n"
           f"Brier score: {m_raw['Brier']:.3f} \u2192 {m_cal['Brier']:.3f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted Probability", fontweight="bold")
    ax.set_ylabel("Observed Proportion", fontweight="bold")
    ax.set_title(f"(A) Calibration Curve — {cohort}")
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.3)

    # --- (B) 预测概率分布迁移 ---
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    mean_raw = float(np.mean(y_prob_raw))
    mean_cal = float(np.mean(y_prob_cal))
    ax.hist(y_prob_raw, bins=bins, density=True, alpha=0.45, color=COL_RAW_LOCAL,
            edgecolor=COL_RAW_LOCAL, linewidth=0.8,
            label=f"Raw (mean={mean_raw:.3f})")
    ax.hist(y_prob_cal, bins=bins, density=True, alpha=0.45, color=COL_CAL_LOCAL,
            edgecolor=COL_CAL_LOCAL, linewidth=0.8,
            label=f"Recalibrated (mean={mean_cal:.3f})")
    ax.axvline(mean_raw, color=COL_RAW_LOCAL, ls="--", lw=1.4)
    ax.axvline(mean_cal, color=COL_CAL_LOCAL, ls="--", lw=1.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted Probability", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title(f"(B) Predicted Probability Shift — {cohort}")
    ax.legend(loc="upper center", fontsize=8.5, frameon=True, framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Calibration Improvement after Temporal Recalibration — "
                 f"{cohort} Cohort ({model_name})", fontsize=FONT_TITLE + 1)
    fig.tight_layout()
    save_figure(fig, f"Geographical_Calibration_Improvement_{model_name}", result_dir)
    print(f"  Geographical calibration improvement figure saved.")


# =============================================================================
# Plot 4: SHAP Summary
# =============================================================================
def plot_shap(clf, X_train, feature_names, model_name, result_dir):
    """SHAP bee-swarm or bar plot."""
    try:
        # Use full dataset for formal run
        X_sample = X_train

        if hasattr(clf, "predict_proba"):
            if model_name == "L2_LR":
                # LinearExplainer: exact SHAP values for linear models
                explainer = shap.LinearExplainer(clf, X_train, feature_perturbation="interventional")
                shap_values = explainer.shap_values(X_sample)
            elif model_name in ("RF", "LightGBM", "CatBoost", "DT", "XGBoost"):
                # TreeExplainer for tree-based models
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    # 新版 SHAP 对二分类树模型返回 (n, n_features, 2) 3D 数组，取正类
                    shap_values = shap_values[:, :, 1]
            else:
                # KernelExplainer fallback for other models (SVM, KNN, GaussianNB)
                background = shap.kmeans(X_train, min(50, X_train.shape[0]))
                explainer = shap.KernelExplainer(
                    lambda x: clf.predict_proba(x)[:, 1], background
                )
                shap_values = explainer.shap_values(X_sample, nsamples=100)
        elif hasattr(clf, "decision_function"):
            explainer = shap.KernelExplainer(clf.decision_function, shap.kmeans(X_train, min(50, X_train.shape[0])))
            shap_values = explainer.shap_values(X_sample, nsamples=100)
        else:
            print(f"  SHAP: model type not supported for SHAP.")
            return

        # ===== Cache SHAP importance data for reuse by plot_combined_shap =====
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order_idx = np.argsort(mean_abs)  # ascending: least→most important
        ordered_features = [feature_names[i] for i in order_idx]
        shap_cache = {
            "feature_names": feature_names,
            "shap_values": shap_values,
            "mean_abs": mean_abs,
            "order_idx": order_idx,
            "ordered_features": ordered_features,
        }
        cache_path = os.path.join(result_dir, f"shap_importance_{model_name}.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(shap_cache, f)
        print(f"  SHAP importance cached ({len(ordered_features)} features, ascending order).")

        # Bar plot (supplementary)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          plot_type="bar", show=False)
        fig_bar = plt.gcf()
        fig_bar.set_size_inches(8, 6)
        if fig_bar.axes:
            fig_bar.axes[0].set_title(f"SHAP Feature Importance (Bar) — {model_name}")
        save_figure(fig_bar, f"shap_summary_bar_{model_name}", result_dir)

        # Beeswarm plot (main)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          plot_type="dot", show=False)
        fig_dot = plt.gcf()
        fig_dot.set_size_inches(10, 7)
        if fig_dot.axes:
            fig_dot.axes[0].set_title(f"SHAP Feature Importance (Beeswarm) — {model_name}")
        save_figure(fig_dot, f"shap_summary_beeswarm_{model_name}", result_dir)
        print(f"  SHAP summary saved (bar + beeswarm).")

        # Dependence plots for top 6 features (Mixed Data Type Support)
        try:
            # Identify top 6 features by mean absolute SHAP value (using cached mean_abs)
            top6_idx = np.argsort(mean_abs)[-6:][::-1]

            for rank, feat_idx in enumerate(top6_idx):
                feature_name = feature_names[feat_idx]
                feature_val = X_sample[:, feat_idx]
                shap_val = shap_values[:, feat_idx]

                # --- Determine feature type: ≤2 unique values → binary ---
                unique_vals = np.unique(feature_val)
                is_binary = len(unique_vals) <= 2

                # Create canvas
                fig, ax = plt.subplots(figsize=(7, 5))

                if is_binary:
                    # ===== Binary feature: violin plot =====
                    clean_label = feature_name.replace('_', ' ').title()
                    # Build a DataFrame for seaborn
                    df_violin = pd.DataFrame({
                        "SHAP": shap_val,
                        "Category": np.where(feature_val == 0,
                                             f"{clean_label} = 0",
                                             f"{clean_label} = 1")
                    })
                    sns.violinplot(
                        data=df_violin, x="Category", y="SHAP",
                        palette={"0": "#1f77b4", "1": "#d62728"}.values(),
                        inner="quartile", cut=0, ax=ax
                    )
                    # Add colorbar for symmetry with continuous features
                    vmin_bin = np.nanpercentile(X_sample.ravel(), 2.5)
                    vmax_bin = np.nanpercentile(X_sample.ravel(), 97.5)
                    sm_bin = plt.cm.ScalarMappable(cmap="coolwarm",
                                                   norm=plt.Normalize(vmin_bin, vmax_bin))
                    sm_bin.set_array([])
                    cbar_bin = plt.colorbar(sm_bin, ax=ax)
                    ax.set_ylabel("SHAP value", fontsize=12)
                    ax.set_xlabel("")

                else:
                    # ===== Continuous feature handling (standard scatter) =====
                    scatter = ax.scatter(
                        feature_val, shap_val,
                        c=feature_val, cmap='coolwarm',
                        alpha=0.7, s=20, edgecolors='none'
                    )
                    ax.set_xlabel(feature_name, fontsize=12)
                    cbar = plt.colorbar(scatter, ax=ax)

                # ----- Common styling (independent of feature type) -----
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
                ax.set_ylabel("SHAP value", fontsize=12)
                ax.set_title(
                    f"SHAP Dependence — {feature_name} (Rank #{rank+1}) — {model_name}",
                    fontsize=13, fontweight='bold'
                )
                ax.grid(True, alpha=0.2, linestyle='--')
                ax.set_axisbelow(True)

                save_figure(fig, f"shap_dependence_{model_name}_rank{rank+1}", result_dir)

            print(f"  SHAP dependence plots saved (top 6 features, mixed types).")
        except Exception as e:
            print(f"  SHAP dependence plots failed: {e}")

    except Exception as e:
        print(f"  SHAP failed: {e}")


# =============================================================================
# Plot 5: Feature Importance / OR Forest Plot
# =============================================================================
def plot_feature_importance(clf, feature_names, model_name, X_train, y_train, result_dir):
    """Feature importance: OR forest plot for L2_LR only; others use SHAP."""

    # =========================================================================
    # 仅对 L2_LR 生成特征重要性图（OR forest plot）
    # 其他模型已由 SHAP 覆盖（包括 XGBoost 作为树模型）
    # =========================================================================
    if model_name != "L2_LR":
        print(f"  Feature importance: Skipped for {model_name} (already covered by SHAP).")
        return

    # --- OR forest plot for L2_LR (clinical standard) ---
    coef = clf.coef_.ravel() if hasattr(clf, "coef_") else None
    if coef is None:
        print(f"  Feature importance: no coefficients available.")
        return

    odds_ratios = np.exp(coef)
    # Bootstrap CI for ORs
    n_boot = BOOTSTRAP_ITER
    rng = np.random.RandomState(RANDOM_STATE)
    n = X_train.shape[0]
    or_boot = np.zeros((n_boot, len(feature_names)))
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            clf_boot = LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=5000,
                C=clf.C, random_state=RANDOM_STATE
            )
            clf_boot.fit(X_train[idx], y_train[idx])
            if np.any(np.abs(clf_boot.coef_) > 10):  # complete separation
                or_boot[b] = np.nan
            else:
                or_boot[b] = np.exp(clf_boot.coef_.ravel())
        except Exception:
            or_boot[b] = np.nan

    or_lo = np.nanpercentile(or_boot, 2.5, axis=0)
    or_hi = np.nanpercentile(or_boot, 97.5, axis=0)

    # Sort by OR magnitude
    sort_idx = np.argsort(np.abs(odds_ratios - 1))[::-1]
    odds_ratios = odds_ratios[sort_idx]
    or_lo = or_lo[sort_idx]
    or_hi = or_hi[sort_idx]
    feat_sorted = np.array(feature_names)[sort_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = range(len(feat_sorted))
    ax.errorbar(odds_ratios, y_pos, xerr=[odds_ratios - or_lo, or_hi - odds_ratios],
                fmt="o", capsize=4, color="steelblue", ecolor="gray", markersize=6)
    ax.axvline(x=1.0, color="red", linestyle="--", lw=1.0, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_sorted)
    ax.set_xlabel("Odds Ratio (95% CI)\nOR per 1 IQR increase (features are RobustScaler-transformed)",
                  fontsize=10)
    ax.set_title(f"Odds Ratios — {model_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, f"odds_ratios_forest_{model_name}", result_dir)
    print(f"  OR forest plot saved.")


# =============================================================================
# Plot 6: Radar Chart (Dynamic Cohorts)
# =============================================================================
def plot_radar_chart(metrics_dict, model_name, result_dir, title_prefix="AB_Score"):
    """Enhanced radar chart: 5 metrics across 3 cohorts with value labels at vertices.

    Parameters:
        title_prefix: "AB_Score" or "AUC" — determines chart title and filename.
    """
    metrics_list = ["AUC", "1-Brier", "Sensitivity", "Specificity", "F1"]
    available_cohorts = [c for c in ["Derivation", "Temporal", "Geographical"]
                         if c in metrics_dict and metrics_dict[c].get("AUC", 0) > 0]
    # 色盲友好配色：固定队列顺序映射（与全脚本 COLORBLIND_PALETTE 一致）
    colors = {c: COLORBLIND_PALETTE.get(c, COHORT_FALLBACK_COLORS[i % len(COHORT_FALLBACK_COLORS)])
              for i, c in enumerate(available_cohorts)}

    n_metrics = len(metrics_list)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

    for cohort in available_cohorts:
        m = metrics_dict[cohort]
        # Build values: use 1-Brier for calibration dimension
        values = [
            m.get("AUC", 0),
            1 - m.get("Brier", 0),
            m.get("Sensitivity", 0),
            m.get("Specificity", 0),
            m.get("F1", 0),
        ]
        values += values[:1]
        ax.fill(angles, values, alpha=0.12, color=colors[cohort])
        ax.plot(angles, values, "o-", color=colors[cohort], lw=2.0,
                markersize=6, label=cohort)
        # Annotate specific values at vertices
        for j, (angle, val) in enumerate(zip(angles[:-1], values[:-1])):
            ax.annotate(f"{val:.2f}", xy=(angle, val),
                        fontsize=7, fontweight="bold", color=colors[cohort],
                        ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_list, fontsize=FONT_LABEL - 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Optimal Model (Selected by {title_prefix}) — {model_name}",
                 fontsize=FONT_TITLE, pad=25, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=FONT_LEGEND)
    ax.grid(True, alpha=0.3)

    save_figure(fig, f"Radar_{title_prefix}_Optimal_Model", result_dir)
    print(f"  Radar chart ({title_prefix}) saved.")


# =============================================================================
# Plot 7b: Combined SHAP (Beeswarm + Bar + Dependence 6x2x3)
# =============================================================================
def plot_combined_shap(model, X_train_scaled, X_plot, y_plot, feature_names,
                       model_name, result_dir, scaler=None):
    """Combined SHAP: beeswarm + bar (left) + 6 dependence plots (right, 2x3 grid).

    左面板为 bar + beeswarm 叠加（基于 plot_shap 缓存的 SHAP 值重绘）；
    右面板 Top 6 dependence 图同样在本函数内重绘（不读取预存 rank1-6 PNG）。
    """
    try:
        # ===== Step 1: Load cached SHAP data from plot_shap =====
        shap_cache_path = os.path.join(result_dir, f"shap_importance_{model_name}.pkl")
        if not os.path.exists(shap_cache_path):
            print(f"  Combined SHAP: cached SHAP data not found, skipping.")
            return

        with open(shap_cache_path, "rb") as f:
            cached = pickle.load(f)

        X_plot_arr = np.asarray(X_plot)
        shap_values = cached["shap_values"]
        ordered_features = cached["ordered_features"]  # ascending: least→most important
        feature_names_cached = cached["feature_names"]
        mean_abs_cached = cached["mean_abs"]
        feat_to_imp = dict(zip(feature_names_cached, mean_abs_cached))
        imp_df = pd.DataFrame({
            "feature": ordered_features,
            "importance": [feat_to_imp[f] for f in ordered_features],
        })

        # ===== Step 2: Extract beeswarm point layout from native shap =====
        beeswarm_collections = []
        fig_tmp = plt.figure(figsize=(4, 4))
        try:
            shap.summary_plot(shap_values, X_plot_arr, feature_names=feature_names,
                              plot_type="dot", show=False, max_display=len(feature_names))
            ax_tmp = fig_tmp.axes[0] if fig_tmp.axes else None
            if ax_tmp is not None:
                for coll in ax_tmp.collections:
                    offsets = coll.get_offsets()
                    if len(offsets) == 0:
                        continue
                    fc = coll.get_facecolors()
                    sizes = coll.get_sizes()
                    if len(sizes) == 1:
                        sizes = np.full(len(offsets), sizes[0])
                    beeswarm_collections.append((offsets, fc, sizes))
                ytick_labels = [t.get_text() for t in ax_tmp.get_yticklabels()]
                shap_native_order = [t for t in ytick_labels if t in feature_names]
                if len(shap_native_order) == len(feature_names):
                    rank_map = {f: i for i, f in enumerate(shap_native_order)}
                    imp_df["rank"] = imp_df["feature"].map(rank_map)
                    imp_df = imp_df.sort_values("rank").drop(columns=["rank"]).reset_index(drop=True)
            print(f"  Extracted {len(beeswarm_collections)} beeswarm collections")
        except Exception as e:
            print(f"  WARNING: shap beeswarm extraction failed ({e})")
        finally:
            plt.close(fig_tmp)

        # ===== Step 3: Create figure with left/right panels =====
        fig = plt.figure(figsize=(20, 9))
        gs = gridspec.GridSpec(1, 2, width_ratios=[0.42, 0.58], figure=fig)

        # ===== Left panel: bar + beeswarm overlay =====
        ax_left = fig.add_subplot(gs[0])
        n_feat = len(imp_df)
        y_pos = np.arange(n_feat)

        # Dual-axis: bar on top axis
        ax_top = ax_left.twiny()
        ax_top.barh(y_pos, imp_df["importance"].values, color="gray",
                    alpha=0.25, height=0.6, edgecolor="none", zorder=1)
        ax_top.xaxis.set_ticks_position("top")
        ax_top.xaxis.set_label_position("top")
        ax_top.set_xlabel("Feature Importance (Mean |SHAP|)", fontsize=10)
        ax_top.set_yticklabels([])

        # Global percentile range for color consistency
        vmin = np.nanpercentile(X_plot_arr.ravel(), 2.5)
        vmax = np.nanpercentile(X_plot_arr.ravel(), 97.5)

        # Beeswarm scatter on main axis
        if beeswarm_collections:
            for i, (offsets, fc, sizes) in enumerate(beeswarm_collections):
                feat_name = imp_df["feature"].iloc[i]
                feat_idx = feature_names.index(feat_name)
                feat_vals = X_plot_arr[:, feat_idx]
                feat_shap = shap_values[:, feat_idx]
                sort_order = np.argsort(feat_shap)
                feat_vals_sorted = feat_vals[sort_order]
                ax_left.scatter(offsets[:, 0], offsets[:, 1],
                                c=feat_vals_sorted, cmap="coolwarm",
                                s=20, alpha=1.0, edgecolors="none",
                                vmin=vmin, vmax=vmax, zorder=3)
        else:
            # Fallback: simple scatter with global coolwarm
            print("  WARNING: No beeswarm data extracted, using fallback scatter")
            ax_left.scatter(shap_values.ravel(),
                            np.tile(np.arange(len(feature_names)), shap_values.shape[0]),
                            c=X_plot_arr.ravel(), cmap="coolwarm", s=20, alpha=1.0,
                            edgecolors="none", vmin=vmin, vmax=vmax, zorder=3)

        ax_left.axvline(x=0, color="black", linestyle="--", lw=0.8, zorder=2)
        ax_left.set_yticks(np.arange(n_feat))
        ax_left.set_yticklabels(imp_df["feature"].values, fontsize=10, fontweight="bold")
        ax_left.set_xlabel("SHAP value", fontsize=10)
        ax_left.set_ylim(-0.5, n_feat - 0.5)
        ax_left.grid(axis="x", alpha=0.2, linestyle="--")
        ax_left.set_title("(a) Feature Importance + Beeswarm", fontsize=12,
                          fontweight="bold", loc="left")

        # Colorbar for left panel
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin, vmax))
        sm.set_array([])
        cbar_left = fig.colorbar(sm, ax=ax_left, shrink=0.7, pad=0.02)
        cbar_left.set_label("Feature value", fontsize=9)
        cbar_left.ax.tick_params(labelsize=8)

        # ===== Right panel: 2x3 grid of dependence plots (re-drawn) =====
        # Restore clinical units if scaler is available
        if scaler is not None:
            X_plot_raw = scaler.inverse_transform(X_plot_arr)
        else:
            X_plot_raw = X_plot_arr
        df_plot = pd.DataFrame(X_plot_raw, columns=feature_names)

        # imp_df is ascending (least→most important), tail(6) reversed = most important first
        top6 = imp_df.iloc[-6:].iloc[::-1].reset_index(drop=True)

        gs_right = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1],
                                                     hspace=0.18, wspace=0.22)

        for rank in range(len(top6)):
            ax = fig.add_subplot(gs_right[rank // 3, rank % 3])
            feat_name = top6["feature"].iloc[rank]
            feat_idx = feature_names.index(feat_name)
            feat_vals = df_plot[feat_name].values
            feat_shap = shap_values[:, feat_idx]
            n_unique = len(np.unique(feat_vals))

            if n_unique <= 2:
                # Binary feature: violin plot
                clean_label = feat_name.replace('_', ' ').title()
                cat0 = "No"
                cat1 = "Yes"
                plot_df_violin = pd.DataFrame({
                    "SHAP": feat_shap,
                    "Category": np.where(feat_vals == 0, cat0, cat1)
                })
                palette_violin = {cat0: "#1f77b4", cat1: "#d62728"}
                sns.violinplot(data=plot_df_violin, x="Category", y="SHAP",
                               order=[cat0, cat1],
                               palette=palette_violin, inner="quartile", density_norm="width",
                               ax=ax, linewidth=1.0)
                # Colorbar for symmetry
                sm_sub = plt.cm.ScalarMappable(cmap="coolwarm",
                                               norm=plt.Normalize(vmin, vmax))
                sm_sub.set_array([])
                cbar_sub = fig.colorbar(sm_sub, ax=ax, shrink=0.8, pad=0.02)
                cbar_sub.ax.tick_params(labelsize=5)
                ax.set_xlabel("")
                ax.set_ylabel("SHAP", fontsize=7)
                ax.tick_params(labelsize=6)
            else:
                # Continuous feature: scatter + LOWESS with coolwarm gradient
                scatter = ax.scatter(feat_vals, feat_shap, c=feat_vals,
                                     cmap="coolwarm", s=8, alpha=0.5,
                                     edgecolors="none", vmin=vmin, vmax=vmax)
                sort_idx = np.argsort(feat_vals)
                x_sorted = feat_vals[sort_idx]
                y_sorted = feat_shap[sort_idx]
                lowess_res = lowess(y_sorted, x_sorted, frac=0.3, it=3, return_sorted=True)
                ax.plot(lowess_res[:, 0], lowess_res[:, 1], color="dimgray", lw=2.0)
                cbar_sub = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
                cbar_sub.ax.tick_params(labelsize=5)
                ax.set_xlabel("")
                ax.set_ylabel("SHAP", fontsize=7)
                ax.tick_params(labelsize=6)

            # Common styling
            ax.axhline(y=0, color="gray", linestyle="--", lw=0.6, alpha=0.4)
            ax.set_title(feat_name, fontsize=8, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, linestyle="--")

        fig.text(0.525, 0.97, "(b) SHAP Dependence (Top 6)", fontsize=12,
                 fontweight="bold", ha="center")

        save_figure(fig, f"combined_shap_{model_name}", result_dir)
        print(f"  Combined SHAP saved.")

    except Exception as e:
        print(f"  Combined SHAP failed: {e}")
        traceback.print_exc()  # traceback 已在模块顶部导入


# =============================================================================
# Plot 8: Confusion Matrices
# =============================================================================
def plot_confusion_matrices(preds_dict, model_name, cutoff, result_dir, nested_cv_df=None,
                            cutoff_raw=None):
    """Dynamic confusion matrix heatmaps. Derivation uses nested-CV predictions if available.

    cutoff      : cutoff for calibrated probabilities (external cohorts).
                  可为标量，或 {cohort: cutoff} 字典（各地理/时间集校准器映射的截断值不同时）。
    cutoff_raw  : cutoff for raw probabilities (Derivation nested-CV OOF probs).
                  If None, falls back to cutoff (raw==calibrated scale assumed).
    """
    cohorts_in_order = ["Derivation"] + [c for c in preds_dict if c != "Derivation"]
    n_coh = len(cohorts_in_order)
    fig, axes = plt.subplots(1, n_coh, figsize=(4.5 * n_coh, 4.5))
    if n_coh == 1:
        axes = [axes]

    for i, cohort in enumerate(cohorts_in_order):
        if cohort == "Derivation" and nested_cv_df is not None and len(nested_cv_df) > 0:
            # Derivation panel uses RAW nested-CV OOF probabilities → apply the raw cutoff
            y_true = nested_cv_df["y_true"].values
            y_prob = nested_cv_df["y_pred_prob"].values
            cuf = cutoff_raw if cutoff_raw is not None else cutoff
        elif cohort in preds_dict:
            y_true, y_prob = preds_dict[cohort]
            cuf = cutoff.get(cohort, np.nan) if isinstance(cutoff, dict) else cutoff
        else:
            continue
        y_pred = (y_prob >= cuf).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    cbar=False, annot_kws={"size": FONT_LABEL})
        axes[i].set_title(f"{cohort}\n(cutoff={cuf:.3f})", fontsize=FONT_TITLE)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    fig.suptitle(f"Confusion Matrices — {model_name}\n"
                 f"Derivation: raw cutoff={cutoff_raw:.3f}; "
                 f"Temporal/Geographical: recalibrated cutoff",
                 fontsize=FONT_TITLE + 1)
    fig.tight_layout()
    save_figure(fig, f"confusion_matrices_{model_name}", result_dir)
    print(f"  Confusion matrices saved.")


# =============================================================================
# Supplementary Figure: AUPRC Relative Gain Bar Plot
# =============================================================================
def plot_auprc_gain_bar(metrics_dict, preds_dict, model_name, result_dir):
    """Relative gain bar plot comparing model AUPRC vs random baseline (prevalence).

    Displays gain multiplier (AUPRC / prevalence) above each model bar.
    Enhanced with high-contrast colors, arrow annotations, and grid lines.
    """
    cohort_order = ["Derivation", "Temporal", "Geographical"]
    available = [c for c in cohort_order if c in metrics_dict and "AUPRC" in metrics_dict[c]]

    # Compute prevalence (random baseline) from preds_dict
    prevalence = {}
    for c in available:
        if c in preds_dict:
            y_true, _ = preds_dict[c]
            prevalence[c] = np.mean(y_true)
        else:
            prevalence[c] = 0.5

    model_auprc = {c: metrics_dict[c]["AUPRC"] for c in available}
    auprc_lo = {c: metrics_dict[c].get("AUPRC_CI_Lower", np.nan) for c in available}
    auprc_hi = {c: metrics_dict[c].get("AUPRC_CI_Upper", np.nan) for c in available}

    # Compute relative gain (AUPRC / prevalence)
    gain = {c: model_auprc[c] / prevalence[c] for c in available}

    x = np.arange(len(available))
    width = 0.35

    # 高对比度配色：从全局 COLORBLIND_PALETTE 取色（与全脚本一致），
    # 未知队列回退 COHORT_FALLBACK_COLORS
    cohort_colors = {c: COLORBLIND_PALETTE.get(c, COHORT_FALLBACK_COLORS[i % len(COHORT_FALLBACK_COLORS)])
                     for i, c in enumerate(available)}
    baseline_color = "#D3D3D3"     # light gray

    bars_baseline = [prevalence[c] for c in available]
    bars_model = [model_auprc[c] for c in available]
    colors_model = [cohort_colors.get(c, "#333333") for c in available]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Baseline bars (light gray with border)
    ax.bar(x - width / 2, bars_baseline, width, color=baseline_color, edgecolor="#A0A0A0",
           linewidth=1.0, label="Random Baseline (Prevalence)", zorder=2)

    # Model bars (saturated color, black border)
    ax.bar(x + width / 2, bars_model, width, color=colors_model, edgecolor="black",
           linewidth=1.0, label="Model AUPRC", zorder=3)

    # CI error bars
    yerr_lo = [model_auprc[c] - auprc_lo[c] for c in available]
    yerr_hi = [auprc_hi[c] - model_auprc[c] for c in available]
    ax.errorbar(x + width / 2, bars_model,
                yerr=[yerr_lo, yerr_hi],
                fmt="none", ecolor="black", capsize=5, capthick=1.2, lw=1.2, zorder=4)

    # --- Annotate values and gain multipliers ---
    # Two-line structure: AUPRC value (top) + gain multiplier (bottom)
    # Geo: red bold large — visual focal point; Temporal: deliberately subdued
    gain_style = {
        "Derivation":   {"color": "black",   "size": 10, "weight": "normal"},
        "Temporal":     {"color": "#666666", "size": 9,  "weight": "normal"},
        "Geographical": {"color": "#CC0000", "size": 14, "weight": "bold"},
    }
    for i, c in enumerate(available):
        val = model_auprc[c]
        gain_val = gain[c]
        gs = gain_style.get(c, {"color": "black", "size": 10, "weight": "normal"})
        # AUPRC value (first line) — black bold for all
        ax.text(x[i] + width / 2, val + 0.05, f"{val:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="black", zorder=5)
        # Gain multiplier (second line) — cohort-specific style
        ax.text(x[i] + width / 2, val + 0.02, f"{gain_val:.2f}\u00d7",
                ha="center", va="bottom", fontsize=gs["size"],
                fontweight=gs["weight"], color=gs["color"], zorder=5)
        # Baseline prevalence value above gray bar
        base = prevalence[c]
        ax.text(x[i] - width / 2, base + 0.03, f"{base:.3f}",
                ha="center", va="bottom", fontsize=9, color="#555555",
                fontweight="bold", zorder=5)

    # Axis labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(available, fontsize=FONT_LABEL + 1, fontweight="bold")
    ax.set_ylabel("AUPRC (Average Precision)", fontsize=FONT_LABEL)
    ax.set_title(f"AUPRC Gain Over Random Baseline\n{model_name}",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Conclusion text box — 右上角，动态标注增益最高的队列
    best_cohort = max(gain, key=gain.get)
    conclusion_text = f"Highest gain: {best_cohort} ({gain[best_cohort]:.2f}×)"
    ax.text(0.98, 0.95, conclusion_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold",
            color="#CC0000", zorder=6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CC0000", alpha=0.9))

    fig.tight_layout()
    save_figure(fig, f"auprc_gain_bar_{model_name}", result_dir)
    print(f"  AUPRC gain bar plot saved (gain values displayed).")


# =============================================================================
# Brier Skill Score Plot
# =============================================================================
def plot_brier_skill_score(metrics_dict, preds_dict, model_name, result_dir, preds_dict_raw=None):
    """Horizontal dot plot of Brier Skill Score with 95% CI for 3 cohorts.

    BSS = 1 - (Model Brier / Null Brier), where Null Brier = prevalence * (1 - prevalence).
    BSS > 0 means the model improves upon random guessing.
    Bootstrapped 95% CI reflects sampling uncertainty.

    Derivation 点估计取 metrics_dict 的 raw apparent Brier，故其 bootstrap 也必须
    用 raw 概率（preds_dict_raw），否则 CI 会围绕一个不同口径的 Brier 展开。
    """
    cohort_order = ["Derivation", "Temporal", "Geographical"]
    available = [c for c in cohort_order if c in metrics_dict and "Brier" in metrics_dict[c]
                 and c in preds_dict]

    if not available:
        print("  Brier Skill Score plot: no valid cohorts, skipping.")
        return

    # Compute BSS and bootstrap CI for each cohort
    rng = np.random.RandomState(RANDOM_STATE)
    n_bs = 1000
    bss_vals = {}
    bss_ci_lo = {}
    bss_ci_hi = {}
    prevalence_vals = {}
    brier_vals = {}

    for cohort in available:
        y_true, y_prob = preds_dict[cohort]
        # Derivation 的 Brier 点估计为 raw apparent 口径 → bootstrap 也用 raw 概率
        if cohort == "Derivation" and preds_dict_raw is not None and cohort in preds_dict_raw:
            y_true, y_prob = preds_dict_raw[cohort]
        n = len(y_true)
        prevalence = np.mean(y_true)
        prevalence_vals[cohort] = prevalence
        null_brier = prevalence * (1 - prevalence)
        model_brier = metrics_dict[cohort]["Brier"]
        brier_vals[cohort] = model_brier

        bss_point = 1 - (model_brier / null_brier) if null_brier > 0 else np.nan
        bss_vals[cohort] = bss_point

        # Bootstrap 95% CI for BSS
        # Protection: if null_b == 0 (e.g., extreme prevalence 0% or 100% in
        # a bootstrap sample), BSS is undefined — set to NaN and skip this iter.
        bs_bss = np.empty(n_bs)
        for b in range(n_bs):
            idx = rng.choice(n, size=n, replace=True)
            y_true_b = y_true[idx]
            y_prob_b = y_prob[idx]
            p_b = np.mean(y_true_b)
            null_b = p_b * (1 - p_b)
            if null_b == 0:
                # Null_Brier = 0 means extreme prevalence; BSS is undefined
                bs_bss[b] = np.nan
                continue
            brier_b = brier_score_loss(y_true_b, y_prob_b)
            bs_bss[b] = 1 - (brier_b / null_b)
        bs_bss = bs_bss[~np.isnan(bs_bss)]
        # Protection: if fewer than 50 valid bootstrap iterations, CI is unreliable
        if len(bs_bss) >= 50:
            bss_ci_lo[cohort] = np.percentile(bs_bss, 2.5)
            bss_ci_hi[cohort] = np.percentile(bs_bss, 97.5)
        else:
            bss_ci_lo[cohort] = np.nan
            bss_ci_hi[cohort] = np.nan
            print(f"  WARNING: {cohort} has only {len(bs_bss)} valid bootstrap iters, "
                  f"BSS CI set to NaN.")

    # Plot horizontal dot plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(available))
    # 从全局 COLORBLIND_PALETTE 取色（与全脚本一致），未知队列回退 COHORT_FALLBACK_COLORS
    colors = {c: COLORBLIND_PALETTE.get(c, COHORT_FALLBACK_COLORS[i % len(COHORT_FALLBACK_COLORS)])
              for i, c in enumerate(available)}

    for i, cohort in enumerate(available):
        bss_val = bss_vals[cohort]
        lo = bss_ci_lo.get(cohort, np.nan)
        hi = bss_ci_hi.get(cohort, np.nan)
        color = colors.get(cohort, "#333333")
        # Error bar
        if not np.isnan(lo) and not np.isnan(hi):
            ax.errorbar(bss_val, i, xerr=[[bss_val - lo], [hi - bss_val]],
                        fmt="o", color=color, ecolor=color, capsize=5, capthick=1.5,
                        lw=2.0, markersize=12, zorder=3)
        else:
            ax.scatter(bss_val, i, color=color, s=120, zorder=3, edgecolors="white", linewidth=1.0)

        # Annotate: BSS value + Brier in parentheses + prevalence below
        prev_pct = prevalence_vals[cohort] * 100
        ann_text = f"BSS = {bss_val:.3f}  (Brier = {brier_vals[cohort]:.3f})"
        ax.text(bss_val + 0.03, i + 0.12, ann_text,
                fontsize=9, fontweight="bold", color=color, va="bottom",
                fontfamily="monospace")
        ax.text(bss_val + 0.03, i - 0.12, f"Prevalence = {prev_pct:.1f}%",
                fontsize=7.5, color="gray", va="top", fontfamily="monospace")

    # Vertical reference line at BSS = 0 (no predictive value)
    ax.axvline(x=0, color="black", linestyle="--", lw=1.5, alpha=0.7,
               label="No predictive value (BSS = 0)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(available, fontsize=11, fontweight="bold")
    ax.set_xlabel("Brier Skill Score (BSS)", fontsize=12, fontweight="bold")
    # Title removed to keep the figure clean (cohort names serve as labels)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9, edgecolor="none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    # Auto-adjust x-axis limits with padding
    all_bss = [v for v in bss_vals.values() if not np.isnan(v)]
    if all_bss:
        x_min = min(all_bss + [0])
        x_max = max(all_bss + [0])
        x_pad = (x_max - x_min) * 0.15 if (x_max - x_min) > 0.01 else 0.15
        ax.set_xlim(x_min - x_pad, x_max + x_pad + 0.25)

    fig.tight_layout()
    save_figure(fig, f"brier_skill_score_{model_name}", result_dir)
    print(f"  Brier Skill Score plot saved.")


# =============================================================================
# Individual SHAP Force (Waterfall) — 2 cases (lowest / highest predicted probability)
# =============================================================================
def plot_individual_shap_force(clf, X_train, feature_names, model_name,
                                result_dir):
    """Individual SHAP waterfall plots for 2 representative cases.

    Layout: 1 row x 2 columns. 病例按最终模型的预测概率中位数二分抽样，
    不代表任何临床风险分层：
      - Left  : 预测概率最低的 1 例
      - Right : 预测概率最高的 1 例

    Parameters
    ----------
    clf : trained model
    X_train : ndarray, Derivation cohort features
    feature_names : list of str
    model_name : str
    result_dir : str
    """
    import joblib
    
    print(f"\n  [Individual SHAP] Creating SHAP explainer for {model_name}...")
    
    # ===== 加载预处理对象，用于反标准化 =====
    # 01_data_preprocessing.py 保存的 preprocessing_objects.joblib 内含自定义类
    # ScaledImputer（imputer 变量），由模块顶部的 stub 支持反序列化。
    # 其结构为：
    #   scaler_robust_params: {原始列名: {'median': ..., 'iqr': ...}}（RobustScaler 参数）
    #   transformed_log_cols : log1p 变换过的原始列名（反标准化后需 expm1 还原）
    # 注意：脚本内不存在 sklearn 的 scaler 对象，需手动按 median/IQR 还原。
    scaler_params = None
    log_cols = set()
    orig_name_by_r = {}
    try:
        preprocessing_path = os.path.join(BASE_DIR, "data_preprocessing",
                                          "preprocessing_objects.joblib")
        if os.path.exists(preprocessing_path):
            preprocessing_objects = joblib.load(preprocessing_path)
            scaler_params = preprocessing_objects.get("scaler_robust_params")
            log_cols = set(preprocessing_objects.get("transformed_log_cols", []))
            final_names = preprocessing_objects.get("final_feature_names", [])
            # R 列名 -> 原始列名（如 'mean.arterial.pressure' -> 'mean arterial pressure'）
            orig_name_by_r = {_r_make_names(n): n for n in final_names}
            if scaler_params is not None:
                print(f"  [Individual SHAP] Loaded scaler params "
                      f"({len(scaler_params)} features) for inverse transformation")
            else:
                print("  [Individual SHAP] Warning: no scaler_robust_params in "
                      "preprocessing_objects, using standardized values")
        else:
            print(f"  [Individual SHAP] Warning: preprocessing_objects.joblib not "
                  f"found, using standardized values")
    except Exception as e:
        print(f"  [Individual SHAP] Warning: Failed to load preprocessing objects: "
              f"{e}, using standardized values")
        scaler_params = None

    # Step 1: Create SHAP explainer (same logic as plot_shap)
    X_sample = X_train
    if hasattr(clf, "predict_proba"):
        if model_name == "L2_LR":
            explainer = shap.LinearExplainer(clf, X_train,
                                             feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_sample)
        elif model_name in ("RF", "LightGBM", "CatBoost", "DT", "XGBoost"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
        else:
            background = shap.kmeans(X_train, min(50, X_train.shape[0]))
            explainer = shap.KernelExplainer(
                lambda x: clf.predict_proba(x)[:, 1], background
            )
            shap_values = explainer.shap_values(X_sample, nsamples=100)
    elif hasattr(clf, "decision_function"):
        explainer = shap.KernelExplainer(
            clf.decision_function,
            shap.kmeans(X_train, min(50, X_train.shape[0]))
        )
        shap_values = explainer.shap_values(X_sample, nsamples=100)
    else:
        print("  Individual SHAP: model type not supported, skipping.")
        return

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    # Step 2: Predicted probabilities from the final model
    # 必须与 SHAP 值同源（同一模型、同一批特征），否则图上 f(x) 与预测概率不一致。
    prob_der = clf.predict_proba(X_train)[:, 1]

    # Step 3: 按预测概率中位数二分，仅用于抽样展示，不做临床风险分层
    median_prob = np.median(prob_der)
    low_indices = np.where(prob_der < median_prob)[0]
    high_indices = np.where(prob_der >= median_prob)[0]

    print(f"  Predicted-probability bands: "
          f"Low<{median_prob:.3f} (n={len(low_indices)}), "
          f"High>={median_prob:.3f} (n={len(high_indices)})")

    if len(low_indices) < 1 or len(high_indices) < 1:
        print("  WARNING: Empty predicted-probability band, "
              "skipping individual SHAP analysis.")
        return

    # Step 4: Select 1 case per band (Low: 最低的 1 例；High: 最高的 1 例)
    low_selected = low_indices[np.argmin(prob_der[low_indices])]

    high_selected = high_indices[np.argmax(prob_der[high_indices])]

    all_selected = [low_selected, high_selected]
    all_categories = ["Low", "High"]

    print(f"  Selected cases (indices): {all_selected}")
    for i, idx in enumerate(all_selected):
        print(f"    [{all_categories[i]}] idx={idx}, prob={prob_der[idx]:.4f}")

    # Step 5: Create combined 1x2 SHAP waterfall figure
    # 标准 SHAP 瀑布图（参考 shap.plots._waterfall 官方实现）：
    #   - 最重要特征（绝对值最大）在顶部，依次向下
    #   - 累积从 f(x) 开始，逐步减去特征贡献回到 E[f(x)]
    #   - 条形使用 plt.arrow() 绘制，head_width=width 使末端呈箭头状
    n_display = min(10, len(feature_names))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Individual SHAP Waterfall Analysis — {model_name}\n"
                 f"Derivation cohort (n={len(prob_der)}); 2 representative cases "
                 f"(lowest / highest predicted probability)",
                 fontsize=14, fontweight="bold", y=0.99)

    for idx, col in enumerate([0, 1]):
        ax = axes[col]
        patient_idx = all_selected[idx]

        # Get SHAP values and feature values for this patient
        sv = shap_values[patient_idx]
        fv = X_sample[patient_idx]
        prob = prob_der[patient_idx]

        # Sort by absolute SHAP value (descending), keep top N
        # SHAP 官方：最重要的特征在顶部（y=num_features-1），依次向下
        order = np.argsort(np.abs(sv))[::-1][:n_display]
        sv_top = sv[order]
        fv_top = fv[order]
        fn_top = [feature_names[i] for i in order]
        
        # Inverse transform to raw values if scaler params are available
        if scaler_params is not None:
            try:
                fv_top_raw = fv_top.copy()
                for k, i in enumerate(order):
                    orig_name = orig_name_by_r.get(feature_names[i])
                    if orig_name is None or orig_name not in scaler_params:
                        continue
                    params = scaler_params[orig_name]
                    median_val = params["median"]
                    iqr_val = params["iqr"]
                    if iqr_val > 0:
                        raw = fv_top_raw[k] * iqr_val + median_val
                        if orig_name in log_cols:
                            raw = np.expm1(raw)
                        fv_top_raw[k] = raw
                fv_top = fv_top_raw
            except Exception as e:
                print(f"  [Individual SHAP] Warning: Failed to inverse transform: {e}")

        n_feat = len(sv_top)
        final_val = base_value + np.sum(sv_top)

        # ===== 标准 SHAP 瀑布图布局（参考 shap v0.46+ 官方实现） =====
        # 官方 rng = range(num_features - 1, -1, -1)：最重要特征在顶部(y=n_feat-1)
        # 累积从 f(x) 开始，逐步减去特征贡献回到 E[f(x)]（底部）
        bar_width = 0.8
        rng = np.arange(n_feat - 1, -1, -1)  # 最重要=顶部, 最不重要=底部

        pos_lefts = []; pos_inds = []; pos_widths = []
        neg_lefts = []; neg_inds = []; neg_widths = []
        loc = base_value + np.sum(sv_top)  # 从 f(x) 开始

        for i in range(n_feat):
            sval = sv_top[i]
            loc -= sval  # 减去当前特征贡献，回到上一步累积值
            if sval >= 0:
                pos_inds.append(rng[i])
                pos_widths.append(sval)
                pos_lefts.append(loc)
            else:
                neg_inds.append(rng[i])
                neg_widths.append(sval)
                neg_lefts.append(loc)

            # 连接线（虚线）
            if i < n_feat - 1:
                ax.plot([loc, loc],
                        [rng[i] - 0.4, rng[i] + 0.4],
                        color='#888888', linestyle='--', linewidth=0.5, zorder=-1)

        # --- 计算箭头 head_length（数据坐标） ---
        all_points = (pos_lefts + [p + w for p, w in zip(pos_lefts, pos_widths)]
                      + neg_lefts + [p + w for p, w in zip(neg_lefts, neg_widths)]
                      + [base_value, final_val])
        dataw = np.max(all_points) - np.min(all_points)
        if dataw == 0:
            dataw = 1.0

        # 用 invisible bar 撑开 x 轴范围（官方做法）
        label_pad = 0.1 * dataw
        for i in range(len(pos_widths)):
            ax.barh(pos_inds[i], pos_widths[i] + (label_pad if pos_widths[i] < 1 else 0) + 0.02 * dataw,
                    left=pos_lefts[i] - 0.01 * dataw, color='red', alpha=0)
        for i in range(len(neg_widths)):
            ax.barh(neg_inds[i], -neg_widths[i] + (label_pad if -neg_widths[i] < 1 else 0) + 0.02 * dataw,
                    left=neg_lefts[i] - 0.02 * dataw, color='blue', alpha=0)

        # 计算 head_length（数据坐标）
        head_length_inches = 0.08
        xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_width_inches = max(bbox.width, 0.1)
        bbox_to_xscale = xlen / ax_width_inches
        hl_scaled = bbox_to_xscale * head_length_inches

        # --- 绘制正向贡献箭头（红色） ---
        for i in range(len(pos_inds)):
            dist = pos_widths[i]
            ax.arrow(pos_lefts[i], pos_inds[i], max(dist - hl_scaled, 0.000001), 0,
                     head_length=min(dist, hl_scaled),
                     color='#ff0051', width=bar_width, head_width=bar_width,
                     zorder=3)
            # SHAP 值标注：短条形文字放外侧，长条形居中
            txt = f"{dist:+.3f}"
            text_width_est = len(txt) * 0.012 * dataw  # 估算文字宽度
            if dist > text_width_est:
                ax.text(pos_lefts[i] + 0.5 * dist, pos_inds[i],
                        txt, ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white',
                        zorder=4)
            else:
                ax.text(pos_lefts[i] + dist + 0.01 * dataw, pos_inds[i],
                        txt, ha='left', va='center',
                        fontsize=8, fontweight='bold', color='#ff0051',
                        zorder=4)

        # --- 绘制负向贡献箭头（蓝色） ---
        for i in range(len(neg_inds)):
            dist = neg_widths[i]
            ax.arrow(neg_lefts[i], neg_inds[i], -(-dist - hl_scaled), 0,
                     head_length=min(-dist, hl_scaled),
                     color='#008bfb', width=bar_width, head_width=bar_width,
                     zorder=3)
            # SHAP 值标注：短条形文字放外侧
            txt = f"{dist:+.3f}"
            text_width_est = len(txt) * 0.012 * dataw
            if -dist > text_width_est:
                ax.text(neg_lefts[i] + 0.5 * dist, neg_inds[i],
                        txt, ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white',
                        zorder=4)
            else:
                ax.text(neg_lefts[i] + dist - 0.01 * dataw, neg_inds[i],
                        txt, ha='right', va='center',
                        fontsize=8, fontweight='bold', color='#008bfb',
                        zorder=4)

        # --- Y 轴标签：SHAP 官方双遍格式 ---
        # 第一遍：完整标签 "value = name"（灰色）
        # 第二遍：仅特征名（黑色），覆盖在第一遍上，形成"灰色值 = 黑色特征名"效果
        ytick_labels_full = ["" for _ in range(n_feat + 1)]
        ytick_labels_name = ["" for _ in range(n_feat + 1)]
        for i in range(n_feat):
            ytick_labels_full[rng[i]] = f"{fv_top[i]:.2f} = {fn_top[i]}"
            ytick_labels_name[rng[i]] = fn_top[i]

        yticks = list(range(n_feat)) + list(np.arange(n_feat) + 1e-8)
        yticklabels_combined = (ytick_labels_full[:-1] + ytick_labels_name[:-1])
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels_combined, fontsize=9)
        # 完整标签（前半部分）着灰色
        for i in range(n_feat):
            ax.get_yticklabels()[i].set_color('#888888')
        # 特征名标签（后半部分）着黑色，覆盖在灰色上
        for i in range(n_feat, 2 * n_feat):
            ax.get_yticklabels()[i].set_color('black')

        # --- 水平虚线（每行分隔） ---
        for i in range(n_feat):
            ax.axhline(i, color='#cccccc', lw=0.5, dashes=(1, 5), zorder=-1)

        # --- E[f(x)] 和 f(x) 垂直参考线 ---
        ax.axvline(base_value, 0, 1.0 / n_feat, color='#888888',
                   linestyle='--', linewidth=0.5, zorder=-1)
        ax.axvline(final_val, 0, 1, color='#888888',
                   linestyle='--', linewidth=0.5, zorder=-1)

        # --- E[f(x)] 标注（底部） ---
        ax.text(base_value, -0.8, f"E[f(x)] = {base_value:.3f}",
                ha='center', va='top', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # --- f(x) 标注（顶部） ---
        ax.text(final_val, n_feat - 0.2, f"f(x) = {final_val:.3f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # --- 预测概率标注（右下角） ---
        ax.text(0.98, 0.02,
                f"Predicted probability = {prob:.4f} ({prob * 100:.1f}%)",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                         alpha=0.9))

        # --- 样式 ---
        ax.set_xlabel("SHAP value (log-odds; x-axis in RobustScaler units)", fontsize=9)
        ax.set_title(f"{all_categories[idx]} predicted probability",
                     fontsize=10, fontweight="bold")
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # --- X 轴范围 ---
        x_pad = dataw * 0.15
        ax.set_xlim(np.min(all_points) - x_pad, np.max(all_points) + x_pad)
        ax.set_ylim(-1.2, n_feat + 0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, f"individual_shap_waterfall_{model_name}", result_dir)
    plt.close(fig)
    print(f"  Individual SHAP waterfall saved.")


# =============================================================================
# Main
# =============================================================================
print("=" * 70)
print("05_Model_Development.py — Final Model Training & Full Validation")
print("=" * 70)

# --- Load best parameters ---
pkl_path = os.path.join(SELECTION_DIR, "best_params_all_models.pkl")
if not os.path.exists(pkl_path):
    print(f"ERROR: {pkl_path} not found. Run 04_Model_Selection.py first.")
    sys.exit(1)

with open(pkl_path, "rb") as f:
    best_params_dict = pickle.load(f)
print(f"Loaded best params for {len(best_params_dict)} models.")

# --- Load selection metrics to find recommended model ---
metrics_csv = os.path.join(SELECTION_DIR, "model_selection_metrics.csv")
if os.path.exists(metrics_csv):
    metrics_df = pd.read_csv(metrics_csv)
    # Use AB_Score (consistent with 04_Model_Selection.py)
    if "AB_Score" in metrics_df.columns:
        recommended_model = metrics_df.loc[metrics_df["AB_Score"].idxmax(), "Model"]
    else:
        recommended_model = metrics_df.loc[metrics_df["AUC"].idxmax(), "Model"]
    print(f"Recommended model from selection: {recommended_model}")
else:
    recommended_model = list(best_params_dict.keys())[0]
    print(f"Fallback model: {recommended_model}")

# User override
if SELECTED_MODEL is not None:
    model_name = SELECTED_MODEL
    print(f"User-specified model: {model_name}")
else:
    model_name = recommended_model
    print(f"Using recommended model: {model_name}")

if model_name not in best_params_dict:
    print(f"ERROR: {model_name} not found in best_params_dict.")
    print(f"Available: {list(best_params_dict.keys())}")
    sys.exit(1)

best_params = best_params_dict[model_name]
print(f"Best params: {best_params}")

# Also get the cutoff from selection metrics
if os.path.exists(metrics_csv):
    cutoff_row = metrics_df[metrics_df["Model"] == model_name]
    if len(cutoff_row) > 0:
        final_cutoff = cutoff_row["Optimal_Cutoff"].values[0]
    else:
        final_cutoff = 0.5
else:
    final_cutoff = 0.5
print(f"Optimal cutoff (from 04 nested CV): {final_cutoff:.3f}")

# --- Load metrics from 04 for Derivation cohort ---
selection_metrics_path = os.path.join(SELECTION_DIR, "model_selection_metrics.csv")
nested_cv_metrics = {}
if os.path.exists(selection_metrics_path):
    df_sel = pd.read_csv(selection_metrics_path)
    row = df_sel[df_sel["Model"] == model_name]
    if len(row) > 0:
        nested_cv_metrics = {
            "AUC": row["AUC"].values[0],
            "AUC_CI_Lower": row["AUC_CI_Lower"].values[0],
            "AUC_CI_Upper": row["AUC_CI_Upper"].values[0],
            "Brier": row["Brier"].values[0],
            "Sensitivity": row["Sensitivity"].values[0],
            "Specificity": row["Specificity"].values[0],
            "Accuracy": row["Accuracy"].values[0],
            "F1": row["F1"].values[0],
            "PPV": row["PPV"].values[0],
            "NPV": row["NPV"].values[0],
            "Optimal_Cutoff": row["Optimal_Cutoff"].values[0],
            # 重复嵌套 CV 新增列（缺失时回退 NaN，兼容旧版 04 输出）
            "AUC_SE": (row["AUC_SE"].values[0] if "AUC_SE" in row.columns else np.nan),
            "AUC_SD": (row["AUC_SD"].values[0] if "AUC_SD" in row.columns else np.nan),
            "AUC_N_ValidRepeats": (row["AUC_N_ValidRepeats"].values[0]
                                   if "AUC_N_ValidRepeats" in row.columns else np.nan),
            "Optimal_Cutoff_SD": (row["Optimal_Cutoff_SD"].values[0]
                                  if "Optimal_Cutoff_SD" in row.columns else np.nan),
        }
        print(f"  Loaded metrics for {model_name}: AUC={nested_cv_metrics['AUC']:.3f}")
else:
    print("  WARNING: model_selection_metrics.csv not found, falling back to recomputed metrics.")

# --- Load Data ---
print("\n=== Loading Data ===")
train_df = pd.read_csv(os.path.join(DATA_DIR, "Derivation cohort.csv"))
feature_cols = [c for c in train_df.columns if c not in ("ID", "label")]
print(f"  Features ({len(feature_cols)}): {feature_cols}")

# Check which external validation cohorts exist
has_temporal = os.path.exists(os.path.join(DATA_DIR, "Temporal external validation cohort.csv"))
has_geo = os.path.exists(os.path.join(DATA_DIR, "Geographical external validation cohort.csv"))

X_train = train_df[feature_cols].values.astype(np.float64)
y_train = train_df["label"].values.astype(np.int32)

# 注：原 raw_data_dict（外部集截距重校准用）已随重校准移除而删除

if has_temporal:
    temporal_df = pd.read_csv(os.path.join(DATA_DIR, "Temporal external validation cohort.csv"))
    X_temporal = temporal_df[feature_cols].values.astype(np.float64)
    y_temporal = temporal_df["label"].values.astype(np.int32)
if has_geo:
    geo_df = pd.read_csv(os.path.join(DATA_DIR, "Geographical external validation cohort.csv"))
    X_geo = geo_df[feature_cols].values.astype(np.float64)
    y_geo = geo_df["label"].values.astype(np.int32)

print(f"  Derivation: {X_train.shape}, Pos/Neg: {np.sum(y_train == 1)}/{np.sum(y_train == 0)}")
if has_temporal:
    print(f"  Temporal: {X_temporal.shape}, Pos/Neg: {np.sum(y_temporal == 1)}/{np.sum(y_temporal == 0)}")
if has_geo:
    print(f"  Geo: {X_geo.shape}, Pos/Neg: {np.sum(y_geo == 1)}/{np.sum(y_geo == 0)}")

# --- Use feature_selection outputs directly (already preprocessed/scaled) ---
X_scaled = {"Derivation": X_train}
y_data = {"Derivation": y_train}
if has_temporal:
    X_scaled["Temporal"] = X_temporal
    y_data["Temporal"] = y_temporal
if has_geo:
    X_scaled["Geographical"] = X_geo
    y_data["Geographical"] = y_geo
# NOTE: 特征已由 01/03 预处理流水线完成 RobustScaler 标准化（scaler 参数存于
# preprocessing_objects.joblib，06 脚本据此反标准化）。05 不重新拟合 scaler，
# 因此不再生成占位用的 scaler_{model_name}.pkl 假文件。

# --- Train Final Model ---
print(f"\n=== Training Final Model: {model_name} ===")
clf = build_model(model_name, best_params)
X_train_scaled = X_train
X_train_scaled_df = pd.DataFrame(X_train_scaled)
if model_name in ("LightGBM", "XGBoost", "CatBoost"):
    # train_test_split 已在模块顶部导入
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train_scaled_df, y_train, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_train
    )
    if model_name == "LightGBM":
        import lightgbm as lgb
        clf.fit(X_tr_sub, y_tr_sub,
                eval_set=[(X_val_sub, y_val_sub)],
                callbacks=[lgb.early_stopping(50, verbose=False)])
    elif model_name == "XGBoost":
        clf.fit(X_tr_sub, y_tr_sub,
                eval_set=[(X_val_sub, y_val_sub)],
                verbose=False)
    elif model_name == "CatBoost":
        clf.fit(X_tr_sub, y_tr_sub,
                eval_set=(X_val_sub, y_val_sub),
                verbose=False)
else:
    clf.fit(X_train_scaled_df, y_train)
print("  Training complete.")

# --- Model Formula (for linear models) ---
model_formula = None
if model_name in ("L2_LR",):
    coef = clf.coef_[0] if hasattr(clf, "coef_") else None
    intercept = clf.intercept_[0] if hasattr(clf, "intercept_") else None
    if coef is not None and intercept is not None:
        feature_names = feature_cols
        if len(feature_names) == len(coef):
            terms = [f"{intercept:.4f}"]
            for name, c in zip(feature_names, coef):
                if abs(c) > 0.0001:
                    terms.append(f"{c:+.4f}*{name}")
            model_formula = "logit(P) = " + " ".join(terms)
            print(f"  Model Formula: {model_formula}")

# --- Predictions (校准策略) ---
# Step 3a: Derivation 校准器（Platt，仅在 Derivation 上拟合），用于 Derivation 与 Temporal。
# Step 3b: 地理外部验证集改用「时间集校准」——与 07 盲法验证完全一致的做法：
#          模型仍只在 Derivation 上训练，Temporal 仅用于拟合重校准器，
#          再将该校准器应用于 Geographical 队列（不做任何重训）。
# preds_dict_raw 始终保存未校准的原始概率，用于 Raw vs Recalibrated 对比。
print("\n=== Predictions (Calibration: Derivation for internal/Temporal, Temporal for Geographical) ===")
preds_dict = {}       # Recalibrated（Derivation / Temporal 用 Derivation 校准器；Geographical 用 Temporal 校准器）
preds_dict_raw = {}   # Raw (uncalibrated)

# Step 1: collect raw predictions for ALL cohorts
# 统一使用 predict_proba 获取概率（SVM 已启用 probability=True），
# 避免外部集用 decision_function+sigmoid、Derivation 用 predict_proba
# 导致同一模型在内部/外部集概率尺度不等价。
raw_logits_all = {}
raw_probs_all = {}
for cohort_name in X_scaled:
    X_df = pd.DataFrame(X_scaled[cohort_name])
    y_prob = clf.predict_proba(X_df)[:, 1]
    y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
    raw_logits_all[cohort_name] = np.log(y_prob_clipped / (1 - y_prob_clipped))
    raw_probs_all[cohort_name] = y_prob
    preds_dict_raw[cohort_name] = (y_data[cohort_name], y_prob)

# Step 2: compute apparent AUC on Derivation using raw probs
apparent_auc = roc_auc_score(y_data["Derivation"], raw_probs_all["Derivation"])
print(f"  Apparent AUC (Derivation, raw probs): {apparent_auc:.4f}")

# Step 3: fit ONE calibrator on Derivation, apply to all cohorts
calib_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
calib_model.fit(raw_logits_all["Derivation"].reshape(-1, 1), y_data["Derivation"])
new_intercept = calib_model.intercept_[0]
new_slope = calib_model.coef_[0][0]
print(f"  Derivation-fitted calibrator: Intercept={new_intercept:.4f}, Slope={new_slope:.4f}")
# 保存校准器供 06 复用
calibrator_path = os.path.join(RESULT_DIR, f"calibrator_{model_name}.pkl")
with open(calibrator_path, "wb") as f:
    pickle.dump(calib_model, f)
print(f"  Calibrator saved to: {calibrator_path}")
final_cutoff_logit = np.log(np.clip(final_cutoff, 1e-6, 1 - 1e-6) / (1 - np.clip(final_cutoff, 1e-6, 1 - 1e-6)))
final_cutoff_calibrated = calib_model.predict_proba(np.array([[final_cutoff_logit]]))[:, 1][0]
print(f"  Calibrated cutoff for recalibrated probabilities: {final_cutoff_calibrated:.4f}")

# Step 3b: 地理外部验证集 —— 用 Temporal 拟合的校准器（与 07 盲法验证口径一致）
# 说明：Temporal 在本策略中仅贡献 2 个校准参数（斜率/截距），不参与训练、特征选择与标准化；
#       地理队列全程不参与训练或校准器拟合，仍为独立外部验证集。
# ⚠️ 方法学注意：Geographical 校准器由 Temporal 结局拟合，其校准后概率间接依赖 Temporal 结局；若论文要称其"完全独立"，应改用 Derivation 校准器。
geo_calibrator = None
final_cutoff_geo = final_cutoff_calibrated
if has_geo and has_temporal:
    geo_calibrator = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    geo_calibrator.fit(raw_logits_all["Temporal"].reshape(-1, 1), y_data["Temporal"])
    geo_intercept = geo_calibrator.intercept_[0]
    geo_slope = geo_calibrator.coef_[0][0]
    final_cutoff_geo = geo_calibrator.predict_proba(np.array([[final_cutoff_logit]]))[:, 1][0]
    print(f"  Geographical calibrator (fitted on Temporal, n={len(y_data['Temporal'])}): "
          f"Intercept={geo_intercept:.4f}, Slope={geo_slope:.4f}")
    print(f"  Geographical calibrated cutoff: {final_cutoff_geo:.4f}")
    geo_calibrator_path = os.path.join(RESULT_DIR, f"calibrator_geo_{model_name}.pkl")
    with open(geo_calibrator_path, "wb") as f:
        pickle.dump(geo_calibrator, f)
    print(f"  Geographical calibrator saved to: {geo_calibrator_path}")
else:
    print("  WARNING: Temporal/Geographical cohort missing — Geographical falls back to the Derivation calibrator.")
    geo_slope = np.nan
    geo_intercept = np.nan

for cohort_name in X_scaled:
    y_true = y_data[cohort_name]
    if cohort_name == "Geographical" and geo_calibrator is not None:
        y_prob = geo_calibrator.predict_proba(raw_logits_all[cohort_name].reshape(-1, 1))[:, 1]
    else:
        y_prob = calib_model.predict_proba(raw_logits_all[cohort_name].reshape(-1, 1))[:, 1]
    preds_dict[cohort_name] = (y_true, y_prob)
    n = len(y_true)
    print(f"  {cohort_name}: {n} samples | "
          f"Mean raw={raw_probs_all[cohort_name].mean():.3f}, Mean cal={y_prob.mean():.3f}")

# --- Compute Metrics ---
print("\n=== Computing Metrics ===")
metrics_dict = {}       # 重校准后（默认）
metrics_dict_raw = {}   # 未校准（原始概率）

for cohort, (y_true, y_prob) in preds_dict.items():
    if cohort == "Derivation" and nested_cv_metrics:
        # Derivation: use nested-CV AUC from 04 (UC estimate),
        # but Brier and clinical metrics must come from RAW apparent predictions
        # (the Derivation-fitted calibrator over-fits, so its Brier is optimistic).
        y_true_d, y_prob_raw_d = preds_dict_raw[cohort]  # raw (uncalibrated) probs
        m_apparent = compute_metrics(y_true_d, y_prob_raw_d, final_cutoff)
        # Start with nested-CV AUC/Apparent_AUC (for comparability with 04)
        metrics_dict[cohort] = dict(nested_cv_metrics)  # copy
        # Override Brier, Calib_Intercept/Slope, and clinical metrics with RAW apparent values
        for k in ("Brier", "Accuracy", "Sensitivity", "Specificity", "F1", "PPV", "NPV",
                  "Cutoff", "Calib_Intercept", "Calib_Slope",
                  "Calib_Slope_CI_Lower", "Calib_Slope_CI_Upper"):
            if k in m_apparent:
                metrics_dict[cohort][k] = m_apparent[k]
        # Compute AUPRC from raw predictions
        auprc_mean, auprc_lo, auprc_hi = bootstrap_auprc_ci(y_true_d, y_prob_raw_d)
        metrics_dict[cohort]["AUPRC"] = auprc_mean
        metrics_dict[cohort]["AUPRC_CI_Lower"] = auprc_lo
        metrics_dict[cohort]["AUPRC_CI_Upper"] = auprc_hi
        # For raw dict: store the same raw apparent values
        metrics_dict_raw[cohort] = dict(metrics_dict[cohort])
        print(f"  Derivation: Using nested-CV AUC + RAW apparent Brier "
              f"(AUC_nested_CV={nested_cv_metrics['AUC']:.3f}, Brier_raw={m_apparent['Brier']:.3f}), "
              f"AUPRC={auprc_mean:.3f} ({auprc_lo:.3f}-{auprc_hi:.3f})")
    else:
        # External validation: compute two sets of metrics
        # 1. Raw (uncalibrated)
        y_true_raw, y_prob_raw = preds_dict_raw[cohort]
        m_raw = compute_metrics(y_true_raw, y_prob_raw, final_cutoff)
        metrics_dict_raw[cohort] = m_raw
        # 2. Recalibrated（地理集用 Temporal 校准器映射后的截断值）
        cohort_cutoff = (final_cutoff_geo
                         if (cohort == "Geographical" and geo_calibrator is not None)
                         else final_cutoff_calibrated)
        m = compute_metrics(y_true, y_prob, cohort_cutoff)
        metrics_dict[cohort] = m
    auc_str = f"{metrics_dict[cohort]['AUC']:.3f} ({metrics_dict[cohort]['AUC_CI_Lower']:.3f}-{metrics_dict[cohort]['AUC_CI_Upper']:.3f})"
    auprc_str = (f"{metrics_dict[cohort]['AUPRC']:.3f} "
                 f"({metrics_dict[cohort].get('AUPRC_CI_Lower', 0):.3f}-"
                 f"{metrics_dict[cohort].get('AUPRC_CI_Upper', 0):.3f})")
    print(f"  {cohort} (Recalibrated): AUC={auc_str}, AUPRC={auprc_str}, Brier={metrics_dict[cohort]['Brier']:.3f}, "
          f"Acc={metrics_dict[cohort]['Accuracy']:.3f}, Sens={metrics_dict[cohort]['Sensitivity']:.3f}, "
          f"Spec={metrics_dict[cohort]['Specificity']:.3f}, F1={metrics_dict[cohort]['F1']:.3f}")
    if cohort != "Derivation":
        raw_brier = metrics_dict_raw[cohort]["Brier"]
        raw_calib_slope = metrics_dict_raw[cohort]["Calib_Slope"]
        print(f"  {cohort} (Raw): Brier={raw_brier:.3f}, Calib_Slope={raw_calib_slope:.3f}")

# --- Sensitivity Analysis: Self-optimal cutoff per external validation set ---
print("\n=== Sensitivity Analysis (Self-optimal Cutoff for External Sets) ===")
sensitivity_rows = []
for cohort, (y_true, y_prob) in preds_dict.items():
    if cohort == "Derivation":
        continue
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    if len(thresholds) > 0:
        self_cutoff = thresholds[best_idx]
        # roc_curve 返回的首个阈值为 inf（sklearn 约定），非有限值回退 0.5
        if not np.isfinite(self_cutoff):
            self_cutoff = 0.5
    else:
        self_cutoff = 0.5
    y_pred_self = (y_prob >= self_cutoff).astype(int)
    self_sens = recall_score(y_true, y_pred_self)
    # 显式 labels=[0,1]，防止某一类缺失时 ravel() 返回维度不足而报错
    tn_s, fp_s, _, _ = confusion_matrix(y_true, y_pred_self, labels=[0, 1]).ravel()
    self_spec = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0
    acc_self = accuracy_score(y_true, y_pred_self)
    f1_self = f1_score(y_true, y_pred_self, zero_division=0)
    sensitivity_rows.append({
        "Cohort": cohort,
        "Self_Optimal_Cutoff": self_cutoff,
        "Self_Sensitivity": self_sens,
        "Self_Specificity": self_spec,
        "Self_Accuracy": acc_self,
        "Self_F1": f1_self,
        "Applied_Cutoff_from_Derivation_Raw": final_cutoff,
        "Applied_Cutoff_Recalibrated": (final_cutoff_geo
                                        if (cohort == "Geographical" and geo_calibrator is not None)
                                        else final_cutoff_calibrated),
        "Applied_Sensitivity": metrics_dict[cohort]["Sensitivity"],
        "Applied_Specificity": metrics_dict[cohort]["Specificity"],
    })
    print(f"  {cohort}: Self-cutoff={self_cutoff:.3f}, "
          f"Sens(Self)={self_sens:.3f}, Spec(Self)={self_spec:.3f} | "
          f"Sens(Applied)={metrics_dict[cohort]['Sensitivity']:.3f}, "
          f"Spec(Applied)={metrics_dict[cohort]['Specificity']:.3f}")

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_csv = os.path.join(RESULT_DIR, "sensitivity_cutoff_analysis.csv")
sensitivity_df.to_csv(sensitivity_csv, index=False, float_format='%.6f')
print(f"  Sensitivity analysis saved: {sensitivity_csv}")

# --- Save Final Summary ---
summary_rows = []

# 1. Derivation Apparent 行 (使用原始概率在完整训练集上计算)
y_true_der, y_prob_der_raw = preds_dict_raw["Derivation"]
m_apparent_der = compute_metrics(y_true_der, y_prob_der_raw, final_cutoff)
row_der_app = {"Cohort": "Derivation", "Model": model_name, "Calibration_Status": "Apparent (Raw)",
               "AUC_Estimation": "Bootstrap mean (apparent, raw probs)"}
row_der_app.update(m_apparent_der)
row_der_app["Apparent_AUC"] = m_apparent_der["AUC"]
row_der_app["Cutoff_Source"] = "raw (Derivation nested-CV median)"
summary_rows.append(row_der_app)

# 2. Derivation Optimism-Corrected 行 (使用 nested-CV AUC + 原始Brier)
if nested_cv_metrics:
    row_der_cv = {"Cohort": "Derivation", "Model": model_name, "Calibration_Status": "Optimism-Corrected (Nested-CV)",
                  "AUC_Estimation": "Nested-CV (from 04_Model_Selection.py)"}
    row_der_cv.update(nested_cv_metrics)
    # Brier 沿用 apparent 的 Brier（该行其余指标为 nested-CV 口径；04 自 v3 起
    # 也提供重复级 Brier，但为使本行与外部队列行的 Brier 定义一致，仍取 apparent）
    row_der_cv["Brier"] = m_apparent_der.get("Brier", np.nan)
    # AUPRC（nested-CV 口径）：从 04 的 OOF 预测用 bootstrap_auprc_ci 计算，补齐该行缺失的 AUPRC 及 95%CI
    oof_path_sum = os.path.join(SELECTION_DIR, "oof_predictions_all_models.csv")
    if os.path.exists(oof_path_sum):
        try:
            _oof_sum = pd.read_csv(oof_path_sum)
            _oof_sum = _oof_sum[_oof_sum["model"] == model_name]
            if len(_oof_sum) > 0:
                _am, _alo, _ahi = bootstrap_auprc_ci(
                    _oof_sum["y_true"].values.astype(np.int32),
                    _oof_sum["y_pred_prob"].values.astype(np.float64))
                row_der_cv["AUPRC"] = _am
                row_der_cv["AUPRC_CI_Lower"] = _alo
                row_der_cv["AUPRC_CI_Upper"] = _ahi
                print(f"  Nested-CV AUPRC (from OOF): {_am:.3f} ({_alo:.3f}-{_ahi:.3f})")
        except Exception as e:
            print(f"  WARNING: nested-CV AUPRC 计算失败 ({e})，该行 AUPRC 留空。")
    row_der_cv["Apparent_AUC"] = apparent_auc
    # 该行各指标口径不同，CSV 内显式标注来源，避免读者误以为全部为 nested-CV 口径：
    #   AUC + CI  = 重复级均值 ± 1.96×重复级 SE（04）
    #   Brier     = raw apparent（与外部验证行同定义）
    #   AUPRC     = repeat 1 代表性 OOF 的 BCa bootstrap
    row_der_cv["Brier_Source"] = "raw apparent"
    row_der_cv["AUPRC_Source"] = "repeat-1 OOF bootstrap (BCa)"
    row_der_cv["Cutoff_Source"] = "raw (Derivation nested-CV median)"
    summary_rows.append(row_der_cv)

# 3. 外部验证集: 校准后 (Recalibrated) + 原始 (Raw)
for cohort in preds_dict:
    if cohort == "Derivation":
        continue
    m = metrics_dict[cohort]
    row_cal = {"Cohort": cohort, "Model": model_name, "Calibration_Status": "Recalibrated",
               "AUC_Estimation": "Bootstrap mean (calibrated probs)"}
    row_cal.update(m)
    row_cal["Apparent_AUC"] = m["AUC"]
    row_cal["Cutoff_Source"] = "mapped via Platt (or Derivation calibrator)"
    summary_rows.append(row_cal)

    if cohort in metrics_dict_raw:
        m_raw = metrics_dict_raw[cohort]
        row_raw = {"Cohort": cohort, "Model": model_name, "Calibration_Status": "Raw",
                   "AUC_Estimation": "Bootstrap mean (raw probs)"}
        row_raw.update(m_raw)
        row_raw["Apparent_AUC"] = m_raw["AUC"]
        row_raw["Cutoff_Source"] = "raw (Derivation nested-CV median)"
        summary_rows.append(row_raw)

summary_df = pd.DataFrame(summary_rows)
# 列名统一为 Applied_Cutoff：本表记录的是「实际施加于该队列的阈值」，
# 与 04 的 Optimal_Cutoff（Derivation 重复级中位数）区分，避免读者误读。
summary_df = summary_df.rename(columns={"Optimal_Cutoff": "Applied_Cutoff"})
summary_csv = os.path.join(RESULT_DIR, "final_model_summary.csv")
summary_df.to_csv(summary_csv, index=False, float_format='%.6f')
print(f"\n  Final summary saved: {summary_csv}")

# =========================================================================
# Calibration Status
# =========================================================================
# 说明：Derivation 校准器（Platt，penalty=None）仅在 Derivation 上拟合，用于
# Derivation 与 Temporal；地理队列改用「时间集校准」，即用 Temporal 拟合的
# 校准器（斜率+截距，2 个参数）重新映射地理队列的原始概率，与 07 盲法验证
# 完全一致。任何外部队列的标签都不参与模型训练，不产生数据泄漏。
# External validation reports only Raw (Table 2) and Platt (eTable 1) metrics. Intercept
# recalibration is NOT performed because it uses the external cohort's own labels and would
# constitute partial in-sample fitting.

# --- Save Prediction Probabilities ---
for cohort in preds_dict.keys():
    y_true, y_prob_cal = preds_dict[cohort]
    y_true_raw, y_prob_raw = preds_dict_raw[cohort]
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_prob_raw": y_prob_raw,
        "y_pred_prob_calibrated": y_prob_cal
    })
    fname = f"predicted_probabilities_{model_name}_{cohort.lower()}.csv"
    df.to_csv(os.path.join(RESULT_DIR, fname), index=False, float_format='%.6f')
print("  Prediction probabilities saved (raw + calibrated).")

# --- Statistical Tests ---
print("\n=== Statistical Tests ===")

# Independent AUC comparison: all available cohort pairs
# Derivation 侧优先使用 04 脚本 nested-CV OOF 预测参与比较（表观 calibrated
# 概率过于乐观）；若 OOF 文件缺失则回退表观概率，并在 AUC_Estimation 列标注口径
oof_preds_auc = None
oof_path_auc = os.path.join(SELECTION_DIR, "oof_predictions_all_models.csv")
if os.path.exists(oof_path_auc):
    try:
        _oof_all = pd.read_csv(oof_path_auc)
        _oof_sub = _oof_all[_oof_all["model"] == model_name]
        if len(_oof_sub) > 0:
            oof_preds_auc = (_oof_sub["y_true"].values.astype(np.int32),
                             _oof_sub["y_pred_prob"].values.astype(np.float64))
            print(f"  AUC comparison: Derivation uses nested-CV OOF predictions "
                  f"(n={len(_oof_sub)}).")
    except Exception as e:
        print(f"  WARNING: failed to load OOF for AUC comparison ({e}), "
              f"falling back to apparent probs.")
else:
    print(f"  WARNING: {oof_path_auc} not found, Derivation side falls back to apparent probs.")

def _auc_estimation_label(cohort):
    """AUC 估计口径标注：Derivation 按 OOF/表观区分，外部队列标注 External。"""
    if cohort == "Derivation":
        return "Nested-CV OOF (Derivation)" if oof_preds_auc is not None else "Apparent (Derivation)"
    return "External"

cohort_list = list(preds_dict.keys())
comparisons = []
for i in range(len(cohort_list)):
    for j in range(i + 1, len(cohort_list)):
        c1 = cohort_list[i]
        c2 = cohort_list[j]
        y1, prob1 = preds_dict[c1]
        y2, prob2 = preds_dict[c2]
        # Derivation 侧替换为 OOF 预测（若可用）
        if c1 == "Derivation" and oof_preds_auc is not None:
            y1, prob1 = oof_preds_auc
        if c2 == "Derivation" and oof_preds_auc is not None:
            y2, prob2 = oof_preds_auc
        comparisons.append((c1, y1, prob1, c2, y2, prob2))

n_comparisons = len(comparisons)
bonferroni_factor = max(n_comparisons, 1)

auc_comparison_results = []
for label1, y1, prob1, label2, y2, prob2 in comparisons:
    auc1, auc2, z_stat, p_val, auc_diff, auc_diff_lo, auc_diff_hi = \
        auc_comparison_test(y1, prob1, y2, prob2, label1, label2)
    auc_comparison_results.append({
        "Comparison": f"{label1} vs {label2}",
        f"AUC_{label1}": auc1,
        f"AUC_{label2}": auc2,
        "AUC_Diff": auc_diff,
        "AUC_Diff_95CI_Lower": auc_diff_lo,
        "AUC_Diff_95CI_Upper": auc_diff_hi,
        "Z_Statistic": z_stat,
        "P_Value": p_val,
        # Bonferroni 保留作敏感性分析；实际多重比较校正使用 BH-FDR（见下）
        "P_Value_Bonferroni": min(p_val * bonferroni_factor, 1.0) if not np.isnan(p_val) else 1.0,
        "AUC_Estimation": f"{_auc_estimation_label(label1)} vs {_auc_estimation_label(label2)}",
    })
    print(f"  {label1} vs {label2}: AUC={auc1:.3f} vs {auc2:.3f}, "
          f"Diff={auc_diff:.3f} ({auc_diff_lo:.3f}-{auc_diff_hi:.3f}), "
          f"Z={z_stat:.3f}, p={p_val:.3f} (Bonferroni: {min(p_val * bonferroni_factor, 1.0):.3f})")

# --- 多重比较校正：Benjamini-Hochberg FDR（实际使用），Bonferroni 列保留作敏感性 ---
_p_vals = np.array([r["P_Value"] if not np.isnan(r["P_Value"]) else 1.0
                    for r in auc_comparison_results])
if len(_p_vals) > 0:
    _p_bh = multipletests(_p_vals, method="fdr_bh")[1]
    for r, p_corr in zip(auc_comparison_results, _p_bh):
        r["P_Value_BH_FDR"] = p_corr
        print(f"  {r['Comparison']}: BH-FDR corrected p={p_corr:.3f}")

auc_comparison_df = pd.DataFrame(auc_comparison_results)
auc_comparison_csv = os.path.join(RESULT_DIR, "auc_comparison_results.csv")
auc_comparison_df.to_csv(auc_comparison_csv, index=False, float_format='%.6f')

# Hosmer-Lemeshow test
# 基于 RAW（未校准）预测概率计算，与外部验证集 raw 指标口径一致；
# 校准后概率会人为改善拟合优度，导致 HL 检验失真（M1）。
hl_results = {}
hl_rows = []
for cohort, (y_true, y_prob) in preds_dict_raw.items():
    # Fixed g=10 for all cohorts to ensure comparability across derivation/external sets.
    # Subject to minimum sample size: if n < 50, use fewer bins so each has ≥5 samples.
    if len(y_true) < 50:
        g = min(10, len(y_true) // 5)
    else:
        g = 10
    chi2_stat, p_val, df = hosmer_lemeshow_test(y_true, y_prob, g=g)
    hl_results[cohort] = {"chi2": chi2_stat, "p": p_val, "df": df}
    hl_rows.append({"Cohort": cohort, "Chi2": chi2_stat, "DF": df, "P_Value": p_val,
                    "Prob_Type": "Raw (uncalibrated)"})
    print(f"  HL {cohort} (raw probs): Chi2={chi2_stat:.3f}, df={df}, p={p_val:.3f}")

hl_df = pd.DataFrame(hl_rows)
hl_csv = os.path.join(RESULT_DIR, "hosmer_lemeshow_results.csv")
hl_df.to_csv(hl_csv, index=False, float_format='%.6f')

# Individual SHAP Waterfall (2 representative cases: lowest / highest probability)
print("\n[Individual SHAP] Waterfall plots for 2 representative cases...")
try:
    plot_individual_shap_force(clf, X_train_scaled, feature_cols, model_name,
                               RESULT_DIR)
except Exception as e:
    print(f"  Individual SHAP failed: {e}")

# --- Visualizations ---
print("\n=== Generating Visualizations ===")

# 1. ROC Curves
print("\n[1/12] ROC Curves")
try:
    plot_roc_curves(preds_dict, model_name, RESULT_DIR,
                nested_cv_auc=nested_cv_metrics.get("AUC"),
                nested_cv_ci_lower=nested_cv_metrics.get("AUC_CI_Lower"),
                nested_cv_ci_upper=nested_cv_metrics.get("AUC_CI_Upper"))
except Exception as e:
    print(f"  ERROR: {e}")

# 1a. Derivation Optimism (Apparent vs Nested-CV AUC)
print("\n[2/12] Derivation Optimism ROC")
try:
    nested_cv_path_opt = os.path.join(SELECTION_DIR, "oof_predictions_all_models.csv")
    if os.path.exists(nested_cv_path_opt):
        nested_cv_all_opt = pd.read_csv(nested_cv_path_opt)
        nested_cv_df_opt = nested_cv_all_opt[nested_cv_all_opt["model"] == model_name].copy()
        if len(nested_cv_df_opt) == len(y_train):
            oof_y_prob = nested_cv_df_opt["y_pred_prob"].values
            oof_y_true = nested_cv_df_opt["y_true"].values
            plot_derivation_optimism(clf, X_train_scaled, y_train, oof_y_prob, oof_y_true, model_name, RESULT_DIR,
                                     nested_cv_auc=nested_cv_metrics.get("AUC"),
                                     nested_cv_ci_lower=nested_cv_metrics.get("AUC_CI_Lower"),
                                     nested_cv_ci_upper=nested_cv_metrics.get("AUC_CI_Upper"))
        else:
            print(f"  WARNING: OOF length {len(nested_cv_df_opt)} != y_train length {len(y_train)}, skipping.")
    else:
        print(f"  WARNING: {nested_cv_path_opt} not found, skipping optimism plot.")
except Exception as e:
    print(f"  ERROR: {e}")

# 1b. PR Curves
print("\n[3/12] PR Curves")
try:
    plot_pr_curves(preds_dict, metrics_dict, model_name, RESULT_DIR,
                   preds_dict_raw=preds_dict_raw)
except Exception as e:
    print(f"  ERROR: {e}")

# 2. Calibration Curves
print("\n[4/12] Calibration Curves")
try:
    plot_calibration_curves(preds_dict, preds_dict_raw, model_name, hl_results, RESULT_DIR)
except Exception as e:
    print(f"  ERROR: {e}")

# 2b. 地理集校准度改善（Raw vs 时间集校准后）
print("\n[5/12] Geographical Calibration Improvement")
try:
    if "Geographical" in preds_dict and "Geographical" in preds_dict_raw:
        y_g, p_g_cal = preds_dict["Geographical"]
        _, p_g_raw = preds_dict_raw["Geographical"]
        hl_g_raw = hosmer_lemeshow_test(y_g, p_g_raw, g=10)[1]
        hl_g_cal = hosmer_lemeshow_test(y_g, p_g_cal, g=10)[1]
        plot_calibration_improvement("Geographical", y_g, p_g_raw, p_g_cal,
                                     metrics_dict_raw["Geographical"],
                                     metrics_dict["Geographical"],
                                     hl_g_raw, hl_g_cal,
                                     model_name, RESULT_DIR)
    else:
        print("  WARNING: Geographical cohort not available, skipping.")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. DCA Curves (Removed per user request — DCA is generated by 06_Clinical_Plots_R.R)

# 4. Simple SHAP (bar + beeswarm + top 3 dependence)
print("\n[6/12] Simple SHAP Summary")
try:
    plot_shap(clf, X_train_scaled, feature_cols, model_name, RESULT_DIR)
except Exception as e:
    print(f"  Simple SHAP failed: {e}")

# 4b. Advanced SHAP (Removed per user request — Combined SHAP below retains the full visualization)

# 4c. Combined SHAP (Beeswarm + Bar + Dependence 6 — 2x3 grid)
print("\n[7/12] Combined SHAP (Beeswarm + Bar + Dependence 6)")
try:
    plot_combined_shap(clf, X_train_scaled, X_train_scaled, y_train,
                       feature_cols, model_name, RESULT_DIR, scaler=None)
except Exception as e:
    print(f"  Combined SHAP failed: {e}")

# 5. Feature Importance / OR Forest
print("\n[8/12] Feature Importance")
try:
    plot_feature_importance(clf, feature_cols, model_name, X_train_scaled, y_train, RESULT_DIR)
except Exception as e:
    print(f"  ERROR: {e}")

# 6. Radar Chart (AB_Score + AUC)
print("\n[9/12] Radar Charts")
try:
    plot_radar_chart(metrics_dict, model_name, RESULT_DIR, title_prefix="AB_Score")
    plot_radar_chart(metrics_dict, model_name, RESULT_DIR, title_prefix="AUC")
except Exception as e:
    print(f"  ERROR: {e}")

# 7. Confusion Matrices
print("\n[10/12] Confusion Matrices")
try:
    # Load nested-CV predictions for Derivation confusion matrix
    nested_cv_path = os.path.join(SELECTION_DIR, "oof_predictions_all_models.csv")
    nested_cv_df = None
    if os.path.exists(nested_cv_path):
        nested_cv_all = pd.read_csv(nested_cv_path)
        nested_cv_df = nested_cv_all[nested_cv_all["model"] == model_name].copy()
    # 各外部队列的校准截断值可能不同（地理集用时间集校准器映射）
    _cal_cutoffs = {c: final_cutoff_calibrated for c in preds_dict if c != "Derivation"}
    if geo_calibrator is not None and "Geographical" in _cal_cutoffs:
        _cal_cutoffs["Geographical"] = final_cutoff_geo
    plot_confusion_matrices(preds_dict, model_name, _cal_cutoffs, RESULT_DIR,
                            nested_cv_df=nested_cv_df, cutoff_raw=final_cutoff)
except Exception as e:
    print(f"  ERROR: {e}")

# Supplementary: AUPRC Relative Gain Bar Plot
print("\n[11/12] AUPRC Gain Bar Plot")
try:
    plot_auprc_gain_bar(metrics_dict, preds_dict, model_name, RESULT_DIR)
except Exception as e:
    print(f"  ERROR: {e}")

# Supplementary: Brier Skill Score
print("\n[12/12] Brier Skill Score")
try:
    plot_brier_skill_score(metrics_dict, preds_dict, model_name, RESULT_DIR,
                           preds_dict_raw=preds_dict_raw)
except Exception as e:
    print(f"  ERROR: {e}")

# --- Text Report ---
print("\n=== Generating Text Report ===")
report_lines = []
report_lines.append("=" * 70)
report_lines.append(f"Model Development Report — {model_name}")
report_lines.append("=" * 70)
report_lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Model: {model_name}")
report_lines.append(f"Python: {sys.version}")
report_lines.append(f"scikit-learn: {_sklearn.__version__}")
report_lines.append(f"Optuna: {_optuna.__version__}")
report_lines.append(f"numpy: {np.__version__}")
report_lines.append(f"pandas: {pd.__version__}")
report_lines.append(f"shap: {shap.__version__ if 'shap' in sys.modules else 'N/A'}")
report_lines.append(f"Bootstrap iterations: {BOOTSTRAP_ITER}")
report_lines.append(f"Optimal Cutoff: {final_cutoff:.3f}")
report_lines.append(f"Best Parameters: {best_params}")
if model_formula:
    report_lines.append(f"Model Formula: {model_formula}")
report_lines.append("")
report_lines.append("Note: Derivation row in final_model_summary.csv is mixed-口径: AUC+CI = repeat-mean ± 1.96×repeat-SE (04); Sensitivity/Specificity/Accuracy/F1/PPV/NPV/Applied_Cutoff = repeat-level mean (Applied_Cutoff = median of repeat-level cutoffs); Brier = raw apparent (see Brier_Source); AUPRC = repeat-1 OOF BCa (see AUPRC_Source). 各行实际施加的阈值来源见 Cutoff_Source 列.")
report_lines.append("External validation sets report both Raw (uncalibrated) and Recalibrated metrics.")
report_lines.append("Calibration: Derivation and Temporal use the Platt scaler (Logistic, penalty=None) fit on Derivation;")
report_lines.append("Geographical uses the calibrator fit on Temporal (same as 07 blinded validation), so its")
report_lines.append("calibrated probabilities depend indirectly on Temporal outcomes (see methodological note in code).")
report_lines.append("")

# --- TRIPOD Reporting Elements ---
report_lines.append("-" * 50)
report_lines.append("TRIPOD Reporting Elements")
report_lines.append("-" * 50)
for cohort_name, (y_t, _) in preds_dict_raw.items():
    n_coh = len(y_t)
    n_ev = int(np.sum(y_t == 1))
    report_lines.append(
        f"  {cohort_name}: n={n_coh}, events={n_ev}, non-events={n_coh - n_ev}, "
        f"prevalence={n_ev / n_coh:.3f}")
n_ev_der = int(np.sum(y_data["Derivation"] == 1))
n_feat = len(feature_cols)
epv = n_ev_der / n_feat if n_feat > 0 else np.nan
report_lines.append(f"  Events per final selected predictor (derivation): {epv:.1f} "
                    f"(events={n_ev_der} / final selected predictors={n_feat})")
report_lines.append(f"  Candidate features (final selection, n={n_feat}): {', '.join(feature_cols)}")
report_lines.append("  Missing data: handled in 01_data_preprocessing.py via NaN-aware scaling -> "
                    "imputation -> inverse scaling (ScaledImputer); feature_selection CSVs contain "
                    "no missing values.")
report_lines.append("  Predictor definitions: features standardized with RobustScaler (median/IQR) "
                    "fitted on Derivation cohort only; age reverse-standardized for subgroup analysis "
                    "in Part 2 (Subgroup Vulnerability Analysis).")
report_lines.append("  Model development: all hyperparameters optimized by nested cross-validation "
                    "on Derivation cohort (04_Model_Selection.py); external cohorts never "
                    "participated in training.")
report_lines.append("")

report_lines.append("-" * 50)
report_lines.append("Performance Metrics (Recalibrated)")
report_lines.append("-" * 50)
for cohort, m in metrics_dict.items():
    # Derivation 条目实为 nested-CV AUC + raw apparent 混合口径，单独标注；
    # 外部队列保留 Recalibrated 标注
    if cohort == "Derivation":
        report_lines.append(f"\n  {cohort} (Nested-CV AUC + raw apparent metrics):")
    else:
        report_lines.append(f"\n  {cohort} (Recalibrated):")
    # AUPRC with CI (explicit formatting for clarity)
    if "AUPRC" in m:
        report_lines.append(f"    AUPRC: {m['AUPRC']:.3f} (95% CI: {m['AUPRC_CI_Lower']:.3f}-{m['AUPRC_CI_Upper']:.3f})")
    for k, v in m.items():
        if k.startswith("AUPRC"):  # skip already-printed AUPRC entries
            continue
        if isinstance(v, float):
            report_lines.append(f"    {k}: {v:.3f}")
        else:
            report_lines.append(f"    {k}: {v}")

# Null Brier & Brier Skill Score
report_lines.append("\n" + "-" * 50)
report_lines.append("Null Brier & Brier Skill Score")
report_lines.append("-" * 50)
for cohort, m in metrics_dict.items():
    if cohort in preds_dict:
        y_true, _ = preds_dict[cohort]
        prevalence = np.mean(y_true)
        null_brier = prevalence * (1 - prevalence)
        # BSS should use the *raw* Brier for Derivation (apparent, no overfit) and
        # the *calibrated* Brier for external validation sets.
        if cohort == "Derivation" and cohort in metrics_dict_raw:
            model_brier = metrics_dict_raw[cohort].get("Brier", np.nan)
        else:
            model_brier = m.get("Brier", np.nan)
        bss = 1 - (model_brier / null_brier) if null_brier > 0 else np.nan
        report_lines.append(f"\n  {cohort}:")
        report_lines.append(f"    Prevalence: {prevalence:.4f} ({int(np.sum(y_true==1))}/{len(y_true)})")
        report_lines.append(f"    Null Brier: {null_brier:.4f}")
        report_lines.append(f"    Model Brier: {model_brier:.4f}")
        report_lines.append(f"    Brier Skill Score (BSS): {bss:.4f}")

# Add raw (uncalibrated) metrics for external validation sets
report_lines.append("\n" + "-" * 50)
report_lines.append("Performance Metrics (Raw / Uncalibrated — External Sets Only)")
report_lines.append("-" * 50)
for cohort, m_raw in metrics_dict_raw.items():
    if cohort == "Derivation":
        continue
    report_lines.append(f"\n  {cohort} (Raw):")
    for k, v in m_raw.items():
        if isinstance(v, float):
            report_lines.append(f"    {k}: {v:.3f}")
        else:
            report_lines.append(f"    {k}: {v}")

report_lines.append("\n" + "-" * 50)
report_lines.append("Independent AUC Comparison (bootstrap z-test, independent samples)")
report_lines.append("Multiple testing correction: Benjamini-Hochberg FDR (used); "
                    "Bonferroni retained as sensitivity analysis.")
report_lines.append("-" * 50)
for r in auc_comparison_results:
    report_lines.append(f"  {r['Comparison']}: AUC diff={r['AUC_Diff']:.3f} "
                        f"({r['AUC_Diff_95CI_Lower']:.3f}-{r['AUC_Diff_95CI_Upper']:.3f}), "
                        f"Z={r['Z_Statistic']:.3f}, p={r['P_Value']:.3f} "
                        f"(BH-FDR: {r['P_Value_BH_FDR']:.3f}; "
                        f"Bonferroni: {r['P_Value_Bonferroni']:.3f})")

report_lines.append("\n" + "-" * 50)
report_lines.append("Hosmer-Lemeshow Test (g=10 bins, based on RAW uncalibrated probabilities; "
                    "fewer bins when n < 50)")
report_lines.append("-" * 50)
for r in hl_rows:
    report_lines.append(f"  {r['Cohort']}: Chi2={r['Chi2']:.3f}, df={r['DF']}, p={r['P_Value']:.3f}")

report_lines.append("")
report_lines.append("-" * 50)
report_lines.append("AUC CI 口径说明 (final_model_summary.csv)")
report_lines.append("-" * 50)
report_lines.append(
    "  Derivation / Optimism-Corrected (Nested-CV) 行的 95% CI = 重复嵌套 CV 的"
    "重复级经验标准误（均值 ± 1.96·SE，SE = SD(重复级 OOF AUC)/sqrt(N_REPEATS)），")
report_lines.append(
    "  反映对外层数据划分的敏感性，属经验标准误而非严格抽样置信区间。")
report_lines.append(
    "  外部队列（Temporal / Geographical, Raw 或 Recalibrated）行的 95% CI = BCa"
    " bootstrap（1000 次重抽 + jackknife），反映样本抽样不确定性。")
report_lines.append(
    "  两类区间含义不同，不可直接横向比较区间宽度；Applied_Cutoff（原 Optimal_Cutoff）"
    "为重复级中位数，各行实际施加的阈值来源见 Cutoff_Source 列。")
report_lines.append(
    "  Derivation / Optimism-Corrected 行的 AUPRC 及 95% CI = 对 04 导出的 repeat 1"
    "代表性 OOF 做 BCa bootstrap（非重复级均值），与 AUC 的重复级口径不同。")
report_lines.append(
    "  该 95% CI 为重复嵌套 CV 的重复级经验标准误（均值 ± 1.96×SE），反映外层数据划分的"
    "敏感性，并非严格意义上的抽样置信区间；外部队列采用 BCa bootstrap（反映样本抽样"
    "不确定性），两者含义不同，不可横向比较区间宽度。")

report_lines.append("")
report_lines.append("-" * 50)
report_lines.append("基于 TRIPOD+AI 声明（Collins et al., BMJ 2024;384:e078378）的条目域整理；")
report_lines.append("正式投稿前请对照期刊要求逐条核对编号与顺序。")
report_lines.append("TRIPOD+AI checklist coverage (script-level; 编号按条目域)")
report_lines.append("-" * 50)
for _ln in [
    "  3  数据来源/参与者        Covered      01_data_preprocessing.py（队列校验与描述统计）",
    "  4  预测因子               Covered      02_vif_analysis.py（相关性/VIF）+ 03_feature_selection.R（4 方法共识）",
    "  5  结局定义               Covered      01（label）; 05 报告事件数与患病率",
    "  6  样本量/EPV             Partially    仅报告 EPV 经验值，未做正式功效计算",
    "  7  缺失数据               Covered      01（mFIM 迭代插补，折内重放）",
    "  8  模型开发与调参         Covered      04_Model_Selection.py（Optuna 嵌套调参）",
    "  9  内部验证（乐观度）      Covered      04 重复嵌套 CV（10×5，外层独立重放全链路）",
    "  10 外部验证               Covered      05（Temporal/Geographical Raw+Platt）、07（盲法队列）",
    "  11 校准                   Covered      05/06（校准曲线、Slope/Intercept、Hosmer-Lemeshow、Brier）",
    "  12 判别性能               Covered      05/06（ROC、AUPRC、DCA、混淆矩阵）",
    "  13 AI 可解释性            Covered      05（SHAP summary/dependence + OR 森林图）",
    "  14 AI 公平性/亚组          Covered      05 Part 2（年龄/性别/HTN 亚组脆弱性）",
    "  15 可复现性               Partially    固定随机种子与 run_meta.txt；仍依赖本地 R/Python 环境",
    "  16 标题/摘要/讨论/注册/资金  Not in scripts（手稿层面内容，本流水线不生成）",
]:
    report_lines.append(_ln)

report_lines.append("")
report_lines.append("-" * 50)
report_lines.append("STARD-AI checklist coverage (script-level)")
report_lines.append("-" * 50)
for _ln in [
    "  1  研究目的与设计          Covered      本流水线的诊断/预后准确性评估框架",
    "  2  参与者纳入/排除        Covered      01_data_preprocessing.py",
    "  3  指标定义与参考标准      Covered      label 定义 + Derivation 重复级阈值（Applied_Cutoff）",
    "  4  样本量与事件数          Covered      05 报告 N/events/prevalence",
    "  5  数据处理与无泄漏        Covered      折内重放（log1p→插补→标准化→VIF→共识选择），见 04 bias_note",
    "  6  判别与校准              Covered      ROC/AUPRC/Brier/校准曲线/HL",
    "  7  决策阈值与临床效用      Covered      06 DCA / CNB / NIA",
    "  8  亚组与敏感性分析        Covered      05 Part 2；05b_NRI_IDI_Analysis.py",
    "  9  局限性/注册/资金来源    Not in scripts（手稿层面内容）",
]:
    report_lines.append(_ln)

report_path = os.path.join(RESULT_DIR, "model_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"  Report saved: {report_path}")

# =============================================================================
# 论文输出表格：外部验证集两套性能指标（Table 2 / eTable 1）
# =============================================================================
# 说明：
#   - Table2_Main_Raw_Validation.csv    : Raw（原始概率）外部验证集性能 —— 主正文 Table 2
#   - eTable1_Platt_Validation.csv      : Platt（Derivation 拟合的校准器）—— 补充 eTable 1
# External validation reports only Raw (Table 2) and Platt (eTable 1) metrics. Intercept
# recalibration is NOT performed because it uses the external cohort's own labels and would
# constitute partial in-sample fitting.
# =============================================================================

EXTERNAL_COHORT_KEYS = ["Temporal", "Geographical"]

# 组织两套表格行（列顺序固定，仅保留存在的列）
_TABLE2_COLS = ["Cohort", "Calibration_Status", "AUC", "AUC_CI_Lower", "AUC_CI_Upper",
                "Brier", "Sensitivity", "Specificity", "Accuracy", "F1", "PPV", "NPV",
                "Calib_Intercept", "Calib_Slope", "Calib_Slope_CI_Lower", "Calib_Slope_CI_Upper"]

def _subset_metrics(m, status, cohort):
    row = {"Cohort": cohort, "Calibration_Status": status}
    for k in _TABLE2_COLS[2:]:
        if k in m:
            row[k] = m[k]
    return row

_raw_rows, _platt_rows = [], []
for _cohort in EXTERNAL_COHORT_KEYS:
    if _cohort not in metrics_dict_raw or _cohort not in metrics_dict:
        continue
    _raw_rows.append(_subset_metrics(metrics_dict_raw[_cohort], "Raw", _cohort))
    _platt_rows.append(_subset_metrics(metrics_dict[_cohort], "Platt", _cohort))

def _write_table(rows, out_name):
    if not rows:
        print(f"  WARNING: no rows for {out_name}, skipping.")
        return
    cols = [c for c in _TABLE2_COLS if any(c in r for r in rows)]
    pd.DataFrame([{c: r.get(c, np.nan) for c in cols} for r in rows]).to_csv(
        os.path.join(RESULT_DIR, out_name), index=False, float_format='%.6f')
    print(f"  Saved: {out_name} ({len(rows)} rows)")

_write_table(_raw_rows, "Table2_Main_Raw_Validation.csv")
_write_table(_platt_rows, "eTable1_Platt_Validation.csv")

# =============================================================================
# Part 2 — 亚组脆弱性分析（原独立成长分析脚本的亚组部分，已并入本脚本）
# =============================================================================
# 说明：本部分直接复用内存中的模型、Derivation/Temporal 校准器与各队列预测概率，
#       不再二次读盘。模型成长曲线已按需求移除（避免与盲法验证的叙事重叠）。
#   2.1 亚组性能：精确反标准化年龄 + SCORE2 三分组 + Bootstrap AUC CI
#   2.2 亚组校准对比：Raw vs Recalibrated 的 Brier/Slope
#   2.3 Brier 改善棒棒糖图
# 输出目录：E:\new\Subgroup Analysis
# =============================================================================
SUBGROUP_DIR = os.path.join(BASE_DIR, "Subgroup Analysis")
PREPROC_DIR = os.path.join(BASE_DIR, "data_preprocessing")
os.makedirs(SUBGROUP_DIR, exist_ok=True)

print("\n" + "=" * 70)
print(f"Part 2: Subgroup Vulnerability Analysis — {model_name}")
print("=" * 70)

def compute_calibration_metrics(y_true, y_prob):
    """Brier + 校准截距/斜率（logit 尺度 GLM，含完全分离异常保护）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    brier = brier_score_loss(y_true, y_prob)

    y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
    logit = np.log(y_prob_clipped / (1 - y_prob_clipped))
    try:
        intercept = sm.GLM(y_true, np.ones((len(y_true), 1)),
                           family=sm.families.Binomial(), offset=logit).fit().params[0]
        slope = sm.GLM(y_true, sm.add_constant(logit),
                       family=sm.families.Binomial()).fit().params[1]
    except Exception as e:
        print(f"  WARNING: 校准 GLM 拟合失败 ({e})，intercept/slope 记为 NaN")
        intercept, slope = np.nan, np.nan
    return brier, intercept, slope


def load_age_scaler_params():
    """从 preprocessing_objects.joblib 读取 age 的 RobustScaler 参数。"""
    import joblib
    preproc_path = os.path.join(PREPROC_DIR, "preprocessing_objects.joblib")
    try:
        pp = joblib.load(preproc_path)
        age_median = pp['scaler_robust_params']['age']['median']
        age_iqr = pp['scaler_robust_params']['age']['iqr']
        is_age_logged = 'age' in pp.get('transformed_log_cols', [])
        print(f"  Age scaler: median={age_median}, iqr={age_iqr}, log_transformed={is_age_logged}")
        return age_median, age_iqr, is_age_logged
    except (KeyError, FileNotFoundError, AttributeError) as e:
        print(f"  WARNING: Could not load preprocessing_objects.joblib ({e}). "
              f"请先运行 01 数据预处理脚本生成该文件；"
              f"本次回退使用 fallback median=60, iqr=15（稍后将强制校验）")
        return 60.0, 15.0, False


def destandardize_age(age_scaled, age_median, age_iqr, is_age_logged):
    """反解 RobustScaler（必要时含 log1p）还原原始年龄（岁）。"""
    age_raw = np.asarray(age_scaled, dtype=np.float64) * age_iqr + age_median
    if is_age_logged:
        age_raw = np.expm1(age_raw)
    return age_raw


if not (has_geo and has_temporal):
    print("  WARNING: Geographical/Temporal cohort missing — "
          "skipping subgroup analysis.")
else:
    geo_ids = geo_df["ID"].values

    # -------------------------------------------------------------------------
    # 2.1 亚组性能（精确反标准化年龄 + SCORE2 三分组）
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2.1 Subgroup Performance — Geographical Set")
    print("-" * 70)

    # 地理集概率直接复用上文（Recalibrated = Temporal 校准器；Raw = 原始概率）
    y_prob_geo = preds_dict["Geographical"][1]
    y_prob_geo_raw = preds_dict_raw["Geographical"][1]
    geo_auc_overall = roc_auc_score(y_geo, y_prob_geo)
    print(f"  Overall Geo AUC (Calibrated) = {geo_auc_overall:.4f}")

    age_median, age_iqr, is_age_logged = load_age_scaler_params()

    orig_geo_path = os.path.join(BASE_DIR, "Geographical external validation cohort.csv")
    if not os.path.exists(orig_geo_path):
        orig_geo_path = os.path.join(PREPROC_DIR, "Geographical external validation cohort.csv")
    orig_geo = pd.read_csv(orig_geo_path)
    subgroup_cols_need = ["ID", "age", "gender", "hypertension"]
    subgroup_cols_avail = [c for c in subgroup_cols_need if c in orig_geo.columns]
    if "hypertension" not in orig_geo.columns:
        print("  WARNING: 'hypertension' column not found. HTN subgroups will be skipped.")
    orig_geo_sub = orig_geo[subgroup_cols_avail].copy()

    geo_eval = pd.DataFrame({
        "ID": geo_ids,
        "y_true": y_geo,
        "y_prob_calibrated": y_prob_geo,
        "y_prob_raw": y_prob_geo_raw,
    }).merge(orig_geo_sub, on="ID", how="left")

    # === 年龄反标准化（RobustScaler → 原始岁数） ===
    age_scaled = geo_df["age"].values.astype(np.float64)
    age_raw = destandardize_age(age_scaled, age_median, age_iqr, is_age_logged)
    # 合理性校验：scaler 参数回退会导致 SCORE2 分组静默算错，故强制拦截
    if age_raw.min() < 18 or age_raw.max() > 100 or not (40 <= age_raw.mean() <= 80):
        sys.exit(
            f"ERROR: 反标准化年龄越界 (min={age_raw.min():.1f}, "
            f"max={age_raw.max():.1f}, mean={age_raw.mean():.1f})，"
            f"超出合理范围（要求取值 18-100、均值 40-80）。"
            f"请确认已先运行 01 数据预处理脚本生成 preprocessing_objects.joblib。"
        )
    geo_eval["age_raw"] = age_raw
    print(f"  age_raw: min={age_raw.min():.1f}, max={age_raw.max():.1f}, "
          f"mean={age_raw.mean():.1f}")

    # === ESC SCORE2 三分类年龄分组 ===
    geo_eval["Age_Group"] = pd.cut(geo_eval["age_raw"], bins=[0, 50, 70, 200],
                                   labels=["<50", "50-69", ">=70"], right=False)
    print(f"  Age_Group distribution: {dict(geo_eval['Age_Group'].value_counts().sort_index())}")

    has_hypertension = "hypertension" in geo_eval.columns
    # gender 0=Male, 1=Female（用户确认）
    geo_eval["Gender"] = geo_eval["gender"].map({0: "Male", 1: "Female"})
    if has_hypertension:
        geo_eval["Hypertension"] = geo_eval["hypertension"].map({0: "No HTN", 1: "HTN"})

    age_labels = sorted(geo_eval["Age_Group"].dropna().unique())
    subgroup_defs = [(f"Age: {lbl}", geo_eval["Age_Group"] == lbl) for lbl in age_labels]
    subgroup_defs += [
        ("Gender: Female", geo_eval["Gender"] == "Female"),
        ("Gender: Male", geo_eval["Gender"] == "Male"),
    ]
    if has_hypertension:
        subgroup_defs += [
            ("HTN: No", geo_eval["Hypertension"] == "No HTN"),
            ("HTN: Yes", geo_eval["Hypertension"] == "HTN"),
        ]

    subgroup_results = []
    for sname, mask in subgroup_defs:
        n = int(mask.sum())
        y_sub = geo_eval.loc[mask, "y_true"].values
        y_prob_sub = geo_eval.loc[mask, "y_prob_calibrated"].values
        n_pos = int(np.sum(y_sub == 1))
        n_neg = int(np.sum(y_sub == 0))

        if n_pos >= 10 and n_neg >= 10:
            auc_est, auc_lo, auc_hi = bootstrap_auc_ci(y_sub, y_prob_sub)
            metric_type = "AUC"
            metric_val = auc_est
        else:
            metric_type = "N/A"
            metric_val, auc_lo, auc_hi = np.nan, np.nan, np.nan
            print(f"  Subgroup too small (n_pos={n_pos}, n_neg={n_neg}) — AUC not reported")

        subgroup_results.append({
            "Subgroup": sname, "N": n, "N_Pos": n_pos, "N_Neg": n_neg,
            "Metric_Type": metric_type, "Metric_Value": metric_val,
            "CI_Lower": auc_lo, "CI_Upper": auc_hi,
            "Signif_Worse": (metric_type == "AUC" and auc_hi < 0.5),
            "P_Value_vs_Random": np.nan,
        })

    subgroup_df = pd.DataFrame(subgroup_results)
    auc_mask = subgroup_df["Metric_Type"] == "AUC"
    n_auc_subgroups = int(auc_mask.sum())
    # 先全量初始化为 False：非 AUC 亚组（样本不足，Metric_Type == "N/A"）不参与
    # FDR 校正，若留空会写入 NaN，下游 06 读图时 ifelse(NA) 会导致颜色缺失
    subgroup_df["Signif_Worse_FDR"] = False
    subgroup_df["P_Value_Bonferroni"] = np.nan
    subgroup_df["Signif_Worse_Bonferroni"] = False
    if n_auc_subgroups > 0:
        for i, (sname, mask) in enumerate(subgroup_defs):
            if not auc_mask.iloc[i]:
                continue
            y_sub = geo_eval.loc[mask, "y_true"].values
            y_prob_sub = geo_eval.loc[mask, "y_prob_calibrated"].values
            n = len(y_sub)
            rng = np.random.RandomState(RANDOM_STATE)
            n_bs = 2000
            bs_auc = np.empty(n_bs)
            for b in range(n_bs):
                b_idx = rng.choice(n, size=n, replace=True)
                try:
                    bs_auc[b] = roc_auc_score(y_sub[b_idx], y_prob_sub[b_idx])
                except ValueError:
                    bs_auc[b] = np.nan
            bs_auc = bs_auc[~np.isnan(bs_auc)]
            if len(bs_auc) > 0:
                p_less = np.mean(bs_auc < 0.5)
                p_val = np.clip(2.0 * min(p_less, 1.0 - p_less), 0.0, 1.0)
            else:
                p_val = 1.0
            subgroup_df.loc[i, "P_Value_vs_Random"] = p_val

        raw_p_vals = subgroup_df.loc[auc_mask, "P_Value_vs_Random"].values
        reject, adj_p_vals, _, _ = multipletests(raw_p_vals, method='fdr_bh')
        subgroup_df.loc[auc_mask, "P_Value_FDR"] = adj_p_vals
        # 方向约束：仅在 FDR 显著且 AUC < 0.5 时判定"显著更差"，
        # 避免把 AUC 显著高于 0.5 的优良亚组误标为脆弱亚组。
        subgroup_df.loc[auc_mask, "Signif_Worse_FDR"] = (
            (adj_p_vals < 0.05)
            & (subgroup_df.loc[auc_mask, "Metric_Value"].values < 0.5)
        )
        adj_p_bonf = np.minimum(raw_p_vals * n_auc_subgroups, 1.0)
        subgroup_df.loc[auc_mask, "P_Value_Bonferroni"] = adj_p_bonf
        subgroup_df.loc[auc_mask, "Signif_Worse_Bonferroni"] = adj_p_bonf < 0.05

    # 布尔列以 "TRUE"/"FALSE" 大写写出：pandas 默认写 "True"/"False"，
    # 而 R 的 as.logical() 只认大写，06 端虽已 toupper 兜底，这里两端都保证一致
    subgroup_out = subgroup_df.copy()
    for _c in ("Signif_Worse", "Signif_Worse_FDR", "Signif_Worse_Bonferroni"):
        if _c in subgroup_out.columns:
            subgroup_out[_c] = subgroup_out[_c].astype(bool).astype(str).str.upper()
    subgroup_out.to_csv(os.path.join(SUBGROUP_DIR, f"Subgroup_Performance_{model_name}.csv"), index=False)

    auc_subgroups = subgroup_df[subgroup_df["Metric_Type"] == "AUC"].copy()
    if len(auc_subgroups) > 0:
        lowest = auc_subgroups.loc[auc_subgroups["Metric_Value"].idxmin()]
        print(f"\n>>> [脆弱亚组] 最低AUC亚组：{lowest['Subgroup']} "
              f"(AUC={lowest['Metric_Value']:.4f}, "
              f"95% CI: [{lowest['CI_Lower']:.4f}-{lowest['CI_Upper']:.4f}])")

    # -------------------------------------------------------------------------
    # 2.2 亚组校准对比（Raw vs Recalibrated）
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2.2 Subgroup Calibration Comparison — Geographical Set")
    print("-" * 70)
    print(f"  {'Subgroup':<25s}  {'Brier_Raw':>12s}  {'Brier_Recal':>12s}  {'Delta_Brier':>12s}")
    print("-" * 75)

    calib_results = []
    for sname, mask in subgroup_defs:
        n = int(mask.sum())
        y_sub = geo_eval.loc[mask, "y_true"].values
        y_prob_before = geo_eval.loc[mask, "y_prob_raw"].values
        y_prob_after = geo_eval.loc[mask, "y_prob_calibrated"].values

        n_pos = int(np.sum(y_sub == 1))
        n_neg = int(np.sum(y_sub == 0))
        if n_pos < 10 or n_neg < 10:
            calib_results.append({
                "Subgroup": sname, "N": n,
                "Slope_Before": np.nan, "Slope_After": np.nan, "Delta_Slope": np.nan,
                "Brier_Before": np.nan, "Brier_After": np.nan, "Delta_Brier": np.nan,
            })
            print(f"  {sname:<25s}  跳过 (n_pos={n_pos}, n_neg={n_neg}，样本量不足)")
            continue

        brier_before, inter_before, slope_before = compute_calibration_metrics(y_sub, y_prob_before)
        brier_after, inter_after, slope_after = compute_calibration_metrics(y_sub, y_prob_after)

        calib_results.append({
            "Subgroup": sname, "N": n,
            "Slope_Before": slope_before, "Slope_After": slope_after,
            "Delta_Slope": slope_after - slope_before,
            "Brier_Before": brier_before, "Brier_After": brier_after,
            "Delta_Brier": brier_after - brier_before,
        })
        delta_sign = "+" if (brier_after - brier_before) >= 0 else ""
        print(f"  {sname:<25s}  {brier_before:>12.4f}  {brier_after:>12.4f}  "
              f"{delta_sign}{brier_after - brier_before:>11.4f}")

    calib_df = pd.DataFrame(calib_results)
    calib_df.to_csv(os.path.join(SUBGROUP_DIR, "Subgroup_Calibration_Comparison.csv"), index=False)

    # -------------------------------------------------------------------------
    # 2.3 Brier 改善棒棒糖图
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2.3 Subgroup Brier Improvement — Lollipop Chart")
    print("-" * 70)

    calib_plot = (calib_df.dropna(subset=["Delta_Brier"])
                  .sort_values("Delta_Brier", ascending=True).reset_index(drop=True))
    n_sub_calib = len(calib_plot)
    if n_sub_calib == 0:
        print("  WARNING: no valid subgroup for Brier lollipop chart, skipping.")
    else:
        n_improved = int(np.sum(calib_plot["Delta_Brier"] < 0))
        point_colors = ["#2CA02C" if d < 0 else "#D62728" for d in calib_plot["Delta_Brier"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(n_sub_calib)
        for i in range(n_sub_calib):
            val = calib_plot["Delta_Brier"].iloc[i]
            ax.plot([0, val], [y_pos[i], y_pos[i]], "-", color="lightgray", lw=2, alpha=0.8, zorder=1)
            ax.scatter(val, y_pos[i], s=150, color=point_colors[i], zorder=3,
                       edgecolors="white", linewidth=0.5)

        ax.axvline(x=0, color="black", linestyle="--", lw=0.8, alpha=0.5, zorder=0)

        for i in range(n_sub_calib):
            val = calib_plot["Delta_Brier"].iloc[i]
            ax.text(val + 0.002, y_pos[i], f"{val:.4f}", fontsize=9, fontweight="bold",
                    ha="left", va="center", color=point_colors[i])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(calib_plot["Subgroup"].values, fontsize=10)
        ax.set_ylabel("")
        ax.set_xlabel("Change in Brier Score (Recalibrated - Raw)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2, axis="x")

        ax.text(0.02, 0.97, f"Brier improved in {n_improved}/{n_sub_calib} subgroups",
                transform=ax.transAxes, fontsize=10, fontweight="bold", color="#2CA02C",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", alpha=0.9))

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2CA02C",
                   markersize=10, label="Brier Improved"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#D62728",
                   markersize=10, label="Brier Worsened"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

        x_min = calib_plot["Delta_Brier"].min()
        x_max = calib_plot["Delta_Brier"].max()
        x_pad = max(abs(x_max - x_min) * 0.25, 0.01)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

        plt.tight_layout()
        save_figure(fig, "subgroup_brier_lollipop", SUBGROUP_DIR)
        print("  Lollipop chart saved.")

# =============================================================================
# Final
# =============================================================================
print("\n" + "=" * 70)
print(f"=== Model Development + Subgroup Analysis Complete — {model_name} ===")
print(f"  Model development outputs: {RESULT_DIR}")
print(f"  Subgroup analysis outputs: {SUBGROUP_DIR}")
print("=" * 70)
