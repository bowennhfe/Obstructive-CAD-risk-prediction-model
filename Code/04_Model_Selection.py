# =============================================================================
# 04_Model_Selection.py — 防泄漏严格嵌套 CV + OOF 评估 + 模型选择
# =============================================================================
# 【重要方法学修订（v2，防数据泄漏）】
# 旧版流程：01→02→03 在全部 649 例 Derivation 上一次性完成 log1p→RF 迭代插补
#   →RobustScaler→VIF→4 方法共识选择，之后才在外层折上跑嵌套 CV——外层测试折
#   的结局信息已通过特征选择进入建模（特征选择 leakage）。
# 新版流程（本脚本）：外层 5 折，每折内完整重放
#     派生比值(log 空间) → log1p(折内重算) → RF 迭代插补(mFIM, 仅 train 拟合)
#     → RobustScaler(仅 train 拟合) → VIF/相关性过滤(02 同阈值, 仅 train)
#     → 4 方法共识选择(subprocess 调 Rscript, 仅 train) → 内层 5 折 Optuna 调参
#     → train 全量重训 → 外层测试折仅 transform + predict。
#   所有统计量（skewness、插补、缩放、相关性、VIF、特征选择）仅在折内 train
#   估计；外层测试折只做 transform 与 predict，杜绝特征选择泄漏。
#
# 【重复嵌套 CV（v3）】
# 外层改为 N_REPEATS 次重复 × OUTER_FOLDS 折：每次重复使用不同的外层数据划分
#   （random_state = RANDOM_STATE + rep），共 N_REPEATS * OUTER_FOLDS 个外层测试折。
# 主性能估计与不确定性口径：
#   主估计 = N_REPEATS 个"重复级 OOF 指标"的均值
#   不确定性 = 重复级经验标准误 SE = SD(重复级指标) / sqrt(N_REPEATS)
#   区间 = 均值 ± 1.96 * SE  → 写入 AUC_CI_Lower / AUC_CI_Upper（下游列名不变）
#   注：该 SE 是重复级经验标准误，不是严格意义上的抽样置信区间。
# 重复嵌套 CV 降低单次划分带来的方差，但不消除针对被选中模型的残余
#   winner's curse；无偏口径仍以嵌套的 InnerSel 混合 OOF 为准。
#
# 口径说明：model_selection_metrics.csv 中 Sensitivity/Specificity/Accuracy/
#       F1/PPV/NPV 等阈值化指标：每次重复用该重复 OOF 的 Youden cutoff 计算，
#       再对 N_REPEATS 次取均值（属于 apparent threshold performance，
#       解释时需注意轻度乐观偏倚）。Optimal_Cutoff 取重复级 cutoff 的中位数。
#
# 输入：
#   - <repo 根>/Derivation cohort.csv（原始、未预处理、含缺失——折内重放用）
#   - <repo>/feature_selection/Derivation cohort.csv（官方面板，仅用于
#     tune_model_on_full_training 生成 best_params，供 05/07 最终模型与
#     外部验证流程使用；不参与外层 OOF 评估）
# 输出（均在 E:\new\Model Selection）：
#   - oof_predictions_all_models.csv（列: model, fold, y_true, y_pred_prob）
#     内容取 repeat 1 的代表性 OOF（每例一行），下游 05/06/07 可直接读
#   - nested_cv_leakage_free_oof.csv（同上，防泄漏版命名副本）
#   - oof_predictions_all_repeats.csv（全量: model, repeat, fold, y_true, y_pred_prob）
#   - model_selection_metrics.csv（重复级均值 + SE 口径，列名向下兼容）
#   - model_selection_metrics_per_repeat.csv（10 重复 × 9 模型 = 90 行，供核对）
#   - nested_cv_repeat_auc_summary.csv（Model, Repeat, AUC, Brier, Optimal_Cutoff）
#   - nested_cv_oof_bca_ci.csv（repeat 1 代表性 OOF 的 BCa CI，仅作对照）
#   - model_selection_stability.csv（逐 repeat 逐 fold 内层/外层胜出模型）
#   - model_selection_stability_summary.csv（各重复一致比例 + 汇总行）
#   - nested_cv_innersel_oof.csv / nested_cv_innersel_metrics.csv
#     （折内自选模型的混合 OOF 及其重复级聚合——对"自动选型流程"的无偏估计）
#   - model_selection_bias_note.txt（三层偏倚口径说明）
#   - foldwise_selected_features.csv / fold_feature_frequency.csv（逐折共识特征+频率）
#   - roc_curves_all_models.png / .pdf / .tif
#   - best_params_all_models.pkl
# 折级中间产物（E:\new\Nested CV Work\repeat_r\fold_k\，不入 git）：
#   - repeat{r}_fold{k}_train_vif.csv / _test_transformed.csv（R 桥输入/折内测试折）
#   - repeat{r}_fold{k}_selected_features.txt / r_bridge_log.txt（R 输出与日志）
#   - repeat{r}_fold{k}_replay_log.json / _oof_predictions.csv（审计用）
#
# 全局开关：
#   N_REPEATS = 10, OUTER_FOLDS = 5, INNER_TRIALS = 50, INNER_FOLDS = 5
# 运行成本提示：总计算量约为单次 5 折嵌套 CV 的 N_REPEATS 倍——R 特征选择
#   subprocess 调用从 5 次变为 N_REPEATS * OUTER_FOLDS 次（默认 50 次），
#   折级中间件磁盘占用同样放大 N_REPEATS 倍。模型固定 9 种（已移除 MLP）。
# =============================================================================

import os
import sys
import gc
import json
import shutil
import subprocess
import time
import pickle
import traceback
import warnings
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
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, average_precision_score)
from sklearn.exceptions import ConvergenceWarning

import optuna
from optuna.samplers import TPESampler

# 折内重放共享库（与 01/02 数值逻辑逐字对齐）
from pipeline_lib import (
    find_raw_cohort_path, replay_preprocess, r_make_names,
    NOISE_FEATURE, NUMERIC_COLS, CATEGORICAL_COLS,
)

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
N_REPEATS = 10     # 重复嵌套 CV 次数（外层重复；每次重复使用不同的外层数据划分）
# 首次试跑建议临时改为 2，验证目录结构/聚合逻辑/下游 05-07 读取正常后再改回 10。
OUTER_FOLDS = 5
INNER_TRIALS = 50  # 多目标优化需要更多 trials 以覆盖帕累托前沿
INNER_FOLDS = 5
BOOTSTRAP_ITER = 1000  # 仅用于「OOF 级 BCa CI」对照文件，不进入主指标表

# 命令行 --fast：冒烟测试模式（必须在任何函数调用/耗时流程之前覆盖模块级常量）
if "--fast" in sys.argv:
    N_REPEATS = 2
    INNER_TRIALS = 10
    print("  FAST MODE: N_REPEATS=2, INNER_TRIALS=10 (smoke test)")

RANDOM_STATE = 123
N_JOBS = max(1, (os.cpu_count() or 1) // 2)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "feature_selection")
RESULT_DIR = os.path.join(BASE_DIR, "Model Selection")
WORK_DIR = os.path.join(BASE_DIR, "Nested CV Work")  # 折级中间件（git-ignore）
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# R 桥配置（禁用 rpy2，仅 subprocess 文件桥）
R_SCRIPT = os.path.join(CODE_DIR, "03_feature_selection.R")
R_TIMEOUT_HOURS = 6  # 单折 R 特征选择超时（Boruta/bestglm/mRMRe 较慢，留足余量）

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
print(f"  N_REPEATS: {N_REPEATS}, OUTER_FOLDS: {OUTER_FOLDS}, "
      f"INNER_FOLDS: {INNER_FOLDS}, INNER_TRIALS: {INNER_TRIALS}")
print(f"  RANDOM_STATE: {RANDOM_STATE} "
      f"(outer seeds: {RANDOM_STATE}..{RANDOM_STATE + N_REPEATS - 1})")
print(f"  Total outer test folds: {N_REPEATS * OUTER_FOLDS} "
      f"| R feature-selection calls: {N_REPEATS * OUTER_FOLDS}")
print(f"  Estimated wall-clock: ~{0.3 * N_REPEATS * OUTER_FOLDS:.1f} - {1.5 * N_REPEATS * OUTER_FOLDS:.1f} hours (R feature selection + Optuna {INNER_TRIALS} trials x {INNER_FOLDS} folds)")
print(f"  NOTE: 总计算量与磁盘占用约为单次 5 折嵌套 CV 的 {N_REPEATS} 倍；")
print(f"        折级中间件请定期清理 {WORK_DIR}")
print(f"  Data dir (official panel, best-params only): {DATA_DIR}")
print(f"  Work dir (fold intermediates): {WORK_DIR}")
print(f"  R script: {R_SCRIPT}")

# =============================================================================
# Model Registry — 9 分类器（MLP 已按方案移除，04 与全部下游一致）
# =============================================================================
MODEL_LIST = [
    "L2_LR",
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
# Optuna Search Spaces
# =============================================================================
def suggest_params(trial, model_name):
    if model_name == "L2_LR":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
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
    elif model_name == "KNN":
        return {"n_jobs": N_JOBS}
    elif model_name == "DT":
        return {"random_state": RANDOM_STATE}
    elif model_name == "RF":
        return {"random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    elif model_name == "LightGBM":
        # 早停通过 fit 的 callbacks=[lgb.early_stopping(50)] 施加（见调用点）
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


def _fit_with_optional_early_stopping(clf, model_name, X_tr_df, y_tr):
    """非树模型直接全量拟合；LightGBM/XGBoost/CatBoost 用分层 20% eval set 早停
    （与旧版 04 的重训逻辑一致）。"""
    if model_name in ("LightGBM", "XGBoost", "CatBoost"):
        X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
            X_tr_df, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr
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
        else:  # CatBoost
            clf.fit(X_tr_sub, y_tr_sub,
                    eval_set=(X_val_sub, y_val_sub),
                    verbose=False)
    else:
        clf.fit(X_tr_df, y_tr)
    return clf


# =============================================================================
# 重复级指标（重复嵌套 CV 口径）
# =============================================================================
# 每个"重复"产生一套完整 OOF（5 个外层测试折拼接），在该重复内计算一次指标；
# 主估计 = 重复级指标的均值，不确定性 = 重复级经验标准误 SE = SD / sqrt(N_REPEATS)，
# 区间 = 均值 ± 1.96 * SE。故此处不再对单一 OOF 做 bootstrap，避免与重复级
# 标准误口径混用。
METRIC_KEYS = ["AUC", "AUPRC", "Brier", "Optimal_Cutoff", "Sensitivity",
               "Specificity", "Accuracy", "F1", "PPV", "NPV"]


# =============================================================================
# OOF 级 BCa CI（仅作对照，不进入主指标表）
# =============================================================================
# 主指标表沿用重复级均值 ± 1.96*SE 口径；此处额外对「repeat 1 的代表性 OOF」
# 计算 BCa bootstrap 区间，写入 nested_cv_oof_bca_ci.csv，供与重复级 SE 并行
# 核对（两种区间的含义不同：BCa 反映样本抽样不确定性，重复级 SE 反映外层
# 数据划分的敏感性）。
def bootstrap_auc_ci_bca(y_true, y_prob, n_iter=BOOTSTRAP_ITER, alpha=0.05,
                         random_state=RANDOM_STATE):
    """BCa (bias-corrected and accelerated) bootstrap CI for AUC. """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    try:
        theta_hat = roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan, np.nan, np.nan
    if np.isnan(theta_hat):
        return np.nan, np.nan, np.nan

    # Jackknife（估计 acceleration）
    jack = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            jack[i] = roc_auc_score(y_true[mask], y_prob[mask])
        except ValueError:
            jack[i] = theta_hat
    jack_mean = np.mean(jack)
    diff = jack_mean - jack
    sum_sq = np.sum(diff ** 2)
    a = 0.0 if sum_sq == 0 else np.sum(diff ** 3) / (6.0 * sum_sq ** 1.5)

    # Bootstrap
    rng = np.random.RandomState(random_state)
    boot = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot[b] = roc_auc_score(y_true[idx], y_prob[idx])
        except ValueError:
            boot[b] = theta_hat

    prop_less = np.clip(np.mean(boot < theta_hat), 0.001, 0.999)
    z0 = norm.ppf(prop_less)
    z_lo, z_hi = norm.ppf(alpha / 2.0), norm.ppf(1.0 - alpha / 2.0)
    a1 = np.clip(norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo))), 0.001, 0.999)
    a2 = np.clip(norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi))), 0.001, 0.999)
    return (float(np.mean(boot)),
            float(np.percentile(boot, 100.0 * a1)),
            float(np.percentile(boot, 100.0 * a2)))



# =============================================================================
# R Bridge — subprocess 文件桥调用 03_feature_selection.R（禁用 rpy2）
# =============================================================================
def run_r_feature_selection(csv_in, txt_out, fold_dir):
    """调用 Rscript 03_feature_selection.R <in.csv> <out.txt>，返回共识特征列表（R 点名）。

    R 端以桥接模式运行：关闭绘图、跳过验证队列改写；stdout/stderr 落盘到
    r_bridge_log.txt 供审计。任何失败立即抛错（防泄漏流程不允许静默回退）。
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "未找到 Rscript 可执行文件。请确认 R 已安装并在 PATH 中"
            "（本流程仅允许 subprocess 文件桥，不使用 rpy2）。")
    cmd = [rscript, "--vanilla", R_SCRIPT, str(csv_in), str(txt_out)]
    log_path = os.path.join(fold_dir, "r_bridge_log.txt")
    print(f"  [R bridge] {' '.join(cmd)}")
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  cwd=str(fold_dir), timeout=R_TIMEOUT_HOURS * 3600)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"R feature selection timed out (>{R_TIMEOUT_HOURS}h, cwd={fold_dir}). "
            f"Consider reducing Boruta maxRuns / feature count.\n{e}"
        )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-40:])
        except OSError:
            pass
        raise RuntimeError(
            f"R 特征选择失败（returncode={proc.returncode}，耗时 {elapsed:.0f}s）。\n"
            f"日志尾部：\n{tail}")
    if not os.path.exists(txt_out):
        raise RuntimeError(f"R 未产出特征文件: {txt_out}（详见 {log_path}）")
    with open(txt_out, "r", encoding="utf-8") as f:
        feats = [ln.strip() for ln in f.readlines() if ln.strip()]
    print(f"  [R bridge] done in {elapsed:.0f}s, {len(feats)} features selected")
    return feats


def _pareto_best_trial(study, context):
    """多目标（AUC↑ Brier↓）Pareto 前沿上取 AB_Score 最大者（与旧版一致）。"""
    try:
        pareto_trials = study.get_pareto_front_trials()
    except AttributeError:
        # optuna 4.9.0 compatibility: best_trials is the correct API
        pareto_trials = study.best_trials
    if len(pareto_trials) == 0:
        # 多目标 study 不存在单一 best_trial，改为从全部完成态 trial 中按 AB_Score 兜底
        print(f"    Warning: {context} Pareto front empty, "
              f"falling back to AB_Score max over completed trials.")
        fallback_trials = [t for t in study.trials if t.values is not None]
        if len(fallback_trials) == 0:
            raise RuntimeError(f"{context}: no completed Optuna trials with valid values")
        pareto_trials = [max(fallback_trials,
                             key=lambda t: 0.5 * t.values[0] + 0.5 * (1 - t.values[1]))]
    best_trial = max(pareto_trials, key=lambda t: 0.5 * t.values[0] + 0.5 * (1 - t.values[1]))
    return best_trial


def _strip_optuna_internal_keys(params):
    params = params.copy()
    for k in ("gamma_use_str", "gamma_str", "gamma_float"):
        params.pop(k, None)
    return params


# =============================================================================
# Inner Tuning（折内 5 折 Optuna，仅使用折内 train 的共识特征子集）
# =============================================================================
def tune_on_fold_train(model_name, X_tr, y_tr, feature_names):
    """在折内 train 上做内层 CV 调参，返回 (Pareto-AB_Score 最优参数, 内层 AB_Score)。

    内层 AB_Score 完全由折内 train 估计（未接触外层测试折），可用于
    折内模型自选（防模型选择偏倚的混合 OOF 敏感性分析）。
    """
    def objective(trial):
        params = suggest_params(trial, model_name)
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        briers = []
        for tr_idx, ev_idx in inner_cv.split(X_tr, y_tr):
            try:
                X_in_tr_df = pd.DataFrame(X_tr[tr_idx], columns=feature_names)
                X_in_ev_df = pd.DataFrame(X_tr[ev_idx], columns=feature_names)
                clf = build_model(model_name, params)
                # ---- 树模型统一使用早停（早停参数见 _get_fixed_params）----
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
                    else:  # CatBoost
                        clf.fit(X_in_tr_df, y_tr[tr_idx],
                                eval_set=(X_in_ev_df, y_tr[ev_idx]),
                                verbose=False)
                else:
                    clf.fit(X_in_tr_df, y_tr[tr_idx])
                proba = clf.predict_proba(X_in_ev_df)[:, 1]
                aucs.append(roc_auc_score(y_tr[ev_idx], proba))
                briers.append(brier_score_loss(y_tr[ev_idx], proba))
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
    study.optimize(objective, n_trials=INNER_TRIALS, n_jobs=1, show_progress_bar=False)

    best_trial = _pareto_best_trial(study, f"[{model_name}] fold-train tuning")
    ab_score = 0.5 * best_trial.values[0] + 0.5 * (1 - best_trial.values[1])
    print(f"    Pareto-selected: AUC={best_trial.values[0]:.4f}, "
          f"Brier={best_trial.values[1]:.4f}, AB_Score={ab_score:.4f}")
    return _strip_optuna_internal_keys(best_trial.params), float(ab_score)


def tune_model_on_full_training(model_name, X, y, feature_names):
    """官方面板（feature_selection/Derivation cohort.csv）上的全量调参。

    仅用于生成 best_params_all_models.pkl（05/07 最终模型与外部验证流程），
    不参与外层 OOF 评估——外部验证集与 Derivation 独立，不存在泄漏。
    """
    def objective(trial):
        params = suggest_params(trial, model_name)
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        briers = []
        for tr_idx, ev_idx in inner_cv.split(X, y):
            try:
                X_in_tr_df = pd.DataFrame(X[tr_idx], columns=feature_names)
                X_in_ev_df = pd.DataFrame(X[ev_idx], columns=feature_names)
                clf = build_model(model_name, params)
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
                    else:  # CatBoost
                        clf.fit(X_in_tr_df, y[tr_idx],
                                eval_set=(X_in_ev_df, y[ev_idx]),
                                verbose=False)
                else:
                    clf.fit(X_in_tr_df, y[tr_idx])
                proba = clf.predict_proba(X_in_ev_df)[:, 1]
                aucs.append(roc_auc_score(y[ev_idx], proba))
                briers.append(brier_score_loss(y[ev_idx], proba))
                try:
                    del X_in_tr_df, X_in_ev_df, clf, proba
                except NameError:
                    pass
            except Exception as e:
                print(f"WARNING: [{model_name}] inner trial failed: {type(e).__name__}: {e}")
                return 0.0, 0.5
        mean_auc = np.mean(aucs) if aucs else 0.0
        mean_brier = np.mean(briers) if briers else 0.5
        return mean_auc, mean_brier

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=INNER_TRIALS, n_jobs=1, show_progress_bar=False)

    best_trial = _pareto_best_trial(study, f"[{model_name}] full-train tuning")
    ab_score = 0.5 * best_trial.values[0] + 0.5 * (1 - best_trial.values[1])
    print(f"  Full-train Pareto-selected: AUC={best_trial.values[0]:.4f}, "
          f"Brier={best_trial.values[1]:.4f}, AB_Score={ab_score:.4f}")
    return _strip_optuna_internal_keys(best_trial.params)


# =============================================================================
# 重复级 / 聚合指标计算
# =============================================================================
def compute_repeat_metrics(y_true, y_prob):
    """单个重复的 OOF 指标（无 bootstrap；重复级 SE 由聚合函数给出）。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        return {k: np.nan for k in METRIC_KEYS}

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = np.nan
    brier = brier_score_loss(y_true, y_prob)

    # 该重复 OOF 上的 Youden cutoff（阈值化指标属 apparent threshold performance）
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = int(np.argmax(tpr - fpr))
    cutoff = thresholds[best_idx] if 0 <= best_idx < len(thresholds) else 0.5
    # roc_curve 首个阈值恒为 inf，若 Youden 恰好选中该点需防护，回退 0.5
    if not np.isfinite(cutoff):
        cutoff = 0.5

    y_pred = (y_prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "AUC": auc,
        "AUPRC": auprc,
        "Brier": brier,
        "Optimal_Cutoff": float(cutoff),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "PPV": precision_score(y_true, y_pred, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
    }


def aggregate_repeat_metrics(repeat_metrics_list):
    """对每个指标列计算重复级均值、SD、SE（SE = SD / sqrt(n_valid_repeats)）。"""
    agg = {}
    for k in METRIC_KEYS:
        vals = np.array([m.get(k, np.nan) for m in repeat_metrics_list], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            agg[k] = agg[k + "_SD"] = agg[k + "_SE"] = np.nan
            agg[k + "_N"] = 0
            continue
        agg[k] = float(np.mean(vals))
        agg[k + "_SD"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg[k + "_SE"] = (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                          if len(vals) > 1 else 0.0)
        agg[k + "_N"] = int(len(vals))
    # Optimal_Cutoff 用中位数（比均值稳健），另给 SD
    cut = np.array([m.get("Optimal_Cutoff", np.nan) for m in repeat_metrics_list], dtype=float)
    cut = cut[np.isfinite(cut)]
    agg["Optimal_Cutoff"] = float(np.median(cut)) if len(cut) else np.nan
    agg["Optimal_Cutoff_SD"] = float(np.std(cut, ddof=1)) if len(cut) > 1 else np.nan
    return agg


def build_metrics_row(label, repeat_metrics, fold_aucs=None, fold_briers=None):
    """把重复级指标聚合为一行模型指标（列名向下兼容 05/06/07）。"""
    agg = aggregate_repeat_metrics(repeat_metrics)
    row = {"Model": label}
    row["AUC"] = agg["AUC"]
    row["AUC_SD"] = agg["AUC_SD"]
    row["AUC_SE"] = agg["AUC_SE"]
    row["AUC_CI_Lower"] = (row["AUC"] - 1.96 * row["AUC_SE"]
                           if np.isfinite(row["AUC_SE"]) else np.nan)
    row["AUC_CI_Upper"] = (row["AUC"] + 1.96 * row["AUC_SE"]
                           if np.isfinite(row["AUC_SE"]) else np.nan)
    row["AUC_N_ValidRepeats"] = agg["AUC_N"]
    row["AUC_PerRepeat"] = ", ".join(
        f"{m.get('AUC', np.nan):.4f}" for m in repeat_metrics)

    row["AUPRC"] = agg["AUPRC"]
    row["AUPRC_SD"] = agg["AUPRC_SD"]
    row["AUPRC_SE"] = agg["AUPRC_SE"]
    row["AUPRC_CI_Lower"] = (row["AUPRC"] - 1.96 * row["AUPRC_SE"]
                             if np.isfinite(row["AUPRC_SE"]) else np.nan)
    row["AUPRC_CI_Upper"] = (row["AUPRC"] + 1.96 * row["AUPRC_SE"]
                             if np.isfinite(row["AUPRC_SE"]) else np.nan)

    row["Brier"] = agg["Brier"]
    row["Brier_SD"] = agg["Brier_SD"]
    row["Brier_SE"] = agg["Brier_SE"]
    row["Optimal_Cutoff"] = agg["Optimal_Cutoff"]
    row["Optimal_Cutoff_SD"] = agg["Optimal_Cutoff_SD"]
    for k in ("Sensitivity", "Specificity", "Accuracy", "F1", "PPV", "NPV"):
        row[k] = agg[k]
        row[k + "_SD"] = agg[k + "_SD"]

    # CV_AUC_* 语义 = 重复级 OOF AUC 的均值/SD（与 AUC / AUC_SD 同值，保留旧列名）
    row["CV_AUC_Mean"] = row["AUC"]
    row["CV_AUC_SD"] = row["AUC_SD"]
    row["CV_AUC_PerRepeat"] = row["AUC_PerRepeat"]
    row["CV_AUC_PerFold"] = (", ".join(f"{a:.4f}" if np.isfinite(a) else "nan"
                                       for a in fold_aucs)
                             if fold_aucs is not None else "")
    row["CV_Brier_Mean"] = (float(np.nanmean(fold_briers)) if fold_briers else np.nan)
    row["CV_Brier_SD"] = (float(np.nanstd(fold_briers)) if fold_briers else np.nan)
    row["AB_Score"] = 0.5 * row["AUC"] + 0.5 * (1 - row["Brier"])
    return row


# =============================================================================
# Save Figure Helper & ROC Curves
# =============================================================================
def save_figure(fig, base_name, result_dir=RESULT_DIR):
    """Save figure in PNG, PDF, TIF formats (DPI=600)."""
    for fmt in ["png", "pdf", "tif"]:
        fpath = os.path.join(result_dir, f"{base_name}.{fmt}")
        fig.savefig(fpath, dpi=IMG_DPI, format=fmt, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)


def save_roc_curves(oof_data_dict, result_dir):
    """Draw ROC curves (one per model) using repeat-1 representative OOF.

    图例中的 AUC 与区间是 N_REPEATS 次重复的均值 ± 1.96*重复级 SE；
    曲线本身来自 repeat 1 的代表性 OOF（与 oof_predictions_all_models.csv 一致）。
    """
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
                      f"(±1.96SE {auc_lo:.3f}-{auc_hi:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("1 - Specificity", fontsize=FONT_LABEL)
    ax.set_ylabel("Sensitivity", fontsize=FONT_LABEL)
    ax.set_title(f"ROC Curves — All Models (Leakage-Free OOF, repeat 1 of {N_REPEATS})",
                 fontsize=FONT_TITLE)
    ax.legend(loc="lower right", fontsize=FONT_LEGEND, frameon=True)
    ax.grid(True, alpha=0.3)

    save_figure(fig, "roc_curves_all_models", result_dir)
    print(f"  ROC curves saved to {result_dir}/roc_curves_all_models.*")


# =============================================================================
# Main: 防泄漏严格嵌套 CV
# =============================================================================
print("\n" + "=" * 70)
print("=== Leakage-Free Nested CV（折内重放 + R 特征选择文件桥）===")
print("=" * 70)

# ---------------------------------------------------------------------------
# Step 1: 加载原始（未预处理）Derivation cohort —— 外层折切分与折内重放的数据源
# ---------------------------------------------------------------------------
print("\n=== Step 1: Loading RAW Derivation cohort (unprocessed, with NaN) ===")
RAW_DATA_PATH = find_raw_cohort_path(BASE_DIR)
raw_df = pd.read_csv(RAW_DATA_PATH)
print(f"  Raw data: {RAW_DATA_PATH}  shape={raw_df.shape}")

_required_cols = ["ID", "label"]
_missing = [c for c in _required_cols + NUMERIC_COLS + CATEGORICAL_COLS if c not in raw_df.columns]
if _missing:
    raise ValueError(f"原始数据缺少必需列: {_missing}")

y_raw = raw_df["label"].values.astype(np.int32)
if not set(np.unique(y_raw)) <= {0, 1}:
    raise ValueError(f"label 列必须为 0/1 二值，实际取值: {np.unique(y_raw)}")
if raw_df["ID"].duplicated().any():
    raise ValueError("原始数据 ID 列存在重复值——请检查数据完整性")
print(f"  Pos/Neg: {np.sum(y_raw == 1)}/{np.sum(y_raw == 0)}")

# 官方面板（01→02→03 冻结产物）——仅用于 best_params 全量调参
OFFICIAL_PANEL_PATH = os.path.join(DATA_DIR, "Derivation cohort.csv")

# ---------------------------------------------------------------------------
# Step 2: 外层重复 × 折切分
#   10 次重复 = 10 套不同的外层数据划分（random_state = RANDOM_STATE + rep），
#   每次重复内部再切 OUTER_FOLDS 折 → 共 N_REPEATS * OUTER_FOLDS 个外层测试折。
#   关键前提：random_state 必须随 rep 变化，否则"重复"只是同一划分跑 10 遍。
# ---------------------------------------------------------------------------
outer_splits_by_repeat = []   # list[list[(tr_idx, te_idx)]]
for rep in range(N_REPEATS):
    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + rep,   # 必须随 rep 变化
    )
    outer_splits_by_repeat.append(list(outer_cv.split(raw_df, y_raw)))
print(f"\nOuter splits prepared: {N_REPEATS} repeats x {OUTER_FOLDS} folds "
      f"(StratifiedKFold, shuffle=True, seeds "
      f"{RANDOM_STATE}..{RANDOM_STATE + N_REPEATS - 1})")

# results[model] 按"重复级"组织：
#   fold_aucs / fold_briers / fold_cutoffs / inner_ab_scores:
#       长度 N_REPEATS*OUTER_FOLDS（按 repeat-major 顺序），仅审计/兼容用
#   oof_by_repeat: 长度 N_REPEATS，每个元素是该重复 5 个测试折拼接的 OOF DataFrame
#   repeat_metrics: 长度 N_REPEATS，每个元素是该重复 OOF 的指标 dict
results = {m: {"fold_aucs": [], "fold_briers": [], "fold_cutoffs": [],
               "inner_ab_scores": [], "oof_by_repeat": [], "repeat_metrics": []}
           for m in MODEL_LIST}
fold_feature_rows = []  # foldwise_selected_features.csv 的长表行（含 repeat 列）

# ---------------------------------------------------------------------------
# Step 3: 外层重复 × 折双层循环
#   每折内重放预处理链 → R 共识选择 → 内层调参 → 重训 → 预测
# ---------------------------------------------------------------------------
for rep_idx, outer_splits in enumerate(outer_splits_by_repeat, 1):
    repeat_records = {m: [] for m in MODEL_LIST}
    print("\n" + "=" * 70)
    print(f"=== REPEAT {rep_idx}/{N_REPEATS}  (outer seed = {RANDOM_STATE + rep_idx - 1}) ===")
    print("=" * 70)

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, 1):
        fold_dir = os.path.join(WORK_DIR, f"repeat_{rep_idx}", f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_tag = f"R{rep_idx}F{fold_idx}"
        tag = f"repeat{rep_idx}_fold{fold_idx}"
        print("\n" + "#" * 70)
        print(f"### REPEAT {rep_idx}/{N_REPEATS} — OUTER FOLD {fold_idx}/{OUTER_FOLDS}")
        print(f"#   train n={len(tr_idx)}, test n={len(te_idx)}, workdir={fold_dir}")
        print("#" * 70)

        # --- 3a. 折内重放：log1p → mFIM → RobustScaler → VIF（仅 train 拟合） ---
        train_raw = raw_df.iloc[tr_idx].reset_index(drop=True)
        test_raw = raw_df.iloc[te_idx].reset_index(drop=True)
        t0 = time.time()
        train_final, test_final, meta = replay_preprocess(train_raw, test_raw, fold_tag=fold_tag)
        print(f"  [replay] done in {time.time() - t0:.0f}s | "
              f"train features (incl. noise): {len(meta['final_cols_with_noise'])}")

        # 折级审计落盘
        train_csv = os.path.join(fold_dir, f"{tag}_train_vif.csv")
        test_csv = os.path.join(fold_dir, f"{tag}_test_transformed.csv")
        train_final.to_csv(train_csv, index=False)
        test_final.to_csv(test_csv, index=False)
        with open(os.path.join(fold_dir, f"{tag}_replay_log.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=float)

        # --- 3b. R 特征选择文件桥（4 方法共识，仅折内 train） ---
        feats_txt = os.path.join(fold_dir, f"{tag}_selected_features.txt")
        selected_r = run_r_feature_selection(train_csv, feats_txt, fold_dir)

        # R 点名（check.names 点名）→ 原始列名；noise_feature 不进入建模特征
        name_map = {r_make_names(c): c for c in train_final.columns}
        selected = []
        for f in selected_r:
            if f == r_make_names(NOISE_FEATURE):
                print(f"  WARNING: noise_feature 被 R 共识选中（随机负对照），已从建模特征剔除")
                continue
            if f not in name_map:
                raise RuntimeError(f"R 选出的特征 '{f}' 无法映射回折内 train 列名")
            selected.append(name_map[f])
        if not selected:
            raise RuntimeError(f"repeat {rep_idx} fold {fold_idx}: R 共识特征为空")
        print(f"  Consensus features ({len(selected)}): {selected}")
        fold_feature_rows.extend(
            [{"repeat": rep_idx, "fold": fold_idx, "feature": f} for f in selected])

        # --- 3c. 折内矩阵（共识特征子集；外层测试折只做 transform 后的取数） ---
        X_tr_full = train_final[selected].values.astype(np.float64)
        y_tr_full = train_final["label"].values.astype(np.int32)
        X_te = test_final[selected].values.astype(np.float64)
        y_te = test_final["label"].values.astype(np.int32)
        ids_te = test_final["ID"].values
        print(f"  Train Pos/Neg: {np.sum(y_tr_full == 1)}/{np.sum(y_tr_full == 0)}"
              f" | test Pos/Neg: {np.sum(y_te == 1)}/{np.sum(y_te == 0)}")

        # --- 3d. 9 模型逐个：内层调参 → 重训 → 预测外层测试折 ---
        fold_pred_records = []  # 折级审计（含 ID）
        for model_name in MODEL_LIST:
            if model_name == "LightGBM" and not HAS_LGBM:
                print(f"  SKIPPED: lightgbm not installed")
                continue
            if model_name == "CatBoost" and not HAS_CATBOOST:
                print(f"  SKIPPED: catboost not installed")
                continue
            if model_name == "XGBoost" and not HAS_XGB:
                print(f"  SKIPPED: xgboost not installed")
                continue

            print(f"  --- [{model_name}] inner tuning ({INNER_FOLDS}x{INNER_TRIALS}) ---")
            try:
                t0 = time.time()
                best_params, inner_ab = tune_on_fold_train(model_name, X_tr_full, y_tr_full, selected)

                # 折内 train 全量重训（树模型早停用 train 内分层 20% eval set）
                X_tr_df = pd.DataFrame(X_tr_full, columns=selected)
                X_te_df = pd.DataFrame(X_te, columns=selected)
                best_clf = build_model(model_name, best_params)
                best_clf = _fit_with_optional_early_stopping(best_clf, model_name, X_tr_df, y_tr_full)
                y_te_prob = best_clf.predict_proba(X_te_df)[:, 1]

                auc_val = roc_auc_score(y_te, y_te_prob)
                brier_val = brier_score_loss(y_te, y_te_prob)

                # 折级 Youden cutoff（仅作折级记录）
                fpr, tpr, thresholds = roc_curve(y_te, y_te_prob)
                best_idx = int(np.argmax(tpr - fpr))
                fold_cutoff = thresholds[best_idx] if 0 <= best_idx < len(thresholds) else 0.5
                if not np.isfinite(fold_cutoff):
                    fold_cutoff = 0.5

                # 四个折级记录必须成对齐追加：仅在本折全部成功后才写入 inner_ab，
                # 若中途失败则统一在 except 里追加 NaN，保证长度与折数一致
                results[model_name]["fold_aucs"].append(auc_val)
                results[model_name]["fold_briers"].append(brier_val)
                results[model_name]["fold_cutoffs"].append(fold_cutoff)
                results[model_name]["inner_ab_scores"].append(inner_ab)
                for i in range(len(y_te)):
                    repeat_records[model_name].append({
                        "model": model_name,
                        "repeat": rep_idx,
                        "fold": fold_idx,
                        "y_true": int(y_te[i]),
                        "y_pred_prob": float(y_te_prob[i]),
                    })
                    fold_pred_records.append({
                        "ID": ids_te[i],
                        "model": model_name,
                        "repeat": rep_idx,
                        "fold": fold_idx,
                        "y_true": int(y_te[i]),
                        "y_pred_prob": float(y_te_prob[i]),
                    })
                print(f"  [{model_name}] R{rep_idx}F{fold_idx} outer AUC={auc_val:.4f}, "
                      f"Brier={brier_val:.4f} ({time.time() - t0:.0f}s)")
            except Exception as e:
                # 模型级失败容忍（与旧版一致）：记录 NaN 并告警，OOF 完整性校验会提示
                print(f"\nFAILED [{model_name}] repeat {rep_idx} fold {fold_idx}: {e}")
                traceback.print_exc()
                results[model_name]["fold_aucs"].append(np.nan)
                results[model_name]["fold_briers"].append(np.nan)
                results[model_name]["fold_cutoffs"].append(np.nan)
                results[model_name]["inner_ab_scores"].append(np.nan)

        # 折级审计文件（含 ID，便于逐例核对）
        if fold_pred_records:
            pd.DataFrame(fold_pred_records).to_csv(
                os.path.join(fold_dir, f"{tag}_oof_predictions.csv"),
                index=False, float_format="%.6f")

        del train_final, test_final, X_tr_full, X_te, meta
        gc.collect()

    # --- 3e. 收尾本重复：拼接 5 个外层测试折 OOF → 计算该重复的 OOF 指标 ---
    for model_name in MODEL_LIST:
        rep_df = pd.DataFrame(repeat_records[model_name])
        results[model_name]["oof_by_repeat"].append(rep_df)
        if len(rep_df) == 0:
            results[model_name]["repeat_metrics"].append(
                {k: np.nan for k in METRIC_KEYS})
            continue
        if len(rep_df) != len(y_raw):
            print(f"*** WARNING: [{model_name}] repeat {rep_idx} OOF incomplete: "
                  f"{len(rep_df)}/{len(y_raw)} rows (some outer folds failed) ***")
        results[model_name]["repeat_metrics"].append(
            compute_repeat_metrics(rep_df["y_true"].values,
                                   rep_df["y_pred_prob"].values))
    n_ok = sum(1 for m in MODEL_LIST if len(results[m]["oof_by_repeat"][-1]) > 0)
    print(f"  Repeat {rep_idx} finished: {n_ok}/{len(MODEL_LIST)} models produced OOF")

# ---------------------------------------------------------------------------
# Step 4: 汇总 OOF 预测
#   4a. 全量：10 重复 × 9 模型 × n 例（重复级配对比较用）
#   4b. 代表性：repeat 1 的 OOF（列结构与旧版一致，下游 05/06/07 直接可读）
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("=== Step 4: Saving OOF Predictions ===")
print("=" * 70)

# --- 4a. 全量 OOF（含 repeat 列） ---
all_repeat_dfs = []
for model_name in MODEL_LIST:
    for rep_df in results[model_name]["oof_by_repeat"]:
        if len(rep_df) > 0:
            all_repeat_dfs.append(rep_df)
if all_repeat_dfs:
    oof_all_repeats = pd.concat(all_repeat_dfs, ignore_index=True)
else:
    oof_all_repeats = pd.DataFrame(
        columns=["model", "repeat", "fold", "y_true", "y_pred_prob"])
oof_repeats_csv = os.path.join(RESULT_DIR, "oof_predictions_all_repeats.csv")
oof_all_repeats.to_csv(oof_repeats_csv, index=False, float_format="%.6f")
print(f"  Saved: {oof_repeats_csv} ({len(oof_all_repeats)} rows)")

# --- 4b. 代表性 OOF = repeat 1（列结构不变，下游兼容） ---
rep1_dfs = []
for model_name in MODEL_LIST:
    rep1 = results[model_name]["oof_by_repeat"][0]
    if len(rep1) == 0:
        continue
    if len(rep1) != len(y_raw):
        print(f"*** WARNING: [{model_name}] repeat 1 OOF incomplete: "
              f"{len(rep1)}/{len(y_raw)} rows (some outer folds failed) ***")
    rep1_dfs.append(rep1[["model", "fold", "y_true", "y_pred_prob"]])
if rep1_dfs:
    oof_all = pd.concat(rep1_dfs, ignore_index=True)
else:
    oof_all = pd.DataFrame(columns=["model", "fold", "y_true", "y_pred_prob"])
oof_csv = os.path.join(RESULT_DIR, "oof_predictions_all_models.csv")
oof_all.to_csv(oof_csv, index=False, float_format="%.6f")
print(f"  Saved: {oof_csv} ({oof_all.shape[0]} rows) — NOTE: repeat 1 representative OOF")
# 防泄漏版命名副本（内容与上相同）
oof_csv_v2 = os.path.join(RESULT_DIR, "nested_cv_leakage_free_oof.csv")
oof_all.to_csv(oof_csv_v2, index=False, float_format="%.6f")
print(f"  Saved: {oof_csv_v2}")

# ---------------------------------------------------------------------------
# Step 4b: 模型选择偏倚加固 —— 折内自选混合 OOF（无偏）+ 选型稳定性表 + 注记
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("=== Step 4b: Model-Selection-Bias Hardening (inner-fold winner OOF) ===")
print("=" * 70)

# 逐 (repeat, fold) 胜出模型：
#   内层胜出 = 该折折内 train 上内层 CV AB_Score 最高的模型（未接触外层测试折，
#              据此拼出的混合 OOF 是对"自动选型流程"的无偏泛化估计）
#   外层胜出 = 该折外层测试 AB_Score 最高者（仅作稳定性展示，不用于性能报告）
stability_rows = []
innersel_repeat_metrics = []   # 每个重复一套混合 OOF 指标
innersel_oof_frames = []       # 每个重复一套混合 OOF（含 repeat 列）
for rep_idx in range(1, N_REPEATS + 1):
    rep_recs = []
    rep_agree = 0
    for fold_i in range(1, OUTER_FOLDS + 1):
        g = (rep_idx - 1) * OUTER_FOLDS + (fold_i - 1)   # 折级列表的全局下标
        inner_cand, outer_cand = [], []
        for m in MODEL_LIST:
            r = results[m]
            rep_df = r["oof_by_repeat"][rep_idx - 1]
            has_oof = (len(rep_df) > 0) and bool((rep_df["fold"] == fold_i).any())
            if (has_oof and len(r["inner_ab_scores"]) > g
                    and np.isfinite(r["inner_ab_scores"][g])):
                inner_cand.append((m, r["inner_ab_scores"][g]))
            if (len(r["fold_aucs"]) > g and np.isfinite(r["fold_aucs"][g])
                    and len(r["fold_briers"]) > g and np.isfinite(r["fold_briers"][g])):
                outer_cand.append((m, 0.5 * r["fold_aucs"][g]
                                   + 0.5 * (1 - r["fold_briers"][g])))
        if not inner_cand:
            raise RuntimeError(
                f"repeat {rep_idx} fold {fold_i}: 无任何模型的内层 AB_Score，无法做折内自选")
        inner_winner, inner_best_ab = max(inner_cand, key=lambda t: t[1])
        outer_winner, outer_best_ab = (max(outer_cand, key=lambda t: t[1])
                                       if outer_cand else ("NA", np.nan))
        sel = results[inner_winner]["oof_by_repeat"][rep_idx - 1]
        sel = sel[sel["fold"] == fold_i]
        if len(sel) == 0:
            raise RuntimeError(
                f"repeat {rep_idx} fold {fold_i}: 内层胜出模型 {inner_winner} 无 OOF 记录")
        rep_recs.append(sel)
        agree = int(inner_winner == outer_winner)
        rep_agree += agree
        stability_rows.append({
            "repeat": rep_idx,
            "fold": fold_i,
            "inner_winner": inner_winner,
            "inner_winner_AB": inner_best_ab,
            "outer_winner": outer_winner,
            "outer_winner_AB": outer_best_ab,
            "agree": agree,
        })
    # 该重复的混合 OOF（5 个折的胜出模型预测拼接）
    rep_mixed = pd.concat(rep_recs, ignore_index=True)
    innersel_oof_frames.append(rep_mixed)
    if len(rep_mixed) == len(y_raw):
        innersel_repeat_metrics.append(
            compute_repeat_metrics(rep_mixed["y_true"].values,
                                   rep_mixed["y_pred_prob"].values))
    else:
        print(f"  WARNING: repeat {rep_idx} 混合 OOF 行数 {len(rep_mixed)} "
              f"≠ 总样本数 {len(y_raw)}，跳过该重复指标")
        innersel_repeat_metrics.append({k: np.nan for k in METRIC_KEYS})
    print(f"  repeat {rep_idx}: inner/outer winner agreement {rep_agree}/{OUTER_FOLDS}")

stability_df = pd.DataFrame(stability_rows)
stability_path = os.path.join(RESULT_DIR, "model_selection_stability.csv")
stability_df.to_csv(stability_path, index=False, float_format="%.6f")
print(f"  Saved: {stability_path} ({len(stability_df)} rows = {N_REPEATS}×{OUTER_FOLDS})")

# 一致性汇总（逐重复 + Overall）
stab_sum_rows = []
for rep_idx in range(1, N_REPEATS + 1):
    sub = stability_df[stability_df["repeat"] == rep_idx]
    stab_sum_rows.append({
        "Repeat": rep_idx,
        "n_pairs": len(sub),
        "n_agree": int(sub["agree"].sum()),
        "agree_pct": float(sub["agree"].mean() * 100.0) if len(sub) else np.nan,
    })
n_agree_total = int(stability_df["agree"].sum())
n_pairs_total = len(stability_df)
stab_sum_rows.append({
    "Repeat": "Overall",
    "n_pairs": n_pairs_total,
    "n_agree": n_agree_total,
    "agree_pct": float(n_agree_total / n_pairs_total * 100.0) if n_pairs_total else np.nan,
})
stab_sum_path = os.path.join(RESULT_DIR, "model_selection_stability_summary.csv")
pd.DataFrame(stab_sum_rows).to_csv(stab_sum_path, index=False, float_format="%.2f")
print(f"  Saved: {stab_sum_path}")
print(f"  Inner/outer winner agreement: {n_agree_total}/{n_pairs_total} "
      f"({n_agree_total / n_pairs_total * 100:.1f}%)")

# 混合 OOF（每个重复一套；沿用旧文件名，含 repeat 列）
innersel_df = pd.concat(innersel_oof_frames, ignore_index=True)
innersel_oof_path = os.path.join(RESULT_DIR, "nested_cv_innersel_oof.csv")
innersel_df.to_csv(innersel_oof_path, index=False, float_format="%.6f")
print(f"  Saved: {innersel_oof_path} ({len(innersel_df)} rows)")

# 混合 OOF 指标（重复级聚合：无偏口径，估计的是"折内自选流程"的泛化性能）
if len(innersel_repeat_metrics) > 0:
    is_row = build_metrics_row("FoldwiseInnerSel", innersel_repeat_metrics)
    is_row["N_Repeats"] = len(innersel_repeat_metrics)
    is_row["Winner_Agreement"] = f"{n_agree_total}/{n_pairs_total}"
    is_metrics_path = os.path.join(RESULT_DIR, "nested_cv_innersel_metrics.csv")
    pd.DataFrame([is_row]).to_csv(is_metrics_path, index=False, float_format="%.6f")
    print(f"  Saved: {is_metrics_path}")
    print(f"  >>> InnerSel (unbiased) AUC={is_row['AUC']:.4f} "
          f"({is_row['AUC_CI_Lower']:.4f}-{is_row['AUC_CI_Upper']:.4f}), "
          f"Brier={is_row['Brier']:.4f}  — 与 Step 6 全局选型对照")

# 偏倚口径注记文件
bias_note = f"""模型选择偏倚口径说明（04_Model_Selection.py v3 自动生成）
================================================================

三层偏倚的性质与处理：
1) 特征选择泄漏（已修复）：旧版在全部 649 例上完成 log1p/插补/缩放/VIF/4 方法
   共识后再跑嵌套 CV，外层测试折结局信息经特征选择进入建模。新版逐折重放
   全链路（仅折内 train 拟合），已消除。
2) 调参泄漏（已解决）：超参仅由内层 5 折 CV 估计，外层测试折只做 predict。
3) 模型选择偏倚（winner's curse，轻度，未消除也无法完全消除）：
   本目录中 9 个模型在重复级 OOF 指标上比较并全局选型，选中模型的指标存在
   系统性轻度乐观偏倚（量级通常在千分位至 0.02 AUC；9 个模型高度相关，
   偏倚远小于原特征选择泄漏）。该指标用于模型间相对比较（ranking），
   不宜直接当作绝对泛化性能报告。

重复嵌套 CV 口径（N_REPEATS = {N_REPEATS}）：
- 每次重复使用不同的外层数据划分（random_state = RANDOM_STATE + rep，
  即 {RANDOM_STATE}..{RANDOM_STATE + N_REPEATS - 1}），共
  {N_REPEATS}×{OUTER_FOLDS} = {N_REPEATS * OUTER_FOLDS} 个外层测试折。
- 报告值 = N_REPEATS 个"重复级 OOF 指标"的均值。
- 不确定性 = 重复级经验标准误 SE = SD(重复级指标) / sqrt(N_REPEATS)。
  该 SE 是重复级经验标准误（衡量对外层数据划分的敏感性），
  不构成严格意义上的抽样置信区间；CI 列（均值±1.96SE）仅供一致性参考。
- 对照文件 nested_cv_oof_bca_ci.csv：对 repeat 1 的代表性 OOF 另算 BCa
  bootstrap 区间（{BOOTSTRAP_ITER} 次重抽 + jackknife）。两种区间含义不同：
  BCa 反映样本抽样不确定性，重复级 SE 反映外层划分敏感性；两者列并置仅为
  交叉核对，不写入 model_selection_metrics.csv，也不替代重复级 SE 口径。
- 重复嵌套 CV 用于降低单次数据划分带来的方差，改善估计稳定性，
  但不得用于消除针对被选中模型的残余 Winner's Curse；
  模型选择偏倚的无偏口径仍以 nested_cv_innersel_metrics.csv 的重复均值为准。

无偏敏感性分析（本目录提供）：
- model_selection_stability.csv：逐 repeat 逐 fold 记录内层胜出
  （折内 train 内层 CV AB_Score 最高，未接触外层测试折）与外层胜出模型及一致性。
- model_selection_stability_summary.csv：各重复一致比例 + 汇总行。
- nested_cv_innersel_oof.csv / nested_cv_innersel_metrics.csv：每重复取内层
  胜出模型预测拼接的混合 OOF，重复级聚合——它估计的是"整个自动选型流程"的
  泛化性能，不含模型选择偏倚。若其 AUC 与全局选型接近，说明 winner's curse 可忽略。

OOF 文件说明：
- oof_predictions_all_models.csv / nested_cv_leakage_free_oof.csv
  = repeat 1 的代表性 OOF（每例一行，列结构不变，供下游 05/06/07 直接读取）。
- oof_predictions_all_repeats.csv = 全部 {N_REPEATS} 次重复的完整 OOF
  （列: model, repeat, fold, y_true, y_pred_prob）。

泛化性能的主证据：Temporal / Geographical 外部验证（对已选定模型完全独立）。
参考文献：Cawley & Talbot (2010) JMLR 11:2079-2107；Varma & Simon (2006)
BMC Bioinformatics 7:91；TRIPOD (Collins et al. 2015, BMJ 350:g7595)。

生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
重复次数: {N_REPEATS}, 外层折数: {OUTER_FOLDS}, 内层: {INNER_FOLDS}x{INNER_TRIALS},
外层种子: {RANDOM_STATE}..{RANDOM_STATE + N_REPEATS - 1}, 内层种子: {RANDOM_STATE}

该 95% CI 为重复嵌套 CV 的重复级经验标准误（均值 ± 1.96×SE），反映外层数据划分的敏感性，并非严格意义上的抽样置信区间；外部队列采用 BCa bootstrap（反映样本抽样不确定性），两者含义不同，不可横向比较区间宽度。
"""
with open(os.path.join(RESULT_DIR, "model_selection_bias_note.txt"), "w",
          encoding="utf-8") as f:
    f.write(bias_note)
print("  Saved: model_selection_bias_note.txt")

# ---------------------------------------------------------------------------
# Step 5: 逐 (repeat, fold) 共识特征落盘（foldwise + 频率汇总）
#   频率分母 = N_REPEATS * OUTER_FOLDS = 50（不再是单次 5 折）
# ---------------------------------------------------------------------------
fold_features_df = pd.DataFrame(fold_feature_rows)
fold_features_path = os.path.join(RESULT_DIR, "foldwise_selected_features.csv")
fold_features_df.to_csv(fold_features_path, index=False)
print(f"  Saved: {fold_features_path} ({len(fold_features_df)} rows)")

n_total_folds = N_REPEATS * OUTER_FOLDS
freq_df = (fold_features_df.groupby("feature")
           .agg(n_folds_out_of_total=("fold", "count"),
                n_repeats_selected=("repeat", "nunique"))
           .reset_index()
           .rename(columns={"n_folds_out_of_total": f"n_folds_out_of_{n_total_folds}"})
           .sort_values([f"n_folds_out_of_{n_total_folds}", "feature"],
                        ascending=[False, True]))
freq_df["pct_folds"] = freq_df[f"n_folds_out_of_{n_total_folds}"] / n_total_folds * 100.0
freq_df["pct_repeats"] = freq_df["n_repeats_selected"] / N_REPEATS * 100.0
freq_path = os.path.join(RESULT_DIR, "fold_feature_frequency.csv")
freq_df.to_csv(freq_path, index=False, float_format="%.2f")
print(f"  Saved: {freq_path}")
print(freq_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Step 6: 重复级指标聚合与模型选择（列名向下兼容 05/06/07）
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("=== Step 6: OOF-Based Model Selection Metrics (Repeat-Level Aggregation) ===")
print("=" * 70)

oof_data = {}
metrics_rows = []
per_repeat_rows = []
repeat_auc_rows = []
bca_rows = []   # OOF 级 BCa CI 对照（独立文件）
for model_name in MODEL_LIST:
    rep_metrics = results[model_name]["repeat_metrics"]
    if len(rep_metrics) == 0 or all(np.isnan(m.get("AUC", np.nan)) for m in rep_metrics):
        print(f"  {model_name}: SKIPPED (no valid repeat-level OOF metrics)")
        continue

    row = build_metrics_row(
        model_name, rep_metrics,
        fold_aucs=results[model_name]["fold_aucs"],
        fold_briers=results[model_name]["fold_briers"],
    )
    metrics_rows.append(row)

    # 逐重复长表（审稿人核对用）+ 重复级 AUC 汇总
    for i, m in enumerate(rep_metrics, 1):
        per_repeat_rows.append({"Model": model_name, "Repeat": i,
                                **{k: m.get(k, np.nan) for k in METRIC_KEYS}})
        repeat_auc_rows.append({"Model": model_name, "Repeat": i,
                                "AUC": m.get("AUC", np.nan),
                                "Brier": m.get("Brier", np.nan),
                                "Optimal_Cutoff": m.get("Optimal_Cutoff", np.nan)})

    # ROC 图与下游兼容：用 repeat 1 的代表性 OOF，图例给重复级均值 ±1.96SE
    rep1 = results[model_name]["oof_by_repeat"][0]
    oof_data[model_name] = {"y_true": rep1["y_true"].values,
                            "y_prob": rep1["y_pred_prob"].values,
                            "metrics": row}

    # OOF 级 BCa CI 对照（repeat 1 代表性 OOF；仅写入独立文件，不进主表）
    t_bca = time.time()
    bca_mean, bca_lo, bca_hi = bootstrap_auc_ci_bca(
        rep1["y_true"].values, rep1["y_pred_prob"].values)
    bca_rows.append({
        "Model": model_name,
        "OOF_Source": "repeat 1 (representative)",
        "AUC_OOF_BCa_Mean": bca_mean,
        "AUC_BCa_Lower": bca_lo,
        "AUC_BCa_Upper": bca_hi,
        "AUC_RepeatMean": row["AUC"],
        "AUC_RepeatSE": row["AUC_SE"],
        "AUC_RepeatMean_CI_Lower": row["AUC_CI_Lower"],
        "AUC_RepeatMean_CI_Upper": row["AUC_CI_Upper"],
    })
    print(f"    [BCa] OOF AUC={bca_mean:.4f} ({bca_lo:.4f}-{bca_hi:.4f}) "
          f"vs repeat-mean {row['AUC']:.4f} ±1.96SE "
          f"({row['AUC_CI_Lower']:.4f}-{row['AUC_CI_Upper']:.4f}) "
          f"({time.time() - t_bca:.0f}s)")

    print(f"  {model_name}: AUC={row['AUC']:.4f} "
          f"({row['AUC_CI_Lower']:.4f}-{row['AUC_CI_Upper']:.4f}, "
          f"SE={row['AUC_SE']:.4f}, n={int(row['AUC_N_ValidRepeats'])}), "
          f"Cutoff={row['Optimal_Cutoff']:.4f}, "
          f"F1={row['F1']:.4f}")

metrics_df = pd.DataFrame(metrics_rows)
col_order = ["Model", "AUC", "AUC_CI_Lower", "AUC_CI_Upper",
             "AUC_SE", "AUC_SD", "AUC_N_ValidRepeats", "AUC_PerRepeat",
             "AUPRC", "AUPRC_CI_Lower", "AUPRC_CI_Upper", "AUPRC_SE", "AUPRC_SD",
             "Brier", "Brier_SE", "Brier_SD",
             "CV_AUC_Mean", "CV_AUC_SD", "CV_AUC_PerRepeat", "CV_AUC_PerFold",
             "CV_Brier_Mean", "CV_Brier_SD",
             "Optimal_Cutoff", "Optimal_Cutoff_SD",
             "Sensitivity", "Sensitivity_SD", "Specificity", "Specificity_SD",
             "Accuracy", "Accuracy_SD", "F1", "F1_SD", "PPV", "PPV_SD",
             "NPV", "NPV_SD", "AB_Score"]
col_order = [c for c in col_order if c in metrics_df.columns]
metrics_df = metrics_df[col_order]

# 口径说明：AUC/Brier 为 N_REPEATS 个重复级指标的均值；CI = 均值 ± 1.96*重复级 SE；
# 阈值化指标为各重复（各自 Youden cutoff）的均值，属 apparent threshold performance。
metrics_csv = os.path.join(RESULT_DIR, "model_selection_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False, float_format="%.6f")
print(f"\n  Metrics (repeat-level mean ± 1.96SE) saved to: {metrics_csv}")

# 与 05 外部队列 BCa bootstrap CI 的口径区分（列名相同、含义不同，不可横向比较区间宽度）
ci_note_path = os.path.join(RESULT_DIR, "metrics_ci_source.txt")
with open(ci_note_path, "w", encoding="utf-8") as _f:
    _f.write(
        "model_selection_metrics.csv 中 AUC_CI_Lower / AUC_CI_Upper / AUPRC_CI_Lower / "
        "AUPRC_CI_Upper 的含义：\n"
        f"  = N_REPEATS={N_REPEATS} 次重复嵌套 CV 的重复级经验标准误，区间为 均值 ± 1.96×SE，"
        "SE = SD(重复级 OOF 指标)/sqrt(N_REPEATS)。\n"
        "  它反映「外层数据划分」的敏感性，不是抽样置信区间。\n"
        "对比：05_Model_Development.py 中外部队列（Temporal / Geographical）的 95% CI 为 "
        "BCa bootstrap（反映样本抽样不确定性）。\n"
        "两类区间含义不同，不可直接比较区间宽度；"
        "nested_cv_oof_bca_ci.csv 另提供 repeat 1 OOF 的 BCa CI 作对照，仅用于口径对照。\n"
    )
print(f"  Saved: {ci_note_path}")

# 逐重复指标（90 行 = 10 × 9）
pd.DataFrame(per_repeat_rows).to_csv(
    os.path.join(RESULT_DIR, "model_selection_metrics_per_repeat.csv"),
    index=False, float_format="%.6f")
print(f"  Saved: model_selection_metrics_per_repeat.csv ({len(per_repeat_rows)} rows)")
pd.DataFrame(repeat_auc_rows).to_csv(
    os.path.join(RESULT_DIR, "nested_cv_repeat_auc_summary.csv"),
    index=False, float_format="%.6f")
print(f"  Saved: nested_cv_repeat_auc_summary.csv ({len(repeat_auc_rows)} rows)")

# OOF 级 BCa CI 对照（repeat 1 代表性 OOF；不进入主指标表）
if bca_rows:
    bca_path = os.path.join(RESULT_DIR, "nested_cv_oof_bca_ci.csv")
    pd.DataFrame(bca_rows).to_csv(bca_path, index=False, float_format="%.6f")
    print(f"  Saved: {bca_path} ({len(bca_rows)} rows) — 仅供对照，请勿与重复级 SE 混用")

best_auc_row = metrics_df.loc[metrics_df["AUC"].idxmax()]
best_ab_row = metrics_df.loc[metrics_df["AB_Score"].idxmax()]
best_model_name = best_ab_row["Model"]
print(f"\n  >>> BEST by AUC: {best_auc_row['Model']} "
      f"(AUC={best_auc_row['AUC']:.4f}, Brier={best_auc_row['Brier']:.4f})")
print(f"  >>> BEST by AB_Score: {best_model_name} "
      f"(AUC={best_ab_row['AUC']:.4f}, Brier={best_ab_row['Brier']:.4f}, "
      f"AB_Score={best_ab_row['AB_Score']:.4f})")

# ---------------------------------------------------------------------------
# Step 7: ROC 曲线（防泄漏 OOF）
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("=== Step 7: Plotting ROC Curves ===")
print("=" * 70)
save_roc_curves(oof_data, RESULT_DIR)

# ---------------------------------------------------------------------------
# Step 8: 官方面板全量调参 → best_params_all_models.pkl（供 05/07 最终模型）
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("=== Step 8: Best Params on Official Panel (for final model / 05-07) ===")
print("=" * 70)

if not os.path.exists(OFFICIAL_PANEL_PATH):
    raise FileNotFoundError(
        f"官方面板缺失: {OFFICIAL_PANEL_PATH}\n"
        "请先完整运行 01→02→03 生成冻结面板（best_params 调参依赖它）。")
official_df = pd.read_csv(OFFICIAL_PANEL_PATH)
# 官方面板调参特征：显式剔除随机负对照 noise_feature（若 02 输出保留该列）。
# 否则 best_params 是在「含 noise」的列空间上搜出来的（对 max_features=0.5/0.8
# 与 colsample_bytree 等相对参数的实际列数有细微影响），而 05/07 读取的
# feature_selection/Derivation cohort.csv 不含 noise，二者会有偏差。
official_features = [c for c in official_df.columns
                     if c not in ("ID", "label", NOISE_FEATURE)]
X_official = official_df[official_features].values.astype(np.float64)
y_official = official_df["label"].values.astype(np.int32)
print(f"  Official panel: {OFFICIAL_PANEL_PATH} shape={official_df.shape}")

best_params_dict = {}
for model_name in MODEL_LIST:
    if model_name not in oof_data:
        print(f"  {model_name}: SKIPPED (no valid OOF predictions)")
        continue
    final_params = tune_model_on_full_training(
        model_name, X_official, y_official, official_features
    )
    best_params_dict[model_name] = final_params
    print(f"  {model_name}: {best_params_dict[model_name]}")

pkl_path = os.path.join(RESULT_DIR, "best_params_all_models.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(best_params_dict, f)
print(f"\n  Best params saved to: {pkl_path}")

# ---------------------------------------------------------------------------
# Step 9: Plot 1 — AB_Score Lollipop
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("=== Plotting AB_Score Lollipop Chart ===")
print("=" * 70)

df_lollipop = metrics_df.sort_values("AB_Score", ascending=True).reset_index(drop=True)
n_models = len(df_lollipop)

fig, ax = plt.subplots(figsize=(9, max(5, n_models * 0.45)))
y_pos = np.arange(n_models)
ab_scores = df_lollipop["AB_Score"].values

norm_c = plt.Normalize(ab_scores.min(), ab_scores.max())
blue_cmap = plt.cm.Blues

# X 轴范围自适应（避免 AB_Score 落在硬编码区间外导致点被裁掉）
x_lo = max(0.0, float(np.floor((ab_scores.min() - 0.03) * 50) / 50.0))
x_hi = min(1.0, float(np.ceil((ab_scores.max() + 0.06) * 50) / 50.0))

for i in range(n_models):
    score = ab_scores[i]
    ax.plot([x_lo, score], [y_pos[i], y_pos[i]], "-", color="lightgray",
            lw=0.8, alpha=0.8, zorder=1)
    ax.scatter(score, y_pos[i], s=80, c=[blue_cmap(norm_c(score))],
               zorder=3, edgecolors="white", linewidth=0.5)

champion_idx = n_models - 1
champion_score = ab_scores[champion_idx]
ax.scatter(champion_score, y_pos[champion_idx], s=120,
           c="#E63946", zorder=5, edgecolors="darkred", linewidth=1.0)

label_offset = ab_scores.max() * 0.02
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
ax.set_title("Model Selection by AB_Score (Leakage-Free OOF)",
             fontsize=13, fontweight="bold")
ax.set_xlim(x_lo, x_hi)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")
plt.tight_layout()
save_figure(fig, "Model_Selection_Lollipop", RESULT_DIR)
print("  Lollipop chart saved.")

# ---------------------------------------------------------------------------
# Step 10: Plot 2 — Performance Gap Waterfall
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("=== Plotting Performance Gap Waterfall ===")
print("=" * 70)

df_gap = metrics_df.sort_values("AB_Score", ascending=True).reset_index(drop=True)
n_gap = len(df_gap)
auc_gap = df_gap["AUC"].values
cal_gap = 1 - df_gap["Brier"].values
gap_vals = auc_gap - cal_gap

fig, ax = plt.subplots(figsize=(10, max(5.5, n_gap * 0.45)))
y_pos_gap = np.arange(n_gap)

x_min_gap = min(cal_gap.min(), auc_gap.min()) - 0.015
x_max_gap = max(cal_gap.max(), auc_gap.max()) + 0.08
ax.set_xlim(x_min_gap, x_max_gap)

for i in range(n_gap):
    auc_val = auc_gap[i]
    cal_val = cal_gap[i]
    gap = gap_vals[i]
    champion = (i == n_gap - 1)

    bar_color = "#F4A460" if champion else "#C0C0C0"
    bar_lw = 10 if champion else 5
    ax.plot([cal_val, auc_val], [y_pos_gap[i], y_pos_gap[i]],
            "-", color=bar_color, lw=bar_lw, alpha=0.85 if champion else 0.6,
            solid_capstyle="butt", zorder=5 if champion else 2)

    ax.scatter(auc_val, y_pos_gap[i], s=90, c="#1F77B4", zorder=6,
               edgecolors="white", linewidth=0.5)
    ax.scatter(cal_val, y_pos_gap[i], s=90, c="#FF7F0E", marker="s",
               zorder=6, edgecolors="white", linewidth=0.5)

    if champion:
        ax.scatter(auc_val, y_pos_gap[i], s=160, c="#1F77B4",
                   edgecolors="#FFD700", linewidth=3, zorder=7)
        ax.scatter(cal_val, y_pos_gap[i], s=160, c="#FF7F0E", marker="s",
                   edgecolors="#FFD700", linewidth=3, zorder=7)

    mid_x = (auc_val + cal_val) / 2
    if champion:
        ax.text(x_max_gap - 0.005, y_pos_gap[i],
                f"★ {best_model_name}  |  Gap={gap:+.3f}  |  AB={df_gap['AB_Score'].iloc[i]:.3f}",
                fontsize=9, fontweight="bold", color="#E63946",
                ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#E63946", alpha=0.95))
    else:
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

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("=== Leakage-Free Model Selection Complete ===")
print(f"  Recommended model: {best_model_name}")
print(f"  OOF file: {oof_csv_v2}")
print(f"  Fold-wise features: {fold_features_path}")
print(f"  Stability & unbiased InnerSel OOF: model_selection_stability.csv /")
print(f"      nested_cv_innersel_metrics.csv (see model_selection_bias_note.txt)")
print(f"  All outputs saved to: {RESULT_DIR}")
print(f"  Fold intermediates: {WORK_DIR}")
print("=" * 70)
