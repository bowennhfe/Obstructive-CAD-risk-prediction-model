# =============================================================================
# 06_Model_Growth_Analysis.py — Model Growth Analysis & Subgroup Vulnerability
# =============================================================================
# 功能：
#   第一部分：成长曲线数据（带Bootstrap置信区间带）—— 模拟数据积累对模型性能的影响
#            （制图由 07_Clinical_Plots_R.R 的 Growth_Curve_Split 完成）
#   第二部分：亚组性能分析 —— 精确反标准化年龄 + SCORE2三分组 + Bootstrap AUC CI
#            （森林图由 07_Clinical_Plots_R.R 的 Subgroup_Forest_Plot 完成）
#   第三部分：校准前后亚组对比 —— 截距校准后 Brier/Slope 变化
#   第四部分：Brier 改善连接点图 —— 可视化各亚组校准后的 Brier 改善
#
# 输入：
#   - feature_selection/Derivation / Temporal / Geographical cohort.csv
#   - Model Selection/best_params_all_models.pkl
#   - Model Selection/model_selection_metrics.csv（推荐模型）
#   - data_preprocessing/preprocessing_objects.joblib（年龄反标准化）
# 输出（均在 E:\new\Model Growth Analysis）：
#   - growth_curve_data.csv
#   - Subgroup_Performance_{model_name}.csv
#   - Subgroup_Calibration_Comparison.csv
#   - subgroup_brier_improvement.png/pdf/tif
# =============================================================================

import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib

# Optional tqdm import (fallback to plain range if not installed)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from scipy.stats import norm
import statsmodels.api as sm  # 统一在模块顶部导入（原在函数内重复导入）
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

# =============================================================================
# Global Configuration
# =============================================================================
RANDOM_STATE = 123
N_JOBS = max(1, os.cpu_count() or 1)
BOOTSTRAP_ITER = 2000
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "feature_selection")
SELECTION_DIR = os.path.join(BASE_DIR, "Model Selection")
PREPROC_DIR = os.path.join(BASE_DIR, "data_preprocessing")
RESULT_DIR = os.path.join(BASE_DIR, "Model Growth Analysis")
MODEL_DEV_DIR = os.path.join(BASE_DIR, "Model Development")  # 05 output directory
os.makedirs(RESULT_DIR, exist_ok=True)

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
    """Save figure in PNG, PDF, TIF at 600 DPI."""
    for ext in ["png", "pdf", "tif"]:
        path = os.path.join(RESULT_DIR, f"{base_name}.{ext}")
        fig.savefig(path, dpi=IMG_DPI, bbox_inches="tight", transparent=False)
    plt.close(fig)


# =============================================================================
# Bootstrap AUC CI (Percentile Method)
# =============================================================================
def bootstrap_auc_ci(y_true, y_prob, n_iter=BOOTSTRAP_ITER, alpha=0.05, random_state=RANDOM_STATE):
    """Compute AUC and its 95% CI using bootstrap percentile method."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    try:
        theta_hat = roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan, np.nan, np.nan
    if np.isnan(theta_hat):
        return np.nan, np.nan, np.nan

    rng = np.random.RandomState(random_state)
    boot_scores = np.full(n_iter, np.nan)
    for b in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        try:
            boot_scores[b] = roc_auc_score(y_true[idx], y_prob[idx])
        except ValueError:
            # 重采样失败（如该次 resample 只剩单一类别）时记 NaN，
            # 不以点估计 theta_hat 填充，避免人为压缩 CI 宽度
            # （与亚组 bootstrap p 值循环中失败记 NaN 的处理口径一致）
            boot_scores[b] = np.nan

    # 仅用非 NaN 的有效迭代计算分位数
    boot_scores = boot_scores[~np.isnan(boot_scores)]
    if len(boot_scores) < 50:
        print(f"  WARNING: bootstrap 有效迭代仅 {len(boot_scores)} 次 (<50)，"
              f"CI 无法可靠估计，返回 NaN")
        return theta_hat, np.nan, np.nan

    lower = np.percentile(boot_scores, 100 * (alpha / 2))
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return theta_hat, lower, upper


def bootstrap_ci_from_values(values, n_iter=BOOTSTRAP_ITER, alpha=0.05, random_state=RANDOM_STATE):
    """Compute BCa 95% CI from a vector of repeated experiment values.

    Algorithm identical to 05_Model_Development.py bootstrap_auc_ci.
    Suitable for growth curve AUC values from n_repeats independent runs.

    Parameters:
        values : array-like of AUC values from repeated experiments
        n_iter : Bootstrap resampling iterations (default 2000)
        alpha : Significance level (default 0.05 for 95% CI)
        random_state : Random seed

    Returns:
        (theta_hat, lower, upper) : point estimate (mean), CI lower, CI upper
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan, np.nan

    theta_hat = np.mean(values)

    # ===== Constant vector detection: CI cannot be estimated → NaN (not zero-width) =====
    # 零宽度 CI 会误导读者认为估计精确；无变异时 BCa 无法定义分位数区间，
    # 返回 NaN 以明确"无法估计"，下游按缺失处理。
    if np.std(values) == 0:
        return theta_hat, np.nan, np.nan

    # --- Jackknife for acceleration a ---
    jack_estimates = np.empty(n)
    for i in range(n):
        jack_estimates[i] = np.mean(np.delete(values, i))
    jack_mean = np.mean(jack_estimates)
    diff = jack_mean - jack_estimates
    sum_cubed = np.sum(diff ** 3)
    sum_sq = np.sum(diff ** 2)
    a = 0.0 if sum_sq == 0 else sum_cubed / (6.0 * (sum_sq ** 1.5))

    # --- Bootstrap resampling ---
    rng = np.random.RandomState(random_state)
    boot_estimates = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        boot_estimates[b] = np.mean(values[idx])

    # --- Bias correction ---
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

    return theta_hat, lower, upper


# =============================================================================
# Calibration Metrics Helper
# =============================================================================
def compute_calibration_metrics(y_true, y_prob):
    """Compute Brier Score, calibration intercept, and calibration slope.

    Calibration slope/intercept are obtained by regressing the logit of
    predicted probabilities against the true labels (logistic regression
    with no penalty). The slope is the coefficient, intercept is the intercept.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    brier = brier_score_loss(y_true, y_prob)

    y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
    logit = np.log(y_prob_clipped / (1 - y_prob_clipped))
    # GLM 拟合加异常保护：小样本亚组可能出现完全分离/不收敛，
    # 产生极端校准斜率；失败时截距与斜率记 NaN 并打印 warning。
    # Brier 不依赖 GLM，仍正常返回。（sm 已在模块顶部统一导入）
    try:
        intercept = sm.GLM(y_true, np.ones((len(y_true), 1)), family=sm.families.Binomial(), offset=logit).fit().params[0]
        slope = sm.GLM(y_true, sm.add_constant(logit), family=sm.families.Binomial()).fit().params[1]
    except Exception as e:
        print(f"  WARNING: 校准 GLM 拟合失败 ({e})，intercept/slope 记为 NaN")
        intercept, slope = np.nan, np.nan

    return brier, intercept, slope


# =============================================================================
# Model Building (from 04/05)
# =============================================================================
def _get_fixed_params(model_name):
    if model_name == "L2_LR":
        return {"solver": "lbfgs", "max_iter": 5000, "random_state": RANDOM_STATE}
    elif model_name == "SVM":
        return {"probability": True, "random_state": RANDOM_STATE}
    elif model_name == "RF":
        return {"random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise RuntimeError("XGBoost not installed")
        return {"n_estimators": 200, "verbosity": 0, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    elif model_name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return {"max_iter": 2000, "early_stopping": True, "validation_fraction": 0.2,
                "n_iter_no_change": 15, "random_state": RANDOM_STATE, "verbose": False}
    elif model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return {"n_jobs": N_JOBS}
    elif model_name == "DT":
        from sklearn.tree import DecisionTreeClassifier
        return {"random_state": RANDOM_STATE}
    elif model_name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise RuntimeError("LightGBM not installed")
        return {"n_estimators": 200, "verbose": -1, "random_state": RANDOM_STATE, "n_jobs": N_JOBS}
    elif model_name == "CatBoost":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise RuntimeError("CatBoost not installed")
        return {"iterations": 500, "early_stopping_rounds": 50, "verbose": 0,
                "random_seed": RANDOM_STATE, "thread_count": N_JOBS, "train_dir": RESULT_DIR}
    elif model_name == "GaussianNB":
        from sklearn.naive_bayes import GaussianNB
        return {}
    return {}


def build_model(model_name, params):
    fixed = _get_fixed_params(model_name)
    merged = {**fixed, **params}
    if model_name == "L2_LR":
        return LogisticRegression(penalty="l2", **merged)
    elif model_name == "SVM":
        return SVC(**merged)
    elif model_name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**merged)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**merged)
    elif model_name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(**merged)
    elif model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**merged)
    elif model_name == "DT":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**merged)
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**merged)
    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**merged)
    elif model_name == "GaussianNB":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(**merged)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Load Preprocessing Objects (for age de-standardization)
# =============================================================================
class ScaledImputer:
    """Minimal stub for unpickling preprocessing_objects.joblib."""
    def __init__(self, imputer=None, use_scaling=True, feature_names=None):
        self.imputer = imputer
        self.use_scaling = use_scaling
        self.feature_names = feature_names or []
        self.means_ = None
        self.stds_ = None


def load_age_scaler_params():
    """Load RobustScaler params for age from preprocessing_objects.joblib."""
    preproc_path = os.path.join(PREPROC_DIR, "preprocessing_objects.joblib")
    try:
        pp = joblib.load(preproc_path)
        age_median = pp['scaler_robust_params']['age']['median']
        age_iqr = pp['scaler_robust_params']['age']['iqr']
        is_age_logged = 'age' in pp.get('transformed_log_cols', [])
        print(f"  Age scaler: median={age_median}, iqr={age_iqr}, log_transformed={is_age_logged}")
        return age_median, age_iqr, is_age_logged
    except (KeyError, FileNotFoundError, AttributeError) as e:
        # 仅捕获"文件缺失/结构不符"类错误（通常意味着 01 预处理脚本未运行
        # 或输出结构已变更），其余异常直接抛出以便排查。
        # 回退参数将导致反标准化年龄失真，下游 Part 2 会对 age_raw 做
        # 合理性校验，越界即 sys.exit 终止，不再静默继续。
        print(f"  WARNING: Could not load preprocessing_objects.joblib ({e}). "
              f"请先运行 01 数据预处理脚本生成该文件；"
              f"本次回退使用 fallback median=60, iqr=15（稍后将强制校验）")
        return 60.0, 15.0, False


def destandardize_age(age_scaled, age_median, age_iqr, is_age_logged):
    """Reverse RobustScaler (± Log(1+X)) to recover raw age in years."""
    age_raw = np.asarray(age_scaled, dtype=np.float64) * age_iqr + age_median
    if is_age_logged:
        age_raw = np.expm1(age_raw)
    return age_raw


# =============================================================================
# Load Data
# =============================================================================
print("=" * 70)
print("06_Model_Growth_Analysis.py — Model Growth & Subgroup Vulnerability")
print("=" * 70)

# --- Load best params and recommended model ---
params_path = os.path.join(SELECTION_DIR, "best_params_all_models.pkl")
metrics_csv = os.path.join(SELECTION_DIR, "model_selection_metrics.csv")

if not os.path.exists(params_path):
    print(f"ERROR: {params_path} not found. Run 04_Model_Selection.py first.")
    sys.exit(1)

with open(params_path, "rb") as f:
    best_params_dict = pickle.load(f)

# Determine recommended model
if os.path.exists(metrics_csv):
    metrics_df = pd.read_csv(metrics_csv)
    # Use AB_Score (consistent with 04_Model_Selection.py)
    if "AB_Score" in metrics_df.columns:
        model_name = metrics_df.loc[metrics_df["AB_Score"].idxmax(), "Model"]
    else:
        model_name = metrics_df.loc[metrics_df["AUC"].idxmax(), "Model"]
    print(f"Recommended model: {model_name}")
else:
    model_name = list(best_params_dict.keys())[0]
    print(f"Fallback model: {model_name}")

if model_name not in best_params_dict:
    model_name = list(best_params_dict.keys())[0]
    print(f"Model not in params, using: {model_name}")

best_params = best_params_dict[model_name]
print(f"Best params: {best_params}")

# --- Load feature_selection data ---
train_df = pd.read_csv(os.path.join(DATA_DIR, "Derivation cohort.csv"))
temporal_df = pd.read_csv(os.path.join(DATA_DIR, "Temporal external validation cohort.csv"))
geo_df = pd.read_csv(os.path.join(DATA_DIR, "Geographical external validation cohort.csv"))

feature_cols = [c for c in train_df.columns if c not in ("ID", "label")]
print(f"Features ({len(feature_cols)}): {feature_cols}")

# Extract arrays
X_train = train_df[feature_cols].values.astype(np.float64)
y_train = train_df["label"].values.astype(np.int32)
X_temporal = temporal_df[feature_cols].values.astype(np.float64)
y_temporal = temporal_df["label"].values.astype(np.int32)
X_geo = geo_df[feature_cols].values.astype(np.float64)
y_geo = geo_df["label"].values.astype(np.int32)
geo_ids = geo_df["ID"].values

print(f"Derivation: {X_train.shape}, Temporal: {X_temporal.shape}, Geo: {X_geo.shape}")
print(f"Derivation Pos/Neg: {np.sum(y_train==1)}/{np.sum(y_train==0)}")
print(f"Temporal Pos/Neg: {np.sum(y_temporal==1)}/{np.sum(y_temporal==0)}")
print(f"Geo Pos/Neg: {np.sum(y_geo==1)}/{np.sum(y_geo==0)}")

# =============================================================================
# Part 1: Growth Curve — AUC & Brier with BCa CI (Fully Unified + 05 Reuse)
# =============================================================================
percentages = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
# Bootstrap 100 repeats is sufficient to stabilize BCa CI (Efron, 1993),
# saving ~50% runtime vs. 200 repeats.
n_repeats = 100
n_temporal = len(y_temporal)

print("\n" + "=" * 70)
print(f"Part 1: Growth Curve — AUC & Brier (Unified: {n_repeats} repeats, BCa CI)")
print("=" * 70)

# Initialize matrices: separate AUC and Brier
growth_auc_matrix = np.full((len(percentages), n_repeats), np.nan)
growth_brier_matrix = np.full((len(percentages), n_repeats), np.nan)

# =========================================================================
# 加载 05 的校准后概率（仅供 Part 2/3 亚组分析使用）
# 口径统一说明：成长曲线 0% 基线点不再由此处单独计算，统一以 Part 1
# 循环中 pct=0.0（仅用 Derivation 训练 + 统一校准器）的计算结果为唯一口径，
# 下游 07 绘图仅读取 growth_curve_data.csv 中的循环计算值，保证单一数据来源。
# 二者理论上应一致（同一模型、同一校准器、同一评估集），此处不再重复输出
# 0% 的 AUC/Brier，避免双重口径并存。
# 0% 处 CI 为 NaN 的说明：pct=0.0 时每次重复的训练数据完全相同，100 次重复
# AUC/Brier 为常数向量，BCa 无法定义分位数区间，CI 返回 NaN 属预期行为，
# 该问题由 07 端绘图时另行处理，本文件不管。
# =========================================================================

print("\n[加载05概率] Loading calibrated probabilities from 05 (for Part 2/3 subgroup analysis)...")
calib_file = os.path.join(MODEL_DEV_DIR, f"predicted_probabilities_{model_name}_geographical.csv")
if os.path.exists(calib_file):
    calib_df = pd.read_csv(calib_file)
    y_prob_raw_0 = calib_df["y_pred_prob_raw"].values
    y_prob_calibrated = calib_df["y_pred_prob_calibrated"].values
    y_prob_geo = y_prob_calibrated
    print(f"  Loaded calibrated probabilities from {calib_file}")
else:
    print(f"  WARNING: {calib_file} not found, using raw predictions from 06...")
    model_0 = build_model(model_name, best_params)
    model_0.fit(X_train, y_train)
    y_prob_geo = model_0.predict_proba(X_geo)[:, 1]
    y_prob_raw_0 = y_prob_geo.copy()

# 注：此处不再打印基于 05 概率的 0% AUC/Brier —— 成长曲线指标以 Part 1
# 循环计算值为唯一口径（见上方"口径统一说明"），y_prob_geo/y_prob_raw_0
# 仅供 Part 2/3 亚组分析使用。

# =========================================================================
# Step 3: Unified loop for all pct values (including 0%) with Platt calibration
# =========================================================================
print(f"  Running {n_repeats} repeats for all pct values...")

# 预先加载或拟合一个统一的校准器（仅在 Derivation 上拟合）
# 校准状态说明：与 05_Model_Development.py 保持一致——单一 Platt 校准器
# （LogisticRegression, penalty=None）仅在 Derivation 上拟合一次，随后对
# 所有成长曲线迭代与所有队列应用同一校准器，避免校准状态跨脚本漂移。
calib_model_path = os.path.join(MODEL_DEV_DIR, f"calibrator_{model_name}.pkl")
if os.path.exists(calib_model_path):
    with open(calib_model_path, "rb") as f:
        calib_model = pickle.load(f)
    print(f"  Loaded calibrator from {calib_model_path} (Platt, fit on Derivation only)")
else:
    # 在 Derivation 上拟合校准器（使用完整训练集，仅一次）
    print("  WARNING: Calibrator not found, fitting on Derivation (Platt, single fit)...")
    model_temp = build_model(model_name, best_params)
    model_temp.fit(X_train, y_train)
    y_prob_train = model_temp.predict_proba(X_train)[:, 1]
    y_prob_clipped = np.clip(y_prob_train, 1e-10, 1 - 1e-10)
    logit_train = np.log(y_prob_clipped / (1 - y_prob_clipped))
    calib_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    calib_model.fit(logit_train.reshape(-1, 1), y_train)
    print("  Fitted calibrator on Derivation (fallback, Platt)")

# Wrap outer loop with tqdm if available
ri_iter = tqdm(range(n_repeats), desc="  Growth-curve repeats",
               file=sys.stdout, ncols=80) if HAS_TQDM else range(n_repeats)

for ri in ri_iter:
    seed = RANDOM_STATE + ri
    rng = np.random.RandomState(seed)

    for pi, pct in enumerate(percentages):
        # Determine sample size and indices for this repeat
        if pct == 0.0:
            n_sample = 0
            idx = np.array([], dtype=int)
        else:
            n_sample = max(1, int(n_temporal * pct))
            idx = rng.choice(n_temporal, size=n_sample, replace=False)

        # Merge training data
        # 分析框架说明（有意为之，勿改）：本分析为"模型更新/数据积累研究"框架，
        # 将 Temporal 集按 pct 比例并入训练，模拟临床数据逐步积累对模型性能的
        # 增益。因此 Temporal 在本分析中不再作为独立外部验证集；评估始终在
        # Geographical 集上进行，Geographical 全程未参与任何训练/校准拟合。
        if n_sample == 0:
            X_merged = X_train
            y_merged = y_train
        else:
            X_merged = np.vstack([X_train, X_temporal[idx]])
            y_merged = np.hstack([y_train, y_temporal[idx]])

        model = build_model(model_name, best_params)
        model.fit(X_merged, y_merged)

        # 获取原始概率
        y_prob_raw = model.predict_proba(X_geo)[:, 1]

        # 应用统一校准器（固定，来自 05 或在 Derivation 上拟合）
        y_prob_clipped = np.clip(y_prob_raw, 1e-10, 1 - 1e-10)
        logit_geo = np.log(y_prob_clipped / (1 - y_prob_clipped))
        y_prob_geo_eval = calib_model.predict_proba(logit_geo.reshape(-1, 1))[:, 1]

        growth_auc_matrix[pi, ri] = roc_auc_score(y_geo, y_prob_geo_eval)
        growth_brier_matrix[pi, ri] = brier_score_loss(y_geo, y_prob_geo_eval)

    # Progress every 50 repeats
    if (ri + 1) % 50 == 0:
        print(f"    Completed {ri + 1}/{n_repeats} repeats")

# =========================================================================
# Step 4: Unified CI computation (AUC + Brier)
# =========================================================================
n_percentages = len(percentages)

# --- AUC arrays ---
growth_auc_mean = np.full(n_percentages, np.nan)
growth_auc_ci_lo = np.full(n_percentages, np.nan)
growth_auc_ci_hi = np.full(n_percentages, np.nan)
growth_auc_se = np.full(n_percentages, np.nan)

# --- Brier arrays ---
growth_brier_mean = np.full(n_percentages, np.nan)
growth_brier_ci_lo = np.full(n_percentages, np.nan)
growth_brier_ci_hi = np.full(n_percentages, np.nan)
growth_brier_se = np.full(n_percentages, np.nan)

# Apply BCa bootstrap_ci_from_values for all pct values (including 0%)
for pi in range(n_percentages):
    # --- AUC CI ---
    auc_vals = growth_auc_matrix[pi, :]
    auc_vals = auc_vals[~np.isnan(auc_vals)]
    if len(auc_vals) >= 5:
        mean_val, lo, hi = bootstrap_ci_from_values(auc_vals, n_iter=BOOTSTRAP_ITER)
        growth_auc_mean[pi] = mean_val
        growth_auc_ci_lo[pi] = lo
        growth_auc_ci_hi[pi] = hi
        growth_auc_se[pi] = np.std(auc_vals, ddof=1)
    else:
        growth_auc_mean[pi] = np.mean(auc_vals)
        growth_auc_se[pi] = np.std(auc_vals, ddof=1)
        growth_auc_ci_lo[pi] = growth_auc_mean[pi] - 1.96 * growth_auc_se[pi]
        growth_auc_ci_hi[pi] = growth_auc_mean[pi] + 1.96 * growth_auc_se[pi]

    # --- Brier CI ---
    brier_vals = growth_brier_matrix[pi, :]
    brier_vals = brier_vals[~np.isnan(brier_vals)]
    if len(brier_vals) >= 5:
        mean_val, lo, hi = bootstrap_ci_from_values(brier_vals, n_iter=BOOTSTRAP_ITER)
        growth_brier_mean[pi] = mean_val
        growth_brier_ci_lo[pi] = lo
        growth_brier_ci_hi[pi] = hi
        growth_brier_se[pi] = np.std(brier_vals, ddof=1)
    else:
        growth_brier_mean[pi] = np.mean(brier_vals)
        growth_brier_se[pi] = np.std(brier_vals, ddof=1)
        growth_brier_ci_lo[pi] = growth_brier_mean[pi] - 1.96 * growth_brier_se[pi]
        growth_brier_ci_hi[pi] = growth_brier_mean[pi] + 1.96 * growth_brier_se[pi]

# Compute n_added
n_added_vals = [max(1, int(n_temporal * pct)) if pct > 0 else 0 for pct in percentages]

# Print results
source_tag = "Platt + 100 reps"
print(f"\n{'%':>6s}  {'N_added':>8s}  {'AUC_mean':>10s}  {'AUC_CI (BCa)':>22s}  "
      f"{'Brier_mean':>11s}  {'Brier_CI (BCa)':>22s}  {'Source':>12s}")
for i, pct in enumerate(percentages):
    label = f"{pct*100:.0f}%" if pct > 0 else "0%"
    src = source_tag if i == 0 else "BCa"
    print(f" {label:>5s}  {n_added_vals[i]:>8d}  {growth_auc_mean[i]:>8.4f}  "
          f"[{growth_auc_ci_lo[i]:.4f}-{growth_auc_ci_hi[i]:.4f}]  "
          f"{growth_brier_mean[i]:>10.4f}  "
          f"[{growth_brier_ci_lo[i]:.4f}-{growth_brier_ci_hi[i]:.4f}]  {src:>12s}")

baseline_auc = growth_auc_mean[0]
full_auc = growth_auc_mean[-1]
baseline_brier = growth_brier_mean[0]
full_brier = growth_brier_mean[-1]
print(f"\n>>> [微调成长] 仅训练集：地理AUC={baseline_auc:.4f}, Brier={baseline_brier:.4f} "
      f"(Platt calibration)")
print(f">>> 加入时间集100%后：地理AUC={full_auc:.4f} "
      f"(95% CI: [{growth_auc_ci_lo[-1]:.4f}-{growth_auc_ci_hi[-1]:.4f}]) "
      f"(提升{full_auc - baseline_auc:+.4f})")
print(f">>>                   Brier={full_brier:.4f} "
      f"(95% CI: [{growth_brier_ci_lo[-1]:.4f}-{growth_brier_ci_hi[-1]:.4f}]) "
      f"(改善{baseline_brier - full_brier:+.4f})")

# Save CSV
growth_df = pd.DataFrame({
    "pct": percentages,
    "n_added": n_added_vals,
    "auc_mean": growth_auc_mean,
    "auc_se": growth_auc_se,
    "auc_ci_lo": growth_auc_ci_lo,
    "auc_ci_hi": growth_auc_ci_hi,
    "brier_mean": growth_brier_mean,
    "brier_se": growth_brier_se,
    "brier_ci_lo": growth_brier_ci_lo,
    "brier_ci_hi": growth_brier_ci_hi,
})
growth_df.to_csv(os.path.join(RESULT_DIR, "growth_curve_data.csv"), index=False)

# =========================================================================
# Plot: Growth Curve (Removed per user request — Growth_Curve_Split_{model}
#       is generated by 07_Clinical_Plots_R.R from growth_curve_data.csv)
# =========================================================================

# =============================================================================
# Part 2: Subgroup Performance Analysis (with Age De-Standardization)
# =============================================================================
print("\n" + "=" * 70)
print("Part 2: Subgroup Performance Analysis — Geographical Set")
print("=" * 70)

# --- Load age scaler params ---
age_median, age_iqr, is_age_logged = load_age_scaler_params()

# --- Load original data for subgroup columns ---
orig_geo_path = os.path.join(BASE_DIR, "Geographical external validation cohort.csv")
if not os.path.exists(orig_geo_path):
    orig_geo_path = os.path.join(PREPROC_DIR, "Geographical external validation cohort.csv")

orig_geo = pd.read_csv(orig_geo_path)
subgroup_cols_need = ["ID", "age", "gender", "hypertension"]
subgroup_cols_avail = [c for c in subgroup_cols_need if c in orig_geo.columns]
if "hypertension" not in orig_geo.columns:
    print("WARNING: 'hypertension' column not found. "
          "Hypertension-related subgroups will be skipped.")
orig_geo_sub = orig_geo[subgroup_cols_avail].copy()
print(f"  Subgroup columns loaded: {subgroup_cols_avail}")

# --- 使用从 05 加载的校准后概率进行亚组分析 ---
# y_prob_geo 已在上面赋值为校准后概率
y_prob_geo_raw = y_prob_raw_0  # 原始概率
# 说明：本脚本不输出基于阈值二分类的指标（accuracy/sensitivity 等），
# 原 y_pred_geo（默认 0.5 阈值）与 geo_eval["y_pred"] 已删除——0.5 阈值
# 与 05 报告的口径不一致，属隐患。如需阈值化指标，应使用 05 确定的最终
# Youden cutoff 而非默认 0.5。

geo_auc_overall = roc_auc_score(y_geo, y_prob_geo)
print(f"  Overall Geo AUC (Calibrated) = {geo_auc_overall:.4f}")

# --- Build geo dataframe with predictions ---
geo_eval = pd.DataFrame({
    "ID": geo_ids,
    "y_true": y_geo,
    "y_prob_calibrated": y_prob_geo,  # calibrated from 05
    "y_prob_raw": y_prob_geo_raw,     # raw (uncalibrated) from 05
})
geo_eval = geo_eval.merge(orig_geo_sub, on="ID", how="left")

# === Age De-Standardization ===
age_scaled = geo_df["age"].values.astype(np.float64)
age_raw = destandardize_age(age_scaled, age_median, age_iqr, is_age_logged)
# 反标准化年龄合理性校验（对 scaler 参数回退场景强制生效）：
# 若 preprocessing_objects.joblib 加载失败而使用回退参数（median=60/iqr=15），
# 错误的 scaler 参数会使 ESC SCORE2 年龄分组全部静默算错，因此此处校验：
# 取值须在 [18, 100] 区间、均值须在 [40, 80]，越界即终止并给出明确错误信息。
if age_raw.min() < 18 or age_raw.max() > 100 or not (40 <= age_raw.mean() <= 80):
    sys.exit(
        f"ERROR: 反标准化年龄越界 (min={age_raw.min():.1f}, "
        f"max={age_raw.max():.1f}, mean={age_raw.mean():.1f})，"
        f"超出合理范围（要求取值 18-100、均值 40-80）。"
        f"年龄 scaler 参数可能有误——请确认已先运行 01 数据预处理脚本生成 "
        f"preprocessing_objects.joblib，再重新运行本脚本。"
    )
geo_eval["age_raw"] = age_raw
print(f"  Age de-standardization: median={age_median}, iqr={age_iqr}, "
      f"log={is_age_logged}")
print(f"  age_raw: min={age_raw.min():.1f}, max={age_raw.max():.1f}, "
      f"mean={age_raw.mean():.1f}")

# === SCORE2 Three-category Age Groups ===
geo_eval["Age_Group"] = pd.cut(
    geo_eval["age_raw"],
    bins=[0, 50, 70, 200],
    labels=["<50", "50-69", ">=70"],
    right=False
)
age_counts = geo_eval["Age_Group"].value_counts().sort_index()
print(f"  Age_Group distribution: {dict(age_counts)}")

# --- Define subgroups ---
has_hypertension = "hypertension" in geo_eval.columns

# 数据字典：gender 0=男性(Male)，1=女性(Female)（用户确认，2026-08-06）
geo_eval["Gender"] = geo_eval["gender"].map({0: "Male", 1: "Female"})
if has_hypertension:
    geo_eval["Hypertension"] = geo_eval["hypertension"].map({0: "No HTN", 1: "HTN"})

# --- Build subgroup definitions (aligned with 07 Subgroup_Forest_Plot_MLP: Age/Gender/HTN) ---
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

# --- Compute subgroup metrics with Bootstrap AUC CI ---
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
        ci_lo, ci_hi = auc_lo, auc_hi
    else:
        metric_type = "N/A"
        metric_val = np.nan
        ci_lo, ci_hi = np.nan, np.nan
        print(f"  Subgroup too small (n_pos={n_pos}, n_neg={n_neg}) — AUC not reported")

    subgroup_results.append({
        "Subgroup": sname,
        "N": n,
        "N_Pos": n_pos,
        "N_Neg": n_neg,
        "Metric_Type": metric_type,
        "Metric_Value": metric_val,
        "CI_Lower": ci_lo,
        "CI_Upper": ci_hi,
        "Signif_Worse": (metric_type == "AUC" and ci_hi < 0.5),
        "P_Value_vs_Random": np.nan,
    })

subgroup_df = pd.DataFrame(subgroup_results)
# FDR (BH) correction for multiple subgroup comparisons (exploratory analysis)
# Compute bootstrap p-value for each AUC subgroup vs. random (AUC=0.5)
auc_mask = subgroup_df["Metric_Type"] == "AUC"
n_auc_subgroups = int(auc_mask.sum())
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
        # 双侧经验 Bootstrap p 值：2 * min(P(boot < 0.5), P(boot > 0.5))，
        # 与 95% 双侧 Bootstrap CI 口径一致（此前仅用单侧 P(boot < 0.5)）。
        if len(bs_auc) > 0:
            p_less = np.mean(bs_auc < 0.5)
            p_val = np.clip(2.0 * min(p_less, 1.0 - p_less), 0.0, 1.0)
        else:
            p_val = 1.0
        subgroup_df.loc[i, "P_Value_vs_Random"] = p_val
    raw_p_vals = subgroup_df.loc[auc_mask, "P_Value_vs_Random"].values
    # BH-FDR correction for subgroup comparisons (exploratory analysis)
    from statsmodels.stats.multitest import multipletests
    reject, adj_p_vals, _, _ = multipletests(raw_p_vals, method='fdr_bh')
    subgroup_df.loc[auc_mask, "P_Value_FDR"] = adj_p_vals
    # 方向约束（关键修复）：双侧 p 值只说明"与随机（AUC=0.5）存在差异"，
    # 不区分方向——AUC 显著高于 0.5 的优良亚组双侧 p 同样显著，若仅凭
    # adj_p < 0.05 判定会被误标为"显著更差"，导致下游 07 森林图把表现好的
    # 亚组错误标红。因此要求 FDR 显著 且 点估计 AUC < 0.5 才判定"更差"。
    subgroup_df.loc[auc_mask, "Signif_Worse_FDR"] = (
        (adj_p_vals < 0.05)
        & (subgroup_df.loc[auc_mask, "Metric_Value"].values < 0.5)
    )
    # Also keep Bonferroni for reference (used in primary analysis)
    adj_p_bonf = np.minimum(raw_p_vals * n_auc_subgroups, 1.0)
    subgroup_df.loc[auc_mask, "P_Value_Bonferroni"] = adj_p_bonf
    subgroup_df.loc[auc_mask, "Signif_Worse_Bonferroni"] = adj_p_bonf < 0.05
subgroup_df.to_csv(os.path.join(RESULT_DIR, f"Subgroup_Performance_{model_name}.csv"), index=False)

# --- Identify weakest subgroups ---
auc_subgroups = subgroup_df[subgroup_df["Metric_Type"] == "AUC"].copy()
if len(auc_subgroups) > 0:
    lowest = auc_subgroups.loc[auc_subgroups["Metric_Value"].idxmin()]
    print(f"\n>>> [脆弱亚组] 最低AUC亚组：{lowest['Subgroup']} "
          f"(AUC={lowest['Metric_Value']:.4f}, "
          f"95% CI: [{lowest['CI_Lower']:.4f}-{lowest['CI_Upper']:.4f}])，"
          f"建议重训时增加此类患者数据。")

# --- Plot: Forest Plot (Removed per user request — Subgroup_Forest_Plot_MLP
#     is generated by 07_Clinical_Plots_R.R from Subgroup_Performance CSV) ---

# =============================================================================
# Part 3: Calibration Comparison — Subgroup Brier Before/After (05 Calibration)
# =============================================================================
print("\n" + "=" * 70)
print("Part 3: Subgroup Calibration Comparison — Geographical Set")
print("=" * 70)

# 使用 05 的校准概率（geo_eval 已包含 y_prob_calibrated 和 y_prob_raw 列）
print(f"  Using 05 probabilities: raw (Before) vs. recalibrated (After)")

# --- Compute calibration metrics for each subgroup ---
calib_results = []
print(f"\n{'Subgroup':<25s}  {'Brier_Before':>12s}  {'Brier_After':>12s}  {'Delta_Brier':>12s}")
print("-" * 75)

for sname, mask in subgroup_defs:
    n = int(mask.sum())
    y_sub = geo_eval.loc[mask, "y_true"].values
    y_prob_before = geo_eval.loc[mask, "y_prob_raw"].values  # raw from 05
    y_prob_after = geo_eval.loc[mask, "y_prob_calibrated"].values  # calibrated from 05

    # 样本量阈值（与 Part 2 一致：n_pos>=10 且 n_neg>=10）：
    # 样本过小的亚组校准指标（尤其 GLM 校准斜率）不稳定且无意义，
    # 记为 NaN 并跳过——不进入 Part 4 的 Delta_Brier 排序与 lollipop 绘图。
    n_pos = int(np.sum(y_sub == 1))
    n_neg = int(np.sum(y_sub == 0))
    if n_pos < 10 or n_neg < 10:
        calib_results.append({
            "Subgroup": sname,
            "N": n,
            "Slope_Before": np.nan,
            "Slope_After": np.nan,
            "Delta_Slope": np.nan,
            "Brier_Before": np.nan,
            "Brier_After": np.nan,
            "Delta_Brier": np.nan,
        })
        print(f"  {sname:<25s}  跳过 (n_pos={n_pos}, n_neg={n_neg}，样本量不足)")
        continue

    brier_before, inter_before, slope_before = compute_calibration_metrics(y_sub, y_prob_before)
    brier_after, inter_after, slope_after = compute_calibration_metrics(y_sub, y_prob_after)

    delta_brier = brier_after - brier_before
    delta_slope = slope_after - slope_before

    calib_results.append({
        "Subgroup": sname,
        "N": n,
        "Slope_Before": slope_before,
        "Slope_After": slope_after,
        "Delta_Slope": delta_slope,
        "Brier_Before": brier_before,
        "Brier_After": brier_after,
        "Delta_Brier": delta_brier,
    })

    delta_sign = "+" if delta_brier >= 0 else ""
    print(f"  {sname:<25s}  {brier_before:>12.4f}  {brier_after:>12.4f}  "
          f"{delta_sign}{delta_brier:>11.4f}")

calib_df = pd.DataFrame(calib_results)
calib_df.to_csv(os.path.join(RESULT_DIR, "Subgroup_Calibration_Comparison.csv"), index=False)
print(f"\n  Calibration comparison saved: Subgroup_Calibration_Comparison.csv")

# =============================================================================
# Part 4: Horizontal Lollipop Chart — Subgroup Brier Improvement
# =============================================================================
print("\n" + "=" * 70)
print("Part 4: Horizontal Lollipop Chart — Subgroup Brier Improvement")
print("=" * 70)

# Sort by Delta_Brier ascending (most improved / most negative first)
# 仅纳入有效计算的亚组：Part 3 中样本量不足（n_pos/n_neg < 10）或 GLM
# 拟合失败的亚组 Delta_Brier 为 NaN，排序与绘图前予以剔除。
calib_plot = (calib_df.dropna(subset=["Delta_Brier"])
              .sort_values("Delta_Brier", ascending=True).reset_index(drop=True))
n_sub_calib = len(calib_plot)
n_improved = int(np.sum(calib_plot["Delta_Brier"] < 0))

# Color and style rules
point_colors = ["#2CA02C" if d < 0 else "#D62728" for d in calib_plot["Delta_Brier"]]

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(n_sub_calib)

# ---- Draw horizontal lollipops ----
for i in range(n_sub_calib):
    val = calib_plot["Delta_Brier"].iloc[i]
    color = point_colors[i]
    # Stem (thin horizontal line)
    ax.plot([0, val], [y_pos[i], y_pos[i]], "-", color="lightgray", lw=2, alpha=0.8, zorder=1)
    # Dot at end
    ax.scatter(val, y_pos[i], s=150, color=color, zorder=3, edgecolors="white", linewidth=0.5)

# ---- Vertical reference line at x=0 ----
ax.axvline(x=0, color="black", linestyle="--", lw=0.8, alpha=0.5, zorder=0)

# ---- Data labels at dot ends ----
for i in range(n_sub_calib):
    val = calib_plot["Delta_Brier"].iloc[i]
    color = point_colors[i]
    x_offset = 0.002
    ax.text(val + x_offset, y_pos[i], f"{val:.4f}",
            fontsize=9, fontweight="bold", ha="left", va="center",
            color=color)

# ---- Y-axis ----
ax.set_yticks(y_pos)
ax.set_yticklabels(calib_plot["Subgroup"].values, fontsize=10)
ax.set_ylabel("")  # Remove Y-axis title

# ---- X-axis ----
ax.set_xlabel("Change in Brier Score (After - Before)", fontsize=11)

# ---- Remove top and right spines ----
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---- Light vertical grid ----
ax.grid(True, alpha=0.2, axis="x")

# ---- Summary box (upper left) ----
summary_text = f"Brier improved in {n_improved}/{n_sub_calib} subgroups"
ax.text(0.02, 0.97, summary_text, transform=ax.transAxes,
        fontsize=10, fontweight="bold", color="#2CA02C",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.9))

# ---- Legend (lower right) ----
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2CA02C",
           markersize=10, label="Brier Improved"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#D62728",
           markersize=10, label="Brier Worsened"),
]
ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

# ---- Adjust x-limits for labels ----
x_min = calib_plot["Delta_Brier"].min()
x_max = calib_plot["Delta_Brier"].max()
x_pad = max(abs(x_max - x_min) * 0.25, 0.01)
ax.set_xlim(x_min - x_pad, x_max + x_pad)

plt.tight_layout()

save_figure(fig, "subgroup_brier_lollipop")
print("  Lollipop chart saved.")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print(f"=== Model Growth Analysis Complete — {model_name} ===")
print(f"  Growth curve data:    {RESULT_DIR}/growth_curve_data.csv")
print(f"  Subgroup performance:  {RESULT_DIR}/Subgroup_Performance_{model_name}.csv")
print(f"  Calibration comparison: {RESULT_DIR}/Subgroup_Calibration_Comparison.csv")
print(f"  Calibration lollipop chart: {RESULT_DIR}/subgroup_brier_lollipop.*")
print("=" * 70)