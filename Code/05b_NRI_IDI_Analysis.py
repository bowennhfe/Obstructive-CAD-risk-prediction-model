# =============================================================================
# 05b_NRI_IDI_Analysis.py — NRI / IDI 敏感性分析
# =============================================================================
# 功能：
#   以推荐模型（model_selection_metrics.csv 中 AB_Score 最高者）为「新模型」，
#   与「基线模型」（优先单变量 age 的临床模型；若不可得则用 Treat All 常数
#   患病率）逐队列对比，计算 NRI（连续 + 分类）与 IDI，并给出分层 bootstrap
#   95% CI。仅作敏感性分析，不替代主表（AUC / Brier / 校准）。
#
# 输入：
#   - <repo>/Model Selection/model_selection_metrics.csv（推荐模型 + Optimal_Cutoff）
#   - <repo>/Model Selection/oof_predictions_all_models.csv（Derivation，repeat-1
#     代表性 OOF，列: model, fold, y_true, y_pred_prob）
#   - <repo>/Model Development/predicted_probabilities_{model}_{cohort}.csv
#     （Temporal / Geographical，列: y_true, y_pred_prob_raw, y_pred_prob_calibrated）
#   - <repo>/Derivation cohort.csv、<repo>/Temporal external validation cohort.csv、
#     <repo>/Geographical external validation cohort.csv（age/label 列，必须为原始
#     未标准化尺度；feature_selection/ 副本的 age 已做 RobustScaler，不可用于基线）
#
# 输出（均在 <repo>/Model Development）：
#   - nri_idi_results.csv
#   - nri_idi_report.txt
#
# 口径说明：
#   - Derivation 的新模型概率取 04 的 repeat-1 OOF（非表观拟合概率），避免乐观。
#     OOF 行按外层折顺序拼接，需按 StratifiedKFold(n_splits=folds,
#     shuffle=True, random_state=RANDOM_STATE) 复现测试折顺序并与 y_true 逐例
#     校验后，才能把 age 对齐到 OOF 行；校验失败则回退 Treat All。
#   - 外部队列的新模型概率用原始概率 y_pred_prob_raw（NRI/IDI 属排序/分层类
#     指标，不受 Platt 校准单调变换影响；用 raw 与主表口径一致）。
#   - 分类 NRI 统一使用 Derivation 的 Optimal_Cutoff（跨队列同一阈值）。
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# Global Configuration（与 05_Model_Development.py 保持一致）
# =============================================================================
RANDOM_STATE = 123
BOOTSTRAP_ITER = 1000
EPS = 1e-9  # 概率变化方向判定阈值（连续 NRI）

# Paths：由脚本自身路径推导项目根，与 05 完全一致的写法
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "feature_selection")
MODEL_SEL_DIR = os.path.join(BASE_DIR, "Model Selection")
RESULT_DIR = os.path.join(BASE_DIR, "Model Development")
os.makedirs(RESULT_DIR, exist_ok=True)

# 队列 → 05 输出的预测文件后缀 / 原始队列 CSV 文件名
COHORTS_EXTERNAL = [
    ("Temporal", "temporal", "Temporal external validation cohort.csv"),
    ("Geographical", "geographical", "Geographical external validation cohort.csv"),
]


# =============================================================================
# 指标实现（纯 numpy；所有函数对 0 样本 / 单一类别 / 长度不一致返回 NaN）
# =============================================================================
def _prepare_metric_inputs(y, p_new, p_base, tag):
    """统一转为 ndarray 并做边界校验；不合法时打印 WARNING 并返回 None。"""
    y = np.asarray(y).astype(int).ravel()
    p_new = np.asarray(p_new, dtype=float).ravel()
    p_base = np.asarray(p_base, dtype=float).ravel()
    if len(y) == 0:
        print(f"  WARNING: [{tag}] 样本量为 0，返回 NaN")
        return None
    if not (len(y) == len(p_new) == len(p_base)):
        print(f"  WARNING: [{tag}] y / p_new / p_base 长度不一致，返回 NaN")
        return None
    if y.min() == y.max():
        print(f"  WARNING: [{tag}] 只有单一类别（无事件或无对照），返回 NaN")
        return None
    return y, p_new, p_base


def compute_nri_continuous(y, p_new, p_base, eps=EPS, tag="NRI_continuous"):
    """连续 NRI（按概率变化方向判定，超过 eps 记为上升/下降）。

    事件组:  P(new > base) - P(new < base)
    非事件组: P(new < base) - P(new > base)   （方向相反，越往下越好）
    合计 = 事件组 + 非事件组
    返回 (NRI_events, NRI_nonevents, NRI_total)
    """
    prepared = _prepare_metric_inputs(y, p_new, p_base, tag)
    if prepared is None:
        return np.nan, np.nan, np.nan
    y, p_new, p_base = prepared
    ev = y == 1
    ne = y == 0
    diff = p_new - p_base
    up = diff > eps
    down = diff < -eps
    nri_ev = float(np.mean(up[ev]) - np.mean(down[ev]))
    nri_ne = float(np.mean(down[ne]) - np.mean(up[ne]))
    return nri_ev, nri_ne, nri_ev + nri_ne


def compute_nri_categorical(y, p_new, p_base, cutoff, tag="NRI_categorical"):
    """分类 NRI（以 Derivation 的 Optimal_Cutoff 为阈值）。

    事件组:  P(new >= c) - P(base >= c)   （向上重分类）
    非事件组: P(base >= c) - P(new >= c)  （向下重分类）
    返回 (NRI_cat_events, NRI_cat_nonevents, NRI_cat_total)
    """
    if cutoff is None or not np.isfinite(cutoff):
        print(f"  WARNING: [{tag}] 阈值不可用（NaN），返回 NaN")
        return np.nan, np.nan, np.nan
    prepared = _prepare_metric_inputs(y, p_new, p_base, tag)
    if prepared is None:
        return np.nan, np.nan, np.nan
    y, p_new, p_base = prepared
    ev = y == 1
    ne = y == 0
    new_hi = p_new >= cutoff
    base_hi = p_base >= cutoff
    nri_ev = float(np.mean(new_hi[ev]) - np.mean(base_hi[ev]))
    nri_ne = float(np.mean(base_hi[ne]) - np.mean(new_hi[ne]))
    return nri_ev, nri_ne, nri_ev + nri_ne


def compute_idi(y, p_new, p_base, tag="IDI"):
    """IDI = 事件组平均概率差 - 非事件组平均概率差。返回 (IDI, IDI_events, IDI_nonevents)。"""
    prepared = _prepare_metric_inputs(y, p_new, p_base, tag)
    if prepared is None:
        return np.nan, np.nan, np.nan
    y, p_new, p_base = prepared
    ev = y == 1
    ne = y == 0
    idi_ev = float(np.mean(p_new[ev]) - np.mean(p_base[ev]))
    idi_ne = float(np.mean(p_new[ne]) - np.mean(p_base[ne]))
    return idi_ev - idi_ne, idi_ev, idi_ne


def bootstrap_nri_idi(y, p_new, p_base, cutoff, n_boot=BOOTSTRAP_ITER,
                      seed=RANDOM_STATE, tag="bootstrap"):
    """分层 bootstrap：事件组与非事件组各自有放回重抽（保持原样本量）。

    返回 (NRI_total_CI_low, NRI_total_CI_high, IDI_CI_low, IDI_CI_high)，
    区间取 2.5 / 97.5 百分位。
    """
    prepared = _prepare_metric_inputs(y, p_new, p_base, tag)
    if prepared is None:
        return np.nan, np.nan, np.nan, np.nan
    y, p_new, p_base = prepared
    ev_idx = np.where(y == 1)[0]
    ne_idx = np.where(y == 0)[0]
    if len(ev_idx) == 0 or len(ne_idx) == 0:
        print(f"  WARNING: [{tag}] 事件组或非事件组样本为 0，无法分层 bootstrap")
        return np.nan, np.nan, np.nan, np.nan

    rng = np.random.RandomState(seed)
    nri_vals = np.full(n_boot, np.nan, dtype=float)
    idi_vals = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        sel_ev = rng.choice(ev_idx, size=len(ev_idx), replace=True)
        sel_ne = rng.choice(ne_idx, size=len(ne_idx), replace=True)
        idx = np.concatenate([sel_ev, sel_ne])
        # 重抽后事件/非事件均非空，内部指标不会触发边界 WARNING
        _, _, nri_total = compute_nri_continuous(y[idx], p_new[idx], p_base[idx], tag=tag)
        idi, _, _ = compute_idi(y[idx], p_new[idx], p_base[idx], tag=tag)
        nri_vals[b] = nri_total
        idi_vals[b] = idi

    if np.all(np.isnan(nri_vals)) or np.all(np.isnan(idi_vals)):
        print(f"  WARNING: [{tag}] bootstrap 全部为 NaN，无法给出区间")
        return np.nan, np.nan, np.nan, np.nan
    nri_lo, nri_hi = np.nanpercentile(nri_vals, [2.5, 97.5])
    idi_lo, idi_hi = np.nanpercentile(idi_vals, [2.5, 97.5])
    return float(nri_lo), float(nri_hi), float(idi_lo), float(idi_hi)


# =============================================================================
# 数据加载与基线概率构造
# =============================================================================
def find_raw_derivation_path():
    """与 04/pipeline_lib 一致：优先 <repo>/Derivation cohort.csv，其次 feature_selection。"""
    exact = os.path.join(BASE_DIR, "Derivation cohort.csv")
    if os.path.exists(exact):
        return exact
    fallback = os.path.join(DATA_DIR, "Derivation cohort.csv")
    if os.path.exists(fallback):
        return fallback
    return None


def align_derivation_age(oof_sub):
    """把 Derivation 的 age 对齐到 OOF 行（OOF 按外层测试折顺序拼接）。

    复现 04 的 repeat-1 外层划分（StratifiedKFold, shuffle=True,
    random_state=RANDOM_STATE），按折顺序拼接测试索引，并用 y_true 逐例校验。
    任何一步不满足（文件缺失 / 缺 age / 含缺失 / 长度或标签不一致）返回 None。
    """
    raw_path = find_raw_derivation_path()
    if raw_path is None:
        print("  WARNING: 找不到原始 Derivation cohort.csv，age 基线不可用")
        return None
    raw = pd.read_csv(raw_path)
    if "age" not in raw.columns or "label" not in raw.columns:
        print("  WARNING: Derivation cohort.csv 缺少 age/label 列，age 基线不可用")
        return None
    if len(raw) != len(oof_sub):
        print(f"  WARNING: Derivation 行数({len(raw)}) 与 OOF 行数({len(oof_sub)}) 不一致，"
              f"age 基线不可用")
        return None
    if raw["age"].isna().any():
        print("  WARNING: Derivation age 含缺失值，age 基线不可用")
        return None
    try:
        n_splits = int(oof_sub["fold"].nunique())
        if n_splits < 2:
            print(f"  WARNING: OOF 折数异常（{n_splits}），age 基线不可用")
            return None
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        order = np.concatenate([te for _, te in cv.split(raw, raw["label"].values)])
    except Exception as exc:
        print(f"  WARNING: 复现外层折顺序失败（{exc}），age 基线不可用")
        return None
    if len(order) != len(oof_sub) or not np.array_equal(
            raw["label"].values.astype(int)[order],
            oof_sub["y_true"].values.astype(int)):
        print("  WARNING: OOF 行顺序与复现的外层折顺序不一致（y_true 校验失败），"
              "age 基线不可用")
        return None
    return raw["age"].values[order].astype(float)


def load_derivation_oof(model_name):
    """读取 04 的 repeat-1 代表性 OOF，仅保留推荐模型。"""
    path = os.path.join(MODEL_SEL_DIR, "oof_predictions_all_models.csv")
    if not os.path.exists(path):
        print(f"  WARNING: 缺少 OOF 文件（{path}），跳过 Derivation")
        return None
    df = pd.read_csv(path)
    needed = {"model", "y_true", "y_pred_prob"}
    if not needed.issubset(df.columns):
        print(f"  WARNING: OOF 列名不符（实际: {list(df.columns)}；需要 {sorted(needed)}），"
              f"跳过 Derivation")
        return None
    sub = df.loc[df["model"] == model_name].reset_index(drop=True)
    if len(sub) == 0:
        print(f"  WARNING: OOF 中找不到推荐模型 {model_name}，跳过 Derivation")
        return None
    if "fold" not in sub.columns:
        print("  WARNING: OOF 缺少 fold 列，无法对齐 age，Derivation 回退 Treat All")
        age = None
    else:
        age = align_derivation_age(sub)
    y = sub["y_true"].values.astype(int)
    p_new = sub["y_pred_prob"].values.astype(float)
    return {"Cohort": "Derivation", "y": y, "p_new": p_new, "age": age}


def load_external_cohort(cohort_name, suffix, csv_name, model_name):
    """读取 05 写出的 {model}_{cohort} 预测文件；缺失/列名不符时打印 WARNING 并跳过。"""
    pred_path = os.path.join(
        RESULT_DIR, f"predicted_probabilities_{model_name}_{suffix}.csv")
    if not os.path.exists(pred_path):
        print(f"  WARNING: 缺少预测文件（{pred_path}），跳过 {cohort_name}")
        return None
    pred = pd.read_csv(pred_path)
    if not {"y_true", "y_pred_prob_raw"}.issubset(pred.columns):
        print(f"  WARNING: {cohort_name} 预测文件列名不符（实际: {list(pred.columns)}），"
              f"跳过 {cohort_name}")
        return None
    y = pred["y_true"].values.astype(int)
    p_new = pred["y_pred_prob_raw"].values.astype(float)

    # age 对齐：05 的预测文件按队列原始行顺序写出，用 label 逐例校验。
    # 只能用项目根目录下的原始队列文件：feature_selection/ 中的副本 age 已做
    # RobustScaler 标准化，若与原始尺度拟合的 age-only 模型搭配会得到错误基线概率。
    age = None
    cohort_path = os.path.join(BASE_DIR, csv_name)
    if not os.path.exists(cohort_path):
        print(f"  WARNING: 找不到原始队列文件（{cohort_name} 需 {cohort_path}），"
              f"{cohort_name} 回退 Treat All（feature_selection/ 副本的 age 已标准化，"
              f"不可用于 age-only 基线）")
    else:
        raw = pd.read_csv(cohort_path)
        if "age" not in raw.columns or "label" not in raw.columns:
            print(f"  WARNING: {csv_name} 缺少 age/label 列，{cohort_name} 回退 Treat All")
        elif len(raw) != len(pred):
            print(f"  WARNING: {cohort_name} 行数({len(raw)}) 与预测文件({len(pred)}) 不一致，"
                  f"回退 Treat All")
        elif not np.array_equal(raw["label"].values.astype(int), y):
            print(f"  WARNING: {cohort_name} 原始队列与预测文件行顺序不一致（label 校验失败），"
                  f"回退 Treat All")
        elif raw["age"].isna().any():
            print(f"  WARNING: {cohort_name} age 含缺失值，回退 Treat All")
        else:
            age = raw["age"].values.astype(float)
    return {"Cohort": cohort_name, "y": y, "p_new": p_new, "age": age}


def fit_age_logistic(age, y):
    """在 Derivation 上拟合单变量 age 逻辑回归（基线模型）；不可用时返回 None。"""
    if age is None:
        return None
    y = np.asarray(y).astype(int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        print("  WARNING: Derivation 单一类别，age 基线模型不可用")
        return None
    try:
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
        model.fit(np.asarray(age, dtype=float).reshape(-1, 1), y)
    except Exception as exc:
        print(f"  WARNING: age 基线模型拟合失败（{exc}），回退 Treat All")
        return None
    return model


def build_baseline(record, age_model):
    """构造基线概率并返回 (p_base, comparator 描述)。

    优先 age 单变量模型（统一在 Derivation 拟合，再应用于各队列 age）；
    否则 Treat All：常数概率 = 该队列患病率。
    """
    y = record["y"]
    age = record["age"]
    if age_model is not None and age is not None:
        p_base = age_model.predict_proba(np.asarray(age, dtype=float).reshape(-1, 1))[:, 1]
        return p_base, "Age-only logistic (single clinical variable, fitted on Derivation)"
    prevalence = float(np.mean(y)) if len(y) else np.nan
    if not np.isfinite(prevalence):
        print(f"  WARNING: {record['Cohort']} 患病率不可用，Treat All 基线设为 NaN")
    return np.full(len(y), prevalence, dtype=float), \
        f"Treat All (constant probability = cohort prevalence {prevalence:.4f})"


# =============================================================================
# Main
# =============================================================================
print("=" * 70)
print("=== NRI / IDI Sensitivity Analysis ===")
print("=" * 70)

# --- Step 1: 读取推荐模型（AB_Score 最高）与 Optimal_Cutoff ---
metrics_path = os.path.join(MODEL_SEL_DIR, "model_selection_metrics.csv")
if not os.path.exists(metrics_path):
    raise FileNotFoundError(
        f"缺少模型选择指标文件：{metrics_path}（请先运行 04_Model_Selection.py）")
metrics_df = pd.read_csv(metrics_path)
if not {"Model", "AB_Score", "Optimal_Cutoff"}.issubset(metrics_df.columns):
    raise ValueError(f"model_selection_metrics.csv 列名不符：{list(metrics_df.columns)}")
best_row = metrics_df.loc[metrics_df["AB_Score"].idxmax()]
model_name = str(best_row["Model"])
cutoff = float(best_row["Optimal_Cutoff"])
print(f"  推荐模型（AB_Score 最高）: {model_name}  "
      f"(AB_Score={best_row['AB_Score']:.4f}, Optimal_Cutoff={cutoff:.4f})")

# --- Step 2: 加载三个队列 ---
print("\n=== Loading cohorts ===")
records = []

der = load_derivation_oof(model_name)
if der is not None:
    print(f"  Derivation: n={len(der['y'])}, 事件={int(np.sum(der['y'] == 1))}, "
          f"非事件={int(np.sum(der['y'] == 0))}, "
          f"age 对齐={'是' if der['age'] is not None else '否（回退 Treat All）'}")
    records.append(der)

for cohort_name, suffix, csv_name in COHORTS_EXTERNAL:
    rec = load_external_cohort(cohort_name, suffix, csv_name, model_name)
    if rec is not None:
        print(f"  {cohort_name}: n={len(rec['y'])}, 事件={int(np.sum(rec['y'] == 1))}, "
              f"非事件={int(np.sum(rec['y'] == 0))}, "
              f"age 对齐={'是' if rec['age'] is not None else '否（回退 Treat All）'}")
        records.append(rec)

if not records:
    print("\nWARNING: 无任何可用队列，仅写出方法说明报告，不生成结果行。")

# --- Step 3: 基线模型（age-only，仅在 Derivation 拟合）---
print("\n=== Baseline model ===")
der_record = next((r for r in records if r["Cohort"] == "Derivation"), None)
age_model = None
if der_record is not None:
    age_model = fit_age_logistic(der_record["age"], der_record["y"])
    print("  基线 = age-only logistic（Derivation 拟合）"
          if age_model is not None else "  基线 = Treat All（age 不可用）")
else:
    print("  Derivation 不可用 → 外部队列使用 Treat All 基线")

# --- Step 4: 逐队列计算 NRI / IDI 及 bootstrap CI ---
print("\n=== Computing NRI / IDI ===")
rows = []
for rec in records:
    cohort_name = rec["Cohort"]
    y, p_new = rec["y"], rec["p_new"]
    p_base, comparator = build_baseline(rec, age_model)
    print(f"\n  [{cohort_name}] comparator = {comparator}")

    nri_ev, nri_ne, nri_total = compute_nri_continuous(
        y, p_new, p_base, tag=f"{cohort_name} NRI_continuous")
    cat_ev, cat_ne, cat_total = compute_nri_categorical(
        y, p_new, p_base, cutoff, tag=f"{cohort_name} NRI_categorical")
    idi, idi_ev, idi_ne = compute_idi(y, p_new, p_base, tag=f"{cohort_name} IDI")
    nri_lo, nri_hi, idi_lo, idi_hi = bootstrap_nri_idi(
        y, p_new, p_base, cutoff, tag=f"{cohort_name} bootstrap")

    print(f"    NRI(continuous)={nri_total:.4f} "
          f"[{nri_lo:.4f}, {nri_hi:.4f}]  "
          f"(events={nri_ev:.4f}, nonevents={nri_ne:.4f})")
    print(f"    NRI(categorical)={cat_total:.4f} "
          f"(events={cat_ev:.4f}, nonevents={cat_ne:.4f})")
    print(f"    IDI={idi:.4f} [{idi_lo:.4f}, {idi_hi:.4f}] "
          f"(events={idi_ev:.4f}, nonevents={idi_ne:.4f})")

    rows.append({
        "Model": model_name,
        "Cohort": cohort_name,
        "Comparator": comparator,
        "NRI_continuous": nri_total,
        "NRI_categorical": cat_total,
        "NRI_total": nri_total,       # 连续 NRI 合计（= 事件组 + 非事件组）
        "NRI_events": nri_ev,
        "NRI_nonevents": nri_ne,
        "NRI_CI_Lower": nri_lo,
        "NRI_CI_Upper": nri_hi,
        "IDI": idi,
        "IDI_events": idi_ev,
        "IDI_nonevents": idi_ne,
        "IDI_CI_Lower": idi_lo,
        "IDI_CI_Upper": idi_hi,
        "N_Events": int(np.sum(y == 1)),
        "N_NonEvents": int(np.sum(y == 0)),
        "Bootstrap_Iter": BOOTSTRAP_ITER,
    })

result_df = pd.DataFrame(rows, columns=[
    "Model", "Cohort", "Comparator",
    "NRI_continuous", "NRI_categorical", "NRI_total",
    "NRI_events", "NRI_nonevents", "NRI_CI_Lower", "NRI_CI_Upper",
    "IDI", "IDI_events", "IDI_nonevents", "IDI_CI_Lower", "IDI_CI_Upper",
    "N_Events", "N_NonEvents", "Bootstrap_Iter",
])
csv_path = os.path.join(RESULT_DIR, "nri_idi_results.csv")
result_df.to_csv(csv_path, index=False, float_format="%.6f")
print(f"\n  Saved: {csv_path} ({len(result_df)} rows)")

# --- Step 5: 文本报告 ---
print("\n=== Writing report ===")
comparator_lines = []
for rec in records:
    _, comparator = build_baseline(rec, age_model)
    comparator_lines.append(f"  - {rec['Cohort']}: {comparator}")
comparator_text = "\n".join(comparator_lines) if comparator_lines else "  - （无可用队列）"

report_lines = [
    "=" * 70,
    "NRI / IDI 敏感性分析报告",
    "=" * 70,
    f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"推荐模型（model_selection_metrics.csv 中 AB_Score 最高）: {model_name}",
    f"分类 NRI 阈值 Optimal_Cutoff（Derivation）: {cutoff:.6f}",
    "",
    "-" * 70,
    "一、方法说明",
    "-" * 70,
    "1. 新模型概率来源",
    "   - Derivation: 04_Model_Selection.py 的 repeat-1 代表性 OOF 预测",
    "     （oof_predictions_all_models.csv，列 model/fold/y_true/y_pred_prob，",
    "     仅取推荐模型行），非表观拟合概率，避免乐观偏倚。",
    "   - Temporal / Geographical: 05_Model_Development.py 写出的",
    "     predicted_probabilities_{model}_{cohort}.csv 中的 y_true + y_pred_prob_raw。",
    "     NRI/IDI 只依赖概率排序与分层，Platt 校准是单调变换，故用 raw 概率与主表一致。",
    "",
    "2. 基线概率（Comparator）",
    comparator_text,
    "   说明：age 单变量逻辑回归统一在 Derivation 队列拟合（1 个临床变量，",
    "   不含其他协变量、不做交叉验证），再将该模型应用于各队列的 age 得到基线概率；",
    "   当 age 无法可靠对齐或缺失时，该队列回退为 Treat All（常数概率 = 该队列患病率）。",
    "   Derivation 的 OOF 行按外层测试折顺序拼接，脚本用 StratifiedKFold(",
    "   n_splits=该 OOF 折数, shuffle=True, random_state=123) 复现折顺序，",
    "   并以 y_true 逐例校验后才把 age 对齐；校验失败即回退 Treat All。",
    "",
    "3. 连续 NRI（阈值 eps = 1e-9，按概率变化方向判定）",
    "   事件组  : NRI_events    = P(new > base) - P(new < base)",
    "   非事件组: NRI_nonevents = P(new < base) - P(new > base)",
    "   合计    : NRI_continuous = NRI_events + NRI_nonevents",
    "",
    "4. 分类 NRI（阈值 = 上述 Optimal_Cutoff，统一用于新旧模型）",
    "   事件组  : NRI_cat_events    = P(new >= c) - P(base >= c)",
    "   非事件组: NRI_cat_nonevents = P(base >= c) - P(new >= c)",
    "   合计    : NRI_categorical   = 两者之和（向下/向上重分类）",
    "",
    "5. IDI",
    "   IDI_events    = mean(p_new | event)     - mean(p_base | event)",
    "   IDI_nonevents = mean(p_new | non-event) - mean(p_base | non-event)",
    "   IDI = IDI_events - IDI_nonevents",
    "",
    f"6. 95% CI: 分层 bootstrap，{BOOTSTRAP_ITER} 次；事件组与非事件组各自有放回",
    "   重抽并保持原样本量；区间取第 2.5 / 97.5 百分位；",
    f"   随机数发生器 np.random.RandomState({RANDOM_STATE})（固定种子，可复现）。",
    "   CI 仅覆盖 NRI_continuous(合计) 与 IDI；分类 NRI（NRI_categorical）不报告 CI，",
    "   因为低患病率下分类 NRI 对阈值位置高度敏感、bootstrap 区间不稳定，",
    "   仅作为点估计参考（known instability of categorical NRI under low prevalence）。",
    "",
    "-" * 70,
    "二、结果表（nri_idi_results.csv）",
    "-" * 70,
    (result_df.round(6).to_string(index=False) if len(result_df) else "（无可用队列，无结果行）"),
    "",
    "-" * 70,
    "三、限制说明",
    "-" * 70,
    "NRI/IDI 依赖风险分层，跨队列比较时基线模型不同会改变数值含义，仅作敏感性分析解读。",
    "",
]

report_path = os.path.join(RESULT_DIR, "nri_idi_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"  Saved: {report_path}")

print("\n" + "=" * 70)
print("NRI / IDI analysis completed.")
print("=" * 70)
