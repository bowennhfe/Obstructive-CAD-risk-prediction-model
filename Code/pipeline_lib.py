"""
pipeline_lib.py — 折内重放共享库（防泄漏嵌套 CV 专用）
=======================================================
目的：04_Model_Selection.py 的外层每折需要在折内训练子集上重放
     「派生比值(log 空间) → log1p → RF 迭代插补(mFIM) → RobustScaler
       → VIF/相关性过滤」全链路，外层测试折仅做 transform。
     本库将 01_data_preprocessing.py 与 02_vif_analysis.py 的数值逻辑
     原样封装为「在 train 上 fit、对 train/test 分别 transform」的函数，
     保证与官方面板流水线数值行为等价，但全部统计量仅在折内 train 上估计。

与原脚本的一致性约定（勿随意改动参数）：
  - log1p 列：由折内 train 原始数值列 skewness 决定（|skew|>1 且 min>-1），
    对应 01 的 transformed_log_cols 逻辑（折内重算）。
  - 插补：mFIM = IterativeImputer(RandomForestRegressor(n_estimators=100,
    random_state=123), max_iter=20, random_state=123)，log 空间、不缩放
    （01 的 ScaledImputer use_scaling=False 路径）。
  - RobustScaler：log 空间插补后值的 median/IQR（(x-median)/IQR，IQR=0 → 0），
    仅 train 估计。
  - 分类变量：train 众数填补 + 按 train sorted unique one-hot（首位为参照，丢弃）。
  - VIF 阶段（02 复刻）：注入 noise_feature（np.random.seed(123) 后 randn）→
    迭代 Spearman 相关过滤（|rho|>0.6，保留列序靠前者）→ 迭代 VIF>10 剔除
    （保护 noise_feature；VIF inf/nan 记 999）。
  - 折 test 永不注入 noise_feature，也不参与任何 fit。
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 — IterativeImputer 需要显式启用
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 123

# 与 01_data_preprocessing.py 完全一致的列定义
NUMERIC_COLS = ['age', 'mean arterial pressure', 'pulse pressure index', 'pulse pressure',
                'total cholesterol', 'triglycerides', 'LDL-C', 'HDL-C',
                'LDL-C/HDL-C', 'total cholesterol/HDL-C', 'triglycerides/HDL-C',
                'uric acid', 'plasma fibrinogen', 'creatinine', 'urea']
CATEGORICAL_COLS = ['gender', 'hypertension', 'diabetes', 'smoking']

# 与 02_vif_analysis.py 一致的阈值
CORR_THRESHOLD = 0.6
VIF_THRESHOLD = 10.0
NOISE_FEATURE = 'noise_feature'


# =============================================================================
# 原始数据定位
# =============================================================================
def find_raw_cohort_path(base_dir):
    """定位原始（未预处理、含缺失值）Derivation cohort.csv。

    与 01_data_preprocessing.py 的 _find_cohort_file 逻辑一致：
    优先精确匹配 <base_dir>/Derivation cohort.csv（避免 Windows 上
    "Derivation cohort - 副本.csv" 之类的文件被 glob 误选），
    找不到再 glob 根目录，最后搜一层子目录。
    """
    exact = os.path.join(base_dir, 'Derivation cohort.csv')
    if os.path.exists(exact):
        return exact
    matches = glob.glob(os.path.join(base_dir, '*Derivation cohort*.csv'))
    if not matches:
        matches = glob.glob(os.path.join(base_dir, '*', '*Derivation cohort*.csv'))
    if not matches:
        raise FileNotFoundError(
            f"Cannot find raw 'Derivation cohort.csv' in {base_dir} (or one level below).")
    if len(matches) > 1:
        print(f"  WARNING: multiple raw Derivation matches: {matches}\n  Using: {matches[0]}")
    return matches[0]


# =============================================================================
# 01 链路复刻：log1p → mFIM 插补 → RobustScaler → 分类变量编码
# =============================================================================
def compute_log_cols(train_raw_numeric):
    """01 口径（折内重算）：基于 train 原始数值列 skewness 决定 log1p 列。

    |skew|>1 且 min>-1（log1p 有定义）才入选；NaN 不参与 skewness 计算。
    """
    log_cols = []
    for col in train_raw_numeric.columns:
        vals = train_raw_numeric[col].dropna()
        if len(vals) > 1:
            col_skew = skew(vals)
            if abs(col_skew) > 1.0:
                if vals.min() > -1:
                    log_cols.append(col)
                else:
                    print(f"  [log1p] {col}: skew={col_skew:.3f} 但 min={vals.min():.3f} <= -1，跳过 log 变换")
    return log_cols


def build_mfim_imputer():
    """构造与 01 选定方法 mFIM 完全一致的 RF 迭代插补器（log 空间、不缩放）。"""
    return IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        max_iter=20,
        random_state=RANDOM_STATE,
    )


def fit_apply_imputation(train_log_df, test_log_df):
    """在折 train 的 log 空间数据上拟合 mFIM，再对 train/test 分别插补。

    返回与输入同列名、同列序的 (train_imputed_df, test_imputed_df)。
    IterativeImputer 在全缺失列上会丢弃该列（keep_empty_features=False），
    此时列数不一致，直接报错（原流水线不存在该情形，出现即为折内数据异常）。
    """
    imputer = build_mfim_imputer()
    train_arr = imputer.fit_transform(train_log_df.values.astype(np.float64))
    test_arr = imputer.transform(test_log_df.values.astype(np.float64))
    if train_arr.shape[1] != train_log_df.shape[1]:
        raise ValueError(
            "折内 train 存在全缺失数值列，IterativeImputer 丢弃了列 "
            f"({train_arr.shape[1]}/{train_log_df.shape[1]})——检查折内数据质量。")
    if test_arr.shape[1] != test_log_df.shape[1]:
        raise ValueError(
            "折内 test 插补输出列数与 train 不一致 "
            f"({test_arr.shape[1]}/{train_log_df.shape[1]})。")
    train_imputed = pd.DataFrame(train_arr, columns=train_log_df.columns,
                                 index=train_log_df.index)
    test_imputed = pd.DataFrame(test_arr, columns=test_log_df.columns,
                                index=test_log_df.index)
    return train_imputed, test_imputed


def fit_robust_params(train_imputed_df):
    """01 口径：RobustScaler 参数 = train 插补后 log 空间值的 median/IQR。"""
    params = {}
    for col in train_imputed_df.columns:
        median_val = train_imputed_df[col].median()
        q1 = train_imputed_df[col].quantile(0.25)
        q3 = train_imputed_df[col].quantile(0.75)
        params[col] = {'median': median_val, 'iqr': q3 - q1}
    return params


def apply_robust_scale(df, params):
    """对 df 逐列施加 (x-median)/IQR；IQR=0 的列全置 0（与 01 一致并告警）。"""
    scaled = pd.DataFrame(index=df.index)
    for col in df.columns:
        median_val = params[col]['median']
        iqr_val = params[col]['iqr']
        if iqr_val == 0:
            print(f"  WARNING: {col} 在折内 train 的 IQR=0，缩放后全置 0")
            scaled[col] = 0.0
        else:
            scaled[col] = (df[col] - median_val) / iqr_val
    return scaled


def fit_categorical(train_cat_df):
    """01 口径：分类变量众数（train）与 sorted unique 类别集合（train）。"""
    fit = {}
    for col in train_cat_df.columns:
        non_null = train_cat_df[col].dropna()
        if len(non_null) == 0:
            raise ValueError(f"折内 train 分类列 {col} 全缺失——检查数据！")
        fit[col] = {'mode': non_null.mode().iloc[0],
                    'uniques': sorted(non_null.unique())}
    return fit


def apply_categorical(cat_df, cat_fit):
    """对分类变量做 train 众数填补 + train 类别 one-hot（首位为参照，丢弃）。

    test 中出现 train 未见的类别时，所有 dummy 均为 0（参照类编码，与 01 一致）。
    """
    out = pd.DataFrame(index=cat_df.index)
    for col, cfg in cat_fit.items():
        s = cat_df[col].fillna(cfg['mode'])
        for val in cfg['uniques'][1:]:
            out[f'{col}_{val}'] = (s == val).astype(int)
    return out


def assemble_processed(id_df, label_df, scaled_num_df, dummies_df):
    """按 01 的 combine_processed_data 顺序组装：ID, label, 数值列, dummy 列。"""
    processed = pd.concat([id_df, label_df], axis=1)
    for col in scaled_num_df.columns:
        processed[col] = scaled_num_df[col].values
    for col in dummies_df.columns:
        processed[col] = dummies_df[col].values
    return processed


# =============================================================================
# 02 链路复刻：noise 注入 → 相关性过滤 → 迭代 VIF 剔除
# =============================================================================
def calculate_vif(df):
    """02 口径逐字复刻：逐列 VIF，inf/nan 记 999.0。"""
    if df.shape[1] == 0:
        return pd.DataFrame(columns=['Feature', 'VIF'])
    vif_list = []
    for i, col in enumerate(df.columns):
        try:
            v = variance_inflation_factor(df.values, i)
            if np.isinf(v) or np.isnan(v):
                v = 999.0
        except Exception:
            v = 999.0
        vif_list.append({'Feature': col, 'VIF': round(v, 2)})
    return pd.DataFrame(vif_list).sort_values('VIF', ascending=False)


def correlation_filter(X, threshold=CORR_THRESHOLD):
    """02 口径逐字复刻：迭代剔除 |Spearman rho|>threshold 的特征对中列序靠后者。"""
    X_filtered = X.copy()
    removed = []
    while True:
        corr = X_filtered.corr(method='spearman')
        corr_arr = corr.values.copy()
        np.fill_diagonal(corr_arr, 0)
        np.tril(corr_arr)[:] = 0
        corr_abs = np.abs(corr_arr)
        max_corr_abs = corr_abs.max()
        if max_corr_abs <= threshold:
            break
        i, j = np.unravel_index(corr_abs.argmax(), corr_arr.shape)
        fa, fb = corr.index[i], corr.columns[j]
        max_corr = corr_arr[i, j]
        rem, keep = fb, fa
        removed.append({'removed': rem, 'kept': keep, 'rho': max_corr})
        X_filtered = X_filtered.drop(columns=[rem])
        print(f"  [corr] Removed: {rem} (kept: {keep}, rho={max_corr:.3f})")
    return X_filtered, removed


def vif_stage(train_feat_df):
    """02 复刻（仅 train）：注入 noise_feature → 相关性过滤 → VIF>10 迭代剔除。

    返回 (final_cols, logs, X_clean)：
      - final_cols: 清洗后保留的列名（含 noise_feature 若存活，与 02 官方面板一致）
      - X_clean:    清洗后的矩阵（已含注入的 noise_feature 列），供调用方直接组装
    """
    np.random.seed(RANDOM_STATE)  # 与 02 一致：噪声注入前重置全局种子
    X_noisy = train_feat_df.copy()
    X_noisy[NOISE_FEATURE] = np.random.randn(len(X_noisy))

    X_filtered, removed_corr = correlation_filter(X_noisy, CORR_THRESHOLD)
    print(f"  [corr] removed {len(removed_corr)} features, {X_filtered.shape[1]} remaining")

    X_clean = X_filtered.copy()
    vif_log = []
    while True:
        vif_current = calculate_vif(X_clean)
        high_vif = vif_current[vif_current['VIF'] > VIF_THRESHOLD]
        if high_vif.empty:
            break
        high_vif_sorted = high_vif.sort_values('VIF', ascending=False)
        feature_to_remove = None
        for _, row in high_vif_sorted.iterrows():
            if row['Feature'] != NOISE_FEATURE:  # 与 02 一致：保护 noise_feature
                feature_to_remove = row['Feature']
                break
        if feature_to_remove is None:
            print("  [vif] 仅 noise_feature 的 VIF>10，停止剔除（noise 受保护）")
            break
        vif_value = high_vif_sorted[high_vif_sorted['Feature'] == feature_to_remove]['VIF'].values[0]
        print(f"  [vif] Removing: {feature_to_remove} (VIF={vif_value:.1f})")
        vif_log.append({'removed': feature_to_remove, 'VIF': vif_value})
        X_clean = X_clean.drop(columns=[feature_to_remove])

    return X_clean.columns.tolist(), {
        'corr_removed': removed_corr,
        'vif_removed': vif_log,
    }, X_clean


# =============================================================================
# 折内重放总入口
# =============================================================================
def replay_preprocess(train_raw, test_raw, fold_tag=""):
    """在折内重放 01→02 全链路：train 上 fit，train/test 分别 transform。

    参数：train_raw / test_raw 均为含 ID、label 与原始（未预处理、可含缺失）
    特征列的 DataFrame（来自原始 Derivation cohort.csv 按外层折切分）。

    返回 (train_df, test_df, meta)：
      - train_df: ID, label + VIF 清洗后特征（含 noise_feature 若存活）
      - test_df:  ID, label + 同一特征集（不含 noise_feature）
      - meta: 折内各阶段日志（log 列表、剔除记录等），供落盘审计
    """
    prefix = f"[fold {fold_tag}] " if fold_tag else ""

    missing_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS
                    if c not in train_raw.columns or c not in test_raw.columns]
    if missing_cols:
        raise ValueError(f"{prefix}原始数据缺少列: {missing_cols}")

    tr_id, tr_y = train_raw[['ID']].copy(), train_raw[['label']].copy()
    te_id, te_y = test_raw[['ID']].copy(), test_raw[['label']].copy()

    # --- 1) log1p（折内重算 skewness；NaN 保留） ---
    tr_num_raw = train_raw[NUMERIC_COLS].copy()
    te_num_raw = test_raw[NUMERIC_COLS].copy()
    log_cols = compute_log_cols(tr_num_raw)
    print(f"{prefix}log1p columns (fold-train skewness): {log_cols}")
    tr_log = tr_num_raw.copy()
    te_log = te_num_raw.copy()
    for c in log_cols:
        tr_log[c] = np.log1p(tr_log[c])
        te_log[c] = np.log1p(te_log[c])

    # --- 2) mFIM 插补（仅 train 拟合） ---
    tr_imp, te_imp = fit_apply_imputation(tr_log, te_log)
    print(f"{prefix}mFIM imputation done (fit on train only)")

    # --- 3) RobustScaler（log 空间，仅 train 估计 median/IQR） ---
    robust_params = fit_robust_params(tr_imp)
    tr_scaled = apply_robust_scale(tr_imp, robust_params)
    te_scaled = apply_robust_scale(te_imp, robust_params)

    # --- 4) 分类变量：train 众数填补 + train 类别 one-hot ---
    cat_fit = fit_categorical(train_raw[CATEGORICAL_COLS])
    tr_dum = apply_categorical(train_raw[CATEGORICAL_COLS], cat_fit)
    te_dum = apply_categorical(test_raw[CATEGORICAL_COLS], cat_fit)

    # --- 5) 组装（列序 = 01：ID, label, 数值, dummy） ---
    train_proc = assemble_processed(tr_id, tr_y, tr_scaled, tr_dum)
    test_proc = assemble_processed(te_id, te_y, te_scaled, te_dum)

    # --- 6) VIF 阶段（02 复刻，仅 train） ---
    feat_matrix = train_proc.drop(columns=['ID', 'label'])
    final_cols, logs, X_clean = vif_stage(feat_matrix)
    print(f"{prefix}VIF stage done: {len(final_cols)} features kept (incl. noise if alive)")

    # X_clean 含折内注入的 noise_feature 负对照（与 02 官方面板结构一致），
    # 直接在其上拼 ID/label 作为写盘矩阵，供 R 桥做 4 方法共识选择。
    train_final = pd.concat([train_proc[['ID', 'label']], X_clean], axis=1)

    # --- 7) 外层测试折建模特征：剔除 noise_feature（test 不含该列） ---
    model_cols = [c for c in final_cols if c != NOISE_FEATURE]
    missing_in_test = [c for c in model_cols if c not in test_proc.columns]
    if missing_in_test:
        raise ValueError(f"{prefix}test 折缺少特征列: {missing_in_test}")
    test_final = test_proc[['ID', 'label'] + model_cols].copy()

    meta = {
        # 折内重算（基于折内 train 的 skewness），不是复用 01 的全局 transformed_log_cols 列表
        'transformed_log_cols': log_cols,
        'log_cols': log_cols,
        'robust_params': robust_params,
        'cat_fit': cat_fit,
        'final_cols_with_noise': final_cols,
        'model_cols': model_cols,
        **logs,
    }
    return train_final, test_final, meta


def r_make_names(name):
    """模拟 R read.csv(check.names=TRUE) 的列名改写（与 05 的 _r_make_names 一致）。"""
    import re
    return re.sub(r"[^A-Za-z0-9_.]", ".", str(name))
