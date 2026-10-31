
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde, skew
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor

import joblib
import os
import glob
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# =============================================================================
# Path Configuration — auto-detect relative to Code folder
# =============================================================================
# BASE_DIR = Code folder's parent (e.g., e:\new\)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory: BASE_DIR/data_preprocessing/
RESULT_DIR = os.path.join(BASE_DIR, 'data_preprocessing')
os.makedirs(RESULT_DIR, exist_ok=True)

# Auto-search for cohort CSV files in BASE_DIR
# 本项目自身会向下列目录写出同名 *_cohort*.csv（01/02/03/04/05/07 的输出），
# 二次运行时若把它们当原始输入，会把已处理数据再加工一遍，故必须排除。
EXCLUDE_DIRS = {"data_preprocessing", "vif_analysis", "feature_selection",
                "Model Selection", "Model Development", "Subgroup Analysis",
                "Blinded Validation", "Clinical_Plots_R", "Nested CV Work"}

def _find_cohort_file(pattern, required=True):
    """Search for a cohort CSV file matching the pattern in BASE_DIR.

    只接受项目根目录下的文件；根目录没有结果时才向一层子目录回退，
    且跳过本项目的输出目录（见 EXCLUDE_DIRS）。
    If required=False, returns None when file not found (optional cohort)."""
    base_abs = os.path.abspath(BASE_DIR)
    candidates = [p for p in glob.glob(os.path.join(BASE_DIR, pattern))
                  if os.path.dirname(os.path.abspath(p)) == base_abs]
    # Also search one level deeper (e.g., subfolders), excluding own outputs
    if not candidates:
        for p in glob.glob(os.path.join(BASE_DIR, '*', pattern)):
            parent = os.path.basename(os.path.dirname(os.path.abspath(p)))
            if parent not in EXCLUDE_DIRS:
                candidates.append(p)
    if not candidates:
        if required:
            raise FileNotFoundError(
                f"Cannot find file matching '{pattern}' in {BASE_DIR}. "
                f"Please ensure the cohort CSV file exists."
            )
        return None
    candidates.sort()   # glob 顺序不保证稳定，排序使多匹配时选择可复现
    if len(candidates) > 1:
        print(f"  WARNING: Multiple matches for '{pattern}': {candidates}")
        print(f"  Using: {candidates[0]}")
    return candidates[0]

print(f"BASE_DIR:  {BASE_DIR}")
print(f"RESULT_DIR: {RESULT_DIR}")

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 9

def save_figure(fig, base_name):
    pdf_path = os.path.join(RESULT_DIR, f'{base_name}.pdf')
    png_path = os.path.join(RESULT_DIR, f'{base_name}.png')
    tiff_path = os.path.join(RESULT_DIR, f'{base_name}.tif')
    
    fig.savefig(pdf_path, dpi=600, bbox_inches='tight', transparent=False)
    fig.savefig(png_path, dpi=600, bbox_inches='tight', transparent=False)
    fig.savefig(tiff_path, dpi=600, bbox_inches='tight', transparent=False, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)

# =============================================================================
# NaN-aware scaling helpers — compute mean/std from non-missing values only,
# keeping NaN as NaN during transform. Used instead of StandardScaler which
# cannot handle NaN in log-space data before imputation.
# =============================================================================
def _get_scale_params(train_df, cols):
    """Compute per-column mean and std from non-missing values only."""
    means = {}
    stds = {}
    for col in cols:
        vals = train_df[col].dropna()
        means[col] = vals.mean()
        s = vals.std()
        stds[col] = s if s > 0 else 1.0  # Avoid division by zero for constant columns
    return means, stds

def _apply_scale(df, means, stds):
    """Apply (X - mean) / std, keeping NaN as NaN."""
    df_scaled = df.copy()
    for col in means:
        df_scaled[col] = (df_scaled[col] - means[col]) / stds[col]
    return df_scaled

def _inverse_scale(df, means, stds):
    """Apply X * std + mean to reverse scaling."""
    df_inv = df.copy()
    for col in means:
        df_inv[col] = df_inv[col] * stds[col] + means[col]
    return df_inv

# ScaledImputer: wrapper that applies NaN-aware scaling→impute→inverse scaling
# for scale-sensitive methods (KNNIM, LRIM, BRIM); MIM/MedIM/mFIM go through directly.
# Defined at module level so it can be pickled by joblib.
class ScaledImputer:
    """Wrapper that does NaN-aware scaling → impute → inverse scaling.
    Mean/std are computed from non-missing values only; NaN stays NaN during scaling,
    then KNNImputer/IterativeImputer handles the NaN imputation.
    
    Parameters
    ----------
    imputer : sklearn imputer
    use_scaling : bool
    feature_names : list, optional
        Column names to use for the DataFrame. If provided, means_ and stds_
        use these names as keys, consistent with external _get_scale_params logic.
        If None, integer column indices are used (for backward compatibility).
    """
    def __init__(self, imputer, use_scaling=True, feature_names=None):
        self.imputer = imputer
        self.use_scaling = use_scaling
        self.feature_names = feature_names or []
        self.means_ = None
        self.stds_ = None

    def fit_transform(self, X):
        if self.use_scaling:
            cols = self.feature_names if self.feature_names else list(range(X.shape[1]))
            df = pd.DataFrame(X, columns=cols)
            self.means_, self.stds_ = _get_scale_params(df, cols)
            X_scaled = _apply_scale(df, self.means_, self.stds_).values
            X_imputed = self.imputer.fit_transform(X_scaled)
            df_imputed = pd.DataFrame(X_imputed, columns=cols)
            X_inv = _inverse_scale(df_imputed, self.means_, self.stds_).values
            return X_inv
        else:
            return self.imputer.fit_transform(X)

    def transform(self, X):
        if self.use_scaling:
            cols = self.feature_names if self.feature_names else list(range(X.shape[1]))
            df = pd.DataFrame(X, columns=cols)
            X_scaled = _apply_scale(df, self.means_, self.stds_).values
            X_imputed = self.imputer.transform(X_scaled)
            df_imputed = pd.DataFrame(X_imputed, columns=cols)
            X_inv = _inverse_scale(df_imputed, self.means_, self.stds_).values
            return X_inv
        else:
            return self.imputer.transform(X)

def main():
    print("=" * 80)
    print("Data Preprocessing Pipeline")
    print("=" * 80)
    
    # Global random seed for reproducibility
    RANDOM_STATE = 123
    np.random.seed(RANDOM_STATE)
    
    print("\n[Step 1/9] Loading data...")
    train_path = _find_cohort_file('*Derivation cohort*.csv')
    temporal_path = _find_cohort_file('*Temporal external validation cohort*.csv', required=False)
    geo_path = _find_cohort_file('*Geographical external validation cohort*.csv', required=False)
    
    has_temporal = temporal_path is not None
    has_geo = geo_path is not None
    
    print(f"  Derivation: {train_path}")
    if has_temporal:
        print(f"  Temporal:   {temporal_path}")
    else:
        print(f"  Temporal:   (not found — skipping)")
    if has_geo:
        print(f"  Geo:        {geo_path}")
    else:
        print(f"  Geo:        (not found — skipping)")
    
    train_data = pd.read_csv(train_path)
    # Robust column name cleaning: strip whitespace, normalize case-insensitive aliases
    train_data.columns = [col.strip() for col in train_data.columns]
    # Create a case-insensitive lookup map for common column name variants
    _col_map = {col.lower(): col for col in train_data.columns}
    print(f"  Columns loaded: {list(train_data.columns)}")
    
    if has_temporal:
        temporal_data = pd.read_csv(temporal_path)
        temporal_data.columns = [col.strip() for col in temporal_data.columns]
        # Align temporal columns to training column names (case-insensitive)
        _temporal_rename = {}
        for tc in temporal_data.columns:
            if tc.lower() in _col_map and tc != _col_map[tc.lower()]:
                _temporal_rename[tc] = _col_map[tc.lower()]
        if _temporal_rename:
            print(f"  Temporal column name alignment: {_temporal_rename}")
            temporal_data = temporal_data.rename(columns=_temporal_rename)
    if has_geo:
        geo_data = pd.read_csv(geo_path)
        geo_data.columns = [col.strip() for col in geo_data.columns]
        _geo_rename = {}
        for gc in geo_data.columns:
            if gc.lower() in _col_map and gc != _col_map[gc.lower()]:
                _geo_rename[gc] = _col_map[gc.lower()]
        if _geo_rename:
            print(f"  Geographical column name alignment: {_geo_rename}")
            geo_data = geo_data.rename(columns=_geo_rename)
    
    print(f"Derivation cohort (Training): {train_data.shape}")
    if has_temporal:
        print(f"Temporal external validation cohort (Test 1): {temporal_data.shape}")
    if has_geo:
        print(f"Geographical external validation cohort (Test 2): {geo_data.shape}")
    print(f"Columns: {list(train_data.columns)}")
    
    train_data.describe().to_csv(os.path.join(RESULT_DIR, 'train_data_stats.csv'))
    if has_temporal:
        temporal_data.describe().to_csv(os.path.join(RESULT_DIR, 'temporal_data_stats.csv'))
    if has_geo:
        geo_data.describe().to_csv(os.path.join(RESULT_DIR, 'geo_data_stats.csv'))
    
    id_cols = ['ID']
    target_col = ['label']
    numeric_cols = ['age', 'mean arterial pressure', 'pulse pressure index', 'pulse pressure',
                   'total cholesterol', 'triglycerides', 'LDL-C', 'HDL-C',
                   'LDL-C/HDL-C', 'total cholesterol/HDL-C', 'triglycerides/HDL-C',
                   'uric acid', 'plasma fibrinogen', 'creatinine', 'urea']
    categorical_cols = ['gender', 'hypertension', 'diabetes', 'smoking']
    
    train_id = train_data[id_cols].copy()
    train_label = train_data[target_col].copy()
    train_preprocess = train_data.drop(columns=id_cols + target_col).copy()

    # 输入校验：label 必须为 0/1 二值，ID 必须唯一
    if not set(train_data['label'].unique()) <= {0, 1}:
        raise ValueError(f"Derivation 队列 label 必须为 0/1 二值，实际取值: {list(train_data['label'].unique())}")
    if not train_data['ID'].is_unique:
        raise ValueError("Derivation 队列 ID 存在重复值，ID 必须唯一")

    if has_temporal:
        temporal_id = temporal_data[id_cols].copy()
        temporal_label = temporal_data[target_col].copy()
        temporal_preprocess = temporal_data.drop(columns=id_cols + target_col).copy()
        # 输入校验：label 必须为 0/1 二值，ID 必须唯一
        if not set(temporal_data['label'].unique()) <= {0, 1}:
            raise ValueError(f"Temporal 队列 label 必须为 0/1 二值，实际取值: {list(temporal_data['label'].unique())}")
        if not temporal_data['ID'].is_unique:
            raise ValueError("Temporal 队列 ID 存在重复值，ID 必须唯一")
    if has_geo:
        geo_id = geo_data[id_cols].copy()
        geo_label = geo_data[target_col].copy()
        geo_preprocess = geo_data.drop(columns=id_cols + target_col).copy()
        # 输入校验：label 必须为 0/1 二值，ID 必须唯一
        if not set(geo_data['label'].unique()) <= {0, 1}:
            raise ValueError(f"Geographical 队列 label 必须为 0/1 二值，实际取值: {list(geo_data['label'].unique())}")
        if not geo_data['ID'].is_unique:
            raise ValueError("Geographical 队列 ID 存在重复值，ID 必须唯一")
    
    print("\nVerifying columns:")
    for col in numeric_cols + categorical_cols:
        train_exists = col in train_preprocess.columns
        temporal_exists = col in temporal_preprocess.columns if has_temporal else True
        geo_exists = col in geo_preprocess.columns if has_geo else True
        status = '✓' if (train_exists and temporal_exists and geo_exists) else '✗'
        print(f"  {col}: {status}")
    
    print(f"\nID columns (excluded): {id_cols}")
    print(f"Target column (excluded): {target_col}")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    print("\n[Step 2/9] Analyzing missing values...")
    
    def analyze_missing(data, dataset_name):
        missing_stats = []
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_percent = (missing_count / len(data)) * 100
            missing_stats.append({
                'Variable': col,
                'Missing_Count': missing_count,
                'Missing_Percent': missing_percent
            })
        return pd.DataFrame(missing_stats).sort_values('Missing_Count', ascending=False)
    
    train_missing_df = analyze_missing(train_preprocess, 'train')
    train_missing_df.to_csv(os.path.join(RESULT_DIR, 'train_missing_value_analysis.csv'), index=False)
    
    if has_temporal:
        temporal_missing_df = analyze_missing(temporal_preprocess, 'temporal')
        temporal_missing_df.to_csv(os.path.join(RESULT_DIR, 'temporal_missing_value_analysis.csv'), index=False)
    if has_geo:
        geo_missing_df = analyze_missing(geo_preprocess, 'geo')
        geo_missing_df.to_csv(os.path.join(RESULT_DIR, 'geo_missing_value_analysis.csv'), index=False)
    
    # Build dynamic subplot grid
    # 色盲友好调色板（项目统一）：Derivation #4C72B0 / Temporal #DD8452 / Geographical #55A868
    missing_cohorts = [("Derivation Cohort", train_missing_df, '#4C72B0')]
    if has_temporal:
        missing_cohorts.append(("Temporal External Validation Cohort", temporal_missing_df, '#DD8452'))
    if has_geo:
        missing_cohorts.append(("Geographical External Validation Cohort", geo_missing_df, '#55A868'))
    
    n_missing_plots = len(missing_cohorts)
    fig, axes = plt.subplots(1, n_missing_plots, figsize=(28 // 3 * n_missing_plots, 8))
    if n_missing_plots == 1:
        axes = [axes]
    
    max_missing = max(df['Missing_Percent'].max() for _, df, _ in missing_cohorts)
    y_limit = max_missing + 0.5
    
    for idx, (title, missing_df, color) in enumerate(missing_cohorts):
        x_pos = np.arange(len(missing_df))
        bars = axes[idx].bar(x_pos, missing_df['Missing_Percent'], color=color, alpha=0.8)
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(missing_df['Variable'], rotation=45, ha='right')
        axes[idx].set_xlabel('Variables')
        axes[idx].set_ylabel('Proportion of Missing Values (%)')
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_ylim(0, y_limit)
        for i, v in enumerate(missing_df['Missing_Percent']):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, 'missing_values_plot')
    
    print("\nTraining set missing values:")
    print(train_missing_df.to_string(index=False))
    if has_temporal:
        print("\nTemporal validation set missing values:")
        print(temporal_missing_df.to_string(index=False))
    if has_geo:
        print("\nGeographical validation set missing values:")
        print(geo_missing_df.to_string(index=False))
    
    # =========================================================================
    # Pre-compute transformed_log_cols BEFORE Step 3 (needed by CV evaluation)
    # Compute skewness on raw training numeric data (drop NaN for skewness calc)
    # Columns with |skew| > 1 will be log1p-transformed; NaN stays NaN.
    # =========================================================================
    print("\n[Pre-compute] Determining columns for log(1+X) transformation (based on raw training data)...")
    train_numeric_raw = train_preprocess[numeric_cols].copy()
    transformed_log_cols = []
    for col in numeric_cols:
        col_data = train_numeric_raw[col].dropna()
        if len(col_data) > 1:
            col_skew = skew(col_data)
            if abs(col_skew) > 1.0:
                # Check if log1p is valid for this column
                min_val = col_data.min()
                if min_val <= -1:
                    print(f"  ⚠️ {col}: skewness={col_skew:.3f} but min={min_val:.3f} <= -1, "
                          f"log1p undefined — skipping log transform for this column.")
                else:
                    transformed_log_cols.append(col)
                    print(f"  {col}: skewness = {col_skew:.3f} → will apply log(1+X)")
            else:
                print(f"  {col}: skewness = {col_skew:.3f} → kept as-is")
        else:
            print(f"  {col}: insufficient data for skewness → kept as-is")
    print(f"  Columns identified for log(1+X): {transformed_log_cols}")
    
    print("\n[Step 3/9] Global imputation method selection on ALL numeric columns with missing...")
    
    # Identify numeric columns with any missing values in the training set
    numeric_cols_with_missing = [
        col for col in numeric_cols
        if train_preprocess[col].isnull().sum() > 0
    ]
    if not numeric_cols_with_missing:
        print("  No numeric columns have missing values — skipping imputation method selection.")
    else:
        missing_rates = {col: train_preprocess[col].isnull().mean() for col in numeric_cols_with_missing}
        sorted_cols = sorted(missing_rates, key=missing_rates.get, reverse=True)
        eval_cols = sorted_cols  # 评估所有有缺失的数值列（不再限制前5个）
        print(f"  Numeric columns with missing: {numeric_cols_with_missing}")
        print(f"  Evaluating on top {len(eval_cols)} columns (by missing rate): {eval_cols}")
    
    # --- Imputation method functions (aligned with new main flow: Log→[NaN-aware Scale]→Impute→[InvScale]→InvLog) ---
    # All validate functions must:
    #   1. Apply log1p to transformed_log_cols (NaN kept as NaN)
    #   2. [For KNNIM/LRIM/BRIM] NaN-aware scaling fit on train (non-missing), transform both
    #   3. Impute in the scaled space (KNNImputer/IterativeImputer handles NaN)
    #   4. [For KNNIM/LRIM/BRIM] Inverse scaling back to log space
    #   5. Inverse log (expm1) back to original clinical scale
    #   6. Return target column values in original scale for NRMSE calculation
    
    def _apply_log_transform(df):
        """Apply log1p to transformed_log_cols, keeping NaN as NaN."""
        df_log = df.copy()
        for c in transformed_log_cols:
            if c in df_log.columns:
                df_log[c] = np.log1p(df_log[c])
        return df_log
    
    def _inverse_log_transform(df):
        """Apply expm1 to transformed_log_cols to revert to original scale."""
        df_orig = df.copy()
        for c in transformed_log_cols:
            if c in df_orig.columns:
                df_orig[c] = np.expm1(df_orig[c])
        return df_orig
    
    def validate_mim(train_data, val_data, col):
        # Step 1: Log transform
        train_log = _apply_log_transform(train_data)
        val_log = _apply_log_transform(val_data)
        # Step 2: Impute in log space (simple mean, no scaling needed)
        imputer = SimpleImputer(strategy='mean')
        val_imputed_log = imputer.fit(train_log[[col]]).transform(val_log[[col]])
        val_imputed_log_df = pd.DataFrame(val_imputed_log, columns=[col], index=val_data.index)
        # Step 3: Inverse log to original scale
        val_imputed = _inverse_log_transform(val_imputed_log_df)
        return val_imputed[col].values
    
    def validate_medim(train_data, val_data, col):
        # Step 1: Log transform
        train_log = _apply_log_transform(train_data)
        val_log = _apply_log_transform(val_data)
        # Step 2: Impute in log space (simple median, no scaling needed)
        imputer = SimpleImputer(strategy='median')
        val_imputed_log = imputer.fit(train_log[[col]]).transform(val_log[[col]])
        val_imputed_log_df = pd.DataFrame(val_imputed_log, columns=[col], index=val_data.index)
        # Step 3: Inverse log to original scale
        val_imputed = _inverse_log_transform(val_imputed_log_df)
        return val_imputed[col].values
    
    def validate_knnim(train_data, val_data, col, n_neighbors=None):
        if n_neighbors is None:
            # Dynamic neighbors: sqrt(sample_size), capped at 1/3 of training fold size
            n_neighbors = max(5, min(int(np.sqrt(len(train_data))), len(train_data) // 3))
        # Step 1: Log transform
        train_log = _apply_log_transform(train_data)
        val_log = _apply_log_transform(val_data)
        # Step 2: NaN-aware scaling on log-space data (fit on train non-missing, NaN stays NaN)
        train_numeric = train_log[numeric_cols].select_dtypes(include=[np.number])
        val_numeric = val_log[numeric_cols].select_dtypes(include=[np.number])
        means, stds = _get_scale_params(train_numeric, numeric_cols)
        train_scaled = _apply_scale(train_numeric, means, stds)
        val_scaled = _apply_scale(val_numeric, means, stds)
        # Step 3: Impute in scaled log space (KNNImputer handles NaN)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        val_imputed_scaled = imputer.fit(train_scaled).transform(val_scaled)
        # Step 4: Inverse scaling back to log space
        val_imputed_log_df = _inverse_scale(
            pd.DataFrame(val_imputed_scaled, columns=numeric_cols), means, stds
        )
        val_imputed_log_df.index = val_data.index
        # Step 5: Inverse log to original clinical scale
        val_imputed = _inverse_log_transform(val_imputed_log_df)
        return val_imputed[col].values
    
    def validate_mfim(train_data, val_data, col):
        # Step 1: Log transform (RF is scale-invariant, no scaling needed)
        train_log = _apply_log_transform(train_data)
        val_log = _apply_log_transform(val_data)
        # Step 2: Impute in log space
        # 迭代式回归插补（单值）：未设 sample_posterior=True，仅产出单套插补数据，非真正多重插补（MI）
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            max_iter=20,
            random_state=RANDOM_STATE
        )
        train_numeric = train_log[numeric_cols].select_dtypes(include=[np.number])
        val_numeric = val_log[numeric_cols].select_dtypes(include=[np.number])
        val_imputed_log_array = imputer.fit(train_numeric).transform(val_numeric)
        val_imputed_log_df = pd.DataFrame(val_imputed_log_array, columns=numeric_cols, index=val_data.index)
        # Step 3: Inverse log to original clinical scale
        val_imputed = _inverse_log_transform(val_imputed_log_df)
        return val_imputed[col].values
    
    def validate_lrim(train_data, val_data, col):
        # Step 1: Log transform
        train_log = _apply_log_transform(train_data)
        val_log = _apply_log_transform(val_data)
        # Step 2: NaN-aware scaling on log-space data (LinearRegression is scale-sensitive)
        train_numeric = train_log[numeric_cols].select_dtypes(include=[np.number])
        val_numeric = val_log[numeric_cols].select_dtypes(include=[np.number])
        means, stds = _get_scale_params(train_numeric, numeric_cols)
        train_scaled = _apply_scale(train_numeric, means, stds)
        val_scaled = _apply_scale(val_numeric, means, stds)
        # Step 3: Impute in scaled log space (IterativeImputer handles NaN)
        # 迭代式回归插补（单值）：未设 sample_posterior=True，仅产出单套插补数据，非真正多重插补（MI）
        imputer = IterativeImputer(
            estimator=LinearRegression(),
            max_iter=20,
            random_state=RANDOM_STATE
        )
        val_imputed_scaled = imputer.fit(train_scaled).transform(val_scaled)
        # Step 4: Inverse scaling back to log space
        val_imputed_log_df = _inverse_scale(
            pd.DataFrame(val_imputed_scaled, columns=numeric_cols), means, stds
        )
        val_imputed_log_df.index = val_data.index
        # Step 5: Inverse log to original clinical scale
        val_imputed = _inverse_log_transform(val_imputed_log_df)
        return val_imputed[col].values
    
    def validate_brim(train_data, val_data, col):
        # Step 1: Log transform
        train_log = _apply_log_transform(train_data)
        val_log = _apply_log_transform(val_data)
        # Step 2: NaN-aware scaling on log-space data (BayesianRidge is scale-sensitive)
        train_numeric = train_log[numeric_cols].select_dtypes(include=[np.number])
        val_numeric = val_log[numeric_cols].select_dtypes(include=[np.number])
        means, stds = _get_scale_params(train_numeric, numeric_cols)
        train_scaled = _apply_scale(train_numeric, means, stds)
        val_scaled = _apply_scale(val_numeric, means, stds)
        # Step 3: Impute in scaled log space (IterativeImputer handles NaN)
        # 迭代式回归插补（单值）：未设 sample_posterior=True，仅产出单套插补数据，非真正多重插补（MI）
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            random_state=RANDOM_STATE
        )
        val_imputed_scaled = imputer.fit(train_scaled).transform(val_scaled)
        # Step 4: Inverse scaling back to log space
        val_imputed_log_df = _inverse_scale(
            pd.DataFrame(val_imputed_scaled, columns=numeric_cols), means, stds
        )
        val_imputed_log_df.index = val_data.index
        # Step 5: Inverse log to original clinical scale
        val_imputed = _inverse_log_transform(val_imputed_log_df)
        return val_imputed[col].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    methods = {
        'MIM': validate_mim,
        'MedIM': validate_medim,
        'KNNIM': validate_knnim,
        'mFIM': validate_mfim,
        'LRIM': validate_lrim,
        'BRIM': validate_brim
    }
    
    if numeric_cols_with_missing:
        # ================================================================
        # Per-fold NRMSE collection (correct 1-SE definition: SE across folds)
        # Each fold produces a mean NRMSE averaged across all eval columns,
        # giving 5 fold-level means per method. SE = std_folds / √5.
        # ================================================================
        # method_fold_scores[method][fold_idx] = list of per-column NRMSEs in that fold
        method_fold_scores = {m: [[] for _ in range(5)] for m in methods}
        
        for col_to_test in eval_cols:
            print(f"\n  Evaluating column: {col_to_test}")
            complete_data = train_preprocess.dropna(subset=[col_to_test])
            if len(complete_data) < 10:
                print(f"    Too few complete samples ({len(complete_data)}) — skipping")
                continue
            
            # Per-column per-method NRMSE accumulator (for printing only)
            col_nrmse = {m: [] for m in methods}
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(complete_data)):
                train_fold = complete_data.iloc[train_idx].copy()
                val_fold = complete_data.iloc[val_idx].copy()
                true_values = val_fold[col_to_test].copy()
                val_fold[col_to_test] = np.nan
                
                for method_name, method_func in methods.items():
                    try:
                        imputed_values = method_func(train_fold, val_fold, col_to_test)
                        denom = np.std(true_values, ddof=1)
                        if denom == 0:
                            nrmse = 0.0
                        else:
                            nrmse = np.sqrt(np.mean((imputed_values - true_values) ** 2)) / denom
                        method_fold_scores[method_name][fold_idx].append(nrmse)
                        col_nrmse[method_name].append(nrmse)
                    except Exception as e:
                        print(f"      {method_name} error on col={col_to_test}, fold={fold_idx}: {e}")
                        col_nrmse[method_name].append(np.nan)
            
            # Print column summary (mean across folds for this column)
            for method_name in methods:
                valid = [x for x in col_nrmse[method_name] if not np.isnan(x)]
                if valid:
                    print(f"    {method_name}: NRMSE = {np.mean(valid):.4f}")
                else:
                    print(f"    {method_name}: all folds failed")
        
        # ================================================================
        # Compute fold-level means: each method gets 5 fold-level averages
        # ================================================================
        nrmse_matrix = {}
        for method_name in methods:
            fold_means = []
            for fold_idx in range(5):
                scores = method_fold_scores[method_name][fold_idx]
                if scores:
                    fold_means.append(np.mean(scores))  # mean across columns in this fold
            if fold_means:
                nrmse_matrix[method_name] = {
                    'mean': np.mean(fold_means),           # grand mean across folds
                    'std':  np.std(fold_means, ddof=0),    # std of fold-level means
                    'se':   np.std(fold_means, ddof=0) / np.sqrt(len(fold_means))  # SE = SD/√K
                }
        
        # ================================================================
        # 核心决策：改良版 1-标准误（1-SE）准则（优先迭代式回归插补）
        # 方法学说明：mFIM/BRIM/LRIM 基于 sklearn IterativeImputer（max_iter=20，
        # 未设 sample_posterior=True），仅生成单套插补数据，属于迭代式回归插补
        # （单值）/ iterative regression imputation (single)，并非生成多套插补
        # 再经 Rubin's rules 合并的真正多重插补（MI）。此处"优先"仅因为回归类
        # 方法利用了变量间相关性，而非基于 MI 的理论依据。
        # ================================================================
        iter_methods = ['mFIM', 'BRIM', 'LRIM']        # 迭代式回归插补（单值，优先）
        simple_methods = ['MIM', 'MedIM', 'KNNIM']     # 简单单值插补（备选）
        
        # Build stats DataFrame (includes SE column for transparency)
        all_methods_stats = pd.DataFrame([
            {'Method': m, 'Mean': v['mean'], 'Std': v['std'], 'SE': v['se']}
            for m, v in nrmse_matrix.items()
        ])
        
        if not all_methods_stats.empty:
            # 1. 找出全局平均 NRMSE 最小的方法（作为基准）
            best_overall_row = all_methods_stats.loc[all_methods_stats['Mean'].idxmin()]
            best_global_mean = best_overall_row['Mean']
            best_global_se = best_overall_row['SE']       # 使用折叠标准误（已含 /√K）
            
            # 2. 计算 1-SE 容忍阈值（直接使用 SE，无需再次除以 √n）
            threshold = best_global_mean + best_global_se
            print(f"\n  全局最优 NRMSE: {best_global_mean:.4f} (SD={best_overall_row['Std']:.4f}, SE={best_global_se:.4f})")
            print(f"  1-SE 容忍阈值 (Mean + SE): {threshold:.4f}")
            
            # 3. 【核心改良】在阈值范围内，优先筛选迭代式回归插补（单值）方法
            candidate_iter = all_methods_stats[
                (all_methods_stats['Method'].isin(iter_methods)) &
                (all_methods_stats['Mean'] <= threshold)
            ]

            if not candidate_iter.empty:
                selected = candidate_iter.loc[candidate_iter['Mean'].idxmin()]
                best_method_name = selected['Method']
                best_nrmse = selected['Mean']
                print(f"  >>> 选择策略：迭代式回归插补（单值）优先 -> {best_method_name} (Mean NRMSE={best_nrmse:.4f})，其在全局最优的 1-SE 范围内。")
            else:
                best_method_name = best_overall_row['Method']
                best_nrmse = best_global_mean
                print(f"  >>> 选择策略：降级为简单单值插补 -> {best_method_name} (Mean NRMSE={best_nrmse:.4f})，因无迭代式回归插补达标。")
        else:
            best_method_name = 'MIM'
            best_nrmse = 0.0
            print("  >>> 评估失败，保底使用 MIM")
        
        # Build global_df for chart and CSV export
        if not all_methods_stats.empty:
            global_df = all_methods_stats[['Method', 'Mean', 'Std', 'SE']].copy()
            global_df = global_df.rename(columns={'Mean': 'Avg_NRMSE'})
            global_df = global_df.sort_values('Avg_NRMSE', ascending=True)
        else:
            global_df = pd.DataFrame(columns=['Method', 'Avg_NRMSE', 'Std', 'SE'])
        
        print(f"\n  Global imputation comparison (fold-level NRMSE across {len(eval_cols)} columns):")
        print(global_df.to_string(index=False))
        print(f"\n  >>> Selected method: {best_method_name} (avg NRMSE = {best_nrmse:.4f})")
        
        # Save simplified comparison CSV (Method + fold-level stats + selection flag)
        comparison_df = global_df.copy()
        comparison_df['Selected'] = comparison_df['Method'].apply(
            lambda m: '★ SELECTED' if m == best_method_name else ''
        )
        comparison_df.to_csv(os.path.join(RESULT_DIR, 'imputation_method_comparison.csv'), index=False)
    else:
        # No missing values — default to MIM (mean imputation) as fallback
        best_method_name = 'MIM'
        best_nrmse = 0.0
        print("  No missing numeric columns. Defaulting to MIM for safety.")
        global_df = pd.DataFrame(columns=['Method', 'Avg_NRMSE', 'Std', 'SE'])
        pd.DataFrame([{'Method': 'MIM', 'Avg_NRMSE': 0.0, 'Selected': '★ SELECTED'}]).to_csv(
            os.path.join(RESULT_DIR, 'imputation_method_comparison.csv'), index=False)
    
    # --- Imputation comparison bar chart ---
    if numeric_cols_with_missing and not global_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = global_df.dropna(subset=['Avg_NRMSE'])
        colors = ['#E91E63' if m == best_method_name else 'steelblue' for m in plot_df['Method']]
        bars = ax.bar(plot_df['Method'], plot_df['Avg_NRMSE'], color=colors, alpha=0.8)
        ax.set_xlabel('Imputation Method')
        ax.set_ylabel('Average NRMSE (fold-level, across all numeric columns with missing)')
        for i, v in enumerate(plot_df['Avg_NRMSE']):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
        plt.tight_layout()
        save_figure(fig, 'imputation_comparison')
    
    # =========================================================================
    # Step 4: Apply log(1+X) transformation to raw numeric data (NaN stays NaN)
    # =========================================================================
    print(f"\n[Step 4/9] Applying log(1+X) transformation to raw numeric data...")
    print(f"  Columns to transform: {transformed_log_cols}")
    
    train_numeric = train_preprocess[numeric_cols].copy()
    train_categorical = train_preprocess[categorical_cols].copy()
    
    if has_temporal:
        temporal_numeric = temporal_preprocess[numeric_cols].copy()
        temporal_categorical = temporal_preprocess[categorical_cols].copy()
    if has_geo:
        geo_numeric = geo_preprocess[numeric_cols].copy()
        geo_categorical = geo_preprocess[categorical_cols].copy()
    
    # Apply log1p to columns identified by skewness; NaN stays NaN
    train_numeric_log_df = train_numeric.copy()
    for col in transformed_log_cols:
        train_numeric_log_df[col] = np.log1p(train_numeric_log_df[col])
    
    if has_temporal:
        temporal_numeric_log_df = temporal_numeric.copy()
        for col in transformed_log_cols:
            temporal_numeric_log_df[col] = np.log1p(temporal_numeric_log_df[col])
    if has_geo:
        geo_numeric_log_df = geo_numeric.copy()
        for col in transformed_log_cols:
            geo_numeric_log_df[col] = np.log1p(geo_numeric_log_df[col])
    
    print(f"  Log transformation complete. Data now in log space (NaN preserved).")
    
    # =========================================================================
    # Step 5: Imputation in log space
    # =========================================================================
    print(f"\n[Step 5/9] Imputing datasets in log space with {best_method_name} (fitted ONLY on training data)...")
    
    # Build imputer dynamically based on best_method_name
    def _build_imputer(method_name):
        if method_name == 'MIM':
            return ScaledImputer(SimpleImputer(strategy='mean'), use_scaling=False, feature_names=numeric_cols)
        elif method_name == 'MedIM':
            return ScaledImputer(SimpleImputer(strategy='median'), use_scaling=False, feature_names=numeric_cols)
        elif method_name == 'KNNIM':
            # Dynamic neighbors: sqrt(sample_size), capped at 1/3 of training size
            n_neighbors = max(5, min(int(np.sqrt(len(train_numeric))), len(train_numeric) // 3))
            return ScaledImputer(KNNImputer(n_neighbors=n_neighbors), use_scaling=True, feature_names=numeric_cols)
        elif method_name == 'mFIM':
            return ScaledImputer(IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
                max_iter=20,
                random_state=RANDOM_STATE
            ), use_scaling=False, feature_names=numeric_cols)  # RF is scale-invariant, no scaling needed
        elif method_name == 'LRIM':
            return ScaledImputer(IterativeImputer(
                estimator=LinearRegression(),
                max_iter=20,
                random_state=RANDOM_STATE
            ), use_scaling=True, feature_names=numeric_cols)
        elif method_name == 'BRIM':
            return ScaledImputer(IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=20,
                random_state=RANDOM_STATE
            ), use_scaling=True, feature_names=numeric_cols)
        else:
            print(f"  WARNING: Unknown method '{method_name}', falling back to mFIM")
            return ScaledImputer(IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
                max_iter=20,
                random_state=RANDOM_STATE
            ), use_scaling=False, feature_names=numeric_cols)
    
    imputer = _build_imputer(best_method_name)
    print(f"  Using imputer: {type(imputer.imputer).__name__}" + 
          (f" (estimator={type(imputer.imputer.estimator).__name__})" if hasattr(imputer.imputer, 'estimator') else "") +
          (f" + NaN-aware scaling" if imputer.use_scaling else ""))
    
    # Impute in log space (ScaledImputer handles NaN-aware scaling round-trip internally)
    train_numeric_imputed = imputer.fit_transform(train_numeric_log_df.values)
    train_numeric_imputed_df = pd.DataFrame(train_numeric_imputed, columns=numeric_cols, index=train_preprocess.index)
    
    if has_temporal:
        print("  Transforming TEMPORAL validation data with fitted imputer...")
        temporal_numeric_imputed = imputer.transform(temporal_numeric_log_df.values)
        temporal_numeric_imputed_df = pd.DataFrame(temporal_numeric_imputed, columns=numeric_cols, index=temporal_preprocess.index)
    if has_geo:
        print("  Transforming GEOGRAPHICAL validation data with fitted imputer...")
        geo_numeric_imputed = imputer.transform(geo_numeric_log_df.values)
        geo_numeric_imputed_df = pd.DataFrame(geo_numeric_imputed, columns=numeric_cols, index=geo_preprocess.index)
    
    print("  Imputation complete (in log space)!")
    
    # Create inverse-log-transformed versions for sensitivity analysis (original clinical scale)
    train_numeric_imputed_original_df = train_numeric_imputed_df.copy()
    for col in transformed_log_cols:
        train_numeric_imputed_original_df[col] = np.expm1(train_numeric_imputed_df[col])
    
    if has_temporal:
        temporal_numeric_imputed_original_df = temporal_numeric_imputed_df.copy()
        for col in transformed_log_cols:
            temporal_numeric_imputed_original_df[col] = np.expm1(temporal_numeric_imputed_df[col])
    if has_geo:
        geo_numeric_imputed_original_df = geo_numeric_imputed_df.copy()
        for col in transformed_log_cols:
            geo_numeric_imputed_original_df[col] = np.expm1(geo_numeric_imputed_df[col])
    
    # Save intermediate state: after imputation (inverse-log-transformed to original scale)
    # NOTE: This file is in original clinical scale for debugging imputation quality
    pd.concat([train_id, train_label, train_numeric_imputed_original_df], axis=1).to_csv(
        os.path.join(RESULT_DIR, 'train_after_imputation.csv'), index=False)
    
    print("\nGenerating sensitivity analysis plots...")
    
    missing_vars = [col for col in numeric_cols if train_numeric[col].isnull().sum() > 0]
    print(f"Variables with missing values in training data: {missing_vars}")
    
    color_before = '#2196F3'
    color_after = '#E91E63'
    
    # =========================================================================
    # Step A: Pre-classify variables by whether external validation set has missing
    # =========================================================================
    vars_with_external_missing = []
    vars_without_external_missing = []
    
    for var_name in missing_vars:
        has_ext_missing = False
        if has_temporal and temporal_numeric[var_name].isnull().sum() > 0:
            has_ext_missing = True
        if has_geo and geo_numeric[var_name].isnull().sum() > 0:
            has_ext_missing = True
        
        if has_ext_missing:
            vars_with_external_missing.append(var_name)
        else:
            vars_without_external_missing.append(var_name)
    
    print(f"  Variables WITH external missing: {vars_with_external_missing}")
    print(f"  Variables WITHOUT external missing: {vars_without_external_missing}")
    
    # =========================================================================
    # Step B: Variables WITH external missing → individual 3×2 figures
    # =========================================================================
    for var_name in vars_with_external_missing:
        print(f"  Processing {var_name} (has external missing)...")
        
        # --- Training data: Before = non-missing only; After = all values (includes imputed) ---
        mask = ~train_numeric[var_name].isnull()
        before_data = train_numeric.loc[mask, var_name].values
        after_data = train_numeric_imputed_original_df[var_name].values  # ALL values after imputation (original scale)
        
        # --- External validation data: Before = non-missing only; After = all values ---
        ext_before_list = []
        ext_after_all_list = []
        
        if has_temporal:
            temporal_mask = ~temporal_numeric[var_name].isnull()
            ext_before_list.append(temporal_numeric.loc[temporal_mask, var_name].values)
            ext_after_all_list.append(temporal_numeric_imputed_original_df[var_name].values)  # ALL values, original scale
        
        if has_geo:
            geo_mask = ~geo_numeric[var_name].isnull()
            ext_before_list.append(geo_numeric.loc[geo_mask, var_name].values)
            ext_after_all_list.append(geo_numeric_imputed_original_df[var_name].values)      # ALL values, original scale
        
        ext_before_data = np.concatenate(ext_before_list) if ext_before_list else np.array([])
        ext_after_data = np.concatenate(ext_after_all_list) if ext_after_all_list else np.array([])
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        
        # --- (a) Derivation KDE ---
        ax = axes[0, 0]
        if len(before_data) > 1 and np.std(before_data) > 1e-10:
            kde_before = gaussian_kde(before_data)
            x_before_kde = np.linspace(before_data.min(), before_data.max(), 200)
            y_before_kde = kde_before(x_before_kde)
            ax.fill_between(x_before_kde, y_before_kde, alpha=0.6, color=color_before)
            ax.plot(x_before_kde, y_before_kde, color=color_before, linewidth=2, label='Before Imputation')
        if len(after_data) > 1 and np.std(after_data) > 1e-10:
            kde_after = gaussian_kde(after_data)
            x_after_kde = np.linspace(after_data.min(), after_data.max(), 200)
            y_after_kde = kde_after(x_after_kde)
            ax.fill_between(x_after_kde, y_after_kde, alpha=0.6, color=color_after)
            ax.plot(x_after_kde, y_after_kde, color=color_after, linewidth=2, label='After Imputation')
        ax.set_ylabel('density')
        ax.set_title('(a) Derivation Cohort')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # --- (b) External KDE ---
        ax = axes[0, 1]
        if len(ext_before_data) > 0:
            if len(ext_before_data) > 1 and np.std(ext_before_data) > 1e-10:
                kde_ext_before = gaussian_kde(ext_before_data)
                x_extb_kde = np.linspace(ext_before_data.min(), ext_before_data.max(), 200)
                y_extb_kde = kde_ext_before(x_extb_kde)
                ax.fill_between(x_extb_kde, y_extb_kde, alpha=0.6, color=color_before)
                ax.plot(x_extb_kde, y_extb_kde, color=color_before, linewidth=2, label='Before Imputation')
            if len(ext_after_data) > 1 and np.std(ext_after_data) > 1e-10:
                kde_ext_after = gaussian_kde(ext_after_data)
                x_exta_kde = np.linspace(ext_after_data.min(), ext_after_data.max(), 200)
                y_exta_kde = kde_ext_after(x_exta_kde)
                ax.fill_between(x_exta_kde, y_exta_kde, alpha=0.6, color=color_after)
                ax.plot(x_exta_kde, y_exta_kde, color=color_after, linewidth=2, label='After Imputation')
            ax.set_ylabel('density')
            ax.legend(loc='upper right', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No missing values\nin test data',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes,
                    style='italic', color='gray')
        ax.set_title('(b) External Validation Cohort')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # --- (c) Derivation Boxplot ---
        ax = axes[1, 0]
        box_data = [before_data, after_data]
        bp = ax.boxplot(box_data, tick_labels=['Before', 'After'], patch_artist=True, widths=0.6, showfliers=True)
        bp['boxes'][0].set_facecolor(color_before)
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(color_after)
        bp['boxes'][1].set_alpha(0.6)
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1)
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        ax.set_ylabel(var_name)
        ax.set_title('(c) Derivation Cohort')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # --- (d) External Boxplot ---
        ax = axes[1, 1]
        if len(ext_before_data) > 0:
            box_data_test = [ext_before_data, ext_after_data]
            bp_test = ax.boxplot(box_data_test, tick_labels=['Before', 'After'], patch_artist=True, widths=0.6, showfliers=True)
            bp_test['boxes'][0].set_facecolor(color_before)
            bp_test['boxes'][0].set_alpha(0.6)
            bp_test['boxes'][1].set_facecolor(color_after)
            bp_test['boxes'][1].set_alpha(0.6)
            for whisker in bp_test['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(1)
            for cap in bp_test['caps']:
                cap.set_color('black')
                cap.set_linewidth(1)
            for median in bp_test['medians']:
                median.set_color('black')
                median.set_linewidth(2)
            ax.set_ylabel(var_name)
        else:
            ax.text(0.5, 0.5, 'No missing values\nin test data',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes,
                    style='italic', color='gray')
        ax.set_title('(d) External Validation Cohort')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # --- (e) Derivation Scatter ---
        ax = axes[2, 0]
        jitter_amount = 0.12
        x_before = np.full(len(before_data), 0.2) + np.random.uniform(-jitter_amount, jitter_amount, len(before_data))
        x_after = np.full(len(after_data), 0.8) + np.random.uniform(-jitter_amount, jitter_amount, len(after_data))
        ax.scatter(x_before, before_data, alpha=0.5, color=color_before, s=20)
        ax.scatter(x_after, after_data, alpha=0.5, color=color_after, s=20)
        ax.set_xticks([0.2, 0.8])
        ax.set_xticklabels(['Before\nImputation', 'After\nImputation'])
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylabel(var_name)
        ax.set_title('(e) Derivation Cohort')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # --- (f) External Scatter ---
        ax = axes[2, 1]
        if len(ext_before_data) > 0:
            test_x_before = np.full(len(ext_before_data), 0.2) + np.random.uniform(-jitter_amount, jitter_amount, len(ext_before_data))
            test_x_after = np.full(len(ext_after_data), 0.8) + np.random.uniform(-jitter_amount, jitter_amount, len(ext_after_data))
            ax.scatter(test_x_before, ext_before_data, alpha=0.5, color=color_before, s=20)
            ax.scatter(test_x_after, ext_after_data, alpha=0.5, color=color_after, s=20)
            ax.set_xticks([0.2, 0.8])
            ax.set_xticklabels(['Before\nImputation', 'After\nImputation'])
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylabel(var_name)
        else:
            ax.text(0.5, 0.5, 'No missing values\nin test data',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes,
                    style='italic', color='gray')
        ax.set_title('(f) External Validation Cohort')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.suptitle(f'Sensitivity Analysis of Imputation Results - {var_name}',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_figure(fig, f'sensitivity_analysis_{var_name.replace(" ", "_")}')
    
    # =========================================================================
    # Step C: Variables WITHOUT external missing → combined 3×N figure
    # =========================================================================
    if vars_without_external_missing:
        print(f"\n  Generating combined figure for {len(vars_without_external_missing)} variables without external missing...")
        N = len(vars_without_external_missing)
        fig, axes = plt.subplots(3, N, figsize=(4 * N, 12))
        if N == 1:
            axes = axes.reshape(3, 1)
        
        for col_idx, var_name in enumerate(vars_without_external_missing):
            print(f"    {var_name} (column {col_idx+1}/{N})")
            
            # Training data only: Before = non-missing; After = all values (includes imputed)
            mask = ~train_numeric[var_name].isnull()
            before_data = train_numeric.loc[mask, var_name].values
            after_data = train_numeric_imputed_original_df[var_name].values  # ALL values after imputation (original scale)
            
            # --- Row 1: KDE ---
            ax = axes[0, col_idx]
            if len(before_data) > 1 and np.std(before_data) > 1e-10:
                kde_before = gaussian_kde(before_data)
                x_before_kde = np.linspace(before_data.min(), before_data.max(), 200)
                y_before_kde = kde_before(x_before_kde)
                ax.fill_between(x_before_kde, y_before_kde, alpha=0.6, color=color_before)
                ax.plot(x_before_kde, y_before_kde, color=color_before, linewidth=2, label='Before Imputation')
            if len(after_data) > 1 and np.std(after_data) > 1e-10:
                kde_after = gaussian_kde(after_data)
                x_after_kde = np.linspace(after_data.min(), after_data.max(), 200)
                y_after_kde = kde_after(x_after_kde)
                ax.fill_between(x_after_kde, y_after_kde, alpha=0.6, color=color_after)
                ax.plot(x_after_kde, y_after_kde, color=color_after, linewidth=2, label='After Imputation')
            ax.set_title(var_name, fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel('(a) KDE\nDensity')
            else:
                ax.set_ylabel('')
            
            # --- Row 2: Boxplot ---
            ax = axes[1, col_idx]
            box_data = [before_data, after_data]
            bp = ax.boxplot(box_data, tick_labels=['Before', 'After'], patch_artist=True, widths=0.6, showfliers=True)
            bp['boxes'][0].set_facecolor(color_before)
            bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor(color_after)
            bp['boxes'][1].set_alpha(0.6)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(1)
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(1)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel('(b) Boxplot\nValue')
            else:
                ax.set_ylabel('')
            
            # --- Row 3: Scatter ---
            ax = axes[2, col_idx]
            jitter_amount = 0.12
            x_before = np.full(len(before_data), 0.2) + np.random.uniform(-jitter_amount, jitter_amount, len(before_data))
            x_after = np.full(len(after_data), 0.8) + np.random.uniform(-jitter_amount, jitter_amount, len(after_data))
            ax.scatter(x_before, before_data, alpha=0.5, color=color_before, s=20)
            ax.scatter(x_after, after_data, alpha=0.5, color=color_after, s=20)
            ax.set_xticks([0.2, 0.8])
            ax.set_xticklabels(['Before\nImputation', 'After\nImputation'])
            ax.set_xlim(-0.1, 1.1)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel('(c) Scatter\nValue')
            else:
                ax.set_ylabel('')
        
        fig.suptitle('Sensitivity Analysis — Variables Without External Missing',
                     fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_figure(fig, 'sensitivity_analysis_no_external_missing')
    
    print("\n[Step 6/9] RobustScaler scaling in log space (median/IQR, fitted ONLY on training data)...")
    
    # Data is already in log space from Step 4/5; RobustScaler operates on log-space values
    scaler_robust_params = {}
    for col in numeric_cols:
        median_val = train_numeric_imputed_df[col].median()
        q1 = train_numeric_imputed_df[col].quantile(0.25)
        q3 = train_numeric_imputed_df[col].quantile(0.75)
        iqr_val = q3 - q1
        scaler_robust_params[col] = {'median': median_val, 'iqr': iqr_val}
    
    train_numeric_scaled_df = train_numeric_imputed_df.copy()
    if has_temporal:
        temporal_numeric_scaled_df = temporal_numeric_imputed_df.copy()
    if has_geo:
        geo_numeric_scaled_df = geo_numeric_imputed_df.copy()
    
    for col in numeric_cols:
        median_val = scaler_robust_params[col]['median']
        iqr_val = scaler_robust_params[col]['iqr']
        if iqr_val == 0:
            # No variability — set all scaled values to 0
            print(f"  WARNING: {col} has IQR=0, setting scaled values to 0")
            train_numeric_scaled_df[f'{col}_scaled'] = 0
            if has_temporal:
                temporal_numeric_scaled_df[f'{col}_scaled'] = 0
            if has_geo:
                geo_numeric_scaled_df[f'{col}_scaled'] = 0
        else:
            train_numeric_scaled_df[f'{col}_scaled'] = (train_numeric_scaled_df[col] - median_val) / iqr_val
            if has_temporal:
                temporal_numeric_scaled_df[f'{col}_scaled'] = (temporal_numeric_scaled_df[col] - median_val) / iqr_val
            if has_geo:
                geo_numeric_scaled_df[f'{col}_scaled'] = (geo_numeric_scaled_df[col] - median_val) / iqr_val
    
    # --- Scaling comparison plot (log-space before vs. log-space after RobustScaler) ---
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            ax = axes[i]
            
            # "Before scaling" = log-space data (from train_numeric_imputed_df)
            orig_data = train_numeric_imputed_df[col].dropna()
            if len(orig_data) > 1 and np.std(orig_data) > 1e-10:
                kde_orig = gaussian_kde(orig_data)
                x_orig = np.linspace(orig_data.min(), orig_data.max(), 100)
                y_orig = kde_orig(x_orig)
                ax.plot(x_orig, y_orig, color='steelblue', linewidth=2, label='Before scaling')
                ax.fill_between(x_orig, y_orig, alpha=0.3, color='steelblue')
            elif len(orig_data) > 0:
                # Constant data: skip KDE, draw a single vertical line at the constant value
                const_val = orig_data.iloc[0]
                ax.axvline(const_val, color='steelblue', linewidth=2, linestyle='-', label='Before scaling')
            
            scaled_data = train_numeric_scaled_df[f'{col}_scaled'].dropna()
            if len(scaled_data) > 1 and np.std(scaled_data) > 1e-10:
                kde_scaled = gaussian_kde(scaled_data)
                x_scaled = np.linspace(scaled_data.min(), scaled_data.max(), 100)
                y_scaled = kde_scaled(x_scaled)
                
                median_val = scaler_robust_params[col]['median']
                iqr_val = scaler_robust_params[col]['iqr']
                if iqr_val > 0:
                    x_scaled_orig = x_scaled * iqr_val + median_val
                else:
                    x_scaled_orig = x_scaled
                
                ax.plot(x_scaled_orig, y_scaled, color='coral', linewidth=2, linestyle='--', label='After scaling')
                ax.fill_between(x_scaled_orig, y_scaled, alpha=0.3, color='coral')
            elif len(scaled_data) > 0:
                # Constant data: skip KDE
                const_val = scaled_data.iloc[0]
                median_val = scaler_robust_params[col]['median']
                iqr_val = scaler_robust_params[col]['iqr']
                if iqr_val > 0:
                    const_val_orig = const_val * iqr_val + median_val
                else:
                    const_val_orig = const_val
                ax.axvline(const_val_orig, color='coral', linewidth=2, linestyle='--', label='After scaling')
            
            ax.set_xlabel('')
            ax.set_ylabel('Density')
            ax.set_title(f'({chr(97+i)}) {col}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, 'scaling_comparison')
    
    scaling_stats = []
    for col in numeric_cols:
        scaling_stats.append({
            'Variable': col,
            'Median': scaler_robust_params[col]['median'],
            'IQR': scaler_robust_params[col]['iqr']
        })
    pd.DataFrame(scaling_stats).to_csv(os.path.join(RESULT_DIR, 'scaling_comparison.csv'), index=False)
    
    # Save intermediate state: after RobustScaler in log space
    # NOTE: named "train_after_scaling_log" to reflect that data is RobustScaled in log-space
    pd.concat([train_id, train_label, train_numeric_scaled_df], axis=1).to_csv(
        os.path.join(RESULT_DIR, 'train_after_scaling_log.csv'), index=False)
    
    print("\n[Step 7/9] Handling categorical missing values (mode imputation from training data)...")
    
    cat_mode = {}
    for col in categorical_cols:
        # Compute mode from training set (before any encoding)
        # Check for entirely-missing column FIRST, before mode() to avoid IndexError
        if len(train_categorical[col].dropna()) == 0:
            raise ValueError(f"Column {col} is entirely missing in training set — check raw data!")
        cat_mode[col] = train_categorical[col].dropna().mode().iloc[0]
        
        # Fill training set
        missing_train = train_categorical[col].isnull().sum()
        if missing_train > 0:
            train_categorical[col] = train_categorical[col].fillna(cat_mode[col])
            print(f"  {col}: filled {missing_train} missing in Training with mode={cat_mode[col]}")
        
        # Fill temporal (if exists)
        if has_temporal:
            missing_temp = temporal_categorical[col].isnull().sum()
            if missing_temp > 0:
                temporal_categorical[col] = temporal_categorical[col].fillna(cat_mode[col])
                print(f"  {col}: filled {missing_temp} missing in Temporal with mode={cat_mode[col]}")
        
        # Fill geo (if exists)
        if has_geo:
            missing_geo = geo_categorical[col].isnull().sum()
            if missing_geo > 0:
                geo_categorical[col] = geo_categorical[col].fillna(cat_mode[col])
                print(f"  {col}: filled {missing_geo} missing in Geographical with mode={cat_mode[col]}")
    
    print(f"  Category modes (from training): {cat_mode}")
    
    print("\n[Step 8/9] Encoding categorical variables (fitted ONLY on training data)...")
    
    encoders = {}
    train_categorical_encoded_df = train_categorical.copy()
    if has_temporal:
        temporal_categorical_encoded_df = temporal_categorical.copy()
    if has_geo:
        geo_categorical_encoded_df = geo_categorical.copy()
    
    all_unseen_warnings = []  # Collect across all categorical columns
    
    for col in categorical_cols:
        print(f"\nProcessing {col}:")
        
        unique_vals = sorted(train_categorical[col].unique())
        print(f"  Training unique values: {unique_vals}")
        
        # P0: Check for unseen categories in validation sets before encoding
        col_unseen_warnings = []
        if has_temporal:
            temporal_unique_vals = sorted(temporal_categorical[col].unique())
            print(f"  Temporal unique values: {temporal_unique_vals}")
            temporal_unseen = set(temporal_unique_vals) - set(unique_vals)
            if temporal_unseen:
                col_unseen_warnings.append(f"Temporal: {sorted(temporal_unseen)}")
        if has_geo:
            geo_unique_vals = sorted(geo_categorical[col].unique())
            print(f"  Geographical unique values: {geo_unique_vals}")
            geo_unseen = set(geo_unique_vals) - set(unique_vals)
            if geo_unseen:
                col_unseen_warnings.append(f"Geographical: {sorted(geo_unseen)}")
        
        if col_unseen_warnings:
            warning_msg = f"  ⚠ WARNING: Unseen categories in validation set(s) for '{col}': {'; '.join(col_unseen_warnings)}. "
            warning_msg += f"These will be encoded as all-zeros (reference category)."
            print(warning_msg)
            all_unseen_warnings.append(warning_msg.strip())
        
        print(f"  One-hot encoding, generating {len(unique_vals)-1} new features")
        
        for val in unique_vals[1:]:
            train_categorical_encoded_df[f'{col}_{val}'] = (train_categorical_encoded_df[col] == val).astype(int)
            if has_temporal:
                temporal_categorical_encoded_df[f'{col}_{val}'] = (temporal_categorical_encoded_df[col] == val).astype(int)
            if has_geo:
                geo_categorical_encoded_df[f'{col}_{val}'] = (geo_categorical_encoded_df[col] == val).astype(int)
        
        encoders[col] = {'type': 'onehot', 'values': unique_vals}
    
    # Pre-compute final feature names (numeric + one-hot categorical dummies)
    final_features = [col for col in numeric_cols]
    for col in categorical_cols:
        for val in encoders[col]['values'][1:]:
            final_features.append(f'{col}_{val}')
    
    # Save intermediate state: after categorical encoding (with ID and label)
    pd.concat([train_id, train_label, train_categorical_encoded_df], axis=1).to_csv(
        os.path.join(RESULT_DIR, 'train_categorical_encoded.csv'), index=False)
    
    # Save unseen_warnings to preprocessing_log.txt
    if all_unseen_warnings:
        log_path = os.path.join(RESULT_DIR, 'preprocessing_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("Preprocessing Warnings (Unseen Categories in Validation Sets):\n")
            f.write("=" * 60 + "\n")
            for warning in all_unseen_warnings:
                f.write(f"- {warning}\n")
        print(f"\n  Unseen category warnings saved to: {log_path}")
    else:
        print(f"\n  No unseen categories detected — all validation set categories present in training.")
    
    print("\n[Step 9/9] Saving preprocessing objects and processed data...")
    
    preprocessing_objects = {
        'imputer': imputer,
        'best_method_name': best_method_name,
        'best_nrmse': best_nrmse,
        'scaler_robust_params': scaler_robust_params,
        'transformed_log_cols': transformed_log_cols,
        'cat_mode': cat_mode,
        'encoders': encoders,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'id_cols': id_cols,
        'target_col': target_col,
        'final_feature_names': final_features
    }
    joblib.dump(preprocessing_objects, os.path.join(RESULT_DIR, 'preprocessing_objects.joblib'))
    
    # Generate feature description file for downstream modeling reference
    feature_desc_lines = []
    feature_desc_lines.append("Feature Description for Processed Data")
    feature_desc_lines.append("=" * 60)
    feature_desc_lines.append("")
    feature_desc_lines.append(f"NUMERICAL FEATURES ({len(numeric_cols)} features):")
    feature_desc_lines.append("  Transformation chain: Raw → Log(1+X) if |skew|>1 → RobustScaler(median/IQR)")
    feature_desc_lines.append(f"  Log-transformed columns: {transformed_log_cols if transformed_log_cols else 'None'}")
    feature_desc_lines.append("  All numerical column names in processed data store the scaled values.")
    feature_desc_lines.append("")
    for col in numeric_cols:
        median_val = scaler_robust_params[col]['median']
        iqr_val = scaler_robust_params[col]['iqr']
        log_note = f"  - {col}: scaled via (X - {median_val:.4f}) / {iqr_val:.4f}" if iqr_val > 0 else f"  - {col}: IQR=0, all scaled to 0"
        if col in transformed_log_cols:
            log_note += " [log1p applied before scaling]"
        else:
            log_note += " [no log transform]"
        feature_desc_lines.append(log_note)
    feature_desc_lines.append("")
    feature_desc_lines.append("CATEGORICAL FEATURES (one-hot encoded from 4 categorical columns):")
    feature_desc_lines.append("  Encoding: Manual one-hot, reference category encoded as all-zeros across dummy columns.")
    feature_desc_lines.append(f"  Category modes used for missing value filling: {cat_mode}")
    feature_desc_lines.append("")
    for col in categorical_cols:
        vals = encoders[col]['values']
        reference = vals[0]
        dummies = vals[1:]
        feature_desc_lines.append(f"  - {col}: reference={reference}, dummies={[f'{col}_{v}' for v in dummies]}")
    feature_desc_lines.append("")
    feature_desc_lines.append("TARGET & ID:")
    feature_desc_lines.append("  - ID: Patient identifier (not used in modeling)")
    feature_desc_lines.append("  - label: Binary outcome variable (0/1)")
    feature_desc_lines.append("")
    feature_desc_lines.append("TOTAL FEATURES (excl. ID & label):")
    feature_desc_lines.append(f"  {len(final_features)} features: {len(numeric_cols)} numerical + {len(final_features) - len(numeric_cols)} one-hot dummies")
    feature_desc_lines.append(f"  Full list: {final_features}")
    
    feature_desc_path = os.path.join(RESULT_DIR, 'feature_description.txt')
    with open(feature_desc_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(feature_desc_lines))
    print(f"  Feature description saved to: {feature_desc_path}")
    
    def combine_processed_data(id_data, label_data, numeric_scaled_df, categorical_encoded_df):
        processed = pd.concat([id_data, label_data], axis=1)
        
        # Add numeric columns (log-transformed + RobustScaler scaled)
        for col in numeric_cols:
            processed[col] = numeric_scaled_df[f'{col}_scaled'].values
        
        # Add categorical one-hot dummy variables (delete original column, add all non-reference dummies)
        for col in categorical_cols:
            # Delete original categorical column if present
            if col in processed.columns:
                processed = processed.drop(columns=[col])
            # Add all non-reference category dummy columns
            unique_vals = encoders[col]['values']
            for val in unique_vals[1:]:
                new_col_name = f'{col}_{val}'
                if new_col_name in categorical_encoded_df.columns:
                    processed[new_col_name] = categorical_encoded_df[new_col_name].values
                else:
                    print(f"  WARNING: {new_col_name} not found in encoded dataframe — filling with 0")
                    processed[new_col_name] = 0
        
        return processed
    
    train_processed = combine_processed_data(train_id, train_label, train_numeric_scaled_df, train_categorical_encoded_df)
    if has_temporal:
        temporal_processed = combine_processed_data(temporal_id, temporal_label, temporal_numeric_scaled_df, temporal_categorical_encoded_df)
    if has_geo:
        geo_processed = combine_processed_data(geo_id, geo_label, geo_numeric_scaled_df, geo_categorical_encoded_df)
    
    print("\nProcessing complete!")
    print(f"Training data shape: {train_processed.shape}")
    if has_temporal:
        print(f"Temporal validation data shape: {temporal_processed.shape}")
    if has_geo:
        print(f"Geographical validation data shape: {geo_processed.shape}")
    
    print("\nFinal features:")
    for feat in final_features:
        print(f"  - {feat}")
    
    train_processed.to_csv(os.path.join(RESULT_DIR, 'Derivation cohort.csv'), index=False)
    if has_temporal:
        temporal_processed.to_csv(os.path.join(RESULT_DIR, 'Temporal external validation cohort.csv'), index=False)
    if has_geo:
        geo_processed.to_csv(os.path.join(RESULT_DIR, 'Geographical external validation cohort.csv'), index=False)
    
    print("\n" + "=" * 80)
    print(f"All results saved to: {RESULT_DIR}")
    print(f"\nImputation method used: {best_method_name} (avg NRMSE = {best_nrmse:.4f})")
    print(f"Log-transformed columns: {transformed_log_cols if transformed_log_cols else 'None'}")
    print(f"Scaler: RobustScaler (median / IQR)")
    print(f"Category modes: {cat_mode}")
    print("\nData leakage prevention summary:")
    print("  ✓ Imputer fitted on training data only")
    print("  ✓ Log-transformation list derived from training data only")
    print("  ✓ RobustScaler parameters (median/IQR) computed on training data only")
    print("  ✓ Category modes computed on training data only")
    print("  ✓ Encoders fitted on training data only")
    print("  ✓ Test data transformed using training data statistics")
    print("=" * 80)

if __name__ == "__main__":
    main()
