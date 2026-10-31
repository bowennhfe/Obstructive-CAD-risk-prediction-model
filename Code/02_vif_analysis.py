"""
VIF Analysis Pipeline
- Based on reference code from vif_analysis.py
- Performs correlation filtering (rho < 0.6) and VIF calculation on Derivation cohort
- Applies the same filtering logic to Temporal and Geographical external validation cohorts
- Avoids data leakage by fitting all parameters on training data only
"""

from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import glob
import warnings
import joblib
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 — required for IterativeImputer unpickling
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer  # noqa: F401 — required for ScaledImputer unpickling
from sklearn.exceptions import ConvergenceWarning
# 仅屏蔽已知无害的警告类别，避免全量屏蔽掩盖真实问题（参照 01/04/05 脚本做法）
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# Minimal ScaledImputer class for unpickling preprocessing_objects.joblib
# (Matches the class defined in 01_data_preprocessing.py)
# =============================================================================
class ScaledImputer:
    """Wrapper for NaN-aware scaling -> impute -> inverse scaling.
    Minimal definition for joblib unpickling compatibility.

    !! 警告：本桩类禁止用于实际数据转换，仅供 unpickling 读取特征名 !!
    其行为与 01_data_preprocessing.py 中的真实 ScaledImputer（含缩放）不一致，
    调用 transform/fit_transform 会得到错误结果，因此直接 raise 以防止误用。
    （joblib unpickling 通过 __new__ + __setstate__ 恢复实例，不会调用这些方法。）
    """
    def __init__(self, imputer, use_scaling=True, feature_names=None):
        self.imputer = imputer
        self.use_scaling = use_scaling
        self.feature_names = feature_names or []
        self.means_ = None
        self.stds_ = None
    def fit_transform(self, X):
        raise NotImplementedError(
            "ScaledImputer 桩类仅供 unpickling 读取特征名，禁止用于实际数据转换")
    def transform(self, X):
        raise NotImplementedError(
            "ScaledImputer 桩类仅供 unpickling 读取特征名，禁止用于实际数据转换")

# =============================================================================
# Path Configuration — auto-detect relative to Code folder
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: auto-search from data_preprocessing folder
DATA_PREPROCESSING_DIR = os.path.join(BASE_DIR, 'data_preprocessing')

def _find_cohort_file(pattern, required=True):
    """Search for a cohort CSV file matching the pattern in data_preprocessing folder.
    If required=False, returns None when file not found (optional cohort)."""
    matches = glob.glob(os.path.join(DATA_PREPROCESSING_DIR, pattern))
    if not matches:
        if required:
            raise FileNotFoundError(
                f"Cannot find file matching '{pattern}' in {DATA_PREPROCESSING_DIR}. "
                f"Please ensure 01_data_preprocessing.py has been run first."
            )
        return None
    if len(matches) > 1:
        print(f"  WARNING: Multiple matches for '{pattern}': {matches}")
        print(f"  Using: {matches[0]}")
    return matches[0]

# Set font and DPI parameters
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Arial'
rcParams['axes.titlesize'] = 9
rcParams['axes.labelsize'] = 8
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8

# Paths - auto-detected from data_preprocessing
TRAIN_PATH = _find_cohort_file('*Derivation cohort*.csv')
TEMPORAL_PATH = _find_cohort_file('*Temporal external validation cohort*.csv', required=False)
GEO_PATH = _find_cohort_file('*Geographical external validation cohort*.csv', required=False)

HAS_TEMPORAL = TEMPORAL_PATH is not None
HAS_GEO = GEO_PATH is not None

# Output: auto-create vif_analysis folder at same level as Code
OUTPUT_ROOT = os.path.join(BASE_DIR, 'vif_analysis')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print(f"BASE_DIR:   {BASE_DIR}")
print(f"Input dir:  {DATA_PREPROCESSING_DIR}")
print(f"Output dir: {OUTPUT_ROOT}")

# Correlation threshold
CORR_THRESHOLD = 0.6

# =============================================================================
# Load final_feature_names from preprocessing_objects.joblib
# =============================================================================
PREPROCESSING_OBJECTS_PATH = os.path.join(DATA_PREPROCESSING_DIR, 'preprocessing_objects.joblib')
_preproc_objects = joblib.load(PREPROCESSING_OBJECTS_PATH)
_raw_feature_names = _preproc_objects['final_feature_names']

# Strip only the _scaled suffix from feature names for cleaner output
def _strip_suffix(name):
    """Remove the _scaled suffix from column/feature names."""
    if name.endswith('_scaled'):
        return name[:-7]
    return name

FINAL_FEATURE_NAMES = [_strip_suffix(n) for n in _raw_feature_names]
print(f"Loaded {len(FINAL_FEATURE_NAMES)} final feature names from preprocessing_objects.joblib")
print(f"  Raw:      {_raw_feature_names}")
print(f"  Cleaned:  {FINAL_FEATURE_NAMES}")

# =============================================================================
# Utility Functions
# =============================================================================

def save_figure(fig, base_path_without_ext, formats=['pdf', 'png', 'tif']):
    """Save figure in multiple formats"""
    for fmt in formats:
        fpath = f"{base_path_without_ext}.{fmt}"
        fig.savefig(fpath, format=fmt, dpi=600, bbox_inches='tight', transparent=False)
        print(f"  Saved: {fpath}")

def calculate_vif(df):
    """Calculate VIF for each feature, with detailed warning logs for failures."""
    if df.shape[1] == 0:
        return pd.DataFrame(columns=['Feature', 'VIF'])
    vif_list = []
    for i, col in enumerate(df.columns):
        try:
            v = variance_inflation_factor(df.values, i)
            if np.isinf(v) or np.isnan(v):
                print(f"  WARNING: VIF for '{col}' is inf/nan. Setting VIF=999.0")
                v = 999.0
        except Exception as e:
            print(f"  WARNING: VIF calculation failed for '{col}': {e}. Setting VIF=999.0")
            v = 999.0
        vif_list.append({'Feature': col, 'VIF': round(v, 2)})
    return pd.DataFrame(vif_list).sort_values('VIF', ascending=False)

def heatmap(x, y, **kwargs):
    """Custom heatmap function"""
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256
        palette = sns.color_palette("Blues", n_colors)

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color)

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min)
            val_position = min(max(val_position, 0), 1)
            ind = int(val_position * (n_colors - 1))
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01
            val_position = min(max(val_position, 0), 1)
            return val_position * size_scale

    if 'x_order' in kwargs:
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]: p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs:
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]: p[0] for p in enumerate(y_names)}

    subplot_spec = kwargs.get('subplot_spec', None)
    if subplot_spec is not None:
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        plot_grid = GridSpecFromSubplotSpec(1, 15, subplot_spec=subplot_spec, hspace=0.2, wspace=0.1)
    else:
        plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)
    ax = plt.subplot(plot_grid[:, :-1])

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k: v for k, v in kwargs.items() if k not in [
        'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 
        'marker', 'x_order', 'y_order', 'xlabel', 'ylabel', 'subplot_spec'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in color],
        rasterized=True,
        **kwargs_pass_on
    )
    ax.set_xticks([v for k, v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k, v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F9F9F9')

    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))

    if color_min < color_max:
        ax = plt.subplot(plot_grid[:, -1])
        col_x = [0] * len(palette)
        bar_y = np.linspace(color_min, color_max, n_colors)
        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5] * len(palette),
            left=col_x,
            height=bar_height,
            color=palette,
            linewidth=0,
            rasterized=True
        )
        ax.set_xlim(1, 2)
        ax.grid(False)
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
        ax.yaxis.tick_right()

def corrplot(data, size_scale=500, marker='s', subplot_spec=None):
    """Create correlation plot"""
    corr = pd.melt(data.reset_index(), id_vars='index').replace(np.nan, 0)
    corr.columns = ['x', 'y', 'value']
    heatmap_kwargs = dict(
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0, 1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )
    if subplot_spec is not None:
        heatmap_kwargs['subplot_spec'] = subplot_spec
    heatmap(
        corr['x'], corr['y'],
        **heatmap_kwargs
    )

def plot_correlation_heatmap(df, out_dir, name):
    """Plot correlation heatmap"""
    print(f"\n>>> Plotting correlation heatmap")
    fig = plt.figure(figsize=(max(10, len(df.columns) * 0.5), max(10, len(df.columns) * 0.5)))
    corr = df.corr(method='spearman')
    corrplot(corr, size_scale=400)
    base_path = os.path.join(out_dir, name)
    save_figure(fig, base_path)
    plt.close()
    print(f"  Correlation heatmap saved: {base_path}.*")

def plot_vif_bar(vif_df, out_dir, name_prefix):
    """Plot VIF bar chart"""
    v = vif_df.sort_values('VIF', ascending=True)
    n_features = len(v)
    fig_height = max(6, n_features * 0.4)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    # Color scheme
    colors = ['#E66100' if x > 10 else '#FDB863' if x > 5 else '#87CEEB' for x in v['VIF']]
    bars = ax.barh(v['Feature'], v['VIF'], color=colors, edgecolor='white', linewidth=0.5)
    
    ax.axvline(5, c='#F39C12', ls='--', lw=1.5, label='VIF=5')
    ax.axvline(10, c='#E74C3C', ls='--', lw=1.5, label='VIF=10')
    ax.set_xscale('log')
    ax.set_xlim(0.9, max(v['VIF'].max() * 1.2, 15))
    ax.set_xlabel('VIF', fontsize=9, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=9, fontweight='bold')
    
    for bar, vif in zip(bars, v['VIF']):
        width = bar.get_width()
        ax.text(width * 1.05, bar.get_y() + bar.get_height()/2, 
                f'{vif:.1f}', va='center', ha='left', fontsize=8)
    
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    base_path = os.path.join(out_dir, name_prefix)
    save_figure(fig, base_path)
    plt.close()
    print(f"  VIF plot saved: {base_path}.*")

def correlation_filter(X, threshold=0.6):
    """
    Perform iterative correlation filtering
    - Removes features with |Spearman rho| > threshold
    - When two features are highly correlated, keeps the feature that appears
      earlier in the original column order (fa), removes the later one (fb).
      This ensures deterministic results without introducing bias from the
      outcome variable (no data leakage).
    - When two variables exceed |rho| > 0.6, the one appearing earlier in the
      pre-specified variable list (from 01_data_preprocessing numeric_cols order)
      is retained. No outcome variable is used — avoiding selection bias.
    - Returns filtered features and removal log
    """
    X_filtered = X.copy()
    removed = []
    
    while True:
        corr = X_filtered.corr(method='spearman')
        corr_arr = corr.values.copy()
        np.fill_diagonal(corr_arr, 0)
        corr_arr[np.tril_indices_from(corr_arr)] = 0
        # 按绝对值判定：|rho| > threshold 即剔除（强负相关同样构成共线性）
        corr_abs = np.abs(corr_arr)
        max_corr_abs = corr_abs.max()
        
        if max_corr_abs <= threshold:
            break
        
        i, j = np.unravel_index(corr_abs.argmax(), corr_arr.shape)
        fa, fb = corr.index[i], corr.columns[j]
        max_corr = corr_arr[i, j]  # 记录带符号的实际 rho（供日志）
        
        # Keep the earlier feature (fa), remove the later one (fb)
        # Decision is based solely on feature order — no outcome variable
        # used, avoiding selective bias (data leakage)
        rem, keep = fb, fa
        
        removed.append({'removed': rem, 'kept': keep, 'rho': max_corr})
        X_filtered = X_filtered.drop(columns=[rem])
        print(f"  Removed: {rem} (kept: {keep}, rho={max_corr:.3f})")
    
    return X_filtered, removed

def plot_combined_figure(vif_before, vif_after, X_before, X_after, out_dir):
    """Combine all four figures into a single 2x2 grid:
    (a) VIF Before Filtering | (b) VIF After Filtering
    (c) Correlation Before   | (d) Correlation After
    Panel labels placed at the upper-left corner of each subplot.
    """
    print("\n>>> Creating combined 2x2 figure...")
    
    fig = plt.figure(figsize=(16, 14))
    main_grid = plt.GridSpec(2, 2, wspace=0.30, hspace=0.35)
    
    def add_panel_label(ax, label):
        ax.text(-0.12, 1.08, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', ha='left', va='top')
    
    def draw_vif(ax, vif_df):
        v = vif_df.sort_values('VIF', ascending=True)
        colors = ['#E66100' if x > 10 else '#FDB863' if x > 5 else '#87CEEB' for x in v['VIF']]
        bars = ax.barh(v['Feature'], v['VIF'], color=colors, edgecolor='white', linewidth=0.5)
        for bar, vif in zip(bars, v['VIF']):
            width = bar.get_width()
            ax.text(width * 1.03, bar.get_y() + bar.get_height()/2,
                    f'{vif:.1f}', va='center', ha='left', fontsize=7)
        ax.axvline(5, c='#FDB863', ls='--', lw=1.2)
        ax.axvline(10, c='#E66100', ls='--', lw=1.2)
        ax.set_xscale('log')
        ax.set_xlim(0.9, max(v['VIF'].max() * 1.2, 15))
        ax.set_xlabel('VIF', fontsize=9, fontweight='bold')
        ax.set_ylabel('Variable', fontsize=9, fontweight='bold')
        ax.tick_params(axis='both', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # === Panel (a): VIF Before Filtering (左上) ===
    ax_a = fig.add_subplot(main_grid[0, 0])
    draw_vif(ax_a, vif_before)
    add_panel_label(ax_a, 'a')
    
    # === Panel (b): VIF After Filtering (右上) ===
    ax_b = fig.add_subplot(main_grid[0, 1])
    draw_vif(ax_b, vif_after)
    add_panel_label(ax_b, 'b')
    
    # === Panel (c): Correlation Before Filtering (左下) ===
    corr_before = X_before.corr(method='spearman')
    corrplot(corr_before, size_scale=400, subplot_spec=main_grid[1, 0])
    ax_c = fig.axes[-2]
    add_panel_label(ax_c, 'c')
    
    # === Panel (d): Correlation After Filtering (右下) ===
    corr_after = X_after.corr(method='spearman')
    corrplot(corr_after, size_scale=400, subplot_spec=main_grid[1, 1])
    ax_d = fig.axes[-2]
    add_panel_label(ax_d, 'd')
    
    fig.subplots_adjust(wspace=0.30, hspace=0.35)
    base_path = os.path.join(out_dir, 'Combined_VIF_Correlation')
    
    # Save directly as vector PDF, PNG, and compressed TIFF
    for fmt in ['pdf', 'png', 'tif']:
        fpath = f"{base_path}.{fmt}"
        kwargs = dict(dpi=600, bbox_inches='tight', transparent=False)
        if fmt == 'tif':
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(fpath, format=fmt, **kwargs)
        print(f"  Saved: {fpath}", flush=True)
    
    plt.close(fig)
    print(f"  Combined figure saved: {base_path}.*", flush=True)

def main():
    print("="*60)
    print("  VIF Analysis Pipeline")
    print("  - Correlation Filtering (|rho| > 0.6 removed)")
    print("  - Avoids data leakage: fits on Derivation cohort only")
    print("="*60)

    # ============================================================
    # Step 1: Load training data (Derivation cohort)
    # ============================================================
    print(f"\n>>> Loading training data: {TRAIN_PATH}")
    df_train = pd.read_csv(TRAIN_PATH)
    # Strip _1 and _2 suffixes from column names to match FINAL_FEATURE_NAMES
    df_train.columns = [_strip_suffix(col) for col in df_train.columns]
    print(f"Training data shape: {df_train.shape}")
    print(f"  Columns: {df_train.columns.tolist()}")

    # ============================================================
    # Step 2: Prepare features (exclude ID and label)
    # ============================================================
    # Use FINAL_FEATURE_NAMES from joblib (already suffix-stripped)
    feature_cols = FINAL_FEATURE_NAMES
    X_train = df_train[feature_cols].copy()
    y_train = df_train['label']
    
    print(f"Initial features ({len(feature_cols)}): {X_train.columns.tolist()}")

    # ============================================================
    # Step 2.5: Inject noise feature for analysis (following original code)
    # ============================================================
    print("\n>>> Setting random seed for reproducibility")
    np.random.seed(123)  # Unified global seed for reproducibility, consistent with all other scripts

    print(">>> Injecting noise feature")
    X_train_noisy = X_train.copy()
    X_train_noisy['noise_feature'] = np.random.randn(len(X_train_noisy))
    
    print(f"Features after noise injection: {X_train_noisy.columns.tolist()}")

    # ============================================================
    # Step 3: VIF Before Correlation Filtering (on training data)
    # ============================================================
    print("\n" + "="*60)
    print("  VIF Analysis Before Correlation Filtering")
    print("  (Based on Derivation cohort)")
    print("="*60)
    
    vif_before = calculate_vif(X_train_noisy)
    vif_before.to_csv(os.path.join(OUTPUT_ROOT, 'vif_before_filtering.csv'), index=False)
    print("\nVIF Before Filtering:")
    print(vif_before.to_string())
    
    # Plot VIF before filtering
    plot_vif_bar(vif_before, OUTPUT_ROOT, 'FigS1_VIF_Before_Filtering')
    
    # Plot correlation heatmap before filtering (using noisy data)
    plot_correlation_heatmap(X_train_noisy, OUTPUT_ROOT, 'FigS2_Correlation_Before_Filtering')

    # ============================================================
    # Step 4: Correlation Filtering (on training data only - NO DATA LEAKAGE)
    # ============================================================
    print("\n" + "="*60)
    print("  Correlation Filtering (|rho| > 0.6 removed)")
    print("  (Fitted on Derivation cohort only)")
    print("="*60)
    
    X_train_filtered, removed_features = correlation_filter(X_train_noisy, threshold=CORR_THRESHOLD)
    
    print(f"\nFeatures removed: {len(removed_features)}")
    print(f"Features remaining: {len(X_train_filtered.columns)}")
    
    # Save removal log
    removal_df = pd.DataFrame(removed_features)
    if not removal_df.empty:
        removal_df.to_csv(os.path.join(OUTPUT_ROOT, 'removed_features_log.csv'), index=False)

    # ============================================================
    # Step 5: VIF After Correlation Filtering (on training data)
    # ============================================================
    print("\n" + "="*60)
    print("  VIF Analysis After Correlation Filtering")
    print("  (Based on Derivation cohort)")
    print("="*60)
    
    vif_after = calculate_vif(X_train_filtered)
    vif_after.to_csv(os.path.join(OUTPUT_ROOT, 'vif_after_filtering.csv'), index=False)
    print("\nVIF After Filtering:")
    print(vif_after.to_string())
    
    # Plot VIF after filtering
    plot_vif_bar(vif_after, OUTPUT_ROOT, 'FigS3_VIF_After_Filtering')
    
    # Plot correlation heatmap after filtering
    plot_correlation_heatmap(X_train_filtered, OUTPUT_ROOT, 'FigS4_Correlation_After_Filtering')

    # ============================================================
    # Step 5.5: Iterative VIF Removal (VIF > 10)
    # ============================================================
    print("\n" + "="*60)
    print("  Iterative VIF Removal (VIF > 10)")
    print("  (Based on Derivation cohort)")
    print("="*60)
    
    X_vif_cleaned = X_train_filtered.copy()
    vif_removal_log = []
    
    while True:
        vif_current = calculate_vif(X_vif_cleaned)
        high_vif = vif_current[vif_current['VIF'] > 10]
        
        if high_vif.empty:
            print("  All VIF <= 10. Iterative removal complete.")
            break
        
        # Find the feature with the highest VIF, excluding noise_feature
        high_vif_sorted = high_vif.sort_values('VIF', ascending=False)
        feature_to_remove = None
        for _, row in high_vif_sorted.iterrows():
            if row['Feature'] != 'noise_feature':
                feature_to_remove = row['Feature']
                break
        
        if feature_to_remove is None:
            print("  WARNING: Only noise_feature has VIF > 10. Stopping removal (noise_feature is protected).")
            break
        
        vif_value = high_vif_sorted[high_vif_sorted['Feature'] == feature_to_remove]['VIF'].values[0]
        print(f"  Removing: {feature_to_remove} (VIF={vif_value:.1f})")
        vif_removal_log.append({'removed': feature_to_remove, 'VIF': vif_value})
        X_vif_cleaned = X_vif_cleaned.drop(columns=[feature_to_remove])
    
    # Save VIF removal log
    if vif_removal_log:
        vif_removal_df = pd.DataFrame(vif_removal_log)
        vif_removal_df.to_csv(os.path.join(OUTPUT_ROOT, 'vif_removal_log.csv'), index=False)
        print(f"\n  VIF removal log saved: vif_removal_log.csv ({len(vif_removal_log)} features removed)")
    else:
        print("\n  No features removed by VIF iteration (all VIF <= 10).")
        pd.DataFrame(columns=['removed', 'VIF']).to_csv(
            os.path.join(OUTPUT_ROOT, 'vif_removal_log.csv'), index=False)
    
    # Final VIF after iterative removal
    vif_final = calculate_vif(X_vif_cleaned)
    vif_final.to_csv(os.path.join(OUTPUT_ROOT, 'vif_final.csv'), index=False)
    print("\nVIF After Iterative Removal:")
    print(vif_final.to_string())
    print(f"\nFeatures remaining after VIF cleaning: {len(X_vif_cleaned.columns)}")
    print(f"  {X_vif_cleaned.columns.tolist()}")

    # --- noise_feature VIF warning check ---
    if 'noise_feature' in X_vif_cleaned.columns:
        noise_vif = vif_final[vif_final['Feature'] == 'noise_feature']['VIF'].values[0]
        if noise_vif > 10:
            print("\n" + "!"*60)
            print("  WARNING: noise_feature has VIF = {:.1f} (> 10)".format(noise_vif))
            print("  This may indicate small sample size or random instability.")
            print("  Monitor this feature carefully in downstream feature selection.")
            print("!"*60)

    # ============================================================
    # Step 6: Apply same filtering to Validation cohorts
    # ============================================================
    print("\n" + "="*60)
    print("  Applying Filtering to Validation Cohorts")
    print("  (Using removal list from Derivation cohort - NO DATA LEAKAGE)")
    print("="*60)
    
    removed_feature_names = [r['removed'] for r in removed_features]
    
    if HAS_TEMPORAL:
        print(f"\n>>> Loading Temporal validation data: {TEMPORAL_PATH}")
        df_temporal = pd.read_csv(TEMPORAL_PATH)
        # Strip _1/_2 suffixes
        df_temporal.columns = [_strip_suffix(col) for col in df_temporal.columns]
        # Column name alignment (case-insensitive match with training data)
        _col_map = {col.lower(): col for col in df_train.columns}
        for tc in df_temporal.columns:
            if tc.lower() in _col_map and tc != _col_map[tc.lower()]:
                df_temporal = df_temporal.rename(columns={tc: _col_map[tc.lower()]})
        print(f"Temporal validation data shape before filtering: {df_temporal.shape}")
        print(f"  Columns: {df_temporal.columns.tolist()}")
    
    if HAS_GEO:
        print(f"\n>>> Loading Geographical validation data: {GEO_PATH}")
        df_geo = pd.read_csv(GEO_PATH)
        # Strip _1/_2 suffixes
        df_geo.columns = [_strip_suffix(col) for col in df_geo.columns]
        # Column name alignment
        _col_map = {col.lower(): col for col in df_train.columns}
        for tc in df_geo.columns:
            if tc.lower() in _col_map and tc != _col_map[tc.lower()]:
                df_geo = df_geo.rename(columns={tc: _col_map[tc.lower()]})
        print(f"Geographical validation data shape before filtering: {df_geo.shape}")
        print(f"  Columns: {df_geo.columns.tolist()}")
    
    # 直接基于最终特征列表（相关性剔除 + VIF 迭代剔除之后）报告验证队列的剩余特征，
    # 不再使用仅按相关性剔除清单过滤的中间结果，避免与最终产物不一致造成误导
    final_val_features = [f for f in X_vif_cleaned.columns if f != 'noise_feature']
    print(f"\nRemoved features (correlation filtering): {removed_feature_names}")
    if HAS_TEMPORAL:
        _temporal_remaining = [f for f in final_val_features if f in df_temporal.columns]
        print(f"Temporal validation remaining features ({len(_temporal_remaining)}): {_temporal_remaining}")
    if HAS_GEO:
        _geo_remaining = [f for f in final_val_features if f in df_geo.columns]
        print(f"Geographical validation remaining features ({len(_geo_remaining)}): {_geo_remaining}")

    # ============================================================
    # Step 7: Save processed datasets (using X_vif_cleaned — after VIF removal)
    # ============================================================
    print("\n" + "="*60)
    print("  Saving Processed Datasets")
    print("="*60)
    
    # IMPORTANT: Use X_vif_cleaned (after VIF iterative removal), NOT X_train_filtered,
    # to ensure that features with VIF > 10 are excluded from the final dataset.
    final_features_train = X_vif_cleaned.columns.tolist()
    df_train_filtered = df_train[['ID', 'label']].copy()
    
    for feat in final_features_train:
        if feat == 'noise_feature':
            # noise_feature from the noisy dataset (if survived)
            df_train_filtered[feat] = X_train_noisy['noise_feature'].values
        else:
            # other features from original training data
            df_train_filtered[feat] = df_train[feat].values
    
    df_train_filtered.to_csv(os.path.join(OUTPUT_ROOT, 'Derivation cohort.csv'), index=False)
    print(f"  Saved: Derivation cohort.csv ({df_train_filtered.shape[0]} rows, {df_train_filtered.shape[1]} columns)")
    if 'noise_feature' in final_features_train:
        print(f"  Note: noise_feature retained (VIF={vif_final[vif_final['Feature']=='noise_feature']['VIF'].values[0]})")
    
    # Save Temporal validation data (NO noise_feature added)
    final_features_val = [f for f in final_features_train if f != 'noise_feature']
    if HAS_TEMPORAL:
        df_temporal_filtered_out = df_temporal[['ID', 'label']].copy()
        for feat in final_features_val:
            if feat in df_temporal.columns:
                df_temporal_filtered_out[feat] = df_temporal[feat].values
    
        df_temporal_filtered_out.to_csv(os.path.join(OUTPUT_ROOT, 'Temporal external validation cohort.csv'), index=False)
        print(f"  Saved: Temporal external validation cohort.csv ({df_temporal_filtered_out.shape[0]} rows, {df_temporal_filtered_out.shape[1]} columns)")
    
    # Save Geographical validation data (NO noise_feature added)
    if HAS_GEO:
        df_geo_filtered_out = df_geo[['ID', 'label']].copy()
        for feat in final_features_val:
            if feat in df_geo.columns:
                df_geo_filtered_out[feat] = df_geo[feat].values
    
        df_geo_filtered_out.to_csv(os.path.join(OUTPUT_ROOT, 'Geographical external validation cohort.csv'), index=False)
        print(f"  Saved: Geographical external validation cohort.csv ({df_geo_filtered_out.shape[0]} rows, {df_geo_filtered_out.shape[1]} columns)")
    
    # Save remaining features list
    with open(os.path.join(OUTPUT_ROOT, 'remaining_features.txt'), 'w') as f:
        f.write("Features after correlation filtering (VIF analysis):\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total features (Derivation cohort): {len(final_features_train)}\n")
        f.write(f"Total features (Validation cohorts): {len(final_features_val)}\n\n")
        f.write("Derivation cohort features:\n")
        for i, feat in enumerate(final_features_train, 1):
            f.write(f"  {i}. {feat}\n")
        f.write("\nValidation cohort features (without noise_feature):\n")
        for i, feat in enumerate(final_features_val, 1):
            f.write(f"  {i}. {feat}\n")
    print(f"  Saved: remaining_features.txt")

    # ============================================================
    # Step 8: Create combined figure
    # ============================================================
    plot_combined_figure(vif_before, vif_after, X_train_noisy, X_train_filtered, OUTPUT_ROOT)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Correlation threshold:                {CORR_THRESHOLD}")
    print(f"  - Features removed (correlation):       {len(removed_features)}")
    print(f"  - Features removed (VIF > 10):           {len(vif_removal_log)}")
    print(f"  - Features remaining:                    {len(X_vif_cleaned.columns)}")
    print(f"\nOutput directory: {OUTPUT_ROOT}")
    print(f"\nFiles generated:")
    print(f"  - Derivation cohort.csv (filtered)")
    if HAS_TEMPORAL:
        print(f"  - Temporal external validation cohort.csv (filtered)")
    if HAS_GEO:
        print(f"  - Geographical external validation cohort.csv (filtered)")
    print(f"  - vif_before_filtering.csv")
    print(f"  - vif_after_filtering.csv")
    print(f"  - removed_features_log.csv")
    print(f"  - vif_removal_log.csv")
    print(f"  - vif_final.csv")
    print(f"  - remaining_features.txt")
    print(f"  - FigS1_VIF_Before_Filtering.*")
    print(f"  - FigS2_Correlation_Before_Filtering.*")
    print(f"  - FigS3_VIF_After_Filtering.*")
    print(f"  - FigS4_Correlation_After_Filtering.*")
    print(f"  - Combined_VIF_Correlation.*")

if __name__ == "__main__":
    main()