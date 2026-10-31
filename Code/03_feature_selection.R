# =============================================================================
# 03_feature_selection.R
# Four complementary feature selection methods:
#   1. Boruta (Wrapper)      - Random Forest + shadow features
#   2. LASSO   (Embedded)    - L1 regularized logistic regression
#   3. Exhaustive AIC (Statistical) - Best subset selection via AIC
#   4. mRMRe   (Filter)      - Minimum Redundancy Maximum Relevance
# 
# All selection performed ONLY on Derivation cohort (no data leakage).
# Results applied to Temporal and Geographical external validation cohorts.
# Output: 600 DPI, Arial font, PDF + PNG + TIFF, SCI-compliant.
# =============================================================================

# --- Path Configuration: auto-detect relative to Code folder ---
script_dir <- dirname(normalizePath(
  sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
))
BASE_DIR <- dirname(script_dir)  # parent of Code/

INPUT_DIR <- file.path(BASE_DIR, "vif_analysis")        # input from 02_vif_analysis
OUTPUT_DIR <- file.path(BASE_DIR, "feature_selection")   # output for 03_feature_selection

# ===================== CLI Bridge Mode（供 04_Model_Selection.py 折内重放调用） =====================
# 用法：Rscript 03_feature_selection.R <input.csv> <output_features.txt>
# 无参数时保持原有行为完全不变；带 2 个及以上参数时进入桥接模式：
#   - 从 <input.csv> 读取数据（替代 vif_analysis/Derivation cohort.csv）
#   - 工作目录切换到 <output_features.txt> 所在目录（所有中间产物写入该目录）
#   - 关闭绘图输出、跳过第 9 节（验证队列改写），仅执行 4 方法共识特征选择
#   - 最终共识特征（每行一个）写入 <output_features.txt>
bridge_args <- commandArgs(trailingOnly = TRUE)
BRIDGE_MODE <- length(bridge_args) >= 2
PLOTS_ENABLED <- TRUE
if (BRIDGE_MODE) {
  BRIDGE_INPUT <- normalizePath(bridge_args[1], mustWork = TRUE)
  # normalizePath 对"尚不存在"的输出文件在 Windows 下可能报错，故只规范化其
  # 所在目录（调用方已事先创建），再拼接文件名，保证跨平台稳妥。
  bridge_out_dir <- normalizePath(dirname(bridge_args[2]), mustWork = TRUE)
  BRIDGE_OUTPUT <- file.path(bridge_out_dir, basename(bridge_args[2]))
  PLOTS_ENABLED <- FALSE
  cat("BRIDGE_MODE: TRUE\n")
  cat("BRIDGE_INPUT: ", BRIDGE_INPUT, "\n", sep = "")
  cat("BRIDGE_OUTPUT: ", BRIDGE_OUTPUT, "\n", sep = "")
}

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

if (BRIDGE_MODE) {
  OUTPUT_DIR <- dirname(BRIDGE_OUTPUT)
  if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
  }
}

cat("BASE_DIR:  ", BASE_DIR, "\n")
cat("INPUT_DIR: ", INPUT_DIR, "\n")
cat("OUTPUT_DIR:", OUTPUT_DIR, "\n")

setwd(OUTPUT_DIR)

# ===================== 0. Package Loading & Setup =====================
# 低危修复2：先统一自检并安装缺失包，全部就绪后再集中 library() 加载。
# 原写法先 library() 后 require/install 自检，包缺失时会在 library() 处直接报错中断。
required_packages <- c("Boruta", "ggplot2", "dplyr", "tidyr", "scales",
                       "glmnet", "caret", "grid", "pheatmap", "pROC",
                       "ResourceSelection", "ggridges", "bestglm", "mRMRe")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

# 自检完成后统一加载所有包（grid 为 R 自带包，无需安装但一并加载）
library(Boruta)
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(glmnet)
library(caret)
library(grid)
library(pheatmap)
library(pROC)              # ROC/AUC 计算
library(ResourceSelection)  # Hosmer-Lemeshow 检验
library(ggridges)          # geom_density_ridges_gradient
library(bestglm)           # Exhaustive AIC（穷举最佳子集）
library(mRMRe)             # mRMR 过滤法特征选择

options(bitmapType = "cairo")

# ===================== 1. Data Loading =====================
data <- read.csv(if (BRIDGE_MODE) BRIDGE_INPUT else file.path(INPUT_DIR, "Derivation cohort.csv"),
                 header = TRUE)
print(str(data))

id_col <- which(colnames(data) == "ID")
label_col <- which(colnames(data) == "label")
if (length(id_col) == 0) stop("Column 'ID' not found in Derivation cohort.csv")
if (length(label_col) == 0) stop("Column 'label' not found in Derivation cohort.csv")
predictor_cols <- setdiff(seq_len(ncol(data)), c(id_col, label_col))

# 01_data_preprocessing.py and 02_vif_analysis.py ensure all features
# are numeric and y is integer 0/1 — no additional conversion needed.
X <- data[, predictor_cols]
y <- data[, label_col]

feature_names <- colnames(X)
n_features <- ncol(X)
set.seed(123)  # Unified global seed — all methods use this

# ===================== 2. Boruta (Wrapper) - FULLY PRESERVED =====================
print("\n[1/4] Running Boruta feature selection...")

# maxRuns=500: sufficient for stable importance estimates (Kursa & Rudnicki, 2010, J Stat Softw)
Var_Boruta <- Boruta(
  x = X, 
  y = as.factor(y), 
  maxRuns = 500, 
  pValue = 0.05, 
  getImp = getImpRfZ
)

print(Var_Boruta)

boruta_selected <- getSelectedAttributes(Var_Boruta, withTentative = FALSE)
print(boruta_selected)

# Boruta Ridge Plot
imp_history <- as.data.frame(Var_Boruta$ImpHistory)

decision_all <- data.frame(
  feature = names(Var_Boruta$finalDecision),
  decision = Var_Boruta$finalDecision
)

data_long <- imp_history %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "importance") %>%
  left_join(decision_all, by = "feature") %>%
  filter(is.finite(importance))

feature_order <- data_long %>%
  group_by(feature) %>%
  summarise(median_imp = median(importance, na.rm = TRUE)) %>%
  arrange(median_imp) %>%
  pull(feature)

# Unified output helper: PDF (pdf+sans), PNG, TIFF (both png/tiff use sans)
save_plot_triple <- function(plot_obj, pdf_name, png_name, tif_name) {
  if (!PLOTS_ENABLED) return(invisible(NULL))
  # PDF via pdf() device with sans font family
  pdf(pdf_name, width = 10, height = 8, family = "sans")
  print(plot_obj)
  dev.off()
  # PNG via png() device
  png(png_name, width = 6000, height = 4800, res = 600, family = "sans")
  print(plot_obj)
  dev.off()
  # TIFF via tiff() device
  tiff(tif_name, width = 6000, height = 4800, res = 600, family = "sans")
  print(plot_obj)
  dev.off()
}

boruta_ridge_plot <- ggplot(data_long, aes(x = importance, y = factor(feature, levels = feature_order), fill = decision)) +
  geom_density_ridges_gradient(scale = 2, rel_min_height = 0.01, color = "black") +
  scale_fill_manual(values = c(
    "Confirmed" = alpha("#02BBC1", 0.6),
    "Tentative" = alpha("#FFC107", 0.6),
    "Rejected" = alpha("#E53935", 0.6),
    "Shadow" = alpha("#757575", 0.6)
  )) +
  labs(
    title = "Feature Importance Ridge Plot Based on Boruta",
    x = "Importance (Z-Score)",
    y = "Features"
  ) +
  theme_bw() +
  theme(
    legend.position = "right",
    panel.spacing = unit(0.1, "lines"),
    axis.text.y = element_text(size = 10)
  )

save_plot_triple(boruta_ridge_plot, 
                 "boruta_ridge_plot.pdf",
                 "boruta_ridge_plot.png",
                 "boruta_ridge_plot.tif")

write.csv(boruta_selected, "boruta_selected_features.csv", row.names = FALSE)

# ===================== 3. LASSO (Embedded) - standardize = TRUE =====================
print("\n[2/4] Running LASSO feature selection (standardize = TRUE)...")

X_matrix <- as.matrix(X)
y_vector <- y  # y is already 0/1 numeric

LASSO_single <- function(seed) {
  set.seed(seed)
  boot_idx <- sample(1:nrow(X_matrix), size = nrow(X_matrix), replace = TRUE)
  X_boot <- X_matrix[boot_idx, ]
  y_boot <- y_vector[boot_idx]
  LASSO_model <- cv.glmnet(
    X_boot, 
    y_boot, 
    family = "binomial", 
    standardize = TRUE,
    alpha = 1
  )
  # Use lambda.min: maximize predictive performance / variable discovery
  # (consistent with internal bootstrap which already uses lambda.min)
  best_lambda <- LASSO_model$lambda.min
  LASSO_coefficients <- coef(LASSO_model, s = best_lambda)
  important_features <- rownames(LASSO_coefficients)[LASSO_coefficients[, 1] != 0]
  # FIX: Exclude intercept from selected features
  important_features <- setdiff(important_features, "(Intercept)")
  return(important_features)
}

n_LASSO_reps <- 50

all_LASSO_selections <- list()
for (i in 1:n_LASSO_reps) {
  # 低危修复1：种子由 i + 500 改为 123 + i，与全管线统一 set.seed(123)（见上文数据加载部分）的种子基数保持一致。
  # 选择"保留逐次显式种子"而非"删除 set.seed 依赖全局随机流推进"，理由：
  # 显式种子使每次 Bootstrap 的结果独立于前文已消耗的随机数（如 Boruta 内部抽样），
  # 即使单独重跑本节也能完全复现，可复现性更稳健。
  all_LASSO_selections[[i]] <- LASSO_single(seed = 123 + i)
}

all_LASSO_features <- unlist(all_LASSO_selections)
LASSO_freq_table <- table(all_LASSO_features)

# FIX: Ensure all 14 features are included, even those with 0 frequency
all_feature_names <- colnames(X)
LASSO_freq_full <- setNames(rep(0, length(all_feature_names)), all_feature_names)
for (feat in names(LASSO_freq_table)) {
  LASSO_freq_full[feat] <- LASSO_freq_table[feat]
}
LASSO_freq <- LASSO_freq_full / n_LASSO_reps

LASSO_freq_sorted <- sort(LASSO_freq, decreasing = TRUE)
print(head(LASSO_freq_sorted, 10))

# ========== Optimize LASSO stability threshold via inner bootstrap + outer 10-fold CV ==========
print("\nOptimizing LASSO stability threshold via inner bootstrap + outer 10-fold CV...")

candidate_thresholds_lasso <- seq(0.5, 0.95, by = 0.05)
print(paste("Testing thresholds:", paste(candidate_thresholds_lasso, collapse = ", ")))

n_folds_outer_lasso <- 10
fold_ids_outer_lasso <- createFolds(as.factor(y), k = n_folds_outer_lasso, list = TRUE, returnTrain = FALSE)
threshold_cv_auc_lasso <- matrix(NA, nrow = n_folds_outer_lasso, ncol = length(candidate_thresholds_lasso))
colnames(threshold_cv_auc_lasso) <- paste0("thr_", candidate_thresholds_lasso)

for (f in 1:n_folds_outer_lasso) {
  test_idx <- fold_ids_outer_lasso[[f]]
  train_idx <- setdiff(1:nrow(X), test_idx)
  
  X_train <- X[train_idx, , drop = FALSE]
  y_train <- y[train_idx]
  X_test  <- X[test_idx, , drop = FALSE]
  y_test  <- y[test_idx]
  
  # ---- Inner Bootstrap to compute LASSO selection frequency ----
  n_bootstrap_inner_lasso <- 50
  freq_inner_lasso <- setNames(rep(0, ncol(X_train)), colnames(X_train))
  
  for (b in 1:n_bootstrap_inner_lasso) {
    boot_idx <- sample(1:nrow(X_train), size = nrow(X_train), replace = TRUE)
    X_boot <- X_train[boot_idx, , drop = FALSE]
    y_boot <- y_train[boot_idx]
    
    lasso_boot <- cv.glmnet(as.matrix(X_boot), y_boot,
                            family = "binomial", alpha = 1, standardize = TRUE)
    coef_boot <- coef(lasso_boot, s = lasso_boot$lambda.min)
    selected_feats <- rownames(coef_boot)[coef_boot[, 1] != 0]
    selected_feats <- setdiff(selected_feats, "(Intercept)")
    
    for (feat in selected_feats) {
      if (feat %in% names(freq_inner_lasso)) {
        freq_inner_lasso[feat] <- freq_inner_lasso[feat] + 1
      }
    }
  }
  
  freq_pct_inner_lasso <- (freq_inner_lasso / n_bootstrap_inner_lasso) * 100
  
  # ---- Evaluate AUC for each candidate threshold ----
  for (t_idx in seq_along(candidate_thresholds_lasso)) {
    thr <- candidate_thresholds_lasso[t_idx]
    selected_feats <- names(freq_pct_inner_lasso[freq_pct_inner_lasso >= thr * 100])
    
    if (length(selected_feats) == 0) {
      threshold_cv_auc_lasso[f, t_idx] <- NA
      next
    }
    
    df_train_lr <- data.frame(X_train[, selected_feats, drop = FALSE],
                              y = y_train)
    lr_model <- glm(y ~ ., data = df_train_lr, family = binomial)
    pred_prob <- predict(lr_model, newdata = data.frame(X_test[, selected_feats, drop = FALSE]),
                         type = "response")
    roc_obj <- roc(y_test, pred_prob, quiet = TRUE)
    threshold_cv_auc_lasso[f, t_idx] <- as.numeric(auc(roc_obj))
  }
}

# 1-SE rule: median threshold within 1 SE of the peak AUC.
# This balances parsimony and stability, between peak and highest-threshold.
mean_auc_lasso <- colMeans(threshold_cv_auc_lasso, na.rm = TRUE)
if (all(is.na(mean_auc_lasso))) stop("All LASSO threshold CV AUCs are NA.")
max_mean_auc_lasso <- max(mean_auc_lasso, na.rm = TRUE)
se_peak_lasso <- apply(threshold_cv_auc_lasso, 2, function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))))
se_peak_lasso <- se_peak_lasso[which.max(mean_auc_lasso)]
candidate_1se_lasso <- candidate_thresholds_lasso[mean_auc_lasso >= max_mean_auc_lasso - se_peak_lasso]
if (length(candidate_1se_lasso) == 0) {
  best_threshold_lasso <- candidate_thresholds_lasso[which.max(mean_auc_lasso)]
} else {
  # Median position within 1-SE range (rounded to nearest candidate threshold)
  idx_1se_lasso <- which(candidate_thresholds_lasso %in% candidate_1se_lasso)
  med_idx_lasso <- idx_1se_lasso[ceiling(length(idx_1se_lasso) / 2)]
  best_threshold_lasso <- candidate_thresholds_lasso[med_idx_lasso]
}
print(paste("LASSO stability threshold (1-SE median):", best_threshold_lasso))
print(paste("  Peak CV-AUC:", round(max_mean_auc_lasso, 4),
            "| 1-SE median AUC:", round(mean_auc_lasso[candidate_thresholds_lasso == best_threshold_lasso], 4)))

# ---- Apply optimized threshold to LASSO feature selection ----
threshold_LASSO_optimized <- best_threshold_lasso
stable_LASSO_features <- names(LASSO_freq[LASSO_freq >= threshold_LASSO_optimized])
print(paste("LASSO stable features (threshold =", threshold_LASSO_optimized * 100, "%):"))
print(stable_LASSO_features)

# ---- Generate LASSO Threshold Optimization Plot ----
lasso_thr_df <- data.frame(
  Threshold = candidate_thresholds_lasso,
  Mean_AUC = mean_auc_lasso,
  SD_AUC = apply(threshold_cv_auc_lasso, 2, sd, na.rm = TRUE)
)
lasso_thr_df <- lasso_thr_df[!is.na(lasso_thr_df$Mean_AUC), ]

# 1-SE selected threshold index for plot annotation
best_idx_lasso <- which(lasso_thr_df$Threshold == best_threshold_lasso)
if (length(best_idx_lasso) == 0) best_idx_lasso <- which.max(lasso_thr_df$Mean_AUC)

lasso_thr_plot <- ggplot(lasso_thr_df, aes(x = Threshold, y = Mean_AUC)) +
  geom_errorbar(aes(ymin = Mean_AUC - SD_AUC, ymax = Mean_AUC + SD_AUC),
                width = 0.02, color = "gray50") +
  geom_line(color = "#FF7F0E", linewidth = 0.8) +
  geom_point(size = 3, color = "#FF7F0E") +
  geom_point(data = lasso_thr_df[best_idx_lasso, ],
             aes(x = Threshold, y = Mean_AUC), color = "red", size = 4) +
  geom_text(data = lasso_thr_df[best_idx_lasso, ],
            aes(x = Threshold, y = Mean_AUC,
                label = sprintf("1-SE\n%.2f\nAUC=%.4f", Threshold, Mean_AUC)),
            vjust = -1.5, color = "red", size = 3.5, family = "sans") +
  scale_x_continuous(breaks = candidate_thresholds_lasso) +
  labs(title = "Optimization of LASSO Stability Threshold via Bootstrap Frequency + 10-fold CV (1-SE)",
       subtitle = "Bootstrap resampling (50 reps) for selection frequency; 10-fold CV on the threshold; 1-SE median rule (Harrell, 2015)",
       x = "Stability Threshold", y = "Mean CV-AUC") +
  theme_bw() +
  theme(plot.title = element_text(size = 12, face = "bold", family = "sans"),
        plot.subtitle = element_text(size = 9, color = "gray40"),
        text = element_text(family = "sans"))

save_plot_triple(lasso_thr_plot,
                 "LASSO_Threshold_Optimization.pdf",
                 "LASSO_Threshold_Optimization.png",
                 "LASSO_Threshold_Optimization.tif")

# Final LASSO model for coefficient path
LASSO_model <- cv.glmnet(
  X_matrix, 
  y_vector, 
  family = "binomial", 
  standardize = TRUE,
  alpha = 1
)
# Use lambda.min: maximize predictive performance, consistent with internal bootstrap
best_lambda <- LASSO_model$lambda.min

# LASSO Coefficient Path - base plot with sans font
# filename=NULL 时仅在当前设备上绘制（供 2×2 组合图复用，不新建设备）
plot_LASSO_path <- function(model, best_lambda, filename, width, height, res = NULL) {
  if (!is.null(filename) && !PLOTS_ENABLED) return(invisible(NULL))
  if (!is.null(filename)) {
    if (!is.null(res)) {
      png(filename, width = width, height = height, res = res, family = "sans")
    } else {
      pdf(filename, width = width, height = height, family = "sans")
    }
  }
  par(cex = 1.0)
  n_features <- nrow(model$beta)
  line_colors <- hue_pal()(n_features)
  feature_names <- rownames(model$beta)
  plot(model, xvar = "lambda", label = FALSE, main = "",
       yaxt = "n", ylab = "", xaxt = "n", xlab = "", col = line_colors)
  axis(2, las = 2, cex.axis = 1.0)
  title(ylab = "Coefficients", line = 2.5, cex.lab = 1.1)
  x_ticks <- axTicks(1)
  axis(1, at = x_ticks, labels = round(exp(x_ticks), 4), cex.axis = 0.9)
  title(xlab = "Log Lambda", line = 2.8, cex.lab = 1.1)
  abline(v = log(best_lambda), lty = 2, col = "red")
  legend("topright", legend = feature_names, col = line_colors,
         lty = 1, lwd = 1, cex = 0.7, bty = "n", title = "Features")
  if (!is.null(filename)) dev.off()
  invisible(NULL)
}

plot_LASSO_path(LASSO_model$glmnet.fit, best_lambda, "LASSO_coefficient_path.pdf", 10, 8)
plot_LASSO_path(LASSO_model$glmnet.fit, best_lambda, "LASSO_coefficient_path.png", 6000, 4800, 600)
plot_LASSO_path(LASSO_model$glmnet.fit, best_lambda, "LASSO_coefficient_path.tif", 6000, 4800, 600)

# Helper: save base R plot to PDF/PNG/TIFF
save_base_plot_triple <- function(plot_fn, pdf_name, png_name, tif_name,
                                   pdf_w = 10, pdf_h = 8, rast_w = 6000, rast_h = 4800) {
  if (!PLOTS_ENABLED) return(invisible(NULL))
  pdf(pdf_name, width = pdf_w, height = pdf_h, family = "sans")
  plot_fn()
  dev.off()
  png(png_name, width = rast_w, height = rast_h, res = 600, family = "sans")
  plot_fn()
  dev.off()
  tiff(tif_name, width = rast_w, height = rast_h, res = 600, family = "sans")
  plot_fn()
  dev.off()
}

# =============================================================================
# 2×2 组合图生成（SCI 投稿规范：600 DPI、sans 字体）
#   panels : 长度为 4 的列表；元素为 ggplot 对象或零参 base 绘图函数。
#            base 函数仅绘制到当前设备（不新建设备、不关闭设备）。
#   out_base: 输出文件基名（不含扩展名，自动生成 .pdf/.png/.tif）
# 实现说明：在同一设备上以 par(mfrow=c(2,2)) 布局，base 面板直接调用
#   （自行推进区域），ggplot 面板先 plot.new() 推进区域后以
#   print(..., vp=viewport()) 绘制到对应区域（见 print.ggplot 的 vp 参数）。
#   a/b/c/d 标签不在此处绘制，由用户后期在成品图中自行添加。PDF 保持矢量。
# =============================================================================
combine_sci_2x2 <- function(panels, out_base,
                            pdf_w = 12, pdf_h = 10) {
  stopifnot(length(panels) == 4)
  if (!PLOTS_ENABLED) return(invisible(NULL))

  draw_combined <- function() {
    par(mfrow = c(2, 2), family = "sans")
    for (i in seq_len(4)) {
      pnl <- panels[[i]]
      if (inherits(pnl, "ggplot")) {
        # 组合图中去除 ggplot 面板的标题与副标题（a/b/c/d 标签由用户后期自行添加）
        pnl <- pnl + theme(plot.title = element_blank(), plot.subtitle = element_blank())
        # 推进到当前区域并绘制 ggplot
        plot.new()
        fig <- par("fig")
        print(pnl, vp = viewport(x = (fig[1] + fig[2]) / 2, y = (fig[3] + fig[4]) / 2,
                                 width = fig[2] - fig[1], height = fig[4] - fig[3]))
      } else if (is.function(pnl)) {
        # base 绘图函数内部自行推进区域
        pnl()
      } else {
        stop("Unsupported panel type: must be ggplot object or base plotting function")
      }
    }
  }
  
  # PDF（矢量）
  pdf(paste0(out_base, ".pdf"), width = pdf_w, height = pdf_h, family = "sans")
  draw_combined()
  dev.off()
  # PNG / TIFF（600 DPI）
  png(paste0(out_base, ".png"), width = pdf_w * 600, height = pdf_h * 600,
      res = 600, family = "sans")
  draw_combined()
  dev.off()
  tiff(paste0(out_base, ".tif"), width = pdf_w * 600, height = pdf_h * 600,
       res = 600, family = "sans", compression = "lzw")
  draw_combined()
  dev.off()
  
  cat("  Combined 2x2 figure saved: ", out_base, ".pdf / .png / .tif\n", sep = "")
  invisible(NULL)
}

# LASSO CV Error Plot
plot_lasso_cv_error <- function() {
  plot(LASSO_model, main = "", yaxt = "n", cex.lab = 1.1, cex.axis = 1.0)
}
save_base_plot_triple(plot_lasso_cv_error,
  "LASSO_cv_error_plot.pdf",
  "LASSO_cv_error_plot.png",
  "LASSO_cv_error_plot.tif")

# LASSO Frequency Plot
# output_file=NULL 时仅在当前设备上绘制（供 2×2 组合图复用，不新建设备）
# main_title="" 时去除标题（组合图用；单图保留原标题）
plot_LASSO_freq <- function(freq_table, output_file, width, height, res = NULL,
                            main_title = "LASSO Feature Selection Frequency") {
  if (!is.null(output_file) && !PLOTS_ENABLED) return(invisible(NULL))
  if (!is.null(output_file)) {
    if (!is.null(res)) {
      png(output_file, width = width, height = height, res = res, family = "sans")
    } else {
      pdf(output_file, width = width, height = height, family = "sans")
    }
  }
  par(cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.4, mar = c(5, 10, 4, 2) + 0.1)
  freq_sorted <- sort(freq_table, decreasing = TRUE) * 100
  feature_names <- names(freq_sorted)
  n_f <- length(freq_sorted)
  stable_idx <- freq_sorted >= threshold_LASSO_optimized * 100
  plot(0, 0, xlim = c(0, 110), ylim = c(n_f + 0.5, 0.5), 
       type = "n", xlab = "Selection Frequency (%)", ylab = "",
       main = main_title, yaxt = "n")
  abline(v = threshold_LASSO_optimized * 100, col = "#ffcccc", lwd = 2, lty = 2)
  text(threshold_LASSO_optimized * 100, n_f, paste0(round(threshold_LASSO_optimized * 100), "% Threshold"), pos = 4, col = "#999999", cex = 0.8)
  for (i in 1:n_f) {
    lines(c(0, freq_sorted[i]), c(i, i), col = "#cccccc", lwd = 1)
    if (stable_idx[i]) {
      points(freq_sorted[i], i, pch = 19, col = "#ff7f0e", cex = 1.5)
      text(freq_sorted[i] + 2, i, 
           sprintf("%.0f", freq_sorted[i] * n_LASSO_reps),
           pos = 4, cex = 0.7, col = "#ff7f0e")
    } else {
      points(freq_sorted[i], i, pch = 19, col = "#cccccc", cex = 0.8)
    }
  }
  axis(2, at = 1:n_f, labels = feature_names, las = 1, cex.axis = 0.8)
  if (!is.null(output_file)) dev.off()
  invisible(NULL)
}

plot_LASSO_freq(LASSO_freq, "LASSO_feature_frequency.pdf", 12, 8)
plot_LASSO_freq(LASSO_freq, "LASSO_feature_frequency.png", 7200, 4800, 600)
plot_LASSO_freq(LASSO_freq, "LASSO_feature_frequency.tif", 7200, 4800, 600)

# =============================================================================
# LASSO 四图 2×2 组合（SCI 规范；顺序 = 生成顺序）
#   a = LASSO_Threshold_Optimization（阈值优化）
#   b = LASSO_coefficient_path（系数路径）
#   c = LASSO_cv_error_plot（CV 误差）
#   d = LASSO_feature_frequency（特征频率）
# =============================================================================
combine_sci_2x2(
  list(
    lasso_thr_plot,
    function() plot_LASSO_path(LASSO_model$glmnet.fit, best_lambda, NULL, NULL, NULL),
    plot_lasso_cv_error,
    function() plot_LASSO_freq(LASSO_freq, NULL, NULL, NULL, main_title = "")
  ),
  out_base = file.path(OUTPUT_DIR, "Combined_LASSO_2x2")
)

write.csv(stable_LASSO_features, "LASSO_selected_features.csv", row.names = FALSE)

# ===================== 4. Exhaustive AIC (Statistical) =====================
# FIX: Changed from BIC to AIC to retain more clinically meaningful features
print("\n[3/4] Running Exhaustive AIC feature selection...")

# Prepare data: numeric y (0/1) as last column
df_aic <- as.data.frame(X)
df_aic$y <- y  # y is already 0/1 numeric
df_aic <- df_aic[, c(setdiff(colnames(df_aic), "y"), "y")]

# Run AIC selection.
# 注意：bestglm 的二项 GLM 穷举实现硬性限制 p <= 15（需枚举 2^p 个模型）。
# 折内重放后候选特征数通常为 15，但个别折因 VIF/相关性过滤保留更多列（如 16），
# 此时穷举不可行，退化为双向逐步 AIC（stats::step），并在日志中显式记录口径。
p_aic <- ncol(df_aic) - 1
aic_method <- if (p_aic <= 15) "exhaustive" else "stepwise"
cat(sprintf("AIC selection method: %s (candidate predictors p = %d; bestglm exhaustive cap = 15)\n",
            aic_method, p_aic))

if (aic_method == "exhaustive") {
  aic_result <- bestglm(df_aic, IC = "AIC", family = binomial, method = "exhaustive")
  print(aic_result)
  best_model <- aic_result$BestModel
} else {
  cat("NOTE: p > 15 exceeds bestglm's exhaustive cap; using bidirectional stepwise AIC.\n")
  full_model <- glm(y ~ ., data = df_aic, family = binomial)
  best_model <- stats::step(full_model, direction = "both", trace = 0)
  aic_result <- NULL
}

# Extract best model features
best_formula <- formula(best_model)
feat_exhaustive <- attr(terms(best_formula), "term.labels")
if (length(feat_exhaustive) == 0) feat_exhaustive <- character(0)
print(paste(sprintf("AIC selected features (%s):", aic_method),
            paste(feat_exhaustive, collapse = ", ")))
print(paste("Number of features:", length(feat_exhaustive)))

write.csv(feat_exhaustive, "Exhaustive_AIC_features.csv", row.names = FALSE)

# --- AIC: Model Size vs AIC Plot ---
# 仅穷举模式有完整子集表；逐步回退时无子集表，留空数据（桥接模式本就不绘图）
if (!is.null(aic_result)) {
  aic_subset <- aic_result$Subsets
  # 修复：行名是 "0 ", "1 ", "11*" 等格式，带空格和可能的 * 标记
  # 需要去掉空格和 * 后转换为数字
  n_features_raw <- gsub("\\s+|\\*", "", rownames(aic_subset))
  aic_df <- data.frame(
    n_features = as.numeric(n_features_raw),
    AIC = aic_subset$AIC
  )
  best_idx <- which.min(aic_df$AIC)
} else {
  aic_df <- data.frame(n_features = integer(0), AIC = numeric(0))
  best_idx <- integer(0)
}

# Exhaustive AIC: Model Size vs AIC Plot
# main_title="" 时去除标题（组合图用；单图保留原标题）
plot_aic_model_size <- function(main_title = "Exhaustive AIC: Model Size vs AIC") {
  par(cex = 1.0, mar = c(5, 5, 4, 2) + 0.1)
  plot(aic_df$n_features, aic_df$AIC, type = "p", pch = 16, col = "#1f77b4",
       xlab = "Number of Features", ylab = "AIC",
       main = main_title,
       cex.axis = 1.0, cex.lab = 1.1, cex.main = 1.2)
  points(aic_df$n_features[best_idx], aic_df$AIC[best_idx], 
         col = "red", pch = 16, cex = 2)
  text(aic_df$n_features[best_idx], aic_df$AIC[best_idx] + 5,
       paste("Optimal: ", aic_df$n_features[best_idx], " features"),
       pos = 3, col = "red", cex = 0.9)
  lines(aic_df$n_features, aic_df$AIC, col = "#1f77b4", lwd = 1, lty = 2)
  grid()
}
save_base_plot_triple(plot_aic_model_size,
  "Exhaustive_AIC_Model_Size_vs_AIC.pdf",
  "Exhaustive_AIC_Model_Size_vs_AIC.png",
  "Exhaustive_AIC_Model_Size_vs_AIC.tif")

# ===================== 5. mRMRe (Filter) - CV-optimized k + Bootstrap stability =====================
print("\n[4/4] Running mRMRe with CV-based k optimization and Bootstrap stability...")

# ---------- Prepare data ----------
# mRMRe requires strictly numeric (double) columns, not integer.
# Convert X columns to numeric type for mRMRe compatibility.
X_mrmr <- as.data.frame(lapply(X, as.numeric))
y_mrmr <- as.numeric(y)  # y is already 0/1; ensure numeric (not integer) for mRMRe

# Check for NA or non-numeric issues
if (any(is.na(X_mrmr))) stop("NA values found in features. Please clean data first.")
if (any(is.na(y_mrmr))) stop("NA values found in outcome.")

df_mrmre <- data.frame(X_mrmr, y_num = y_mrmr)
data_mrmre <- mRMR.data(data = df_mrmre)

# ---------- Step 1: CV to determine optimal k ----------
# FIX (Issue 1): Run mRMRe WITHIN each CV fold to prevent information leakage.
# The mRMR.classic call must be on the training fold only, not the full dataset.
print("Step 1: Running 5-fold CV to determine optimal feature count k (mRMRe inside folds)...")
# FIX: Limit candidate k range to min(ncol(X), 14) since we have 14 features
candidate_k <- 1:ncol(X)  # Allow all possible feature counts
cv_auc_mean <- numeric(length(candidate_k))
cv_auc_sd   <- numeric(length(candidate_k))

n_folds <- 5
fold_ids <- createFolds(as.factor(y), k = n_folds, list = TRUE, returnTrain = FALSE)

for (k in candidate_k) {
  cat("  Testing k =", k, "...\n")
  
  fold_aucs <- numeric(n_folds)
  
  for (f in 1:n_folds) {
    test_idx <- fold_ids[[f]]
    train_idx <- setdiff(1:nrow(X), test_idx)
    
    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test  <- X[test_idx, , drop = FALSE]
    y_test  <- y[test_idx]
    
    # Build mRMRe data on training fold only (convert to numeric for mRMRe compatibility)
    df_train_mrmr <- data.frame(lapply(X_train, as.numeric), y_num = as.numeric(y_train))
    data_train_mrmr <- mRMR.data(data = df_train_mrmr)
    
    filter_k <- tryCatch({
      mRMR.classic(
        data = data_train_mrmr,
        target_indices = ncol(df_train_mrmr),
        feature_count = k
      )
    }, error = function(e) NULL)
    
    if (is.null(filter_k)) { fold_aucs[f] <- NA; next }
    
    selected_idx <- solutions(filter_k)[[1]]
    if (length(selected_idx) == 0) { fold_aucs[f] <- NA; next }
    
    # Train LR on training fold with selected features
    X_train_sel <- X_train[, selected_idx, drop = FALSE]
    X_test_sel  <- X_test[, selected_idx, drop = FALSE]
    
    df_train_lr <- data.frame(X_train_sel, y = y_train)
    lr_model <- glm(y ~ ., data = df_train_lr, family = binomial)
    
    # Predict on test fold
    pred_prob <- predict(lr_model, newdata = data.frame(X_test_sel), type = "response")
    
    # Compute AUC using pROC
    if (!require("pROC", quietly = TRUE)) {
      install.packages("pROC", repos = "https://cloud.r-project.org")
      library(pROC)
    }
    roc_obj <- roc(y_test, pred_prob, quiet = TRUE)
    fold_aucs[f] <- as.numeric(auc(roc_obj))
  }
  
  cv_auc_mean[k] <- mean(fold_aucs, na.rm = TRUE)
  cv_auc_sd[k]   <- sd(fold_aucs, na.rm = TRUE)
}

# 1-SE rule: smallest k whose mean CV-AUC is within 1 SE of the peak
# (Harrell, 2015 — consistent with the "one standard error" principle)
max_mean_auc_k <- max(cv_auc_mean, na.rm = TRUE)
se_peak_k <- cv_auc_sd[which.max(cv_auc_mean)] / sqrt(n_folds)
candidate_k_1se <- candidate_k[cv_auc_mean >= max_mean_auc_k - se_peak_k]
if (length(candidate_k_1se) == 0) {
  optimal_k_final <- candidate_k[which.max(cv_auc_mean)]
} else {
  optimal_k_final <- min(candidate_k_1se, na.rm = TRUE)
}

print(paste0("Peak k: ", candidate_k[which.max(cv_auc_mean)],
             " (AUC = ", round(max_mean_auc_k, 4), ")"))
print(paste0("1-SE selected k: ", optimal_k_final,
             " (AUC = ", round(cv_auc_mean[optimal_k_final], 4), ")"))

# --- Chart 1: k vs CV-AUC line plot ---
cv_auc_df <- data.frame(
  k = candidate_k,
  AUC = cv_auc_mean,
  SD  = cv_auc_sd
)

k_auc_plot <- ggplot(cv_auc_df, aes(x = k, y = AUC)) +
  geom_errorbar(aes(ymin = AUC - SD, ymax = AUC + SD), width = 0.2, color = "gray60") +
  geom_line(color = "#2E86AB", linewidth = 0.8) +
  geom_point(size = 2.5, color = "#2E86AB") +
  geom_point(data = cv_auc_df[optimal_k_final, ], 
             aes(x = k, y = AUC), color = "red", size = 4) +
  geom_text(data = cv_auc_df[optimal_k_final, ],
            aes(x = k, y = AUC, 
                label = sprintf("1-SE k=%d\nAUC=%.4f", k, AUC)),
            vjust = -1.5, color = "red", size = 4) +
  scale_x_continuous(breaks = candidate_k) +
  labs(title = "mRMRe: Optimal Feature Count Selection via 5-fold CV",
       subtitle = "1-SE criterion: smallest k within 1 SE of peak AUC (Harrell, 2015)",
       x = "Number of Features (k)", y = "CV-AUC (Mean +/- SD)") +
  theme_bw() +
  theme(plot.title = element_text(size = 13, face = "bold"),
        plot.subtitle = element_text(size = 9, color = "gray40"),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 11))

save_plot_triple(k_auc_plot,
                 "mRMRe_Optimal_k_Selection.pdf",
                 "mRMRe_Optimal_k_Selection.png",
                 "mRMRe_Optimal_k_Selection.tif")

# ========== Step 1.5: Optimize stability threshold via inner bootstrap + outer 10-fold CV ==========
print("\nStep 1.5: Optimizing stability threshold via inner bootstrap + outer 10-fold CV...")

# 候选阈值（从 0.5 到 0.95，步长 0.05）
candidate_thresholds <- seq(0.5, 0.95, by = 0.05)
print(paste("Testing thresholds:", paste(candidate_thresholds, collapse = ", ")))

n_folds_outer <- 10
fold_ids_outer <- createFolds(as.factor(y), k = n_folds_outer, list = TRUE, returnTrain = FALSE)
threshold_cv_auc <- matrix(NA, nrow = n_folds_outer, ncol = length(candidate_thresholds))
colnames(threshold_cv_auc) <- paste0("thr_", candidate_thresholds)

for (f in 1:n_folds_outer) {
  test_idx <- fold_ids_outer[[f]]
  train_idx <- setdiff(1:nrow(X), test_idx)
  
  X_train <- X[train_idx, , drop = FALSE]
  y_train <- y[train_idx]
  X_test  <- X[test_idx, , drop = FALSE]
  y_test  <- y[test_idx]
  
  # ---- 在训练集内做 Bootstrap 计算频率 ----
  n_bootstrap_inner <- 50  # 内层用50次与外层保持一致
  freq_inner <- setNames(rep(0, ncol(X_train)), colnames(X_train))
  
  for (b in 1:n_bootstrap_inner) {
    boot_idx <- sample(1:nrow(X_train), size = nrow(X_train), replace = TRUE)
    X_boot <- X_train[boot_idx, , drop = FALSE]
    y_boot <- y_train[boot_idx]
    
    df_boot <- data.frame(lapply(X_boot, as.numeric), y_num = as.numeric(y_boot))
    data_boot <- mRMR.data(data = df_boot)
    
    # 低危修复4：与 Step 1 一致，内层 mRMR.classic 增加 tryCatch 保护；
    # 失败时打印 warning 并跳过本次 Bootstrap（选择频率不累计），避免整个嵌套循环中断
    filter_boot <- tryCatch({
      mRMR.classic(
        data = data_boot, 
        target_indices = ncol(df_boot), 
        feature_count = optimal_k_final  # 1-SE determined optimal k
      )
    }, error = function(e) {
      warning(paste("Step 1.5 mRMR.classic failed (outer fold", f, ", bootstrap", b, "):", e$message))
      NULL
    })
    if (is.null(filter_boot)) next
    selected_idx <- solutions(filter_boot)[[1]]
    if (length(selected_idx) > 0) {
      for (feat in colnames(X_train)[selected_idx]) {
        freq_inner[feat] <- freq_inner[feat] + 1
      }
    }
  }
  
  freq_pct_inner <- (freq_inner / n_bootstrap_inner) * 100
  
  # ---- 对每个候选阈值，筛选特征并在测试集上评估 AUC ----
  for (t_idx in seq_along(candidate_thresholds)) {
    thr <- candidate_thresholds[t_idx]
    selected_feats <- names(freq_pct_inner[freq_pct_inner >= thr * 100])
    
    if (length(selected_feats) == 0) {
      threshold_cv_auc[f, t_idx] <- NA
      next
    }
    
    # 在测试集上评估
    X_test_sel <- X_test[, selected_feats, drop = FALSE]
    if (ncol(X_test_sel) == 0) {
      threshold_cv_auc[f, t_idx] <- NA
      next
    }
    
    # 用训练集拟合逻辑回归
    df_train_lr <- data.frame(X_train[, selected_feats, drop = FALSE], 
                              y = y_train)
    lr_model <- glm(y ~ ., data = df_train_lr, family = binomial)
    pred_prob <- predict(lr_model, newdata = data.frame(X_test_sel), type = "response")
    roc_obj <- roc(y_test, pred_prob, quiet = TRUE)
    threshold_cv_auc[f, t_idx] <- as.numeric(auc(roc_obj))
  }
}

# 1-SE rule: median threshold within 1 SE of the peak AUC.
# (Harrell, 2015 — consistent with LASSO threshold logic above)
mean_auc_by_threshold <- colMeans(threshold_cv_auc, na.rm = TRUE)
if (all(is.na(mean_auc_by_threshold))) stop("All mRMRe threshold CV AUCs are NA.")
max_mean_auc_thr <- max(mean_auc_by_threshold, na.rm = TRUE)
se_peak_thr <- apply(threshold_cv_auc, 2, function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))))
se_peak_thr <- se_peak_thr[which.max(mean_auc_by_threshold)]
candidate_1se_thr <- candidate_thresholds[mean_auc_by_threshold >= max_mean_auc_thr - se_peak_thr]
if (length(candidate_1se_thr) == 0) {
  best_threshold <- candidate_thresholds[which.max(mean_auc_by_threshold)]
} else {
  # Median position within 1-SE range (rounded to nearest candidate threshold)
  idx_1se_thr <- which(candidate_thresholds %in% candidate_1se_thr)
  med_idx_thr <- idx_1se_thr[ceiling(length(idx_1se_thr) / 2)]
  best_threshold <- candidate_thresholds[med_idx_thr]
}

# 绘图展示阈值 vs CV-AUC
threshold_opt_df <- data.frame(
  Threshold = candidate_thresholds,
  Mean_AUC = mean_auc_by_threshold,
  SD_AUC = apply(threshold_cv_auc, 2, sd, na.rm = TRUE)
)
threshold_opt_df <- threshold_opt_df[!is.na(threshold_opt_df$Mean_AUC), ]

# 1-SE selected threshold index for plot annotation
best_idx_thr <- which(threshold_opt_df$Threshold == best_threshold)
if (length(best_idx_thr) == 0) best_idx_thr <- which.max(threshold_opt_df$Mean_AUC)

thr_opt_plot <- ggplot(threshold_opt_df, aes(x = Threshold, y = Mean_AUC)) +
  geom_errorbar(aes(ymin = Mean_AUC - SD_AUC, ymax = Mean_AUC + SD_AUC), 
                width = 0.02, color = "gray50") +
  geom_line(color = "#2E86AB", linewidth = 0.8) +
  geom_point(size = 3, color = "#2E86AB") +
  geom_point(data = threshold_opt_df[best_idx_thr, ], 
             aes(x = Threshold, y = Mean_AUC), color = "red", size = 4) +
  geom_text(data = threshold_opt_df[best_idx_thr, ], 
            aes(x = Threshold, y = Mean_AUC, 
                label = sprintf("1-SE\n%.2f\nAUC=%.4f", Threshold, Mean_AUC)), 
            vjust = -1.5, color = "red", size = 3.5) +
  scale_x_continuous(breaks = candidate_thresholds) +
  labs(title = "Optimization of mRMRe Stability Threshold via Bootstrap Frequency + 10-fold CV (1-SE)",
       subtitle = "Bootstrap resampling (50 reps) for selection frequency; 10-fold CV on the threshold; 1-SE median rule (Harrell, 2015)", 
       x = "Stability Threshold", y = "Mean CV-AUC") +
  theme_bw() +
  theme(plot.title = element_text(size = 12, face = "bold"),
        plot.subtitle = element_text(size = 9, color = "gray40"))

save_plot_triple(thr_opt_plot, 
                  "mRMRe_Threshold_Optimization.pdf", 
                  "mRMRe_Threshold_Optimization.png", 
                  "mRMRe_Threshold_Optimization.tif")

# 打印最佳阈值
print(paste0("mRMRe stability threshold (1-SE median): ", best_threshold))
print(paste0("  Peak CV-AUC: ", round(max_mean_auc_thr, 4),
             " | 1-SE chosen AUC: ", round(mean_auc_by_threshold[candidate_thresholds == best_threshold], 4)))

# ---- 用最优阈值更新 mrmre_threshold ----
mrmre_threshold <- best_threshold * 100  # 转换为百分比
print(paste("Using optimized stability threshold:", mrmre_threshold, "%"))

# ---------- Step 2: Bootstrap stability with fixed optimal_k_final ----------
print(paste0("Step 2: 50 Bootstrap iterations with k = ", optimal_k_final, "..."))
n_bootstrap <- 50
selection_freq <- setNames(rep(0, ncol(X)), colnames(X))

for (i in 1:n_bootstrap) {
  boot_idx <- sample(1:nrow(X), size = nrow(X), replace = TRUE)
  X_boot <- X[boot_idx, , drop = FALSE]
  y_boot <- y[boot_idx]
  
  df_boot <- data.frame(lapply(X_boot, as.numeric), y_num = as.numeric(y_boot))
  data_boot <- mRMR.data(data = df_boot)
  
  # 低危修复4：与 Step 1 一致，mRMR.classic 增加 tryCatch 保护；
  # 失败时打印 warning 并跳过本次 Bootstrap（选择频率不累计），避免整个循环中断
  filter_boot <- tryCatch({
    mRMR.classic(
      data = data_boot,
      target_indices = ncol(df_boot),
      feature_count = optimal_k_final
    )
  }, error = function(e) {
    warning(paste("Step 2 mRMR.classic failed (bootstrap", i, "):", e$message))
    NULL
  })
  if (is.null(filter_boot)) next
  selected_idx_boot <- solutions(filter_boot)[[1]]
  
  if (length(selected_idx_boot) > 0) {
    selected_feats_boot <- colnames(X)[selected_idx_boot]
    for (feat in selected_feats_boot) {
      selection_freq[feat] <- selection_freq[feat] + 1
    }
  }
}

# Calculate frequency percentages
selection_freq_pct <- (selection_freq / n_bootstrap) * 100
selection_freq_sorted <- sort(selection_freq_pct, decreasing = TRUE)

# Note: mrmre_threshold is already optimized in Step 1.5 (via inner bootstrap + outer 10-fold CV)
# mrmre_threshold now contains the CV-optimized value (e.g., 55%)
stable_mrmre_features <- names(selection_freq_pct[selection_freq_pct >= mrmre_threshold])
print(paste0("mRMRe high-stability features (>=", mrmre_threshold, "% Bootstrap frequency):"))
print(stable_mrmre_features)
print(paste("Number of stable mRMRe features:", length(stable_mrmre_features)))

# Export CSV
write.csv(
  data.frame(Feature = names(selection_freq_sorted), 
             Frequency_pct = selection_freq_sorted),
  "mRMRe_Bootstrap_Stability.csv", row.names = FALSE
)
write.csv(stable_mrmre_features, "mRMRe_Stable_Features.csv", row.names = FALSE)

# --- Chart 2: Bootstrap selection frequency bar chart ---
freq_df <- data.frame(
  feature = names(selection_freq_sorted),
  frequency = selection_freq_sorted,
  stable = names(selection_freq_sorted) %in% stable_mrmre_features
)
freq_df$feature <- factor(freq_df$feature, levels = rev(names(selection_freq_sorted)))

bootstrap_bar_plot <- ggplot(freq_df, aes(x = frequency, y = feature, fill = stable)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_vline(xintercept = mrmre_threshold, linetype = "dashed", color = "red", linewidth = 0.8) +
  annotate("text", x = mrmre_threshold + 2, y = 0.5, 
           label = paste0(mrmre_threshold, "% Threshold"), hjust = 0, vjust = 0, color = "red", size = 4) +
  scale_fill_manual(values = c("TRUE" = "#2E86AB", "FALSE" = "#cccccc")) +
  scale_x_continuous(limits = c(0, 105), breaks = seq(0, 100, 20)) +
  labs(title = paste0("mRMRe Bootstrap Stability (k = ", optimal_k_final, ", 50 iterations)"),
       x = "Selection Frequency (%)", y = "") +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.y = element_text(size = 10),
        plot.title = element_text(size = 12, face = "bold"))

save_plot_triple(bootstrap_bar_plot,
                 "mRMRe_Bootstrap_Stability.pdf",
                 "mRMRe_Bootstrap_Stability.png",
                 "mRMRe_Bootstrap_Stability.tif")

# =============================================================================
# 第二张 2×2 组合图（SCI 规范；顺序 = 生成顺序）
#   a = Exhaustive_AIC_Model_Size_vs_AIC（穷举 AIC 模型规模）
#   b = mRMRe_Optimal_k_Selection（最优 k 选择）
#   c = mRMRe_Threshold_Optimization（稳定性阈值优化）
#   d = mRMRe_Bootstrap_Stability（Bootstrap 稳定性）
# =============================================================================
combine_sci_2x2(
  list(
    function() plot_aic_model_size(main_title = ""),
    k_auc_plot,
    thr_opt_plot,
    bootstrap_bar_plot
  ),
  out_base = file.path(OUTPUT_DIR, "Combined_mRMRe_AIC_2x2")
)

# ===================== 6. Consensus Analysis =====================
print("\nGenerating consensus analysis...")

# Collect all method results
method_list <- list(
  LASSO = stable_LASSO_features,
  Boruta = boruta_selected,
  Exhaustive_AIC = feat_exhaustive,
  mRMRe = stable_mrmre_features
)

methods <- names(method_list)

# --- 6a. 3-Method Consensus Venn Diagram (using VennDiagram package) ---
if (!require("VennDiagram", quietly = TRUE)) {
  install.packages("VennDiagram", repos = "https://cloud.r-project.org")
  library(VennDiagram)
}

# 低危修复6：将 VennDiagramLogger 日志阈值提升为 ERROR，抑制 venn.diagram 的 INFO 级日志，
# 避免在输出目录残留 VennDiagram.*.log 文件。
# futile.logger 是 VennDiagram 的依赖通常已安装，仍加 requireNamespace 检查以防缺失。
if (requireNamespace("futile.logger", quietly = TRUE)) {
  futile.logger::flog.threshold(futile.logger::ERROR, name = "VennDiagramLogger")
}

# 低危修复3：删除与上方 method_list 内容完全重复的 venn_data，直接复用 method_list

# Save Venn diagram (venn.diagram returns grob list, needs grid.draw)
draw_venn <- function() {
  # FIX: Ensure clean canvas before drawing
  grid::grid.newpage()
  venn_grob <- venn.diagram(
    x = method_list,
    filename = NULL,
    col = "black",
    fill = c("#FF7F0E", "#02BBC1", "#E53935", "#2E86AB"),
    alpha = 0.3,
    label.col = "black",
    cex = 1.5,
    fontface = "bold",
    fontfamily = "sans",
    cat.col = "black",
    cat.cex = 1.2,
    cat.fontface = "bold",
    cat.fontfamily = "sans",
    margin = 0.1
  )
  grid::grid.draw(venn_grob)
}
save_base_plot_triple(draw_venn,
  "Venn_Diagram_3Methods_Consensus.pdf",
  "Venn_Diagram_3Methods_Consensus.png",
  "Venn_Diagram_3Methods_Consensus.tif",
  pdf_w = 10, pdf_h = 10, rast_w = 6000, rast_h = 6000)

# --- 6a2. UpSet Plot (3-Method Consensus Visualization) ---
# UpSet 图：左上条形图 = 每个方法的选择总数，主图 = 交集大小（点阵+条形）
# 静态图中通过下方标签组合可辨认交集所属特征
if (!require("ComplexUpset", quietly = TRUE)) {
  install.packages("ComplexUpset", repos = "https://cloud.r-project.org")
  library(ComplexUpset)
}

# Build upset data: strictly 0/1 binary matrix
upset_input <- data.frame(
  Feature = feature_names,
  LASSO = as.integer(feature_names %in% stable_LASSO_features),
  Boruta = as.integer(feature_names %in% boruta_selected),
  Exhaustive_AIC = as.integer(feature_names %in% feat_exhaustive),
  mRMRe = as.integer(feature_names %in% stable_mrmre_features)
)

# --- Validate: ensure all columns are strictly 0/1 ---
cat("\n--- Validating UpSet input matrix (must be strictly 0/1) ---\n")
upset_method_cols <- c("LASSO", "Boruta", "Exhaustive_AIC", "mRMRe")
for (col in upset_method_cols) {
  col_vals <- upset_input[[col]]
  cat(sprintf("  %s: min=%d, max=%d, unique=%s\n", 
              col, min(col_vals), max(col_vals),
              paste(sort(unique(col_vals)), collapse = ", ")))
}
stopifnot("ERROR: UpSet columns must be strictly 0/1!" = 
            all(sapply(upset_input[, upset_method_cols], max) == 1))
stopifnot("ERROR: UpSet columns must be strictly 0/1!" = 
            all(sapply(upset_input[, upset_method_cols], min) == 0))
cat("  Validation passed: all columns are strictly 0/1\n")

# --- Add Consensus_3 column: 1 if selected by >=3 methods ---
selection_counts_temp <- rowSums(upset_input[, upset_method_cols])
upset_input$Consensus_3 <- as.integer(selection_counts_temp >= 3)
cat(sprintf("  Consensus_3 features (>=3 methods): %d\n", sum(upset_input$Consensus_3)))

# --- Generate Consensus_3_Feature_Table.csv ---
consensus_3_table <- data.frame(
  Feature = feature_names,
  LASSO = ifelse(feature_names %in% stable_LASSO_features, "Yes", "No"),
  Boruta = ifelse(feature_names %in% boruta_selected, "Yes", "No"),
  Exhaustive_AIC = ifelse(feature_names %in% feat_exhaustive, "Yes", "No"),
  mRMRe = ifelse(feature_names %in% stable_mrmre_features, "Yes", "No"),
  Selection_Count = selection_counts_temp,
  Consensus_3 = ifelse(selection_counts_temp >= 3, "Yes (>=3)", "No"),
  stringsAsFactors = FALSE
)
consensus_3_table <- consensus_3_table[order(-consensus_3_table$Selection_Count), ]
write.csv(consensus_3_table, "Consensus_3_Feature_Table.csv", row.names = FALSE)
cat("  Saved Consensus_3_Feature_Table.csv\n")

# Compute intersection details for annotation
find_intersection_members <- function(data, sets) {
  binary_mat <- as.matrix(data[, sets])
  combos <- unique(binary_mat)
  result <- list()
  for (i in 1:nrow(combos)) {
    combo <- combos[i, , drop = FALSE]
    match_idx <- apply(binary_mat, 1, function(r) all(r == combo[1, ]))
    members <- data$Feature[match_idx]
    if (length(members) > 0 && sum(combo) > 0) {
      combo_name <- paste(sets[combo[1, ] == 1], collapse = " ∩ ")
      if (combo_name == "") combo_name <- "None"
      result[[combo_name]] <- members
    }
  }
  return(result)
}

intersection_members <- find_intersection_members(upset_input, 
  c("LASSO", "Boruta", "Exhaustive_AIC", "mRMRe"))

upset_plot_input <- upset_input[, upset_method_cols, drop = FALSE]

# Color scheme matching the overall project palette
# Matrix dots & set size bars use method-specific colors
# Intersection bars use a blue gradient by degree (number of methods)
method_colors <- c(
  LASSO           = "#FF7F0E",  # orange
  Boruta          = "#02BBC1",  # teal
  Exhaustive_AIC  = "#E53935",  # red
  mRMRe           = "#2E86AB"   # steel blue
)

# Queries: color the matrix dots by method
upset_queries <- lapply(names(method_colors), function(s) {
  upset_query(set = s, fill = method_colors[[s]], color = method_colors[[s]])
})

# Compute max set size for Y-axis limits
max_set_size <- max(colSums(upset_plot_input))

upset_plot <- ComplexUpset::upset(
  upset_plot_input,
  intersect = upset_method_cols,
  min_size = 0,
  sort_sets = FALSE,
  sort_intersections_by = c("degree", "cardinality"),
  name = "Method Combination",
  base_annotations = list(
    "Intersection Size" = intersection_size(
      mapping = aes(fill = "intersection"),
      counts = TRUE,
      bar_number_threshold = 1,
      text = list(size = 4.5)
    ) +
      scale_fill_manual(values = c(intersection = "#1565C0"), guide = "none") +
      scale_y_continuous(limits = c(0, 10), breaks = seq(0, 10, 2)) +
      ylab("Intersection Size")
  ),
  set_sizes = upset_set_size() +
    scale_fill_manual(values = method_colors, guide = "none") +
    scale_x_reverse(breaks = 0:max_set_size, limits = c(max_set_size, 0)) +
    xlab("Features per Method"),
  queries = upset_queries,
  themes = upset_default_themes(
    text = element_text(family = "sans", color = "black")
  )
) +
  theme(
    legend.position = "right",
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white"),
    axis.text = element_text(size = 10, color = "black"),
    axis.title = element_text(size = 11, color = "black"),
    text = element_text(color = "black")
  )

save_plot_triple(
  upset_plot,
  "UpSet_Plot_3Methods_Consensus.pdf",
  "UpSet_Plot_3Methods_Consensus.png",
  "UpSet_Plot_3Methods_Consensus.tif"
)

# Export intersection membership details to CSV for reference
intersection_df <- data.frame(
  Intersection = names(intersection_members),
  Count = sapply(intersection_members, length),
  Features = sapply(intersection_members, paste, collapse = ", "),
  stringsAsFactors = FALSE
)
intersection_df <- intersection_df[order(-intersection_df$Count), ]
write.csv(intersection_df, "UpSet_Intersection_Details.csv", row.names = FALSE)

# Print intersection details for user reference
cat("\n--- UpSet Intersection Details ---\n")
for (i in 1:nrow(intersection_df)) {
  cat(sprintf("  %s (n=%d): %s\n", 
              intersection_df$Intersection[i], 
              intersection_df$Count[i],
              intersection_df$Features[i]))
}

# --- 6b. 3-Method Consensus Matrix (pheatmap) ---
consensus_matrix <- matrix(0, nrow = length(feature_names), ncol = length(methods),
                           dimnames = list(feature_names, methods))

for (i in seq_along(feature_names)) {
  f <- feature_names[i]
  consensus_matrix[i, "LASSO"] <- ifelse(f %in% stable_LASSO_features, 1, 0)
  consensus_matrix[i, "Boruta"] <- ifelse(f %in% boruta_selected, 1, 0)
  consensus_matrix[i, "Exhaustive_AIC"] <- ifelse(f %in% feat_exhaustive, 1, 0)
  consensus_matrix[i, "mRMRe"] <- ifelse(f %in% stable_mrmre_features, 1, 0)
}

# Sort by selection count
selection_counts <- rowSums(consensus_matrix)
consensus_matrix <- consensus_matrix[order(-selection_counts), , drop = FALSE]

# pheatmap consensus matrix
# IMPROVEMENT: Use temporary file instead of pdf(NULL) for safer device handling
# (pheatmap internally opens a device for layout computation even with filename=NA)
if (PLOTS_ENABLED) {
  tmp_pdf <- tempfile(fileext = ".pdf")
  pdf(tmp_pdf)
  consensus_hm <- pheatmap(consensus_matrix,
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = c("#E8E8E8", "#2E86AB"),
    legend = FALSE,
    display_numbers = TRUE,
    number_color = "black",
    fontsize_number = 10,
    fontsize_row = 10,
    fontsize_col = 11,
    main = "Consensus Matrix: Feature Selection by Four Methods\n(>=3 Methods = Final Consensus)",
    angle_col = 0,
    border_color = "white",
    filename = NA
  )
  dev.off()
  unlink(tmp_pdf)

  pdf("Consensus_Matrix_3Methods_Consensus.pdf", width = 8, height = 10, family = "sans")
  grid::grid.newpage()
  grid::grid.draw(consensus_hm$gtable)
  dev.off()

  png("Consensus_Matrix_3Methods_Consensus.png", width = 4800, height = 6000, res = 600, family = "sans")
  grid::grid.newpage()
  grid::grid.draw(consensus_hm$gtable)
  dev.off()

  tiff("Consensus_Matrix_3Methods_Consensus.tif", width = 4800, height = 6000, res = 600, family = "sans")
  grid::grid.newpage()
  grid::grid.draw(consensus_hm$gtable)
  dev.off()
}

# --- 6c. Consensus_3 Feature Count Bar Chart ---
# 显示每个特征被几种方法选中，红色高亮 >=3 的共识特征
selection_count_df <- data.frame(
  Feature = names(selection_counts),
  Count = as.integer(selection_counts),
  Consensus_3 = selection_counts >= 3
)
selection_count_df <- selection_count_df[order(-selection_count_df$Count), ]
selection_count_df$Feature <- factor(selection_count_df$Feature, 
                                     levels = rev(selection_count_df$Feature))

consensus_count_plot <- ggplot(selection_count_df, 
                                aes(x = Count, y = Feature, fill = Consensus_3)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = Count), hjust = -0.5, size = 5, color = "black") +
  geom_vline(xintercept = 2.5, linetype = "dashed", color = "red", linewidth = 1) +
  annotate("text", x = 2.7, y = 0.8, 
           label = ">=3 Methods (Consensus)", hjust = 0, color = "red", size = 4.5,
           fontface = "bold") +
  scale_fill_manual(values = c("TRUE" = "#E53935", "FALSE" = "#B0BEC5")) +
  scale_x_continuous(limits = c(0, 4.8), breaks = 0:4) +
  labs(title = "Feature Selection Consensus: Number of Methods Selecting Each Feature",
       subtitle = "Red: >=3 methods (Consensus features)  |  Grey: <3 methods",
       x = "Number of Methods", y = "") +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.y = element_text(size = 10),
        plot.title = element_text(size = 12, face = "bold"),
        plot.subtitle = element_text(size = 10, color = "gray40"))

save_plot_triple(consensus_count_plot,
                 "Consensus_3_Feature_Count.pdf",
                 "Consensus_3_Feature_Count.png",
                 "Consensus_3_Feature_Count.tif")

# --- 6d. Feature Count Comparison Bar Chart ---
feature_counts <- sapply(method_list, length)
count_df <- data.frame(
  Method = methods,
  Count = feature_counts
)
count_df$Method <- factor(count_df$Method, levels = methods)

count_bar_plot <- ggplot(count_df, aes(x = Method, y = Count, fill = Method)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = Count), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("#FF7F0E", "#02BBC1", "#E53935", "#2E86AB")) +
  labs(title = "Number of Features Selected by Each Method",
       x = "", y = "Number of Features") +
  theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(size = 11, face = "bold"))

save_plot_triple(count_bar_plot,
                 "Feature_Count_Comparison.pdf",
                 "Feature_Count_Comparison.png",
                 "Feature_Count_Comparison.tif")

# --- 6e. Final Feature Selection (>=3 Methods Consensus) ---
# Rule 1: Majority voting (>=3 out of 4 methods)
final_features <- names(selection_counts[selection_counts >= 3])

# Rule 2: If empty, use >=2 methods
if (length(final_features) == 0) {
  cat("Majority voting (>=3) empty. Falling back to >=2 methods...\n")
  final_features <- names(selection_counts[selection_counts >= 2])
}

# Rule 3: If still empty, use Exhaustive AIC
if (length(final_features) == 0) {
  cat(">=2 also empty. Using Exhaustive AIC result...\n")
  final_features <- feat_exhaustive
}

print(paste("Final selected features:", paste(final_features, collapse = ", ")))
print(paste("Number of final features:", length(final_features)))

write.csv(final_features, "final_selected_features.csv", row.names = FALSE)

# ===================== 7. Final Model & Diagnostics =====================
print("\nGenerating final model, diagnostics, and forest plot...")

if (!BRIDGE_MODE && length(final_features) > 0) {
  # Fit final logistic regression model
  final_formula <- as.formula(paste("y_vector ~", paste(final_features, collapse = " + ")))
  final_model <- glm(final_formula, data = as.data.frame(X_matrix), family = binomial)
  
  # --- FIX #4: 10-fold Cross-Validated AUC ---
  # Manual 10-fold CV for robust AUC performance estimate
  fold_ids_cv <- createFolds(as.factor(y_vector), k = 10, list = TRUE, returnTrain = FALSE)
  cv_auc_folds <- numeric(10)
  for (cv_i in 1:10) {
    test_idx <- fold_ids_cv[[cv_i]]
    train_idx <- setdiff(1:nrow(X_matrix), test_idx)
    # Build train data with y_vector included for formula compatibility
    cv_train <- as.data.frame(X_matrix[train_idx, , drop = FALSE])
    cv_train$y_vector <- y_vector[train_idx]
    cv_model <- glm(final_formula, data = cv_train, family = binomial)
    cv_test <- as.data.frame(X_matrix[test_idx, , drop = FALSE])
    cv_pred <- predict(cv_model, newdata = cv_test, type = "response")
    cv_roc <- roc(y_vector[test_idx], cv_pred, quiet = TRUE)
    cv_auc_folds[cv_i] <- as.numeric(auc(cv_roc))
  }
  final_cv_auc_mean <- mean(cv_auc_folds)
  final_cv_auc_sd <- sd(cv_auc_folds)
  print(paste("  Final Model 10-fold CV AUC:", round(final_cv_auc_mean, 4), 
              "(SD =", round(final_cv_auc_sd, 4), ")"))
  
  # --- FIX #5: Variance Inflation Factor (VIF) ---
  if (!require("car", quietly = TRUE)) {
    install.packages("car", repos = "https://cloud.r-project.org")
    library(car)
  }
  vif_values <- tryCatch({
    vif(final_model)
  }, error = function(e) {
    print(paste("  VIF calculation warning:", e$message))
    NULL
  })
  if (!is.null(vif_values)) {
    print("  Final Model VIF values:")
    print(round(vif_values, 2))
    high_vif <- vif_values[vif_values > 10]
    if (length(high_vif) > 0) {
      warning(paste("  High VIF (>10) detected in:", paste(names(high_vif), collapse = ", ")))
    } else {
      print("  All VIF values <= 10 (no multicollinearity concerns)")
    }
  }
  
  # --- Issue 4: Model Discrimination (AUC / C-statistic) ---
  if (!require("pROC", quietly = TRUE)) {
    install.packages("pROC", repos = "https://cloud.r-project.org")
    library(pROC)
  }
  pred_prob_final <- predict(final_model, type = "response")
  roc_final <- roc(y_vector, pred_prob_final, quiet = TRUE)
  auc_final <- as.numeric(auc(roc_final))
  print(paste("  Final Model AUC (C-statistic):", round(auc_final, 4)))
  
  # --- Issue 4: Calibration (Hosmer-Lemeshow test) ---
  if (!require("ResourceSelection", quietly = TRUE)) {
    install.packages("ResourceSelection", repos = "https://cloud.r-project.org")
    library(ResourceSelection)
  }
  hl_test <- hoslem.test(y_vector, pred_prob_final, g = 10)
  print(paste("  Hosmer-Lemeshow test: chi-sq =", round(hl_test$statistic, 3),
              ", df =", hl_test$parameter, ", p =", round(hl_test$p.value, 4)))
  
  # Extract OR and CI
  coef_summary <- summary(final_model)$coefficients
  coef_df <- as.data.frame(coef_summary)
  coef_df <- coef_df[rownames(coef_df) != "(Intercept)", , drop = FALSE]
  
  coef_df$OR <- exp(coef_df[, 1])
  coef_df$CI_lower <- exp(coef_df[, 1] - 1.96 * coef_df[, 2])
  coef_df$CI_upper <- exp(coef_df[, 1] + 1.96 * coef_df[, 2])
  coef_df$Feature <- rownames(coef_df)
  coef_df$p_value <- coef_df[, 4]
  
  # Format p-value label (Issue 3)
  coef_df$p_label <- ifelse(coef_df$p_value < 0.001, "p<0.001",
                            sprintf("p=%.3f", coef_df$p_value))
  
  # Sort by OR
  coef_df <- coef_df[order(coef_df$OR), ]
  coef_df$Feature <- factor(coef_df$Feature, levels = coef_df$Feature)
  
  # Determine x-axis range: labels start at CI_upper + offset, need enough space
  max_or <- max(coef_df$CI_upper, na.rm = TRUE)
  # Estimate max label width (approximate character width in data units)
  label_text <- sprintf("%.2f (%.2f-%.2f) %s", coef_df$OR, coef_df$CI_lower, coef_df$CI_upper, coef_df$p_label)
  max_label_chars <- max(nchar(label_text))
  # Each char ~0.08 units at size 3.5, add buffer
  label_offset <- max_or * 0.15  # Gap between CI_upper and label start
  label_width_estimate <- max_label_chars * 0.08
  x_max <- max_or + label_offset + label_width_estimate + 0.5

  forest_plot <- ggplot(coef_df, aes(x = OR, y = Feature)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
    geom_errorbar(aes(xmin = CI_lower, xmax = CI_upper, y = Feature), 
                  width = 0.2, color = "#2E86AB") +
    geom_point(size = 3, color = "#2E86AB") +
    # Labels positioned at CI_upper + offset to avoid overlap with error bars
    geom_text(aes(x = CI_upper + label_offset, 
                  label = sprintf("%.2f (%.2f-%.2f) %s", OR, CI_lower, CI_upper, p_label)), 
              hjust = 0, size = 3.5) +  # Start text at CI_upper + offset, left-aligned
    coord_cartesian(xlim = c(0, x_max), clip = "off") +
    labs(title = "Final Model: Odds Ratio with 95% Confidence Interval",
         x = "Odds Ratio (95% CI)", y = "") +
    theme_bw() +
    theme(axis.text.y = element_text(size = 10),
          plot.title = element_text(size = 12, face = "bold"),
          plot.margin = margin(5, 180, 5, 5))  # Extra right margin for labels
  
  save_plot_triple(forest_plot,
                   "Final_Model_Forest_Plot.pdf",
                   "Final_Model_Forest_Plot.png",
                   "Final_Model_Forest_Plot.tif")
} else if (BRIDGE_MODE) {
  cat("BRIDGE_MODE: skipping final-model diagnostics and forest plot.\n")
} else {
  print("No final features selected. Skipping diagnostics and forest plot.")
}

# ===================== 8. Summary Text File =====================
print("\nGenerating feature selection summary...")

summary_lines <- c(
  "========================================",
  "Feature Selection Summary",
  "========================================",
  "",
  paste("Date:", Sys.Date()),
  "",
  "--- Methods ---",
  paste("1. Boruta (Wrapper):", length(boruta_selected), "features"),
  paste("   Features:", paste(boruta_selected, collapse = ", ")),
  "",
  paste("2. LASSO (Embedded, standardize=TRUE, lambda.min, threshold=", threshold_LASSO_optimized * 100, "%):",
        length(stable_LASSO_features), "features"),
  paste("   Features:", paste(stable_LASSO_features, collapse = ", ")),
  "",
  paste("3. Exhaustive AIC (Statistical):", length(feat_exhaustive), "features"),
  paste("   Features:", paste(feat_exhaustive, collapse = ", ")),
  "",
  paste("4. mRMRe (Filter, CV-optimized k=", optimal_k_final, ", Bootstrap >=", mrmre_threshold, "%):", 
        length(stable_mrmre_features), "features"),
  paste("   Features:", paste(stable_mrmre_features, collapse = ", ")),
  "",
  "--- Consensus (>=3 Methods Majority Voting) ---",
  "",
  paste("Intersection of 4 methods:", length(Reduce(intersect, method_list)), "features"),
  paste("Features selected by >= 3 methods:", 
        paste(names(selection_counts[selection_counts >= 3]), collapse = ", ")),
  "",
  paste("FINAL FEATURE SET (>=3 Methods Consensus, ", length(final_features), " features):", sep = ""),
  paste(final_features, collapse = ", "),
  "",
  "--- Model Diagnostics ---",
  if (exists("final_cv_auc_mean")) paste("  10-fold CV AUC:", round(final_cv_auc_mean, 4), 
                                         "(SD =", round(final_cv_auc_sd, 4), ")") else "  CV AUC: not computed",
  if (exists("auc_final")) paste("  Apparent AUC (C-statistic):", round(auc_final, 4)) else "  Apparent AUC: not computed",
  if (exists("hl_test")) paste("  Hosmer-Lemeshow: chi-sq =", round(hl_test$statistic, 3),
                                ", df =", hl_test$parameter,
                                ", p =", round(hl_test$p.value, 4)) else "  HL test: not computed",
  if (exists("vif_values") && !is.null(vif_values)) {
    c("  VIF (Variance Inflation Factor):",
      paste("    ", names(vif_values), "=", round(vif_values, 2)))
  } else "  VIF: not computed",
  if (exists("high_vif") && length(high_vif) > 0) 
    paste("  WARNING: High VIF (>10) in:", paste(names(high_vif), collapse = ", ")) else "",
  "========================================",
  "Selection Rule (>=3 Methods Majority Voting):",
  "  1. Features selected by >=3 out of 4 methods (majority voting)",
  "  2. If empty -> >=2 methods",
  "  3. If still empty -> Exhaustive AIC (strongest statistical support)",
  "========================================"
)

# 低危修复5：summary 文件路径定义为变量，供第 9 节追加验证队列特征删减记录及末尾追加 sessionInfo
summary_file <- "feature_selection_summary.txt"
writeLines(summary_lines, summary_file)

print(summary_lines)

# ===================== 9. Apply to Validation Cohorts =====================
# 桥接模式跳过本节（折内重放不需要改写任何验证队列文件）
if (!BRIDGE_MODE) {
print("\nApplying feature selection to validation cohorts (no data leakage)...")

if (length(final_features) > 0) {
  # Derivation cohort - keep original filename
  selected_data <- data[, c(id_col, label_col, match(final_features, colnames(data)))]
  write.csv(selected_data, "Derivation cohort.csv", row.names = FALSE)
  
  # FIX: Use tryCatch for error handling when loading validation files
  # Temporal external validation
  # 低危修复5：致命缺失（缺失特征超过一半）信息先存入变量，在 tryCatch 之外 stop，
  # 避免被下方 error 处理函数吞掉而静默继续
  fatal_missing_temporal <- NULL
  tryCatch({
    temporal_data <- read.csv(file.path(INPUT_DIR, "Temporal external validation cohort.csv"), header = TRUE)

    # FIX: Check feature matching
    missing_in_temporal <- setdiff(final_features, colnames(temporal_data))
    if (length(missing_in_temporal) > 0) {
      if (length(missing_in_temporal) > length(final_features) / 2) {
        # 低危修复5：缺失特征超过最终特征数的一半，验证已无意义，记录致命信息（在 tryCatch 外中止）
        fatal_missing_temporal <- paste0(
          "Temporal external validation cohort 缺失特征数 (", length(missing_in_temporal),
          ") 超过最终特征数 (", length(final_features), ") 的一半，流程中止。缺失特征: ",
          paste(missing_in_temporal, collapse = ", "))
      } else {
        warning(paste("Missing features in Temporal cohort:", paste(missing_in_temporal, collapse = ", ")))
        final_features_temporal <- setdiff(final_features, missing_in_temporal)
        # 低危修复5：删减事件追加写入 summary 文件，明确记录队列名与被删减特征名
        sink(summary_file, append = TRUE)
        cat("\n--- Validation Cohort Feature Dropping (验证队列特征删减记录) ---\n")
        cat("Cohort (队列): Temporal external validation cohort\n")
        cat("Dropped features (被删减特征):", paste(missing_in_temporal, collapse = ", "), "\n")
        cat("Retained features (保留特征):", paste(final_features_temporal, collapse = ", "), "\n")
        sink()
      }
    } else {
      final_features_temporal <- final_features
    }

    if (is.null(fatal_missing_temporal)) {
      temporal_selected <- temporal_data[, c(id_col, label_col, match(final_features_temporal, colnames(temporal_data)))]
      write.csv(temporal_selected, "Temporal external validation cohort.csv", row.names = FALSE)
      print(paste("  Temporal external validation: ", nrow(temporal_selected), "samples, ",
                  ncol(temporal_selected) - 2, "features"))
      print(paste("    Saved:", file.path(OUTPUT_DIR, "Temporal external validation cohort.csv")))
    }
  }, error = function(e) {
    warning(paste("Could not load Temporal validation cohort:", e$message))
    print("  Skipping Temporal external validation")
  })
  # 低危修复5：缺失特征超过一半时在此真正中止流程
  if (!is.null(fatal_missing_temporal)) stop(fatal_missing_temporal)
  
  # Geographical external validation
  # 低危修复5：致命缺失（缺失特征超过一半）信息先存入变量，在 tryCatch 之外 stop，
  # 避免被下方 error 处理函数吞掉而静默继续
  fatal_missing_geo <- NULL
  tryCatch({
    geo_data <- read.csv(file.path(INPUT_DIR, "Geographical external validation cohort.csv"), header = TRUE)

    # FIX: Check feature matching
    missing_in_geo <- setdiff(final_features, colnames(geo_data))
    if (length(missing_in_geo) > 0) {
      if (length(missing_in_geo) > length(final_features) / 2) {
        # 低危修复5：缺失特征超过最终特征数的一半，验证已无意义，记录致命信息（在 tryCatch 外中止）
        fatal_missing_geo <- paste0(
          "Geographical external validation cohort 缺失特征数 (", length(missing_in_geo),
          ") 超过最终特征数 (", length(final_features), ") 的一半，流程中止。缺失特征: ",
          paste(missing_in_geo, collapse = ", "))
      } else {
        warning(paste("Missing features in Geographical cohort:", paste(missing_in_geo, collapse = ", ")))
        final_features_geo <- setdiff(final_features, missing_in_geo)
        # 低危修复5：删减事件追加写入 summary 文件，明确记录队列名与被删减特征名
        sink(summary_file, append = TRUE)
        cat("\n--- Validation Cohort Feature Dropping (验证队列特征删减记录) ---\n")
        cat("Cohort (队列): Geographical external validation cohort\n")
        cat("Dropped features (被删减特征):", paste(missing_in_geo, collapse = ", "), "\n")
        cat("Retained features (保留特征):", paste(final_features_geo, collapse = ", "), "\n")
        sink()
      }
    } else {
      final_features_geo <- final_features
    }

    if (is.null(fatal_missing_geo)) {
      geo_selected <- geo_data[, c(id_col, label_col, match(final_features_geo, colnames(geo_data)))]
      write.csv(geo_selected, "Geographical external validation cohort.csv", row.names = FALSE)
      print(paste("  Geographical external validation: ", nrow(geo_selected), "samples, ",
                  ncol(geo_selected) - 2, "features"))
      print(paste("    Saved:", file.path(OUTPUT_DIR, "Geographical external validation cohort.csv")))
    }
  }, error = function(e) {
    warning(paste("Could not load Geographical validation cohort:", e$message))
    print("  Skipping Geographical external validation")
  })
  # 低危修复5：缺失特征超过一半时在此真正中止流程
  if (!is.null(fatal_missing_geo)) stop(fatal_missing_geo)
  
  print("\nFeature selection completed successfully for all available cohorts!")
} else {
  print("No final features selected. Cannot apply to validation cohorts.")
}
}  # end if (!BRIDGE_MODE) — 第 9 节在桥接模式下整体跳过

# 低危修复2：在 summary 文件末尾追加 sessionInfo() 输出，记录运行环境以保证结果可复现
sink(summary_file, append = TRUE)
cat("\n========================================\n")
cat("Session Info (运行环境记录，用于结果复现)\n")
cat("========================================\n")
print(sessionInfo())
sink()

# ===================== 10. Bridge Output（桥接模式写出共识特征） =====================
if (BRIDGE_MODE) {
  if (length(final_features) == 0) {
    stop("BRIDGE: 最终共识特征为空，无法向调用方返回结果")
  }
  writeLines(final_features, BRIDGE_OUTPUT)
  cat("BRIDGE: final features written to ", BRIDGE_OUTPUT, "\n", sep = "")
}

print("\n========================================")
print(paste("All tasks completed. Results saved to", OUTPUT_DIR))
print("========================================")