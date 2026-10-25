# =============================================================================
# 07_Prospective_Validation.R — 前瞻队列外部验证（R 实现，风格参照 05/06）
# =============================================================================
# 做法由 temporal_mode.txt（Python 07 写出）决定，两种策略二选一：
#   [calibration，默认] 前瞻集先按模型原始概率算一遍（Raw），再用时间集（Temporal）
#                       拟合校准器（Platt / 逻辑回归）后算第二遍（Recalibrated）。
#   [merge]             Derivation-only 模型（重训前） vs 并入 Temporal 重训后的模型
#                       （重训后）两套结果对比；重训后不再做时间集重校准。
#
# 数据流：
#   - 模型对 Derivation / Temporal / Prospective 的概率由建模流程（Python 07）导出为 CSV；
#     merge 模式下额外导出 probsp_retrained（重训后模型概率）。
#   - 可视化风格参照 05_Model_Development.py（Part 2 成长/亚组）与 06_Clinical_Plots_R.R。
#
# 输入（Prospective Validation/）：
#   - temporal_mode.txt             (calibration | merge)
#   - prospective_predictions.csv   (label, probsp_raw[, probsp_retrained])
#   - temporal_predictions.csv      (label, probsp_raw[, probsp_retrained])
#   - derivation_predictions.csv    (label, probsp_raw[, probsp_retrained])
#   - Model Selection/model_selection_metrics.csv          —— 推荐模型 + 原始最优截断值
# 输出：
#   - calibration 模式 → Prospective Validation/R_outputs/
#   - merge 模式       → Prospective Validation/R_outputs_Temporal_Merged/
#   - 表格：Prospective_Validation_Summary.csv, Prospective_HosmerLemeshow.csv,
#           Prospective_Cutoff_Sensitivity.csv, Prospective_Risk_Stratification.csv
#   - 图（各 TIFF+PDF+PNG）：Prospective_ROC_AUPRC_*（ROC + PR 双图，merge 模式含两条曲线）,
#           Prospective_Calibration_Curves_*, Prospective_DCA_Curves_*,
#           Prospective_ConfusionMatrix*, Prospective_RiskStratification*,
#           Prospective_Probability_Distribution_*,
#           Prospective_Comparison_*（两套结果对比拼图：校准曲线 + 决策曲线）
#   - 报告：Prospective_Report.txt
# =============================================================================

# --- 1. 加载/安装必需包 ---
required_pkgs <- c("ggplot2", "dplyr", "tidyr", "pROC", "scales", "grid", "gridExtra", "patchwork")
for (pkg in required_pkgs) {
  if (!suppressWarnings(require(pkg, character.only = TRUE, quietly = TRUE))) {
    suppressWarnings(install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE))
    if (!suppressWarnings(require(pkg, character.only = TRUE, quietly = TRUE))) {
      stop(sprintf("R 包 %s 自动安装/加载失败，请手动执行 install.packages(\"%s\") 后重跑。", pkg, pkg))
    }
  }
}
options(bitmapType = "cairo")
set.seed(123)

# =============================================================================
# --- 2. 全局参数 ---
# =============================================================================
COL_CAL   <- "#d62728"   # 校准后（红）
COL_RAW   <- "#1f77b4"   # 原始概率（蓝）
COL_RED   <- "#d62728"
COL_GEO   <- "#2ca02c"
COL_DERIV <- "#1f77b4"
COL_TEMP  <- "#ff7f0e"
COL_GRAY  <- "gray50"

FONT_TITLE  <- 14
FONT_LABEL  <- 12
FONT_LEGEND <- 10
FONT_TICK   <- 10

# --- 路径解析（与 07 一致：由 --file= 推导项目根目录） ---
cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE))
  BASE_DIR <- dirname(script_dir)
} else {
  warning("无法从 commandArgs 解析脚本路径（--file= 缺失），回退使用 dirname(getwd())。")
  BASE_DIR <- dirname(getwd())
}
MODEL_SEL_DIR <- file.path(BASE_DIR, "Model Selection")
PROSP_DIR     <- file.path(BASE_DIR, "Prospective Validation")

# --- Temporal 策略开关（由 Python 07 写入 temporal_mode.txt） ---
#   calibration : Temporal 仅用于拟合重校准器（默认）
#   merge       : Temporal 并入训练集重训模型（model revision）
mode_file <- file.path(PROSP_DIR, "temporal_mode.txt")
TEMPORAL_MODE <- if (file.exists(mode_file)) trimws(readLines(mode_file, warn = FALSE)[1]) else "calibration"
if (!TEMPORAL_MODE %in% c("calibration", "merge")) TEMPORAL_MODE <- "calibration"
cat(sprintf(">>> Temporal strategy: %s\n", TEMPORAL_MODE))

OUTPUT_DIR <- if (TEMPORAL_MODE == "merge") file.path(PROSP_DIR, "R_outputs_Temporal_Merged") else file.path(PROSP_DIR, "R_outputs")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# --- 关键输入校验 ---
for (fn in c("prospective_predictions.csv", "temporal_predictions.csv", "derivation_predictions.csv")) {
  fp <- file.path(PROSP_DIR, fn)
  if (!file.exists(fp)) stop("缺少输入文件（请先运行建模流程生成）：", fp)
}

# =============================================================================
# --- 3. 统一保存函数（沿用 07） ---
# =============================================================================
save_sci_plot <- function(plot_obj, base_name, width, height, dpi_type = "color") {
  dpi_val <- ifelse(dpi_type == "line", 1200, 600)
  base_path <- file.path(OUTPUT_DIR, base_name)
  cat(sprintf("  Saving to: %s\n", base_path))
  safe_print <- function() {
    on.exit({ if (dev.cur() > 1) dev.off() }, add = TRUE)
    print(plot_obj)
  }
  tryCatch({
    tiff(paste0(base_path, ".tif"), width = width, height = height,
         units = "in", res = dpi_val, compression = "lzw", family = "sans")
    safe_print(); cat(sprintf("  -> %s.tif saved\n", base_path))
  }, error = function(e) { if (dev.cur() > 1) dev.off(); cat(sprintf("  ERROR TIFF: %s\n", e$message)) })
  tryCatch({
    pdf(paste0(base_path, ".pdf"), width = width, height = height, family = "sans")
    safe_print(); cat(sprintf("  -> %s.pdf saved\n", base_path))
  }, error = function(e) { if (dev.cur() > 1) dev.off(); cat(sprintf("  ERROR PDF: %s\n", e$message)) })
  tryCatch({
    png(paste0(base_path, ".png"), width = width, height = height,
        units = "in", res = dpi_val, family = "sans")
    safe_print(); cat(sprintf("  -> %s.png saved\n", base_path))
  }, error = function(e) { if (dev.cur() > 1) dev.off(); cat(sprintf("  ERROR PNG: %s\n", e$message)) })
  cat(sprintf("  Done: %s.* (%.0f DPI)\n", base_path, dpi_val))
}

theme_sci <- function() {
  theme_bw(base_size = FONT_LABEL, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      axis.title = element_text(size = FONT_LABEL),
      axis.text = element_text(size = FONT_TICK),
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
      legend.text = element_text(size = FONT_LEGEND - 1),
      legend.key.size = unit(0.5, "cm"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3, color = "gray90"),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5)
    )
}

# =============================================================================
# --- 4. 读取模型名 & 数据 ---
# =============================================================================
read_model_name <- function() {
  p <- file.path(MODEL_SEL_DIR, "model_selection_metrics.csv")
  if (!file.exists(p)) { cat("  WARNING: model_selection_metrics.csv 缺失，默认 MLP\n"); return("MLP") }
  df <- read.csv(p, stringsAsFactors = FALSE)
  best <- df[which.max(df$AB_Score), ]
  cat(sprintf(">>> Recommended model: %s (AB_Score=%.4f)\n", best$Model, best$AB_Score))
  best$Model
}
model_name <- read_model_name()
cat(sprintf("Model: %s\n", model_name))

pred  <- read.csv(file.path(PROSP_DIR, "prospective_predictions.csv"), stringsAsFactors = FALSE)
tempd <- read.csv(file.path(PROSP_DIR, "temporal_predictions.csv"), stringsAsFactors = FALSE)
derid <- read.csv(file.path(PROSP_DIR, "derivation_predictions.csv"), stringsAsFactors = FALSE)

y     <- as.integer(pred$label)
p_raw <- as.numeric(pred$probsp_raw)
y_t   <- as.integer(tempd$label)
p_t   <- as.numeric(tempd$probsp_raw)
y_d   <- as.integer(derid$label)
p_d   <- as.numeric(derid$probsp_raw)

# 原始最优截断值（来自 04）
cutoff_raw <- 0.5
ms_path <- file.path(MODEL_SEL_DIR, "model_selection_metrics.csv")
if (file.exists(ms_path)) {
  ms <- read.csv(ms_path, stringsAsFactors = FALSE)
  row <- ms[ms$Model == model_name, ]
  if (nrow(row) > 0) cutoff_raw <- as.numeric(row$Optimal_Cutoff[1])
}

# --- 两套结果的命名与数据来源（随 Temporal 策略） ---
if (TEMPORAL_MODE == "merge") {
  if (!"probsp_retrained" %in% names(pred))
    stop("merge 模式需要 prospective_predictions.csv 含 probsp_retrained 列；请用 --temporal-mode merge 重跑 Python 07。")
  if (!"probsp_retrained" %in% names(derid))
    stop("merge 模式需要 derivation_predictions.csv 含 probsp_retrained 列；请用 --temporal-mode merge 重跑 Python 07。")
  LAB_A_LEG <- "Raw"
  LAB_B_LEG <- "Recalibrated"
  LAB_A_SHORT <- "Raw"
  LAB_B_SHORT <- "Recalibrated"
  STATUS_A  <- "Before retraining (Derivation only)"
  STATUS_B  <- "After retraining (Derivation + Temporal)"
  COMP_TITLE <- "Before vs After retraining"
  SFX_A <- "_Before_Retraining"
  SFX_B <- "_After_Retraining"
} else {
  LAB_A_LEG <- "Without calibration"
  LAB_B_LEG <- "With calibration"
  LAB_A_SHORT <- "Without calibration"
  LAB_B_SHORT <- "With calibration"
  STATUS_A  <- "Raw"
  STATUS_B  <- "Recalibrated (Temporal)"
  COMP_TITLE <- "Without vs With calibration"
  SFX_A <- "_Pre_Calibration"
  SFX_B <- ""
}

# =============================================================================
# --- 5. 生成前瞻两套概率 ---
#   calibration 模式：在 R 中拟合 Platt 校准器（时间集）→ p_cal_vals
#   merge 模式      ：第二套 = 并入 Temporal 重训后模型的原始概率
# =============================================================================
if (TEMPORAL_MODE == "merge") {
  cat("\n=== Recalibration skipped (Temporal is part of the training set) ===\n")
  qlogis_safe <- function(p, eps = 1e-6) qlogis(pmin(pmax(p, eps), 1 - eps))
  cal_a <- NA_real_; cal_b <- NA_real_
  p_cal_vals   <- as.numeric(pred$probsp_retrained)
  p_d_cal_vals <- as.numeric(derid$probsp_retrained)
  cutoff_cal   <- cutoff_raw
  cat(sprintf("  Cutoff applied to both models: %.4f (Derivation optimal)\n", cutoff_raw))
  cat(sprintf("  Prospective n=%d (events=%d, prevalence=%.3f)\n", length(y), sum(y), mean(y)))
} else {
  cat("\n=== Fitting recalibrator on Temporal (in R) ===\n")
  qlogis_safe <- function(p, eps = 1e-6) qlogis(pmin(pmax(p, eps), 1 - eps))
  fit_cal <- glm(y_t ~ qlogis_safe(p_t), family = binomial())
  cal_a <- unname(coef(fit_cal)[2])   # 斜率（logit 尺度）
  cal_b <- unname(coef(fit_cal)[1])   # 截距
  cat(sprintf("  Calibrator: slope=%.4f, intercept=%.4f\n", cal_a, cal_b))

  p_cal_vals   <- plogis(cal_a * qlogis_safe(p_raw) + cal_b)   # 前瞻校准概率
  cutoff_cal   <- plogis(cal_a * qlogis_safe(cutoff_raw) + cal_b)
  p_d_cal_vals <- plogis(cal_a * qlogis_safe(p_d) + cal_b)     # Derivation 校准概率（分层用）
  cat(sprintf("  Cutoff: raw=%.4f -> recalibrated=%.4f\n", cutoff_raw, cutoff_cal))
  cat(sprintf("  Prospective n=%d (events=%d, prevalence=%.3f)\n", length(y), sum(y), mean(y)))
}

# =============================================================================
# --- 6. 指标计算 ---
# =============================================================================
auc_ci <- function(y, p) {
  r <- pROC::roc(y, p, quiet = TRUE)
  ci <- as.numeric(pROC::ci.auc(r, method = "delong"))
  list(auc = as.numeric(pROC::auc(r)), lo = ci[1], hi = ci[3])
}
auprc <- function(y, p) {
  o <- order(p, decreasing = TRUE); ys <- y[o]
  tps <- cumsum(ys); fps <- cumsum(1 - ys)
  prec <- tps / (tps + fps); rec <- tps / sum(ys)
  rec <- c(0, rec); prec <- c(1, prec)
  sum(diff(rec) * prec[-1])   # 与 sklearn average_precision_score 口径一致
}
calib_slope_intercept <- function(y, p) {
  fit <- glm(y ~ qlogis_safe(p), family = binomial())
  c(slope = unname(coef(fit)[2]), intercept = unname(coef(fit)[1]))
}
hl_test <- function(y, p, g = 10) {
  br <- unique(quantile(p, probs = seq(0, 1, length.out = g + 1), na.rm = TRUE))
  if (length(br) < 3) return(c(chi2 = NA, df = NA, p = NA))
  bin <- cut(p, breaks = br, include.lowest = TRUE, labels = FALSE)
  agg <- aggregate(cbind(obs = y, exp = p, n = 1), by = list(bin = bin), FUN = sum)
  agg <- agg[!is.na(agg$bin), ]
  eps <- 1e-10
  chi2 <- sum((agg$obs - agg$exp)^2 / (agg$exp + eps) +
              ((agg$n - agg$obs) - (agg$n - agg$exp))^2 / ((agg$n - agg$exp) + eps))
  df <- nrow(agg) - 2
  pv <- if (df > 0) 1 - pchisq(chi2, df) else NA
  c(chi2 = chi2, df = df, p = pv)
}
clinical_metrics <- function(y, p, cutoff) {
  pred <- as.integer(p >= cutoff)
  tp <- sum(pred == 1 & y == 1); tn <- sum(pred == 0 & y == 0)
  fp <- sum(pred == 1 & y == 0); fn <- sum(pred == 0 & y == 1)
  sens <- if ((tp + fn) > 0) tp / (tp + fn) else 0
  spec <- if ((tn + fp) > 0) tn / (tn + fp) else 0
  ppv  <- if ((tp + fp) > 0) tp / (tp + fp) else 0
  npv  <- if ((tn + fn) > 0) tn / (tn + fn) else 0
  acc  <- (tp + tn) / length(y)
  f1   <- if ((ppv + sens) > 0) 2 * ppv * sens / (ppv + sens) else 0
  list(tp = tp, tn = tn, fp = fp, fn = fn, Sens = sens, Spec = spec,
       PPV = ppv, NPV = npv, Acc = acc, F1 = f1)
}
self_cutoff <- function(y, p) {
  r <- pROC::roc(y, p, quiet = TRUE)
  co <- pROC::coords(r, "best", best.method = "youden", ret = "threshold", transpose = FALSE)
  as.numeric(co$threshold[1])
}

metrics_row <- function(y, p, cutoff, calstatus) {
  a <- auc_ci(y, p); ap <- auprc(y, p); cm <- clinical_metrics(y, p, cutoff)
  cs <- calib_slope_intercept(y, p); hl <- hl_test(y, p)
  data.frame(Cohort = "Prospective", Calibration_Status = calstatus,
             N = length(y), Events = sum(y),
             AUC = a$auc, AUC_CI_Lower = a$lo, AUC_CI_Upper = a$hi, AUPRC = ap,
             Brier = mean((y - p)^2),
             Accuracy = cm$Acc, Sensitivity = cm$Sens, Specificity = cm$Spec,
             PPV = cm$PPV, NPV = cm$NPV, F1 = cm$F1,
             Calib_Intercept = cs["intercept"], Calib_Slope = cs["slope"],
             HL_chi2 = hl["chi2"], HL_df = hl["df"], HL_p = hl["p"],
             Cutoff = cutoff, stringsAsFactors = FALSE)
}

cat("\n=== Metrics ===\n")
summary_df <- rbind(
  metrics_row(y, p_cal_vals, cutoff_cal, STATUS_B),
  metrics_row(y, p_raw, cutoff_raw, STATUS_A)
)
rownames(summary_df) <- NULL
write.csv(summary_df, file.path(OUTPUT_DIR, "Prospective_Validation_Summary.csv"), row.names = FALSE)
print(summary_df[, c("Cohort","Calibration_Status","N","Events","AUC","AUPRC","Brier","Calib_Slope")])

# Hosmer-Lemeshow
hl_cal <- hl_test(y, p_cal_vals); hl_raw <- hl_test(y, p_raw)
write.csv(data.frame(
  Cohort = "Prospective",
  Calibration_Status = c(STATUS_A, STATUS_B),
  Chi2 = c(hl_raw["chi2"], hl_cal["chi2"]),
  df = c(hl_raw["df"], hl_cal["df"]),
  P_value = c(hl_raw["p"], hl_cal["p"])
), file.path(OUTPUT_DIR, "Prospective_HosmerLemeshow.csv"), row.names = FALSE)

# 截断值敏感性（两套）
cm_raw <- clinical_metrics(y, p_raw, cutoff_raw)
cm_cal <- clinical_metrics(y, p_cal_vals, cutoff_cal)
sc_raw <- self_cutoff(y, p_raw); sc_cal <- self_cutoff(y, p_cal_vals)
cm_sc_raw <- clinical_metrics(y, p_raw, sc_raw)
cm_sc_cal <- clinical_metrics(y, p_cal_vals, sc_cal)
write.csv(data.frame(
  Calibration = c(STATUS_A, STATUS_A, STATUS_B, STATUS_B),
  Cutoff_Type = c("Applied (Derivation optimal)", "Self-optimal (Youden)",
                  if (TEMPORAL_MODE == "merge") "Applied (Derivation optimal)" else "Applied (mapped from Derivation optimal)",
                  "Self-optimal (Youden)"),
  Cutoff = c(cutoff_raw, sc_raw, cutoff_cal, sc_cal),
  Sensitivity = c(cm_raw$Sens, cm_sc_raw$Sens, cm_cal$Sens, cm_sc_cal$Sens),
  Specificity = c(cm_raw$Spec, cm_sc_raw$Spec, cm_cal$Spec, cm_sc_cal$Spec)
), file.path(OUTPUT_DIR, "Prospective_Cutoff_Sensitivity.csv"), row.names = FALSE)

# 风险分层（Derivation 三分位；两套）
risk_table <- function(y, p, low_cut, high_cut) {
  grp <- factor(ifelse(p < low_cut, "Low", ifelse(p < high_cut, "Intermediate", "High")),
                levels = c("Low", "Intermediate", "High"))
  data.frame(Risk_Group = levels(grp),
             N = as.integer(table(grp)),
             Events = as.integer(tapply(y, grp, sum)),
             Event_Rate = as.numeric(tapply(y, grp, mean)),
             Mean_Probability = as.numeric(tapply(p, grp, mean)))
}
risk_raw <- risk_table(y, p_raw, quantile(p_d, 1/3), quantile(p_d, 2/3))
risk_cal <- risk_table(y, p_cal_vals, quantile(p_d_cal_vals, 1/3), quantile(p_d_cal_vals, 2/3))
risk_out <- rbind(cbind(Calibration = STATUS_A, risk_raw),
                  cbind(Calibration = STATUS_B, risk_cal))
write.csv(risk_out, file.path(OUTPUT_DIR, "Prospective_Risk_Stratification.csv"), row.names = FALSE)
cat("\n--- Risk stratification (calibrated) ---\n"); print(risk_cal)

# =============================================================================
# --- 7. 绘图（Raw 与 Recalibrated 各一套；参照 06/07 风格） ---
# =============================================================================
cat("\n=== Generating figures ===\n")

# ---- 7.1 ROC（merge 模式：同时画“重训前/后”两条曲线；calibration 模式：单条） ----
roc_coords <- function(y, p) {
  o <- order(p, decreasing = TRUE); ys <- y[o]
  tps <- cumsum(ys); fps <- cumsum(1 - ys)
  data.frame(fpr = c(0, fps / sum(1 - ys)), tpr = c(0, tps / sum(ys)))
}
a_cal <- auc_ci(y, p_cal_vals)
a_raw <- auc_ci(y, p_raw)
if (TEMPORAL_MODE == "merge") {
  roc_df <- rbind(
    transform(roc_coords(y, p_raw),
              Curve = sprintf("%s\n(AUC=%.3f, %.3f-%.3f)", LAB_A_SHORT, a_raw$auc, a_raw$lo, a_raw$hi)),
    transform(roc_coords(y, p_cal_vals),
              Curve = sprintf("%s\n(AUC=%.3f, %.3f-%.3f)", LAB_B_SHORT, a_cal$auc, a_cal$lo, a_cal$hi))
  )
  roc_cols <- setNames(c(COL_RAW, COL_CAL), unique(roc_df$Curve))
} else {
  roc_df <- roc_coords(y, p_cal_vals)
  roc_df$Curve <- sprintf("Prospective AUC=%.3f (%.3f-%.3f)", a_cal$auc, a_cal$lo, a_cal$hi)
  roc_cols <- setNames(COL_CAL, unique(roc_df$Curve))
}
p_roc <- ggplot(roc_df, aes(fpr, tpr, color = Curve)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", linewidth = 0.6) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(name = NULL, values = roc_cols) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "1 - Specificity", y = "Sensitivity", title = "(A) ROC Curve") +
  theme_sci() +
  theme(plot.title = element_text(size = FONT_LABEL, face = "bold", hjust = 0.5),
        legend.position = c(0.98, 0.02), legend.justification = c(1, 0),
        legend.text = element_text(size = FONT_LEGEND - 2),
        legend.key.height = unit(16, "pt"))
cat("[1/7] ROC built (AUC).\n")

# ---- 7.2 校准曲线（07 风格：分位数分箱 + 局部合并），Raw 与 Calibrated 各一张 ----
make_cal_bins <- function(y, p) {
  o <- order(p); p <- p[o]; y <- y[o]
  edges <- unique(quantile(p, probs = seq(0, 1, length.out = 11), na.rm = TRUE))
  repeat {
    bi <- cut(p, breaks = edges, include.lowest = TRUE, labels = FALSE)
    ne <- length(unique(bi[!is.na(bi)]))
    if (ne <= 1) break
    tmp <- data.frame(bin = bi, y = y); tmp <- tmp[!is.na(tmp$bin), ]
    agg <- aggregate(y ~ bin, data = tmp, FUN = function(v) c(n_pos = sum(v), n_bin = length(v)))
    n_pos <- agg$y[, 1]; n_bin <- agg$y[, 2]
    bad <- which(n_pos < 5)
    if (length(bad) == 0) break
    mt <- bad[which.min(n_pos[bad])]
    nb <- c(mt - 1, mt + 1); nb <- nb[nb >= 1 & nb <= ne]
    if (length(nb) == 0) break
    ns <- nb[which.min(n_bin[nb])]
    lo_bin <- min(mt, ns)
    if ((lo_bin + 1) <= length(edges)) edges <- edges[-(lo_bin + 1)] else break
  }
  bin <- cut(p, breaks = edges, include.lowest = TRUE, labels = FALSE)
  d <- data.frame(y = y, p = p, bin = bin); d <- d[!is.na(d$bin), ]
  agg <- aggregate(cbind(obs = y, pp = p) ~ bin, data = d, FUN = mean)
  agg[, c("pp", "obs")]
}
plot_calibration <- function(y, p, hl_p, base_name) {
  bins <- make_cal_bins(y, p)
  brier <- mean((y - p)^2)
  lab <- sprintf("Prospective (Brier=%.3f, HL p=%.3f)", brier, hl_p)
  pb <- ggplot(bins, aes(pp, obs)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", linewidth = 0.7) +
    geom_line(color = COL_CAL, linewidth = 0.9) +
    geom_point(color = COL_CAL, shape = 16, size = 2.6) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(x = "Mean Predicted Probability", y = "Observed Proportion") +
    theme_sci() + theme(legend.position = "none")
  save_sci_plot(pb, base_name, 6.5, 6.0, "color")
}
plot_calibration(y, p_raw, hl_raw["p"], paste0("Prospective_Calibration_Curves", SFX_A, "_", model_name))
plot_calibration(y, p_cal_vals, hl_cal["p"], paste0("Prospective_Calibration_Curves", SFX_B, "_", model_name))
cat("[2/7] Calibration curves (set 1 + set 2) done.\n")

# ---- 7.3 DCA（07 风格），Raw 与 Calibrated 各一张 ----
nb_curve <- function(y, p, thr) {
  n <- length(y)
  sapply(thr, function(t) { pr <- p >= t
    sum(pr & y == 1) / n - (sum(pr & y == 0) / n) * (t / (1 - t)) })
}
plot_dca <- function(y, p, base_name) {
  thr <- seq(0.01, 0.80, by = 0.01); prev <- mean(y)
  nb <- nb_curve(y, p, thr); nb_all <- prev - (1 - prev) * (thr / (1 - thr))
  ddf <- rbind(
    data.frame(Threshold = thr, Net_Benefit = nb, Strategy = "Model"),
    data.frame(Threshold = thr, Net_Benefit = 0, Strategy = "Treat None"),
    data.frame(Threshold = thr, Net_Benefit = nb_all, Strategy = "Treat All"),
    data.frame(Threshold = thr, Net_Benefit = prev, Strategy = "Ideal (perfect)")
  )
  ddf$Strategy <- factor(ddf$Strategy, levels = c("Ideal (perfect)", "Treat None", "Treat All", "Model"))
  cols <- c("Ideal (perfect)" = "darkred", "Treat None" = "black", "Treat All" = "gray50", "Model" = COL_CAL)
  lt <- c("Ideal (perfect)" = "dotted", "Treat None" = "dashed", "Treat All" = "dashed", "Model" = "solid")
  reg <- data.frame(Threshold = thr, ymin = nb_all, ymax = pmax(nb, nb_all))
  y_hi <- max(max(nb), prev) * 1.05 + 0.02
  pd <- ggplot() +
    geom_ribbon(data = reg, aes(Threshold, ymin = ymin, ymax = ymax), fill = COL_CAL, alpha = 0.15) +
    geom_line(data = ddf, aes(Threshold, Net_Benefit, color = Strategy, linetype = Strategy, linewidth = Strategy)) +
    scale_color_manual(name = NULL, values = cols, breaks = names(cols)) +
    scale_linetype_manual(name = NULL, values = lt) +
    scale_linewidth_manual(name = NULL, values = c("Ideal (perfect)" = 0.7, "Treat None" = 0.5,
                                                   "Treat All" = 0.8, "Model" = 1.2)) +
    scale_x_continuous(limits = c(0, 0.80), breaks = seq(0, 0.8, by = 0.1)) +
    coord_cartesian(ylim = c(-0.15, y_hi)) +
    labs(x = "Threshold Probability", y = "Net Benefit") +
    guides(color = guide_legend(override.aes = list(linewidth = 1.2)), linetype = "none", linewidth = "none") +
    theme_sci() + theme(legend.position = c(0.80, 0.80), legend.justification = c(0.5, 0.5))
  save_sci_plot(pd, base_name, 7.0, 6.0, "line")
}
plot_dca(y, p_raw, paste0("Prospective_DCA_Curves", SFX_A, "_", model_name))
plot_dca(y, p_cal_vals, paste0("Prospective_DCA_Curves", SFX_B, "_", model_name))
cat("[3/7] DCA curves (set 1 + set 2) done.\n")

# ---- 7.4 混淆矩阵（Raw 与 Calibrated 各一张） ----
plot_cm <- function(y, p, cutoff, base_name) {
  pred <- as.integer(p >= cutoff)
  tab <- as.data.frame(table(
    Predicted = factor(pred, levels = c(1, 0), labels = c("Pred Pos", "Pred Neg")),
    Reference = factor(y, levels = c(1, 0), labels = c("True Pos", "True Neg"))))
  pcm <- ggplot(tab, aes(Reference, Predicted, fill = Freq)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = Freq), size = 6, fontface = "bold",
              color = ifelse(tab$Freq > max(tab$Freq) / 2, "white", "black")) +
    scale_fill_gradient(low = "#D6E4F0", high = "#1f4e79", guide = "none") +
    labs(x = NULL, y = NULL, title = sprintf("Confusion Matrix (cutoff=%.3f)", cutoff)) +
    theme_minimal(base_size = FONT_LABEL, base_family = "sans") +
    theme(plot.title = element_text(size = FONT_TITLE - 2, hjust = 0.5))
  save_sci_plot(pcm, base_name, 5.5, 4.5, "color")
}
plot_cm(y, p_raw, cutoff_raw, paste0("Prospective_ConfusionMatrix", SFX_A))
plot_cm(y, p_cal_vals, cutoff_cal, paste0("Prospective_ConfusionMatrix", SFX_B))
cat("[4/7] Confusion matrices (raw + calibrated) done.\n")

# ---- 7.5 风险分层（Raw 与 Calibrated 各一张；参照 06 风格） ----
plot_risk <- function(rt, base_name) {
  rt$Risk_Group <- factor(rt$Risk_Group, levels = c("Low", "Intermediate", "High"))
  pr <- ggplot(rt, aes(Risk_Group, Event_Rate, fill = Risk_Group)) +
    geom_col(width = 0.62, alpha = 0.9, color = "gray30") +
    geom_text(aes(label = sprintf("%.1f%%\n(n=%d)", Event_Rate * 100, N)), vjust = -0.25, size = 3.4) +
    scale_fill_manual(values = c("Low" = COL_GEO, "Intermediate" = COL_TEMP, "High" = COL_RED)) +
    scale_y_continuous(limits = c(0, max(rt$Event_Rate) * 1.35 + 0.03), labels = scales::label_percent(accuracy = 1)) +
    labs(x = NULL, y = "Observed Event Rate") +
    theme_sci() + theme(legend.position = "none")
  save_sci_plot(pr, base_name, 6.5, 5.0, "color")
}
plot_risk(risk_raw, paste0("Prospective_RiskStratification", SFX_A))
plot_risk(risk_cal, paste0("Prospective_RiskStratification", SFX_B))
cat("[5/7] Risk stratification (raw + calibrated) done.\n")

# ---- 7.6 PR 曲线（merge 模式：两条曲线；calibration 模式：单条；与 ROC 融合为双图） ----
pr_coords <- function(y, p) {
  o <- order(p, decreasing = TRUE); ys <- y[o]
  tps <- cumsum(ys); fps <- cumsum(1 - ys)
  data.frame(Recall = c(0, tps / sum(ys)), Precision = c(1, tps / (tps + fps)))
}
ap_cal <- auprc(y, p_cal_vals); ap_raw <- auprc(y, p_raw)
if (TEMPORAL_MODE == "merge") {
  pr_df <- rbind(
    transform(pr_coords(y, p_raw), Curve = sprintf("%s\n(AUPRC=%.3f)", LAB_A_SHORT, ap_raw)),
    transform(pr_coords(y, p_cal_vals), Curve = sprintf("%s\n(AUPRC=%.3f)", LAB_B_SHORT, ap_cal))
  )
  pr_cols <- setNames(c(COL_RAW, COL_CAL), unique(pr_df$Curve))
  pr_title <- "(B) Precision-Recall"
  p_pr <- ggplot(pr_df, aes(Recall, Precision, color = Curve)) +
    geom_line(linewidth = 1.2) +
    scale_color_manual(name = NULL, values = pr_cols) +
    theme(legend.text = element_text(size = FONT_LEGEND - 2),
          legend.key.height = unit(16, "pt"))
} else {
  pr_df <- pr_coords(y, p_cal_vals)
  pr_title <- sprintf("(B) Precision-Recall (AUPRC = %.3f)", ap_cal)
  p_pr <- ggplot(pr_df, aes(Recall, Precision)) +
    geom_line(color = COL_CAL, linewidth = 1.2)
}
p_pr <- p_pr +
  geom_hline(yintercept = mean(y), linetype = "dashed", color = "gray40", linewidth = 0.7) +
  annotate("text", x = 0.98, y = mean(y) + 0.03, hjust = 1, size = 3.4, color = "gray30",
           label = sprintf("Baseline = %.3f", mean(y))) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Recall (Sensitivity)", y = "Precision (PPV)", title = pr_title) +
  theme_sci() + theme(plot.title = element_text(size = FONT_LABEL, face = "bold", hjust = 0.5))

# ---- 7.6b ROC (AUC) + PR (AUPRC) 融合为双图 ----
disc <- p_roc + p_pr
save_sci_plot(disc, paste0("Prospective_ROC_AUPRC_", model_name), 9.5, 5.5, "color")
cat("[6/7] ROC + PR (AUC + AUPRC) combined figure done.\n")

# ---- 7.7 概率分布（Raw 与 Calibrated 各一张） ----
plot_dist <- function(y, p, cutoff, base_name) {
  d <- data.frame(prob = p, Outcome = factor(y, levels = c(0, 1), labels = c("No Event", "Event")))
  pdist <- ggplot(d, aes(prob, fill = Outcome)) +
    geom_histogram(aes(y = after_stat(density)), bins = 22, alpha = 0.55, color = "white", position = "identity") +
    geom_vline(xintercept = cutoff, linetype = "dashed", color = "black", linewidth = 0.8) +
    scale_fill_manual(values = c("No Event" = COL_RAW, "Event" = COL_CAL), name = NULL) +
    labs(x = "Predicted Probability", y = "Density") +
    theme_sci() + theme(legend.position = c(0.80, 0.85))
  save_sci_plot(pdist, base_name, 6.5, 5.0, "color")
}
plot_dist(y, p_raw, cutoff_raw, paste0("Prospective_Probability_Distribution", SFX_A))
plot_dist(y, p_cal_vals, cutoff_cal, paste0("Prospective_Probability_Distribution", SFX_B))
cat("[7/7] Probability distribution (raw + calibrated) done.\n")

# ---- 7.8 两套结果对比拼图（校准曲线 + 决策曲线） ----
cat(sprintf("[7b] Comparison composite (%s)...\n", COMP_TITLE))
bins_raw <- make_cal_bins(y, p_raw); bins_raw$Set <- LAB_A_LEG
bins_cal <- make_cal_bins(y, p_cal_vals); bins_cal$Set <- LAB_B_LEG
bins_all <- rbind(bins_raw, bins_cal)
bins_all$Set <- factor(bins_all$Set, levels = c(LAB_A_LEG, LAB_B_LEG))
lab_map <- setNames(
  c(sprintf("%s (Brier=%.3f, HL p=%.3f)", LAB_A_LEG, mean((y - p_raw)^2), hl_raw["p"]),
    sprintf("%s (Brier=%.3f, HL p=%.3f)", LAB_B_LEG, mean((y - p_cal_vals)^2), hl_cal["p"])),
  c(LAB_A_LEG, LAB_B_LEG)
)
p_comp_cal <- ggplot(bins_all, aes(pp, obs, color = Set)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", linewidth = 0.6) +
  geom_line(linewidth = 0.9) + geom_point(shape = 16, size = 2.4) +
  scale_color_manual(name = NULL,
                     values = setNames(c(COL_RAW, COL_CAL), c(LAB_A_LEG, LAB_B_LEG)),
                     labels = function(x) lab_map[x]) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Mean Predicted Probability", y = "Observed Proportion", title = "(A) Calibration") +
  theme_sci() + theme(plot.title = element_text(size = FONT_LABEL, face = "bold"),
                      legend.position = c(0.98, 0.02), legend.justification = c(1, 0),
                      legend.text = element_text(size = FONT_LEGEND - 3))

thr2 <- seq(0.01, 0.80, by = 0.01); prev2 <- mean(y)

# 决策阈值处的净收益（Net Benefit）数值，直接写入图例括号内
op_thr    <- cutoff_cal
nb_raw_op <- nb_curve(y, p_raw, op_thr)
nb_cal_op <- nb_curve(y, p_cal_vals, op_thr)
lvl_raw  <- c("Treat None", "Treat All", LAB_A_LEG, LAB_B_LEG)
lvl_disp <- c("Treat None", "Treat All",
              sprintf("%s (NB=%.3f)", LAB_A_LEG, nb_raw_op),
              sprintf("%s (NB=%.3f)", LAB_B_LEG, nb_cal_op))

ddf2 <- rbind(
  data.frame(Threshold = thr2, Net_Benefit = nb_curve(y, p_raw, thr2), Strategy = LAB_A_LEG),
  data.frame(Threshold = thr2, Net_Benefit = nb_curve(y, p_cal_vals, thr2), Strategy = LAB_B_LEG),
  data.frame(Threshold = thr2, Net_Benefit = 0, Strategy = "Treat None"),
  data.frame(Threshold = thr2, Net_Benefit = prev2 - (1 - prev2) * (thr2 / (1 - thr2)), Strategy = "Treat All")
)
ddf2$Strategy <- factor(ddf2$Strategy, levels = lvl_raw, labels = lvl_disp)
cols2 <- setNames(c("black", "gray50", COL_RAW, COL_CAL), lvl_disp)
lt2   <- setNames(c("dashed", "dashed", "solid", "solid"), lvl_disp)

p_comp_dca <- ggplot(ddf2, aes(Threshold, Net_Benefit, color = Strategy, linetype = Strategy)) +
  geom_line(linewidth = 1.1) +
  geom_vline(xintercept = op_thr, linetype = "dashed", color = "gray45", linewidth = 0.5) +
  annotate("point", x = op_thr, y = nb_raw_op, color = COL_RAW, size = 2.6) +
  annotate("point", x = op_thr, y = nb_cal_op, color = COL_CAL, size = 2.6) +
  scale_color_manual(name = NULL, values = cols2) +
  scale_linetype_manual(name = NULL, values = lt2) +
  scale_x_continuous(limits = c(0, 0.80), breaks = seq(0, 0.8, by = 0.1)) +
  coord_cartesian(ylim = c(-0.15, max(prev2, 0.05) * 1.1 + 0.02)) +
  labs(x = "Threshold Probability", y = "Net Benefit", title = "(B) Decision Curve") +
  theme_sci() + theme(plot.title = element_text(size = FONT_LABEL, face = "bold"),
                      legend.position = c(0.98, 0.98), legend.justification = c(1, 1))
comp <- p_comp_cal + p_comp_dca
comp_name <- if (TEMPORAL_MODE == "merge") {
  paste0("Prospective_Comparison_Before_vs_After_Retraining_", model_name)
} else {
  paste0("Prospective_Comparison_Raw_vs_Recalibrated_", model_name)
}
save_sci_plot(comp, comp_name, 13, 5.5, "color")
cat("[7b] Comparison composite done.\n")

# =============================================================================
# --- 8. 文字报告 ---
# =============================================================================
fmt_m <- function(m) sprintf("AUC=%.3f (%.3f-%.3f), AUPRC=%.3f, Brier=%.4f",
                             m$AUC, m$AUC_CI_Lower, m$AUC_CI_Upper, m$AUPRC, m$Brier)
m_cal <- metrics_row(y, p_cal_vals, cutoff_cal, STATUS_B)
m_raw <- metrics_row(y, p_raw, cutoff_raw, STATUS_A)
rec_line <- if (TEMPORAL_MODE == "merge") {
  "Recalibrator: none (Temporal is part of the training set)"
} else {
  sprintf("Recalibrator (Platt, logit): slope=%.4f, intercept=%.4f (fitted on Temporal n=%d)",
          cal_a, cal_b, length(y_t))
}
cut_line <- if (TEMPORAL_MODE == "merge") {
  sprintf("Cutoff: %.4f (Derivation optimal, applied to both models)", cutoff_raw)
} else {
  sprintf("Cutoff: raw=%.4f | recalibrated=%.4f", cutoff_raw, cutoff_cal)
}
rep_lines <- c(
  "======================================================================",
  "Prospective Cohort External Validation Report (R)",
  "======================================================================",
  sprintf("Temporal strategy: %s", TEMPORAL_MODE),
  sprintf("Model: %s", model_name),
  rec_line,
  cut_line,
  sprintf("Prospective n=%d (events=%d, prevalence=%.3f)", length(y), sum(y), mean(y)),
  "",
  sprintf("=== [Set 1] %s ===", STATUS_A),
  sprintf("  %s", fmt_m(m_raw)),
  sprintf("  Acc=%.3f, Sens=%.3f, Spec=%.3f, PPV=%.3f, NPV=%.3f, F1=%.3f",
          m_raw$Accuracy, m_raw$Sensitivity, m_raw$Specificity, m_raw$PPV, m_raw$NPV, m_raw$F1),
  sprintf("  Calib slope=%.3f, intercept=%.3f | HL chi2=%.3f, p=%.4f",
          m_raw$Calib_Slope, m_raw$Calib_Intercept, m_raw$HL_chi2, m_raw$HL_p),
  "",
  sprintf("=== [Set 2] %s ===", STATUS_B),
  sprintf("  %s", fmt_m(m_cal)),
  sprintf("  Acc=%.3f, Sens=%.3f, Spec=%.3f, PPV=%.3f, NPV=%.3f, F1=%.3f",
          m_cal$Accuracy, m_cal$Sensitivity, m_cal$Specificity, m_cal$PPV, m_cal$NPV, m_cal$F1),
  sprintf("  Calib slope=%.3f, intercept=%.3f | HL chi2=%.3f, p=%.4f",
          m_cal$Calib_Slope, m_cal$Calib_Intercept, m_cal$HL_chi2, m_cal$HL_p),
  sprintf("  Delta AUC=%.4f, Delta AUPRC=%.4f (Set 2 - Set 1)", m_cal$AUC - m_raw$AUC, m_cal$AUPRC - m_raw$AUPRC),
  "",
  sprintf("[Risk stratification — %s]", STATUS_B),
  apply(risk_cal, 1, function(r) sprintf("  %-14s n=%s events=%s rate=%.1f%%",
                                         r["Risk_Group"], r["N"], r["Events"], as.numeric(r["Event_Rate"]) * 100)),
  "",
  sprintf("Outputs: %s", OUTPUT_DIR),
  "======================================================================"
)
writeLines(rep_lines, file.path(OUTPUT_DIR, "Prospective_Report.txt"))
cat(paste(rep_lines, collapse = "\n"), "\n")

cat("\n=== Prospective Validation (R) Complete ===\n")
