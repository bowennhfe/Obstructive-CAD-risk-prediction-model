# =============================================================================
# 06_Clinical_Plots_R.R — SCI标准临床预测模型图表生成（与Python风格一致）
# =============================================================================
# 功能：分 10 步生成临床图表，完全符合医学SCI顶刊出版要求：
#   亚组森林图、风险分层点图、临床影响曲线、校准曲线、决策曲线（DCA）、
#   连续净收益、避免不必要干预数（NIA）。
#   其中校准曲线与 DCA 各含校准前/后双版本（各计 3 步）。
# 输入：各CSV文件（来自 01~06 Python流程）
# 输出：TIFF (LZW) + PDF + PNG，按要求DPI
# =============================================================================

# --- 1. 加载/安装必需包 ---
# 注：Rcpp 已从依赖列表移除（本脚本未直接使用）
required_pkgs <- c(
  "ggplot2", "patchwork", "forestplot", "corrplot", "ggrepel",
  "dplyr", "tidyr", "pROC", "gridExtra", "grid", "scales"
)
for (pkg in required_pkgs) {
  if (!suppressWarnings(require(pkg, character.only = TRUE, quietly = TRUE))) {
    suppressWarnings(install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE))
    # 安装后再次加载确认；仍失败则中止并提示用户手动安装（避免 library() 抛出无包名上下文的报错）
    if (!suppressWarnings(require(pkg, character.only = TRUE, quietly = TRUE))) {
      stop(sprintf("R 包 %s 自动安装/加载失败，请手动执行 install.packages(\"%s\") 后重跑本脚本。",
                   pkg, pkg))
    }
  }
}
options(bitmapType = "cairo")
set.seed(123)

# =============================================================================
# --- 2. 全局参数设置（与Python 05_Model_Development.py 一致） ---
# =============================================================================

# 颜色（与Python palette = ["#1f77b4", "#ff7f0e", "#2ca02c"] 一致）
COL_DERIV  <- "#1f77b4"
COL_TEMP   <- "#ff7f0e"
COL_GEO    <- "#2ca02c"
COL_RED    <- "#d62728"
COL_GRAY   <- "gray50"

# 字体与字号
FONT_TITLE  <- 14
FONT_LABEL  <- 12
FONT_LEGEND <- 10
FONT_TICK   <- 10
FONT_SMALL  <- 8

# 图片输入路径（数据来源）
# 优先从脚本自身路径推导项目根目录（解析 commandArgs 的 --file=，与 setwd/启动目录无关）；
# 解析失败（如交互式会话中 source）再回退 dirname(getwd()) 并警告。
cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1]),
                                      winslash = "/", mustWork = FALSE))
  BASE_DIR <- dirname(script_dir)  # 脚本位于 <项目根>/Code/，其父目录即项目根
} else {
  warning("无法从 commandArgs 解析脚本路径（--file= 缺失），回退使用 dirname(getwd())；请确认当前工作目录为 Code/ 目录。")
  BASE_DIR <- dirname(getwd())  # Code 目录的父目录
}
MODEL_SEL_DIR <- file.path(BASE_DIR, "Model Selection")
MODEL_DEV_DIR <- file.path(BASE_DIR, "Model Development")
SUBGROUP_DIR <- file.path(BASE_DIR, "Subgroup Analysis")
VIF_DIR <- file.path(BASE_DIR, "vif_analysis")
# 07 脚本所有图表统一输出到新文件夹
OUTPUT_DIR <- file.path(BASE_DIR, "Clinical_Plots_R")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# 启动时校验关键输入 CSV 存在性（来自上游 01~06 流程），缺失即中止并给出明确路径
required_inputs <- c(
  file.path(MODEL_SEL_DIR, "model_selection_metrics.csv"),
  file.path(MODEL_DEV_DIR, "risk_stratification_results.csv")
)
missing_inputs <- required_inputs[!file.exists(required_inputs)]
if (length(missing_inputs) > 0) {
  stop("关键输入文件缺失，请先运行上游 01~06 流程生成以下文件：\n  ",
       paste(missing_inputs, collapse = "\n  "))
}

# =============================================================================
# --- 3. 统一保存函数  ---
# =============================================================================
# 自动选择图例位置：将数据区域分为 2x2 网格，选择数据点最少的角落（空白区）
auto_legend_pos <- function(x, y) {
  n <- length(x)
  if (n == 0) return(c(0.85, 0.85))
  mx <- max(x, na.rm = TRUE); mn <- min(x, na.rm = TRUE)
  my <- max(y, na.rm = TRUE); myn <- min(y, na.rm = TRUE)
  xh <- if (mx == mn) mn else (mx + mn) / 2
  yh <- if (my == myn) myn else (my + myn) / 2
  cnt <- c(
    TL = sum(x <= xh & y >= yh, na.rm = TRUE),
    TR = sum(x >  xh & y >= yh, na.rm = TRUE),
    BL = sum(x <= xh & y <  yh, na.rm = TRUE),
    BR = sum(x >  xh & y <  yh, na.rm = TRUE)
  )
  switch(names(which.min(cnt)),
    TL = c(0.15, 0.88), TR = c(0.85, 0.88),
    BL = c(0.15, 0.12), BR = c(0.85, 0.12))
}

save_sci_plot <- function(plot_obj, base_name, width, height, dpi_type = "line") {  # dpi_type: "line" -> 1200 DPI, "color" -> 600 DPI
  dpi_val <- ifelse(dpi_type == "line", 1200, 600)
  out_dir <- OUTPUT_DIR
  base_path <- file.path(out_dir, base_name)
  cat(sprintf("  Saving to: %s\n", base_path))
  
  # 通用安全打印：无论 print(plot_obj) 是否报错，都确保图形设备被关闭
  safe_print <- function() {
    on.exit({ if (dev.cur() > 1) dev.off() }, add = TRUE)
    print(plot_obj)
  }
  
  # TIFF (LZW 压缩)
  tryCatch({
    tiff(paste0(base_path, ".tif"), width = width, height = height,
         units = "in", res = dpi_val, compression = "lzw", family = "sans")
    safe_print()
    cat(sprintf("  -> %s.tif saved\n", base_path))
  }, error = function(e) {
    if (dev.cur() > 1) dev.off()
    cat(sprintf("  ERROR saving TIFF: %s\n", e$message))
  })
  
  # PDF（矢量）
  tryCatch({
    pdf(paste0(base_path, ".pdf"), width = width, height = height, family = "sans")
    safe_print()
    cat(sprintf("  -> %s.pdf saved\n", base_path))
  }, error = function(e) {
    if (dev.cur() > 1) dev.off()
    cat(sprintf("  ERROR saving PDF: %s\n", e$message))
  })
  
  # PNG（预览）
  tryCatch({
    png(paste0(base_path, ".png"), width = width, height = height,
        units = "in", res = dpi_val, family = "sans")
    safe_print()
    cat(sprintf("  -> %s.png saved\n", base_path))
  }, error = function(e) {
    if (dev.cur() > 1) dev.off()
    cat(sprintf("  ERROR saving PNG: %s\n", e$message))
  })
  
  cat(sprintf("  Done: %s.* (%.0f DPI, %s)\n", base_path, dpi_val, dpi_type))
}

# =============================================================================
# --- 4. 读取推荐模型名称 ---
# =============================================================================
read_model_name <- function() {
  metrics_path <- file.path(MODEL_SEL_DIR, "model_selection_metrics.csv")
  if (!file.exists(metrics_path)) {
    stop("Cannot find model_selection_metrics.csv at: ", metrics_path)
  }
  df <- read.csv(metrics_path, stringsAsFactors = FALSE)
  # AB_Score 最高者为推荐模型
  best <- df[which.max(df$AB_Score), ]
  cat(sprintf(">>> Recommended model: %s (AB_Score = %.4f)\n", best$Model, best$AB_Score))
  return(best$Model)
}

# =============================================================================
# --- 5. 绘图函数 ---
# =============================================================================

# ---------------------------------------------------------------
# 5.1 亚组森林图
# ---------------------------------------------------------------
plot_subgroup_forest <- function(model_name) {
  cat("[1/10] 绘制亚组森林图...\n")
  
  data_path <- file.path(SUBGROUP_DIR, paste0("Subgroup_Performance_", model_name, ".csv"))
  if (!file.exists(data_path)) {
    cat("  WARNING: Subgroup_Performance_", model_name, ".csv not found, skipping.\n", sep = "")
    return(NULL)
  }
  
  df <- read.csv(data_path, stringsAsFactors = FALSE)
  
  # CSV 中 Signif_Worse_FDR 存为 "True"/"False"，转为逻辑型以匹配 scale 名称
  df$Signif_Worse_FDR <- as.logical(df$Signif_Worse_FDR)
  
  # 筛选亚组：仅保留 Age:, Gender:, HTN: 开头的
  df <- df[grepl("^Age:|^Gender:|^HTN:", df$Subgroup), ]
  if (nrow(df) < 2) {
    cat("  WARNING: Fewer than 2 subgroups after filtering, skipping.\n")
    return(NULL)
  }
  
  # 解析主类别和水平
  df$Group <- gsub(":.*$", "", df$Subgroup)
  df$Level <- gsub("^.*?:\\s*", "", df$Subgroup)
  
  # 定义主类别顺序
  group_order <- c("Age", "Gender", "HTN")
  df$Group <- factor(df$Group, levels = group_order)

  # 临床意义的水平排序（ESC SCORE2 年龄组；性别 Female→Male；HTN No→Yes），
  # 替代无临床意义的字母降序。decreasing=TRUE 使 y_pos 顶部为第一个水平。
  clin_level_rank <- c(
    "Age: <50" = 1,  "Age: 50-69" = 2, "Age: >=70" = 3,
    "Gender: Female" = 1, "Gender: Male" = 2,
    "HTN: No" = 1, "HTN: Yes" = 2
  )
  df$level_rank <- as.numeric(clin_level_rank[as.character(df$Subgroup)])
  df$level_rank[is.na(df$level_rank)] <- rank(df$Level[is.na(df$level_rank)], ties.method = "first")
  df <- df[order(df$Group, df$level_rank, decreasing = TRUE), ]
  df$y_pos <- nrow(df):1
  
  # 颜色
  df$point_color <- ifelse(df$Signif_Worse_FDR, COL_RED, COL_DERIV)
  
  # ---- 计算 O/E 比率 ----
  overall_prev <- sum(df$N_Pos) / sum(df$N)
  oe_list <- lapply(seq_len(nrow(df)), function(i) {
    n_pos <- df$N_Pos[i]
    n <- df$N[i]
    obs_rate <- n_pos / n
    oe <- obs_rate / overall_prev
    # Wilson CI for obs_rate, then divide by overall_prev
    if (n > 0) {
      z <- qnorm(0.975)
      p_hat <- n_pos / n
      denom <- 1 + z^2 / n
      center <- (p_hat + z^2 / (2 * n)) / denom
      margin <- z * sqrt((p_hat * (1 - p_hat) + z^2 / (4 * n)) / n) / denom
      ci_lower <- (center - margin) / overall_prev
      ci_upper <- (center + margin) / overall_prev
    } else {
      ci_lower <- NA
      ci_upper <- NA
    }
    data.frame(OE = oe, OE_CI_lower = ci_lower, OE_CI_upper = ci_upper)
  })
  oe_df <- bind_rows(oe_list)
  df$OE <- oe_df$OE
  df$OE_CI_lower <- oe_df$OE_CI_lower
  df$OE_CI_upper <- oe_df$OE_CI_upper
  
  # 标签列
  df$N_label <- sprintf("%d/%d", df$N_Pos, df$N)
  # Metric_Value 为 NA 的小样本亚组不打印 "NA (NA–NA)"，改显示样本量不足提示
  df$AUC_label <- ifelse(is.na(df$Metric_Value),
                         "N/A (insufficient sample)",
                         sprintf("%.3f (%.3f–%.3f)", df$Metric_Value, df$CI_Lower, df$CI_Upper))
  df$OE_label <- sprintf("%.2f (%.2f–%.2f)", df$OE, df$OE_CI_lower, df$OE_CI_upper)
  
  # x 轴下限自适应：硬编码 0.3 会静默裁剪 CI 下限更低的低 AUC 亚组
  x_lo <- min(0.3, min(df$CI_Lower, na.rm = TRUE) - 0.02)
  
  # 右侧文本 x 坐标（超出数据范围）
  x_text_start <- 1.03
  x_text_auc  <- 1.20
  x_text_oe   <- 1.42
  
  # 检测主类别变化位置，添加分隔线
  sep_y <- c()
  for (i in 2:nrow(df)) {
    if (as.character(df$Group[i]) != as.character(df$Group[i - 1])) {
      sep_y <- c(sep_y, df$y_pos[i] + 0.5)
    }
  }
  
  # 构建 ggplot
  p <- ggplot(df, aes(x = Metric_Value, y = y_pos)) +
    # 参考线
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray50", linewidth = 0.6) +
    # 类别分隔线
    geom_segment(data = data.frame(y = sep_y),
                 aes(x = -Inf, xend = Inf, y = y, yend = y),
                 linetype = "dotted", color = "gray70", linewidth = 0.4, inherit.aes = FALSE) +
    # 误差线：geom_errorbarh 已于 ggplot2 4.0 弃用；geom_errorbar(orientation="x")
    # 在 4.0.3 存在 bug，故改用 geom_linerange（无端帽的水平误差线，经验证可用）
    geom_linerange(aes(xmin = CI_Lower, xmax = CI_Upper),
                   linewidth = 0.8, color = df$point_color) +
    # 点
    geom_point(aes(color = Signif_Worse_FDR), size = 3) +
    scale_color_manual(values = c("FALSE" = COL_DERIV, "TRUE" = COL_RED), guide = "none") +
    # 右侧文本：N (Events)
    geom_text(aes(x = x_text_start, label = N_label),
              hjust = 0, size = 2.8, family = "sans", color = "black") +
    # 右侧文本：AUC (95% CI)
    geom_text(aes(x = x_text_auc, label = AUC_label),
              hjust = 0, size = 2.8, family = "sans", color = "black") +
    # 右侧文本：O/E (95% CI)
    geom_text(aes(x = x_text_oe, label = OE_label),
              hjust = 0, size = 2.8, family = "sans", color = "black") +
    # 列标题
    annotate("text", x = x_text_start, y = max(df$y_pos) + 0.8,
             label = "N (Events)", fontface = "bold",
             hjust = 0, size = 3.2, family = "sans", color = "black") +
    annotate("text", x = x_text_auc, y = max(df$y_pos) + 0.8,
             label = "AUC (95% CI)", fontface = "bold",
             hjust = 0, size = 3.2, family = "sans", color = "black") +
    annotate("text", x = x_text_oe, y = max(df$y_pos) + 0.8,
             label = "O/E (95% CI)", fontface = "bold",
             hjust = 0, size = 3.2, family = "sans", color = "black") +
    # 坐标轴
    scale_x_continuous(limits = c(min(0.2, x_lo), 1.80),
                       breaks = seq(max(0, floor(x_lo * 10) / 10), 1.0, by = 0.1),
                       labels = scales::label_number(accuracy = 0.1)) +
    scale_y_continuous(breaks = df$y_pos, labels = df$Subgroup) +
    coord_cartesian(xlim = c(x_lo, 0.95), clip = "off") +
    labs(x = "AUC (95% CI)", y = NULL) +
    theme_bw(base_size = FONT_LABEL, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      axis.title.x = element_text(size = FONT_LABEL),
      axis.text.y = element_text(size = FONT_TICK, hjust = 1),
      axis.text.x = element_text(size = FONT_TICK),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5),
      plot.margin = margin(10, 180, 5, 5)
    )
  
  plot_height <- max(4.5, nrow(df) * 0.45 + 1.2)
  save_sci_plot(p, paste0("Subgroup_Forest_Plot_", model_name),
                width = 8.0, height = plot_height, dpi_type = "line")
  cat("  Subgroup forest plot done.\n")
  return(p)
}

# ---------------------------------------------------------------
# 5.2 风险分层点图（Cleveland Dot Plot）
# ---------------------------------------------------------------
plot_risk_stratification <- function() {
  cat("[2/10] 绘制风险分层点图...\n")
  
  data_path <- file.path(MODEL_DEV_DIR, "risk_stratification_results.csv")
  if (!file.exists(data_path)) {
    cat("  WARNING: risk_stratification_results.csv not found, skipping.\n")
    return(NULL)
  }
  
  df <- read.csv(data_path, stringsAsFactors = FALSE)
  
  # 计算 Wilson 95% CI
  calc_wilson_ci <- function(x, n, conf.level = 0.95) {
    if (n == 0) return(c(0, 0))
    z <- qnorm((1 + conf.level) / 2)
    p_hat <- x / n
    denom <- 1 + z^2 / n
    center <- (p_hat + z^2 / (2 * n)) / denom
    margin <- z * sqrt((p_hat * (1 - p_hat) + z^2 / (4 * n)) / n) / denom
    c(center - margin, center + margin)
  }
  
  # 风险分层结果来自 Python 输出，仅含 Observed_Rate，通过 Observed_Rate * N 反推事件数
  
  # 文件列：Cohort,Risk_Group,N,Observed_Rate
  # 通过 Observed_Rate * N 反推事件数，用于 Wilson CI
  
  df$Pos <- round(df$Observed_Rate * df$N)
  ci_list <- lapply(1:nrow(df), function(i) {
    calc_wilson_ci(df$Pos[i], df$N[i])
  })
  df$CI_Lower <- sapply(ci_list, `[`, 1)
  df$CI_Upper <- sapply(ci_list, `[`, 2)
  
  # 动态风险水平顺序：从 CSV 元数据列或 Risk_Group 字符串中提取截断值排序
  # 优先使用 Low_Cutoff / High_Cutoff 元数据列（由 Python 05 脚本写入）
  if ("Low_Cutoff" %in% colnames(df) && "High_Cutoff" %in% colnames(df) &&
      nrow(df) > 0) {
    low_val <- df$Low_Cutoff[1]
    high_val <- df$High_Cutoff[1]
  } else {
    # 回退：从 Risk_Group 字符串中解析数字（如 "Low (<0.12)" 提取 0.12）
    parse_first_num <- function(s) {
      m <- regmatches(s, regexpr("[0-9]+\\.?[0-9]*", s))
      if (length(m) > 0) as.numeric(m[1]) else NA
    }
    nums <- sapply(df$Risk_Group, parse_first_num)
    low_val <- min(nums, na.rm = TRUE)
    high_val <- max(nums, na.rm = TRUE)
  }

  # 根据截断值构建标签及其顺序
  low_label  <- sprintf("Low (<%.0f%%)", low_val * 100)
  inter_label <- sprintf("Intermediate (%.0f%%-%.0f%%)", low_val * 100, high_val * 100)
  high_label <- sprintf("High (%.0f%%)", high_val * 100)

  # 匹配实际出现的 Risk_Group 值（可能因浮点精度与 sprintf 不精确一致）
  # 改用模糊匹配：包含 "Low"、"Intermediate"、"High" 即可
  group_levels <- c(
    grep("Low", df$Risk_Group, value = TRUE)[1],
    grep("Intermediate", df$Risk_Group, value = TRUE)[1],
    grep("High", df$Risk_Group, value = TRUE)[1]
  )
  group_levels <- unique(na.omit(group_levels))
  df$Risk_Group <- factor(df$Risk_Group, levels = group_levels)
  
  # 队列颜色（风险分层仅覆盖 Derivation / Temporal，地理集已按需求移除）
  cohort_colors <- c("Derivation" = COL_DERIV, "Temporal" = COL_TEMP)
  cohort_shapes <- c("Derivation" = 16, "Temporal" = 15)
  
  # Y 轴偏移
  y_offset <- c("Derivation" = 0.18, "Temporal" = -0.18)
  
  df$y_pos <- as.numeric(df$Risk_Group) + y_offset[df$Cohort]
  
  # 计算 Derivation 总体患病率
  deriv_total <- subset(df, Cohort == "Derivation")
  deriv_overall_rate <- sum(deriv_total$Pos) / sum(deriv_total$N)
  
  p <- ggplot(df, aes(x = Observed_Rate * 100, y = y_pos)) +
    # 队列间连接线
    geom_line(aes(group = Risk_Group), color = "gray70", linewidth = 0.5, alpha = 0.5) +
    # Wilson 误差线：geom_errorbarh 已于 ggplot2 4.0 弃用，改用 geom_linerange（同森林图）
    geom_linerange(aes(xmin = CI_Lower * 100, xmax = CI_Upper * 100, color = Cohort),
                   linewidth = 0.8, show.legend = FALSE) +
    # 点
    geom_point(aes(color = Cohort, shape = Cohort), size = 3.5, stroke = 0.8) +
    # 标签
    geom_text(aes(label = paste0(round(Observed_Rate * 100), "%"),
                  color = Cohort),
              hjust = -0.3, vjust = 0, size = 2.8, show.legend = FALSE) +
    scale_color_manual(name = "Cohort", values = cohort_colors,
                       breaks = names(cohort_colors), labels = names(cohort_colors)) +
    scale_shape_manual(name = "Cohort", values = cohort_shapes,
                       breaks = names(cohort_shapes), labels = names(cohort_shapes)) +
    # 参考线：Derivation 总体患病率
    geom_vline(xintercept = deriv_overall_rate * 100, linetype = "dashed",
               color = "gray50", linewidth = 0.6) +
    scale_y_continuous(breaks = 1:3, labels = levels(df$Risk_Group),
                       limits = c(0.5, 3.5)) +
    scale_x_continuous(limits = c(0, 100), labels = scales::label_number(suffix = "%")) +
    labs(x = "Observed Event Rate (%)", y = "Risk Group") +
    theme_bw(base_size = FONT_LABEL, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      axis.title = element_text(size = FONT_LABEL),
      axis.text = element_text(size = FONT_TICK),
      legend.position = auto_legend_pos(df$Observed_Rate * 100, df$y_pos),
      legend.justification = c(0.5, 0.5),
      legend.background = element_rect(fill = "white", color = "gray80"),
      legend.title = element_text(size = FONT_LEGEND),
      legend.text = element_text(size = FONT_LEGEND - 1),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5)
    )
  
  save_sci_plot(p, "Risk_Stratification_Dotplot",
                width = 7.0, height = 4.5, dpi_type = "line")
  cat("  Risk stratification dot plot done.\n")
  return(p)
}

# ---------------------------------------------------------------
# 5.2b 临床影响曲线（Clinical Impact Curve）
# ---------------------------------------------------------------
plot_clinical_impact <- function(model_name) {
  cat("[2.5/10] 绘制临床影响曲线...\n")

  # 读取三个队列的预测概率（来自 05_Model_Development 输出）
  cohorts <- list(
    list(name = "Derivation", file = file.path(MODEL_DEV_DIR, paste0("predicted_probabilities_", model_name, "_derivation.csv"))),
    list(name = "Temporal", file = file.path(MODEL_DEV_DIR, paste0("predicted_probabilities_", model_name, "_temporal.csv"))),
    list(name = "Geographical", file = file.path(MODEL_DEV_DIR, paste0("predicted_probabilities_", model_name, "_geographical.csv")))
  )

  # 阈值序列（从 0.01 到 0.50，步长 0.01，聚焦临床相关区间）
  thresholds <- seq(0.01, 0.50, by = 0.01)

  plot_list <- list()

  for (cohort_info in cohorts) {
    fpath <- cohort_info$file
    if (!file.exists(fpath)) {
      cat(sprintf("  WARNING: %s not found, skipping.\n", basename(fpath)))
      next
    }

    dat <- read.csv(fpath)
    y_true <- dat$y_true
    y_prob <- dat$y_pred_prob_calibrated  # 使用校准后概率

    n_total <- length(y_true)
    n_events <- sum(y_true == 1)

    # 初始化向量
    n_high_risk <- numeric(length(thresholds))
    n_high_risk_pos <- numeric(length(thresholds))

    for (i in seq_along(thresholds)) {
      t <- thresholds[i]
      pred_high <- y_prob >= t
      n_high_risk[i] <- sum(pred_high)
      n_high_risk_pos[i] <- sum(pred_high & y_true == 1)
    }

    # 计算阳性预测值（PPV）用于右侧标注
    ppv <- n_high_risk_pos / n_high_risk
    ppv[is.nan(ppv)] <- 0

    df <- data.frame(
      Threshold = thresholds,
      N_High_Risk = n_high_risk,
      N_High_Risk_Pos = n_high_risk_pos,
      PPV = ppv,
      Cohort = cohort_info$name
    )
    plot_list[[cohort_info$name]] <- df
  }

  if (length(plot_list) == 0) {
    cat("  WARNING: No data for clinical impact curve, skipping.\n")
    return(NULL)
  }

  # 合并数据
  df_all <- do.call(rbind, plot_list)
  df_all$Cohort <- factor(df_all$Cohort, levels = c("Derivation", "Temporal", "Geographical"))

  # 色盲友好配色（与全脚本保持一致）
  cohort_colors <- c("Derivation" = COL_DERIV, "Temporal" = COL_TEMP, "Geographical" = COL_GEO)

  # ---- 绘制双轴图 ----
  # 提取 Derivation 队列作为主参考，确定右侧 y 轴范围（PPV）
  der_data <- subset(df_all, Cohort == "Derivation")
  max_ppv <- max(der_data$PPV, na.rm = TRUE)
  ylim_right <- c(0, max(1, max_ppv * 1.15))

  # 计算左轴缩放因子：使两条曲线在视觉上不重叠打架
  max_left <- max(df_all$N_High_Risk, na.rm = TRUE)
  scale_factor <- max_left / max(ylim_right)

  p <- ggplot() +
    # ---- 左侧条形图：高风险人数 ----
    geom_bar(data = df_all,
             aes(x = Threshold, y = N_High_Risk, fill = Cohort),
             stat = "identity", alpha = 0.25, width = 0.008, position = "dodge") +
    # ---- 左侧折线：高风险中阳性人数 ----
    geom_line(data = df_all,
              aes(x = Threshold, y = N_High_Risk_Pos, color = Cohort, linetype = Cohort),
              linewidth = 1.0) +
    # ---- 右侧折线：PPV（阳性预测值） ----
    geom_line(data = df_all,
              aes(x = Threshold, y = PPV * scale_factor, color = Cohort),
              linewidth = 1.2, linetype = "dashed") +
    # ---- 右轴标注 ----
    scale_y_continuous(
      name = "Number of Patients (High Risk / True Positives)",
      sec.axis = sec_axis(~ . / scale_factor,
                          name = "Positive Predictive Value (PPV)",
                          labels = scales::percent_format(accuracy = 1))
    ) +
    scale_color_manual(values = cohort_colors) +
    scale_fill_manual(values = cohort_colors) +
    scale_linetype_manual(values = c("Derivation" = "solid", "Temporal" = "solid", "Geographical" = "solid")) +
    labs(x = "Threshold Probability",
         title = paste("Clinical Impact Curve —", model_name),
         subtitle = "Bars: High-risk patients (opacity); Solid lines: True positives (left axis); Dashed lines: PPV (right axis)") +
    theme_bw(base_family = "sans") +
    theme(
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.title = element_blank(),
      axis.title.y.left = element_text(color = "black", size = FONT_LABEL),
      axis.title.y.right = element_text(color = "black", size = FONT_LABEL),
      axis.text = element_text(size = FONT_TICK),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5)
    )

  # 保存
  save_sci_plot(p, paste0("Clinical_Impact_Curve_", model_name),
                width = 8.0, height = 6.0, dpi_type = "line")
  cat("  Clinical Impact Curve saved.\n")

  return(p)
}

# ---------------------------------------------------------------
# 5.3 校准曲线（校准前/校准后双版本）
# ---------------------------------------------------------------
plot_calibration_curves <- function(model_name, use_calibrated = TRUE, file_suffix = "") {
  # 步骤号/标签随 file_suffix 与 use_calibrated 变化
  #   file_suffix=="" & use_calibrated==FALSE -> Raw 主图（Pre-Calibration）
  #   file_suffix=="" & use_calibrated==TRUE  -> Platt 补充图（Post-Calibration）
  #   file_suffix=="_intercept_recalibrated"  -> 截距重校准补充图
  if (file_suffix == "_intercept_recalibrated") {
    calib_label <- "Intercept-Calibrated"
    step_no <- "5"
  } else if (use_calibrated) {
    calib_label <- "Post-Calibration (Platt)"
    step_no <- "4"
  } else {
    calib_label <- "Pre-Calibration (Raw)"
    step_no <- "3"
  }
  cat(sprintf("[%s/10] 绘制校准曲线 (%s)...\n", step_no, calib_label))
  
  cohorts <- list(
    list(name = "Derivation", color = COL_DERIV),
    list(name = "Temporal", color = COL_TEMP),
    list(name = "Geographical", color = COL_GEO)
  )
  
  all_data <- list()
  for (cohort in cohorts) {
    # 05 输出的概率文件：默认无后缀（Raw / Platt）；file_suffix 非空时读取
    # 对应版本（如截距重校准 *_intercept_recalibrated.csv，仅外部集生成）。
    cohort_tag <- tolower(cohort$name)
    fname <- paste0("predicted_probabilities_", model_name, "_", cohort_tag, file_suffix, ".csv")
    fpath <- file.path(MODEL_DEV_DIR, fname)
    # Derivation 无截距重校准文件，回退到默认文件（无后缀）
    if (!file.exists(fpath) && cohort$name == "Derivation" && file_suffix == "_intercept_recalibrated") {
      fname <- paste0("predicted_probabilities_", model_name, "_derivation.csv")
      fpath <- file.path(MODEL_DEV_DIR, fname)
      cat(sprintf("  Derivation %s file not found, falling back to %s\n", file_suffix, fname))
    }
    if (!file.exists(fpath)) {
      cat(sprintf("  WARNING: %s not found, skipping %s.\n", fname, cohort$name))
      next
    }
    dat <- read.csv(fpath, stringsAsFactors = FALSE)
    dat$Cohort <- cohort$name
    all_data[[length(all_data) + 1]] <- dat
  }
  
  if (length(all_data) == 0) {
    cat("  WARNING: No calibration data found, skipping.\n")
    return(NULL)
  }
  
  df <- bind_rows(all_data)
  
  # 根据 use_calibrated 选择概率列（校准后 / 校准前原始概率）
  if (use_calibrated) {
    prob_col <- "y_pred_prob_calibrated"
  } else {
    if ("y_pred_prob_raw" %in% names(df)) {
      prob_col <- "y_pred_prob_raw"
    } else {
      cat("  WARNING: y_pred_prob_raw not found, falling back to calibrated.\n")
      prob_col <- "y_pred_prob_calibrated"
    }
  }
  
  # 计算 Brier Score（使用所选概率列）
  brier_scores <- df %>%
    group_by(Cohort) %>%
    summarise(
      Brier = mean((y_true - .data[[prob_col]])^2),
      .groups = "drop"
    )
  
  brier_labels <- setNames(
    sprintf("%s (Brier = %.3f)", brier_scores$Cohort, brier_scores$Brier),
    brier_scores$Cohort
  )
  
  cohort_colors <- setNames(c(COL_DERIV, COL_TEMP, COL_GEO),
                            c("Derivation", "Temporal", "Geographical"))
  
  # ---- 分位数分箱（quantile bin）+ 局部合并（Local Merging）----
  # 核心思想：起始保持 10 个分位数箱子；仅对阳性事件不足（n_pos < 5）的
  #           个别箱子与其邻近箱子局部合并，尽量保留更多箱子
  bin_data <- list()
  for (cohort_name in unique(df$Cohort)) {
    sub <- df[df$Cohort == cohort_name, ]
    sub <- sub[order(sub[[prob_col]]), ]
    
    # 起始 10 个分位数箱子
    n_bins <- 10
    bin_edges <- quantile(sub[[prob_col]],
                          probs = seq(0, 1, length.out = n_bins + 1),
                          na.rm = TRUE)
    # 去除重复分位点（预测概率存在大量相同值时）
    bin_edges <- unique(bin_edges)
    
    # 局部合并循环：逐箱检查，仅合并 n_pos < 5 的箱子，直至全部达标
    repeat {
      bin_indices <- cut(sub[[prob_col]], breaks = bin_edges,
                         include.lowest = TRUE, labels = FALSE)
      n_bins_eff <- length(unique(bin_indices[!is.na(bin_indices)]))
      if (n_bins_eff <= 1) break  # 只剩一个箱子，无法再合并
      
      # 当前各箱的阳性事件数与样本量
      bins_tmp <- sub %>%
        mutate(bin = bin_indices) %>%
        filter(!is.na(bin)) %>%
        group_by(bin) %>%
        summarise(n_pos = sum(y_true), n_bin = n(), .groups = "drop")
      
      bad_idx <- which(bins_tmp$n_pos < 5)
      if (length(bad_idx) == 0) break  # 所有箱子均达标
      
      # 优先处理阳性事件最少的箱子
      merge_target <- bad_idx[which.min(bins_tmp$n_pos[bad_idx])]
      
      # 选择邻居：与样本量较小的邻近箱子合并（保持各箱均衡），tie 时向左
      neighbors <- c(merge_target - 1, merge_target + 1)
      neighbors <- neighbors[neighbors >= 1 & neighbors <= n_bins_eff]
      if (length(neighbors) == 0) break
      neighbor_sel <- neighbors[which.min(bins_tmp$n_bin[neighbors])]
      
      # 局部合并：删除两箱之间的分位点边界（内部边界 bin_edges[lo+1]）
      lo_bin <- min(merge_target, neighbor_sel)
      bin_edges <- bin_edges[-(lo_bin + 1)]
    }
    
    sub$bin <- cut(sub[[prob_col]], breaks = bin_edges,
                   include.lowest = TRUE, labels = FALSE)
    
    bins <- sub %>%
      group_by(bin) %>%
      summarise(
        mean_pred = mean(.data[[prob_col]]),
        n_bin = n(),
        n_pos = sum(y_true),
        obs_prop = mean(y_true),
        .groups = "drop"
      ) %>%
      # 按 bin 排序以确保 mean_pred 递增
      arrange(bin)
    
    # 过滤可能产生的 NA 箱
    bins <- bins[!is.na(bins$bin), ]
    
    bins$Cohort <- cohort_name
    bin_data[[length(bin_data) + 1]] <- bins
  }
  bin_df <- bind_rows(bin_data)
  
  # ---- 绘制校准曲线 ----
  p <- ggplot() +
    # 理想线
    geom_abline(intercept = 0, slope = 1, linetype = "dashed",
                color = "gray40", linewidth = 0.7) +
    # 连接线（按 mean_pred 升序）
    geom_line(data = bin_df,
              aes(x = mean_pred, y = obs_prop, color = Cohort, group = Cohort),
              linewidth = 0.8) +
    # 分箱点
    geom_point(data = bin_df,
               aes(x = mean_pred, y = obs_prop, color = Cohort),
               shape = 16, size = 2.5) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_color_manual(name = NULL,
                       values = cohort_colors,
                       breaks = brier_scores$Cohort,
                       labels = brier_labels) +
    labs(x = "Mean Predicted Probability",
         y = "Observed Proportion") +
    theme_bw(base_size = FONT_LABEL, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      axis.title = element_text(size = FONT_LABEL),
      axis.text = element_text(size = FONT_TICK),
      legend.position = c(0.85, 0.15),
      legend.justification = c(0.5, 0.5),
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
      legend.text = element_text(size = FONT_LEGEND - 1),
      legend.key.size = unit(0.5, "cm"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3, color = "gray90"),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5)
    )
  
  # 文件名：根据 use_calibrated 和 file_suffix 生成基名（临时名，随后在 main 中
  # 重命名为论文最终文件名）
  #   file_suffix=="" & use_calibrated==FALSE -> Calibration_Curves_Pre_Calibration_*
  #   file_suffix=="" & use_calibrated==TRUE  -> Calibration_Curves_*（Platt）
  #   file_suffix=="_intercept_recalibrated"  -> Calibration_Curves_Intercept_*
  if (file_suffix == "_intercept_recalibrated") {
    tmp_suffix <- "_Intercept"
  } else if (use_calibrated) {
    tmp_suffix <- ""
  } else {
    tmp_suffix <- "_Pre_Calibration"
  }
  save_sci_plot(p, paste0("Calibration_Curves", tmp_suffix, "_", model_name),
                width = 7.0, height = 6.0, dpi_type = "color")
  cat(sprintf("  Calibration curves (%s) done.\n", calib_label))
  return(p)
}

# ---------------------------------------------------------------
# 5.4 决策曲线（DCA）
# ---------------------------------------------------------------
plot_dca_curves <- function(model_name, use_calibrated = TRUE, file_suffix = "") {
  # 步骤号/标签随 file_suffix 与 use_calibrated 变化（与校准曲线分组对应）
  #   file_suffix=="" & use_calibrated==FALSE -> Raw 主图
  #   file_suffix=="" & use_calibrated==TRUE  -> Platt 补充图
  #   file_suffix=="_intercept_recalibrated"  -> 截距重校准补充图
  if (file_suffix == "_intercept_recalibrated") {
    calib_label <- "Intercept-Calibrated"
    step_no <- "8"
  } else if (use_calibrated) {
    calib_label <- "Post-Calibration (Platt)"
    step_no <- "7"
  } else {
    calib_label <- "Pre-Calibration (Raw)"
    step_no <- "6"
  }
  cat(sprintf("[%s/10] 绘制决策曲线 (%s)...\n", step_no, calib_label))
  cat(sprintf("  DCA %s: Treat None / Treat All / Model curves + colored clinical value regions.\n", calib_label))
  
  # 阈值概率 0.01~0.80（与连续净收益 / NIA / 贝叶斯DCA 口径统一；>0.80 临床意义有限）
  thresholds <- seq(0.01, 0.80, by = 0.01)
  
  cohorts <- list(
    list(name = "Derivation"),
    list(name = "Temporal"),
    list(name = "Geographical")
  )
  
  all_dca <- list()
  prev_list <- list()
  
  for (cohort in cohorts) {
    # 05 输出的概率文件：默认无后缀（Raw / Platt）；file_suffix 非空时读取
    # 对应版本（如截距重校准 *_intercept_recalibrated.csv，仅外部集生成）。
    cohort_tag <- tolower(cohort$name)
    fname <- paste0("predicted_probabilities_", model_name, "_", cohort_tag, file_suffix, ".csv")
    fpath <- file.path(MODEL_DEV_DIR, fname)
    # Derivation 无截距重校准文件，回退到默认文件（无后缀）
    if (!file.exists(fpath) && cohort$name == "Derivation" && file_suffix == "_intercept_recalibrated") {
      fname <- paste0("predicted_probabilities_", model_name, "_derivation.csv")
      fpath <- file.path(MODEL_DEV_DIR, fname)
      cat(sprintf("  Derivation %s file not found, falling back to %s\n", file_suffix, fname))
    }
    if (!file.exists(fpath)) {
      cat(sprintf("  WARNING: %s not found, skipping %s.\n", fname, cohort$name))
      next
    }
    dat <- read.csv(fpath, stringsAsFactors = FALSE)
    y_true <- dat$y_true
    
    # 根据 use_calibrated 参数选择概率列
    if (use_calibrated) {
      y_pred <- dat$y_pred_prob_calibrated
    } else {
      # 校准前使用原始概率
      if ("y_pred_prob_raw" %in% colnames(dat)) {
        y_pred <- dat$y_pred_prob_raw
      } else {
        cat(sprintf("  WARNING: y_pred_prob_raw not found for %s, falling back to calibrated.\n", cohort$name))
        y_pred <- dat$y_pred_prob_calibrated
      }
    }
    
    n <- length(y_true)
    prev_list[[cohort$name]] <- mean(y_true)
    
    # 向量化计算 NB = TP/n - FP/n * t/(1-t)
    nb <- sapply(thresholds, function(t) {
      tp <- sum(y_pred >= t & y_true == 1)
      fp <- sum(y_pred >= t & y_true == 0)
      (tp / n) - (fp / n) * (t / (1 - t))
    })
    
    all_dca[[length(all_dca) + 1]] <- data.frame(
      Threshold = thresholds,
      Net_Benefit = nb,
      Strategy = cohort$name,
      stringsAsFactors = FALSE
    )
  }
  
  if (length(all_dca) == 0) {
    cat("  WARNING: No DCA data found, skipping.\n")
    return(NULL)
  }
  dca_df <- bind_rows(all_dca)
  
  # ---- 基准线 ----
  make_treat_all <- function(prev) {
    prev - (1 - prev) * (thresholds / (1 - thresholds))
  }
  # Ideal 参考线：完美预测模型的净收益恒等于患病率（TP=n_events, FP=0）。
  # 使用 Derivation 患病率作为统一上界参考（模型与校准器均在 Derivation 开发）。
  ideal_prev <- if (!is.null(prev_list[["Derivation"]])) prev_list[["Derivation"]] else unlist(prev_list)[1]
  baseline_list <- list(
    data.frame(Threshold = thresholds, Net_Benefit = 0,
               Strategy = "Treat None", stringsAsFactors = FALSE),
    data.frame(Threshold = thresholds,
               Net_Benefit = rep(ideal_prev, length(thresholds)),
               Strategy = "Ideal (perfect)", stringsAsFactors = FALSE)
  )
  for (nm in names(prev_list)) {
    baseline_list[[length(baseline_list) + 1]] <- data.frame(
      Threshold = thresholds, Net_Benefit = make_treat_all(prev_list[[nm]]),
      Strategy = paste0("Treat All (", nm, ")"), stringsAsFactors = FALSE)
  }
  
  # ---- 合并全部曲线 ----
  strategy_levels <- c(
    "Ideal (perfect)",
    "Treat None",
    "Treat All (Derivation)", "Derivation",
    "Treat All (Temporal)", "Temporal",
    "Treat All (Geographical)", "Geographical"
  )
  all_lines <- bind_rows(c(list(dca_df), baseline_list))
  all_lines$Strategy <- factor(all_lines$Strategy, levels = strategy_levels)
  
  # ---- 色盲友好配色方案 ----
  # Derivation = 蓝色 #1f77b4, Temporal = 橙色 #ff7f0e, Geographical = 绿色 #2ca02c
  all_colors <- c(
    "Ideal (perfect)"        = "darkred",
    "Treat None"             = "black",
    "Treat All (Derivation)" = "gray30",
    "Derivation"             = COL_DERIV,   # 蓝色 #1f77b4
    "Treat All (Temporal)"   = "gray50",
    "Temporal"               = COL_TEMP,    # 橙色 #ff7f0e
    "Treat All (Geographical)" = "gray70",
    "Geographical"           = COL_GEO      # 绿色 #2ca02c
  )
  all_linetypes <- c(
    "Ideal (perfect)" = "dotted",
    "Treat None" = "dashed",  # 黑色虚线：所有不治疗，净获益恒 0
    "Treat All (Derivation)" = "solid", "Derivation" = "solid",
    "Treat All (Temporal)" = "solid", "Temporal" = "solid",
    "Treat All (Geographical)" = "solid", "Geographical" = "solid"
  )
  all_sizes <- c(
    "Ideal (perfect)" = 0.7,
    "Treat None" = 0.5,
    "Treat All (Derivation)" = 0.8, "Derivation" = 1.2,
    "Treat All (Temporal)" = 0.8, "Temporal" = 1.2,
    "Treat All (Geographical)" = 0.8, "Geographical" = 1.2
  )
  
  # ---- 曲线同色填充区域：各队列模型 NB > max(Treat None=0, Treat All) 的临床价值区间 ----
  region_colors <- c(
    "Derivation"   = COL_DERIV,   # 蓝色
    "Temporal"     = COL_TEMP,    # 橙色
    "Geographical" = COL_GEO      # 绿色
  )
  region_df <- do.call(rbind, lapply(names(prev_list), function(nm) {
    sub <- dca_df[dca_df$Strategy == nm, ]
    base <- pmax(make_treat_all(prev_list[[nm]]), 0)
    data.frame(Threshold = thresholds,
               ymin = base,
               ymax = pmax(sub$Net_Benefit, base),
               Region = nm, stringsAsFactors = FALSE)
  }))
  
  # ---- y 轴范围 ----
  max_nb <- max(dca_df$Net_Benefit, na.rm = TRUE)
  max_prev <- max(unlist(prev_list), na.rm = TRUE)
  y_lo <- -0.15
  y_hi <- max(max_nb, max_prev, na.rm = TRUE) * 1.05 + 0.02
  
  # 图例固定右上角，稍向下移位
  leg_pos <- c(0.85, 0.82)
  
  p <- ggplot() +
    # 临床价值区域（曲线同色填充，各颜色可交叉重叠）
    geom_ribbon(data = region_df,
                aes(x = Threshold, ymin = ymin, ymax = ymax, fill = Region),
                alpha = 0.20, color = NA) +
    # 模型 + 基准线
    geom_line(data = all_lines, aes(x = Threshold, y = Net_Benefit,
                                    color = Strategy, linetype = Strategy,
                                    linewidth = Strategy)) +
    scale_color_manual(name = NULL, values = all_colors, breaks = names(all_colors)) +
    scale_linetype_manual(name = NULL, values = all_linetypes) +
    scale_linewidth_manual(name = NULL, values = all_sizes) +
    scale_fill_manual(name = NULL, values = region_colors) +
    coord_cartesian(xlim = c(0, 0.80), ylim = c(y_lo, y_hi)) +
    scale_x_continuous(limits = c(0, 0.80), breaks = seq(0, 0.8, by = 0.1),
                       labels = scales::label_number(accuracy = 0.1)) +
    labs(x = "Threshold Probability", y = "Net Benefit") +
    guides(
      color = guide_legend(override.aes = list(linewidth = 1.2), order = 1),
      fill = guide_legend(override.aes = list(alpha = 0.35), order = 2),
      linetype = "none", linewidth = "none"
    ) +
    theme_bw(base_size = FONT_LABEL, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      axis.title = element_text(size = FONT_LABEL),
      axis.text = element_text(size = FONT_TICK),
      legend.position = leg_pos,
      legend.justification = c(0.5, 0.5),
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
      legend.text = element_text(size = FONT_LEGEND - 2),
      legend.key.size = unit(0.4, "cm"),
      legend.spacing.y = unit(0.02, "cm"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3, color = "gray90"),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5)
    )
  
  # 文件名：根据 use_calibrated 和 file_suffix 生成基名（临时名，随后在 main 中
  # 重命名为论文最终文件名）
  #   file_suffix=="" & use_calibrated==FALSE -> DCA_Curves_Pre_Calibration_*
  #   file_suffix=="" & use_calibrated==TRUE  -> DCA_Curves_*（Platt）
  #   file_suffix=="_intercept_recalibrated"  -> DCA_Curves_Intercept_*
  if (file_suffix == "_intercept_recalibrated") {
    tmp_suffix <- "_Intercept"
  } else if (use_calibrated) {
    tmp_suffix <- ""
  } else {
    tmp_suffix <- "_Pre_Calibration"
  }
  save_sci_plot(p, paste0("DCA_Curves", tmp_suffix, "_", model_name),
                width = 7.0, height = 6.0, dpi_type = "line")
  cat(sprintf("  DCA curves (%s) done.\n", calib_label))
  return(p)
}

# ---------------------------------------------------------------
# 5.5 连续净收益图（Continuous Net Benefit）
# ---------------------------------------------------------------
plot_continuous_nb <- function(model_name) {
  cat("[9/10] 绘制连续净收益曲线...\n")
  
  # 强制使用 dcurves 包：安装后再次 require 确认，仍失败则中止并提示手动安装
  if (!require("dcurves", quietly = TRUE)) {
    cat("  dcurves not available, installing from CRAN...\n")
    tryCatch(
      install.packages("dcurves", repos = "https://cloud.r-project.org", quiet = TRUE),
      error = function(e) cat(sprintf("  dcurves installation error: %s\n", e$message))
    )
    if (!require("dcurves", quietly = TRUE)) {
      stop("R 包 dcurves 自动安装/加载失败，请手动执行 install.packages(\"dcurves\") 后重跑本脚本（连续净收益图强制要求 dcurves 计算）。")
    }
  }
  
  # pracma 用于梯形积分（已安装）
  has_pracma <- require("pracma", quietly = TRUE)
  
  cohorts <- list(
    list(name = "Derivation", file = paste0("predicted_probabilities_", model_name, "_derivation.csv"),
         color = COL_DERIV),
    list(name = "Temporal", file = paste0("predicted_probabilities_", model_name, "_temporal.csv"),
         color = COL_TEMP),
    list(name = "Geographical", file = paste0("predicted_probabilities_", model_name, "_geographical.csv"),
         color = COL_GEO)
  )
  
  thresholds <- seq(0.01, 0.80, by = 0.01)
  all_data <- list()
  prev_list <- list()
  
  for (cohort in cohorts) {
    # 统一使用 05 输出的概率文件（含 Platt 校准后概率，单一校准器仅 Derivation 拟合）。
    # 不再读取 *_intercept_recalibrated.csv（外部集重校准会引入泄漏，且与 06 口径不一致）。
    fpath <- file.path(MODEL_DEV_DIR, cohort$file)
    if (!file.exists(fpath)) {
      cat(sprintf("  WARNING: %s not found, skipping %s.\n", cohort$file, cohort$name))
      next
    }
    dat <- read.csv(fpath, stringsAsFactors = FALSE)
    y_true <- dat$y_true
    y_pred <- dat$y_pred_prob_calibrated
    n <- length(y_true)
    prevalence <- mean(y_true)
    
    prev_list[[cohort$name]] <- prevalence
    
    # 使用 dcurves::dca 计算净收益（从返回对象提取模型变量）
    # 注意：dcurves::dca 不接受 smooth/standardize 参数（此前版本误传导致
    # 全部回退手工计算），dcurves 的标准化在返回对象中体现。
    dca_obj <- tryCatch(
      dca(y_true ~ y_pred_prob_calibrated, data = dat,
          thresholds = seq(0.01, 0.80, by = 0.01)),
      error = function(e) {
        cat(sprintf("  WARNING: dcurves::dca failed for %s: %s\n", cohort$name, e$message))
        NULL
      }
    )
    
    if (!is.null(dca_obj)) {
      # dcurves 返回包含 model/all/none 三个变量的 tibble
      dca_tbl <- dca_obj$dca
      # 提取模型变量（非 all 且非 none）
      model_rows <- dca_tbl[dca_tbl$variable != "all" & dca_tbl$variable != "none", ]
      model_rows <- model_rows[model_rows$threshold >= 0.01, ]
      if (nrow(model_rows) == length(thresholds)) {
        nb <- model_rows$net_benefit
        # 优先使用 standardized_net_benefit（若 dcurves 版本支持）
        if ("standardized_net_benefit" %in% colnames(model_rows)) {
          snb <- model_rows$standardized_net_benefit
        } else {
          snb <- if (prevalence > 0) nb / prevalence else rep(0, length(nb))
        }
      } else {
        # 回退：手工计算（dcurves 返回行数与阈值网格不匹配时）
        cat(sprintf("  !!! WARNING: %s 非dcurves结果，为手工回退计算（manual fallback），与强制使用dcurves的口径不一致，结果解读需谨慎 !!!\n",
                    cohort$name))
        nb <- sapply(thresholds, function(t) {
          tp <- sum(y_pred >= t & y_true == 1)
          fp <- sum(y_pred >= t & y_true == 0)
          (tp / n) - (fp / n) * (t / (1 - t))
        })
        snb <- if (prevalence > 0) nb / prevalence else rep(0, length(nb))
      }
    } else {
      # 回退：手工计算（dcurves::dca 执行失败时）
      cat(sprintf("  !!! WARNING: %s 非dcurves结果，为手工回退计算（manual fallback），与强制使用dcurves的口径不一致，结果解读需谨慎 !!!\n",
                  cohort$name))
      nb <- sapply(thresholds, function(t) {
        tp <- sum(y_pred >= t & y_true == 1)
        fp <- sum(y_pred >= t & y_true == 0)
        (tp / n) - (fp / n) * (t / (1 - t))
      })
      snb <- if (prevalence > 0) nb / prevalence else rep(0, length(nb))
    }
    
    all_data[[length(all_data) + 1]] <- data.frame(
      Threshold = thresholds,
      Net_Benefit = nb,
      Standardized_NB = snb,
      Cohort = cohort$name,
      stringsAsFactors = FALSE
    )
  }
  
  if (length(all_data) == 0) {
    cat("  WARNING: No data found for Continuous Net Benefit, skipping.\n")
    return(NULL)
  }
  
  df <- bind_rows(all_data)
  
  # Treat All：各队列基于自身患病率的标准化净收益（与 DCA 图基准一致），
  # 使模型曲线与对应队列的 Treat All 基准可直接对比。
  treat_all_df <- do.call(rbind, lapply(names(prev_list), function(nm) {
    prev_i <- prev_list[[nm]]
    ta_nb <- prev_i - (1 - prev_i) * (thresholds / (1 - thresholds))
    data.frame(Threshold = thresholds,
               Standardized_NB = if (prev_i > 0) ta_nb / prev_i else rep(0, length(thresholds)),
               Cohort = paste0("Treat All (", nm, ")"),
               stringsAsFactors = FALSE)
  }))
  
  # Treat None
  treat_none_snb <- rep(0, length(thresholds))
  
  # 计算各队列 AUC-NB（标准化净收益曲线下面积）
  auc_info <- sapply(unique(df$Cohort), function(cname) {
    sub <- df[df$Cohort == cname, ]
    if (has_pracma) {
      trapz(sub$Threshold, sub$Standardized_NB)
    } else {
      sum(diff(sub$Threshold) * (head(sub$Standardized_NB, -1) + tail(sub$Standardized_NB, -1)) / 2)
    }
  })
  
  cohort_colors <- setNames(c(COL_DERIV, COL_TEMP, COL_GEO),
                            c("Derivation", "Temporal", "Geographical"))
  
  # 图例标签（含 AUC-NB）
  legend_labels <- sapply(names(auc_info), function(n) {
    sprintf("%s (AUC-NB = %.3f)", n, auc_info[n])
  })
  
  all_lines <- rbind(
    df[, c("Threshold", "Standardized_NB", "Cohort")],
    treat_all_df[, c("Threshold", "Standardized_NB", "Cohort")],
    data.frame(Threshold = thresholds, Standardized_NB = treat_none_snb,
               Cohort = "Treat None", stringsAsFactors = FALSE)
  )
  colnames(all_lines)[2] <- "Value"
  all_lines$Cohort <- factor(all_lines$Cohort,
    levels = c("Derivation", "Temporal", "Geographical",
               "Treat All (Derivation)", "Treat All (Temporal)", "Treat All (Geographical)",
               "Treat None"))
  
  all_colors <- c(
    "Derivation"   = COL_DERIV,
    "Temporal"     = COL_TEMP,
    "Geographical" = COL_GEO,
    "Treat All (Derivation)"   = COL_DERIV,
    "Treat All (Temporal)"     = COL_TEMP,
    "Treat All (Geographical)" = COL_GEO,
    "Treat None"   = "gray30"
  )
  
  all_linetypes <- c(
    "Derivation"   = "solid",
    "Temporal"     = "solid",
    "Geographical" = "solid",
    "Treat All (Derivation)"   = "dashed",
    "Treat All (Temporal)"     = "dashed",
    "Treat All (Geographical)" = "dashed",
    "Treat None"   = "solid"
  )
  
  all_sizes <- c(
    "Derivation"   = 1.2,
    "Temporal"     = 1.2,
    "Geographical" = 1.2,
    "Treat All (Derivation)"   = 0.7,
    "Treat All (Temporal)"     = 0.7,
    "Treat All (Geographical)" = 0.7,
    "Treat None"   = 0.5
  )
  
  legend_breaks <- c("Derivation", "Temporal", "Geographical",
                     "Treat All (Derivation)", "Treat All (Temporal)", "Treat All (Geographical)",
                     "Treat None")
  legend_labels_full <- c(
    legend_labels[["Derivation"]],
    legend_labels[["Temporal"]],
    legend_labels[["Geographical"]],
    "Treat All (Derivation)", "Treat All (Temporal)", "Treat All (Geographical)",
    "Treat None"
  )
  
  # y 轴范围：标准化净收益理论范围 0~1，下限封顶 -0.2（避免 Treat All 高阈值剧降压扁曲线）
  value_min <- min(all_lines$Value, na.rm = TRUE)
  value_max <- max(all_lines$Value, na.rm = TRUE)
  y_lo_cnb <- max(-0.20, value_min)
  y_hi_cnb <- min(1.05, max(0.05, value_max) * 1.05)
  
  p <- ggplot(all_lines, aes(x = Threshold, y = Value,
                             color = Cohort, linetype = Cohort, linewidth = Cohort)) +
    geom_line() +
    scale_color_manual(name = NULL, values = all_colors,
                       breaks = legend_breaks, labels = legend_labels_full) +
    scale_linetype_manual(name = NULL, values = all_linetypes,
                          breaks = legend_breaks, labels = legend_labels_full) +
    scale_linewidth_manual(name = NULL, values = all_sizes,
                           breaks = legend_breaks, labels = legend_labels_full) +
    coord_cartesian(xlim = c(0, 0.80), ylim = c(y_lo_cnb, y_hi_cnb)) +
    scale_x_continuous(limits = c(0, 0.80), breaks = seq(0, 0.8, by = 0.1),
                       labels = scales::label_number(accuracy = 0.1)) +
    labs(x = "Threshold Probability", y = "Standardized Net Benefit") +
    guides(color = guide_legend(override.aes = list(linewidth = 1.2)),
           linetype = "none", linewidth = "none") +
    theme_bw(base_size = FONT_LABEL, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      axis.title = element_text(size = FONT_LABEL),
      axis.text = element_text(size = FONT_TICK),
      legend.position = c(0.85, 0.88),
      legend.justification = c(0.5, 0.5),
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
      legend.text = element_text(size = FONT_LEGEND - 1),
      legend.key.size = unit(0.45, "cm"),
      legend.spacing.y = unit(0.02, "cm"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3, color = "gray90"),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5)
    )
  
  save_sci_plot(p, paste0("Continuous_Net_Benefit_", model_name),
                width = 7.0, height = 6.0, dpi_type = "color")
  cat("  Continuous Net Benefit curve done.\n")
  
  return(invisible(p))
}

# ---------------------------------------------------------------
# 5.5b 避免不必要干预数（Net Intervention Avoided, NIA）
# ---------------------------------------------------------------
# Vickers & Elkin (2006): NIA(t) = 100 * (NB_model - NB_TreatAll) / (t/(1-t))
# 表示在阈值 t 下，模型相对"全部治疗"策略每 100 名被治疗患者中
# 避免的不必要干预数。保留负值区间：NIA < 0 表示模型劣于 Treat All（有害），
# 图中以 y=0 参考线区分益（>0）/害（<0）区，不做截断以免掩盖有害区间。
plot_net_intervention_avoided <- function(model_name) {
  cat("[10/10] 绘制避免不必要干预数（NIA）曲线...\n")
  
  thresholds <- seq(0.01, 0.80, by = 0.01)
  cohorts <- list(
    list(name = "Derivation",   file = paste0("predicted_probabilities_", model_name, "_derivation.csv")),
    list(name = "Temporal",     file = paste0("predicted_probabilities_", model_name, "_temporal.csv")),
    list(name = "Geographical", file = paste0("predicted_probabilities_", model_name, "_geographical.csv"))
  )
  
  all_nia <- list()
  prev_list <- list()
  
  for (cohort in cohorts) {
    # 统一使用 05 输出的概率文件（Platt 校准后概率，单一校准器仅 Derivation 拟合）
    fpath <- file.path(MODEL_DEV_DIR, cohort$file)
    if (!file.exists(fpath)) {
      cat(sprintf("  WARNING: %s not found, skipping %s.\n", cohort$file, cohort$name))
      next
    }
    dat <- read.csv(fpath, stringsAsFactors = FALSE)
    y_true <- dat$y_true
    y_pred <- dat$y_pred_prob_calibrated
    n <- length(y_true)
    prev <- mean(y_true)
    prev_list[[cohort$name]] <- prev
    
    # 模型净收益
    nb_model <- sapply(thresholds, function(t) {
      tp <- sum(y_pred >= t & y_true == 1)
      fp <- sum(y_pred >= t & y_true == 0)
      (tp / n) - (fp / n) * (t / (1 - t))
    })
    # 该队列自身的 Treat All 净收益（与 DCA 图基准一致）
    nb_treat_all <- prev - (1 - prev) * (thresholds / (1 - thresholds))
    # NIA（每 100 名治疗患者）。保留负值：负值表示该阈值下模型劣于 Treat All（有害），
    # 截断为 0 会掩盖这一信息；绘图时以 y=0 虚线区分益/害区。
    odds_t <- thresholds / (1 - thresholds)
    nia <- 100 * (nb_model - nb_treat_all) / odds_t
    
    all_nia[[length(all_nia) + 1]] <- data.frame(
      Threshold = thresholds,
      NIA = nia,
      Cohort = cohort$name,
      stringsAsFactors = FALSE
    )
  }
  
  if (length(all_nia) == 0) {
    cat("  WARNING: No NIA data found, skipping.\n")
    return(NULL)
  }
  nia_df <- bind_rows(all_nia)
  nia_df$Cohort <- factor(nia_df$Cohort, levels = c("Derivation", "Temporal", "Geographical"))
  
  nia_colors <- c("Derivation" = COL_DERIV, "Temporal" = COL_TEMP, "Geographical" = COL_GEO)
  
  # y 轴下限须覆盖负值区间（有害区），y=0 虚线区分益/害
  nia_y_lo <- min(0, min(nia_df$NIA, na.rm = TRUE) * 1.05)
  nia_y_hi <- max(nia_df$NIA, na.rm = TRUE) * 1.05 + 1
  
  p <- ggplot(nia_df, aes(x = Threshold, y = NIA, color = Cohort)) +
    geom_line(linewidth = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
    scale_color_manual(values = nia_colors) +
    coord_cartesian(xlim = c(0, 0.80), ylim = c(nia_y_lo, nia_y_hi)) +
    scale_x_continuous(limits = c(0, 0.80), breaks = seq(0, 0.8, by = 0.1),
                       labels = scales::label_number(accuracy = 0.1)) +
    labs(x = "Threshold Probability",
         y = "Net Interventions Avoided (per 100 patients)",
         color = "Cohort") +
    theme_bw(base_size = FONT_LABEL, base_family = "sans") +
    theme(
      plot.title = element_blank(),
      axis.title = element_text(size = FONT_LABEL),
      axis.text = element_text(size = FONT_TICK),
      legend.position = auto_legend_pos(nia_df$Threshold, nia_df$NIA),
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3),
      legend.text = element_text(size = FONT_LEGEND - 1),
      legend.key.size = unit(0.45, "cm"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3, color = "gray90"),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5)
    )
  
  save_sci_plot(p, paste0("Net_Intervention_Avoided_", model_name),
                width = 7.0, height = 6.0, dpi_type = "line")
  cat("  NIA curve done.\n")
  return(invisible(p))
}

# =============================================================================
# --- 6. 主函数：顺序执行 ---
# =============================================================================
main <- function() {
  cat("=", strrep("=", 76), "=\n", sep = "")
  cat("  06_Clinical_Plots_R.R — SCI标准临床预测模型图表生成\n")
  cat("=", strrep("=", 76), "=\n", sep = "")
  
  # 读取推荐模型
  model_name <- tryCatch(read_model_name(), error = function(e) {
    cat("  ERROR: ", e$message, "\n")
    cat("  Using default: MLP\n")
    return("MLP")
  })
  
  cat(sprintf("Model: %s\n", model_name))
  cat(sprintf("Output dir: %s\n", OUTPUT_DIR))
  cat("\n")
  
  # [1/10] 亚组森林图
  tryCatch(plot_subgroup_forest(model_name), error = function(e) {
    cat("  ERROR in subgroup forest plot: ", e$message, "\n")
  })
  
  # [2/10] 风险分层点图
  tryCatch(plot_risk_stratification(), error = function(e) {
    cat("  ERROR in risk stratification: ", e$message, "\n")
  })
  
  # [2.5/10] 临床影响曲线
  tryCatch(plot_clinical_impact(model_name), error = function(e) {
    cat("  ERROR in clinical impact curve: ", e$message, "\n")
  })
  
  # [3/10] 校准曲线（Raw 主图）
  tryCatch(plot_calibration_curves(model_name, use_calibrated = FALSE, file_suffix = ""),
           error = function(e) {
    cat("  ERROR in calibration curves (raw): ", e$message, "\n")
  })
  # [4/10] 校准曲线（Platt 补充图）
  tryCatch(plot_calibration_curves(model_name, use_calibrated = TRUE, file_suffix = ""),
           error = function(e) {
    cat("  ERROR in calibration curves (platt): ", e$message, "\n")
  })
  # [5/10] 校准曲线（Intercept 补充图）
  tryCatch(plot_calibration_curves(model_name, use_calibrated = TRUE,
                                   file_suffix = "_intercept_recalibrated"),
           error = function(e) {
    cat("  ERROR in calibration curves (intercept): ", e$message, "\n")
  })
  
  # [6/10] 决策曲线 DCA（Raw 主图）
  tryCatch(plot_dca_curves(model_name, use_calibrated = FALSE, file_suffix = ""),
           error = function(e) {
    cat("  ERROR in DCA curves (raw): ", e$message, "\n")
  })
  # [7/10] 决策曲线 DCA（Platt 补充图）
  tryCatch(plot_dca_curves(model_name, use_calibrated = TRUE, file_suffix = ""),
           error = function(e) {
    cat("  ERROR in DCA curves (platt): ", e$message, "\n")
  })
  # [8/10] 决策曲线 DCA（Intercept 补充图）
  tryCatch(plot_dca_curves(model_name, use_calibrated = TRUE,
                           file_suffix = "_intercept_recalibrated"),
           error = function(e) {
    cat("  ERROR in DCA curves (intercept): ", e$message, "\n")
  })
  
  # [9/10] 连续净收益曲线
  tryCatch(plot_continuous_nb(model_name), error = function(e) {
    cat("  ERROR in continuous net benefit: ", e$message, "\n")
  })
  
  # [10/10] 避免不必要干预数（NIA）
  tryCatch(plot_net_intervention_avoided(model_name), error = function(e) {
    cat("  ERROR in NIA curve: ", e$message, "\n")
  })
  
  # ---- 输出文件重命名为论文最终文件名（Figure2 / eFigure1 / eFigure2）----
  rename_to_final <- function(tmp_base, final_base) {
    for (ext in c("tif", "pdf", "png")) {
      tmp_f <- file.path(OUTPUT_DIR, paste0(tmp_base, ".", ext))
      final_f <- file.path(OUTPUT_DIR, paste0(final_base, ".", ext))
      if (file.exists(tmp_f)) {
        if (file.rename(tmp_f, final_f)) {
          cat(sprintf("  Renamed: %s -> %s\n", basename(tmp_f), basename(final_f)))
        } else {
          cat(sprintf("  WARNING: rename failed for %s\n", basename(tmp_f)))
        }
      }
    }
  }
  
  # 校准曲线三套：Raw 主图 / Platt 补充图 / Intercept 补充图
  rename_to_final(paste0("Calibration_Curves_Pre_Calibration_", model_name),
                  paste0("Figure2_Main_Calibration_Raw_", model_name))
  rename_to_final(paste0("Calibration_Curves_", model_name),
                  paste0("eFigure1_Calibration_Platt_", model_name))
  rename_to_final(paste0("Calibration_Curves_Intercept_", model_name),
                  paste0("eFigure2_Calibration_Intercept_", model_name))
  # DCA 三套：Raw 主图 / Platt 补充图 / Intercept 补充图
  rename_to_final(paste0("DCA_Curves_Pre_Calibration_", model_name),
                  paste0("Figure2_Main_DCA_Raw_", model_name))
  rename_to_final(paste0("DCA_Curves_", model_name),
                  paste0("eFigure1_DCA_Platt_", model_name))
  rename_to_final(paste0("DCA_Curves_Intercept_", model_name),
                  paste0("eFigure2_DCA_Intercept_", model_name))
  
  cat("\n")
  cat("=", strrep("=", 76), "=\n", sep = "")
  cat("  所有图表生成完成！\n")
  cat("=", strrep("=", 76), "=\n", sep = "")
}

# 执行
main()