import os, pandas as pd, numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "Model Development")

summary_df = pd.read_csv(os.path.join(RESULT_DIR, "final_model_summary.csv"))
hl_df = pd.read_csv(os.path.join(RESULT_DIR, "hosmer_lemeshow_results.csv"))
delong_df = pd.read_csv(os.path.join(RESULT_DIR, "delong_test_results.csv"))
model_name = summary_df["Model"].dropna().iloc[0] if "Model" in summary_df.columns and not summary_df.empty else "Unknown"

report_lines = []
report_lines.append("=" * 70)
report_lines.append(f"Model Development Report — {model_name}")
report_lines.append("=" * 70)
report_lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Model: {model_name}")
report_lines.append("")
report_lines.append("Note: Derivation cohort metrics are nested-CV estimates from 04_Model_Selection.py.")
report_lines.append("")
report_lines.append("-" * 50)
report_lines.append("Performance Metrics")
report_lines.append("-" * 50)

for _, row in summary_df.iterrows():
    cohort = row["Cohort"]
    report_lines.append(f"\n  {cohort}:")
    # AUPRC with CI (explicit formatting for clarity)
    if pd.notna(row.get("AUPRC")):
        report_lines.append(f"    AUPRC: {row['AUPRC']:.3f} (95% CI: {row['AUPRC_CI_Lower']:.3f}-{row['AUPRC_CI_Upper']:.3f})")
    for col in summary_df.columns:
        if col in ("Cohort", "Model"):
            continue
        if col.startswith("AUPRC"):  # skip already-printed AUPRC entries
            continue
        val = row[col]
        if pd.notna(val):
            report_lines.append(f"    {col}: {val:.3f}")

report_lines.append("\n" + "-" * 50)
report_lines.append("Independent AUC Comparison (bootstrap z-test, independent samples)")
report_lines.append("-" * 50)
for _, r in delong_df.iterrows():
    report_lines.append(
        f"  {r['Comparison']}: AUC diff={r['AUC_Diff']:.3f} "
        f"({r['AUC_Diff_95CI_Lower']:.3f}-{r['AUC_Diff_95CI_Upper']:.3f}), "
        f"Z={r['Z_Statistic']:.3f}, p={r['P_Value']:.3f} "
        f"(Bonferroni: {r['P_Value_Bonferroni']:.3f})"
    )

report_lines.append("\n" + "-" * 50)
report_lines.append("Hosmer-Lemeshow Test")
report_lines.append("-" * 50)
for _, r in hl_df.iterrows():
    report_lines.append(f"  {r['Cohort']}: Chi2={r['Chi2']:.3f}, df={r['DF']}, p={r['P_Value']:.3f}")

report_path = os.path.join(RESULT_DIR, "model_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Report saved: {report_path}")