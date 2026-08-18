"""
eval_v4_advanced.py
-------------------
Score-based + threshold-tuning metrics for v4. Loads cached scores from
`scores_cache_v4.npz` (produced by eval_v4.py) so it runs in seconds with no
model inference.

Adds the metrics that eval_v4.py does not cover:
  - AUC-PR (Precision-Recall AUC)
  - d-prime (separation of genuine vs impostor distributions)
  - DET curve (NIST-standard FAR vs FRR, log-log)
  - PR curve
  - FNMR @ fixed FMR table (ISO/IEC 19795 operating points)
  - 5 optimal-threshold methods compared side-by-side:
      1. EER threshold
      2. Youden's J  (max TPR - FPR)
      3. F1-max
      4. Fixed-FAR operating point (0.01%, 0.1%, 1%)
      5. Cost-sensitive (weighted FAR/FRR)

Outputs (in V4/):
  - advanced_metrics_v4.csv         (single-number summary)
  - fnmr_at_fmr_v4.csv              (FNMR at FMR = 10%, 1%, 0.1%, 0.01%)
  - optimal_thresholds_v4.csv       (5 methods compared)
  - det_curve_v4.png
  - pr_curve_v4.png
  - threshold_methods_v4.png        (visual comparison of where each method lands)

Usage:
    python V4/eval_v4_advanced.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix,
)

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except Exception:
    pass

HERE        = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH  = os.path.join(HERE, "scores_cache_v4.npz")

if not os.path.exists(CACHE_PATH):
    raise FileNotFoundError(
        f"Score cache not found: {CACHE_PATH}\n"
        "Run V4/eval_v4.py first to generate it."
    )

print(f"Loading cached scores: {CACHE_PATH}", flush=True)
data = np.load(CACHE_PATH)
y_true  = data["y_true"].astype(int)
y_score = data["y_score"].astype(float)
print(f"  Total pairs : {len(y_true):,}", flush=True)
print(f"  Genuine     : {int((y_true == 1).sum()):,}", flush=True)
print(f"  Impostor    : {int((y_true == 0).sum()):,}", flush=True)

gen = y_score[y_true == 1]
imp = y_score[y_true == 0]

# ============================================================================
# 1. Score-based metrics (threshold-independent)
# ============================================================================
print("\n=== Score-based metrics ===", flush=True)

# AUC-ROC
fpr, tpr, roc_thr = roc_curve(y_true, y_score)
auc_roc = auc(fpr, tpr)

# AUC-PR
precision, recall, pr_thr = precision_recall_curve(y_true, y_score)
auc_pr = average_precision_score(y_true, y_score)

# d-prime: (mu_gen - mu_imp) / sqrt((var_gen + var_imp) / 2)
d_prime = (gen.mean() - imp.mean()) / np.sqrt((gen.var() + imp.var()) / 2)

# EER
fnr = 1 - tpr
eer_idx    = int(np.nanargmin(np.abs(fpr - fnr)))
eer        = (fpr[eer_idx] + fnr[eer_idx]) / 2
eer_thresh = roc_thr[eer_idx]

# Separation gap
sep_gap = gen.min() - imp.max()

print(f"  AUC-ROC : {auc_roc:.6f}", flush=True)
print(f"  AUC-PR  : {auc_pr:.6f}", flush=True)
print(f"  d-prime : {d_prime:.4f}", flush=True)
print(f"  EER     : {eer*100:.4f}% (threshold {eer_thresh:.4f})", flush=True)
print(f"  Separation gap (min_gen - max_imp): {sep_gap:.4f}", flush=True)

with open(os.path.join(HERE, "advanced_metrics_v4.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Value", "Interpretation"])
    w.writerow(["AUC-ROC",  f"{auc_roc:.6f}",  "1.0 = perfect, 0.5 = chance"])
    w.writerow(["AUC-PR",   f"{auc_pr:.6f}",   "Better than ROC under class imbalance"])
    w.writerow(["d-prime",  f"{d_prime:.4f}",  ">3 = excellent separation"])
    w.writerow(["EER (%)",  f"{eer*100:.4f}",  "Lower = better; biometric standard"])
    w.writerow(["EER threshold", f"{eer_thresh:.4f}", "Threshold where FAR == FRR"])
    w.writerow(["Separation gap", f"{sep_gap:.4f}",
                "min(genuine) - max(impostor); >0 = linearly separable"])
    w.writerow(["Genuine count",  f"{len(gen)}", ""])
    w.writerow(["Impostor count", f"{len(imp)}", ""])
print("Saved advanced_metrics_v4.csv", flush=True)

# ============================================================================
# 2. FNMR @ fixed FMR table (ISO/IEC 19795 operating points)
# ============================================================================
print("\n=== FNMR @ fixed FMR (ISO/IEC 19795) ===", flush=True)
fmr_targets = [0.10, 0.01, 0.001, 0.0001]  # 10%, 1%, 0.1%, 0.01%
fnmr_rows = [["Target FMR (%)", "Actual FMR (%)", "FNMR (%)", "Threshold",
              "Use case"]]
use_cases = {
    0.10:   "Low security / convenience",
    0.01:   "Consumer (phone unlock)",
    0.001:  "Banking / payment",
    0.0001: "High security (border, gov)",
}
for target in fmr_targets:
    idx = int(np.argmin(np.abs(fpr - target)))
    actual_fmr = fpr[idx]
    fnmr_val   = fnr[idx]
    thr_val    = roc_thr[idx]
    print(f"  FMR target {target*100:6.3f}%  ->  "
          f"actual {actual_fmr*100:6.3f}%  "
          f"FNMR {fnmr_val*100:6.2f}%  thr {thr_val:.4f}", flush=True)
    fnmr_rows.append([f"{target*100:.3f}", f"{actual_fmr*100:.4f}",
                      f"{fnmr_val*100:.4f}", f"{thr_val:.4f}",
                      use_cases[target]])
with open(os.path.join(HERE, "fnmr_at_fmr_v4.csv"), "w", newline="",
          encoding="utf-8") as f:
    csv.writer(f).writerows(fnmr_rows)
print("Saved fnmr_at_fmr_v4.csv", flush=True)

# ============================================================================
# 3. Optimal-threshold methods (5 ways)
# ============================================================================
print("\n=== Optimal threshold (5 methods) ===", flush=True)

def metrics_at(thr):
    yp = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    far = fp / (fp + tn) if (fp + tn) else 0
    frr = fn / (fn + tp) if (fn + tp) else 0
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return tp, tn, fp, fn, acc, far, frr, p, r, f1

# Method 1: EER
m1_thr = eer_thresh

# Method 2: Youden's J = TPR - FPR (max)
j = tpr - fpr
m2_idx = int(np.argmax(j))
m2_thr = roc_thr[m2_idx]

# Method 3: F1-max (sweep finely)
sweep = np.linspace(0.001, 0.999, 999)
f1s = []
for t in sweep:
    yp = (y_score >= t).astype(int)
    cm = confusion_matrix(y_true, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1s.append(2 * p * r / (p + r) if (p + r) else 0)
f1s = np.array(f1s)
m3_thr = float(sweep[int(np.argmax(f1s))])

# Method 4: Fixed-FAR operating points (0.1% chosen as the "phone unlock" default)
target_far_default = 0.001
m4_idx = int(np.argmin(np.abs(fpr - target_far_default)))
m4_thr = roc_thr[m4_idx]

# Method 5: Cost-sensitive (C_FA = 100 * C_FR)
C_FA, C_FR = 100.0, 1.0
costs = C_FA * fpr + C_FR * fnr
m5_idx = int(np.argmin(costs))
m5_thr = roc_thr[m5_idx]

methods = [
    ("1. EER (balanced)",            m1_thr,
     "Point where FAR == FRR"),
    ("2. Youden's J (max TPR-FPR)",  m2_thr,
     "Maximizes separation; rewards both correctness"),
    ("3. F1-max",                    m3_thr,
     "Best harmonic mean of Precision & Recall"),
    (f"4. Fixed FAR = {target_far_default*100:.1f}%",   m4_thr,
     "Industry operating-point pick (phone unlock)"),
    (f"5. Cost-sensitive (C_FA={int(C_FA)}, C_FR={int(C_FR)})", m5_thr,
     "Security-weighted; FAR is 100x more costly"),
]

opt_rows = [["Method", "Threshold", "TP", "TN", "FP", "FN",
             "Accuracy", "FAR", "FRR", "Precision", "Recall", "F1",
             "Recommendation"]]
print(f"\n  {'Method':<42} {'Thr':>7} {'Acc':>7} {'FAR':>7} {'FRR':>7} "
      f"{'F1':>7}", flush=True)
for name, thr, note in methods:
    tp, tn, fp, fn, acc, far, frr, p, r, f1 = metrics_at(thr)
    print(f"  {name:<42} {thr:>7.4f} {acc*100:>6.2f}% "
          f"{far*100:>6.3f}% {frr*100:>6.2f}% {f1:>7.4f}", flush=True)
    opt_rows.append([name, f"{thr:.4f}", tp, tn, fp, fn,
                     f"{acc:.4f}", f"{far:.4f}", f"{frr:.4f}",
                     f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", note])

# Also add current "0.70 default" for reference
tp, tn, fp, fn, acc, far, frr, p, r, f1 = metrics_at(0.70)
opt_rows.append(["(current default 0.70)", "0.7000", tp, tn, fp, fn,
                 f"{acc:.4f}", f"{far:.4f}", f"{frr:.4f}",
                 f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", "Reference"])

with open(os.path.join(HERE, "optimal_thresholds_v4.csv"), "w", newline="",
          encoding="utf-8") as f:
    csv.writer(f).writerows(opt_rows)
print("\nSaved optimal_thresholds_v4.csv", flush=True)

# ============================================================================
# 4. DET curve (NIST-standard)
# ============================================================================
# DET curve plots FAR vs FRR with both axes on probit (normal-deviate) scale
def probit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return norm.ppf(p)

# Filter out the (1,1) endpoints that compress the plot
mask = (fpr > 0) & (fpr < 1) & (fnr > 0) & (fnr < 1)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(probit(fpr[mask]), probit(fnr[mask]), color="#065A82", lw=2,
        label=f"v4 DET")
ax.scatter(probit(fpr[eer_idx]), probit(fnr[eer_idx]), color="red", s=80,
           zorder=5, label=f"EER = {eer*100:.2f}%")

ticks_pct = [0.01, 0.1, 1, 2, 5, 10, 20, 40]
ticks     = [probit(t / 100) for t in ticks_pct]
labels    = [f"{t:g}%" for t in ticks_pct]
ax.set_xticks(ticks); ax.set_xticklabels(labels)
ax.set_yticks(ticks); ax.set_yticklabels(labels)
ax.set_xlim(probit(0.0001), probit(0.5))
ax.set_ylim(probit(0.0001), probit(0.5))
ax.set_xlabel("False Match Rate (FMR / FAR)")
ax.set_ylabel("False Non-Match Rate (FNMR / FRR)")
ax.set_title("DET Curve - v4 (NIST-style, log-log probit)")
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "det_curve_v4.png"), dpi=150)
plt.close()
print("Saved det_curve_v4.png", flush=True)

# ============================================================================
# 5. PR curve
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall, precision, color="#10B981", lw=2,
        label=f"v4 PR (AP = {auc_pr:.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve - v4")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "pr_curve_v4.png"), dpi=150)
plt.close()
print("Saved pr_curve_v4.png", flush=True)

# ============================================================================
# 6. Threshold-methods visual comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: score histograms with each method's threshold marked
ax = axes[0]
ax.hist(imp, bins=60, alpha=0.55, color="#EF4444",
        label=f"Impostor (n={len(imp)})")
ax.hist(gen, bins=60, alpha=0.55, color="#10B981",
        label=f"Genuine (n={len(gen)})")
colors = ["#1F2937", "#7C3AED", "#F59E0B", "#0EA5E9", "#DC2626"]
for (name, thr, _), c in zip(methods, colors):
    ax.axvline(thr, color=c, linestyle="--", lw=2,
               label=f"{name.split(' ')[0]} thr={thr:.3f}")
ax.set_xlabel("Similarity score")
ax.set_ylabel("Count")
ax.set_yscale("log")
ax.set_title("Score distribution with optimal thresholds")
ax.legend(fontsize=8, loc="upper center")
ax.grid(True, alpha=0.3)

# Right: ROC with each operating point marked
ax = axes[1]
ax.plot(fpr, tpr, color="#065A82", lw=2, label=f"v4 ROC (AUC={auc_roc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
for (name, thr, _), c in zip(methods, colors):
    idx = int(np.argmin(np.abs(roc_thr - thr)))
    ax.scatter(fpr[idx], tpr[idx], color=c, s=80, zorder=5,
               label=f"{name.split(' ')[0]}")
ax.set_xlabel("FAR (False Match Rate)")
ax.set_ylabel("TPR (True Match Rate)")
ax.set_xlim(-0.01, 0.2); ax.set_ylim(0.7, 1.01)
ax.set_title("ROC curve - operating points (zoomed)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(HERE, "threshold_methods_v4.png"), dpi=150)
plt.close()
print("Saved threshold_methods_v4.png", flush=True)

# ============================================================================
# Final summary
# ============================================================================
print("\n" + "=" * 72, flush=True)
print("DONE - advanced v4 metrics computed.", flush=True)
print("=" * 72, flush=True)
print("\nFiles written to V4/:", flush=True)
for fn in ["advanced_metrics_v4.csv", "fnmr_at_fmr_v4.csv",
           "optimal_thresholds_v4.csv", "det_curve_v4.png",
           "pr_curve_v4.png", "threshold_methods_v4.png"]:
    p = os.path.join(HERE, fn)
    sz = os.path.getsize(p) if os.path.exists(p) else 0
    print(f"  {fn:<35} ({sz:,} bytes)", flush=True)
