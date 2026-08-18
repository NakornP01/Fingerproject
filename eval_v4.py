"""
eval_v4.py
----------
Open-set evaluation of v4 on the 200 held-out subjects from
`../subject_split.json`. Mirrors `eval_disjoint.py` so the numbers are
directly comparable to v3.

Adds one focused metric: same-person-different-finger FAR. This is the case
the user observed v3 still failing on (e.g. 99.96% on visibly different
fingers of the same person). v4 targets this specifically by raising the
hard-negative ratio to 50%.

Outputs (in V4/):
  - scores_cache_v4.npz
  - roc_curve_v4.png
  - score_distribution_v4.png
  - confusion_matrix_v4.png
  - score_stats_v4.csv
  - threshold_results_v4.csv
  - model_comparison_v3_vs_v4.csv
  - hard_neg_v4.csv          (same-person-different-finger FAR breakdown)

Usage:
    python V4/eval_v4.py
"""
import os
import sys
import csv
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except Exception:
    pass

# --- Paths ------------------------------------------------------------------
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATASET      = os.path.join(PROJECT_ROOT, "dataset")
MODEL_PATH   = os.path.join(HERE, "fingerprint_model_v4.h5")
SPLIT_PATH   = os.path.join(PROJECT_ROOT, "subject_split.json")

SEED        = 42
N_PAIRS     = 4000
THRESHOLD   = 0.70
IMG_SIZE    = 90

random.seed(SEED)
np.random.seed(SEED)

# --- Sanity checks ----------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"v4 model not found: {MODEL_PATH}\n"
        "Train it first:  python V4/train_v4.py"
    )
if not os.path.exists(SPLIT_PATH):
    raise FileNotFoundError(f"Subject split not found: {SPLIT_PATH}")

# --- Load subject split + data ----------------------------------------------
with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split = json.load(f)
test_subjects = set(split["test_subjects"])
print(f"Test subjects: {len(test_subjects)} (held out)", flush=True)

with np.load(os.path.join(DATASET, "x_real.npz")) as f: x_real = f["data"]
y_real = np.load(os.path.join(DATASET, "y_real.npy"))
with np.load(os.path.join(DATASET, "x_hard.npz")) as f: x_hard = f["data"]
y_hard = np.load(os.path.join(DATASET, "y_hard.npy"))

real_mask = np.isin(y_real[:, 0], list(test_subjects))
hard_mask = np.isin(y_hard[:, 0], list(test_subjects))
x_real_te, y_real_te = x_real[real_mask], y_real[real_mask]
x_hard_te, y_hard_te = x_hard[hard_mask], y_hard[hard_mask]
print(f"  Real (held-out):        {len(x_real_te):,}", flush=True)
print(f"  Altered-Hard (held-out): {len(x_hard_te):,}", flush=True)


def to_f(a):
    return (a.astype(np.float32) / 255.0).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


real_db = {}
for img, lbl in zip(x_real_te, y_real_te):
    real_db[tuple(int(v) for v in lbl)] = to_f(img[np.newaxis])[0]
real_keys = list(real_db.keys())

person_to_keys = {}
for k in real_keys:
    person_to_keys.setdefault(k[0], []).append(k)

hard_list = []
for img, lbl in zip(x_hard_te, y_hard_te):
    k = tuple(int(v) for v in lbl)
    if k in real_db:
        hard_list.append((k, to_f(img[np.newaxis])[0]))
print(f"  Hard probes matched to gallery: {len(hard_list):,}", flush=True)


# --- Build v4 architecture & load weights -----------------------------------
def build_v4():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    f = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    f = layers.MaxPooling2D(2)(f)
    f = layers.Conv2D(64, 3, padding="same", activation="relu")(f)
    f = layers.MaxPooling2D(2)(f)
    f = layers.Conv2D(64, 3, padding="same", activation="relu")(f)
    f = layers.MaxPooling2D(2)(f)
    feat = Model(inputs=inp, outputs=f, name="feature_extractor")

    x1 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x2 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    a = feat(x1); b = feat(x2)
    diff = layers.Lambda(lambda t: K.abs(t[0] - t[1]),
                         name="abs_diff",
                         output_shape=lambda s: s[0])([a, b])
    h = layers.Conv2D(32, 3, padding="same", activation="relu")(diff)
    h = layers.MaxPooling2D(2)(h)
    h = layers.Flatten()(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(1, activation="sigmoid")(h)
    return Model(inputs=[x1, x2], outputs=out)


print(f"\nLoading v4 model: {MODEL_PATH}", flush=True)
model = build_v4()
model.load_weights(MODEL_PATH)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# --- Standard test pairs (50% genuine, 50% diff-person impostor) ------------
print(f"\nGenerating {N_PAIRS} standard test pairs...", flush=True)
i1, i2, y = [], [], []
for _ in range(N_PAIRS):
    if random.random() > 0.5:
        k, alt = random.choice(hard_list)
        i1.append(alt); i2.append(real_db[k]); y.append(1)
    else:
        k1 = random.choice(real_keys); k2 = random.choice(real_keys)
        while k2[0] == k1[0]:
            k2 = random.choice(real_keys)
        i1.append(real_db[k1]); i2.append(real_db[k2]); y.append(0)
img1 = np.array(i1, dtype=np.float32)
img2 = np.array(i2, dtype=np.float32)
y_true = np.array(y, dtype=np.int32)

print("Running inference...", flush=True)
y_score = model.predict([img1, img2], batch_size=64, verbose=1).flatten()
np.savez(os.path.join(HERE, "scores_cache_v4.npz"),
         y_true=y_true, y_score=y_score)

# --- Metrics ----------------------------------------------------------------
fpr, tpr, thr = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
fnr = 1 - tpr
eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
eer_thresh = thr[eer_idx]

y_pred = (y_score >= THRESHOLD).astype(np.int32)
TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
acc = (TP + TN) / (TP + TN + FP + FN)
far = FP / (FP + TN) if (FP + TN) else 0
frr = FN / (FN + TP) if (FN + TP) else 0

print(f"\n=== v4 open-set metrics ({len(test_subjects)} held-out subjects) ===",
      flush=True)
print(f"  AUC       : {roc_auc:.4f}", flush=True)
print(f"  EER       : {eer*100:.2f}%  (threshold {eer_thresh:.4f})", flush=True)
print(f"  Accuracy@{int(THRESHOLD*100)}%: {acc*100:.2f}%", flush=True)
print(f"  FAR       : {far*100:.2f}%", flush=True)
print(f"  FRR       : {frr*100:.2f}%", flush=True)
print(f"  TP={TP} TN={TN} FP={FP} FN={FN}", flush=True)

# --- Targeted: same-person-different-finger FAR -----------------------------
# This is the case v4 is built to fix.
print("\n=== Same-person-different-finger FAR (the v4 target case) ===",
      flush=True)
sp_pairs = []
for k1 in real_keys:
    sibs = [k for k in person_to_keys[k1[0]] if k != k1]
    for k2 in sibs:
        sp_pairs.append((real_db[k1], real_db[k2]))
random.shuffle(sp_pairs)
sp_pairs = sp_pairs[:2000]
if sp_pairs:
    sp_i1 = np.array([p[0] for p in sp_pairs], dtype=np.float32)
    sp_i2 = np.array([p[1] for p in sp_pairs], dtype=np.float32)
    sp_scores = model.predict([sp_i1, sp_i2], batch_size=64, verbose=0).flatten()
    sp_rows = [["Threshold", "FAR (%)", "Mean score", "P95 score", "Max score"]]
    for t in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        far_sp = (sp_scores >= t).mean() * 100
        sp_rows.append([t, f"{far_sp:.2f}", f"{sp_scores.mean():.4f}",
                        f"{np.percentile(sp_scores, 95):.4f}",
                        f"{sp_scores.max():.4f}"])
        print(f"  thr={t:.2f}  FAR={far_sp:5.2f}%  "
              f"mean={sp_scores.mean():.4f}  "
              f"p95={np.percentile(sp_scores, 95):.4f}  "
              f"max={sp_scores.max():.4f}", flush=True)
    with open(os.path.join(HERE, "hard_neg_v4.csv"), "w", newline="",
              encoding="utf-8") as f:
        csv.writer(f).writerows(sp_rows)
    print("Saved hard_neg_v4.csv", flush=True)

# --- ROC curve --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, color="#065A82", lw=2, label=f"v4 ROC (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
ax.scatter(fpr[eer_idx], tpr[eer_idx], color="red", s=60, zorder=5,
           label=f"EER = {eer*100:.2f}% (thr={eer_thresh:.3f})")
ax.set_xlabel("False Match Rate")
ax.set_ylabel("True Match Rate")
ax.set_title(f"ROC — v4 (open-set, {len(test_subjects)} unseen subjects)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "roc_curve_v4.png"), dpi=150)
plt.close()
print("\nSaved roc_curve_v4.png", flush=True)

# --- Score distribution -----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
gen = y_score[y_true == 1]; imp = y_score[y_true == 0]
ax.hist(imp, bins=50, alpha=0.6, color="#EF4444", label=f"Impostor (n={len(imp)})")
ax.hist(gen, bins=50, alpha=0.6, color="#10B981", label=f"Genuine (n={len(gen)})")
ax.axvline(THRESHOLD, color="black", linestyle="--", lw=1.5,
           label=f"Threshold = {THRESHOLD}")
ax.set_xlabel("Similarity score")
ax.set_ylabel("Count")
ax.set_title("Score Distribution — v4 (open-set)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "score_distribution_v4.png"), dpi=150)
plt.close()
print("Saved score_distribution_v4.png", flush=True)

# --- Confusion matrix -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap([[TN, FP], [FN, TP]], annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted NO", "Predicted YES"],
            yticklabels=["Actual NO", "Actual YES"],
            annot_kws={"size": 16, "weight": "bold"}, ax=ax)
ax.set_title(f"Confusion Matrix — v4\nThreshold={THRESHOLD}  "
             f"Accuracy={acc*100:.2f}%  N={N_PAIRS}")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "confusion_matrix_v4.png"), dpi=150)
plt.close()
print("Saved confusion_matrix_v4.png", flush=True)

# --- Score stats CSV --------------------------------------------------------
def stats(a):
    return [f"{a.mean():.6f}", f"{np.median(a):.6f}", f"{a.std():.6f}",
            f"{a.min():.6f}", f"{a.max():.6f}",
            f"{np.percentile(a,25):.6f}", f"{np.percentile(a,75):.6f}"]


with open(os.path.join(HERE, "score_stats_v4.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Genuine", "Impostor"])
    headers = ["Mean", "Median", "Std", "Min", "Max", "P25", "P75"]
    gs, is_ = stats(gen), stats(imp)
    for h, g, i in zip(headers, gs, is_):
        w.writerow([h, g, i])
    w.writerow(["Count", len(gen), len(imp)])
    w.writerow(["Separation gap (min_gen - max_imp)",
                f"{gen.min() - imp.max():.6f}", ""])
print("Saved score_stats_v4.csv", flush=True)

# --- Threshold sweep --------------------------------------------------------
thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
with open(os.path.join(HERE, "threshold_results_v4.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Threshold", "TP", "TN", "FP", "FN", "Accuracy",
                "FAR", "FRR", "Precision", "Recall", "F1"])
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        a = (tp + tn) / (tp + tn + fp + fn)
        fa = fp / (fp + tn) if (fp + tn) else 0
        fr = fn / (fn + tp) if (fn + tp) else 0
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        w.writerow([t, tp, tn, fp, fn, f"{a:.4f}", f"{fa:.4f}",
                    f"{fr:.4f}", f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])
print("Saved threshold_results_v4.csv", flush=True)

# --- v3 vs v4 comparison ----------------------------------------------------
v3_metrics = {
    "AUC":              1.0000,
    "EER (%)":          0.05,
    "Accuracy@0.70 (%)": 99.30,
    "FAR@0.70 (%)":     0.00,
    "FRR@0.70 (%)":     1.38,
}
with open(os.path.join(HERE, "model_comparison_v3_vs_v4.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "v3 (disjoint)", "v4 (disjoint + hard-neg + aug)"])
    w.writerow(["AUC",                f"{v3_metrics['AUC']:.4f}",              f"{roc_auc:.4f}"])
    w.writerow(["EER (%)",            f"{v3_metrics['EER (%)']:.2f}",          f"{eer*100:.2f}"])
    w.writerow(["Accuracy@0.70 (%)",  f"{v3_metrics['Accuracy@0.70 (%)']:.2f}", f"{acc*100:.2f}"])
    w.writerow(["FAR@0.70 (%)",       f"{v3_metrics['FAR@0.70 (%)']:.2f}",     f"{far*100:.2f}"])
    w.writerow(["FRR@0.70 (%)",       f"{v3_metrics['FRR@0.70 (%)']:.2f}",     f"{frr*100:.2f}"])
print("Saved model_comparison_v3_vs_v4.csv", flush=True)

print("\n" + "=" * 72, flush=True)
print("DONE — v4 open-set evaluation complete.", flush=True)
print("=" * 72, flush=True)
