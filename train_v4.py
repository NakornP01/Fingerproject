"""
train_v4.py
-----------
v4 of the Siamese fingerprint trainer. Builds on v3's subject-disjoint split
and addresses the residual "different finger, same person -> false positive"
cases that v3 still misses on real (live-scan / cross-domain) inputs.

Improvements over v3:
  1. Heavier hard negatives:
       v3:  33% pos / 33% diff-person / 33% same-person-diff-finger
       v4:  25% pos / 25% diff-person / 50% same-person-diff-finger
     (Doubles the proportion of the negative case the system still struggles
     with.)
  2. Symmetric merge: replace asymmetric Subtract([a, b]) with
     Lambda(abs(a - b)). The API no longer needs to predict both orderings.
  3. Per-image data augmentation (rotation +/-10 deg, translation +/-5%,
     contrast/brightness jitter) applied independently to each image of a
     pair. Improves robustness to capture-time variation (rotation, pressure,
     contrast) typical of live-scan sensors.
  4. Slightly deeper feature extractor (3 conv blocks vs 2). Adds a Dropout
     before the final classifier to reduce overfitting on the larger augmented
     pair set.
  5. Reuses `subject_split.json` from v3 unchanged -- evaluation on the same
     200 held-out subjects gives an apples-to-apples comparison with v3.

Input resolution stays at 90x90 (matches the preprocessed npz arrays; the
source SOCOFing BMPs are 96x103 so a higher target would be pure
interpolation).

Saves:
  - V4/fingerprint_model_v4.h5
  - V4/training_history_v4.json

Usage (in conda env "fingerprint"):
    python V4/train_v4.py
"""
import os
import sys
import json
import time
import random
import numpy as np
import cv2
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

try:
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
except Exception:
    pass

# --- Paths ------------------------------------------------------------------
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATASET      = os.path.join(PROJECT_ROOT, "dataset")
SPLIT_PATH   = os.path.join(PROJECT_ROOT, "subject_split.json")
MODEL_OUTPUT = os.path.join(HERE, "fingerprint_model_v4.h5")
HIST_OUTPUT  = os.path.join(HERE, "training_history_v4.json")

# --- Config -----------------------------------------------------------------
SEED          = 42
IMG_SIZE      = 90
N_PAIRS       = 24000
EPOCHS        = 15
BATCH_SIZE    = 32
VAL_FRACTION  = 0.10

# Pair mix
P_POSITIVE        = 0.25
P_NEG_DIFF_PERSON = 0.25
# remainder (0.50) = same-person-different-finger

# Augmentation
AUG_PROB     = 0.7
AUG_ROT_DEG  = 10.0
AUG_SHIFT    = 0.05
AUG_ALPHA    = (0.85, 1.15)
AUG_BETA_PCT = 0.10

random.seed(SEED)
np.random.seed(SEED)

# --- Load subject-disjoint split from v3 ------------------------------------
print("=" * 72, flush=True)
if not os.path.exists(SPLIT_PATH):
    raise FileNotFoundError(
        f"Subject split not found: {SPLIT_PATH}\n"
        "Run train_v3_disjoint.py first to create subject_split.json."
    )
with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split = json.load(f)
train_subjects = set(split["train_subjects"])
test_subjects  = set(split["test_subjects"])
print(f"Reusing v3 subject split: {len(train_subjects)} train / "
      f"{len(test_subjects)} test  (overlap: 0)", flush=True)

# --- Load preprocessed SOCOFing arrays --------------------------------------
print("\nLoading preprocessed SOCOFing arrays...", flush=True)
with np.load(os.path.join(DATASET, "x_real.npz")) as f:
    x_real = f["data"]
y_real = np.load(os.path.join(DATASET, "y_real.npy"))
with np.load(os.path.join(DATASET, "x_easy.npz")) as f:
    x_easy = f["data"]
y_easy = np.load(os.path.join(DATASET, "y_easy.npy"))
print(f"  x_real: {x_real.shape}   y_real: {y_real.shape}", flush=True)
print(f"  x_easy: {x_easy.shape}   y_easy: {y_easy.shape}", flush=True)

# Filter to train subjects only
real_mask = np.isin(y_real[:, 0], list(train_subjects))
easy_mask = np.isin(y_easy[:, 0], list(train_subjects))
x_real_tr, y_real_tr = x_real[real_mask], y_real[real_mask]
x_easy_tr, y_easy_tr = x_easy[easy_mask], y_easy[easy_mask]
print(f"\nAfter filtering to {len(train_subjects)} train subjects:", flush=True)
print(f"  Real:        {len(x_real_tr):,} images", flush=True)
print(f"  Altered-Easy: {len(x_easy_tr):,} images", flush=True)

# --- Build keyed dictionaries -----------------------------------------------
# Keep images as uint8 here; convert + augment at pair-build time.
real_db = {}
for img, lbl in zip(x_real_tr, y_real_tr):
    real_db[tuple(int(v) for v in lbl)] = img.reshape(IMG_SIZE, IMG_SIZE)

altered_list = []
for img, lbl in zip(x_easy_tr, y_easy_tr):
    k = tuple(int(v) for v in lbl)
    if k in real_db:
        altered_list.append((k, img.reshape(IMG_SIZE, IMG_SIZE)))

real_keys = list(real_db.keys())
person_to_keys = {}
for k in real_keys:
    person_to_keys.setdefault(k[0], []).append(k)
print(f"  real_db entries:  {len(real_db):,}", flush=True)
print(f"  altered probes:   {len(altered_list):,}", flush=True)

# --- Augmentation -----------------------------------------------------------
def augment(img_u8: np.ndarray) -> np.ndarray:
    """
    Apply geometric + photometric augmentation to a single (H, W) uint8 image.
    Returns a (H, W) uint8 image.
    """
    if random.random() > AUG_PROB:
        return img_u8

    h, w = img_u8.shape
    cx, cy = w / 2.0, h / 2.0

    angle = random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG)
    tx    = random.uniform(-AUG_SHIFT, AUG_SHIFT) * w
    ty    = random.uniform(-AUG_SHIFT, AUG_SHIFT) * h
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(
        img_u8, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,  # SOCOFing background is white
    )

    alpha = random.uniform(*AUG_ALPHA)
    beta  = random.uniform(-AUG_BETA_PCT, AUG_BETA_PCT) * 255.0
    out = warped.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def to_float(img_u8: np.ndarray) -> np.ndarray:
    return (img_u8.astype(np.float32) / 255.0).reshape(IMG_SIZE, IMG_SIZE, 1)


# --- Generate balanced pairs ------------------------------------------------
print(f"\nGenerating {N_PAIRS} pairs ({P_POSITIVE:.0%} pos / "
      f"{P_NEG_DIFF_PERSON:.0%} diff-person / "
      f"{1 - P_POSITIVE - P_NEG_DIFF_PERSON:.0%} same-person-diff-finger)...",
      flush=True)

img1_list, img2_list, label_list = [], [], []
counts = {"pos": 0, "neg_diff_person": 0, "neg_same_person_diff_finger": 0,
          "fallback": 0}

cum_pos = P_POSITIVE
cum_neg = P_POSITIVE + P_NEG_DIFF_PERSON

for _ in range(N_PAIRS):
    r = random.random()
    if r < cum_pos:
        k, alt = random.choice(altered_list)
        a, b = augment(alt), augment(real_db[k])
        img1_list.append(to_float(a)); img2_list.append(to_float(b))
        label_list.append(1.0)
        counts["pos"] += 1
    elif r < cum_neg:
        k1 = random.choice(real_keys); k2 = random.choice(real_keys)
        while k2[0] == k1[0]:
            k2 = random.choice(real_keys)
        a, b = augment(real_db[k1]), augment(real_db[k2])
        img1_list.append(to_float(a)); img2_list.append(to_float(b))
        label_list.append(0.0)
        counts["neg_diff_person"] += 1
    else:
        k1 = random.choice(real_keys)
        sibs = [k for k in person_to_keys[k1[0]] if k != k1]
        if sibs:
            k2 = random.choice(sibs)
            a, b = augment(real_db[k1]), augment(real_db[k2])
            img1_list.append(to_float(a)); img2_list.append(to_float(b))
            label_list.append(0.0)
            counts["neg_same_person_diff_finger"] += 1
        else:
            others = [k for k in real_keys if k[0] != k1[0]]
            k2 = random.choice(others)
            a, b = augment(real_db[k1]), augment(real_db[k2])
            img1_list.append(to_float(a)); img2_list.append(to_float(b))
            label_list.append(0.0)
            counts["fallback"] += 1

print(f"  Pair counts: {counts}", flush=True)
img1_arr = np.array(img1_list, dtype=np.float32)
img2_arr = np.array(img2_list, dtype=np.float32)
labels   = np.array(label_list, dtype=np.float32)

# Pair-level val split
idx = np.random.permutation(len(labels))
img1_arr = img1_arr[idx]; img2_arr = img2_arr[idx]; labels = labels[idx]
split_at = int(len(labels) * (1 - VAL_FRACTION))
tr_i1, vl_i1 = img1_arr[:split_at], img1_arr[split_at:]
tr_i2, vl_i2 = img2_arr[:split_at], img2_arr[split_at:]
tr_y,  vl_y  = labels[:split_at],   labels[split_at:]
print(f"  Train pairs: {len(tr_y):,}   Val pairs: {len(vl_y):,}", flush=True)

# --- Build Siamese model ----------------------------------------------------
print("\nBuilding v4 Siamese model (symmetric abs-diff, 3-conv extractor)...",
      flush=True)

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

# Symmetric merge: abs(a - b). Makes score(x1, x2) == score(x2, x1).
diff = layers.Lambda(lambda t: K.abs(t[0] - t[1]),
                     name="abs_diff",
                     output_shape=lambda s: s[0])([a, b])

h = layers.Conv2D(32, 3, padding="same", activation="relu")(diff)
h = layers.MaxPooling2D(2)(h)
h = layers.Flatten()(h)
h = layers.Dense(64, activation="relu")(h)
h = layers.Dropout(0.3)(h)
out = layers.Dense(1, activation="sigmoid")(h)

model = Model(inputs=[x1, x2], outputs=out)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# --- Train ------------------------------------------------------------------
print(f"\nTraining up to {EPOCHS} epochs (EarlyStopping patience=3)...", flush=True)
es = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1,
)
history = model.fit(
    [tr_i1, tr_i2], tr_y,
    validation_data=([vl_i1, vl_i2], vl_y),
    batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[es], verbose=1,
)

# --- Save -------------------------------------------------------------------
model.save(MODEL_OUTPUT)
print(f"\nModel saved -> {MODEL_OUTPUT}", flush=True)
print(f"Final val_loss: {history.history['val_loss'][-1]:.4f}", flush=True)
print(f"Final val_acc:  {history.history['val_accuracy'][-1]*100:.2f}%", flush=True)

with open(HIST_OUTPUT, "w", encoding="utf-8") as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()},
              f, indent=2)
print(f"Training history -> {HIST_OUTPUT}", flush=True)

print("\nNext step: run `python V4/eval_v4.py` to evaluate on the 200 held-out subjects.",
      flush=True)
