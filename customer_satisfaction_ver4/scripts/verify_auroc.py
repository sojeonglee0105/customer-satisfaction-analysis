"""신규 합성 데이터로 학습 시 AUROC가 충분히 높은지 검증.

다음 모델을 빠르게 학습하여 Macro/Micro AUROC, Macro F1, Accuracy를 측정한다.
- Logistic Regression
- Random Forest
- LightGBM (optional)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
train_df = pd.read_csv(DATA_DIR / "lgd_csat_strong_signal_train.csv")
test_df  = pd.read_csv(DATA_DIR / "lgd_csat_strong_signal_test.csv")

target = "rpi"
meta_cols = ["year", "area", "product", "client"]
feature_cols = [c for c in train_df.columns if c not in meta_cols + [target]]

X_train = train_df[feature_cols].values
y_train = train_df[target].values
X_test  = test_df[feature_cols].values
y_test  = test_df[target].values

print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}, Features: {len(feature_cols)}")
print(f"[INFO] Train class: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"[INFO] Test  class: {dict(zip(*np.unique(y_test, return_counts=True)))}")
print()

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(
        random_state=42, max_iter=2000, class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced",
    ),
}

try:
    import lightgbm as lgb
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, random_state=42,
        class_weight="balanced", verbose=-1,
    )
except Exception:
    pass

results = []
for name, mdl in models.items():
    if name == "Logistic Regression":
        Xtr, Xte = X_train_sc, X_test_sc
    else:
        Xtr, Xte = X_train, X_test
    mdl.fit(Xtr, y_train)
    y_pred = mdl.predict(Xte)
    y_prob = mdl.predict_proba(Xte)

    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    if y_test_bin.shape[1] == 1:  # binary
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

    macro_auroc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    micro_auroc = roc_auc_score(y_test_bin, y_prob, average="micro", multi_class="ovr")
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")
    acc = accuracy_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Macro AUROC": macro_auroc,
        "Micro AUROC": micro_auroc,
        "Macro F1": macro_f1,
        "Micro F1": micro_f1,
        "Accuracy": acc,
    })

res_df = pd.DataFrame(results)
print("=" * 70)
print("[RESULTS] Model Performance on lgd_csat_strong_signal data")
print("=" * 70)
print(res_df.to_string(index=False, float_format="%.4f"))

best_au = res_df["Macro AUROC"].max()
print()
if best_au >= 0.95:
    print(f"[VERDICT] Excellent — best Macro AUROC = {best_au:.4f} (>= 0.95)")
elif best_au >= 0.90:
    print(f"[VERDICT] Strong    — best Macro AUROC = {best_au:.4f} (>= 0.90)")
elif best_au >= 0.85:
    print(f"[VERDICT] Good      — best Macro AUROC = {best_au:.4f} (>= 0.85)")
else:
    print(f"[VERDICT] Weak      — best Macro AUROC = {best_au:.4f} (< 0.85)")
