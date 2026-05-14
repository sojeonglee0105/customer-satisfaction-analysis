"""모델 팩토리, 5-seed 학습, 평가 지표, 통계 검정."""
from __future__ import annotations

import time
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


MODEL_LIST = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "LightGBM",
]
if XGB_AVAILABLE:
    MODEL_LIST.append("XGBoost")

DEFAULT_HPARAMS = {
    "Logistic Regression": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
    "Decision Tree":       {"max_depth": 8, "min_samples_split": 2},
    "Random Forest":       {"n_estimators": 200, "max_depth": None},
    "LightGBM":            {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31},
    "XGBoost":             {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
}

NEEDS_SCALING = {"Logistic Regression"}


def make_model(name: str, seed: int, class_weight: str | None,
               hparams: dict | None = None) -> Any:
    """모델 인스턴스 생성. class_weight=None or 'balanced'."""
    p = (hparams or DEFAULT_HPARAMS.get(name, {})).copy()
    if name == "Logistic Regression":
        return LogisticRegression(random_state=seed, class_weight=class_weight, **p)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed, class_weight=class_weight, **p)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=seed, class_weight=class_weight,
                                       n_jobs=-1, **p)
    if name == "LightGBM" and LGB_AVAILABLE:
        return lgb.LGBMClassifier(random_state=seed, class_weight=class_weight,
                                    verbose=-1, **p)
    if name == "XGBoost" and XGB_AVAILABLE:
        return xgb.XGBClassifier(random_state=seed, eval_metric="mlogloss",
                                   use_label_encoder=False, n_jobs=-1, **p)
    raise ValueError(f"Unknown or unavailable model: {name}")


# ── 평가 ───────────────────────────────────────────────────────────
def evaluate(model: Any, X_tr: np.ndarray, y_tr: np.ndarray,
              X_te: np.ndarray, y_te: np.ndarray,
              all_classes: list | None = None) -> dict:
    """모델 학습 후 Macro/Micro F1·AUROC + Accuracy + train_sec 반환."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_sec = time.perf_counter() - t0

    y_pred = model.predict(X_te)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_te)
    except Exception:
        pass

    all_classes = all_classes or sorted(np.unique(np.concatenate([y_tr, y_te])))
    present = sorted(np.unique(y_te))
    present_idx = [all_classes.index(c) for c in present]

    macro_f1 = f1_score(y_te, y_pred, labels=present, average="macro", zero_division=0)
    micro_f1 = f1_score(y_te, y_pred, labels=present, average="micro", zero_division=0)
    acc      = accuracy_score(y_te, y_pred)

    macro_auroc = micro_auroc = np.nan
    if y_prob is not None and len(present) >= 2:
        try:
            classes_attr = list(getattr(model, "classes_", all_classes))
            n_full = len(all_classes)
            y_prob_full = np.zeros((len(y_te), n_full))
            for i, c in enumerate(all_classes):
                if c in classes_attr:
                    y_prob_full[:, i] = y_prob[:, classes_attr.index(c)]
            y_te_bin = label_binarize(y_te, classes=all_classes)[:, present_idx]
            y_prob_sub = y_prob_full[:, present_idx]
            if y_te_bin.ndim == 1:
                y_te_bin = y_te_bin.reshape(-1, 1)
            macro_auroc = roc_auc_score(y_te_bin, y_prob_sub, average="macro", multi_class="ovr")
            micro_auroc = roc_auc_score(y_te_bin, y_prob_sub, average="micro")
        except Exception:
            pass

    return {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "macro_f1":    round(float(macro_f1), 4),
        "micro_f1":    round(float(micro_f1), 4),
        "macro_auroc": round(float(macro_auroc), 4) if not np.isnan(macro_auroc) else None,
        "micro_auroc": round(float(micro_auroc), 4) if not np.isnan(micro_auroc) else None,
        "accuracy":    round(float(acc), 4),
        "train_sec":   round(float(train_sec), 4),
    }


# ── 5-seed 반복 실험 ──────────────────────────────────────────────
def run_repeated(model_names: list[str],
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray,  y_test: np.ndarray,
                  class_weight: str | None,
                  hparams_map: dict[str, dict] | None,
                  seeds: list[int],
                  progress_cb=None) -> dict:
    """모든 모델 × 모든 seed 학습. progress_cb(frac, message)로 진행 보고."""
    hparams_map = hparams_map or {}
    all_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    needs_scale = any(n in NEEDS_SCALING for n in model_names)
    if needs_scale:
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)
    else:
        X_tr_sc = X_te_sc = None

    raw: dict[int, dict[str, dict]] = {}
    last_models: dict[str, Any] = {}
    total = len(seeds) * len(model_names)
    done = 0
    for seed in seeds:
        raw[seed] = {}
        for name in model_names:
            mdl = make_model(name, seed, class_weight, hparams_map.get(name))
            X_tr = X_tr_sc if name in NEEDS_SCALING else X_train
            X_te = X_te_sc if name in NEEDS_SCALING else X_test
            res = evaluate(mdl, X_tr, y_train, X_te, y_test, all_classes=all_classes)
            raw[seed][name] = {k: v for k, v in res.items() if k != "model"}
            last_models[name] = res["model"]
            done += 1
            if progress_cb:
                progress_cb(done / total, f"Seed {seed} / {name}")
    return {
        "raw": raw,
        "last_models": last_models,
        "all_classes": all_classes,
        "X_tr_sc": X_tr_sc, "X_te_sc": X_te_sc,
        "class_weight": class_weight,
        "seeds": seeds,
        "model_names": model_names,
    }


# ── 통계 집계 ─────────────────────────────────────────────────────
METRIC_KEYS = ["macro_f1", "micro_f1", "macro_auroc", "micro_auroc", "accuracy"]
METRIC_LABELS = {
    "macro_f1": "Macro F1", "micro_f1": "Micro F1",
    "macro_auroc": "Macro AUROC", "micro_auroc": "Micro AUROC",
    "accuracy": "Accuracy",
}

def aggregate_results(raw: dict, model_names: list[str], seeds: list[int]) -> pd.DataFrame:
    """Mean ± Std DataFrame 반환 (long-form)."""
    rows = []
    for name in model_names:
        for k in METRIC_KEYS:
            vals = [raw[s][name].get(k) for s in seeds]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if not vals:
                rows.append({"model": name, "metric": k, "mean": None, "std": None, "values": []})
                continue
            rows.append({
                "model": name, "metric": k,
                "mean": round(float(np.mean(vals)), 4),
                "std":  round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                "values": vals,
            })
        train_secs = [raw[s][name]["train_sec"] for s in seeds]
        rows.append({"model": name, "metric": "train_sec",
                     "mean": round(float(np.mean(train_secs)), 4),
                     "std":  round(float(np.std(train_secs, ddof=1)) if len(train_secs) > 1 else 0.0, 4),
                     "values": train_secs})
    return pd.DataFrame(rows)


def aggregate_pivot(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Mean 값을 모델 × 지표 매트릭스로 피벗."""
    return agg_df.pivot(index="model", columns="metric", values="mean")


# ── 통계 검정 ─────────────────────────────────────────────────────
def friedman_test(raw: dict, model_names: list[str], seeds: list[int],
                   metric: str) -> dict:
    """Friedman test — 4모델 × 5seed."""
    if len(model_names) < 3 or len(seeds) < 3:
        return {"statistic": None, "pvalue": None,
                "note": "샘플 또는 모델 수 부족 (>=3 필요)"}
    groups = []
    for m in model_names:
        vals = [raw[s][m].get(metric) for s in seeds]
        if any(v is None for v in vals):
            return {"statistic": None, "pvalue": None, "note": "결측 지표"}
        groups.append(vals)
    s, p = stats.friedmanchisquare(*groups)
    return {"statistic": round(float(s), 4), "pvalue": round(float(p), 4)}


def pairwise_wilcoxon(raw: dict, model_names: list[str], seeds: list[int],
                       metric: str) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank + Bonferroni 보정."""
    pairs = list(combinations(model_names, 2))
    n_pairs = len(pairs)
    rows = []
    for a, b in pairs:
        v1 = [raw[s][a].get(metric) for s in seeds]
        v2 = [raw[s][b].get(metric) for s in seeds]
        if any(v is None for v in v1 + v2):
            rows.append({"model_a": a, "model_b": b, "stat": None,
                          "pvalue": None, "pvalue_bonf": None})
            continue
        diff = [x - y for x, y in zip(v1, v2)]
        if all(d == 0 for d in diff):
            rows.append({"model_a": a, "model_b": b, "stat": 0.0,
                          "pvalue": 1.0, "pvalue_bonf": 1.0})
            continue
        try:
            s, p = stats.wilcoxon(v1, v2, alternative="two-sided")
            rows.append({"model_a": a, "model_b": b,
                          "stat": round(float(s), 4),
                          "pvalue": round(float(p), 4),
                          "pvalue_bonf": round(min(float(p) * n_pairs, 1.0), 4)})
        except Exception:
            rows.append({"model_a": a, "model_b": b, "stat": None,
                          "pvalue": None, "pvalue_bonf": None})
    return pd.DataFrame(rows)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list | None = None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)
