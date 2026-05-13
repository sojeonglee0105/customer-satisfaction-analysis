"""
RPI(1~5) 분류: 전체 특성 전처리(이전 벤치마크 실험 1과 동일)로
Decision Tree / Random Forest / LightGBM / Logistic Regression 학습·평가 후 HTML 리포트.

5개 서로 다른 random_state로 동일 실험 반복 → 평균±표준편차, Friedman 검정,
베이스라인(Decision Tree) 대비 Wilcoxon 부호-순위 검정(다중비교 Bonferroni).

HTML 출력: 이 스크립트와 동일 폴더(customer_satisfaction_ver3).
"""
from __future__ import annotations

import html
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "customer_satisfaction_ver3"
REPORTS_DIR = Path(__file__).resolve().parent
TARGET = "rpi"
CAT_COLS = ["year", "area", "product", "client"]

# 5회 반복 시드 (고정; 재현성)
REPEAT_SEEDS: tuple[int, ...] = (42, 1337, 7, 99, 31415)
REFERENCE_SEED = 42  # 단일 실행 비교·이전 리포트 연속성

# 하이퍼파라미터 공통( random_state 는 실행 시 주입 )
DT_BASE = dict(
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=20,
    class_weight="balanced",
)
RF_BASE = dict(
    n_estimators=300,
    max_depth=16,
    min_samples_leaf=4,
    class_weight="balanced_subsample",
    n_jobs=-1,
)
LGBM_BASE = dict(
    objective="multiclass",
    num_class=5,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    verbosity=-1,
    force_col_wise=True,
)
LR_BASE = dict(
    max_iter=5000,
    solver="lbfgs",
    C=1.0,
    class_weight="balanced",
)


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
        ],
        remainder="drop",
    )


def as_lgbm_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(X, dtype=np.float32),
        columns=[f"f{i}" for i in range(X.shape[1])],
    )


def eval_all(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    labels = np.arange(proba.shape[1])
    out: dict[str, float] = {}
    out["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )
    out["micro_f1"] = float(
        f1_score(y_true, y_pred, average="micro", labels=labels, zero_division=0)
    )
    try:
        out["macro_auroc"] = float(
            roc_auc_score(
                y_true,
                proba,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        out["micro_auroc"] = float(
            roc_auc_score(
                y_true,
                proba,
                labels=labels,
                multi_class="ovr",
                average="micro",
            )
        )
    except ValueError:
        out["macro_auroc"] = float("nan")
        out["micro_auroc"] = float("nan")
    return out


def make_models(seed: int) -> list[tuple[str, object]]:
    return [
        ("0. Decision Tree (baseline)", DecisionTreeClassifier(**DT_BASE, random_state=seed)),
        ("1. Random Forest", RandomForestClassifier(**RF_BASE, random_state=seed)),
        ("2. LightGBM", LGBMClassifier(**LGBM_BASE, random_state=seed)),
        ("3. Logistic Regression", LogisticRegression(**LR_BASE, random_state=seed)),
    ]


def fit_eval_one(
    name: str,
    clf: object,
    Xtr: np.ndarray,
    Xte: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | str]:
    if isinstance(clf, LGBMClassifier):
        tr_in, te_in = as_lgbm_frame(Xtr), as_lgbm_frame(Xte)
    else:
        tr_in, te_in = np.asarray(Xtr), np.asarray(Xte)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        clf.fit(tr_in, y_train)
    train_sec = time.perf_counter() - t0

    pred = clf.predict(te_in)
    proba = clf.predict_proba(te_in)
    m = eval_all(y_test, pred, proba)
    return {
        "algorithm": name,
        "macro_f1": m["macro_f1"],
        "micro_f1": m["micro_f1"],
        "macro_auroc": m["macro_auroc"],
        "micro_auroc": m["micro_auroc"],
        "train_sec": train_sec,
    }


def _wilcoxon_safe(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None, str | None]:
    """Paired Wilcoxon; returns (statistic, p-value, note)."""
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return None, None, "차이가 모두 0이라 검정 생략"
    if len(d) < 3:
        return None, None, "표본 수 부족"
    try:
        r = stats.wilcoxon(a, b, alternative="two-sided", method="auto")
        return float(r.statistic), float(r.pvalue), None
    except ValueError as e:
        return None, None, str(e)


def main() -> None:
    train_path = DATA_DIR / "customer_satisfaction_train.csv"
    test_path = DATA_DIR / "customer_satisfaction_test.csv"
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    feature_cols = [c for c in df_tr.columns if c != TARGET]
    numeric_cols = [c for c in feature_cols if c not in CAT_COLS]

    X_train = df_tr[feature_cols]
    X_test = df_te[feature_cols]
    le = LabelEncoder()
    y_train = le.fit_transform(df_tr[TARGET].astype(int).values)
    y_test = le.transform(df_te[TARGET].astype(int).values)

    pre = build_preprocessor(numeric_cols)
    Xtr = pre.fit_transform(X_train, y_train)
    Xte = pre.transform(X_test)
    n_feat = Xtr.shape[1]

    # --- 반복 실험: 시드별 결과 ---
    all_runs: list[list[dict[str, float | str]]] = []
    for seed in REPEAT_SEEDS:
        row_run: list[dict[str, float | str]] = []
        for name, clf in make_models(seed):
            row_run.append(fit_eval_one(name, clf, Xtr, Xte, y_train, y_test))
        all_runs.append(row_run)

    algo_names = [r["algorithm"] for r in all_runs[0]]
    n_algo = len(algo_names)
    n_rep = len(REPEAT_SEEDS)

    metric_keys = ["macro_f1", "micro_f1", "macro_auroc", "micro_auroc", "train_sec"]
    arr = np.zeros((n_rep, n_algo, len(metric_keys)))
    for si, run in enumerate(all_runs):
        for aj, r in enumerate(run):
            for mk, mki in zip(metric_keys, range(len(metric_keys))):
                arr[si, aj, mki] = float(r[metric_keys[mki]])

    # 평균 ± 표준편차
    mean_m = arr.mean(axis=0)
    std_m = arr.std(axis=0, ddof=1) if n_rep > 1 else np.zeros_like(mean_m)

    summary_rows = []
    for aj, aname in enumerate(algo_names):
        summary_rows.append(
            {
                "algorithm": aname,
                "macro_f1": f"{mean_m[aj, 0]:.4f} ± {std_m[aj, 0]:.4f}",
                "micro_f1": f"{mean_m[aj, 1]:.4f} ± {std_m[aj, 1]:.4f}",
                "macro_auroc": f"{mean_m[aj, 2]:.4f} ± {std_m[aj, 2]:.4f}",
                "micro_auroc": f"{mean_m[aj, 3]:.4f} ± {std_m[aj, 3]:.4f}",
                "train_sec": f"{mean_m[aj, 4]:.4f} ± {std_m[aj, 4]:.4f}",
            }
        )

    # seed=42 단일 실행 (참고)
    ref_idx = REPEAT_SEEDS.index(REFERENCE_SEED) if REFERENCE_SEED in REPEAT_SEEDS else 0
    rows_single = all_runs[ref_idx]

    baseline_single = rows_single[0]
    delta_rows = []
    for r in rows_single[1:]:
        delta_rows.append(
            {
                "vs": str(r["algorithm"]),
                "d_macro_f1": float(r["macro_f1"]) - float(baseline_single["macro_f1"]),
                "d_micro_f1": float(r["micro_f1"]) - float(baseline_single["micro_f1"]),
                "d_macro_auroc": float(r["macro_auroc"]) - float(baseline_single["macro_auroc"]),
                "d_micro_auroc": float(r["micro_auroc"]) - float(baseline_single["micro_auroc"]),
                "d_time": float(r["train_sec"]) - float(baseline_single["train_sec"]),
            }
        )

    # 평균 기준 Δ
    mean_delta = []
    for j in range(1, n_algo):
        mean_delta.append(
            {
                "vs": algo_names[j],
                "d_macro_f1": mean_m[j, 0] - mean_m[0, 0],
                "d_micro_f1": mean_m[j, 1] - mean_m[0, 1],
                "d_macro_auroc": mean_m[j, 2] - mean_m[0, 2],
                "d_micro_auroc": mean_m[j, 3] - mean_m[0, 3],
                "d_time": mean_m[j, 4] - mean_m[0, 4],
            }
        )

    # --- Friedman: 지표별 4개 알고리즘 (블록 = 시드) ---
    friedman_rows = []
    for mki, mk in enumerate(metric_keys):
        samples = [arr[:, j, mki] for j in range(n_algo)]
        try:
            stat, p = stats.friedmanchisquare(*samples)
            friedman_rows.append(
                {
                    "metric": mk,
                    "statistic": float(stat),
                    "p_value": float(p),
                    "sig": "p < 0.05" if p < 0.05 else "p ≥ 0.05",
                }
            )
        except Exception as e:
            friedman_rows.append({"metric": mk, "statistic": None, "p_value": None, "sig": str(e)})

    # --- Wilcoxon: 각 비베이스라인 vs DT, 지표별 (paired by seed) ---
    n_comp = n_algo - 1
    bonf = n_comp * len(metric_keys)  # 보수적: 3알고리즘 × 5지표 동시 다중비교
    wilcox_rows = []
    for j in range(1, n_algo):
        for mk, mki in zip(metric_keys, range(len(metric_keys))):
            a = arr[:, j, mki]
            b = arr[:, 0, mki]
            stat, p_raw, note = _wilcoxon_safe(a, b)
            p_adj = min(1.0, p_raw * bonf) if p_raw is not None else None
            mean_diff = float(np.mean(a - b))
            wilcox_rows.append(
                {
                    "comparison": f"{algo_names[j]} vs {algo_names[0]}",
                    "metric": mk,
                    "mean_diff": mean_diff,
                    "statistic": stat,
                    "p_raw": p_raw,
                    "p_bonferroni": p_adj,
                    "sig_005": "예" if (p_adj is not None and p_adj < 0.05) else "아니오",
                    "note": note,
                }
            )

    # 시드별 상세 (long table)
    long_rows_html = []
    for si, seed in enumerate(REPEAT_SEEDS):
        for r in all_runs[si]:
            long_rows_html.append(
                "<tr>"
                f"<td class='num'>{seed}</td>"
                f"<td>{html.escape(str(r['algorithm']))}</td>"
                f"<td class='num'>{float(r['macro_f1']):.4f}</td>"
                f"<td class='num'>{float(r['micro_f1']):.4f}</td>"
                f"<td class='num'>{float(r['macro_auroc']):.4f}</td>"
                f"<td class='num'>{float(r['micro_auroc']):.4f}</td>"
                f"<td class='num'>{float(r['train_sec']):.4f}</td>"
                "</tr>"
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "customer_satisfaction_algorithm_comparison_report.html"

    prev_dt_macro_auc = 0.9499
    prev_dt_macro_f1 = 0.8227

    def tr_cells_summary(r: dict) -> str:
        return (
            f"<td>{html.escape(r['algorithm'])}</td>"
            f"<td class='num'>{html.escape(r['macro_f1'])}</td>"
            f"<td class='num'>{html.escape(r['micro_f1'])}</td>"
            f"<td class='num'>{html.escape(r['macro_auroc'])}</td>"
            f"<td class='num'>{html.escape(r['micro_auroc'])}</td>"
            f"<td class='num'>{html.escape(r['train_sec'])}</td>"
        )

    def tr_cells_single(r: dict) -> str:
        return (
            f"<td>{html.escape(str(r['algorithm']))}</td>"
            f"<td class='num'>{float(r['macro_f1']):.4f}</td>"
            f"<td class='num'>{float(r['micro_f1']):.4f}</td>"
            f"<td class='num'>{float(r['macro_auroc']):.4f}</td>"
            f"<td class='num'>{float(r['micro_auroc']):.4f}</td>"
            f"<td class='num'>{float(r['train_sec']):.4f}</td>"
        )

    delta_html = []
    for d in delta_rows:
        delta_html.append(
            "<tr>"
            f"<td>{html.escape(d['vs'])}</td>"
            f"<td class='num'>{d['d_macro_f1']:+.4f}</td>"
            f"<td class='num'>{d['d_micro_f1']:+.4f}</td>"
            f"<td class='num'>{d['d_macro_auroc']:+.4f}</td>"
            f"<td class='num'>{d['d_micro_auroc']:+.4f}</td>"
            f"<td class='num'>{d['d_time']:+.4f}</td>"
            "</tr>"
        )

    mean_delta_html = []
    for d in mean_delta:
        mean_delta_html.append(
            "<tr>"
            f"<td>{html.escape(d['vs'])}</td>"
            f"<td class='num'>{d['d_macro_f1']:+.4f}</td>"
            f"<td class='num'>{d['d_micro_f1']:+.4f}</td>"
            f"<td class='num'>{d['d_macro_auroc']:+.4f}</td>"
            f"<td class='num'>{d['d_micro_auroc']:+.4f}</td>"
            f"<td class='num'>{d['d_time']:+.4f}</td>"
            "</tr>"
        )

    fried_html = []
    for fr in friedman_rows:
        st = f"{fr['statistic']:.4f}" if fr["statistic"] is not None else "—"
        pv = f"{fr['p_value']:.6f}" if fr["p_value"] is not None else "—"
        fried_html.append(
            "<tr>"
            f"<td>{html.escape(fr['metric'])}</td>"
            f"<td class='num'>{st}</td>"
            f"<td class='num'>{pv}</td>"
            f"<td>{html.escape(str(fr['sig']))}</td>"
            "</tr>"
        )

    wilcox_html = []
    for w in wilcox_rows:
        pr = f"{w['p_raw']:.6f}" if w["p_raw"] is not None else "—"
        pa = f"{w['p_bonferroni']:.6f}" if w["p_bonferroni"] is not None else "—"
        st = f"{w['statistic']:.6f}" if w["statistic"] is not None else "—"
        note = w["note"] or ""
        wilcox_html.append(
            "<tr>"
            f"<td>{html.escape(w['comparison'])}</td>"
            f"<td>{html.escape(w['metric'])}</td>"
            f"<td class='num'>{w['mean_diff']:+.6f}</td>"
            f"<td class='num'>{st}</td>"
            f"<td class='num'>{pr}</td>"
            f"<td class='num'>{pa}</td>"
            f"<td>{html.escape(w['sig_005'])}</td>"
            f"<td>{html.escape(note)}</td>"
            "</tr>"
        )

    seeds_str = ", ".join(str(s) for s in REPEAT_SEEDS)
    params_json = html.escape(
        json.dumps(
            {
                "repeat_seeds": list(REPEAT_SEEDS),
                "n_repeats": n_rep,
                "reference_single_seed": REFERENCE_SEED,
                "feature_engineering": "실험 1: 전체 특성 (중앙값 대치+StandardScaler, OneHotEncoder)",
                "n_features_after_preprocess": n_feat,
                "DecisionTree": {**DT_BASE, "random_state": "<per run>"},
                "RandomForest": {**RF_BASE, "random_state": "<per run>"},
                "LightGBM": {**LGBM_BASE, "random_state": "<per run>"},
                "LogisticRegression": {**LR_BASE, "random_state": "<per run>"},
                "friedman": "지표별 Friedman 검정 — 동일 시드에서 4알고리즘 점수를 블록으로 취급",
                "wilcoxon": f"베이스라인 대비 쌍체 Wilcoxon, Bonferroni 보정 계수={bonf} (3알고리즘×5지표)",
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    body_summary = "\n".join("<tr>" + tr_cells_summary(r) + "</tr>" for r in summary_rows)
    body_single = "\n".join("<tr>" + tr_cells_single(r) + "</tr>" for r in rows_single)

    doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>고객만족 알고리즘 비교 리포트</title>
  <style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 24px 32px 48px; color: #111; max-width: 1100px; line-height: 1.55; }}
    h1 {{ font-size: 1.4rem; border-bottom: 2px solid #0d9488; padding-bottom: 8px; }}
    h2 {{ font-size: 1.05rem; color: #134e4a; margin-top: 26px; }}
    h3 {{ font-size: 0.98rem; color: #0f766e; margin-top: 18px; }}
    p.meta {{ color: #555; font-size: 0.9rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 0.86rem; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 7px 9px; }}
    th {{ background: #ecfdf5; text-align: left; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .note {{ background: #f8fafc; border-left: 4px solid #64748b; padding: 12px 16px; margin: 14px 0; }}
    pre.params {{ background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 8px; font-size: 0.74rem; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>고객만족 RPI 예측 — 알고리즘 비교</h1>
  <p class="meta">
    학습: <code>{html.escape(str(train_path.relative_to(ROOT)))}</code><br/>
    평가: <code>{html.escape(str(test_path.relative_to(ROOT)))}</code><br/>
    타깃: <strong>{TARGET}</strong> (1~5 등급, 지표 계산용 0~4 인코딩)<br/>
    생성: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
  </p>

  <div class="note">
    <strong>Feature engineering</strong><br/>
    이전 <code>customer_satisfaction_fe_benchmark_report.html</code>에서 테스트 Macro AUROC가 가장 높았던 조합과 동일한
    <strong>실험 1·전체 특성</strong> 전처리(수치: 중앙값 대치+StandardScaler, 범주: OneHotEncoder, <strong>{n_feat}</strong>차원)를 사용합니다.
  </div>

  <div class="note">
    <strong>5회 반복 실험 (random seed)</strong><br/>
    동일 데이터·동일 하이퍼파라미터에서 <code>random_state</code>만
    <strong>{html.escape(seeds_str)}</strong> 로 바꿔 각각 전체 파이프라인을 학습·평가했습니다 (총 <strong>{n_rep}</strong>회/알고리즘).
    트리/포레스트/LightGBM의 내부 무작위성과 로지스틱 수치해에 영향을 줍니다. 표본 수가 작아(n={n_rep}) 검정은 탐색적 해석에 한합니다.
  </div>

  <h2>1. 학습 결과표 (5회 반복: 평균 ± 표본표준편차)</h2>
  <p class="meta">동일 테스트 분할에서의 지표입니다. AUROC는 one-vs-rest, F1은 sklearn 정의의 Macro/Micro입니다.</p>
  <table>
    <thead>
      <tr>
        <th>알고리즘</th>
        <th>Macro F1</th>
        <th>Micro F1</th>
        <th>Macro AUROC</th>
        <th>Micro AUROC</th>
        <th>학습시간 (초)</th>
      </tr>
    </thead>
    <tbody>
{body_summary}
    </tbody>
  </table>

  <h2>2. 통계 검정</h2>
  <h3>2-1. Friedman 검정 (알고리즘 간 차이, 시드=블록)</h3>
  <p class="meta">각 지표에 대해 5개 시드에서의 4개 알고리즘 점수를 대응 표본으로 두고 Friedman χ² 검정을 수행했습니다.
     유의하면 “시드에 따른 변동을 통제한 상태에서 알고리즘 간 분포가 동일하지 않다”는 탐색적 결론입니다.</p>
  <table>
    <thead>
      <tr><th>지표</th><th>χ² 통계량</th><th>p-value</th><th>유의성 (α=0.05)</th></tr>
    </thead>
    <tbody>
{chr(10).join(fried_html)}
    </tbody>
  </table>

  <h3>2-2. Wilcoxon 부호-순위 검정 (vs Decision Tree, 시드별 쌍체)</h3>
  <p class="meta">각 지표·각 비교마다 5개 시드에서의 점수 쌍에 Wilcoxon <code>two-sided</code> 검정.
     다중비교 완화를 위해 원시 p-value에 Bonferroni 계수 <strong>{bonf}</strong> (3 알고리즘 × 5 지표)를 곱해 <code>p_bonferroni</code>를 보고했습니다.
     <code>mean_diff</code>는 (비교 알고리즘 − 베이스라인)의 시드별 평균입니다.</p>
  <table>
    <thead>
      <tr>
        <th>비교</th><th>지표</th><th>mean_diff</th><th>통계량</th><th>p (raw)</th><th>p (Bonferroni)</th><th>p_adj&lt;0.05?</th><th>비고</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(wilcox_html)}
    </tbody>
  </table>

  <h2>3. 시드별 상세 결과 (원시값)</h2>
  <table>
    <thead>
      <tr>
        <th>random_state</th><th>알고리즘</th><th>Macro F1</th><th>Micro F1</th><th>Macro AUROC</th><th>Micro AUROC</th><th>학습시간(초)</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(long_rows_html)}
    </tbody>
  </table>

  <h2>4. 대표 실행 (random_state={REFERENCE_SEED}) — 직전 리포트와 동일 조건</h2>
  <p class="meta">단일 시드 기준 표·베이스라인 대비 차이(절대차)입니다.</p>
  <table>
    <thead>
      <tr>
        <th>알고리즘</th>
        <th>Macro F1</th>
        <th>Micro F1</th>
        <th>Macro AUROC</th>
        <th>Micro AUROC</th>
        <th>학습시간 (초)</th>
      </tr>
    </thead>
    <tbody>
{body_single}
    </tbody>
  </table>

  <h3>4-1. Decision Tree 베이스라인 대비 차이 (seed={REFERENCE_SEED})</h3>
  <table>
    <thead>
      <tr>
        <th>알고리즘</th>
        <th>Δ Macro F1</th>
        <th>Δ Micro F1</th>
        <th>Δ Macro AUROC</th>
        <th>Δ Micro AUROC</th>
        <th>Δ 학습시간 (초)</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(delta_html)}
    </tbody>
  </table>

  <h3>4-2. 베이스라인 대비 평균 차이 (5회 평균 지표 기준)</h3>
  <table>
    <thead>
      <tr>
        <th>알고리즘</th>
        <th>Δ Macro F1</th>
        <th>Δ Micro F1</th>
        <th>Δ Macro AUROC</th>
        <th>Δ Micro AUROC</th>
        <th>Δ 학습시간 (초)</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(mean_delta_html)}
    </tbody>
  </table>

  <div class="note">
    <strong>이전 실험 Decision Tree 참고값</strong><br/>
    <code>customer_satisfaction_fe_benchmark_report.html</code> 실험 1 기준 Macro AUROC <strong>{prev_dt_macro_auc}</strong>,
    Macro F1 <strong>{prev_dt_macro_f1}</strong>. 대표 실행(seed={REFERENCE_SEED})의 Decision Tree 행과 비교하시면 됩니다.
  </div>

  <h2>5. 하이퍼파라미터·설정 (JSON)</h2>
  <pre class="params">{params_json}</pre>
</body>
</html>
"""

    out.write_text(doc, encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
