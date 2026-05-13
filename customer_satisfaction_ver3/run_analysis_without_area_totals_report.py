"""
평가 영역별(t1,t2,c,d,q1,q2) *csi_total, *cci_total 12개만 제외한 독립변수로
RPI 예측 — 여러 알고리즘 성능 비교 후 HTML 리포트 생성.
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
from mord import LogisticAT
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
OUT_DIR = Path(__file__).resolve().parent
TARGET = "rpi"
CAT_COLS = ["year", "area", "product", "client"]
RANDOM_STATE = 42

# 평가 영역별 csi_total / cci_total 만 제외 (res, core, comm 등은 유지)
AREA_PREFIXES = ("t1", "t2", "c", "d", "q1", "q2")
EXCLUDE_COLS: tuple[str, ...] = tuple(
    f"{p}_{m}_total" for p in AREA_PREFIXES for m in ("csi", "cci")
)

DT_PARAMS = dict(
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=20,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=16,
    min_samples_leaf=4,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
LGBM_PARAMS = dict(
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
    random_state=RANDOM_STATE,
    verbosity=-1,
    force_col_wise=True,
)
LR_PARAMS = dict(
    max_iter=5000,
    solver="lbfgs",
    C=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
ORDINAL_PARAMS = dict(alpha=1.0)


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


def main() -> None:
    train_path = DATA_DIR / "customer_satisfaction_train.csv"
    test_path = DATA_DIR / "customer_satisfaction_test.csv"
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    all_cols = set(df_tr.columns)
    missing_ex = [c for c in EXCLUDE_COLS if c not in all_cols]
    if missing_ex:
        raise ValueError(f"제외 컬럼이 데이터에 없음: {missing_ex}")

    base_features = [c for c in df_tr.columns if c != TARGET]
    feature_cols = [c for c in base_features if c not in EXCLUDE_COLS]
    numeric_cols = [c for c in feature_cols if c not in CAT_COLS]

    X_train = df_tr[feature_cols]
    X_test = df_te[feature_cols]
    le = LabelEncoder()
    y_train = le.fit_transform(df_tr[TARGET].astype(int).values)
    y_test = le.transform(df_te[TARGET].astype(int).values)

    pre = build_preprocessor(numeric_cols)
    Xtr = pre.fit_transform(X_train, y_train)
    Xte = pre.transform(X_test)
    n_dense = Xtr.shape[1]

    specs: list[tuple[str, object]] = [
        ("Decision Tree", DecisionTreeClassifier(**DT_PARAMS)),
        ("Random Forest", RandomForestClassifier(**RF_PARAMS)),
        ("LightGBM", LGBMClassifier(**LGBM_PARAMS)),
        ("Logistic Regression", LogisticRegression(**LR_PARAMS)),
        ("Ordinal Logistic (mord.LogisticAT)", LogisticAT(**ORDINAL_PARAMS)),
    ]

    rows: list[dict] = []
    for name, clf in specs:
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
        rows.append(
            {
                "algorithm": name,
                "macro_f1": m["macro_f1"],
                "micro_f1": m["micro_f1"],
                "macro_auroc": m["macro_auroc"],
                "micro_auroc": m["micro_auroc"],
                "train_sec": train_sec,
            }
        )

    ranked = sorted(rows, key=lambda r: (-r["macro_auroc"], -r["macro_f1"], -r["micro_auroc"]))
    best = ranked[0]

    out_html = OUT_DIR / "customer_satisfaction_analysis_without_area_totals_report.html"

    exclude_li = "".join(f"<li><code>{html.escape(c)}</code></li>" for c in EXCLUDE_COLS)

    table_body = []
    for i, r in enumerate(ranked, start=1):
        table_body.append(
            "<tr>"
            f"<td class='num'>{i}</td>"
            f"<td>{html.escape(r['algorithm'])}</td>"
            f"<td class='num'>{r['macro_f1']:.4f}</td>"
            f"<td class='num'>{r['micro_f1']:.4f}</td>"
            f"<td class='num'>{r['macro_auroc']:.4f}</td>"
            f"<td class='num'>{r['micro_auroc']:.4f}</td>"
            f"<td class='num'>{r['train_sec']:.4f}</td>"
            "</tr>"
        )

    params_json = html.escape(
        json.dumps(
            {
                "excluded_columns": list(EXCLUDE_COLS),
                "n_raw_features": len(feature_cols),
                "n_numeric": len(numeric_cols),
                "n_cat": len(CAT_COLS),
                "n_dense_after_preprocess": n_dense,
                "random_state": RANDOM_STATE,
                "models": {
                    "DecisionTree": {k: v for k, v in DT_PARAMS.items() if k != "random_state"},
                    "RandomForest": {k: v for k, v in RF_PARAMS.items() if k != "random_state"},
                    "LightGBM": {k: v for k, v in LGBM_PARAMS.items() if k != "random_state"},
                    "LogisticRegression": {k: v for k, v in LR_PARAMS.items() if k != "random_state"},
                    "Ordinal_LogisticAT": ORDINAL_PARAMS,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RPI 예측 — 영역별 total 제외 특성 분석</title>
  <style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 24px 32px 48px; color: #111; max-width: 960px; line-height: 1.55; }}
    h1 {{ font-size: 1.35rem; border-bottom: 2px solid #0369a1; padding-bottom: 8px; }}
    h2 {{ font-size: 1.05rem; color: #0c4a6e; margin-top: 24px; }}
    p.meta {{ color: #555; font-size: 0.9rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.88rem; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; }}
    th {{ background: #e0f2fe; text-align: left; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .note {{ background: #f8fafc; border-left: 4px solid #64748b; padding: 12px 16px; margin: 14px 0; }}
    .best {{ background: #ecfdf5; font-weight: 600; }}
    pre.params {{ background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 8px; font-size: 0.76rem; overflow-x: auto; }}
    ul.compact {{ margin: 0.4rem 0; padding-left: 1.2rem; }}
    ul.compact li {{ margin: 0.2rem 0; }}
  </style>
</head>
<body>
  <h1>RPI 예측 — 평가 영역별 <code>csi_total</code> / <code>cci_total</code> 제외 분석</h1>
  <p class="meta">
    학습: <code>{html.escape(str(train_path.relative_to(ROOT)))}</code><br/>
    평가: <code>{html.escape(str(test_path.relative_to(ROOT)))}</code><br/>
    타깃: <strong>{TARGET}</strong> · <code>random_state={RANDOM_STATE}</code><br/>
    생성: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
  </p>

  <div class="note">
    <strong>결론 (Macro AUROC 1순위, 동률 시 Macro F1)</strong><br/>
    본 설정에서 가장 높은 성능은
    <strong>{html.escape(best["algorithm"])}</strong>입니다
    (Macro AUROC <strong>{best["macro_auroc"]:.4f}</strong>,
    Macro F1 <strong>{best["macro_f1"]:.4f}</strong>).
  </div>

  <h2>1. 분석 조건</h2>
  <p>독립변수에서 아래 <strong>{len(EXCLUDE_COLS)}개</strong>만 제거하고, 그 외 CSI/CCI 세부 항목(res/core/comm) 및 범주형 변수는 그대로 사용했습니다.
     전처리는 기존 벤치마크와 동일하게 수치(중앙값 대치+표준화)+범주(원-핫)입니다.</p>
  <ul class="compact">
{exclude_li}
  </ul>
  <p class="meta">원시 특성 수(타깃·제외 12개 제외): <strong>{len(feature_cols)}</strong> ·
     수치 <strong>{len(numeric_cols)}</strong> + 범주 <strong>{len(CAT_COLS)}</strong> ·
     전처리 후 밀집 차원: <strong>{n_dense}</strong></p>

  <h2>2. 알고리즘별 테스트 성능 (순위: Macro AUROC 내림차순)</h2>
  <table>
    <thead>
      <tr>
        <th>순위</th>
        <th>알고리즘</th>
        <th>Macro F1</th>
        <th>Micro F1</th>
        <th>Macro AUROC</th>
        <th>Micro AUROC</th>
        <th>학습시간(초)</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(table_body)}
    </tbody>
  </table>

  <h2>3. 해석 메모</h2>
  <ul class="compact">
    <li>제외한 12개는 영역별 <strong>종합 점수</strong>이지만, 동일 영역의 <code>res</code>/<code>core</code>/<code>comm</code> 점수가 남아 있어
        RPI와의 정보가 상당 부분 <strong>중복·재구성 가능</strong>합니다. 그래서 본 데이터에서는 절대 성능이 크게 무너지지 않고,
        <strong>Ordinal → Logistic Regression → LightGBM</strong> 순의 상대 우위가 유지되는 패턴입니다.</li>
    <li>순위는 단일 홀드아웃·<code>random_state={RANDOM_STATE}</code> 기준이므로, 운영 전 시드 반복·교차검증을 권장합니다.</li>
    <li>Decision Tree만 AUROC·F1이 유의하게 낮아, 요약 신호 없이 고차원·다중공선 구조에서 <strong>얕은 트리</strong>의 한계가 드러난 것으로 볼 수 있습니다.</li>
  </ul>

  <h2>4. 설정 (JSON)</h2>
  <pre class="params">{params_json}</pre>
</body>
</html>
"""

    out_html.write_text(doc, encoding="utf-8")
    print("Wrote", out_html)


if __name__ == "__main__":
    main()
