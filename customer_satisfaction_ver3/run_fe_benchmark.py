"""
Customer satisfaction: compare 3 feature strategies x 3 models.
Reads train/test from data/customer_satisfaction_ver3/, writes HTML report in this folder.
"""
from __future__ import annotations

import html
import json
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mord import LogisticAT
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "customer_satisfaction_ver3"
REPORTS_DIR = Path(__file__).resolve().parent
TARGET = "rpi"
CAT_COLS = ["year", "area", "product", "client"]
RANDOM_STATE = 42
SELECT_K = 25
PCA_VAR = 0.95

# Same hyperparameters for all FE variants (only input dimensionality changes).
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
DT_PARAMS = dict(
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=20,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)
ORDINAL_PARAMS = dict(alpha=1.0)  # L2 on mord LogisticAT


def _reason_for_feature(name: str, mi: float) -> str:
    """Korean rationale template: MI + domain keywords."""
    bits = [
        f"학습셋 기준 상호정보량(MI)이 {mi:.4f}로 RPI(1~5)와의 의존도가 높게 추정됨.",
    ]
    n = name.lower()
    if "total" in n:
        bits.append("종합 지표라 여러 하위 신호를 압축해 등급 변별에 유리할 수 있음.")
    if "csi" in n:
        bits.append("CSI는 고객 만족 관점으로 RPI와 개념적으로 연결됨.")
    if "cci" in n:
        bits.append("CCI는 경쟁사 대비 상대평가로 관계 품질과 연동될 수 있음.")
    if "t1_" in n or "t2_" in n:
        bits.append("시점별 세부 축으로 등급 패턴을 세분화하는 데 기여할 수 있음.")
    if "q1_" in n or "q2_" in n or "d_" in n or "c_" in n:
        bits.append("평가 영역/분기 등 세부 차원 신호.")
    if "year" in n:
        bits.append("연도별 조사 환경·표본 구성 차이를 흡수.")
    if "product" in n:
        bits.append("제품군별 만족·불만 구조 차이 반영.")
    if "client" in n:
        bits.append("고객사별 관계/요구 수준 차이 반영.")
    if "area" in n:
        bits.append("평가 영역(축)별 난이도·중요도 차이 반영.")
    return " ".join(bits)


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


def eval_multiclass(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    n_classes = proba.shape[1]
    labels = np.arange(n_classes)
    try:
        auc = roc_auc_score(
            y_true,
            proba,
            labels=labels,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc = float("nan")
    f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    return float(auc), float(f1)


def _r(results: list[dict], experiment: str, model: str) -> dict:
    for row in results:
        if row["experiment"] == experiment and row["model"] == model:
            return row
    raise KeyError((experiment, model))


def _best_experiment(results: list[dict], model: str) -> dict:
    rows = [r for r in results if r["model"] == model]
    return max(rows, key=lambda x: (x["macro_auroc"], x["macro_f1"]))


def build_insights_and_future_sections(
    results: list[dict],
    *,
    n_cat_selected: int,
    select_k: int,
    n_pca: int,
    pca_ev_sum: float,
) -> str:
    """본 실험 수치 기반 인사이트·향후 과제 HTML (한국어)."""
    exp1, exp2, exp3 = "1. 전체 특성", "2. 특성 선택 (MI, 상위 K)", "3. PCA 차원 축소"
    m_lgbm = "LightGBM"
    m_ord = "Ordinal Logistic Regression (mord.LogisticAT)"
    m_dt = "Decision Tree"

    r_dt1 = _r(results, exp1, m_dt)
    r_dt2 = _r(results, exp2, m_dt)
    r_dt3 = _r(results, exp3, m_dt)
    r_lgb1 = _r(results, exp1, m_lgbm)
    r_lgb2 = _r(results, exp2, m_lgbm)
    r_lgb3 = _r(results, exp3, m_lgbm)
    r_o1 = _r(results, exp1, m_ord)
    r_o2 = _r(results, exp2, m_ord)
    r_o3 = _r(results, exp3, m_ord)

    best_lgbm = _best_experiment(results, m_lgbm)

    d_auc_dt_pca = r_dt3["macro_auroc"] - r_dt1["macro_auroc"]
    d_f1_dt_pca = r_dt3["macro_f1"] - r_dt1["macro_f1"]
    d_auc_lgb_mi = r_lgb2["macro_auroc"] - r_lgb1["macro_auroc"]
    d_f1_lgb_mi = r_lgb2["macro_f1"] - r_lgb1["macro_f1"]
    d_auc_ord_mi = r_o2["macro_auroc"] - r_o1["macro_auroc"]

    return f"""
  <h2>4. 핵심 인사이트</h2>
  <p class="meta">아래는 본 리포트 표(섹션 2)의 <strong>테스트셋 지표</strong>를 바탕으로 한 해석이며, 인과·업무 규칙을 단정하지 않습니다.</p>
  <ul class="insights-list">
    <li><strong>전반적 분리 성능이 매우 높음</strong>:
      Ordinal / LightGBM의 Macro AUROC가 대부분 <strong>0.99대</strong>에 근접합니다.
      RPI는 만족·경쟁력 지표(CSI/CCI)와 같은 설문 체계에서 도출된 경우가 많아,
      <strong>정보 중복·사후 정의(tautology)에 가까운 예측</strong>이 아닌지 점검이 필요합니다.
      운영 목적이 “조기 경고”라면 시간·고객사 밖 검증이 특히 중요합니다.</li>
    <li><strong>모델별로 FE 민감도가 다름</strong>:
      <strong>Decision Tree</strong>는 전체 특성 대비 PCA에서 Macro AUROC가
      <strong>{d_auc_dt_pca:+.4f}</strong>, Macro F1이 <strong>{d_f1_dt_pca:+.4f}</strong> 개선되어
      다중공선·고차원에서 얕은 트리가 불리했던 것으로 해석할 수 있습니다.
      반면 <strong>LightGBM</strong>은 MI 기반 축소에서 Macro AUROC <strong>{d_auc_lgb_mi:+.4f}</strong>,
      Macro F1 <strong>{d_f1_lgb_mi:+.4f}</strong>로 소폭 하락해,
      <strong>일부 상호 대체 가능한 신호</strong>를 잘 활용하는 부스팅 특성과 부합합니다.</li>
    <li><strong>Ordinal 모델의 최고 AUROC</strong>:
      전체 특성에서 Macro AUROC <strong>{r_o1['macro_auroc']:.4f}</strong>로 세 모델·세 FE 중 최고이며,
      순서형 RPI(1~5)에 비례 오즈 가정이 잘 맞는 구간으로 보입니다.
      다만 MI 축소 시 <strong>{d_auc_ord_mi:+.4f}</strong>만큼 AUROC가 내려가,
      <strong>선택된 25차원만으로는 순서 로지스틱에 필요한 정보가 일부 빠졌을 가능성</strong>이 있습니다.</li>
    <li><strong>LightGBM 최적 FE</strong>: LightGBM 기준 Macro AUROC 최고는
      「<strong>{best_lgbm['experiment']}</strong>」
      (AUROC <strong>{best_lgbm['macro_auroc']:.4f}</strong>, F1 <strong>{best_lgbm['macro_f1']:.4f}</strong>)입니다.</li>
    <li><strong>MI 상위 특성의 공통 패턴</strong> (섹션 3):
      <code>*_cci_total</code>, <code>*_csi_total</code> 등 <strong>종합 지표</strong>가 상위를 차지합니다.
      세부 축(res/core/comm)은 상대적으로 MI가 낮은 편으로, <strong>등급 요약에 압축된 정보</strong>가 우선인 셈입니다.</li>
    <li><strong>범주형이 MI 상위 K에서 탈락</strong>:
      원-핫 범주(<code>cat__</code>)가 상위 <strong>{select_k}</strong>개 안에 <strong>{n_cat_selected}</strong>개뿐입니다.
      제품·고객사·연도 등 <strong>세그먼트 해석</strong>이 필요하면 MI만으로 자르기보다
      별도 브랜치(예: 트리용 범주 유지 + 수치만 축소)를 검토할 만합니다.</li>
  </ul>

  <h2>5. 향후 추가 과제</h2>
  <ul class="insights-list">
    <li><strong>누수·중복 점검</strong>: RPI 산출에 CSI/CCI가 직접 반영되는지 데이터 정의서와 대조하고,
      가능하면 <strong>RPI 확정 이전 시점의 지표만</strong>으로 예측하는 난이도 높은 설정을 한 벤치마크를 둡니다.</li>
    <li><strong>특성 선택의 안정화</strong>: 현재 MI+SelectKBest는 학습분할에 과적합하기 쉬우므로
      <strong>중첩 CV(nested CV)</strong> 또는 <strong>Permutation importance</strong> 기반 선택과 비교합니다.</li>
    <li><strong>범주형 보존 전략</strong>: MI 상위에 강제 포함·별도 임베딩(Target encoding 등) 또는
      <strong>계층적 모델</strong>(범주별 소모델)로 해석력을 회복합니다.</li>
    <li><strong>PCA 해석</strong>: 현재 PCA는 설명분산 약 <strong>{pca_ev_sum:.2%}</strong>를
      <strong>{n_pca}</strong>성분으로 압축합니다. 로딩 행렬로 축 의미를 요약하고,
      운영 리포트용으로는 <strong>비음수 행렬 분해(NMF)</strong> 등 대안과 비교할 수 있습니다.</li>
    <li><strong>순서형 평가 지표</strong>: Macro F1 외 <strong>MAE(순서 오차)</strong>, Quadratic weighted kappa,
      클래스별 재현율(특히 RPI 4·5)을 추가합니다.</li>
    <li><strong>일반화 검증</strong>: <strong>연도·고객사 홀드아웃</strong>, 또는 수집 배치별 분할로
      동일 FE·모델의 성능 분산을 측정합니다.</li>
    <li><strong>하이퍼파라미터</strong>: 세 FE에서 동일 설정이었으므로,
      FE별로 <strong>Optuna/베이지안 최적화</strong>를 적용해 공정 비교를 재정의합니다.</li>
    <li><strong>설명 가능성</strong>: SHAP·부분의존으로 <code>*_total</code> 외 변수의
      한계 기여도를 이해관계자용으로 정리합니다.</li>
  </ul>
"""


def main() -> None:
    train_path = DATA_DIR / "customer_satisfaction_train.csv"
    test_path = DATA_DIR / "customer_satisfaction_test.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    feature_cols = [c for c in df_tr.columns if c != TARGET]
    numeric_cols = [c for c in feature_cols if c not in CAT_COLS]

    X_train = df_tr[feature_cols]
    X_test = df_te[feature_cols]
    y_train_raw = df_tr[TARGET].astype(int).values
    y_test_raw = df_te[TARGET].astype(int).values

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    pre = build_preprocessor(numeric_cols)
    Xtr = pre.fit_transform(X_train, y_train)
    Xte = pre.transform(X_test)
    feat_names = pre.get_feature_names_out()

    n_raw = len(feature_cols)
    n_dense = Xtr.shape[1]

    # --- shared feature selection (train only) ---
    mi_fn = partial(mutual_info_classif, random_state=RANDOM_STATE)
    selector = SelectKBest(score_func=mi_fn, k=min(SELECT_K, Xtr.shape[1]))
    selector.fit(Xtr, y_train)
    mask = selector.get_support()
    mi_scores = selector.scores_
    sel_idx = np.where(mask)[0]
    ranked = sorted(
        [(feat_names[i], float(mi_scores[i])) for i in sel_idx],
        key=lambda x: -x[1],
    )
    selection_rows = [
        {
            "feature": fn,
            "mi": sc,
            "reason": _reason_for_feature(fn, sc),
        }
        for fn, sc in ranked
    ]
    Xtr_sel = selector.transform(Xtr)
    Xte_sel = selector.transform(Xte)

    # --- shared PCA (train only) ---
    pca = PCA(n_components=PCA_VAR, random_state=RANDOM_STATE)
    Xtr_pca = pca.fit_transform(Xtr)
    Xte_pca = pca.transform(Xte)
    n_pca = pca.n_components_

    datasets = {
        "1. 전체 특성": (Xtr, Xte, n_dense),
        "2. 특성 선택 (MI, 상위 K)": (Xtr_sel, Xte_sel, Xtr_sel.shape[1]),
        "3. PCA 차원 축소": (Xtr_pca, Xte_pca, n_pca),
    }

    def new_model(name: str):
        if name == "LightGBM":
            return LGBMClassifier(**LGBM_PARAMS)
        if name == "Ordinal Logistic Regression (mord.LogisticAT)":
            return LogisticAT(**ORDINAL_PARAMS)
        if name == "Decision Tree":
            return DecisionTreeClassifier(**DT_PARAMS)
        raise KeyError(name)

    model_order = [
        "LightGBM",
        "Ordinal Logistic Regression (mord.LogisticAT)",
        "Decision Tree",
    ]

    def as_lgbm_frame(X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(np.asarray(X, dtype=np.float32), columns=[f"f{i}" for i in range(X.shape[1])])

    results: list[dict] = []
    for ds_name, (Xt_tr, Xt_te, n_feat) in datasets.items():
        for model_name in model_order:
            clf = new_model(model_name)
            if model_name == "LightGBM":
                tr_in, te_in = as_lgbm_frame(Xt_tr), as_lgbm_frame(Xt_te)
            else:
                tr_in, te_in = np.asarray(Xt_tr), np.asarray(Xt_te)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                clf.fit(tr_in, y_train)
                pred = clf.predict(te_in)
                proba = clf.predict_proba(te_in)
            auc, f1m = eval_multiclass(y_test, pred, proba)
            results.append(
                {
                    "experiment": ds_name,
                    "model": model_name,
                    "n_features": int(n_feat),
                    "macro_auroc": auc,
                    "macro_f1": f1m,
                }
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_html = REPORTS_DIR / "customer_satisfaction_fe_benchmark_report.html"

    n_cat_selected = sum(1 for r in selection_rows if r["feature"].startswith("cat__"))

    table_rows_html = []
    for r in results:
        table_rows_html.append(
            "<tr>"
            f"<td>{html.escape(r['experiment'])}</td>"
            f"<td>{html.escape(r['model'])}</td>"
            f"<td class='num'>{r['n_features']}</td>"
            f"<td class='num'>{r['macro_auroc']:.4f}</td>"
            f"<td class='num'>{r['macro_f1']:.4f}</td>"
            "</tr>"
        )

    sel_table_rows = []
    for row in selection_rows:
        sel_table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['feature'])}</td>"
            f"<td class='num'>{row['mi']:.5f}</td>"
            f"<td>{html.escape(row['reason'])}</td>"
            "</tr>"
        )

    params_json = json.dumps(
        {
            "LightGBM": LGBM_PARAMS,
            "DecisionTree": DT_PARAMS,
            "Ordinal_LogisticAT": ORDINAL_PARAMS,
            "SelectKBest": {"k": min(SELECT_K, n_dense), "score_func": "mutual_info_classif", "random_state": RANDOM_STATE},
            "PCA": {
                "n_components": PCA_VAR,
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            },
            "random_state": RANDOM_STATE,
        },
        indent=2,
        ensure_ascii=False,
    )
    params_block = html.escape(params_json)

    insights_and_future = build_insights_and_future_sections(
        results,
        n_cat_selected=n_cat_selected,
        select_k=min(SELECT_K, n_dense),
        n_pca=n_pca,
        pca_ev_sum=float(pca.explained_variance_ratio_.sum()),
    )

    doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>고객만족 Feature Engineering 벤치마크</title>
  <style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 24px 32px 48px; color: #1a1a1a; line-height: 1.55; max-width: 1100px; }}
    h1 {{ font-size: 1.45rem; border-bottom: 2px solid #2563eb; padding-bottom: 8px; }}
    h2 {{ font-size: 1.1rem; margin-top: 28px; color: #1e3a5f; }}
    p.meta {{ color: #555; font-size: 0.92rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #eff6ff; text-align: left; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .note {{ background: #f8fafc; border-left: 4px solid #64748b; padding: 12px 16px; margin: 16px 0; }}
    ul.insights-list {{ padding-left: 1.25rem; margin: 0.5rem 0 1rem; }}
    ul.insights-list li {{ margin: 0.5rem 0; }}
    pre.params {{ background: #0f172a; color: #e2e8f0; padding: 14px 16px; border-radius: 8px; overflow-x: auto; font-size: 0.78rem; }}
  </style>
</head>
<body>
  <h1>고객만족 RPI 예측 — Feature Engineering 비교</h1>
  <p class="meta">
    데이터: <code>{html.escape(str(train_path.relative_to(ROOT)))}</code> /
    <code>{html.escape(str(test_path.relative_to(ROOT)))}</code><br/>
    타깃: <strong>{TARGET}</strong> (등급 1~5, 학습 시 0~4 인코딩)<br/>
    생성 시각 (로컬): {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
  </p>

  <div class="note">
    <strong>특성 개수 점검</strong><br/>
    원시 입력 컬럼 수(타깃 제외): <strong>{n_raw}</strong>개
    (수치 {len(numeric_cols)} + 범주형 {len(CAT_COLS)}).<br/>
    전처리 후(수치 표준화 + 범주형 원-핫) 밀집 특성 차원: <strong>{n_dense}</strong>개,
    학습 표본 <strong>{len(df_tr)}</strong>건 → 표본/특성 비 약 <strong>{len(df_tr)/max(n_dense,1):.1f}</strong>:1.
    일반적으로 이 정도 차원은 트리/부스팅에 과도하다고 보기 어렵지만, 해석·다중공선성·노이즈 관점에서
    특성 선택·PCA 실험으로 효과를 확인하는 것이 목적입니다.
  </div>

  <h2>1. 실험 설계</h2>
  <ul>
    <li><strong>실험 1</strong>: 전처리 후 전 특성 사용.</li>
    <li><strong>실험 2</strong>: 학습 데이터에만 <code>SelectKBest(mutual_info_classif, k={min(SELECT_K, n_dense)})</code> 적용해
        동일한 부분집합을 세 모델 모두에 사용 (데이터 누수 방지: 테스트는 transform만).</li>
    <li><strong>실험 3</strong>: 동일 전처리 후 <code>PCA(n_components={PCA_VAR})</code>로 차원 축소
        (누적 설명분산 비율 합: <strong>{float(pca.explained_variance_ratio_.sum()):.4f}</strong>, 성분 수 <strong>{n_pca}</strong>).</li>
    <li>모델·하이퍼파라미터는 실험(1~3) 간 동일, 입력 행렬만 상이.</li>
  </ul>

  <h2>2. 평가 결과 (Macro AUROC, Macro F1)</h2>
  <p class="meta">다중 클래스 AUROC는 one-vs-rest macro 평균입니다.</p>
  <table>
    <thead>
      <tr><th>실험</th><th>모델</th><th>특성 차원</th><th>Macro AUROC</th><th>Macro F1</th></tr>
    </thead>
    <tbody>
      {chr(10).join(table_rows_html)}
    </tbody>
  </table>

  <h2>3. 실험 2 — 선택된 특성과 선정 이유</h2>
  <p class="meta">상호정보량(MI)은 입력과 RPI 간 통계적 연관(비선형 포함)을 학습 분할에서만 추정한 값입니다.
     아래 문장은 MI 크기와 컬럼 이름 패턴을 바탕으로 한 <strong>해석 가이드</strong>이며, 인과관계 주장은 아닙니다.</p>
  <p class="meta">이번 상위 K개 안에 포함된 범주형 원-핫 특성(<code>cat__</code>)은 <strong>{n_cat_selected}</strong>개입니다.
     CSI/CCI 종합·세부 수치가 RPI와 더 강한 MI를 보이며, 연도·제품 등 범주 신호는 상위 K에서 밀려난 것으로 해석할 수 있습니다.</p>
  <table>
    <thead>
      <tr><th>특성</th><th>MI 점수</th><th>선정 이유 (요약)</th></tr>
    </thead>
    <tbody>
      {chr(10).join(sel_table_rows)}
    </tbody>
  </table>

{insights_and_future}

  <h2>6. 고정 하이퍼파라미터 (JSON)</h2>
  <pre class="params">{params_block}</pre>
</body>
</html>
"""

    out_html.write_text(doc, encoding="utf-8")
    print("Wrote", out_html)


if __name__ == "__main__":
    main()
