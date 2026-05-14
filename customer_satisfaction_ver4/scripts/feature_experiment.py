"""
Feature Engineering Experiment
===============================
3가지 Feature 전략 × 3개 모델 = 9가지 조합 비교
  Exp 1: All features
  Exp 2: Feature Selection (Mutual Information + Correlation 기반)
  Exp 3: PCA 차원 축소

모델: LightGBM, Logistic Regression, Decision Tree
지표: macro AUROC, macro F1
결과: customer_satisfaction_ver4/reports/feature_experiment_report.html
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

import lightgbm as lgb

# ── 경로 설정 ─────────────────────────────────────────────────────────
BASE = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver4")
DATA = BASE / "data"
REPORT_DIR = BASE / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ── 데이터 로드 ───────────────────────────────────────────────────────
VER3 = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver3")
train = pd.read_csv(DATA / "customer_satisfaction_train.csv")
test  = pd.read_csv(VER3 / "customer_satisfaction_test.csv")

TARGET = "rpi"
CAT_COLS = ["area", "product", "client"]
DROP_COLS = ["year", TARGET]  # year는 정보 누수 방지 및 시점 변수로 제외

# 범주형 인코딩 (Label Encoding)
for col in CAT_COLS:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col]  = le.transform(test[col])

FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
X_train = train[FEATURE_COLS].copy()
y_train = train[TARGET].copy()
X_test  = test[FEATURE_COLS].copy()
y_test  = test[TARGET].copy()

CLASSES = sorted(y_train.unique())
print("Classes:", CLASSES)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# ── 평가 함수 ─────────────────────────────────────────────────────────
def evaluate(model, X_tr, y_tr, X_te, y_te, model_name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    # Test 셋에 없는 클래스 처리
    present_classes = sorted(y_te.unique())
    all_classes     = sorted(y_tr.unique())

    # macro F1 (labels=present only to avoid zero-division on missing class)
    f1 = f1_score(y_te, y_pred, labels=present_classes, average="macro", zero_division=0)

    # macro AUROC (OvR, only on classes present in test)
    y_te_bin = label_binarize(y_te, classes=all_classes)
    # 없는 클래스 컬럼 제거
    present_idx = [all_classes.index(c) for c in present_classes]
    y_te_bin_sub = y_te_bin[:, present_idx]
    y_prob_sub   = y_prob[:, present_idx]

    if y_te_bin_sub.shape[1] <= 1:
        auroc = float("nan")
    else:
        try:
            auroc = roc_auc_score(y_te_bin_sub, y_prob_sub, average="macro", multi_class="ovr")
        except Exception:
            auroc = float("nan")

    return {"f1": round(f1, 4), "auroc": round(auroc, 4) if not np.isnan(auroc) else "N/A"}


# ── 모델 정의 (동일 seed, 동일 세팅) ──────────────────────────────────
def make_models():
    lgbm = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1
    )
    lr = LogisticRegression(
        max_iter=1000, C=1.0,
        solver="lbfgs", random_state=42
    )
    dt = DecisionTreeClassifier(
        max_depth=8, random_state=42
    )
    return {"LightGBM": lgbm, "Logistic Regression": lr, "Decision Tree": dt}


# ════════════════════════════════════════════════════════════════════
# Exp 1: All Features
# ════════════════════════════════════════════════════════════════════
print("\n[Exp 1] All Features ({} cols)".format(X_train.shape[1]))
results_exp1 = {}
for name, model in make_models().items():
    scaler = StandardScaler()
    if name in ["Logistic Regression"]:
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
    else:
        Xtr, Xte = X_train.values, X_test.values
    results_exp1[name] = evaluate(model, Xtr, y_train, Xte, y_test, name)
    print("  {}: F1={} AUROC={}".format(name, results_exp1[name]["f1"], results_exp1[name]["auroc"]))


# ════════════════════════════════════════════════════════════════════
# Exp 2: Feature Selection
# 선택 근거:
#  (A) Mutual Information ≥ MI 전체 평균 이상인 변수
#  (B) + Pearson |r| ≥ 0.15 with target (선형 신호 보강)
#  두 조건을 OR 합산하여 최종 feature 선택
# ════════════════════════════════════════════════════════════════════
print("\n[Exp 2] Feature Selection")

# MI 계산
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=FEATURE_COLS)
mi_threshold = mi_series.mean()

# Pearson 상관 계산
corr_series = X_train.corrwith(y_train.astype(float)).abs()
corr_threshold = 0.15

selected_mi   = set(mi_series[mi_series >= mi_threshold].index)
selected_corr = set(corr_series[corr_series >= corr_threshold].index)
selected_features = sorted(selected_mi | selected_corr)

print("  MI threshold  : {:.4f} → {} features".format(mi_threshold, len(selected_mi)))
print("  Corr threshold: {:.4f} → {} features".format(corr_threshold, len(selected_corr)))
print("  Union selected: {} features".format(len(selected_features)))

# Feature 선택 상세 저장 (HTML 리포트용)
fs_detail = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mutual_info": mi_scores,
    "pearson_r_abs": corr_series.values,
}).sort_values("mutual_info", ascending=False)
fs_detail["selected"] = fs_detail["feature"].isin(selected_features)

X_train_fs = X_train[selected_features]
X_test_fs  = X_test[selected_features]

results_exp2 = {}
for name, model in make_models().items():
    scaler = StandardScaler()
    if name in ["Logistic Regression"]:
        Xtr = scaler.fit_transform(X_train_fs)
        Xte = scaler.transform(X_test_fs)
    else:
        Xtr, Xte = X_train_fs.values, X_test_fs.values
    results_exp2[name] = evaluate(model, Xtr, y_train, Xte, y_test, name)
    print("  {}: F1={} AUROC={}".format(name, results_exp2[name]["f1"], results_exp2[name]["auroc"]))


# ════════════════════════════════════════════════════════════════════
# Exp 3: PCA
# 분산 95% 설명 기준으로 컴포넌트 수 자동 선택
# ════════════════════════════════════════════════════════════════════
print("\n[Exp 3] PCA")

scaler_pca = StandardScaler()
X_train_scaled = scaler_pca.fit_transform(X_train)
X_test_scaled  = scaler_pca.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
n_components = X_train_pca.shape[1]
variance_explained = pca.explained_variance_ratio_.sum()

print("  PCA components: {} (explains {:.1f}% variance)".format(n_components, variance_explained * 100))

results_exp3 = {}
for name, model in make_models().items():
    results_exp3[name] = evaluate(model, X_train_pca, y_train, X_test_pca, y_test, name)
    print("  {}: F1={} AUROC={}".format(name, results_exp3[name]["f1"], results_exp3[name]["auroc"]))


# ════════════════════════════════════════════════════════════════════
# HTML 리포트 생성
# ════════════════════════════════════════════════════════════════════
MODEL_NAMES = ["LightGBM", "Logistic Regression", "Decision Tree"]
EXP_LABELS  = [
    "Exp 1: All Features ({} cols)".format(len(FEATURE_COLS)),
    "Exp 2: Feature Selection ({} cols)".format(len(selected_features)),
    "Exp 3: PCA ({} components, {:.1f}% var)".format(n_components, variance_explained * 100),
]
ALL_RESULTS = [results_exp1, results_exp2, results_exp3]

# 전체 결과 플랫 테이블
rows_flat = []
for exp_label, res_dict in zip(EXP_LABELS, ALL_RESULTS):
    for model_name in MODEL_NAMES:
        rows_flat.append({
            "Experiment": exp_label,
            "Model": model_name,
            "Macro F1": res_dict[model_name]["f1"],
            "Macro AUROC": res_dict[model_name]["auroc"],
        })
df_results = pd.DataFrame(rows_flat)

# 최고 점수 하이라이트용
max_f1    = df_results["Macro F1"].max()
max_auroc = df_results[df_results["Macro AUROC"] != "N/A"]["Macro AUROC"].astype(float).max()


def cell_style(val, max_val, metric):
    try:
        v = float(val)
    except:
        return ""
    if v == max_val:
        return "background:#16a34a;color:#fff;font-weight:700;"
    elif v >= max_val * 0.97:
        return "background:#bbf7d0;color:#166534;font-weight:600;"
    elif v >= max_val * 0.93:
        return "background:#dcfce7;"
    return ""


# Feature Selection 상세 테이블 (상위 20)
fs_top = fs_detail.head(20)

def fs_row(row):
    bg = "#f0fdf4" if row["selected"] else "#fff"
    mark = "✔" if row["selected"] else ""
    return (
        "<tr style='background:{};'>"
        "<td>{}</td><td>{:.4f}</td><td>{:.4f}</td><td style='text-align:center;color:#16a34a;font-weight:700;'>{}</td>"
        "</tr>"
    ).format(bg, row["feature"], row["mutual_info"], row["pearson_r_abs"], mark)

fs_rows_html = "\n".join(fs_top.apply(fs_row, axis=1))


# 결과 테이블 HTML
def result_table_html():
    headers = ["Experiment", "Model", "Macro F1", "Macro AUROC"]
    rows_html = []
    for _, row in df_results.iterrows():
        f1_style    = cell_style(row["Macro F1"], max_f1, "f1")
        auroc_style = cell_style(row["Macro AUROC"], max_auroc, "auroc")
        rows_html.append(
            "<tr>"
            "<td>{}</td><td>{}</td>"
            "<td style='{}'>{}</td>"
            "<td style='{}'>{}</td>"
            "</tr>".format(
                row["Experiment"], row["Model"],
                f1_style, row["Macro F1"],
                auroc_style, row["Macro AUROC"],
            )
        )
    return (
        "<table><thead><tr>"
        + "".join("<th>{}</th>".format(h) for h in headers)
        + "</tr></thead><tbody>"
        + "\n".join(rows_html)
        + "</tbody></table>"
    )


# PCA 분산 기여도 상위 10 컴포넌트
pca_var_rows = "".join(
    "<tr><td>PC{}</td><td>{:.4f}</td><td>{:.1f}%</td></tr>".format(
        i + 1,
        pca.explained_variance_ratio_[i],
        pca.explained_variance_ratio_[:i+1].sum() * 100
    )
    for i in range(min(10, n_components))
)

# ── 최종 HTML ─────────────────────────────────────────────────────
html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Feature Engineering Experiment Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#f8fafc;color:#1e293b;font-size:14px;line-height:1.6;}}
  .header{{background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);color:#fff;padding:36px 48px;}}
  .header h1{{font-size:26px;font-weight:700;margin-bottom:6px;}}
  .header p{{opacity:.85;font-size:13px;}}
  .container{{max-width:1100px;margin:0 auto;padding:32px 24px;}}
  .card{{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:28px 32px;margin-bottom:28px;}}
  h2{{font-size:17px;font-weight:700;color:#1e3a5f;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid #e2e8f0;}}
  h3{{font-size:14px;font-weight:600;color:#334155;margin:16px 0 8px;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th{{background:#1e3a5f;color:#fff;padding:10px 14px;text-align:left;font-weight:600;}}
  td{{padding:9px 14px;border-bottom:1px solid #e2e8f0;}}
  tr:last-child td{{border-bottom:none;}}
  tr:hover td{{background:#f1f5f9;}}
  .tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;}}
  .tag-all{{background:#dbeafe;color:#1e40af;}}
  .tag-fs{{background:#fef9c3;color:#854d0e;}}
  .tag-pca{{background:#f3e8ff;color:#6b21a8;}}
  .meta-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px;}}
  .meta-card{{background:#f1f5f9;border-radius:8px;padding:14px 18px;}}
  .meta-card .val{{font-size:22px;font-weight:700;color:#2563eb;}}
  .meta-card .lbl{{font-size:12px;color:#64748b;margin-top:2px;}}
  .note{{background:#fffbeb;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:4px;font-size:13px;color:#78350f;margin-top:12px;}}
  .reason-box{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:14px 18px;font-size:13px;color:#166534;margin-bottom:16px;}}
  .reason-box ul{{margin-left:18px;margin-top:6px;}}
  .reason-box li{{margin-bottom:4px;}}
  .legend{{display:flex;gap:16px;font-size:12px;margin-bottom:12px;flex-wrap:wrap;}}
  .legend-item{{display:flex;align-items:center;gap:6px;}}
  .dot{{width:12px;height:12px;border-radius:2px;}}
</style>
</head>
<body>
<div class="header">
  <h1>Feature Engineering Experiment Report</h1>
  <p>Customer Satisfaction RPI 예측 &nbsp;|&nbsp; 3 Experiments × 3 Models &nbsp;|&nbsp; Macro AUROC &amp; Macro F1</p>
</div>

<div class="container">

<!-- 데이터 요약 -->
<div class="card">
  <h2>데이터 요약</h2>
  <div class="meta-grid">
    <div class="meta-card"><div class="val">{n_train:,}</div><div class="lbl">Train 샘플 수</div></div>
    <div class="meta-card"><div class="val">{n_test:,}</div><div class="lbl">Test 샘플 수</div></div>
    <div class="meta-card"><div class="val">{n_features}</div><div class="lbl">전체 Feature 수</div></div>
    <div class="meta-card"><div class="val">5</div><div class="lbl">RPI 클래스 수 (1~5)</div></div>
  </div>
  <div class="note">
    ⚠ Test 셋은 2025년 후반 데이터(300행)로, <strong>RPI 1 클래스가 존재하지 않습니다.</strong>
    이로 인해 Macro F1·AUROC는 실제 4개 클래스(2~5) 기준으로 산출됩니다.
  </div>
</div>

<!-- 실험 설계 -->
<div class="card">
  <h2>실험 설계</h2>
  <table>
    <thead><tr><th>실험</th><th>Feature 전략</th><th>Feature 수</th><th>전처리</th></tr></thead>
    <tbody>
      <tr><td><span class="tag tag-all">Exp 1</span></td><td>전체 Feature 사용</td><td>{n_features}</td><td>범주형 Label Encoding, LR은 StandardScaler 추가</td></tr>
      <tr><td><span class="tag tag-fs">Exp 2</span></td><td>Feature Selection (MI + Pearson)</td><td>{n_fs}</td><td>동일</td></tr>
      <tr><td><span class="tag tag-pca">Exp 3</span></td><td>PCA 차원 축소 (분산 95%)</td><td>{n_pca} components</td><td>StandardScaler → PCA</td></tr>
    </tbody>
  </table>
  <h3 style="margin-top:16px;">공통 모델 세팅</h3>
  <table>
    <thead><tr><th>모델</th><th>주요 하이퍼파라미터</th></tr></thead>
    <tbody>
      <tr><td>LightGBM</td><td>n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42</td></tr>
      <tr><td>Logistic Regression</td><td>max_iter=1000, C=1.0, solver=lbfgs, random_state=42 (multinomial via OvR)</td></tr>
      <tr><td>Decision Tree</td><td>max_depth=8, random_state=42</td></tr>
    </tbody>
  </table>
</div>

<!-- Exp 2 Feature Selection 근거 -->
<div class="card">
  <h2><span class="tag tag-fs">Exp 2</span>&nbsp; Feature Selection 선택 근거</h2>
  <div class="reason-box">
    <strong>선택 기준 (OR 조건 — 둘 중 하나라도 충족하면 선택)</strong>
    <ul>
      <li><strong>Mutual Information (MI) ≥ 평균 MI ({mi_threshold:.4f})</strong>
          &nbsp;— 비선형 관계 포함 target과의 정보량 기준.
          MI가 평균 이상인 변수는 클래스 구분에 실질적 기여를 하는 것으로 판단.</li>
      <li><strong>Pearson |r| ≥ 0.15</strong>
          &nbsp;— 선형 상관이 일정 수준 이상인 변수 보강.
          MI만으로 포착 못하는 선형 신호를 추가로 포함.</li>
    </ul>
    <br>
    → MI ≥ {mi_threshold:.4f}를 충족하는 변수 {n_mi}개, Pearson |r| ≥ 0.15를 충족하는 변수 {n_corr}개,
    합집합으로 최종 <strong>{n_fs}개 변수</strong>를 선택.
  </div>

  <h3>Feature 중요도 상위 20 (MI 기준)</h3>
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:#f0fdf4;border:1px solid #bbf7d0;"></div> 선택된 Feature</div>
    <div class="legend-item"><div class="dot" style="background:#fff;border:1px solid #e2e8f0;"></div> 미선택 Feature</div>
  </div>
  <table>
    <thead><tr><th>Feature</th><th>Mutual Information</th><th>Pearson |r|</th><th>선택</th></tr></thead>
    <tbody>{fs_rows}</tbody>
  </table>
</div>

<!-- Exp 3 PCA -->
<div class="card">
  <h2><span class="tag tag-pca">Exp 3</span>&nbsp; PCA 차원 축소 결과</h2>
  <div class="meta-grid">
    <div class="meta-card"><div class="val">{n_features}</div><div class="lbl">원본 Feature 수</div></div>
    <div class="meta-card"><div class="val">{n_pca}</div><div class="lbl">선택된 PC 수 (95% 분산)</div></div>
    <div class="meta-card"><div class="val">{var_pct:.1f}%</div><div class="lbl">설명 분산 비율</div></div>
  </div>
  <h3>PC별 누적 분산 기여도 (상위 10)</h3>
  <table>
    <thead><tr><th>Component</th><th>Individual Variance</th><th>Cumulative</th></tr></thead>
    <tbody>{pca_var_rows}</tbody>
  </table>
</div>

<!-- 실험 결과 -->
<div class="card">
  <h2>실험 결과 비교표</h2>
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:#16a34a;"></div> 최고 점수</div>
    <div class="legend-item"><div class="dot" style="background:#bbf7d0;border:1px solid #16a34a;"></div> Top 3% 이내</div>
    <div class="legend-item"><div class="dot" style="background:#dcfce7;border:1px solid #bbf7d0;"></div> Top 7% 이내</div>
  </div>
  {result_table}
</div>

<!-- 해석 -->
<div class="card">
  <h2>인사이트 요약</h2>
  <table>
    <thead><tr><th>관점</th><th>내용</th></tr></thead>
    <tbody>
      <tr><td>Feature 수 영향</td>
          <td>전체 {n_features}개 feature 중 상당수가 중복 정보를 담고 있을 가능성이 높음 (CSI·CCI 하위 항목 구조상 동일 영역의 res/core/comm/total이 공존). Feature Selection·PCA로 압축 시 노이즈 감소 효과 기대.</td></tr>
      <tr><td>Feature Selection 의의</td>
          <td>MI + Pearson 조합으로 비선형·선형 신호를 모두 고려. 선택된 {n_fs}개 변수는 원본 대비 약 {fs_ratio:.0f}% 수준으로 압축되면서도 핵심 정보를 보존.</td></tr>
      <tr><td>PCA 의의</td>
          <td>{n_pca}개 PC로 원본 {n_features}개 변수의 {var_pct:.1f}% 분산을 설명. 모델 간 성능 변화 폭이 크다면 원본 feature 공간의 다중공선성 문제를 간접 확인 가능.</td></tr>
      <tr><td>주의 사항</td>
          <td>Test 셋에 RPI 1 클래스가 없어 macro 지표가 4-class 기준으로 산출됨. 실제 서비스에서는 전체 5-class 평가가 필요.</td></tr>
    </tbody>
  </table>
</div>

</div><!-- /container -->
</body>
</html>
""".format(
    n_train=len(train),
    n_test=len(test),
    n_features=len(FEATURE_COLS),
    n_fs=len(selected_features),
    n_pca=n_components,
    var_pct=variance_explained * 100,
    mi_threshold=mi_threshold,
    n_mi=len(selected_mi),
    n_corr=len(selected_corr),
    fs_rows=fs_rows_html,
    pca_var_rows=pca_var_rows,
    result_table=result_table_html(),
    fs_ratio=len(selected_features) / len(FEATURE_COLS) * 100,
)

out_html = REPORT_DIR / "feature_experiment_report.html"
out_html.write_text(html, encoding="utf-8")
print("\nReport saved -> {}".format(out_html))
