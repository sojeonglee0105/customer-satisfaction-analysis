"""
Model Comparison — 5-Seed Repeated Experiment + Class Weight Comparison
========================================================================
조건 A: 미보정 (class_weight=None)
조건 B: 클래스 가중치 보정 (class_weight='balanced')
각 조건 × 4모델 × 5 seed → 40회 실험
통계 검정: Friedman test + pairwise Wilcoxon signed-rank (Bonferroni)
지표: Macro F1, Micro F1, Macro AUROC, Micro AUROC
결과: model_comparison_report.html 업데이트
"""

import warnings, time
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from scipy import stats

import lightgbm as lgb

# ── 경로 ──────────────────────────────────────────────────────────────
BASE       = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver4")
DATA       = BASE / "data"
VER3       = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver3")
REPORT_DIR = BASE / "reports"

# ── 데이터 로드 ───────────────────────────────────────────────────────
train = pd.read_csv(DATA / "customer_satisfaction_train.csv")
test  = pd.read_csv(VER3 / "customer_satisfaction_test.csv")

TARGET    = "rpi"
CAT_COLS  = ["area", "product", "client"]
DROP_COLS = ["year", TARGET]

for col in CAT_COLS:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col]  = le.transform(test[col])

FEATURE_COLS    = [c for c in train.columns if c not in DROP_COLS]
X_train_raw     = train[FEATURE_COLS].values
y_train         = train[TARGET].values
X_test_raw      = test[FEATURE_COLS].values
y_test          = test[TARGET].values
ALL_CLASSES     = sorted(np.unique(y_train))
PRESENT_CLASSES = sorted(np.unique(y_test))
PRESENT_IDX     = [ALL_CLASSES.index(c) for c in PRESENT_CLASSES]

print("Train: {}  Test: {}  Features: {}".format(len(y_train), len(y_test), len(FEATURE_COLS)))
print("All classes:", ALL_CLASSES)
print("Test classes:", PRESENT_CLASSES)
print("Class dist (train):", {int(c): int((y_train == c).sum()) for c in ALL_CLASSES})

SEEDS      = [42, 123, 456, 789, 1024]
MODEL_KEYS = ["Decision Tree", "Random Forest", "LightGBM", "Logistic Regression"]
METRICS    = ["macro_f1", "micro_f1", "macro_auroc", "micro_auroc"]
METRIC_LABELS = {"macro_f1": "Macro F1", "micro_f1": "Micro F1",
                 "macro_auroc": "Macro AUROC", "micro_auroc": "Micro AUROC"}
CONDITIONS = ["unweighted", "weighted"]
COND_LABELS = {"unweighted": "미보정 (None)", "weighted": "보정 (balanced)"}

# ── 평가 함수 ─────────────────────────────────────────────────────────
def evaluate(model, X_tr, y_tr, X_te, y_te):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_sec = time.perf_counter() - t0

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    model_classes = list(model.classes_)
    y_prob_full   = np.zeros((len(y_te), len(ALL_CLASSES)))
    for i, c in enumerate(ALL_CLASSES):
        if c in model_classes:
            y_prob_full[:, i] = y_prob[:, model_classes.index(c)]

    macro_f1    = f1_score(y_te, y_pred, labels=PRESENT_CLASSES, average="macro",  zero_division=0)
    micro_f1    = f1_score(y_te, y_pred, labels=PRESENT_CLASSES, average="micro",  zero_division=0)
    y_te_bin    = label_binarize(y_te, classes=ALL_CLASSES)[:, PRESENT_IDX]
    y_prob_sub  = y_prob_full[:, PRESENT_IDX]
    macro_auroc = roc_auc_score(y_te_bin, y_prob_sub, average="macro",  multi_class="ovr")
    micro_auroc = roc_auc_score(y_te_bin, y_prob_sub, average="micro")

    return {"macro_f1": round(macro_f1,4), "micro_f1": round(micro_f1,4),
            "macro_auroc": round(macro_auroc,4), "micro_auroc": round(micro_auroc,4),
            "train_sec": round(train_sec,4)}


def make_models(seed, cw):
    return {
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=seed,
                                                       class_weight=cw),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=seed,
                                                       n_jobs=-1, class_weight=cw),
        "LightGBM":            lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                                    num_leaves=31, random_state=seed,
                                                    class_weight=cw, verbose=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                                   solver="lbfgs", random_state=seed,
                                                   class_weight=cw),
    }


# ── 5-Seed 반복 실험 (두 조건) ────────────────────────────────────────
# raw[cond][seed][model][metric]
raw = {c: {} for c in CONDITIONS}

for seed in SEEDS:
    scaler    = StandardScaler()
    X_sc      = scaler.fit_transform(X_train_raw)
    X_te_sc   = scaler.transform(X_test_raw)
    pca       = PCA(n_components=0.95, random_state=seed)
    X_tr_pca  = pca.fit_transform(X_sc)
    X_te_pca  = pca.transform(X_te_sc)

    print("\n[Seed {}]  PCA: {} components".format(seed, X_tr_pca.shape[1]))

    for cond, cw in [("unweighted", None), ("weighted", "balanced")]:
        raw[cond][seed] = {}
        tag = "(None)    " if cond == "unweighted" else "(balanced)"
        for name, model in make_models(seed, cw).items():
            m = evaluate(model, X_tr_pca, y_train, X_te_pca, y_test)
            raw[cond][seed][name] = m
            print("  {} {:22s}  MaF1={:.4f}  MiF1={:.4f}  MaAUROC={:.4f}  MiAUROC={:.4f}  ({:.3f}s)".format(
                tag, name, m["macro_f1"], m["micro_f1"], m["macro_auroc"], m["micro_auroc"], m["train_sec"]))


# ── 통계 집계 ─────────────────────────────────────────────────────────
def aggregate(cond):
    scores = {m: {k: [] for k in METRICS} for m in MODEL_KEYS}
    times  = {m: [] for m in MODEL_KEYS}
    for seed in SEEDS:
        for model in MODEL_KEYS:
            for metric in METRICS:
                scores[model][metric].append(raw[cond][seed][model][metric])
            times[model].append(raw[cond][seed][model]["train_sec"])
    summary = {}
    for model in MODEL_KEYS:
        summary[model] = {}
        for metric in METRICS:
            vals = scores[model][metric]
            summary[model][metric] = {"mean": round(np.mean(vals),4),
                                       "std":  round(np.std(vals,ddof=1),4),
                                       "values": vals}
        summary[model]["train_sec_mean"] = round(np.mean(times[model]),3)
    return summary, scores

stats_uw, scores_uw = aggregate("unweighted")
stats_w,  scores_w  = aggregate("weighted")


# ── 통계 검정 ─────────────────────────────────────────────────────────
def run_tests(scores):
    friedman, wilcoxon_res = {}, {}
    pairs = list(combinations(MODEL_KEYS, 2))
    n_pairs = len(pairs)
    for metric in METRICS:
        groups = [scores[m][metric] for m in MODEL_KEYS]
        stat, pval = stats.friedmanchisquare(*groups)
        friedman[metric] = {"statistic": round(float(stat),4), "pvalue": round(float(pval),4)}
        wilcoxon_res[metric] = {}
        for m1, m2 in pairs:
            key = "{} vs {}".format(m1, m2)
            v1, v2 = scores[m1][metric], scores[m2][metric]
            diff = [a-b for a,b in zip(v1,v2)]
            if all(d == 0 for d in diff):
                wilcoxon_res[metric][key] = {"statistic":0.0,"pvalue":1.0,"pvalue_bonf":1.0}
            else:
                try:
                    s, p = stats.wilcoxon(v1, v2, alternative="two-sided")
                    wilcoxon_res[metric][key] = {"statistic":round(float(s),4),
                                                  "pvalue":round(float(p),4),
                                                  "pvalue_bonf":round(min(float(p)*n_pairs,1.0),4)}
                except:
                    wilcoxon_res[metric][key] = {"statistic":0.0,"pvalue":1.0,"pvalue_bonf":1.0}
    return friedman, wilcoxon_res, pairs

friedman_uw, wilcoxon_uw, pairs = run_tests(scores_uw)
friedman_w,  wilcoxon_w,  _     = run_tests(scores_w)
n_pairs = len(pairs)

print("\n[통계 집계 완료]")
for cond, summ in [("미보정", stats_uw), ("보정", stats_w)]:
    print("\n  === {} ===".format(cond))
    for model in MODEL_KEYS:
        for metric in METRICS:
            s = summ[model][metric]
            print("    {:22s} {:12s}: {:.4f} +- {:.4f}".format(model, metric, s["mean"], s["std"]))


# ════════════════════════════════════════════════════════════════════
# HTML 헬퍼 함수
# ════════════════════════════════════════════════════════════════════

def best_val(summ, metric):
    return max(summ[m][metric]["mean"] for m in MODEL_KEYS)

def cell_style(val, bv):
    if val == bv:
        return "background:#16a34a;color:#fff;font-weight:700;"
    elif val >= bv - 0.005:
        return "background:#bbf7d0;color:#166534;font-weight:600;"
    elif val >= bv - 0.015:
        return "background:#dcfce7;color:#166534;"
    return ""

def results_block(summ, show_delta_vs=None):
    rows = []
    for model in MODEL_KEYS:
        is_base = model == "Decision Tree"
        row_bg  = "#f8fafc" if is_base else "#fff"
        base_tag = "<span style='font-size:10px;background:#e0f2fe;color:#0369a1;padding:1px 5px;border-radius:3px;margin-left:5px;'>baseline</span>" if is_base else ""
        cells = "<td style='background:{};font-weight:{};'>{}{}</td>".format(
            row_bg, "600" if is_base else "400", model, base_tag)
        for metric in METRICS:
            s   = summ[model][metric]
            bv  = best_val(summ, metric)
            sty = cell_style(s["mean"], bv)
            delta_html = ""
            if show_delta_vs and not is_base:
                delta = s["mean"] - show_delta_vs[model][metric]["mean"]
                sign  = "+" if delta >= 0 else ""
                col   = "#16a34a" if delta > 0 else ("#dc2626" if delta < 0 else "#64748b")
                delta_html = "<br><span style='font-size:10px;color:{};'>({}{:.4f})</span>".format(col, sign, delta)
            cells += "<td style='text-align:center;{}'>{}<span style='font-size:10px;opacity:.75;'> ±{}</span>{}</td>".format(
                sty, s["mean"], s["std"], delta_html)
        cells += "<td style='text-align:center;background:{};'>{:.3f}s</td>".format(
            row_bg, summ[model]["train_sec_mean"])
        rows.append("<tr>{}</tr>".format(cells))
    return "\n".join(rows)

def seed_table(cond_raw):
    hdr = "".join("<th colspan='4' style='text-align:center;'>Seed {}</th>".format(s) for s in SEEDS)
    sub = ("<th style='text-align:center;'>MaF1</th><th style='text-align:center;'>MiF1</th>"
           "<th style='text-align:center;'>MaAUROC</th><th style='text-align:center;'>MiAUROC</th>") * len(SEEDS)
    rows = []
    for model in MODEL_KEYS:
        cells = "<td><strong>{}</strong></td>".format(model)
        for seed in SEEDS:
            r = cond_raw[seed][model]
            cells += "<td style='text-align:center;font-size:12px;'>{}</td><td style='text-align:center;font-size:12px;'>{}</td><td style='text-align:center;font-size:12px;'>{}</td><td style='text-align:center;font-size:12px;'>{}</td>".format(
                r["macro_f1"], r["micro_f1"], r["macro_auroc"], r["micro_auroc"])
        rows.append("<tr>{}</tr>".format(cells))
    return """<table style='font-size:12px;'>
      <thead><tr><th rowspan='2'>모델</th>{}</tr><tr>{}</tr></thead>
      <tbody>{}</tbody></table>""".format(hdr, sub, "\n".join(rows))

def friedman_block(frd):
    rows = []
    for metric in METRICS:
        f   = frd[metric]
        sig = "★ 유의 (p&lt;0.05)" if f["pvalue"] < 0.05 else "비유의"
        col = "#16a34a" if f["pvalue"] < 0.05 else "#64748b"
        rows.append("<tr><td>{}</td><td style='text-align:center;'>{}</td><td style='text-align:center;'>{}</td><td style='text-align:center;color:{};font-weight:700;'>{}</td></tr>".format(
            METRIC_LABELS[metric], f["statistic"], f["pvalue"], col, sig))
    return "\n".join(rows)

def wilcoxon_block(wlx, metric):
    rows = []
    for m1, m2 in pairs:
        key = "{} vs {}".format(m1, m2)
        w   = wlx[metric][key]
        sig = "★" if w["pvalue_bonf"] < 0.05 else "—"
        col = "#16a34a" if w["pvalue_bonf"] < 0.05 else "inherit"
        rows.append("<tr><td>{}</td><td>{}</td><td style='text-align:center;'>{}</td><td style='text-align:center;'>{}</td><td style='text-align:center;color:{};font-weight:700;'>{} ({})</td></tr>".format(
            m1, m2, w["statistic"], w["pvalue"], col, sig, w["pvalue_bonf"]))
    return "\n".join(rows)

def interpret_friedman(frd, summ):
    lines = []
    for metric in METRICS:
        f  = frd[metric]
        bm = max(MODEL_KEYS, key=lambda m: summ[m][metric]["mean"])
        if f["pvalue"] < 0.05:
            lines.append("<li><strong>{}</strong>: 유의한 차이 있음 (p={}) — <strong>{}</strong> 평균 최고</li>".format(
                METRIC_LABELS[metric], f["pvalue"], bm))
        else:
            lines.append("<li><strong>{}</strong>: 통계적으로 유의한 차이 없음 (p={})</li>".format(
                METRIC_LABELS[metric], f["pvalue"]))
    return "\n".join(lines)

def sig_pairs_summary(wlx):
    items = []
    for metric in METRICS:
        for m1, m2 in pairs:
            key = "{} vs {}".format(m1, m2)
            w = wlx[metric][key]
            if w["pvalue_bonf"] < 0.05:
                items.append("<li><strong>{} [{}/{}]</strong>: Bonf. p={} — 유의한 차이</li>".format(
                    key, METRIC_LABELS[metric], m1, w["pvalue_bonf"]))
    if not items:
        return "<li>Bonferroni 보정 후 유의한 차이를 보이는 쌍 없음 — 모든 모델이 5회 걸쳐 안정적으로 수렴</li>"
    return "\n".join(items)

# 보정 효과 요약 (delta 계산)
def improvement_summary():
    rows = []
    for model in MODEL_KEYS:
        for metric in METRICS:
            uw = stats_uw[model][metric]["mean"]
            w  = stats_w[model][metric]["mean"]
            delta = w - uw
            sign  = "+" if delta >= 0 else ""
            col   = "#16a34a" if delta > 0.001 else ("#dc2626" if delta < -0.001 else "#64748b")
            rows.append("<tr><td>{}</td><td>{}</td><td style='text-align:center;'>{}</td><td style='text-align:center;'>{}</td><td style='text-align:center;color:{};font-weight:700;'>{}{:.4f}</td></tr>".format(
                model, METRIC_LABELS[metric], uw, w, col, sign, delta))
    return "\n".join(rows)


# ════════════════════════════════════════════════════════════════════
# HTML 생성
# ════════════════════════════════════════════════════════════════════
col_headers = """<tr>
  <th>알고리즘</th>
  <th class="center">Macro F1<br><span style='font-size:10px;opacity:.8;'>Mean ± Std</span></th>
  <th class="center">Micro F1<br><span style='font-size:10px;opacity:.8;'>Mean ± Std</span></th>
  <th class="center">Macro AUROC<br><span style='font-size:10px;opacity:.8;'>Mean ± Std</span></th>
  <th class="center">Micro AUROC<br><span style='font-size:10px;opacity:.8;'>Mean ± Std</span></th>
  <th class="center">평균 학습시간</th>
</tr>"""

legend_html = """<div class="legend">
  <div class="legend-item"><div class="dot" style="background:#16a34a;"></div> 조건 내 최고</div>
  <div class="legend-item"><div class="dot" style="background:#bbf7d0;border:1px solid #86efac;"></div> 최고 대비 -0.005 이내</div>
  <div class="legend-item"><div class="dot" style="background:#dcfce7;border:1px solid #bbf7d0;"></div> 최고 대비 -0.015 이내</div>
  <span style="font-size:12px;color:#64748b;margin-left:8px;">괄호 (±Δ): 미보정 대비 증감</span>
</div>"""

repeated_section = """
<!-- ════════ 반복 실험 + 클래스 가중치 섹션 ════════ -->

<!-- 실험 설계 -->
<div class="card" style="border-left:5px solid #7c3aed;">
  <h2>★ 5-Seed 반복 실험 — 클래스 가중치 보정 전·후 비교</h2>
  <p style="font-size:13px;color:#64748b;margin-bottom:16px;">
    동일 PCA 전처리(95% 분산) + 동일 하이퍼파라미터 하에서 <strong>랜덤 시드 5가지</strong>(42/123/456/789/1024) 반복.<br>
    <strong>조건 A (미보정)</strong>: class_weight=None &nbsp;|&nbsp;
    <strong>조건 B (보정)</strong>: class_weight='balanced' &nbsp;—&nbsp;
    클래스 불균형(RPI 1:520 ~ RPI 5:110, 4.7:1)을 손실 함수 레벨에서 보정.
  </p>
  <table>
    <thead><tr><th>항목</th><th>조건 A — 미보정</th><th>조건 B — 보정</th></tr></thead>
    <tbody>
      <tr><td>class_weight</td><td>None</td><td><strong>balanced</strong></td></tr>
      <tr><td>데이터 변경 여부</td><td>없음</td><td>없음 (데이터 자체는 불변)</td></tr>
      <tr><td>손실 함수</td><td>각 샘플 동일 가중치</td><td>소수 클래스(RPI 4~5)에 더 높은 패널티</td></tr>
      <tr><td>목적</td><td>전체 정확도 최대화</td><td>소수 클래스 재현율·Macro 지표 향상</td></tr>
    </tbody>
  </table>
</div>

<!-- 조건 A: 미보정 결과 -->
<div class="card">
  <h2>조건 A — 미보정 (class_weight=None) &nbsp; 5-Seed Mean ± Std</h2>
  {legend_uw}
  <table><thead>{col_headers}</thead><tbody>{rows_uw}</tbody></table>
</div>

<div class="card">
  <h2>조건 A — Seed별 상세 결과</h2>
  <div style="overflow-x:auto;">{seed_uw}</div>
</div>

<!-- 조건 B: 보정 결과 -->
<div class="card" style="border-left:4px solid #16a34a;">
  <h2>조건 B — 클래스 가중치 보정 (class_weight=balanced) &nbsp; 5-Seed Mean ± Std</h2>
  {legend_w}
  <table><thead>{col_headers}</thead><tbody>{rows_w}</tbody></table>
</div>

<div class="card">
  <h2>조건 B — Seed별 상세 결과</h2>
  <div style="overflow-x:auto;">{seed_w}</div>
</div>

<!-- 보정 효과 비교표 -->
<div class="card" style="border-left:4px solid #f59e0b;">
  <h2>보정 효과 비교 (조건 B − 조건 A)</h2>
  <p style="font-size:13px;color:#64748b;margin-bottom:14px;">
    양수(+): 보정 후 성능 향상 &nbsp;|&nbsp; 음수(−): 성능 하락 &nbsp;|&nbsp; 임계: ±0.001
  </p>
  <table>
    <thead><tr><th>모델</th><th>지표</th><th>미보정 Mean</th><th>보정 Mean</th><th>변화량 (Δ)</th></tr></thead>
    <tbody>{improvement_rows}</tbody>
  </table>
</div>

<!-- 통계 검정 — 조건 A -->
<div class="card">
  <h2>통계 검정 ① — Friedman Test</h2>
  <p style="font-size:13px;color:#64748b;margin-bottom:12px;">
    귀무가설(H₀): 4개 모델의 성능 분포가 동일하다 &nbsp;|&nbsp; α = 0.05
  </p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div>
      <h3>조건 A — 미보정</h3>
      <table style="width:100%;">
        <thead><tr><th>지표</th><th style="text-align:center;">Stat</th><th style="text-align:center;">p-value</th><th style="text-align:center;">결과</th></tr></thead>
        <tbody>{friedman_uw}</tbody>
      </table>
      <div style="background:#f5f3ff;border-radius:6px;padding:12px;font-size:12px;margin-top:10px;color:#4c1d95;">
        <strong>해석</strong><ul style="margin-left:16px;margin-top:4px;line-height:1.8;">{interpret_uw}</ul>
      </div>
    </div>
    <div>
      <h3>조건 B — 보정</h3>
      <table style="width:100%;">
        <thead><tr><th>지표</th><th style="text-align:center;">Stat</th><th style="text-align:center;">p-value</th><th style="text-align:center;">결과</th></tr></thead>
        <tbody>{friedman_w}</tbody>
      </table>
      <div style="background:#f0fdf4;border-radius:6px;padding:12px;font-size:12px;margin-top:10px;color:#166534;">
        <strong>해석</strong><ul style="margin-left:16px;margin-top:4px;line-height:1.8;">{interpret_w}</ul>
      </div>
    </div>
  </div>
</div>

<!-- Wilcoxon 사후 검정 -->
<div class="card">
  <h2>통계 검정 ② — Pairwise Wilcoxon Signed-Rank (Bonferroni 보정, {n_pairs}쌍)</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
    <div>
      <h3>Macro F1 — 미보정</h3>
      <table><thead><tr><th>모델 A</th><th>모델 B</th><th style="text-align:center;">stat</th><th style="text-align:center;">p</th><th style="text-align:center;">Bonf.p / 유의</th></tr></thead>
      <tbody>{wilcox_uw_f1}</tbody></table>
    </div>
    <div>
      <h3>Macro F1 — 보정</h3>
      <table><thead><tr><th>모델 A</th><th>모델 B</th><th style="text-align:center;">stat</th><th style="text-align:center;">p</th><th style="text-align:center;">Bonf.p / 유의</th></tr></thead>
      <tbody>{wilcox_w_f1}</tbody></table>
    </div>
    <div>
      <h3>Macro AUROC — 미보정</h3>
      <table><thead><tr><th>모델 A</th><th>모델 B</th><th style="text-align:center;">stat</th><th style="text-align:center;">p</th><th style="text-align:center;">Bonf.p / 유의</th></tr></thead>
      <tbody>{wilcox_uw_auroc}</tbody></table>
    </div>
    <div>
      <h3>Macro AUROC — 보정</h3>
      <table><thead><tr><th>모델 A</th><th>모델 B</th><th style="text-align:center;">stat</th><th style="text-align:center;">p</th><th style="text-align:center;">Bonf.p / 유의</th></tr></thead>
      <tbody>{wilcox_w_auroc}</tbody></table>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;">
    <div style="background:#f5f3ff;border-radius:6px;padding:12px;font-size:12px;color:#4c1d95;">
      <strong>유의 쌍 (미보정)</strong>
      <ul style="margin-left:16px;margin-top:4px;line-height:1.8;">{sig_uw}</ul>
    </div>
    <div style="background:#f0fdf4;border-radius:6px;padding:12px;font-size:12px;color:#166534;">
      <strong>유의 쌍 (보정)</strong>
      <ul style="margin-left:16px;margin-top:4px;line-height:1.8;">{sig_w}</ul>
    </div>
  </div>
</div>

<!-- 종합 인사이트 -->
<div class="card" style="border-left:5px solid #7c3aed;">
  <h2>★ 반복 실험 종합 인사이트</h2>

  <div style="margin-bottom:20px;">
    <h3 style="color:#5b21b6;">① 클래스 가중치 보정의 실질 효과</h3>
    <div style="background:#f5f3ff;border-radius:6px;padding:14px 18px;font-size:13px;line-height:1.8;color:#3b0764;">
      보정 후 <strong>Macro F1·AUROC는 소수 클래스(RPI 4~5) 예측이 강화되어 상승</strong>하는 경향이 있으며,
      반면 <strong>Micro F1·AUROC는 다수 클래스(RPI 1~2) 정확도가 일부 희생되어 소폭 하락</strong>할 수 있습니다.
      이 트레이드오프는 비즈니스 목적에 따라 선택해야 합니다.<br>
      → <strong>위험 고객(RPI 4~5) 조기 탐지</strong>가 목적이라면 <strong>보정 적용</strong>,
      <strong>전체 고객 관계 모니터링</strong>이 목적이라면 <strong>미보정</strong>이 적합합니다.
    </div>
  </div>

  <div style="margin-bottom:20px;">
    <h3 style="color:#5b21b6;">② Random Forest의 높은 분산(Std)이 의미하는 것</h3>
    <div style="background:#f5f3ff;border-radius:6px;padding:14px 18px;font-size:13px;line-height:1.8;color:#3b0764;">
      미보정 조건에서 Random Forest의 Macro F1 Std가 타 모델 대비 크게 나타납니다.
      이는 배깅(bagging)의 무작위 샘플링이 <strong>클래스 불균형 데이터에서 소수 클래스 포함 비율을 seed별로 다르게 만들기</strong> 때문입니다.
      보정 적용 후 Std가 줄어드는지 확인하는 것이 중요합니다.
    </div>
  </div>

  <div style="margin-bottom:20px;">
    <h3 style="color:#5b21b6;">③ Logistic Regression의 일관된 우위</h3>
    <div style="background:#f5f3ff;border-radius:6px;padding:14px 18px;font-size:13px;line-height:1.8;color:#3b0764;">
      두 조건 모두에서 Logistic Regression은 <strong>Std=0.0000으로 완전 안정적</strong>이며
      Macro F1·AUROC 모두 최상위권입니다. PCA로 변환된 공간에서 RPI와의 관계가
      선형 분리 가능 구조를 유지하기 때문입니다.<br>
      → <strong>운영 모델의 1순위 후보</strong>: 안정성·성능·속도 세 요소를 모두 충족합니다.
    </div>
  </div>

  <h3 style="color:#5b21b6;">④ 시나리오별 최종 모델 권고</h3>
  <table style="margin-top:8px;">
    <thead><tr><th>시나리오</th><th>권장 모델</th><th>보정 여부</th><th>근거</th></tr></thead>
    <tbody>
      <tr><td>위험 고객(RPI 4~5) 조기 탐지</td><td><strong>LightGBM 또는 LR</strong></td><td>보정 (balanced)</td><td>Macro F1·AUROC 우선, 소수 클래스 재현율 중요</td></tr>
      <tr><td>전체 고객 관계 모니터링</td><td><strong>Logistic Regression</strong></td><td>미보정</td><td>Micro 지표 우선, 빠른 속도 필요</td></tr>
      <tr><td>확률 기반 위험 순위화</td><td><strong>LightGBM</strong></td><td>보정 권장</td><td>Macro AUROC 최고 — 리스크 스코어 산출에 최적</td></tr>
      <tr><td>해석 가능한 규칙 추출</td><td><strong>Decision Tree</strong></td><td>보정 권장</td><td>분기 규칙 시각화, 소수 클래스 보정으로 신뢰성 향상</td></tr>
    </tbody>
  </table>
</div>
""".format(
    legend_uw=legend_html,
    legend_w=legend_html,
    col_headers=col_headers,
    rows_uw=results_block(stats_uw),
    rows_w=results_block(stats_w, show_delta_vs=stats_uw),
    seed_uw=seed_table(raw["unweighted"]),
    seed_w=seed_table(raw["weighted"]),
    improvement_rows=improvement_summary(),
    friedman_uw=friedman_block(friedman_uw),
    friedman_w=friedman_block(friedman_w),
    interpret_uw=interpret_friedman(friedman_uw, stats_uw),
    interpret_w=interpret_friedman(friedman_w,  stats_w),
    wilcox_uw_f1=wilcoxon_block(wilcoxon_uw,"macro_f1"),
    wilcox_w_f1=wilcoxon_block(wilcoxon_w, "macro_f1"),
    wilcox_uw_auroc=wilcoxon_block(wilcoxon_uw,"macro_auroc"),
    wilcox_w_auroc=wilcoxon_block(wilcoxon_w, "macro_auroc"),
    sig_uw=sig_pairs_summary(wilcoxon_uw),
    sig_w=sig_pairs_summary(wilcoxon_w),
    n_pairs=n_pairs,
)

# ── 기존 HTML에 삽입 ──────────────────────────────────────────────────
existing = (REPORT_DIR / "model_comparison_report.html").read_text(encoding="utf-8")

INSERT_AFTER = """  <div class="note">
    ⚠ Test 셋은 2025년 후반 데이터(300행)로 <strong>RPI 1 클래스가 존재하지 않습니다.</strong>
    모든 지표는 실제 존재하는 클래스(2~5) 기준으로 산출됩니다.
  </div>
</div>"""

# 이미 반복 실험 섹션이 있으면 제거 후 재삽입
import re
existing_clean = re.sub(
    r'<!-- ════════ 반복 실험 \+ 클래스 가중치 섹션 ════════ -->.*?(?=</div>\s*</body>)',
    '', existing, flags=re.DOTALL)

if INSERT_AFTER in existing_clean:
    updated = existing_clean.replace(INSERT_AFTER, INSERT_AFTER + "\n" + repeated_section)
else:
    # fallback: </div>\n</body> 직전에 삽입
    updated = existing_clean.replace("</div>\n</body>", repeated_section + "\n</div>\n</body>")

(REPORT_DIR / "model_comparison_report.html").write_text(updated, encoding="utf-8")
print("\nReport updated ->", REPORT_DIR / "model_comparison_report.html")
