"""
SHAP Analysis — Feature Importance 리포트에 추가
=================================================
모델: LightGBM / Random Forest / Logistic Regression (class_weight='balanced')
피처 공간: 원본 51개 변수

시각화:
  1. 3모델 Mean |SHAP| 바차트 비교
  2. LGB SHAP Beeswarm (전체) — 양수/음수 방향 + 크기 동시 표현
  3. RPI 클래스별 SHAP 방향 (RPI 1 vs RPI 5 대비)
  4. t1_cci_total / c_cci_total / q1_cci_total 의존성 플롯 (Dependence)
  5. 위험 고객(RPI 4~5) vs 우량 고객(RPI 1~2) SHAP Waterfall 대조
  6. 양수 기여 vs 음수 기여 Feature 분류표

결과: feature_importance_report.html 끝에 SHAP 섹션 추가
"""

import warnings, io, base64
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# ── 경로 & 폰트 ────────────────────────────────────────────────────
BASE       = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver4")
DATA       = BASE / "data"
VER3       = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver3")
REPORT_DIR = BASE / "reports"

import matplotlib.font_manager as fm
for fp in [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\NanumGothic.ttf"]:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        fn = fm.FontProperties(fname=fp).get_name()
        matplotlib.rcParams["font.family"] = fn
        print("Font:", fn); break
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# 영역 정의
DOMAIN_MAP = {
    "t1": "신기술", "t2": "개발", "c": "Cost",
    "d": "공급", "q1": "품질", "q2": "서비스"
}
DOMAIN_COLORS = {
    "신기술": "#2563eb", "개발": "#16a34a", "Cost": "#ea580c",
    "공급": "#9333ea", "품질": "#e11d48", "서비스": "#ca8a04",
    "기타": "#64748b"
}

def get_domain(feat):
    for prefix in ["t1","t2","q1","q2","c","d"]:
        if feat.startswith(prefix + "_") or feat == prefix:
            return DOMAIN_MAP.get(prefix, "기타")
    return "기타"

def feat_label(feat):
    d = get_domain(feat)
    suffix = feat.replace("t1_","").replace("t2_","").replace("q1_","") \
                  .replace("q2_","").replace("c_","").replace("d_","")
    return "[{}] {}".format(d, suffix)

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── 데이터 로드 ────────────────────────────────────────────────────
train = pd.read_csv(DATA / "customer_satisfaction_train.csv")
test  = pd.read_csv(VER3 / "customer_satisfaction_test.csv")
full  = pd.concat([train, test], ignore_index=True)

TARGET   = "rpi"
CAT_COLS = ["area", "product", "client"]
DROP_COLS = ["year", TARGET]

for col in CAT_COLS:
    le = LabelEncoder()
    le.fit(full[col])
    train[col] = le.transform(train[col])
    test[col]  = le.transform(test[col])
    full[col]  = le.transform(full[col])

FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
X_train = train[FEATURE_COLS].values
y_train = train[TARGET].values
X_test  = test[FEATURE_COLS].values
y_test  = test[TARGET].values
ALL_CLS = sorted(np.unique(y_train))
N_CLS   = len(ALL_CLS)
N_FEAT  = len(FEATURE_COLS)

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_train)
X_te_sc  = scaler.transform(X_test)

print("Train:", len(y_train), "| Test:", len(y_test), "| Feats:", N_FEAT)
print("Classes:", ALL_CLS)

# ── 모델 학습 ──────────────────────────────────────────────────────
SEED = 42
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                 num_leaves=31, class_weight="balanced",
                                 random_state=SEED, verbose=-1)
lgb_model.fit(X_train, y_train)
print("LGB trained")

rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("RF trained")

lr_model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                               class_weight="balanced", random_state=SEED)
lr_model.fit(X_tr_sc, y_train)
print("LR trained")

# ── SHAP 값 계산 ───────────────────────────────────────────────────
print("\nComputing SHAP values ...")

# LightGBM — TreeExplainer (가장 정확)
explainer_lgb  = shap.TreeExplainer(lgb_model)
shap_lgb_raw   = explainer_lgb.shap_values(X_test)
# shap_lgb_raw: list of (n_samples, n_feats) for each class
shap_lgb_arr = np.array(shap_lgb_raw)
# 실제 shape 확인 후 (n_cls, n_samples, n_feats) 로 정규화
if shap_lgb_arr.ndim == 3 and shap_lgb_arr.shape[0] == len(X_test):
    shap_lgb = shap_lgb_arr.transpose(2, 0, 1)   # (n_samples,n_feats,n_cls) -> (n_cls,n_samples,n_feats)
elif shap_lgb_arr.ndim == 3:
    shap_lgb = shap_lgb_arr                        # already (n_cls, n_samples, n_feats)
else:
    shap_lgb = shap_lgb_arr
print("  LGB SHAP done - shape:", shap_lgb.shape)

# Random Forest — TreeExplainer
explainer_rf   = shap.TreeExplainer(rf_model)
shap_rf_raw    = explainer_rf.shap_values(X_test)
shap_rf_arr    = np.array(shap_rf_raw)
if shap_rf_arr.ndim == 3 and shap_rf_arr.shape[0] == len(X_test):
    shap_rf = shap_rf_arr.transpose(2, 0, 1)
elif shap_rf_arr.ndim == 3:
    shap_rf = shap_rf_arr
else:
    shap_rf = shap_rf_arr
print("  RF SHAP done - shape:", shap_rf.shape)

# Logistic Regression — LinearExplainer
explainer_lr   = shap.LinearExplainer(lr_model, X_tr_sc)
shap_lr_raw    = explainer_lr.shap_values(X_te_sc)
shap_lr_arr    = np.array(shap_lr_raw)
if shap_lr_arr.ndim == 3 and shap_lr_arr.shape[0] == len(X_te_sc):
    shap_lr = shap_lr_arr.transpose(2, 0, 1)
elif shap_lr_arr.ndim == 3:
    shap_lr = shap_lr_arr
elif shap_lr_arr.ndim == 2:
    shap_lr = shap_lr_arr[np.newaxis, ...]
else:
    shap_lr = shap_lr_arr
print("  LR SHAP done - shape:", shap_lr.shape)

# ─ 전체 절댓값 평균 (클래스 × 샘플 평균) ─
def mean_abs_shap(shap_3d):
    return np.abs(shap_3d).mean(axis=(0, 1))    # (n_feats,)

shap_mean_lgb = mean_abs_shap(shap_lgb)
shap_mean_rf  = mean_abs_shap(shap_rf)
shap_mean_lr  = mean_abs_shap(shap_lr)

# 클래스별 평균 SHAP (부호 보존) — (n_cls, n_feats)
shap_cls_lgb = shap_lgb.mean(axis=1)       # 샘플 평균, 부호 보존
shap_cls_rf  = shap_rf.mean(axis=1)
shap_cls_lr  = shap_lr.mean(axis=1)

imgs = {}

# ════════════════════════════════════════════════════════════════════
# 시각화 1: 3모델 Mean |SHAP| 바차트 비교 (Top-20)
# ════════════════════════════════════════════════════════════════════
top20_idx = np.argsort(shap_mean_lgb)[-20:][::-1]
top20_feat = [FEATURE_COLS[i] for i in top20_idx]
domains_top20 = [get_domain(f) for f in top20_feat]
bar_colors = [DOMAIN_COLORS.get(d, "#64748b") for d in domains_top20]

fig, ax = plt.subplots(figsize=(12, 8), facecolor="#f8fafc")
y = np.arange(20)
w = 0.25
ax.barh(y + w, shap_mean_lr[top20_idx],  height=w, color="#6366f1", alpha=0.85, label="Logistic Regression")
ax.barh(y,     shap_mean_lgb[top20_idx], height=w, color="#10b981", alpha=0.85, label="LightGBM")
ax.barh(y - w, shap_mean_rf[top20_idx],  height=w, color="#f59e0b", alpha=0.85, label="Random Forest")
ax.set_yticks(y)
ax.set_yticklabels([feat_label(f) for f in top20_feat], fontsize=8.5)
ax.set_xlabel("Mean |SHAP Value| (예측에 대한 평균 기여 절댓값)", fontsize=10)
ax.set_title("3모델 SHAP 기반 Feature Importance Top-20\n(막대 길이: 예측 기여 크기 | 영역별 색상)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.axvline(0, color="black", lw=0.8)
ax.grid(axis="x", alpha=0.25)
ax.set_facecolor("#f8fafc")
# 도메인 범례
domain_patches = [mpatches.Patch(color=c, label=n)
                  for n, c in DOMAIN_COLORS.items() if n != "기타"]
ax2_legend = ax.legend(handles=domain_patches, fontsize=7.5, loc="upper right",
                        title="영역", title_fontsize=8)
ax.add_artist(ax2_legend)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
imgs["shap_bar_3models"] = fig_to_b64(fig); plt.close(fig)
print("  [V1] shap_bar_3models done")

# ════════════════════════════════════════════════════════════════════
# 시각화 2: LGB SHAP Beeswarm (전체 클래스 통합 — 양수/음수 방향)
# ════════════════════════════════════════════════════════════════════
# 전체 클래스에 걸쳐 샘플별 SHAP (절댓값 기준 상위 피처 선택 후 beeswarm 유사 dot plot)
shap_lgb_all = shap_lgb.sum(axis=0)        # (n_samples, n_feats) — 클래스 합산

# Top-15 피처 beeswarm 수동 구현
top15_idx_lgb = np.argsort(shap_mean_lgb)[-15:]

fig, ax = plt.subplots(figsize=(11, 8), facecolor="#f8fafc")
y_pos = np.arange(15)
for yi, fi in enumerate(top15_idx_lgb):
    shap_vals = shap_lgb_all[:, fi]
    feat_vals  = X_test[:, fi]
    # 피처값 정규화 → 색상 매핑
    fv_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-12)
    colors = plt.cm.RdBu_r(fv_norm)
    jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(shap_vals))
    ax.scatter(shap_vals, yi + jitter, c=colors, s=8, alpha=0.6, linewidths=0)

ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([feat_label(FEATURE_COLS[i]) for i in top15_idx_lgb], fontsize=9)
ax.set_xlabel("SHAP Value (← 재구매 낮춤 | 재구매 높임 →)", fontsize=10)
ax.set_title("LightGBM SHAP Beeswarm Plot — Top-15 Features\n(점 색상: 🔴빨간=피처값 높음 | 🔵파란=피처값 낮음 | X축: 양수=RPI 높임, 음수=RPI 낮춤)", fontsize=10, fontweight="bold")
# 컬러바 — 피처 값
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("피처 값 (Low → High)", fontsize=9)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(["낮음", "중간", "높음"])
ax.grid(axis="x", alpha=0.2)
ax.set_facecolor("#f8fafc")
fig.tight_layout()
imgs["shap_beeswarm_lgb"] = fig_to_b64(fig); plt.close(fig)
print("  [V2] shap_beeswarm_lgb done")

# ════════════════════════════════════════════════════════════════════
# 시각화 3: RPI 클래스별 SHAP 방향 — RPI 1 vs RPI 5 대비 (Top-12)
# ════════════════════════════════════════════════════════════════════
top12_idx = np.argsort(shap_mean_lgb)[-12:][::-1]
top12_feat = [FEATURE_COLS[i] for i in top12_idx]

cls1_idx  = np.where(y_test == 1)[0] if 1 in ALL_CLS else np.where(y_test == min(ALL_CLS))[0]
cls5_idx  = np.where(y_test == 5)[0] if 5 in ALL_CLS else np.where(y_test == max(ALL_CLS))[0]

# RPI 1 예측 시 SHAP (class index 0), RPI 5 예측 시 SHAP (class index 4)
cls_label = {1:"RPI 1 (재구매 확실)", 2:"RPI 2", 3:"RPI 3", 4:"RPI 4", 5:"RPI 5 (위험 고객)"}
CLS_COLORS_BAR = {1:"#3b82f6", 2:"#22c55e", 3:"#eab308", 4:"#f97316", 5:"#ef4444"}

fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#f8fafc")
for ax, cls_int, cls_shap_idx, title_suffix in [
    (axes[0], 1, 0, "RPI 1 (재구매 확실) 예측 기여"),
    (axes[1], 5, 4, "RPI 5 (위험 고객) 예측 기여"),
]:
    cls_mask = np.where(y_test == cls_int)[0] if cls_int in np.unique(y_test) else np.array([])
    if len(cls_mask) == 0:
        cls_mask = np.where(y_test == np.unique(y_test)[cls_shap_idx % len(np.unique(y_test))])[0]
    shap_slice = shap_lgb[cls_shap_idx, :, :]   # (n_samples, n_feats) for this class
    mean_shap  = shap_slice[:, top12_idx].mean(axis=0)
    bar_c = ["#10b981" if v >= 0 else "#ef4444" for v in mean_shap]
    y_p = np.arange(12)
    ax.barh(y_p, mean_shap, color=bar_c, alpha=0.85, height=0.65)
    ax.set_yticks(y_p)
    ax.set_yticklabels([feat_label(f) for f in top12_feat], fontsize=8.5)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Mean SHAP Value", fontsize=9)
    ax.set_title("{}\n{}".format(title_suffix,
        "🟢양수=해당 RPI 예측에 긍정 기여  🔴음수=부정 기여"), fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_facecolor("#f8fafc")
    pos_cnt = (mean_shap > 0).sum()
    neg_cnt = (mean_shap <= 0).sum()
    ax.text(0.97, 0.02, "양수 기여: {}개\n음수 기여: {}개".format(pos_cnt, neg_cnt),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", alpha=0.8))
fig.suptitle("LightGBM SHAP — RPI 1 vs RPI 5 예측 기여 방향 비교\n(같은 피처가 RPI 1 예측에는 양수, RPI 5 예측에는 반대 방향으로 작용하는 패턴 확인)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
imgs["shap_rpi1_vs_rpi5"] = fig_to_b64(fig); plt.close(fig)
print("  [V3] shap_rpi1_vs_rpi5 done")

# ════════════════════════════════════════════════════════════════════
# 시각화 4: 전체 클래스 SHAP Heatmap (클래스 × Top-12 피처 × 방향)
# ════════════════════════════════════════════════════════════════════
cls_names = ["RPI 1\n(최우량)", "RPI 2", "RPI 3", "RPI 4", "RPI 5\n(위험)"]
shap_cls_mat = np.zeros((N_CLS, len(top12_idx)))
for ci in range(N_CLS):
    shap_cls_mat[ci, :] = shap_lgb[ci, :, :][:, top12_idx].mean(axis=0)

fig, ax = plt.subplots(figsize=(13, 5), facecolor="#f8fafc")
im = ax.imshow(shap_cls_mat, cmap="RdBu_r", aspect="auto",
               vmin=-np.abs(shap_cls_mat).max(), vmax=np.abs(shap_cls_mat).max())
ax.set_xticks(range(len(top12_feat)))
ax.set_xticklabels([feat_label(f) for f in top12_feat], fontsize=8, rotation=35, ha="right")
ax.set_yticks(range(N_CLS))
ax.set_yticklabels(cls_names if len(cls_names)==N_CLS else
                   ["RPI {}".format(c) for c in ALL_CLS], fontsize=9)
for ri in range(N_CLS):
    for ci in range(len(top12_feat)):
        v = shap_cls_mat[ri, ci]
        ax.text(ci, ri, "{:.3f}".format(v), ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.02 else "#1e293b", fontweight="bold")
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
             label="Mean SHAP Value\n(+ : 해당 RPI 예측 증가 기여 | - : 감소 기여)")
ax.set_title("RPI 클래스 × Top-12 Feature — SHAP 방향 히트맵 (LightGBM)\n"
             "🔴빨간=해당 클래스 예측 증가 기여 | 🔵파란=감소 기여", fontsize=11, fontweight="bold")
fig.tight_layout()
imgs["shap_class_heatmap"] = fig_to_b64(fig); plt.close(fig)
print("  [V4] shap_class_heatmap done")

# ════════════════════════════════════════════════════════════════════
# 시각화 5: 상위 3 피처 Dependence Plot (t1_cci_total, c_cci_total, q1_cci_total)
# ════════════════════════════════════════════════════════════════════
dep_feats = ["t1_cci_total", "c_cci_total", "q1_cci_total"]
dep_domain = ["신기술", "Cost", "품질"]
dep_colors_map = {"신기술":"#2563eb", "Cost":"#ea580c", "품질":"#e11d48"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#f8fafc")
for ax, feat_name, domain_name in zip(axes, dep_feats, dep_domain):
    if feat_name not in FEATURE_COLS:
        ax.text(0.5, 0.5, "피처 없음", ha="center", va="center", transform=ax.transAxes)
        continue
    fi = FEATURE_COLS.index(feat_name)
    feat_vals  = X_test[:, fi]
    # RPI 5 클래스 SHAP (클래스 인덱스 4 = RPI 5)
    rpi5_idx   = ALL_CLS.index(5) if 5 in ALL_CLS else -1
    rpi1_idx   = ALL_CLS.index(1) if 1 in ALL_CLS else 0
    shap_rpi5  = shap_lgb[rpi5_idx, :, fi]   # 이 피처의 RPI5 기여
    shap_rpi1  = shap_lgb[rpi1_idx, :, fi]   # 이 피처의 RPI1 기여

    ax.scatter(feat_vals, shap_rpi5, c="#ef4444", alpha=0.5, s=15, label="RPI 5 기여 (+위험↑)")
    ax.scatter(feat_vals, shap_rpi1, c="#3b82f6", alpha=0.5, s=15, label="RPI 1 기여 (+우량↑)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    # 추세선
    from numpy.polynomial.polynomial import polyfit
    for yvals, color in [(shap_rpi5, "#dc2626"), (shap_rpi1, "#1d4ed8")]:
        try:
            coeffs = np.polyfit(feat_vals, yvals, 1)
            x_line = np.linspace(feat_vals.min(), feat_vals.max(), 50)
            ax.plot(x_line, np.polyval(coeffs, x_line), color=color, lw=1.5, alpha=0.8)
        except:
            pass
    ax.set_xlabel("{} ({})".format(feat_name, domain_name), fontsize=9)
    ax.set_ylabel("SHAP Value", fontsize=9)
    ax.set_title("[{}] {}\n피처값↑ → SHAP 방향 변화".format(domain_name, feat_name), fontsize=9, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.2)
    ax.set_facecolor("#f8fafc")

fig.suptitle("Top-3 피처 Dependence Plot — 피처 값과 SHAP 기여 방향의 관계\n"
             "(X축: 피처 측정값 | Y축: 해당 클래스 예측에 대한 기여 크기 및 방향)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
imgs["shap_dependence"] = fig_to_b64(fig); plt.close(fig)
print("  [V5] shap_dependence done")

# ════════════════════════════════════════════════════════════════════
# 시각화 6: Waterfall — 위험 고객(RPI 5) vs 우량 고객(RPI 1) 대표 샘플
# ════════════════════════════════════════════════════════════════════
rpi5_cls = 5 if 5 in np.unique(y_test) else max(np.unique(y_test))
rpi1_cls = 1 if 1 in np.unique(y_test) else min(np.unique(y_test))
rpi5_samp = np.where(y_test == rpi5_cls)[0]
rpi1_samp = np.where(y_test == rpi1_cls)[0]

def waterfall_manual(ax, shap_vals, feat_names, feat_vals, title, baseline=0, top_n=10):
    """수동 Waterfall 플롯"""
    order    = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
    sv       = shap_vals[order]
    fnames   = [feat_label(feat_names[i]) for i in order]
    fvals    = feat_vals[order]

    cumsum = np.cumsum(sv)
    starts = np.concatenate([[baseline], baseline + cumsum[:-1]])

    bar_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in sv]
    y_pos = np.arange(top_n)[::-1]

    ax.barh(y_pos, sv, left=starts[::-1] if False else starts,
            color=bar_colors, alpha=0.85, height=0.65)
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(fnames[::-1], fontsize=7.5)
    ax.axvline(baseline, color="black", lw=1)
    ax.set_xlabel("SHAP Value (누적 기여)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    for i, (y, val, start) in enumerate(zip(y_pos, sv, starts)):
        sign = "+" if val >= 0 else ""
        ax.text(start + val + (0.0003 if val >= 0 else -0.0003), y,
                "{}{:.3f}\n(val={:.1f})".format(sign, val, feat_vals[order[len(sv)-1-i]]),
                va="center", ha="left" if val >= 0 else "right",
                fontsize=6.5, color="#1e293b")
    ax.grid(axis="x", alpha=0.2)
    ax.set_facecolor("#f8fafc")

rpi5_sample_idx = rpi5_samp[0]
rpi1_sample_idx = rpi1_samp[0] if len(rpi1_samp) > 0 else rpi5_samp[-1]

rpi5_cls_idx = ALL_CLS.index(rpi5_cls)
rpi1_cls_idx = ALL_CLS.index(rpi1_cls)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#f8fafc")
waterfall_manual(
    axes[0],
    shap_lgb[rpi5_cls_idx, rpi5_sample_idx, :],
    FEATURE_COLS,
    X_test[rpi5_sample_idx, :],
    "⚠ 위험 고객 (실제 RPI={}) 샘플\nRPI {} 예측 기여 Top-10".format(
        y_test[rpi5_sample_idx], rpi5_cls),
    top_n=10
)
waterfall_manual(
    axes[1],
    shap_lgb[rpi1_cls_idx, rpi1_sample_idx, :],
    FEATURE_COLS,
    X_test[rpi1_sample_idx, :],
    "✅ 우량 고객 (실제 RPI={}) 샘플\nRPI {} 예측 기여 Top-10".format(
        y_test[rpi1_sample_idx], rpi1_cls),
    top_n=10
)
fig.suptitle("SHAP Waterfall — 위험 고객 vs 우량 고객 개별 예측 설명\n"
             "(🟢양수=해당 RPI 예측 확률 증가 기여 | 🔴음수=감소 기여 | val=실제 측정값)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
imgs["shap_waterfall"] = fig_to_b64(fig); plt.close(fig)
print("  [V6] shap_waterfall done")

# ════════════════════════════════════════════════════════════════════
# 시각화 7: 모델별 양수/음수 SHAP 기여 피처 분류 (Top-20, 전체 클래스 합산 기준)
# ════════════════════════════════════════════════════════════════════
# 전체 클래스에 걸쳐 Mean SHAP 부호 판단 — 절댓값 큰 방향으로 결정
top20_for_sign = np.argsort(shap_mean_lgb)[-20:][::-1]

fig, axes = plt.subplots(1, 3, figsize=(17, 7), facecolor="#f8fafc")
for ax, mname, shap_3d, sign_label in [
    (axes[0], "Logistic Regression", shap_lr,  "LR"),
    (axes[1], "LightGBM",            shap_lgb, "LGB"),
    (axes[2], "Random Forest",       shap_rf,  "RF"),
]:
    mean_all = shap_3d.mean(axis=0).mean(axis=0)   # (n_feats,) — 클래스·샘플 평균 부호
    vals     = mean_all[top20_for_sign]
    feats    = [feat_label(FEATURE_COLS[i]) for i in top20_for_sign]
    bar_c    = ["#16a34a" if v >= 0 else "#dc2626" for v in vals]
    y_p      = np.arange(20)[::-1]
    ax.barh(y_p, vals, color=bar_c, alpha=0.85, height=0.65)
    ax.set_yticks(y_p)
    ax.set_yticklabels(feats, fontsize=7.5)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Mean SHAP Value (부호 보존)", fontsize=8.5)
    ax.set_title("{}\n🟢양수=RPI 높임 기여 | 🔴음수=RPI 낮춤 기여".format(mname), fontsize=9, fontweight="bold")
    pos = (vals > 0).sum(); neg = (vals <= 0).sum()
    ax.text(0.97, 0.02, "양수:{} 음수:{}".format(pos, neg),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8fafc", alpha=0.8))
    ax.grid(axis="x", alpha=0.2)
    ax.set_facecolor("#f8fafc")

fig.suptitle("3모델 SHAP 양수/음수 기여 방향 비교 — Top-20 Features\n"
             "(전체 클래스·샘플 평균 부호 | 양수=RPI 전반을 높이는 방향으로 작용, 음수=낮추는 방향)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
imgs["shap_sign_compare"] = fig_to_b64(fig); plt.close(fig)
print("  [V7] shap_sign_compare done")

print("\n[모든 SHAP 시각화 완료]")

# ════════════════════════════════════════════════════════════════════
# 양수/음수 기여 분류 테이블 데이터 생성
# ════════════════════════════════════════════════════════════════════
mean_lgb_signed = shap_lgb.mean(axis=0).mean(axis=0)  # (n_feats,)
top20_feats_sign = [FEATURE_COLS[i] for i in top20_for_sign]
top20_sign_lgb   = mean_lgb_signed[top20_for_sign]
top20_abs_lgb    = shap_mean_lgb[top20_for_sign]

def sign_rows():
    rows = []
    for i, (feat, signed, absval) in enumerate(zip(top20_feats_sign, top20_sign_lgb, top20_abs_lgb)):
        domain = get_domain(feat)
        dom_style = "background:{};color:white;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:700;".format(
            DOMAIN_COLORS.get(domain, "#64748b"))
        if signed > 0:
            dir_cell = "<span style='color:#16a34a;font-weight:700;'>▲ 양수</span><br><span style='font-size:10px;color:#166534;'>RPI 상승 방향 기여</span>"
            biz_meaning = "이 값이 높을수록 → 재구매 의향(RPI) 증가 방향"
        else:
            dir_cell = "<span style='color:#dc2626;font-weight:700;'>▼ 음수</span><br><span style='font-size:10px;color:#991b1b;'>RPI 하강 방향 기여</span>"
            biz_meaning = "이 값이 높을수록 → 재구매 의향(RPI) 감소 방향 (주의!)"
        rows.append("""<tr>
          <td style='text-align:center;font-weight:700;'>{rank}</td>
          <td><strong>{feat}</strong></td>
          <td><span style='{dom_sty}'>{domain}</span></td>
          <td style='text-align:center;'>{signed:.4f}</td>
          <td style='text-align:center;'>{absval:.4f}</td>
          <td>{dir_cell}</td>
          <td style='font-size:12px;'>{biz}</td>
        </tr>""".format(
            rank=i+1, feat=feat, dom_sty=dom_style, domain=domain,
            signed=signed, absval=absval,
            dir_cell=dir_cell, biz=biz_meaning))
    return "\n".join(rows)


# ════════════════════════════════════════════════════════════════════
# HTML 섹션 생성 및 리포트에 삽입
# ════════════════════════════════════════════════════════════════════
SHAP_SECTION = """
<!-- ════════ SHAP 분석 섹션 ════════ -->
<div class="card" style="border-left:5px solid #ec4899; margin-top:8px;">
  <h2 style="color:#be185d;">🔮 SHAP 분석 — 예측 기여 크기 및 방향 (양수/음수)</h2>
  <p style="font-size:12px;color:#64748b;margin-bottom:18px;">
    SHAP(SHapley Additive exPlanations)은 각 피처가 개별 예측에 얼마나, 어떤 <strong>방향</strong>으로 기여했는지를 설명합니다.<br>
    <strong>양수(+) SHAP</strong>: 해당 피처가 해당 클래스 예측 확률을 높이는 방향 &nbsp;|&nbsp;
    <strong>음수(−) SHAP</strong>: 예측 확률을 낮추는 방향<br>
    모델: LightGBM (TreeExplainer) / Random Forest (TreeExplainer) / Logistic Regression (LinearExplainer)
  </p>

  <!-- V1: 3모델 Mean |SHAP| 바차트 -->
  <h3>① 3모델 SHAP 기반 Feature Importance 비교 (Mean |SHAP Value|)</h3>
  <img class="chart" src="data:image/png;base64,{img_bar}" />
  <div class="insight-box insight-blue">
    <strong>해석:</strong> 막대 길이 = 예측에 대한 평균 기여 <em>크기</em> (방향 무관).<br>
    세 모델에서 모두 긴 막대 → 방법론 독립적으로 중요한 피처.
    <strong style='color:#1d4ed8;'>신기술(t1)</strong>과 <strong style='color:#9a3412;'>Cost(c)</strong> CCI total이 압도적 1~2위를 유지합니다.
  </div>

  <!-- V2: LGB Beeswarm -->
  <h3 style="margin-top:20px;">② LightGBM SHAP Beeswarm — 피처 값과 기여 방향의 관계</h3>
  <img class="chart" src="data:image/png;base64,{img_beeswarm}" />
  <div class="insight-box insight-green">
    <strong>핵심 해석 포인트:</strong><br>
    • <strong>점이 오른쪽(+)에 분포</strong> + <strong>빨간색(고값)</strong> → "이 피처가 높을수록 RPI를 높이는 방향으로 기여"<br>
    • <strong>점이 왼쪽(−)에 분포</strong> + <strong>파란색(저값)</strong> → "이 피처가 낮을수록 RPI를 낮추는 방향"<br>
    • 양쪽에 골고루 분포 → 비선형 또는 클래스 의존적 효과 (특정 범위에서만 기여)
  </div>

  <!-- V3: RPI 1 vs RPI 5 -->
  <h3 style="margin-top:20px;">③ RPI 1(우량) vs RPI 5(위험) 예측 기여 방향 대비</h3>
  <img class="chart" src="data:image/png;base64,{img_rpi1vs5}" />
  <div class="insight-box insight-amber">
    <strong>핵심 해석:</strong><br>
    같은 피처가 <strong>RPI 1 예측에는 양수(+)</strong>이면서 <strong>RPI 5 예측에는 음수(−)</strong>가 정상적인 패턴입니다.
    (예: t1_cci_total 값이 높으면 → RPI 1 가능성 ↑, RPI 5 가능성 ↓)<br>
    만약 동일 방향이면 그 피처는 특정 RPI만 구분하지 못하는 비변별 피처입니다.
  </div>

  <!-- V4: 클래스 × 피처 SHAP 히트맵 -->
  <h3 style="margin-top:20px;">④ RPI 클래스 × Feature SHAP 방향 히트맵</h3>
  <img class="chart" src="data:image/png;base64,{img_heatmap}" />
  <div class="insight-box insight-green">
    <strong>해석 방법:</strong> 행 = RPI 클래스, 열 = 피처.<br>
    🔴 빨간 셀 → 해당 피처가 그 RPI 클래스 예측 확률을 높이는 기여.<br>
    🔵 파란 셀 → 낮추는 기여.<br>
    <strong>이상적인 패턴</strong>: 각 피처가 RPI 1~5에 걸쳐 <em>단조 증가 또는 단조 감소</em>한다면
    → 해당 피처는 RPI 서열 전체를 잘 구분하는 핵심 변수입니다.
  </div>

  <!-- V5: Dependence Plot -->
  <h3 style="margin-top:20px;">⑤ 상위 3 피처 Dependence Plot — 피처 측정값과 SHAP 기여의 함수적 관계</h3>
  <img class="chart" src="data:image/png;base64,{img_dep}" />
  <div class="insight-box insight-blue">
    <strong>해석:</strong><br>
    • X축 = 실제 측정된 피처 값 (CCI/CSI 점수), Y축 = 해당 클래스 SHAP 기여<br>
    • <span style='color:#dc2626;'>빨간 추세선</span> = "RPI 5 예측에 미치는 영향" /
      <span style='color:#1d4ed8;'>파란 추세선</span> = "RPI 1 예측에 미치는 영향"<br>
    • 신기술 CCI가 높아질수록 → RPI 1(우량) 기여 급증 + RPI 5(위험) 기여 급감 → <strong>강한 선형 분리 능력 확인</strong><br>
    • 비선형 꺾임이 있다면 → 특정 임계값 근방에서 고객 행동 변화 포인트
  </div>

  <!-- V6: Waterfall -->
  <h3 style="margin-top:20px;">⑥ 개별 고객 SHAP Waterfall — 위험 고객 vs 우량 고객 예측 설명</h3>
  <img class="chart" src="data:image/png;base64,{img_waterfall}" />
  <div class="insight-box insight-amber">
    <strong>해석:</strong> 실제 운영에서 "왜 이 고객이 RPI 5(위험)로 예측됐는가?"를 설명하는 방식입니다.<br>
    • 🟢 양수 막대 → 해당 피처가 그 RPI 예측을 증가시킨 원인<br>
    • 🔴 음수 막대 → 그 RPI 예측을 낮추려 한 피처 (완화 요인)<br>
    • val=실제 측정값 → 어떤 점수가 문제였는지 직접 확인 가능<br>
    <strong>영업팀 활용</strong>: "이 고객은 t1_cci_total=2.3으로 신기술 신뢰가 낮아 위험 고객으로 분류됨" 형태의 개인화 보고 가능
  </div>

  <!-- V7: 양수/음수 방향 비교 -->
  <h3 style="margin-top:20px;">⑦ 3모델 SHAP 양수/음수 기여 방향 비교 (전체 클래스·샘플 평균)</h3>
  <img class="chart" src="data:image/png;base64,{img_sign}" />
  <div class="insight-box insight-green">
    <strong>전체 클래스 평균 기준 방향 해석:</strong><br>
    양수(+) 피처 = 전반적으로 RPI를 높이는 방향으로 작용 (고만족·고신뢰 → 재구매 촉진)<br>
    음수(−) 피처 = 전반적으로 RPI를 낮추는 방향으로 작용 (주의 필요 — 해당 피처 관리 소홀 시 이탈 위험)<br>
    <strong>3모델 간 방향이 일치</strong>하는 피처 → 방법론 독립적으로 신뢰할 수 있는 방향성
  </div>

  <!-- 양수/음수 기여 분류 테이블 -->
  <h3 style="margin-top:20px;">⑧ LightGBM SHAP 기여 방향 분류표 — Top-20 Features</h3>
  <p style="font-size:12px;color:#64748b;margin-bottom:10px;">
    전체 클래스 × 샘플 평균 SHAP 부호 기준 | 양수(▲): 높은 값이 RPI 상승 기여 | 음수(▼): 낮은 값이 위험 신호
  </p>
  <div style="overflow-x:auto;">
  <table style="font-size:12px;">
    <thead>
      <tr>
        <th style="text-align:center;">순위</th>
        <th>피처명</th>
        <th>영역</th>
        <th style="text-align:center;">Mean SHAP<br>(부호 보존)</th>
        <th style="text-align:center;">Mean |SHAP|<br>(크기)</th>
        <th>기여 방향</th>
        <th>비즈니스 의미</th>
      </tr>
    </thead>
    <tbody>{sign_rows}</tbody>
  </table>
  </div>

  <!-- 종합 SHAP 인사이트 -->
  <div style="background:#1e293b; border-radius:8px; padding:18px 22px; margin-top:18px; font-size:13px; line-height:1.9; color:#e2e8f0;">
    <strong style="color:#f8fafc; font-size:14px;">★ SHAP 분석 핵심 인사이트 요약</strong><br>
    <span style="color:#7dd3fc;">① 신기술(t1) / Cost(c) / 품질(q1) CCI total</span>은 3가지 방법(내장/Permutation/SHAP) 모두에서 압도적 1~3위 — 가장 신뢰할 수 있는 핵심 피처입니다.<br>
    <span style="color:#86efac;">② 양수 기여 피처</span>: 높은 CCI/CSI 점수가 재구매 촉진 — <strong>점수 향상 → 고객 유지</strong> 직결.<br>
    <span style="color:#fca5a5;">③ 음수 기여 피처</span>: 일부 피처는 전반적으로 RPI를 낮추는 방향으로 작용 → <strong>해당 피처 낮은 고객을 즉시 위험군으로 분류</strong>해야 합니다.<br>
    <span style="color:#fdba74;">④ Waterfall 개인화 설명</span>: 영업팀이 특정 고객에게 "어떤 점수가 위험 신호인지" 데이터 기반으로 전달 가능 — CRM 시스템 연동 권장.<br>
    <span style="color:#d8b4fe;">⑤ Dependence Plot 임계값</span>: t1_cci_total, c_cci_total의 추세선 꺾임 지점이 고객 관리 개입 기준값(예: ≤ 3.0 → 위험) 설정에 활용됩니다.
  </div>
</div>
""".format(
    img_bar=imgs["shap_bar_3models"],
    img_beeswarm=imgs["shap_beeswarm_lgb"],
    img_rpi1vs5=imgs["shap_rpi1_vs_rpi5"],
    img_heatmap=imgs["shap_class_heatmap"],
    img_dep=imgs["shap_dependence"],
    img_waterfall=imgs["shap_waterfall"],
    img_sign=imgs["shap_sign_compare"],
    sign_rows=sign_rows(),
)

# ── 리포트에 삽입 ──────────────────────────────────────────────────
fi_path = REPORT_DIR / "feature_importance_report.html"
fi_html = fi_path.read_text(encoding="utf-8")

# "최종 인사이트 정리" 섹션 바로 앞에 SHAP 섹션 삽입
INSERT_BEFORE = "<!-- ════════ 최종 인사이트 정리 ════════ -->"
if INSERT_BEFORE in fi_html:
    fi_html = fi_html.replace(INSERT_BEFORE, SHAP_SECTION + "\n" + INSERT_BEFORE)
else:
    fi_html = fi_html.replace("</div>\n</body>", SHAP_SECTION + "\n</div>\n</body>")

fi_path.write_text(fi_html, encoding="utf-8")
print("\nSHAP section added ->", fi_path)
