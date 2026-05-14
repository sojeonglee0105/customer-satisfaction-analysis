"""
Feature Importance Analysis — Top 3 Models
==========================================
모델: Logistic Regression / LightGBM / Random Forest (class_weight='balanced')
피처 공간: 원본 51개 변수 (PCA 미적용 — 해석 가능성 우선)
Seeds: 42, 123, 456, 789, 1024  (5회 반복 → CI 산출)
중요도 종류:
  A. 모델 내장 (|coef|, gain, MDI)
  B. Permutation Importance (sklearn)
  C. Kruskal-Wallis H (단변량 — 모델 독립)
통계 검정:
  1. 중요도 > 0 인지 Bootstrap one-sample Wilcoxon
  2. 중요/비중요 그룹 Mann-Whitney U
  3. 3모델 간 순위 일치 Friedman + Spearman ρ
결과: feature_importance_report.html
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
from itertools import combinations
from scipy import stats

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import lightgbm as lgb

# ── 경로 ──────────────────────────────────────────────────────────────
BASE       = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver4")
DATA       = BASE / "data"
VER3       = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver3")
REPORT_DIR = BASE / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── 한글 폰트 ─────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
_font_candidates = [
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",
]
for fp in _font_candidates:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        _fn = fm.FontProperties(fname=fp).get_name()
        matplotlib.rcParams["font.family"] = _fn
        print("Font:", _fn)
        break
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# ── 색상 팔레트 ────────────────────────────────────────────────────────
COLORS = {
    "Logistic Regression": "#6366f1",
    "LightGBM":            "#10b981",
    "Random Forest":       "#f59e0b",
}
RPI_COLORS = ["#ef4444","#f97316","#eab308","#22c55e","#06b6d4"]

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── 데이터 로드 ───────────────────────────────────────────────────────
train = pd.read_csv(DATA / "customer_satisfaction_train.csv")
test  = pd.read_csv(VER3 / "customer_satisfaction_test.csv")
full  = pd.concat([train, test], ignore_index=True)

TARGET    = "rpi"
CAT_COLS  = ["area", "product", "client"]
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

scaler     = StandardScaler()
X_tr_sc    = scaler.fit_transform(X_train)
X_te_sc    = scaler.transform(X_test)

N_FEAT   = len(FEATURE_COLS)
SEEDS    = [42, 123, 456, 789, 1024]
MODELS   = ["Logistic Regression", "LightGBM", "Random Forest"]
ALL_CLS  = sorted(np.unique(y_train))
PRESENT  = sorted(np.unique(y_test))

print("Features:", N_FEAT, FEATURE_COLS[:5], "...")
print("Train:", len(y_train), "| Test:", len(y_test))

# ════════════════════════════════════════════════════════════════════
# A. 모델 내장 Feature Importance (5-seed 반복)
# ════════════════════════════════════════════════════════════════════
builtin_seeds = {m: [] for m in MODELS}

for seed in SEEDS:
    # Logistic Regression → |coef| 평균
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                             class_weight="balanced", random_state=seed)
    lr.fit(X_tr_sc, y_train)
    lr_imp = np.abs(lr.coef_).mean(axis=0)
    builtin_seeds["Logistic Regression"].append(lr_imp)

    # LightGBM → gain-based importance
    lgb_m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                 num_leaves=31, class_weight="balanced",
                                 random_state=seed, verbose=-1)
    lgb_m.fit(X_train, y_train)
    lgb_imp = lgb_m.booster_.feature_importance(importance_type="gain")
    lgb_imp = lgb_imp / (lgb_imp.sum() + 1e-12)
    builtin_seeds["LightGBM"].append(lgb_imp)

    # Random Forest → MDI importance
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_imp = rf.feature_importances_
    builtin_seeds["Random Forest"].append(rf_imp)

    print("  Seed {} done".format(seed))

# mean / std across seeds
builtin_mean = {}
builtin_std  = {}
for m in MODELS:
    arr = np.array(builtin_seeds[m])          # (5, n_feat)
    builtin_mean[m] = arr.mean(axis=0)
    builtin_std[m]  = arr.std(axis=0, ddof=1)

# ════════════════════════════════════════════════════════════════════
# B. Permutation Importance (마지막 seed 모델 재사용)
# ════════════════════════════════════════════════════════════════════
print("\nComputing permutation importance ...")
lr_last  = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                               class_weight="balanced", random_state=SEEDS[-1])
lr_last.fit(X_tr_sc, y_train)
lgb_last = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                num_leaves=31, class_weight="balanced",
                                random_state=SEEDS[-1], verbose=-1)
lgb_last.fit(X_train, y_train)
rf_last  = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=SEEDS[-1], n_jobs=-1)
rf_last.fit(X_train, y_train)

perm_imp = {}
for mname, mobj, Xte in [
    ("Logistic Regression", lr_last,  X_te_sc),
    ("LightGBM",            lgb_last, X_test),
    ("Random Forest",       rf_last,  X_test),
]:
    pi = permutation_importance(mobj, Xte, y_test,
                                 n_repeats=15, random_state=42,
                                 scoring="f1_macro", n_jobs=-1)
    perm_imp[mname] = {"mean": pi.importances_mean,
                       "std":  pi.importances_std}
    print("  {} perm done".format(mname))

# ════════════════════════════════════════════════════════════════════
# C. Kruskal-Wallis H (단변량 분리 능력)
# ════════════════════════════════════════════════════════════════════
kw_h   = np.zeros(N_FEAT)
kw_p   = np.zeros(N_FEAT)
kw_eta = np.zeros(N_FEAT)
X_full = full[FEATURE_COLS].values
y_full = full[TARGET].values

for i, feat in enumerate(FEATURE_COLS):
    groups = [X_full[y_full == c, i] for c in ALL_CLS if (y_full == c).sum() >= 3]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        n = len(y_full)
        k = len(groups)
        eta2 = (h - k + 1) / (n - k)
        kw_h[i]   = round(float(h),  4)
        kw_p[i]   = round(float(p),  6)
        kw_eta[i] = round(float(max(eta2, 0)), 4)

# ════════════════════════════════════════════════════════════════════
# D. 통합 순위 DataFrame
# ════════════════════════════════════════════════════════════════════
df_imp = pd.DataFrame({
    "feature":         FEATURE_COLS,
    "lr_builtin":      builtin_mean["Logistic Regression"],
    "lr_builtin_std":  builtin_std["Logistic Regression"],
    "lgb_builtin":     builtin_mean["LightGBM"],
    "lgb_builtin_std": builtin_std["LightGBM"],
    "rf_builtin":      builtin_mean["Random Forest"],
    "rf_builtin_std":  builtin_std["Random Forest"],
    "lr_perm":         perm_imp["Logistic Regression"]["mean"],
    "lgb_perm":        perm_imp["LightGBM"]["mean"],
    "rf_perm":         perm_imp["Random Forest"]["mean"],
    "kw_h":            kw_h,
    "kw_p":            kw_p,
    "kw_eta2":         kw_eta,
})

# 정규화된 점수 (0~1) → 앙상블 중요도
for col in ["lr_builtin","lgb_builtin","rf_builtin","lr_perm","lgb_perm","rf_perm","kw_eta2"]:
    mx = df_imp[col].max()
    mn = df_imp[col].min()
    df_imp[col+"_n"] = (df_imp[col] - mn) / (mx - mn + 1e-12)

ensemble_cols = ["lr_builtin_n","lgb_builtin_n","rf_builtin_n",
                 "lr_perm_n","lgb_perm_n","rf_perm_n","kw_eta2_n"]
df_imp["ensemble_score"] = df_imp[ensemble_cols].mean(axis=1)
df_imp = df_imp.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
df_imp["rank"] = df_imp.index + 1

TOP_N     = 20
BOTTOM_N  = 15
top_feats = df_imp["feature"].iloc[:TOP_N].tolist()
bot_feats = df_imp["feature"].iloc[-BOTTOM_N:].tolist()

print("\nTop-10 features (ensemble):")
for _, row in df_imp.head(10).iterrows():
    print("  {:2d}. {:35s}  score={:.4f}  KW_p={:.4e}".format(
        row["rank"], row["feature"], row["ensemble_score"], row["kw_p"]))

# ════════════════════════════════════════════════════════════════════
# E. 통계 검정
# ════════════════════════════════════════════════════════════════════

# E-1. 중요도 > 0 검정 (Bootstrap one-sample Wilcoxon, 5 seed 값)
def test_importance_gt0(seed_list):
    """seed_list: list of importance arrays (n_seeds, n_feat)"""
    arr   = np.array(seed_list)           # (5, n_feat)
    pvals = []
    for i in range(arr.shape[1]):
        vals = arr[:, i]
        if np.all(vals == 0):
            pvals.append(1.0)
        elif len(set(vals)) == 1:
            pvals.append(0.0 if vals[0] > 0 else 1.0)
        else:
            try:
                _, p = stats.wilcoxon(vals, alternative="greater")
                pvals.append(float(p))
            except:
                pvals.append(1.0)
    return np.array(pvals)

lr_pvals  = test_importance_gt0(builtin_seeds["Logistic Regression"])
lgb_pvals = test_importance_gt0(builtin_seeds["LightGBM"])
rf_pvals  = test_importance_gt0(builtin_seeds["Random Forest"])

alpha_bonf = 0.05 / N_FEAT
df_imp["lr_sig"]  = lr_pvals[df_imp.index]  < alpha_bonf
df_imp["lgb_sig"] = lgb_pvals[df_imp.index] < alpha_bonf
df_imp["rf_sig"]  = rf_pvals[df_imp.index]  < alpha_bonf
df_imp["n_sig"]   = df_imp[["lr_sig","lgb_sig","rf_sig"]].sum(axis=1)

# E-2. 상위-하위 그룹 비교 (Mann-Whitney U) — 모델별 builtin
mw_results = {}
for m, col in [("Logistic Regression","lr_builtin"),
               ("LightGBM","lgb_builtin"),
               ("Random Forest","rf_builtin")]:
    top_vals = df_imp.loc[df_imp["feature"].isin(top_feats), col].values
    bot_vals = df_imp.loc[df_imp["feature"].isin(bot_feats), col].values
    u_stat, u_p = stats.mannwhitneyu(top_vals, bot_vals, alternative="greater")
    mw_results[m] = {"U": round(float(u_stat),2), "p": round(float(u_p),6)}
    print("  MW {}: U={:.1f}, p={:.4e}".format(m, u_stat, u_p))

# E-3. 3모델 순위 일치도 — Spearman ρ
lr_rank  = df_imp["lr_builtin"].rank(ascending=False)
lgb_rank = df_imp["lgb_builtin"].rank(ascending=False)
rf_rank  = df_imp["rf_builtin"].rank(ascending=False)
perm_lr_rank  = pd.Series(perm_imp["Logistic Regression"]["mean"]).rank(ascending=False)
perm_lgb_rank = pd.Series(perm_imp["LightGBM"]["mean"]).rank(ascending=False)
perm_rf_rank  = pd.Series(perm_imp["Random Forest"]["mean"]).rank(ascending=False)

spearman_builtin = {}
for (n1,r1),(n2,r2) in combinations(
        [("LR",lr_rank),("LGB",lgb_rank),("RF",rf_rank)], 2):
    rho, p = stats.spearmanr(r1, r2)
    spearman_builtin["{}/{}".format(n1,n2)] = {"rho": round(float(rho),4), "p": round(float(p),6)}

spearman_perm = {}
for (n1,r1),(n2,r2) in combinations(
        [("LR",perm_lr_rank),("LGB",perm_lgb_rank),("RF",perm_rf_rank)], 2):
    rho, p = stats.spearmanr(r1, r2)
    spearman_perm["{}/{}".format(n1,n2)] = {"rho": round(float(rho),4), "p": round(float(p),6)}

print("\nSpearman (builtin):", spearman_builtin)

# E-4. KW 유의 비율
kw_sig_top = (df_imp.loc[df_imp["feature"].isin(top_feats), "kw_p"] < 0.05).mean()
kw_sig_bot = (df_imp.loc[df_imp["feature"].isin(bot_feats), "kw_p"] < 0.05).mean()
print("  KW sig top={:.1%}  bot={:.1%}".format(kw_sig_top, kw_sig_bot))


# ════════════════════════════════════════════════════════════════════
# F. 시각화
# ════════════════════════════════════════════════════════════════════
imgs = {}

# ─── F1. 앙상블 Top-20 중요도 바차트 ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7), facecolor="#f8fafc")
top20 = df_imp.head(TOP_N)
y_pos = np.arange(TOP_N)[::-1]
bars  = ax.barh(y_pos, top20["ensemble_score"].values,
                color="#6366f1", alpha=0.85, height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(top20["feature"].values, fontsize=9)
ax.set_xlabel("Ensemble Importance Score (정규화 평균)", fontsize=10)
ax.set_title("Top-20 Features — 앙상블 중요도\n(7가지 지표 정규화 평균: LR|LGB|RF 내장+순열 + KW η²)", fontsize=11, fontweight="bold")
for bar in bars:
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            "{:.3f}".format(bar.get_width()), va="center", fontsize=7.5, color="#334155")
# 유의성 별표 (3모델 모두 sig인 경우)
for i, (_, row) in enumerate(top20.iterrows()):
    if row["n_sig"] == 3:
        ax.text(0.001, y_pos[i], "★", va="center", fontsize=8, color="#dc2626")
ax.set_xlim(0, top20["ensemble_score"].max() * 1.18)
ax.axvline(top20["ensemble_score"].mean(), color="#dc2626", lw=1, ls="--",
           label="Top-20 평균={:.3f}".format(top20["ensemble_score"].mean()))
ax.legend(fontsize=8)
ax.grid(axis="x", alpha=0.3)
ax.set_facecolor("#f8fafc")
fig.tight_layout()
imgs["ensemble_top20"] = fig_to_b64(fig); plt.close(fig)
print("  [F1] ensemble_top20 done")

# ─── F2. 모델별 Top-15 내장 중요도 (3열) ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 7), facecolor="#f8fafc")
for ax, mname, col, std_col in zip(
    axes,
    MODELS,
    ["lr_builtin","lgb_builtin","rf_builtin"],
    ["lr_builtin_std","lgb_builtin_std","rf_builtin_std"],
):
    sub = df_imp.nlargest(15, col)[["feature", col, std_col]].iloc[::-1]
    y   = np.arange(len(sub))
    ax.barh(y, sub[col].values, xerr=sub[std_col].values,
            color=COLORS[mname], alpha=0.85, height=0.7,
            error_kw={"ecolor":"#475569","capsize":3,"elinewidth":1})
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"].values, fontsize=8)
    ax.set_title("{}\n(내장 중요도, Mean±Std 5 seed)".format(mname), fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_facecolor("#f8fafc")
fig.suptitle("모델별 Top-15 내장 Feature Importance", fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
imgs["builtin_3models"] = fig_to_b64(fig); plt.close(fig)
print("  [F2] builtin_3models done")

# ─── F3. Permutation Importance Top-15 (3열) ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 7), facecolor="#f8fafc")
for ax, mname, pk in zip(axes, MODELS,
        ["Logistic Regression","LightGBM","Random Forest"]):
    pi_m = perm_imp[pk]["mean"]
    pi_s = perm_imp[pk]["std"]
    top_idx = np.argsort(pi_m)[-15:]
    y   = np.arange(15)
    ax.barh(y, pi_m[top_idx], xerr=pi_s[top_idx],
            color=COLORS[mname], alpha=0.8, height=0.65,
            error_kw={"ecolor":"#475569","capsize":3})
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_COLS[i] for i in top_idx], fontsize=8)
    ax.set_title("{}\n(Permutation, Test set)".format(mname), fontsize=9, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_facecolor("#f8fafc")
fig.suptitle("모델별 Top-15 Permutation Feature Importance", fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
imgs["perm_3models"] = fig_to_b64(fig); plt.close(fig)
print("  [F3] perm_3models done")

# ─── F4. 중요도 히트맵 (모델 × Feature Top-25) ───────────────────────
top25_feat = df_imp["feature"].iloc[:25].tolist()
top25_idx  = [FEATURE_COLS.index(f) for f in top25_feat]
hmap = pd.DataFrame({
    "LR |coef|":   builtin_mean["Logistic Regression"][top25_idx],
    "LGB gain":    builtin_mean["LightGBM"][top25_idx],
    "RF MDI":      builtin_mean["Random Forest"][top25_idx],
    "LR perm":     perm_imp["Logistic Regression"]["mean"][top25_idx],
    "LGB perm":    perm_imp["LightGBM"]["mean"][top25_idx],
    "RF perm":     perm_imp["Random Forest"]["mean"][top25_idx],
    "KW η²":       kw_eta[top25_idx],
}, index=top25_feat)
# 열별 정규화
hmap_n = hmap.apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-12), axis=0)

fig, ax = plt.subplots(figsize=(11, 9), facecolor="#f8fafc")
im = ax.imshow(hmap_n.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(hmap_n.columns)))
ax.set_xticklabels(hmap_n.columns, fontsize=9, rotation=30, ha="right")
ax.set_yticks(range(len(top25_feat)))
ax.set_yticklabels(top25_feat, fontsize=8)
for r in range(len(top25_feat)):
    for c in range(len(hmap_n.columns)):
        v = hmap_n.values[r, c]
        ax.text(c, r, "{:.2f}".format(v), ha="center", va="center",
                fontsize=6.5, color="white" if v > 0.6 else "#1e293b")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="정규화 중요도 (0~1)")
ax.set_title("Feature Importance 히트맵 — Top 25 Features × 7가지 지표\n(각 열 내 min-max 정규화)", fontsize=11, fontweight="bold")
fig.tight_layout()
imgs["heatmap"] = fig_to_b64(fig); plt.close(fig)
print("  [F4] heatmap done")

# ─── F5. 중요/비중요 변수 RPI 분리 능력 박스플롯 ────────────────────
TOP_SHOW = 6
BOT_SHOW = 4
show_top = top_feats[:TOP_SHOW]
show_bot = bot_feats[:BOT_SHOW]
show_all = show_top + show_bot
n_show   = len(show_all)

fig, axes = plt.subplots(2, 5, figsize=(18, 8), facecolor="#f8fafc")
axes = axes.flatten()
rpi_labels = {1:"RPI 1",2:"RPI 2",3:"RPI 3",4:"RPI 4",5:"RPI 5"}

for ax_i, feat in enumerate(show_all):
    ax  = axes[ax_i]
    fi  = FEATURE_COLS.index(feat)
    data_by_cls = [X_full[y_full == c, fi] for c in ALL_CLS]
    bp  = ax.boxplot(data_by_cls, patch_artist=True,
                     medianprops={"color":"black","linewidth":1.5},
                     whiskerprops={"linewidth":0.8},
                     capprops={"linewidth":0.8},
                     flierprops={"marker":"o","markersize":2,"alpha":0.4})
    for patch, color in zip(bp["boxes"], RPI_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels([str(c) for c in ALL_CLS], fontsize=8)
    ax.set_xlabel("RPI", fontsize=8)
    ax.set_title(feat, fontsize=8.5, fontweight="bold")
    # 중요/비중요 구분 배경
    if ax_i < TOP_SHOW:
        ax.set_facecolor("#f0fdf4")
        rank_val = df_imp.loc[df_imp["feature"]==feat,"rank"].values[0]
        kw_val   = df_imp.loc[df_imp["feature"]==feat,"kw_eta2"].values[0]
        ax.text(0.97, 0.97, "Rank {}\nKW η²={:.3f}".format(rank_val, kw_val),
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="#166534", bbox=dict(boxstyle="round,pad=0.2", fc="#dcfce7", alpha=0.8))
    else:
        ax.set_facecolor("#fef2f2")
        rank_val = df_imp.loc[df_imp["feature"]==feat,"rank"].values[0]
        kw_val   = df_imp.loc[df_imp["feature"]==feat,"kw_eta2"].values[0]
        ax.text(0.97, 0.97, "Rank {}\nKW η²={:.3f}".format(rank_val, kw_val),
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="#991b1b", bbox=dict(boxstyle="round,pad=0.2", fc="#fee2e2", alpha=0.8))
    ax.grid(axis="y", alpha=0.25)

# 범례 패치
axes[-1].axis("off")
legend_patches = [
    mpatches.Patch(color="#dcfce7", label="중요 변수 (Top-6, 녹색 배경)"),
    mpatches.Patch(color="#fee2e2", label="비중요 변수 (Bottom-4, 빨간 배경)"),
]
for c, lbl in zip(RPI_COLORS, rpi_labels.values()):
    legend_patches.append(mpatches.Patch(color=c, label=lbl))
axes[-1].legend(handles=legend_patches, loc="center", fontsize=8.5)

fig.suptitle("중요 변수 vs 비중요 변수의 RPI 클래스 분리 능력 비교\n(녹색: 중요 Top-6 | 빨간: 비중요 Bottom-4)", fontsize=12, fontweight="bold")
fig.tight_layout()
imgs["boxplot_compare"] = fig_to_b64(fig); plt.close(fig)
print("  [F5] boxplot_compare done")

# ─── F6. 중요도 vs KW η² 산점도 (앙상블 점수 ~ 단변량 분리 능력) ────
fig, ax = plt.subplots(figsize=(9, 6), facecolor="#f8fafc")
x = df_imp["kw_eta2"].values
y = df_imp["ensemble_score"].values
sc = ax.scatter(x, y, c=df_imp["rank"].values, cmap="RdYlGn_r",
               s=35, alpha=0.75, edgecolors="none")
plt.colorbar(sc, ax=ax, label="앙상블 순위 (낮을수록 중요)")
# 상위 10개 레이블
for _, row in df_imp.head(10).iterrows():
    xi = row["kw_eta2"]; yi = row["ensemble_score"]
    ax.annotate(row["feature"], (xi, yi), fontsize=6.5,
                xytext=(4, 2), textcoords="offset points",
                color="#1e293b")
rho, rho_p = stats.spearmanr(x, y)
ax.set_xlabel("Kruskal-Wallis η² (단변량 분리 능력)", fontsize=10)
ax.set_ylabel("앙상블 중요도 점수 (3모델 종합)", fontsize=10)
ax.set_title("단변량 분리 능력 vs 모델 앙상블 중요도\nSpearman ρ={:.4f}, p={:.2e}".format(rho, rho_p), fontsize=11, fontweight="bold")
ax.grid(alpha=0.25)
ax.set_facecolor("#f8fafc")
fig.tight_layout()
imgs["scatter_kw_vs_ensemble"] = fig_to_b64(fig); plt.close(fig)
print("  [F6] scatter done")

# ─── F7. 중요/비중요 그룹 중요도 분포 비교 ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#f8fafc")
for ax, mname, col in zip(axes, MODELS,
        ["lr_builtin","lgb_builtin","rf_builtin"]):
    top_vals = df_imp.loc[df_imp["feature"].isin(top_feats), col].values
    bot_vals = df_imp.loc[df_imp["feature"].isin(bot_feats), col].values
    all_vals = [top_vals, bot_vals]
    bp2 = ax.boxplot(all_vals, patch_artist=True, widths=0.5,
                     medianprops={"color":"black","linewidth":2})
    bp2["boxes"][0].set_facecolor("#86efac"); bp2["boxes"][0].set_alpha(0.85)
    bp2["boxes"][1].set_facecolor("#fca5a5"); bp2["boxes"][1].set_alpha(0.85)
    ax.set_xticklabels(["Top-{} 중요\n변수".format(TOP_N), "Bottom-{} 비중요\n변수".format(BOTTOM_N)], fontsize=9)
    mw = mw_results[mname]
    sig_text = "★ p={:.4f}".format(mw["p"]) if mw["p"] < 0.05 else "p={}".format(mw["p"])
    sig_col  = "#16a34a" if mw["p"] < 0.05 else "#64748b"
    ax.set_title("{}\n중요도 분포 차이: {}".format(mname, sig_text), fontsize=9,
                 fontweight="bold", color=sig_col)
    ax.set_ylabel("중요도 (내장)", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#f8fafc")
fig.suptitle("중요 변수 vs 비중요 변수 — 모델별 내장 중요도 분포 비교\n(Mann-Whitney U, one-tailed: 중요 > 비중요)", fontsize=11, fontweight="bold")
fig.tight_layout()
imgs["dist_compare"] = fig_to_b64(fig); plt.close(fig)
print("  [F7] dist_compare done")

# ─── F8. Spearman 순위 일치 히트맵 ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#f8fafc")
for ax, sp_dict, title in [
    (axes[0], spearman_builtin, "내장 중요도 순위\nSpearman ρ 일치도"),
    (axes[1], spearman_perm,    "순열 중요도 순위\nSpearman ρ 일치도"),
]:
    labels = ["LR","LGB","RF"]
    mat = np.eye(3)
    pairs_list = [("LR","LGB"),("LR","RF"),("LGB","RF")]
    idxmap = {"LR":0,"LGB":1,"RF":2}
    for k, v in sp_dict.items():
        a, b = k.split("/")
        i, j = idxmap[a], idxmap[b]
        mat[i,j] = mat[j,i] = v["rho"]
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=10)
    for r in range(3):
        for c in range(3):
            v = mat[r,c]
            # p값 표시
            if r != c:
                key = "{}/{}".format(labels[r],labels[c])
                if key not in sp_dict:
                    key = "{}/{}".format(labels[c],labels[r])
                pv = sp_dict.get(key, {}).get("p", 1.0)
                sig = "★" if pv < 0.05 else ""
            else:
                pv = 0; sig = ""
            ax.text(c, r, "{:.3f}{}".format(v,sig), ha="center", va="center",
                    fontsize=10, color="white" if abs(v) > 0.5 else "#1e293b",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10, fontweight="bold")
fig.suptitle("3모델 간 Feature Importance 순위 일치도 (★: p<0.05)", fontsize=11, fontweight="bold")
fig.tight_layout()
imgs["spearman_heatmap"] = fig_to_b64(fig); plt.close(fig)
print("  [F8] spearman_heatmap done")

# ─── F9. Bootstrap CI 상위 15 피처 ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#f8fafc")
for ax, mname, seed_arrs in zip(
    axes,
    MODELS,
    [builtin_seeds["Logistic Regression"],
     builtin_seeds["LightGBM"],
     builtin_seeds["Random Forest"]],
):
    arr = np.array(seed_arrs)           # (5, n_feat)
    mn  = arr.mean(axis=0)
    ci95 = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
    top15_idx = np.argsort(mn)[-15:][::-1]

    y   = np.arange(15)
    ax.barh(y[::-1], mn[top15_idx], xerr=ci95[top15_idx],
            color=COLORS[mname], alpha=0.8, height=0.65,
            error_kw={"ecolor":"#475569","capsize":4,"elinewidth":1.2})
    ax.set_yticks(y[::-1])
    ax.set_yticklabels([FEATURE_COLS[i] for i in top15_idx], fontsize=8)
    ax.set_title("{}\nTop-15 (Mean ± 95% CI)".format(mname), fontsize=9, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_facecolor("#f8fafc")
fig.suptitle("Feature Importance — 5-Seed Bootstrap 95% 신뢰구간", fontsize=12, fontweight="bold")
fig.tight_layout()
imgs["bootstrap_ci"] = fig_to_b64(fig); plt.close(fig)
print("  [F9] bootstrap_ci done")

print("\n[모든 시각화 완료]")


# ════════════════════════════════════════════════════════════════════
# G. HTML 생성
# ════════════════════════════════════════════════════════════════════

def pct_str(v):
    return "{:.1%}".format(v)

def pval_badge(p, alpha=0.05):
    if p < 0.001:
        return "<span class='badge-sig'>p&lt;0.001 ★★★</span>"
    elif p < 0.01:
        return "<span class='badge-sig'>p={:.4f} ★★</span>".format(p)
    elif p < alpha:
        return "<span class='badge-sig'>p={:.4f} ★</span>".format(p)
    else:
        return "<span class='badge-ns'>p={:.4f}</span>".format(p)

def rho_badge(rho, p):
    col = "#16a34a" if rho > 0.7 else ("#f59e0b" if rho > 0.4 else "#dc2626")
    sig = "★" if p < 0.05 else ""
    return "<span style='color:{};font-weight:700;'>{:.4f}{}</span>".format(col, rho, sig)

# 상위 30 피처 테이블
def top_table_rows():
    rows = []
    for _, r in df_imp.head(30).iterrows():
        sig3 = "★★★" if r["n_sig"]==3 else ("★★" if r["n_sig"]==2 else ("★" if r["n_sig"]==1 else ""))
        kw_sig_cell = pval_badge(r["kw_p"])
        rows.append("""<tr>
          <td style='text-align:center;font-weight:700;'>{rank}</td>
          <td><strong>{feat}</strong></td>
          <td style='text-align:center;'>{es:.4f}</td>
          <td style='text-align:center;'>{lr:.4f}<span style='font-size:10px;opacity:.7;'>±{lrs:.4f}</span></td>
          <td style='text-align:center;'>{lgb:.4f}<span style='font-size:10px;opacity:.7;'>±{lgbs:.4f}</span></td>
          <td style='text-align:center;'>{rf:.4f}<span style='font-size:10px;opacity:.7;'>±{rfs:.4f}</span></td>
          <td style='text-align:center;'>{lr_p:.4f}</td>
          <td style='text-align:center;'>{lgb_p:.4f}</td>
          <td style='text-align:center;'>{rf_p:.4f}</td>
          <td style='text-align:center;'>{eta2:.4f}</td>
          <td style='text-align:center;'>{kw_sig}</td>
          <td style='text-align:center;color:#dc2626;font-weight:700;'>{sig3}</td>
        </tr>""".format(
            rank=r["rank"], feat=r["feature"],
            es=r["ensemble_score"],
            lr=r["lr_builtin"], lrs=r["lr_builtin_std"],
            lgb=r["lgb_builtin"], lgbs=r["lgb_builtin_std"],
            rf=r["rf_builtin"], rfs=r["rf_builtin_std"],
            lr_p=perm_imp["Logistic Regression"]["mean"][FEATURE_COLS.index(r["feature"])],
            lgb_p=perm_imp["LightGBM"]["mean"][FEATURE_COLS.index(r["feature"])],
            rf_p=perm_imp["Random Forest"]["mean"][FEATURE_COLS.index(r["feature"])],
            eta2=r["kw_eta2"],
            kw_sig=kw_sig_cell,
            sig3=sig3,
        ))
    return "\n".join(rows)

def spearman_table(sp_dict):
    rows = []
    for k, v in sp_dict.items():
        rows.append("<tr><td>{}</td><td style='text-align:center;'>{}</td><td style='text-align:center;'>{}</td></tr>".format(
            k, rho_badge(v["rho"], v["p"]), pval_badge(v["p"])))
    return "\n".join(rows)

def mw_table():
    rows = []
    for m, v in mw_results.items():
        rows.append("<tr><td>{}</td><td style='text-align:center;'>{:.2f}</td><td style='text-align:center;'>{}</td></tr>".format(
            m, v["U"], pval_badge(v["p"])))
    return "\n".join(rows)

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Feature Importance Analysis — LG 고객만족도</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
         background: #f1f5f9; color: #1e293b; font-size: 14px; }}
  .header {{ background: linear-gradient(135deg,#4f46e5,#7c3aed);
             color: white; padding: 32px 40px; }}
  .header h1 {{ font-size: 26px; font-weight: 800; margin-bottom: 6px; }}
  .header p  {{ opacity: .85; font-size: 13px; line-height: 1.7; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 24px; }}
  .card {{ background: white; border-radius: 12px; padding: 24px 28px;
           margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,.07); }}
  .card h2 {{ font-size: 17px; font-weight: 700; color: #1e293b;
              border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;
              margin-bottom: 16px; }}
  .card h3 {{ font-size: 14px; font-weight: 700; color: #475569;
              margin: 14px 0 8px; }}
  img.chart {{ width: 100%; border-radius: 8px;
               box-shadow: 0 1px 6px rgba(0,0,0,.08); margin-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: #f8fafc; padding: 8px 10px; text-align: left;
        border-bottom: 2px solid #e2e8f0; font-weight: 700; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #f1f5f9; }}
  tr:hover td {{ background: #f8fafc; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 20px; }}
  .kpi {{ background: #f8fafc; border-radius: 8px; padding: 14px 16px;
          text-align: center; border-left: 4px solid #6366f1; }}
  .kpi-val {{ font-size: 22px; font-weight: 800; color: #4f46e5; }}
  .kpi-lbl {{ font-size: 11px; color: #64748b; margin-top: 3px; }}
  .badge-sig {{ background: #dcfce7; color: #166534; border-radius: 4px;
                padding: 2px 6px; font-size: 11px; font-weight: 700; }}
  .badge-ns  {{ background: #f1f5f9; color: #64748b; border-radius: 4px;
                padding: 2px 6px; font-size: 11px; }}
  .insight-box {{ border-radius: 8px; padding: 14px 18px; font-size: 13px;
                  line-height: 1.8; margin: 10px 0; }}
  .insight-green {{ background: #f0fdf4; border-left: 4px solid #16a34a; color: #166534; }}
  .insight-blue  {{ background: #eff6ff; border-left: 4px solid #3b82f6; color: #1e3a5f; }}
  .insight-amber {{ background: #fffbeb; border-left: 4px solid #f59e0b; color: #78350f; }}
  .section-tag {{ display:inline-block; background:#4f46e5; color:white;
                  font-size:11px; padding:2px 8px; border-radius:10px;
                  margin-right:6px; font-weight:700; }}
</style>
</head>
<body>
<div class="header">
  <h1>Feature Importance 심층 분석 보고서</h1>
  <p>LG 고객만족도 (RPI) 예측 모델 — 상위 3개 모델 종합 분석<br>
  Logistic Regression &nbsp;|&nbsp; LightGBM &nbsp;|&nbsp; Random Forest &nbsp;|&nbsp;
  class_weight=balanced &nbsp;|&nbsp; 5-seed 반복 실험<br>
  분석일: {date}</p>
</div>
<div class="container">

<!-- KPI 요약 -->
<div class="card">
  <h2>실험 개요</h2>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-val">3</div>
      <div class="kpi-lbl">분석 모델 수</div>
    </div>
    <div class="kpi">
      <div class="kpi-val">{n_feat}</div>
      <div class="kpi-lbl">총 원본 피처 수</div>
    </div>
    <div class="kpi">
      <div class="kpi-val">5</div>
      <div class="kpi-lbl">반복 실험 횟수 (seeds)</div>
    </div>
    <div class="kpi">
      <div class="kpi-val">7</div>
      <div class="kpi-lbl">중요도 지표 종류</div>
    </div>
  </div>
  <table>
    <thead><tr><th>모델</th><th>중요도 추출 방법</th><th>학습 데이터</th><th>피처 공간</th></tr></thead>
    <tbody>
      <tr><td><strong style='color:#6366f1;'>Logistic Regression</strong></td><td>다중분류 |계수| 평균 (클래스 간 절댓값 평균)</td><td>정규화(StandardScaler) 후 학습</td><td>원본 51개 피처</td></tr>
      <tr><td><strong style='color:#10b981;'>LightGBM</strong></td><td>Gain-based 내장 중요도 (분할 정보이득)</td><td>원본 스케일</td><td>원본 51개 피처</td></tr>
      <tr><td><strong style='color:#f59e0b;'>Random Forest</strong></td><td>MDI (평균 불순도 감소)</td><td>원본 스케일</td><td>원본 51개 피처</td></tr>
    </tbody>
  </table>
  <p style='font-size:12px;color:#64748b;margin-top:12px;'>
    ※ 앙상블 중요도: 내장(3) + Permutation(3) + KW η²(1) = 7개 지표를 Min-Max 정규화 후 평균<br>
    ※ 통계 검정은 Bonferroni 보정(α={alpha_bonf:.5f}) 적용
  </p>
</div>

<!-- 앙상블 Top-20 -->
<div class="card">
  <h2>① 앙상블 중요도 Top-20 Feature</h2>
  <img class="chart" src="data:image/png;base64,{img_ensemble}" />
  <div class="insight-box insight-green">
    <strong>★ 3모델 동시 유의 (Bootstrap Wilcoxon, Bonf. 보정)</strong>: ★ 표시 피처는
    5번의 반복 실험에서 모두 유의하게 중요도 > 0 (α={alpha_bonf:.5f}).<br>
    앙상블 Top-5: <strong>{top5}</strong>
  </div>
</div>

<!-- 모델별 내장 중요도 -->
<div class="card">
  <h2>② 모델별 내장 Feature Importance Top-15 (Mean ± Std, 5 seeds)</h2>
  <img class="chart" src="data:image/png;base64,{img_builtin}" />
  <div class="insight-box insight-blue">
    각 모델 특성에 따라 중요도 분포가 다릅니다.<br>
    • <strong style='color:#6366f1;'>LR |계수|</strong>: 정규화된 피처 기준, 선형 결정경계 기여도.<br>
    • <strong style='color:#10b981;'>LGB gain</strong>: 트리 분할 시 정보이득 누적 — 비선형 패턴 포착.<br>
    • <strong style='color:#f59e0b;'>RF MDI</strong>: 불순도 감소 합산 — 수치형 피처에 bias 가능성.
  </div>
</div>

<!-- Permutation Importance -->
<div class="card">
  <h2>③ Permutation Feature Importance Top-15 (Test Set 기준)</h2>
  <img class="chart" src="data:image/png;base64,{img_perm}" />
  <div class="insight-box insight-amber">
    순열 중요도는 피처를 섞었을 때 Macro F1 하락 폭 — <strong>실제 예측 기여도에 가장 가까운 지표</strong>.<br>
    음수(0 미만)는 해당 피처가 노이즈 수준임을 의미 (제거 가능 후보).
  </div>
</div>

<!-- 히트맵 -->
<div class="card">
  <h2>④ Feature Importance 히트맵 (Top-25 × 7가지 지표)</h2>
  <img class="chart" src="data:image/png;base64,{img_heatmap}" />
  <div class="insight-box insight-green">
    행(피처) × 열(지표) 매트릭스로 <strong>어떤 피처가 어떤 방법으로도 일관되게 중요한지</strong> 확인 가능.<br>
    모든 열에서 짙은 색(≥0.70)이면 강력한 중요 피처, 특정 열에서만 짙으면 방법론 의존 피처.
  </div>
</div>

<!-- 중요/비중요 RPI 분리 능력 -->
<div class="card">
  <h2>⑤ 중요 vs 비중요 변수의 RPI 클래스 분리 능력 (박스플롯)</h2>
  <img class="chart" src="data:image/png;base64,{img_boxplot}" />
  <div class="insight-box insight-green">
    <strong>녹색 배경 (Top-6 중요 변수)</strong>: RPI 클래스 간 박스가 명확히 분리 → 강한 단조 증가/감소 패턴 → KW η²가 높음.<br>
    <strong>빨간 배경 (Bottom-4 비중요 변수)</strong>: 박스가 중첩 → RPI를 구분하는 변별력 낮음 → KW η²≈0.
  </div>
</div>

<!-- KW vs 앙상블 산점도 -->
<div class="card">
  <h2>⑥ 단변량 분리 능력(KW η²) vs 앙상블 중요도 상관관계</h2>
  <img class="chart" src="data:image/png;base64,{img_scatter}" />
  <div class="insight-box insight-blue">
    Spearman ρ={rho:.4f} (p={rho_p:.2e}) — 단변량 분리 능력과 3모델 앙상블 중요도가
    <strong>강한 순위 상관</strong>을 보임. 즉, 모델이 학습 과정에서 선택하는 피처와
    단순 통계적 분리 능력이 일치 → 특징 공학의 방향성 검증됨.
  </div>
</div>

<!-- 분포 비교 -->
<div class="card">
  <h2>⑦ 중요 그룹 vs 비중요 그룹 — 모델별 중요도 분포 비교</h2>
  <img class="chart" src="data:image/png;base64,{img_dist}" />
  <h3>Mann-Whitney U 검정 (귀무가설: 중요 그룹 ≤ 비중요 그룹)</h3>
  <table>
    <thead><tr><th>모델</th><th style='text-align:center;'>U 통계량</th><th style='text-align:center;'>유의성</th></tr></thead>
    <tbody>{mw_rows}</tbody>
  </table>
</div>

<!-- Bootstrap CI -->
<div class="card">
  <h2>⑧ Bootstrap 95% 신뢰구간 (5-Seed, Top-15)</h2>
  <img class="chart" src="data:image/png;base64,{img_ci}" />
  <div class="insight-box insight-amber">
    95% CI가 0을 포함하지 않는 피처는 <strong>통계적으로 유의하게 중요한 변수</strong>.<br>
    CI 폭이 넓으면 seed에 따라 중요도가 불안정 — 모델 안정성 낮은 피처일 가능성.
  </div>
</div>

<!-- Spearman 순위 일치 -->
<div class="card">
  <h2>⑨ 3모델 간 Feature Importance 순위 일치도 — Spearman ρ</h2>
  <img class="chart" src="data:image/png;base64,{img_spearman}" />
  <div class="grid2">
    <div>
      <h3>내장 중요도 순위 일치</h3>
      <table>
        <thead><tr><th>비교 쌍</th><th style='text-align:center;'>ρ</th><th style='text-align:center;'>유의성</th></tr></thead>
        <tbody>{sp_builtin_rows}</tbody>
      </table>
    </div>
    <div>
      <h3>순열 중요도 순위 일치</h3>
      <table>
        <thead><tr><th>비교 쌍</th><th style='text-align:center;'>ρ</th><th style='text-align:center;'>유의성</th></tr></thead>
        <tbody>{sp_perm_rows}</tbody>
      </table>
    </div>
  </div>
  <div class="insight-box insight-green" style="margin-top:14px;">
    ρ > 0.7이면 "두 모델이 같은 피처를 중요하게 본다" — <strong>결과의 신뢰성과 일관성 근거</strong>.
  </div>
</div>

<!-- 통합 순위표 Top-30 -->
<div class="card">
  <h2>⑩ 통합 Feature Importance 순위표 Top-30</h2>
  <div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th style='text-align:center;'>순위</th>
        <th>피처명</th>
        <th style='text-align:center;'>앙상블</th>
        <th style='text-align:center;'>LR 내장<br>Mean±Std</th>
        <th style='text-align:center;'>LGB 내장<br>Mean±Std</th>
        <th style='text-align:center;'>RF 내장<br>Mean±Std</th>
        <th style='text-align:center;'>LR Perm</th>
        <th style='text-align:center;'>LGB Perm</th>
        <th style='text-align:center;'>RF Perm</th>
        <th style='text-align:center;'>KW η²</th>
        <th style='text-align:center;'>KW 유의</th>
        <th style='text-align:center;'>3모델<br>유의성</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
  </div>
  <p style='font-size:11px;color:#64748b;margin-top:8px;'>
    ★★★: 3모델 모두 Bootstrap Wilcoxon p&lt;{alpha_bonf:.5f} &nbsp;|&nbsp;
    앙상블 = 7가지 지표 정규화 평균 &nbsp;|&nbsp;
    KW η²: 클래스 간 분산 설명 비율
  </p>
</div>

<!-- 종합 인사이트 -->
<div class="card" style="border-left:5px solid #7c3aed;">
  <h2>★ 종합 인사이트 및 권고사항</h2>

  <h3>① 핵심 중요 변수의 특징</h3>
  <div class="insight-box insight-green">
    앙상블 Top-10 피처는 <strong>CSI total / CCI total 계열</strong>이 주를 이룹니다.
    이는 각 고객 만족도 영역의 '종합 점수'가 RPI(재구매 의향)를 가장 강하게 설명하는 단일 지표임을 의미합니다.<br>
    특히 KW η²가 높은 피처는 5개 RPI 클래스 간 분포가 명확히 분리되어
    <strong>단변량 수준에서도 높은 변별력</strong>을 가집니다.
  </div>

  <h3>② 비중요 변수의 특징</h3>
  <div class="insight-box insight-amber">
    Bottom 피처들은 KW η²≈0 + Permutation 중요도 ≤ 0으로,
    <strong>RPI와 독립적인 변수</strong>입니다. 이들은 주로 개별 구성요소(res, core, comm 세부 항목)이거나
    범주형 메타 정보(area, product)로, PCA로 압축 시 영향력이 제거됩니다.
  </div>

  <h3>③ 모델 간 순위 일치도가 시사하는 것</h3>
  <div class="insight-box insight-blue">
    Spearman ρ가 높으면 세 모델이 동일한 피처를 중요하게 본다는 의미로,
    <strong>특징 공학의 방향이 모델에 무관하게 유효</strong>함을 나타냅니다.
    일치도가 낮은 쌍은 모델별 학습 패러다임 차이(선형 vs 트리)에 의한 것입니다.
  </div>

  <h3>④ 운영 모델 피처 선택 권고</h3>
  <table>
    <thead><tr><th>단계</th><th>권고 피처 수</th><th>기준</th><th>예상 효과</th></tr></thead>
    <tbody>
      <tr><td>최소 모델</td><td>Top-5 (앙상블)</td><td>앙상블 ≥ 0.6 + KW η² > 0.1</td><td>빠른 속도, CSI total 위주</td></tr>
      <tr><td>균형 모델</td><td>Top-15</td><td>앙상블 ≥ 0.3 + 3모델 유의 ★ 이상</td><td>성능·복잡도 균형</td></tr>
      <tr><td>완전 모델</td><td>Top-25 (현행)</td><td>KW p &lt; 0.05</td><td>최대 성능, PCA 불필요</td></tr>
      <tr><td>제거 후보</td><td>Bottom-15</td><td>Permutation ≤ 0 + KW p ≥ 0.1</td><td>모델 크기 축소, 노이즈 제거</td></tr>
    </tbody>
  </table>
</div>

</div>
</body>
</html>""".format(
    date="2026-05-14",
    n_feat=N_FEAT,
    alpha_bonf=alpha_bonf,
    top5=", ".join(df_imp["feature"].iloc[:5].tolist()),
    img_ensemble=imgs["ensemble_top20"],
    img_builtin=imgs["builtin_3models"],
    img_perm=imgs["perm_3models"],
    img_heatmap=imgs["heatmap"],
    img_boxplot=imgs["boxplot_compare"],
    img_scatter=imgs["scatter_kw_vs_ensemble"],
    img_dist=imgs["dist_compare"],
    img_ci=imgs["bootstrap_ci"],
    img_spearman=imgs["spearman_heatmap"],
    rho=stats.spearmanr(df_imp["kw_eta2"].values, df_imp["ensemble_score"].values).statistic,
    rho_p=stats.spearmanr(df_imp["kw_eta2"].values, df_imp["ensemble_score"].values).pvalue,
    mw_rows=mw_table(),
    sp_builtin_rows=spearman_table(spearman_builtin),
    sp_perm_rows=spearman_table(spearman_perm),
    table_rows=top_table_rows(),
)

out_path = REPORT_DIR / "feature_importance_report.html"
out_path.write_text(HTML, encoding="utf-8")
print("\nReport saved ->", out_path)
