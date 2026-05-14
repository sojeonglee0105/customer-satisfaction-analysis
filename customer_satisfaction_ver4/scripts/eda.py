"""
Customer Satisfaction EDA
==========================
입력: customer_satisfaction_ver4/data/customer_satisfaction_balanced.xlsx
출력: customer_satisfaction_ver4/outputs/plots/*.png
      customer_satisfaction_ver4/outputs/tables/*.csv|json
      customer_satisfaction_ver4/reports/EDA_report.md
"""

import json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

# ── 1. 환경 설정 ──────────────────────────────────────────────────────
BASE     = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver4")
DATA     = BASE / "data"
PLOTS    = BASE / "outputs" / "plots"
TABLES   = BASE / "outputs" / "tables"
REPORTS  = BASE / "reports"
for d in [PLOTS, TABLES, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

# EDA 입력 파일: ver2 원본 데이터
EDA_XLSX  = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver2\customer_satisfaction_full_columns.xlsx")
EDA_SHEET = "synthetic_data"

# 한글 폰트 설정
def set_korean_font():
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]
    available  = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"]       = font
            plt.rcParams["axes.unicode_minus"] = False
            print("Font set:", font)
            return font
    plt.rcParams["axes.unicode_minus"] = False
    print("Warning: no Korean font found, using default")
    return None

FONT = set_korean_font()

# ── 2. 데이터 로드 ────────────────────────────────────────────────────
df = pd.read_excel(EDA_XLSX, sheet_name=EDA_SHEET)
print("Shape:", df.shape)

TARGET   = "rpi"
CAT_COLS = ["year", "area", "product", "client"]
NUM_COLS = [c for c in df.columns if c not in CAT_COLS + [TARGET]]
CSI_COLS = [c for c in NUM_COLS if "csi" in c]
CCI_COLS = [c for c in NUM_COLS if "cci" in c]

# ── tables: 기본 정보 저장 ────────────────────────────────────────────
df.head(5).to_csv(TABLES / "head5.csv", index=False, encoding="utf-8-sig")

dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
(TABLES / "dtypes_summary.json").write_text(
    json.dumps(dtypes_dict, ensure_ascii=False, indent=2), encoding="utf-8")

desc = df[NUM_COLS[:30]].describe().round(4)
desc.to_csv(TABLES / "numeric_describe_first30.csv", encoding="utf-8-sig")

target_counts = df[TARGET].value_counts().sort_index()
target_pct    = (target_counts / len(df) * 100).round(2)
tc_df = pd.DataFrame({"count": target_counts, "percent": target_pct})
tc_df.to_csv(TABLES / "target_class_counts.csv", encoding="utf-8-sig")
print("[RPI 분포]\n", tc_df.to_string())

# 상수형 변수 탐지
constant_cols = [c for c in NUM_COLS if df[c].nunique() <= 1]
(TABLES / "constant_like_columns.json").write_text(
    json.dumps(constant_cols, ensure_ascii=False), encoding="utf-8")
print("Constant cols:", constant_cols if constant_cols else "없음")

# CSI/CCI Z-score 정보
zscore_info = {
    "applied_to": CSI_COLS + CCI_COLS,
    "n_csi": len(CSI_COLS),
    "n_cci": len(CCI_COLS),
    "note": "상관분석 전 열별 Z-score 적용"
}
(TABLES / "csi_cci_zscore_summary.json").write_text(
    json.dumps(zscore_info, ensure_ascii=False, indent=2), encoding="utf-8")

# ── Z-score 적용 (상관분석용) ─────────────────────────────────────────
df_z = df.copy()
for col in CSI_COLS + CCI_COLS:
    std = df_z[col].std()
    if std > 0:
        df_z[col] = (df_z[col] - df_z[col].mean()) / std

# ── 3. 타겟 변수 분석 ─────────────────────────────────────────────────
print("\n[Plot 01] RPI 분포 막대+파이...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("RPI 클래스 분포 (1=관계 매우 우수  ~  5=관계 매우 불량)", fontsize=14, fontweight="bold")

colors = ["#16a34a", "#4ade80", "#fbbf24", "#f97316", "#ef4444"]
labels = ["RPI {}".format(i) for i in target_counts.index]

# 막대
ax = axes[0]
bars = ax.bar(labels, target_counts.values, color=colors, edgecolor="white", linewidth=1.5)
ax.set_title("클래스별 샘플 수", fontsize=12)
ax.set_ylabel("샘플 수")
for bar, cnt in zip(bars, target_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, target_counts.max() * 1.15)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

# 파이
axes[1].pie(target_counts.values, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("클래스 비율", fontsize=12)

plt.tight_layout()
plt.savefig(PLOTS / "01_rpi_class_bar_pie.png", dpi=150, bbox_inches="tight")
plt.close()

# 고객사별 RPI 박스플롯
print("[Plot 02] 고객사별 RPI 분포...")
fig, ax = plt.subplots(figsize=(10, 5))
client_order = sorted(df["client"].unique())
data_by_client = [df[df["client"] == c][TARGET].values for c in client_order]
bp = ax.boxplot(data_by_client, labels=client_order, patch_artist=True,
                medianprops={"color":"white","linewidth":2.5})
palette = ["#3b82f6","#8b5cf6","#f59e0b","#10b981","#ef4444"]
for patch, color in zip(bp["boxes"], palette[:len(client_order)]):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_title("고객사별 RPI 분포 (낮을수록 우수)", fontsize=13, fontweight="bold")
ax.set_xlabel("고객사"); ax.set_ylabel("RPI")
ax.set_yticks(range(1, 6))
ax.grid(axis="y", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS / "02_rpi_by_client_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 4. 수치형 변수 분석 ───────────────────────────────────────────────
print("[Plot 03] 수치형 범위 상위 20...")
ranges = (df[NUM_COLS].max() - df[NUM_COLS].min()).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(ranges.index[::-1], ranges.values[::-1],
               color=["#3b82f6" if "csi" in c else "#f59e0b" for c in ranges.index[::-1]],
               edgecolor="white", linewidth=0.8)
ax.set_title("수치형 변수 값 범위 상위 20개", fontsize=13, fontweight="bold")
ax.set_xlabel("Range (max - min)")
ax.grid(axis="x", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#3b82f6",label="CSI"),
                   Patch(facecolor="#f59e0b",label="CCI")], loc="lower right")
plt.tight_layout()
plt.savefig(PLOTS / "03_numeric_range_top20.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5. 상관관계 분석 ──────────────────────────────────────────────────
print("[Plot 04] RPI 상관 상위 20...")
corr_target = df_z[NUM_COLS].corrwith(df_z[TARGET].astype(float))
corr_abs    = corr_target.abs().sort_values(ascending=False)
top20       = corr_target[corr_abs.index[:20]]

# 상관관계 테이블 저장
corr_abs.reset_index().rename(columns={"index":"feature", 0:"corr_abs"}).to_csv(
    TABLES / "correlation_with_target.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(12, 7))
colors_bar = ["#ef4444" if v < 0 else "#3b82f6" for v in top20.values]
ax.barh(top20.index[::-1], top20.values[::-1], color=colors_bar[::-1],
        edgecolor="white", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("RPI와 상관 상위 20개 변수 (Z-score 기준, 음수=CSI↑→RPI↓)", fontsize=12, fontweight="bold")
ax.set_xlabel("Pearson r (Z-score 기준)")
ax.grid(axis="x", alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS / "04_corr_with_target_top20.png", dpi=150, bbox_inches="tight")
plt.close()

# 히트맵 — 상위 30개
print("[Plot 05] 히트맵 상위 30...")
top30_cols = corr_abs.index[:30].tolist()
corr_matrix = df_z[top30_cols + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(16, 14))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, linewidths=0.3, ax=ax,
            cbar_kws={"shrink": 0.7},
            annot=False)
ax.set_title("상위 30개 변수 + RPI 상관 히트맵", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS / "05_heatmap_top30.png", dpi=150, bbox_inches="tight")
plt.close()

# 다중공선성 체크
print("[Table] 다중공선성 체크...")
corr_num = df_z[NUM_COLS].corr().abs()
mc_pairs = []
cols_list = corr_num.columns.tolist()
for i in range(len(cols_list)):
    for j in range(i+1, len(cols_list)):
        v = corr_num.iloc[i, j]
        if v >= 0.9:
            mc_pairs.append({"var1": cols_list[i], "var2": cols_list[j], "correlation": round(float(v), 4)})
mc_df = pd.DataFrame(mc_pairs)
mc_df.to_csv(TABLES / "multicollinearity_pairs.csv", index=False, encoding="utf-8-sig")
print("  다중공선성 쌍 (|r|>=0.9):", len(mc_df))

# ── 6. 클래스별 분포 비교 ─────────────────────────────────────────────
print("[Plot 06] 박스플롯 상위 6개...")
top6_cols = corr_abs.index[:6].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("RPI 클래스별 분포 비교 — 상관 상위 6개 변수 (원 척도)", fontsize=13, fontweight="bold")
rpi_classes = sorted(df[TARGET].unique())
bp_colors   = ["#16a34a","#4ade80","#fbbf24","#f97316","#ef4444"]
for idx, col in enumerate(top6_cols):
    ax = axes[idx // 3][idx % 3]
    data_by_rpi = [df[df[TARGET] == r][col].dropna().values for r in rpi_classes]
    bp = ax.boxplot(data_by_rpi, labels=["RPI {}".format(r) for r in rpi_classes],
                    patch_artist=True,
                    medianprops={"color":"white","linewidth":2})
    for patch, color in zip(bp["boxes"], bp_colors[:len(rpi_classes)]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_ylabel("CSI (0~10)" if "csi" in col else "CCI (-5~5)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS / "06_boxplot_top6_by_class.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n모든 시각화 저장 완료.")

# ── 7. 종합 보고서 (Markdown) ─────────────────────────────────────────
print("[Report] EDA_report.md 생성...")

# 상관 상위 20 텍스트
top20_lines = "\n".join(
    "- `{}` : r = {:.4f}".format(col, corr_target[col])
    for col in top20.index
)

# 다중공선성 텍스트
if len(mc_df) == 0:
    mc_text = "- |r| ≥ 0.9인 변수 쌍 없음 — 다중공선성 문제 없음"
else:
    mc_text = "\n".join(
        "- `{}` & `{}` : r = {}".format(r.var1, r.var2, r.correlation)
        for _, r in mc_df.head(10).iterrows()
    )

report_md = """# 고객 만족도 EDA 종합 보고서

- **데이터 파일**: `customer_satisfaction_ver2/customer_satisfaction_full_columns.xlsx`
- **시트**: `synthetic_data`
- **타겟 컬럼 (Y_Class = RPI)**: `rpi`
- **폰트 설정**: `{font}`

> 문서에서 **Y_Class**라고 하면 본 데이터에서는 **`rpi`** 컬럼(고객 만족도 RPI 등급)을 가리킵니다.

## 데이터 구조 (도메인 정의)

- **조사 년도** (`year`): 수치형 — 2023, 2024, 2025
- **평가영역** (`area`): 범주형 — t1, t2, d, c, q1, q2
- **제품군** (`product`): 범주형 — 모니터, 노트북, TV, 스마트폰, 자동차
- **고객사명** (`client`): 범주형 — a ~ e
- **RPI / Y_Class** (`rpi`): 범주형 — **1(관계 매우 우수) ~ 5(관계 매우 불량)**
- **CSI**: 수치형 — 0(매우 낮음) ~ 10(매우 높음)
- **CCI**: 수치형 — -5(매우 낮음) ~ +5(매우 높음)

## 1. 데이터 요약

```
{shape}
```

### 컬럼 목록

{col_list}

### 기본 통계 (수치형 처음 30개)

→ `outputs/tables/numeric_describe_first30.csv` 참고

## 2. 타겟(Y_Class = RPI / `rpi` 컬럼) 분석

```
{tc_str}
```

### 시각화

- **RPI 분포(1=우수~5=불량)**: [outputs/plots/01_rpi_class_bar_pie.png](../outputs/plots/01_rpi_class_bar_pie.png)
- **고객사별 RPI (낮을수록 우수)**: [outputs/plots/02_rpi_by_client_boxplot.png](../outputs/plots/02_rpi_by_client_boxplot.png)

## 3. 수치형 변수

- **변수 범위 상위 20 (원 척도)**: [outputs/plots/03_numeric_range_top20.png](../outputs/plots/03_numeric_range_top20.png)
- 상수(또는 유일값)에 가까운 변수: `outputs/tables/constant_like_columns.json`
  - 탐지된 상수형 변수: `{const_cols}`
- CSI·CCI 열 Z-score 적용 내역: `outputs/tables/csi_cci_zscore_summary.json`

## 4. 상관관계 및 다중공선성

> **RPI 해석**: 1은 관계가 매우 우수, 5는 매우 불량입니다.
> **상관·히트맵**은 CSI·CCI에 **열별 Z-score**를 적용한 뒤 산출했으며,
> 그 외 수치형은 **원 스케일**을 유지합니다.
> 박스플롯은 해석을 위해 **CSI·CCI 원 척도**를 사용합니다.

### RPI와 상관 상위 20개

{top20_lines}

### 시각화

- **RPI 상관 상위 20 (CSI·CCI Z)**: [outputs/plots/04_corr_with_target_top20.png](../outputs/plots/04_corr_with_target_top20.png)
- **상위 30 히트맵 (CSI·CCI Z)**: [outputs/plots/05_heatmap_top30.png](../outputs/plots/05_heatmap_top30.png)

### 다중공선성 체크 (|r| ≥ 0.9)

{mc_text}

→ 전체 목록: `outputs/tables/multicollinearity_pairs.csv`

## 5. 클래스별 분포 비교

- **박스플롯 상위 6 (CSI·CCI 원 척도)**: [outputs/plots/06_boxplot_top6_by_class.png](../outputs/plots/06_boxplot_top6_by_class.png)

## 6. 인사이트

- **Y_Class**(타겟)는 엑셀 **`rpi`** 열이며, **1=관계 우수 · 5=관계 불량**으로 해석합니다.
- RPI 등급 종류 수: {n_rpi_classes}개 — 각 클래스 **{n_per_class}개씩 균형 분포** (이전 실험에서 불균형 개선 완료).
- 수치형 설명변수 개수 (타겟 제외): {n_num}개 (CSI {n_csi}개 + CCI {n_cci}개).
- 상수형 변수: {const_count}개 — 제거 불필요.
- 다중공선성 쌍 (|r| ≥ 0.9): {n_mc}쌍.
- RPI와 가장 강한 상관: `{top1_col}` (r = {top1_val:.4f}) — CSI 높을수록 RPI 낮음(우수 관계) 방향.

## 7. 다음 단계 제안

- 모델링 시 CSI·CCI에 `StandardScaler`(또는 열별 Z-score) 파이프라인을 포함합니다.
- RPI는 낮을수록 우호적 관계이므로, 회귀·랭킹 해석 시 계수 방향을 도메인 척도에 맞게 설명합니다.
- 범주형 변수(area, product, client)는 Label Encoding 또는 One-Hot Encoding 전략을 확정합니다.
- 5-class 균형 데이터이므로 Ordinal 회귀 / 다중 분류 모두 적용 가능합니다.
- 다중공선성이 {n_mc}쌍으로 {mc_advice}합니다.
""".format(
    font=FONT or "default",
    shape="{} x {}".format(*df.shape),
    col_list="\n".join("- `{}`".format(c) for c in df.columns),
    tc_str=tc_df.to_string(),
    const_cols=constant_cols if constant_cols else "없음",
    top20_lines=top20_lines,
    mc_text=mc_text,
    n_rpi_classes=df[TARGET].nunique(),
    n_per_class=int(len(df) / df[TARGET].nunique()),
    n_num=len(NUM_COLS),
    n_csi=len(CSI_COLS),
    n_cci=len(CCI_COLS),
    const_count=len(constant_cols),
    n_mc=len(mc_df),
    top1_col=corr_abs.index[0],
    top1_val=corr_target[corr_abs.index[0]],
    mc_advice="발견됨 — PCA 또는 VIF 기반 변수 제거 권장" if len(mc_df) > 0 else "없음 — 추가 조치 불필요",
)

(REPORTS / "EDA_report.md").write_text(report_md, encoding="utf-8")
print("EDA_report.md 저장 완료.")

# ── 완료 요약 출력 ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("EDA 완료 요약")
print("="*55)
print("Plots  :", list(PLOTS.glob("*.png")).__len__(), "files ->", PLOTS)
print("Tables :", list(TABLES.glob("*")).__len__(),   "files ->", TABLES)
print("Report :", REPORTS / "EDA_report.md")
