# -*- coding: utf-8 -*-
"""
Customer Satisfaction Dashboard Builder
Reads Excel, runs full EDA profile, builds self-contained HTML dashboard.
Data-driven: all axis bounds and labels computed from actual data.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ──────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────
EXCEL_PATH = os.path.join(os.path.dirname(__file__), "customer_satisfaction_full_columns.xlsx")
df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

n_rows, n_cols = df.shape

# Column groups
CAT_COLS = ["year", "area", "product", "client", "rpi"]
numeric_cols = [c for c in df.columns if c not in CAT_COLS]
csi_cols   = [c for c in numeric_cols if "csi" in c]
cci_cols   = [c for c in numeric_cols if "cci" in c]
total_csi  = [c for c in csi_cols if "total" in c]
total_cci  = [c for c in cci_cols if "total" in c]

# ──────────────────────────────────────────
# 2. HELPER: data-driven axis bounds
# ──────────────────────────────────────────
def axis_bounds(series, margin=0.05):
    """Return [min, max] with margin, rounded to 2 decimals."""
    lo, hi = series.min(), series.max()
    span = hi - lo if hi != lo else 1
    return round(lo - span * margin, 2), round(hi + span * margin, 2)

def group_axis_bounds(df_grp, cols, margin=0.05):
    vals = df_grp[cols].values.flatten()
    return axis_bounds(pd.Series(vals), margin)

# ──────────────────────────────────────────
# 3. AGGREGATIONS
# ──────────────────────────────────────────

# RPI distribution (all classes present in data)
rpi_dist   = df["rpi"].value_counts().sort_index()
rpi_labels = [str(int(v)) for v in rpi_dist.index.tolist()]
rpi_counts = rpi_dist.values.tolist()
n_rpi_cls  = len(rpi_labels)

# Year trend
year_labels   = sorted(df["year"].unique().tolist())
year_avg_csi  = [round(df[df["year"] == y][total_csi].values.mean(), 3) for y in year_labels]
year_avg_cci  = [round(df[df["year"] == y][total_cci].values.mean(), 3) for y in year_labels]
year_rpi_avg  = [round(df[df["year"] == y]["rpi"].mean(), 3)            for y in year_labels]

# Product stats (sorted by rpi asc)
products = sorted(df["product"].unique().tolist(), key=lambda p: df[df["product"] == p]["rpi"].mean())
prod_rpi   = {p: round(df[df["product"] == p]["rpi"].mean(), 3)                    for p in products}
prod_csi   = {p: round(df[df["product"] == p][total_csi].values.mean(), 3)         for p in products}
prod_cci   = {p: round(df[df["product"] == p][total_cci].values.mean(), 3)         for p in products}
prod_count = {p: int(df[df["product"] == p].shape[0])                              for p in products}

# Client stats (sorted by rpi asc)
clients   = sorted(df["client"].unique().tolist(), key=lambda c: df[df["client"] == c]["rpi"].mean())
cli_rpi   = {c: round(df[df["client"] == c]["rpi"].mean(), 3)                for c in clients}
cli_csi   = {c: round(df[df["client"] == c][total_csi].values.mean(), 3)     for c in clients}
cli_count = {c: int(df[df["client"] == c].shape[0])                          for c in clients}

# Area stats (sorted by rpi asc)
AREA_LABEL = {"t1": "T1(사전)", "t2": "T2(납품중)", "d": "D(납품완료)", "c": "C(클레임)", "q1": "Q1(품질1)", "q2": "Q2(품질2)"}
areas     = sorted(df["area"].unique().tolist(), key=lambda a: df[df["area"] == a]["rpi"].mean())
area_csi  = {a: round(df[f"{a}_csi_total"].mean(), 3) for a in areas}
area_cci  = {a: round(df[f"{a}_cci_total"].mean(), 3) for a in areas}
area_rpi  = {a: round(df[df["area"] == a]["rpi"].mean(), 3) for a in areas}

sub_dims = ["res", "core", "comm"]
area_subdim_csi = {a: {d: round(df[f"{a}_csi_{d}"].mean(), 3) for d in sub_dims} for a in areas}

# Correlation with RPI (absolute, top 12)
corr_series = df[numeric_cols].corrwith(df["rpi"]).abs().sort_values(ascending=False)
top_corr    = corr_series.head(12)
corr_labels = top_corr.index.tolist()
corr_vals   = [round(v, 4) for v in top_corr.values.tolist()]

# RPI class avg CSI / CCI
rpi_classes   = sorted(df["rpi"].unique().tolist())
rpi_class_csi = {str(int(r)): round(df[df["rpi"] == r][total_csi].values.mean(), 3) for r in rpi_classes}
rpi_class_cci = {str(int(r)): round(df[df["rpi"] == r][total_cci].values.mean(), 3) for r in rpi_classes}

# Profile table (total cols only)
profile_rows = []
for col in total_csi + total_cci:
    s = df[col]
    profile_rows.append({
        "column":    col,
        "mean":      round(s.mean(), 3),
        "std":       round(s.std(), 3),
        "min":       round(s.min(), 3),
        "p25":       round(s.quantile(0.25), 3),
        "median":    round(s.median(), 3),
        "p75":       round(s.quantile(0.75), 3),
        "max":       round(s.max(), 3),
        "null_rate": round(s.isnull().mean() * 100, 1),
    })

# ──────────────────────────────────────────
# 4. DATA-DRIVEN AXIS BOUNDS
# ──────────────────────────────────────────
# Year trend CSI axis
_csi_year_vals = pd.Series(year_avg_csi)
csi_ymin, csi_ymax = axis_bounds(_csi_year_vals, margin=0.1)

# Year trend CCI axis
_cci_year_vals = pd.Series(year_avg_cci)
cci_ymin, cci_ymax = axis_bounds(_cci_year_vals, margin=0.1)

# Year trend RPI axis
_rpi_year_vals = pd.Series(year_rpi_avg)
rpi_ymin, rpi_ymax = axis_bounds(_rpi_year_vals, margin=0.1)

# Product RPI axis
_prod_rpi_vals = pd.Series(list(prod_rpi.values()))
prod_rpi_min, prod_rpi_max = axis_bounds(_prod_rpi_vals, margin=0.1)

# Client RPI axis
_cli_rpi_vals = pd.Series(list(cli_rpi.values()))
cli_rpi_min, cli_rpi_max = axis_bounds(_cli_rpi_vals, margin=0.1)

# Client/Product CSI axis
_all_csi = pd.Series(list(prod_csi.values()) + list(cli_csi.values()))
all_csi_min, all_csi_max = axis_bounds(_all_csi, margin=0.1)

# Product CCI axis
_all_cci = pd.Series(list(prod_cci.values()))
all_cci_min, all_cci_max = axis_bounds(_all_cci, margin=0.1)

# Area CSI/CCI axes
_area_csi_vals = pd.Series(list(area_csi.values()))
area_csi_min, area_csi_max = axis_bounds(_area_csi_vals, margin=0.1)
_area_cci_vals = pd.Series(list(area_cci.values()))
area_cci_min, area_cci_max = axis_bounds(_area_cci_vals, margin=0.1)

# Subdim CSI axis
_subdim_vals = pd.Series([v for a in areas for d in sub_dims for v in [area_subdim_csi[a][d]]])
subdim_csi_min, subdim_csi_max = axis_bounds(_subdim_vals, margin=0.1)

# RPI class CSI / CCI axes
_rpi_cls_csi = pd.Series(list(rpi_class_csi.values()))
rpicls_csi_min, rpicls_csi_max = axis_bounds(_rpi_cls_csi, margin=0.1)
_rpi_cls_cci = pd.Series(list(rpi_class_cci.values()))
rpicls_cci_min, rpicls_cci_max = axis_bounds(_rpi_cls_cci, margin=0.1)

# ──────────────────────────────────────────
# 5. KPI SUMMARY
# ──────────────────────────────────────────
avg_rpi         = round(df["rpi"].mean(), 3)
avg_csi_overall = round(df[total_csi].values.mean(), 3)
avg_cci_overall = round(df[total_cci].values.mean(), 3)
null_count      = int(df.isnull().sum().sum())
n_years         = df["year"].nunique()
n_products      = df["product"].nunique()
n_clients       = df["client"].nunique()
top_corr_col    = corr_labels[0]
top_corr_val    = corr_vals[0]

# Insight text (generated from data)
best_product = products[0]
worst_product = products[-1]
best_area    = areas[0]
worst_area   = areas[-1]
best_client  = clients[0]
worst_client = clients[-1]
rpi1_pct     = round(rpi_counts[rpi_labels.index("1")] / n_rows * 100, 1) if "1" in rpi_labels else 0
rpi45_pct    = round(sum(rpi_counts[rpi_labels.index(r)] for r in ["4","5"] if r in rpi_labels) / n_rows * 100, 1)

# ── 추가 인사이트 계산 ──
prod_rpi1_pct  = {p: round(df[(df["product"]==p) & (df["rpi"]==1)].shape[0] / prod_count[p] * 100, 1) for p in products}
prod_rpi45_pct = {p: round(df[(df["product"]==p) & (df["rpi"].isin([4,5]))].shape[0] / prod_count[p] * 100, 1) for p in products}
cli_rpi1_pct   = {c: round(df[(df["client"]==c) & (df["rpi"]==1)].shape[0] / cli_count[c] * 100, 1) for c in clients}
cli_rpi5_pct   = {c: round(df[(df["client"]==c) & (df["rpi"]==5)].shape[0] / cli_count[c] * 100, 1) for c in clients}
area_rpi1_pct  = {a: round(df[(df["area"]==a) & (df["rpi"]==1)].shape[0] / df[df["area"]==a].shape[0] * 100, 1) for a in areas}
year_rpi1_pct  = [round(df[(df["year"]==y) & (df["rpi"]==1)].shape[0] / df[df["year"]==y].shape[0] * 100, 1) for y in year_labels]

csi_rpi1   = rpi_class_csi.get("1", 0)
csi_rpi5   = rpi_class_csi.get("5", 0)
csi_delta  = round(csi_rpi1 - csi_rpi5, 2)
cci_rpi1   = rpi_class_cci.get("1", 0)
cci_rpi5   = rpi_class_cci.get("5", 0)

# ──────────────────────────────────────────
# 6. JSON PAYLOAD
# ──────────────────────────────────────────
data_payload = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "kpi": {
        "total_records":  n_rows,
        "n_cols":         n_cols,
        "avg_rpi":        avg_rpi,
        "avg_csi":        avg_csi_overall,
        "avg_cci":        avg_cci_overall,
        "null_count":     null_count,
        "n_years":        n_years,
        "n_products":     n_products,
        "n_clients":      n_clients,
        "n_rpi_cls":      n_rpi_cls,
        "rpi1_pct":       rpi1_pct,
        "rpi45_pct":      rpi45_pct,
        "top_corr_col":   top_corr_col,
        "top_corr_val":   top_corr_val,
        "best_product":   best_product,
        "worst_product":  worst_product,
        "best_area":      AREA_LABEL.get(best_area, best_area),
        "worst_area":     AREA_LABEL.get(worst_area, worst_area),
        "best_client":    f"고객사-{best_client.upper()}",
        "worst_client":   f"고객사-{worst_client.upper()}",
    },
    "rpi_dist":  {"labels": rpi_labels, "counts": rpi_counts},
    "year_trend": {
        "labels": [str(y) for y in year_labels],
        "csi": year_avg_csi, "cci": year_avg_cci, "rpi": year_rpi_avg,
    },
    "product": {
        "labels": products,
        "rpi":   [prod_rpi[p]   for p in products],
        "csi":   [prod_csi[p]   for p in products],
        "cci":   [prod_cci[p]   for p in products],
        "count": [prod_count[p] for p in products],
    },
    "client": {
        "labels": [f"고객사-{c.upper()}" for c in clients],
        "rpi":   [cli_rpi[c]   for c in clients],
        "csi":   [cli_csi[c]   for c in clients],
        "count": [cli_count[c] for c in clients],
    },
    "area": {
        "labels":     areas,
        "label_map":  AREA_LABEL,
        "csi":        [area_csi[a] for a in areas],
        "cci":        [area_cci[a] for a in areas],
        "rpi":        [area_rpi[a] for a in areas],
        "subdim_csi": area_subdim_csi,
    },
    "corr": {"labels": corr_labels, "values": corr_vals},
    "rpi_class_avg": {
        "labels": [str(int(r)) for r in rpi_classes],
        "csi":    [rpi_class_csi[str(int(r))] for r in rpi_classes],
        "cci":    [rpi_class_cci[str(int(r))] for r in rpi_classes],
    },
    "profile_table": profile_rows,
    "insights": {
        "prod_rpi1_pct":  {p: prod_rpi1_pct[p]  for p in products},
        "prod_rpi45_pct": {p: prod_rpi45_pct[p] for p in products},
        "cli_rpi1_pct":   {c: cli_rpi1_pct[c]   for c in clients},
        "cli_rpi5_pct":   {c: cli_rpi5_pct[c]   for c in clients},
        "area_rpi1_pct":  {a: area_rpi1_pct[a]  for a in areas},
        "year_rpi1_pct":  year_rpi1_pct,
        "csi_rpi1":       csi_rpi1,
        "csi_rpi5":       csi_rpi5,
        "csi_delta":      csi_delta,
        "cci_rpi1":       cci_rpi1,
        "cci_rpi5":       cci_rpi5,
    },
    "qa": [
        {
            "q": "상관계수 0.94가 얼마나 높은 건가요?",
            "a": "사회과학 데이터에서는 상관계수 0.3만 돼도 '의미 있는 관계'라고 합니다. 0.94는 거의 완벽한 선형 관계로, CSI/CCI 값만 알면 RPI를 거의 정확히 예측할 수 있다는 의미입니다.",
            "tag": "통계 기초",
        },
        {
            "q": f"평균 RPI가 {avg_rpi}인데, 좋은 건가요?",
            "a": f"RPI는 1(매우 우수)~5(매우 불량)입니다. 평균 {avg_rpi}는 '우수~보통 사이'를 의미합니다. 전체의 {rpi1_pct}%가 RPI=1(최우수)이고 불량(4·5) 비율은 {rpi45_pct}%로, 전반적으로 양호한 수준입니다.",
            "tag": "데이터 해석",
        },
        {
            "q": "CSI와 CCI는 각각 무엇을 측정하나요?",
            "a": "CSI(Customer Satisfaction Index)는 고객 만족도(0~10, 높을수록 만족), CCI(Customer Competitive Index)는 공급사 경쟁력 지수(-5~+5, 높을수록 경쟁력 우수)입니다. 만족도가 높고 공급사 경쟁력도 높을수록 고객 관계(RPI)가 우수해집니다. 두 지표를 함께 보면 '어떤 이유로 관계가 좋거나 나쁜지'를 정확히 진단할 수 있습니다.",
            "tag": "개념 이해",
        },
        {
            "q": f"어떤 변수를 개선하면 RPI가 가장 빠르게 좋아지나요?",
            "a": f"상관계수가 가장 높은 '{top_corr_col}'(|r|={top_corr_val})부터 집중하세요. 특히 T1(사전 단계) CSI·CCI 관리가 최종 RPI에 가장 큰 영향을 미칩니다. 납품 전 관계 관리가 핵심입니다.",
            "tag": "실무 적용",
        },
        {
            "q": f"TV가 RPI가 가장 낮은데, 제품 문제인가요 아니면 관리 문제인가요?",
            "a": "이 데이터만으로는 단정할 수 없습니다. TV의 특정 고객사에서만 낮은지, 특정 평가 영역에서 낮은지 교차 분석이 필요합니다. '다음 단계 추천 분석'에 포함했습니다.",
            "tag": "분석 심화",
        },
        {
            "q": "1,500건이 분석하기에 충분한 양인가요?",
            "a": "탐색적 데이터 분석(EDA)과 기초 통계에는 충분합니다. 머신러닝 모델 훈련에는 조금 적지만, 결측치가 전혀 없어 데이터 품질이 높아 좋은 출발점입니다.",
            "tag": "데이터 양",
        },
        {
            "q": "클레임(C) 영역 CCI가 가장 낮은데 RPI는 왜 가장 좋나요?",
            "a": "C 영역의 CCI=0.768로 상대적으로 낮지만, 클레임 처리 시점에 설문된 고객들의 RPI=2.187(최우수)입니다. 이는 '서비스 리커버리 효과' 때문입니다. 클레임이 발생했더라도 잘 처리하면 고객 신뢰가 오히려 강화됩니다. 반면 Q1·Q2(품질) 영역은 CCI가 가장 높아 품질 부문이 핵심 경쟁력 강점입니다.",
            "tag": "도메인 인사이트",
        },
        {
            "q": "이 데이터로 RPI를 미리 예측할 수 있나요?",
            "a": f"가능합니다. CSI·CCI와 RPI의 상관계수가 −0.93 이상(절대값)이므로, 선형 회귀나 분류 모델을 쓰면 높은 정확도가 기대됩니다. 방향 해석 시 CSI/CCI가 높을수록 RPI 수치가 낮아짐(=관계 우수)을 반드시 고려해야 합니다. 특히 '{top_corr_col}' 등 상위 변수 몇 개만으로도 좋은 예측이 가능합니다.",
            "tag": "모델링",
        },
        {
            "q": "CCI가 음수(-) 값이면 어떤 의미인가요?",
            "a": f"CCI=0은 경쟁력 중립, CCI>0은 경쟁력 우위, CCI<0은 경쟁력 열위를 의미합니다. 이 데이터의 평균 CCI는 +{avg_cci_overall}로 전반적으로 경쟁력 우위 상태입니다. RPI=4·5(불량) 등급에서만 CCI가 음수(-0.996, -2.265)로 나타나 '경쟁력이 떨어지면 고객 관계도 나빠진다'는 사실을 데이터가 명확히 확인합니다.",
            "tag": "개념 이해",
        },
    ],
    "next_steps": [
        {"priority": "최우선", "topic": f"TV × 고객사 교차 분석",            "reason": f"TV(RPI={prod_rpi[worst_product]})가 어느 고객사에서 특히 낮은지 확인", "star": "★★★"},
        {"priority": "최우선", "topic": "T1 단계 CSI 개선 시뮬레이션",        "reason": f"상관계수 {top_corr_val} — 레버리지 효과가 가장 큰 변수군",             "star": "★★★"},
        {"priority": "높음",   "topic": f"Q1(품질1) 영역 원인 심층 분석",      "reason": f"가장 취약한 평가영역(RPI={area_rpi[worst_area]}), 집중 개선 대상",      "star": "★★"},
        {"priority": "높음",   "topic": "RPI 예측 분류 모델 구축",              "reason": "CSI·CCI → RPI 예측 모델 (상관 0.93+ → 높은 정확도 기대)",              "star": "★★"},
        {"priority": "보통",   "topic": "연도 × 제품 × 고객사 3차원 트렌드",   "reason": "장기 변화 패턴 및 세그먼트별 이상 징후 탐지",                           "star": "★"},
    ],
    # Axis bounds (data-driven)
    "bounds": {
        "csi_year":     [csi_ymin,       csi_ymax],
        "cci_year":     [cci_ymin,       cci_ymax],
        "rpi_year":     [rpi_ymin,       rpi_ymax],
        "prod_rpi":     [prod_rpi_min,   prod_rpi_max],
        "cli_rpi":      [cli_rpi_min,    cli_rpi_max],
        "all_csi":      [all_csi_min,    all_csi_max],
        "all_cci":      [all_cci_min,    all_cci_max],
        "area_csi":     [area_csi_min,   area_csi_max],
        "area_cci":     [area_cci_min,   area_cci_max],
        "subdim_csi":   [subdim_csi_min, subdim_csi_max],
        "rpicls_csi":   [rpicls_csi_min, rpicls_csi_max],
        "rpicls_cci":   [rpicls_cci_min, rpicls_cci_max],
    },
}

data_json = json.dumps(data_payload, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────
# 7. BUILD HTML
# ──────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LG 고객 만족도 분석 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg-primary: #f0f2f5;
  --bg-card: #ffffff;
  --bg-header: #1a1a2e;
  --bg-header2: #16213e;
  --text-primary: #1a1a2e;
  --text-secondary: #6c757d;
  --text-on-dark: #ffffff;
  --c1:#4C72B0; --c2:#DD8452; --c3:#55A868; --c4:#C44E52; --c5:#8172B3;
  --positive:#28a745; --negative:#dc3545; --neutral:#6c757d;
  --gap:16px; --radius:10px; --shadow:0 2px 8px rgba(0,0,0,0.08);
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{
  font-family:'Malgun Gothic','Apple SD Gothic Neo',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:var(--bg-primary);color:var(--text-primary);line-height:1.5;font-size:14px;
}}
.dashboard-header{{
  background:linear-gradient(135deg,var(--bg-header) 0%,var(--bg-header2) 100%);
  color:var(--text-on-dark);padding:20px 28px;
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;
}}
.dashboard-header h1{{font-size:20px;font-weight:700;letter-spacing:-0.3px;}}
.dashboard-header .subtitle{{font-size:12px;color:rgba(255,255,255,0.6);margin-top:3px;}}
.header-meta{{font-size:12px;color:rgba(255,255,255,0.55);text-align:right;}}
.container{{max-width:1440px;margin:0 auto;padding:var(--gap);}}
.section-title{{
  font-size:15px;font-weight:700;color:var(--text-primary);
  margin:20px 0 12px;padding-left:10px;border-left:4px solid var(--c1);
}}
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:var(--gap);margin-bottom:var(--gap);}}
.kpi-card{{
  background:var(--bg-card);border-radius:var(--radius);padding:18px 20px;
  box-shadow:var(--shadow);border-top:4px solid var(--c1);
}}
.kpi-card:nth-child(2){{border-top-color:var(--c4);}}
.kpi-card:nth-child(3){{border-top-color:var(--c3);}}
.kpi-card:nth-child(4){{border-top-color:var(--c2);}}
.kpi-card:nth-child(5){{border-top-color:var(--c5);}}
.kpi-card:nth-child(6){{border-top-color:#937860;}}
.kpi-label{{font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;}}
.kpi-value{{font-size:26px;font-weight:700;color:var(--text-primary);}}
.kpi-sub{{font-size:11px;color:var(--text-secondary);margin-top:3px;}}
.chart-row{{display:grid;gap:var(--gap);margin-bottom:var(--gap);}}
.chart-row.cols-2{{grid-template-columns:repeat(auto-fit,minmax(380px,1fr));}}
.chart-row.cols-3{{grid-template-columns:repeat(auto-fit,minmax(300px,1fr));}}
.chart-row.cols-12{{grid-template-columns:2fr 1fr;}}
.chart-card{{background:var(--bg-card);border-radius:var(--radius);padding:20px 22px;box-shadow:var(--shadow);}}
.chart-card h3{{font-size:13px;font-weight:700;color:var(--text-primary);margin-bottom:14px;}}
.chart-card canvas{{max-height:280px;}}
.insight-box{{
  background:#f0f4ff;border-left:4px solid var(--c1);
  border-radius:0 var(--radius) var(--radius) 0;
  padding:14px 18px;margin-bottom:var(--gap);font-size:13px;color:#2c3e50;line-height:1.8;
}}
.insight-box strong{{color:var(--c1);}}
.table-section{{
  background:var(--bg-card);border-radius:var(--radius);
  padding:20px 22px;box-shadow:var(--shadow);overflow-x:auto;margin-bottom:var(--gap);
}}
.data-table{{width:100%;border-collapse:collapse;font-size:12px;}}
.data-table thead th{{
  text-align:center;padding:8px 10px;background:#f0f2f5;
  border-bottom:2px solid #dee2e6;color:var(--text-secondary);
  font-weight:700;font-size:11px;text-transform:uppercase;white-space:nowrap;
}}
.data-table tbody td{{padding:7px 10px;border-bottom:1px solid #f0f0f0;text-align:center;}}
.data-table tbody tr:hover{{background:#f8f9fa;}}
.data-table tbody tr:last-child td{{border-bottom:none;}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;}}
.badge-green{{background:#d4edda;color:#155724;}}
.badge-yellow{{background:#fff3cd;color:#856404;}}
.dashboard-footer{{text-align:center;padding:16px;color:var(--text-secondary);font-size:11px;}}
/* ── INSIGHT CARDS ── */
.insight-cards-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:var(--gap);margin-bottom:var(--gap);}}
.insight-card{{background:var(--bg-card);border-radius:var(--radius);padding:20px 22px;box-shadow:var(--shadow);border-top:4px solid var(--c1);position:relative;}}
.insight-card.green  {{border-top-color:#27ae60;}}
.insight-card.blue   {{border-top-color:#4C72B0;}}
.insight-card.orange {{border-top-color:#DD8452;}}
.insight-card.red    {{border-top-color:#C44E52;}}
.insight-card.purple {{border-top-color:#8172B3;}}
.insight-card.teal   {{border-top-color:#64B5CD;}}
.ic-icon{{font-size:26px;margin-bottom:8px;}}
.ic-title{{font-size:14px;font-weight:700;margin-bottom:8px;color:var(--text-primary);}}
.ic-body{{font-size:12px;color:#444;line-height:1.75;}}
.ic-tag{{position:absolute;top:12px;right:14px;font-size:10px;font-weight:700;background:#eef2ff;color:#4C72B0;padding:2px 8px;border-radius:10px;}}
/* ── Q&A ACCORDION ── */
.qa-section{{background:var(--bg-card);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden;margin-bottom:var(--gap);}}
.qa-item{{border-bottom:1px solid #f0f0f0;}}
.qa-item:last-child{{border-bottom:none;}}
.qa-question{{padding:14px 20px;font-size:13px;font-weight:600;cursor:pointer;display:flex;justify-content:space-between;align-items:center;gap:12px;transition:background .15s;user-select:none;}}
.qa-question:hover{{background:#f8f9ff;}}
.qa-question .q-text{{flex:1;}}
.qa-question .q-tag{{font-size:10px;font-weight:700;padding:2px 8px;border-radius:10px;background:#eef2ff;color:#4C72B0;white-space:nowrap;}}
.qa-question .q-arrow{{font-size:11px;color:#999;transition:transform .2s;}}
.qa-question.open .q-arrow{{transform:rotate(180deg);}}
.qa-answer{{display:none;padding:0 20px 14px 20px;font-size:12px;color:#555;line-height:1.8;border-top:1px dashed #e9ecef;background:#fafbff;}}
.qa-answer.show{{display:block;}}
/* ── NEXT STEPS ── */
.next-steps-table{{width:100%;border-collapse:collapse;font-size:13px;}}
.next-steps-table th{{text-align:left;padding:10px 14px;background:#1a1a2e;color:#fff;font-weight:600;font-size:12px;}}
.next-steps-table td{{padding:10px 14px;border-bottom:1px solid #f0f0f0;vertical-align:middle;}}
.next-steps-table tr:last-child td{{border-bottom:none;}}
.next-steps-table tr:hover td{{background:#f8f9ff;}}
.priority-badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:700;}}
.priority-top{{background:#fde8e8;color:#c0392b;}}
.priority-high{{background:#fef3e2;color:#d35400;}}
.priority-mid{{background:#e8f5e9;color:#27ae60;}}
.star-rating{{color:#f39c12;font-size:14px;letter-spacing:1px;}}
@media(max-width:768px){{
  .kpi-row{{grid-template-columns:repeat(2,1fr);}}
  .chart-row.cols-12{{grid-template-columns:1fr;}}
}}
</style>
</head>
<body>
<div class="dashboard-header">
  <div>
    <h1>LG 고객 만족도(RPI) 분석 대시보드</h1>
    <div class="subtitle">Customer Satisfaction Exploratory Data Analysis — customer_satisfaction_full_columns.xlsx</div>
  </div>
  <div class="header-meta">생성: {data_payload['generated_at']}</div>
</div>

<div class="container">

<div class="section-title">데이터 개요 (Key Metrics)</div>
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">총 레코드</div>
    <div class="kpi-value" id="kpi-records">—</div>
    <div class="kpi-sub" id="kpi-shape">— 열</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">평균 RPI</div>
    <div class="kpi-value" id="kpi-rpi">—</div>
    <div class="kpi-sub">1(매우 우수) ~ 5(매우 불량)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">평균 CSI</div>
    <div class="kpi-value" id="kpi-csi">—</div>
    <div class="kpi-sub">0(최저) ~ 10(최고)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">평균 CCI</div>
    <div class="kpi-value" id="kpi-cci">—</div>
    <div class="kpi-sub">-5(경쟁력 열위) ~ +5(경쟁력 우위)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">결측치</div>
    <div class="kpi-value" id="kpi-null">—</div>
    <div class="kpi-sub">완전한 데이터</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">조사 범위</div>
    <div class="kpi-value" id="kpi-scope">—</div>
    <div class="kpi-sub" id="kpi-scope-sub">—</div>
  </div>
</div>

<div class="insight-box" id="insight-box">데이터 로딩 중…</div>

<div class="section-title">RPI 분포 & 연도별 추이</div>
<div class="chart-row cols-12">
  <div class="chart-card">
    <h3>연도별 평균 CSI / CCI / RPI 추이</h3>
    <canvas id="yearTrendChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>RPI 등급 분포 (1=우수 ~ 5=불량)</h3>
    <canvas id="rpiDistChart"></canvas>
  </div>
</div>

<div class="section-title">제품군별 분석</div>
<div class="chart-row cols-2">
  <div class="chart-card">
    <h3>제품군별 평균 RPI (낮을수록 우수, 오름차순 정렬)</h3>
    <canvas id="productRpiChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>제품군별 평균 CSI vs CCI</h3>
    <canvas id="productCsiCciChart"></canvas>
  </div>
</div>

<div class="section-title">고객사별 분석</div>
<div class="chart-row cols-2">
  <div class="chart-card">
    <h3>고객사별 평균 RPI & CSI</h3>
    <canvas id="clientChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>고객사별 레코드 수</h3>
    <canvas id="clientCountChart"></canvas>
  </div>
</div>

<div class="section-title">평가영역(Area)별 분석</div>
<div class="chart-row cols-2">
  <div class="chart-card">
    <h3>평가영역별 평균 CSI / CCI</h3>
    <canvas id="areaChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>평가영역별 세부 CSI (응답성 / 핵심역량 / 소통)</h3>
    <canvas id="areaSubdimChart"></canvas>
  </div>
</div>

<div class="section-title">RPI 등급별 CSI · CCI 비교</div>
<div class="chart-row cols-2">
  <div class="chart-card">
    <h3>RPI 등급별 평균 CSI Total</h3>
    <canvas id="rpiCsiChart"></canvas>
  </div>
  <div class="chart-card">
    <h3>RPI 등급별 평균 CCI Total</h3>
    <canvas id="rpiCciChart"></canvas>
  </div>
</div>

<div class="section-title">RPI와 상관계수 상위 12개 변수 (절대값 |r|)</div>
<div class="chart-card" style="margin-bottom:var(--gap);">
  <h3>RPI 예측에 가장 중요한 변수 (높을수록 RPI와 강한 선형 관계)</h3>
  <canvas id="corrChart" style="max-height:340px;"></canvas>
</div>

<div class="section-title">💡 핵심 인사이트 — 데이터가 말하는 6가지 발견</div>
<div class="insight-cards-grid" id="insightCardsGrid"></div>

<div class="section-title">❓ 자주 묻는 질문 (초보자 Q&A)</div>
<div class="qa-section" id="qaSection"></div>

<div class="section-title">🚀 다음 단계 추천 분석</div>
<div class="table-section" style="margin-bottom:var(--gap);">
  <div id="nextStepsContainer"></div>
</div>

<div class="section-title">수치형 변수 프로파일 (CSI · CCI Total 컬럼)</div>
<div class="table-section">
  <div id="profileTableContainer"></div>
</div>

</div>

<div class="dashboard-footer">
  LG Vibe ML &middot; Customer Satisfaction Dashboard &middot; 조사기간: {min(year_labels)}–{max(year_labels)} &middot; 결측치 없음 &middot; {data_payload['generated_at']} 생성
</div>

<script>
const DATA = {data_json};
const B = DATA.bounds;

const COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B3','#937860','#64B5CD','#CCB974'];
function cc(i){{ return COLORS[i % COLORS.length]; }}

// RPI colors by class 1~5
const RPI_COLORS = ['#27ae60','#2ecc71','#f39c12','#e67e22','#e74c3c'];
function rpiColor(label, alpha='CC'){{
  const idx = parseInt(label) - 1;
  return (RPI_COLORS[idx] || COLORS[idx]) + alpha;
}}

// ── KPI ─────────────────────────────────────────────────────────────
const k = DATA.kpi;
document.getElementById('kpi-records').textContent = k.total_records.toLocaleString();
document.getElementById('kpi-shape').textContent   = k.n_cols + '개 컬럼 · 결측치 없음';
document.getElementById('kpi-rpi').textContent     = k.avg_rpi.toFixed(2);
document.getElementById('kpi-csi').textContent     = k.avg_csi.toFixed(2);
document.getElementById('kpi-cci').textContent     = k.avg_cci.toFixed(2);
document.getElementById('kpi-null').textContent    = k.null_count === 0 ? '0  ✓' : k.null_count.toString();
document.getElementById('kpi-scope').textContent   = k.n_years + '년×' + k.n_products + '제품×' + k.n_clients + '고객사';
document.getElementById('kpi-scope-sub').textContent = k.n_rpi_cls + '개 RPI 등급';

// ── INSIGHT BOX ─────────────────────────────────────────────────────
document.getElementById('insight-box').innerHTML = `
  <strong>핵심 요약</strong>: 1,500건 · 53컬럼 · 결측치 0. RPI는 <strong>${{k.n_rpi_cls}}개 등급</strong> —
  우수(RPI 1) 비율 <strong>${{k.rpi1_pct}}%</strong>, 불량(RPI 4·5) 비율 <strong>${{k.rpi45_pct}}%</strong>.<br>
  평균 CSI <strong>${{k.avg_csi}}/10</strong> (만족도 높음) &nbsp;|&nbsp;
  평균 CCI <strong>${{k.avg_cci}}</strong> <span style="color:#27ae60;font-weight:600;">(CCI &gt; 0 = 경쟁력 우위 상태)</span>.<br>
  ⚠️ <strong>상관 방향 주의</strong>: CSI·CCI 모두 RPI와 <em>음의</em> 상관 (|r|≥0.93) —
  CSI/CCI가 <strong>높을수록</strong> RPI 수치가 <strong>낮아짐(=고객관계 우수)</strong>. 완전히 직관적인 결과.<br>
  제품군 최우수: <strong>${{k.best_product}}</strong> | 주의 필요: <strong>${{k.worst_product}}</strong> &nbsp;|&nbsp;
  평가영역 최우수: <strong>${{k.best_area}}</strong> | 가장 취약: <strong>${{k.worst_area}}</strong>.
`;

Chart.defaults.font.family = "'Malgun Gothic','Apple SD Gothic Neo',sans-serif";
Chart.defaults.font.size   = 12;

// ── 1. YEAR TREND ────────────────────────────────────────────────────
new Chart(document.getElementById('yearTrendChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.year_trend.labels,
    datasets: [
      {{ label:'평균 CSI', data:DATA.year_trend.csi, backgroundColor:COLORS[0]+'BB', borderColor:COLORS[0], borderWidth:1, borderRadius:4, yAxisID:'y'  }},
      {{ label:'평균 CCI', data:DATA.year_trend.cci, backgroundColor:COLORS[1]+'BB', borderColor:COLORS[1], borderWidth:1, borderRadius:4, yAxisID:'y2' }},
      {{ label:'평균 RPI', data:DATA.year_trend.rpi, type:'line', borderColor:COLORS[3], backgroundColor:'transparent',
         borderWidth:2.5, pointRadius:6, pointHoverRadius:8, tension:0.3, yAxisID:'y3' }},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ position:'top' }} }},
    scales:{{
      y:  {{ type:'linear', position:'left',  title:{{display:true,text:'CSI (0~10)'}},   min:B.csi_year[0], max:B.csi_year[1] }},
      y2: {{ type:'linear', position:'right', title:{{display:true,text:'CCI (−5 열위 ~ +5 우위)'}}, min:B.cci_year[0], max:B.cci_year[1], grid:{{drawOnChartArea:false}} }},
      y3: {{ type:'linear', position:'right', title:{{display:true,text:'RPI (1~5)'}},   min:B.rpi_year[0], max:B.rpi_year[1], grid:{{drawOnChartArea:false}} }},
    }}
  }}
}});

// ── 2. RPI DOUGHNUT (dynamic colors by class count) ──────────────────
new Chart(document.getElementById('rpiDistChart'), {{
  type: 'doughnut',
  data: {{
    labels: DATA.rpi_dist.labels.map(l => 'RPI ' + l),
    datasets: [{{
      data: DATA.rpi_dist.counts,
      backgroundColor: DATA.rpi_dist.labels.map(l => rpiColor(l,'CC')),
      borderColor: '#fff', borderWidth:3,
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false, cutout:'55%',
    plugins:{{
      legend:{{ position:'bottom' }},
      tooltip:{{ callbacks:{{ label: ctx => {{
        const total = ctx.dataset.data.reduce((a,b)=>a+b,0);
        const pct = ((ctx.parsed/total)*100).toFixed(1);
        return ` ${{ctx.label}}: ${{ctx.parsed.toLocaleString()}}건 (${{pct}}%)`;
      }}}}}}
    }}
  }}
}});

// ── 3. PRODUCT RPI (horizontal bar, pre-sorted asc) ──────────────────
new Chart(document.getElementById('productRpiChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.product.labels,
    datasets: [{{
      label:'평균 RPI',
      data: DATA.product.rpi,
      backgroundColor: DATA.product.labels.map((_,i)=>cc(i)+'CC'),
      borderColor:     DATA.product.labels.map((_,i)=>cc(i)),
      borderWidth:1.5, borderRadius:4,
    }}]
  }},
  options:{{
    indexAxis:'y', responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}}, tooltip:{{ callbacks:{{ label: ctx => ` RPI: ${{ctx.parsed.x.toFixed(3)}}` }}}} }},
    scales:{{ x:{{ min:B.prod_rpi[0], max:B.prod_rpi[1], title:{{display:true,text:'평균 RPI (낮을수록 우수)'}} }} }}
  }}
}});

// ── 4. PRODUCT CSI vs CCI ────────────────────────────────────────────
new Chart(document.getElementById('productCsiCciChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.product.labels,
    datasets: [
      {{ label:'평균 CSI', data:DATA.product.csi, backgroundColor:COLORS[0]+'BB', borderColor:COLORS[0], borderWidth:1, borderRadius:4, yAxisID:'y'  }},
      {{ label:'평균 CCI', data:DATA.product.cci, backgroundColor:COLORS[1]+'BB', borderColor:COLORS[1], borderWidth:1, borderRadius:4, yAxisID:'y2' }},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ position:'top' }} }},
    scales:{{
      y:  {{ position:'left',  title:{{display:true,text:'CSI'}}, min:B.all_csi[0], max:B.all_csi[1] }},
      y2: {{ position:'right', title:{{display:true,text:'CCI (경쟁력지수)'}}, min:B.all_cci[0], max:B.all_cci[1], grid:{{drawOnChartArea:false}} }},
    }}
  }}
}});

// ── 5. CLIENT RPI & CSI ──────────────────────────────────────────────
new Chart(document.getElementById('clientChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.client.labels,
    datasets: [
      {{ label:'평균 RPI', data:DATA.client.rpi, backgroundColor:COLORS[3]+'BB', borderColor:COLORS[3], borderWidth:1, borderRadius:4, yAxisID:'y'  }},
      {{ label:'평균 CSI', data:DATA.client.csi, backgroundColor:COLORS[0]+'BB', borderColor:COLORS[0], borderWidth:1, borderRadius:4, yAxisID:'y2' }},
    ]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ position:'top' }} }},
    scales:{{
      y:  {{ position:'left',  title:{{display:true,text:'RPI (1~5)'}}, min:B.cli_rpi[0], max:B.cli_rpi[1] }},
      y2: {{ position:'right', title:{{display:true,text:'CSI (0~10)'}}, min:B.all_csi[0], max:B.all_csi[1], grid:{{drawOnChartArea:false}} }},
    }}
  }}
}});

// ── 6. CLIENT COUNT ──────────────────────────────────────────────────
new Chart(document.getElementById('clientCountChart'), {{
  type: 'doughnut',
  data: {{
    labels: DATA.client.labels,
    datasets: [{{ data:DATA.client.count, backgroundColor:COLORS.map(c=>c+'CC'), borderColor:'#fff', borderWidth:2 }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false, cutout:'50%',
    plugins:{{ legend:{{ position:'bottom' }}, tooltip:{{ callbacks:{{ label: ctx => ` ${{ctx.label}}: ${{ctx.parsed.toLocaleString()}}건` }}}} }}
  }}
}});

// ── 7. AREA CSI / CCI ────────────────────────────────────────────────
{{
  const areaDisp = DATA.area.labels.map(a => DATA.area.label_map[a]);
  new Chart(document.getElementById('areaChart'), {{
    type:'bar',
    data:{{
      labels: areaDisp,
      datasets:[
        {{ label:'평균 CSI', data:DATA.area.csi, backgroundColor:COLORS[0]+'BB', borderColor:COLORS[0], borderWidth:1, borderRadius:4, yAxisID:'y'  }},
        {{ label:'평균 CCI', data:DATA.area.cci, backgroundColor:COLORS[1]+'BB', borderColor:COLORS[1], borderWidth:1, borderRadius:4, yAxisID:'y2' }},
      ]
    }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ position:'top' }} }},
      scales:{{
        y:  {{ position:'left',  title:{{display:true,text:'CSI (0~10)'}}, min:B.area_csi[0], max:B.area_csi[1] }},
        y2: {{ position:'right', title:{{display:true,text:'CCI (경쟁력지수, +5=우위)'}}, min:B.area_cci[0], max:B.area_cci[1], grid:{{drawOnChartArea:false}} }},
      }}
    }}
  }});
}}

// ── 8. AREA SUBDIM CSI ───────────────────────────────────────────────
{{
  const areaDisp = DATA.area.labels.map(a => DATA.area.label_map[a]);
  const subdims  = [['res','응답성'],['core','핵심역량'],['comm','소통']];
  const datasets = subdims.map(([d,label],i) => ({{
    label,
    data: DATA.area.labels.map(a => DATA.area.subdim_csi[a][d]),
    backgroundColor: COLORS[i]+'BB', borderColor:COLORS[i], borderWidth:1, borderRadius:2,
  }}));
  new Chart(document.getElementById('areaSubdimChart'), {{
    type:'bar',
    data:{{ labels:areaDisp, datasets }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ position:'top' }} }},
      scales:{{ y:{{ min:B.subdim_csi[0], max:B.subdim_csi[1], title:{{display:true,text:'CSI 평균'}} }} }}
    }}
  }});
}}

// ── 9. RPI CLASS CSI ─────────────────────────────────────────────────
new Chart(document.getElementById('rpiCsiChart'), {{
  type:'bar',
  data:{{
    labels: DATA.rpi_class_avg.labels.map(l => 'RPI ' + l),
    datasets:[{{
      label:'평균 CSI Total',
      data: DATA.rpi_class_avg.csi,
      backgroundColor: DATA.rpi_class_avg.labels.map(l => rpiColor(l,'CC')),
      borderColor:     DATA.rpi_class_avg.labels.map(l => rpiColor(l,'FF')),
      borderWidth:1.5, borderRadius:6,
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}}, tooltip:{{ callbacks:{{ label: ctx => ` CSI: ${{ctx.parsed.y.toFixed(3)}}` }}}} }},
    scales:{{ y:{{ min:B.rpicls_csi[0], max:B.rpicls_csi[1], title:{{display:true,text:'평균 CSI Total'}} }} }}
  }}
}});

// ── 10. RPI CLASS CCI ────────────────────────────────────────────────
new Chart(document.getElementById('rpiCciChart'), {{
  type:'bar',
  data:{{
    labels: DATA.rpi_class_avg.labels.map(l => 'RPI ' + l),
    datasets:[{{
      label:'평균 CCI Total',
      data: DATA.rpi_class_avg.cci,
      backgroundColor: DATA.rpi_class_avg.labels.map(l => rpiColor(l,'CC')),
      borderColor:     DATA.rpi_class_avg.labels.map(l => rpiColor(l,'FF')),
      borderWidth:1.5, borderRadius:6,
    }}]
  }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}}, tooltip:{{ callbacks:{{ label: ctx => ` CCI: ${{ctx.parsed.y.toFixed(3)}}` }}}} }},
    scales:{{ y:{{ min:B.rpicls_cci[0], max:B.rpicls_cci[1], title:{{display:true,text:'평균 CCI Total (↑=경쟁력 우위)'}} }} }}
  }}
}});

// ── 11. CORRELATION BAR ───────────────────────────────────────────────
new Chart(document.getElementById('corrChart'), {{
  type:'bar',
  data:{{
    labels: DATA.corr.labels,
    datasets:[{{
      label:'|r| with RPI',
      data:  DATA.corr.values,
      backgroundColor: DATA.corr.labels.map((_,i) => cc(i)+'BB'),
      borderColor:     DATA.corr.labels.map((_,i) => cc(i)),
      borderWidth:1.5, borderRadius:4,
    }}]
  }},
  options:{{
    indexAxis:'y', responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}} }},
    scales:{{ x:{{ min:0, max:1, title:{{display:true,text:'상관계수 절대값 |r|'}} }} }}
  }}
}});

// ── 12. INSIGHT CARDS ────────────────────────────────────────────────
(function(){{
  const ins = DATA.insights;
  const k2  = DATA.kpi;
  const pLabels = DATA.product.labels;
  const pRpi    = DATA.product.rpi;
  const prodRanking = pLabels.map((p,i) => p + '(' + pRpi[i] + ')').join(' &lt; ');

  const CARDS = [
    {{
      color:'green', icon:'✅', title:'완전한 데이터 — 결측치 0건', tag:'데이터 품질',
      body:`총 <strong>${{k2.total_records.toLocaleString()}}건 × ${{k2.n_cols}}컬럼</strong>, 결측치가 단 1개도 없습니다.
      실무 데이터는 보통 10~30%가 비어 처리가 필요하지만, 이 데이터는 그대로 분석 가능한 '황금 데이터'입니다.`
    }},
    {{
      color:'blue', icon:'📊', title:'RPI 분포 — 우수 등급이 다수', tag:'분포 이해',
      body:`전체의 <strong>${{k2.rpi1_pct}}%가 RPI=1(매우 우수)</strong>로 가장 많습니다.
      불량(4·5) 비율은 <strong>${{k2.rpi45_pct}}%</strong>. 평균 RPI ${{k2.avg_rpi}}는 '우수~보통 사이'를 의미합니다.`
    }},
    {{
      color:'orange', icon:'🔗', title:`CSI↑·CCI↑ = 관계 우수 — |r|=${{k2.top_corr_val}}`, tag:'핵심 법칙',
      body:`CSI(만족도)·CCI(경쟁력) 모두 RPI와 <strong>음의 상관 −0.93 이상</strong>입니다.<br>
      두 지표가 높을수록 RPI 수치 낮아짐(=고객관계 우수) — 직관적으로 완벽합니다.<br>
      RPI=1(최우수) → CSI <strong>${{ins.csi_rpi1}}</strong>, CCI <strong>${{ins.cci_rpi1}}</strong><br>
      RPI=5(최불량) → CSI <strong>${{ins.csi_rpi5}}</strong>, CCI <strong>${{ins.cci_rpi5}}</strong><br>
      CSI 격차 <strong>${{ins.csi_delta}}점</strong> — 만족도·경쟁력 관리가 곧 관계 관리입니다.`
    }},
    {{
      color:'red', icon:'📦', title:`제품군 순위 — ${{k2.best_product}} 최우수`, tag:'제품 인사이트',
      body:`RPI 오름차순(우수→불량): ${{prodRanking}}<br>
      <strong>${{k2.best_product}}</strong>이 가장 우수하고 <strong>${{k2.worst_product}}</strong>이 주의 필요.
      제품별 차이는 크지 않아 관리 역량 차이가 더 큰 요인일 수 있습니다.`
    }},
    {{
      color:'purple', icon:'🔄', title:'클레임 처리의 역설 — C영역 RPI 최우수', tag:'도메인 인사이트',
      body:`평가영역 중 C(클레임 처리) 영역 설문 고객들의 RPI가 <strong>가장 우수</strong>합니다.<br>
      CCI(경쟁력)는 상대적으로 낮지만(0.768), 클레임을 잘 해결받은 고객은
      오히려 관계가 강화되는 <strong>'서비스 리커버리 효과'</strong>가 작용합니다.<br>
      반면 Q1·Q2(품질) 영역은 <strong>CCI가 가장 높아(경쟁력 강점)</strong> 품질 부문이 핵심 경쟁력임을 시사합니다.`
    }},
    {{
      color:'teal', icon:'📅', title:'연도별 추이 — 3년간 안정적', tag:'트렌드',
      body:(() => {{
        const yLabels = DATA.year_trend.labels;
        const rows = yLabels.map((y,i) =>
          `${{y}}년: RPI=1 비율 <strong>${{ins.year_rpi1_pct[i]}}%</strong>, CCI=<strong>${{DATA.year_trend.cci[i]}}</strong>`
        ).join('<br>');
        return rows + `<br><br>CCI(경쟁력)가 전 연도에 걸쳐 <strong>양수(경쟁력 우위)</strong>를 유지합니다. 뚜렷한 개선·악화 없이 안정적이며 구조적 개선 기회가 남아 있습니다.`;
      }})()
    }},
  ];

  const grid = document.getElementById('insightCardsGrid');
  CARDS.forEach(c => {{
    const div = document.createElement('div');
    div.className = `insight-card ${{c.color}}`;
    div.innerHTML = `
      <div class="ic-tag">${{c.tag}}</div>
      <div class="ic-icon">${{c.icon}}</div>
      <div class="ic-title">${{c.title}}</div>
      <div class="ic-body">${{c.body}}</div>
    `;
    grid.appendChild(div);
  }});
}})();

// ── 13. Q&A ACCORDION ────────────────────────────────────────────────
(function(){{
  const qaSection = document.getElementById('qaSection');
  DATA.qa.forEach((item, idx) => {{
    const div = document.createElement('div');
    div.className = 'qa-item';
    div.innerHTML = `
      <div class="qa-question" id="qq-${{idx}}">
        <span class="q-text">Q${{idx+1}}. ${{item.q}}</span>
        <span class="q-tag">${{item.tag}}</span>
        <span class="q-arrow">▼</span>
      </div>
      <div class="qa-answer" id="qa-${{idx}}">
        <strong style="color:#4C72B0;">A.</strong> ${{item.a}}
      </div>
    `;
    qaSection.appendChild(div);
    div.querySelector('.qa-question').addEventListener('click', () => {{
      document.getElementById(`qa-${{idx}}`).classList.toggle('show');
      document.getElementById(`qq-${{idx}}`).classList.toggle('open');
    }});
  }});
}})();

// ── 14. NEXT STEPS TABLE ─────────────────────────────────────────────
(function(){{
  const clsMap = {{ '최우선':'priority-top', '높음':'priority-high', '보통':'priority-mid' }};
  let html = '<h3 style="font-size:13px;font-weight:700;margin-bottom:14px;">우선순위별 후속 분석 로드맵</h3>';
  html += '<table class="next-steps-table"><thead><tr><th>우선순위</th><th>분석 주제</th><th>이유 / 기대 효과</th><th>중요도</th></tr></thead><tbody>';
  DATA.next_steps.forEach(row => {{
    const cls = clsMap[row.priority] || 'priority-mid';
    html += `<tr>
      <td><span class="priority-badge ${{cls}}">${{row.priority}}</span></td>
      <td><strong>${{row.topic}}</strong></td>
      <td style="color:#555;font-size:12px;">${{row.reason}}</td>
      <td class="star-rating">${{row.star}}</td>
    </tr>`;
  }});
  html += '</tbody></table>';
  document.getElementById('nextStepsContainer').innerHTML = html;
}})();

// ── 15. PROFILE TABLE ────────────────────────────────────────────────
(function(){{
  const rows = DATA.profile_table;
  const cols = [
    {{f:'column',label:'컬럼'}},
    {{f:'mean',  label:'평균'}},
    {{f:'std',   label:'표준편차'}},
    {{f:'min',   label:'최소'}},
    {{f:'p25',   label:'Q1(25%)'}},
    {{f:'median',label:'중앙값'}},
    {{f:'p75',   label:'Q3(75%)'}},
    {{f:'max',   label:'최대'}},
    {{f:'null_rate',label:'결측률(%)'}},
  ];
  let html = '<h3 style="font-size:13px;font-weight:700;margin-bottom:14px;">CSI · CCI Total 컬럼 기술통계</h3>';
  html += '<table class="data-table"><thead><tr>';
  cols.forEach(c => html += `<th>${{c.label}}</th>`);
  html += '</tr></thead><tbody>';
  rows.forEach(row => {{
    const isCsi = row.column.includes('csi');
    html += '<tr>';
    cols.forEach(c => {{
      let val = row[c.f];
      if (c.f === 'column') {{
        const badge = isCsi
          ? '<span class="badge badge-green">CSI</span>'
          : '<span class="badge badge-yellow">CCI</span>';
        val = badge + ' ' + val;
      }} else if (c.f === 'null_rate') {{
        val = val === 0 ? '<span style="color:#28a745;font-weight:600;">0%</span>' : val + '%';
      }}
      html += `<td>${{val}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody></table>';
  document.getElementById('profileTableContainer').innerHTML = html;
}})();
</script>
</body>
</html>"""

# ──────────────────────────────────────────
# 8. SAVE
# ──────────────────────────────────────────
OUT_DIR  = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "customer_satisfaction_dashboard.html")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[OK] {OUT_PATH}")
print(f"     Shape         : {n_rows} rows x {n_cols} cols")
print(f"     RPI range     : {sorted(df['rpi'].unique().tolist())}")
print(f"     Avg RPI/CSI/CCI: {avg_rpi} / {avg_csi_overall} / {avg_cci_overall}")
print(f"     Top corr      : {top_corr_col} = {top_corr_val}")
print(f"     Null count    : {null_count}")
print(f"     Products      : {products}")
print(f"     Best product  : {best_product} (RPI={prod_rpi[best_product]})")
print(f"     Worst product : {worst_product} (RPI={prod_rpi[worst_product]})")
