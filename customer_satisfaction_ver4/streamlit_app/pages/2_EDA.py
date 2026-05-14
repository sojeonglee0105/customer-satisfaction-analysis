"""Page 2 — 동적 EDA (Exploratory Data Analysis)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.insight_utils import render_eda_insights
from utils.progress_utils import render_step_progress
from utils.viz_utils import (
    correlation_heatmap,
    histogram_grid,
    multicollinearity_pairs,
    target_distribution_chart,
    violin_by_class,
)

st.set_page_config(page_title="LGD CSAT — 2. EDA", page_icon="🔍", layout="wide")
render_step_progress(current_step=2)
st.title("🔍 2. 탐색적 데이터 분석 (EDA)")

if "df" not in st.session_state:
    st.warning("⬅ 먼저 **1_Upload** 페이지에서 데이터를 업로드해주세요.")
    st.stop()

df       = st.session_state["df"]
target   = st.session_state["target"]
features = st.session_state["feature_cols"]

c1, c2, c3 = st.columns(3)
c1.metric("샘플 수", f"{len(df):,}")
c2.metric("독립변수", f"{len(features):,}")
c3.metric("타겟 클래스", f"{df[target].nunique()}")

# ── 1. 타겟 분포 ──────────────────────────────────────────────────
st.subheader("1️⃣ 타겟 변수 분포")
st.plotly_chart(target_distribution_chart(df[target], f"{target} 클래스 분포"),
                  use_container_width=True)

cls_count = df[target].value_counts()
imbalance = cls_count.max() / max(cls_count.min(), 1)
if imbalance > 10:
    st.warning(f"⚠ 클래스 불균형 비율 = {imbalance:.1f}:1 — `class_weight='balanced'` 적용을 권장합니다.")
elif imbalance > 3:
    st.info(f"클래스 불균형 비율 = {imbalance:.1f}:1 — 보정 고려 대상.")
else:
    st.success(f"클래스 균형 양호 (비율 {imbalance:.1f}:1)")

# ── 2. 수치형 변수 분포 ───────────────────────────────────────────
st.subheader("2️⃣ 수치형 변수 분포")
numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
if numeric_features:
    n_to_show = st.slider("표시할 변수 수", min_value=3, max_value=min(24, len(numeric_features)),
                            value=min(9, len(numeric_features)))
    selected_dist = st.multiselect(
        "히스토그램 변수 선택",
        options=numeric_features,
        default=numeric_features[:n_to_show],
    )
    overlay = st.toggle("타겟 클래스별 색상 분리", value=True)
    if selected_dist:
        st.plotly_chart(
            histogram_grid(df, selected_dist, target=target if overlay else None),
            use_container_width=True,
        )
else:
    st.info("수치형 독립변수가 없습니다.")

# ── 3. 상관관계 / 히트맵 ─────────────────────────────────────────
st.subheader("3️⃣ 상관관계 분석")
top_n_corr = st.slider("히트맵 변수 수 (타겟과의 |r| 기준 상위)",
                         min_value=5, max_value=min(50, len(numeric_features) + 1),
                         value=min(20, len(numeric_features) + 1))
df_for_corr = df[numeric_features + [target]] if pd.api.types.is_numeric_dtype(df[target]) else df[numeric_features]
target_for_corr = target if pd.api.types.is_numeric_dtype(df[target]) else None
st.plotly_chart(correlation_heatmap(df_for_corr, top_n=top_n_corr, target=target_for_corr),
                  use_container_width=True)

# ── 4. 다중공선성 ────────────────────────────────────────────────
st.subheader("4️⃣ 다중공선성 (변수 간 고상관 쌍)")
threshold = st.slider("|r| 임계값", min_value=0.5, max_value=0.99, value=0.9, step=0.01)
multi_df = multicollinearity_pairs(df[numeric_features], threshold=threshold)
if multi_df.empty:
    st.success(f"|r| ≥ {threshold} 쌍이 없습니다 ✓")
else:
    st.warning(f"|r| ≥ {threshold} 쌍이 {len(multi_df)}개 있습니다.")
    st.dataframe(multi_df, use_container_width=True, height=min(400, 35 * len(multi_df) + 40))

# ── 5. 클래스별 분포 비교 ────────────────────────────────────────
st.subheader("5️⃣ 변수의 클래스별 분포 비교 (Violin Plot)")
if numeric_features:
    feat_for_violin = st.selectbox("변수 선택", options=numeric_features, key="violin_feat")
    st.plotly_chart(violin_by_class(df, feat_for_violin, target),
                      use_container_width=True)

# ── 6. Kruskal-Wallis 단변량 검정 ──────────────────────────────────
st.subheader("6️⃣ Kruskal-Wallis 단변량 검정 — 각 변수의 클래스 분리 능력")
classes = df[target].dropna().unique()
if len(classes) < 2:
    st.info("클래스가 1개뿐이라 검정을 수행할 수 없습니다.")
else:
    rows = []
    n = len(df)
    k = len(classes)
    for col in numeric_features:
        groups = [df.loc[df[target] == c, col].dropna().values for c in classes]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        try:
            h, p = stats.kruskal(*groups)
            eta2 = max((h - k + 1) / (n - k), 0) if n > k else 0
            rows.append({
                "변수": col,
                "H 통계량": round(float(h), 4),
                "p-value": round(float(p), 6),
                "η²": round(float(eta2), 4),
                "유의": "★" if p < 0.05 else "",
            })
        except Exception:
            pass
    kw_df = pd.DataFrame(rows).sort_values("η²", ascending=False).reset_index(drop=True)
    if kw_df.empty:
        st.info("KW 검정 가능한 변수가 없습니다.")
    else:
        st.dataframe(kw_df, use_container_width=True, height=min(420, 35 * len(kw_df) + 40))
        n_sig = (kw_df["p-value"] < 0.05).sum()
        st.caption(f"전체 {len(kw_df)}개 변수 중 {n_sig}개가 p<0.05로 유의 — η²이 클수록 클래스 분리 능력이 강합니다.")
        st.session_state["eda_kw_df"] = kw_df

# ════════════════════════════════════════════════════════════════════
# 💡 가장 중요한 결과 정리 (비전공자 친화)
# ════════════════════════════════════════════════════════════════════
_kw_for_insight = st.session_state.get("eda_kw_df")
_missing_total = int(df.isna().sum().sum())
render_eda_insights(
    n_samples=len(df),
    n_features=len(features),
    n_classes=int(df[target].nunique()),
    target_name=str(target),
    class_counts=cls_count.to_dict(),
    imbalance_ratio=float(imbalance),
    kw_df=_kw_for_insight,
    multi_df=multi_df if not multi_df.empty else None,
    numeric_n=len(numeric_features),
    missing_total=_missing_total,
)
