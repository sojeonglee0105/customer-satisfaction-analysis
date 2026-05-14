"""Page 5 — 모델 성능 비교 + 통계 검정 + Confusion Matrix + ROC."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.insight_utils import render_comparison_insights
from utils.model_utils import (
    METRIC_KEYS,
    METRIC_LABELS,
    aggregate_pivot,
    aggregate_results,
    friedman_test,
    get_confusion_matrix,
    pairwise_wilcoxon,
)
from utils.progress_utils import render_step_progress
from utils.viz_utils import (
    confusion_matrix_chart,
    metric_box_plot,
    metric_delta_bar,
    radar_chart,
    roc_curves_chart,
)

st.set_page_config(page_title="LGD CSAT — 5. 모델 비교", page_icon="📈", layout="wide")
render_step_progress(current_step=5)
st.title("📈 5. 모델 성능 비교")

if "training_result" not in st.session_state:
    st.warning("⬅ 먼저 **4_Model_Training** 페이지에서 모델을 학습해주세요.")
    st.stop()

res     = st.session_state["training_result"]
raw     = res["raw"]
seeds   = res["seeds"]
models  = res["model_names"]
classes = res["all_classes"]

agg_df = aggregate_results(raw, models, seeds)
pivot  = aggregate_pivot(agg_df)

# ── 평가 지표 선택 ───────────────────────────────────────────────
st.subheader("평가 지표 선택")
selected_metrics = st.multiselect(
    "표시할 지표", options=METRIC_KEYS,
    default=["macro_f1", "micro_f1", "macro_auroc", "micro_auroc"],
    format_func=lambda k: METRIC_LABELS[k],
)

if not selected_metrics:
    st.warning("지표를 1개 이상 선택해주세요.")
    st.stop()

# ── 1. 결과 테이블 ───────────────────────────────────────────────
st.subheader("1️⃣ Mean ± Std 결과 테이블")
display_rows = []
for m in models:
    row = {"모델": m}
    for k in selected_metrics:
        sub = agg_df[(agg_df["model"] == m) & (agg_df["metric"] == k)]
        if not sub.empty and sub["mean"].iloc[0] is not None:
            mean = sub["mean"].iloc[0]
            std  = sub["std"].iloc[0]
            row[METRIC_LABELS[k]] = f"{mean:.4f} ± {std:.4f}"
        else:
            row[METRIC_LABELS[k]] = "—"
    sec_sub = agg_df[(agg_df["model"] == m) & (agg_df["metric"] == "train_sec")]
    if not sec_sub.empty:
        row["평균 학습시간(s)"] = f"{sec_sub['mean'].iloc[0]:.3f}"
    display_rows.append(row)
table_df = pd.DataFrame(display_rows)

# 최고값 하이라이트
def highlight_best(s: pd.Series):
    try:
        nums = s.str.split(" ±").str[0].astype(float)
        is_best = nums == nums.max()
        return ["background-color: #16a34a; color: white; font-weight: 700;"
                if v else "" for v in is_best]
    except Exception:
        return [""] * len(s)

styled = table_df.style.apply(highlight_best,
                                subset=[c for c in table_df.columns if c not in ("모델", "평균 학습시간(s)")])
st.dataframe(styled, use_container_width=True)

# ── 2. Radar Chart ───────────────────────────────────────────────
st.subheader("2️⃣ Radar Chart — 모델별 종합 성능")
radar_metrics = [m for m in selected_metrics if m in pivot.columns]
if radar_metrics:
    st.plotly_chart(radar_chart(pivot, radar_metrics), use_container_width=True)

# ── 3. Box Plot — Seed 분산 ──────────────────────────────────────
st.subheader("3️⃣ Seed별 분산 — Box Plot")
metric_for_box = st.selectbox("Box Plot 지표", options=selected_metrics,
                                 format_func=lambda k: METRIC_LABELS[k], key="box_metric")
st.plotly_chart(metric_box_plot(agg_df, metric_for_box), use_container_width=True)

# ── 4. 보정 vs 미보정 비교 (옵션) ─────────────────────────────────
if "training_result_uw" in st.session_state:
    st.subheader("4️⃣ class_weight 보정 vs 미보정 비교")
    res_uw = st.session_state["training_result_uw"]
    agg_uw = aggregate_results(res_uw["raw"], res_uw["model_names"], res_uw["seeds"])
    delta_metric = st.selectbox("비교 지표", options=selected_metrics,
                                  format_func=lambda k: METRIC_LABELS[k], key="delta_metric")
    st.plotly_chart(metric_delta_bar(agg_uw, agg_df, delta_metric),
                      use_container_width=True)

# ── 5. 통계 검정 ─────────────────────────────────────────────────
st.subheader("5️⃣ 통계 검정")
friedman_results: list[dict] = []
with st.expander("Friedman Test (전체 모델 차이 유의성)", expanded=True):
    rows = []
    for k in selected_metrics:
        f = friedman_test(raw, models, seeds, k)
        friedman_results.append({"metric": k, **f})
        rows.append({
            "지표": METRIC_LABELS[k],
            "통계량": f.get("statistic"),
            "p-value": f.get("pvalue"),
            "결과": (
                "★ 유의 (p<0.05)" if f.get("pvalue") is not None and f["pvalue"] < 0.05
                else "비유의" if f.get("pvalue") is not None
                else f.get("note", "—")
            ),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with st.expander("Pairwise Wilcoxon Signed-Rank (Bonferroni)"):
    metric_for_wil = st.selectbox("Wilcoxon 검정 지표", options=selected_metrics,
                                    format_func=lambda k: METRIC_LABELS[k], key="wil_metric")
    wil_df = pairwise_wilcoxon(raw, models, seeds, metric_for_wil)
    if not wil_df.empty:
        wil_df = wil_df.copy()
        wil_df["유의"] = wil_df["pvalue_bonf"].apply(
            lambda p: "★" if p is not None and p < 0.05 else ""
        )
        st.dataframe(wil_df, use_container_width=True)

# ── 6. Confusion Matrix ──────────────────────────────────────────
st.subheader("6️⃣ Confusion Matrix")
last_models = res["last_models"]
cm_model = st.selectbox("모델 선택", options=models, key="cm_model")
selected_seed = seeds[-1]
y_pred = raw[selected_seed][cm_model].get("y_pred")
if y_pred is not None:
    y_test = st.session_state["fe_y_test"]
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    cm = get_confusion_matrix(y_test, y_pred, labels=labels)
    st.plotly_chart(confusion_matrix_chart(cm, labels, f"Confusion Matrix — {cm_model} (seed={selected_seed})"),
                      use_container_width=True)

# ── 7. ROC Curves ────────────────────────────────────────────────
st.subheader("7️⃣ ROC Curves (마지막 seed 모델 기준)")
prob_dict: dict[str, np.ndarray | None] = {}
for m in models:
    p = raw[selected_seed][m].get("y_prob")
    if p is not None:
        prob_dict[m] = np.array(p)
if prob_dict:
    y_test = st.session_state["fe_y_test"]
    st.plotly_chart(roc_curves_chart(y_test, prob_dict, classes,
                                          title=f"ROC Curves — seed {selected_seed}"),
                      use_container_width=True)
else:
    st.info("predict_proba를 지원하지 않는 모델이 포함되어 있어 ROC를 표시할 수 없습니다.")

# ════════════════════════════════════════════════════════════════════
# 💡 가장 중요한 결과 정리 (비전공자 친화)
# ════════════════════════════════════════════════════════════════════
# Friedman 결과 — Macro F1이 selected_metrics에 없을 경우 별도로 보충
if not any(r.get("metric") == "macro_f1" for r in friedman_results):
    f_extra = friedman_test(raw, models, seeds, "macro_f1")
    friedman_results.append({"metric": "macro_f1", **f_extra})

render_comparison_insights(
    agg_df=agg_df,
    models=models,
    n_seeds=len(seeds),
    friedman_results=friedman_results,
    cw_uw_compare="training_result_uw" in st.session_state,
)
