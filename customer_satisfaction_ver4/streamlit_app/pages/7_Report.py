"""Page 7 — 종합 리포트 대시보드.

기존 HTML 리포트(EDA/FE/Model Comparison/Feature Importance)의 항목을 모두
대화형 탭으로 통합하고, 추가 과제 제안 섹션을 명시적으로 포함합니다.
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.data_utils import encode_categorical, load_uploaded_file
from utils.model_utils import (
    METRIC_KEYS,
    METRIC_LABELS,
    aggregate_pivot,
    aggregate_results,
    friedman_test,
    get_confusion_matrix,
    pairwise_wilcoxon,
)
from utils.pptx_utils import build_pptx_report
from utils.progress_utils import render_step_progress
from utils.report_content import (
    ADDITIONAL_TASKS,
    DEFAULT_DOMAIN_MAP,
    INDEX_DEFINITIONS,
    INSIGHT_GUIDES,
    detect_domain,
    domain_legend_html,
    insight_to_html,
    tasks_to_html,
)
from utils.viz_utils import (
    confusion_matrix_chart,
    correlation_heatmap,
    importance_bar,
    metric_box_plot,
    metric_delta_bar,
    multicollinearity_pairs,
    radar_chart,
    roc_curves_chart,
    target_distribution_chart,
)
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LGD CSAT — 7. 종합 리포트", page_icon="📑", layout="wide")
render_step_progress(current_step=7)
st.title("📑 7. 종합 리포트 대시보드")
st.caption(
    "EDA · Feature Engineering · Model Comparison · Feature Importance · 도메인 해석 · "
    "추가 과제 제안을 통합 제공합니다."
)

if "training_result" not in st.session_state:
    st.warning("⬅ 먼저 **4_Model_Training** 페이지에서 모델을 학습해주세요.")
    st.stop()

# ── 세션 데이터 추출 ──────────────────────────────────────────────
res     = st.session_state["training_result"]
raw     = res["raw"]
seeds   = res["seeds"]
models  = res["model_names"]
classes = res["all_classes"]

df       = st.session_state.get("df")
target   = st.session_state.get("target")
features = st.session_state.get("feature_cols", [])
strategy = st.session_state.get("fe_strategy", "—")
fe_feat  = st.session_state.get("fe_feature_names", [])

agg_df = aggregate_results(raw, models, seeds)
pivot  = aggregate_pivot(agg_df)

best_f1_model = pivot["macro_f1"].idxmax() if "macro_f1" in pivot.columns else "-"
best_f1_val   = float(pivot["macro_f1"].max()) if "macro_f1" in pivot.columns else 0.0
best_au_model = pivot["macro_auroc"].idxmax() if "macro_auroc" in pivot.columns else "-"
best_au_val   = float(pivot["macro_auroc"].max()) if "macro_auroc" in pivot.columns else 0.0

# ════════════════════════════════════════════════════════════════════
# 상단 KPI 카드
# ════════════════════════════════════════════════════════════════════
st.subheader("📊 핵심 KPI")
c1, c2, c3, c4 = st.columns(4)
c1.metric("최고 Macro F1 모델", best_f1_model, f"{best_f1_val:.4f}")
c2.metric("최고 Macro AUROC 모델", best_au_model, f"{best_au_val:.4f}")
c3.metric("학습된 모델", f"{len(models)}개")
c4.metric("반복 시드", f"{len(seeds)}회")

# ── 분석 설정 ────────────────────────────────────────────────────
summary_table = pd.DataFrame([
    {"항목": "데이터 크기",
     "값": f"{df.shape[0]:,} 행 × {df.shape[1]:,} 열" if df is not None else "—"},
    {"항목": "종속변수", "값": target or "—"},
    {"항목": "선택한 독립변수 수", "값": str(len(features))},
    {"항목": "FE 전략", "값": str(strategy)},
    {"항목": "변환 후 피처 수", "값": str(len(fe_feat))},
    {"항목": "Train / Test",
     "값": (f"{st.session_state['fe_X_train'].shape[0]} / "
            f"{st.session_state['fe_X_test'].shape[0]}")
            if "fe_X_train" in st.session_state else "—"},
    {"항목": "class_weight", "값": str(res.get("class_weight"))},
    {"항목": "결측 처리", "값": st.session_state.get("impute_strategy", "—")},
])
with st.expander("📌 분석 설정 요약 펼치기", expanded=True):
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

st.divider()

# ════════════════════════════════════════════════════════════════════
# 7개 탭 구조
# ════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📈 종합 요약",
    "🔍 EDA 결과",
    "🛠 Feature Engineering",
    "🤖 모델 비교",
    "🎯 Feature Importance",
    "🏭 도메인 해석",
    "🔬 추가 과제 제안",
])

# ── Tab 0: 종합 요약 ──────────────────────────────────────────────
with tabs[0]:
    st.subheader("모델 성능 종합 테이블")
    display_rows = []
    for m in models:
        row = {"모델": m}
        for k in METRIC_KEYS:
            sub = agg_df[(agg_df["model"] == m) & (agg_df["metric"] == k)]
            if not sub.empty and sub["mean"].iloc[0] is not None:
                row[METRIC_LABELS[k]] = f"{sub['mean'].iloc[0]:.4f} ± {sub['std'].iloc[0]:.4f}"
            else:
                row[METRIC_LABELS[k]] = "—"
        display_rows.append(row)
    table_df = pd.DataFrame(display_rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    metrics_for_radar = [k for k in ["macro_f1", "micro_f1", "macro_auroc", "micro_auroc"]
                           if k in pivot.columns]
    if metrics_for_radar:
        st.plotly_chart(radar_chart(pivot, metrics_for_radar), use_container_width=True)

    # Top Features
    st.subheader("🏆 Top Features")
    top_features_text = "Page 6에서 분석을 실행하면 표시됩니다."
    if "imp_builtin_df" in st.session_state:
        top10 = st.session_state["imp_builtin_df"].head(10).reset_index(drop=True)
        col_t, col_g = st.columns([1, 1])
        with col_t:
            st.dataframe(top10, use_container_width=True, hide_index=True)
        with col_g:
            st.plotly_chart(
                importance_bar(top10, "feature", "importance",
                                "Top-10 Built-in Importance"),
                use_container_width=True,
            )
        top_features_text = ", ".join(top10["feature"].head(5).tolist())
    elif "shap_cache" in st.session_state and st.session_state["shap_cache"].get("shap_3d") is not None:
        from utils.shap_utils import shap_bar_chart, shap_summary_df
        s3d = st.session_state["shap_cache"]["shap_3d"]
        summary = shap_summary_df(s3d, st.session_state["fe_feature_names"]).head(10)
        col_t, col_g = st.columns([1, 1])
        with col_t:
            st.dataframe(summary, use_container_width=True, hide_index=True)
        with col_g:
            st.plotly_chart(shap_bar_chart(summary, top_n=10, signed=False),
                              use_container_width=True)
        top_features_text = ", ".join(summary["feature"].head(5).tolist())
    else:
        st.info(top_features_text)
    st.session_state["_report_top_features_text"] = top_features_text

# ── Tab 1: EDA 결과 ───────────────────────────────────────────────
with tabs[1]:
    st.subheader("타겟 변수 분포")
    if df is not None and target in df.columns:
        st.plotly_chart(target_distribution_chart(df[target], f"{target} 분포"),
                          use_container_width=True)
        cls_count = df[target].value_counts()
        imb = cls_count.max() / max(cls_count.min(), 1)
        if imb > 10:
            st.warning(f"⚠ 클래스 불균형 비율 {imb:.1f}:1 — `class_weight='balanced'` 권장")
        elif imb > 3:
            st.info(f"클래스 불균형 비율 {imb:.1f}:1 — 보정 고려")
        else:
            st.success(f"클래스 균형 양호 (비율 {imb:.1f}:1)")

    st.subheader("상관관계 (Top 30)")
    if df is not None:
        numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        target_for_corr = target if pd.api.types.is_numeric_dtype(df[target]) else None
        df_corr = df[numeric_features + [target]] if target_for_corr else df[numeric_features]
        st.plotly_chart(correlation_heatmap(df_corr, top_n=30, target=target_for_corr),
                          use_container_width=True)

    st.subheader("다중공선성 (|r| ≥ 0.9)")
    if df is not None and numeric_features:
        multi_df = multicollinearity_pairs(df[numeric_features], threshold=0.9)
        if multi_df.empty:
            st.success("|r| ≥ 0.9 쌍 없음 ✓")
        else:
            st.warning(f"{len(multi_df)} 쌍 발견 — PCA / Feature Selection 권장")
            st.dataframe(multi_df.head(30), use_container_width=True, hide_index=True)

    st.subheader("Kruskal-Wallis 단변량 검정 (Top 20)")
    if "eda_kw_df" in st.session_state:
        kw_df = st.session_state["eda_kw_df"]
        kw_top = kw_df.head(20)
        col_t, col_g = st.columns([1, 1])
        with col_t:
            st.dataframe(kw_top, use_container_width=True, hide_index=True)
        with col_g:
            kw_plot = kw_top.iloc[::-1]  # plotly horizontal bar용 역순
            sig_color = ["#16a34a" if p < 0.05 else "#cbd5e1"
                         for p in kw_plot["p-value"]]
            fig_kw = go.Figure(go.Bar(
                x=kw_plot["η²"], y=kw_plot["변수"], orientation="h",
                marker_color=sig_color,
                text=[f"η²={v:.3f}" for v in kw_plot["η²"]],
                textposition="outside",
            ))
            fig_kw.update_layout(
                title="클래스 분리 능력 Top-20 (η², 🟢=유의 / ⚪=비유의)",
                template="plotly_white", height=max(420, 22 * len(kw_plot)),
                xaxis_title="η² (효과 크기)",
                margin=dict(l=160, r=60, t=60, b=40),
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        n_sig = (kw_df["p-value"] < 0.05).sum()
        st.caption(f"전체 {len(kw_df)}개 변수 중 **{n_sig}개**가 p<0.05로 유의 — η²이 클수록 클래스 분리 능력이 강합니다.")
    else:
        st.info("Page 2 (EDA)에서 KW 검정을 실행하면 결과가 표시됩니다.")

    st.markdown("**EDA 핵심 인사이트**")
    g = INSIGHT_GUIDES["eda"]
    for it in g["items"]:
        st.markdown(f"- {it}")

# ── Tab 2: Feature Engineering ────────────────────────────────────
with tabs[2]:
    st.subheader("적용 전략")
    fe_info = pd.DataFrame([
        {"항목": "전략", "값": str(strategy)},
        {"항목": "변환 후 피처 수", "값": str(len(fe_feat))},
        {"항목": "Train shape",
         "값": str(st.session_state["fe_X_train"].shape) if "fe_X_train" in st.session_state else "—"},
        {"항목": "Test shape",
         "값": str(st.session_state["fe_X_test"].shape) if "fe_X_test" in st.session_state else "—"},
    ])
    st.dataframe(fe_info, use_container_width=True, hide_index=True)

    if strategy == "pca" and st.session_state.get("fe_explained_var") is not None:
        from utils.viz_utils import scree_plot
        st.plotly_chart(scree_plot(st.session_state["fe_explained_var"]),
                          use_container_width=True)
        cum = np.cumsum(st.session_state["fe_explained_var"])
        st.caption(f"누적 분산 설명률 {cum[-1]:.2%} (총 {len(cum)} 컴포넌트)")
    elif fe_feat:
        st.markdown(f"**사용 피처 ({len(fe_feat)}개):** "
                      f"{', '.join(fe_feat[:30])}"
                      f"{'...' if len(fe_feat) > 30 else ''}")

    st.markdown("**Feature Engineering 핵심 인사이트**")
    g = INSIGHT_GUIDES["fe"]
    for it in g["items"]:
        st.markdown(f"- {it}")

# ── Tab 3: 모델 비교 ──────────────────────────────────────────────
with tabs[3]:
    st.subheader("Mean ± Std 결과")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.subheader("Seed별 분포 (Box Plot)")
    metric_for_box = st.selectbox("지표 선택",
                                     options=[k for k in METRIC_KEYS if k != "accuracy"],
                                     format_func=lambda k: METRIC_LABELS[k],
                                     key="report_box_metric")
    st.plotly_chart(metric_box_plot(agg_df, metric_for_box), use_container_width=True)

    if "training_result_uw" in st.session_state:
        st.subheader("class_weight 보정 vs 미보정 비교")
        res_uw = st.session_state["training_result_uw"]
        agg_uw = aggregate_results(res_uw["raw"], res_uw["model_names"], res_uw["seeds"])
        delta_metric = st.selectbox("비교 지표",
                                       options=[k for k in METRIC_KEYS if k != "accuracy"],
                                       format_func=lambda k: METRIC_LABELS[k],
                                       key="report_delta_metric")
        st.plotly_chart(metric_delta_bar(agg_uw, agg_df, delta_metric),
                          use_container_width=True)

    st.subheader("통계 검정 — Friedman Test")
    rows = []
    for k in [k for k in METRIC_KEYS if k != "accuracy"]:
        f = friedman_test(raw, models, seeds, k)
        rows.append({
            "지표": METRIC_LABELS[k],
            "통계량": f.get("statistic"),
            "p-value": f.get("pvalue"),
            "결과": ("★ 유의 (p<0.05)" if f.get("pvalue") and f["pvalue"] < 0.05
                       else ("비유의" if f.get("pvalue") is not None else f.get("note", "—"))),
        })
    fri_table_df = pd.DataFrame(rows)
    col_t, col_g = st.columns([1, 1])
    with col_t:
        st.dataframe(fri_table_df, use_container_width=True, hide_index=True)
    with col_g:
        # p-value 막대 + 0.05 임계선
        fri_plot = fri_table_df[fri_table_df["p-value"].notna()].copy()
        if not fri_plot.empty:
            fri_plot["bar_color"] = fri_plot["p-value"].apply(
                lambda p: "#16a34a" if p < 0.05 else "#cbd5e1"
            )
            fig_fri = go.Figure(go.Bar(
                x=fri_plot["지표"], y=fri_plot["p-value"],
                marker_color=fri_plot["bar_color"],
                text=[f"p={p:.4f}" for p in fri_plot["p-value"]],
                textposition="outside",
            ))
            fig_fri.add_hline(y=0.05, line=dict(color="#dc2626", dash="dash"),
                              annotation_text="α=0.05", annotation_position="top right")
            fig_fri.update_layout(
                title="Friedman p-value (🟢 < 0.05 = 모델 차이 유의)",
                template="plotly_white", height=380, yaxis_title="p-value",
                margin=dict(l=50, r=20, t=60, b=40),
            )
            st.plotly_chart(fig_fri, use_container_width=True)

    st.subheader("통계 검정 — Pairwise Wilcoxon (Bonferroni)")
    metric_for_wil = st.selectbox("Wilcoxon 검정 지표",
                                     options=[k for k in METRIC_KEYS if k != "accuracy"],
                                     format_func=lambda k: METRIC_LABELS[k],
                                     key="report_wil_metric")
    wil_df = pairwise_wilcoxon(raw, models, seeds, metric_for_wil)
    if not wil_df.empty:
        wil_df = wil_df.copy()
        wil_df["유의"] = wil_df["pvalue_bonf"].apply(
            lambda p: "★" if p is not None and p < 0.05 else ""
        )
        col_t, col_g = st.columns([1, 1])
        with col_t:
            st.dataframe(wil_df, use_container_width=True, hide_index=True)
        with col_g:
            # 모델 쌍 × p-value Heatmap 형태
            wil_plot = wil_df.dropna(subset=["pvalue_bonf"]).copy()
            if not wil_plot.empty:
                wil_plot["pair"] = wil_plot["model_a"] + " vs " + wil_plot["model_b"]
                wil_plot = wil_plot.sort_values("pvalue_bonf")
                wil_plot["bar_color"] = wil_plot["pvalue_bonf"].apply(
                    lambda p: "#16a34a" if p < 0.05 else "#cbd5e1"
                )
                fig_wil = go.Figure(go.Bar(
                    x=wil_plot["pvalue_bonf"], y=wil_plot["pair"],
                    orientation="h",
                    marker_color=wil_plot["bar_color"],
                    text=[f"{p:.4f}" for p in wil_plot["pvalue_bonf"]],
                    textposition="outside",
                ))
                fig_wil.add_vline(x=0.05, line=dict(color="#dc2626", dash="dash"),
                                  annotation_text="α=0.05",
                                  annotation_position="top right")
                fig_wil.update_layout(
                    title=f"모델 쌍별 Bonferroni p-value ({METRIC_LABELS[metric_for_wil]})",
                    template="plotly_white",
                    height=max(380, 30 * len(wil_plot)),
                    xaxis_title="Bonferroni p-value",
                    margin=dict(l=180, r=80, t=60, b=40),
                )
                st.plotly_chart(fig_wil, use_container_width=True)

    st.subheader("Confusion Matrix & ROC")
    last_seed = seeds[-1]
    cm_model = st.selectbox("모델 선택", options=models, key="report_cm_model")
    y_test = st.session_state["fe_y_test"]
    y_pred = raw[last_seed][cm_model].get("y_pred")
    if y_pred is not None:
        labels = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
        cm = get_confusion_matrix(y_test, y_pred, labels=labels)
        st.plotly_chart(confusion_matrix_chart(cm, labels, f"Confusion Matrix — {cm_model}"),
                          use_container_width=True)

    prob_dict = {m: np.array(raw[last_seed][m]["y_prob"])
                  for m in models if raw[last_seed][m].get("y_prob") is not None}
    if prob_dict:
        st.plotly_chart(roc_curves_chart(y_test, prob_dict, classes,
                                              title=f"ROC Curves — seed {last_seed}"),
                          use_container_width=True)

    st.markdown("**모델 비교 핵심 인사이트**")
    for it in INSIGHT_GUIDES["model"]["items"]:
        st.markdown(f"- {it}")

# ── Tab 4: Feature Importance ─────────────────────────────────────
with tabs[4]:
    has_any = False
    if "imp_builtin_df" in st.session_state:
        has_any = True
        st.subheader("내장 중요도 (Built-in) Top-20")
        builtin_top = st.session_state["imp_builtin_df"].head(20)
        col_t, col_g = st.columns([1, 1])
        with col_t:
            st.dataframe(builtin_top, use_container_width=True, hide_index=True)
        with col_g:
            st.plotly_chart(
                importance_bar(builtin_top, "feature", "importance",
                                "Top-20 Built-in Importance"),
                use_container_width=True,
            )

    if "imp_perm_df" in st.session_state:
        has_any = True
        st.subheader("Permutation Importance Top-20")
        perm_top = st.session_state["imp_perm_df"].head(20)
        col_t, col_g = st.columns([1, 1])
        with col_t:
            st.dataframe(perm_top, use_container_width=True, hide_index=True)
        with col_g:
            err_col = "std" if "std" in perm_top.columns else None
            st.plotly_chart(
                importance_bar(perm_top, "feature", "importance",
                                "Top-20 Permutation Importance",
                                err_col=err_col),
                use_container_width=True,
            )
        st.caption("음수 값 = 셔플 후 성능 향상 → 노이즈 변수 가능성")

    if "shap_cache" in st.session_state and st.session_state["shap_cache"].get("shap_3d") is not None:
        has_any = True
        from utils.shap_utils import (
            shap_bar_chart,
            shap_class_heatmap,
            shap_summary_df,
        )
        s3d = st.session_state["shap_cache"]["shap_3d"]
        feature_names_now = st.session_state["fe_feature_names"]
        st.subheader("SHAP 요약 Top-20")
        shap_df_full = shap_summary_df(s3d, feature_names_now)
        shap_df = shap_df_full.head(20)

        # 표 + Mean |SHAP| 막대
        col_t, col_g = st.columns([1, 1])
        with col_t:
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
        with col_g:
            st.plotly_chart(shap_bar_chart(shap_df_full, top_n=20, signed=False),
                              use_container_width=True)

        # 부호 보존(방향성) 막대 — 🔵 양수 / 🔴 음수
        st.markdown("**🔵 방향성 분석 — 부호 보존 SHAP (🔵 양수=클래스↑ / 🔴 음수=클래스↓)**")
        st.plotly_chart(shap_bar_chart(shap_df_full, top_n=15, signed=True),
                          use_container_width=True)

        # 클래스 × Top Feature 히트맵
        if s3d.shape[0] >= 2:
            st.markdown("**🌡 클래스 × Top Feature SHAP 히트맵**")
            st.plotly_chart(
                shap_class_heatmap(s3d, feature_names_now, classes, top_n=12),
                use_container_width=True,
            )
            st.caption("🔵 파란 셀 = 그 클래스 예측 확률 **증가** 기여 / 🔴 빨간 셀 = **감소** 기여")

    if not has_any:
        st.info("Page 6에서 Feature Importance / SHAP 분석을 실행하면 결과가 여기에 통합됩니다.")
    else:
        st.markdown("**Feature Importance & SHAP 핵심 인사이트**")
        for it in INSIGHT_GUIDES["fi"]["items"]:
            st.markdown(f"- {it}")

# ── Tab 5: 도메인 관점 해석 ───────────────────────────────────────
with tabs[5]:
    st.subheader("📐 영역(도메인) 정의 — 사용자 정의 가능")
    st.caption("변수명 접두사를 영역명에 매핑하면, 중요 피처를 영역별로 그룹화한 해석을 자동 생성합니다.")

    if "custom_domain_map" not in st.session_state:
        st.session_state["custom_domain_map"] = dict(DEFAULT_DOMAIN_MAP)

    edit_cols = st.columns(2)
    with edit_cols[0]:
        st.markdown("**현재 매핑**")
        for prefix, info in st.session_state["custom_domain_map"].items():
            st.markdown(
                f"- <span style='background:{info['color']};color:white;padding:2px 6px;"
                f"border-radius:4px;font-size:11px;'>{prefix}</span> "
                f"&nbsp;**{info['name']}** — {info['desc']}",
                unsafe_allow_html=True,
            )
    with edit_cols[1]:
        with st.expander("➕ 매핑 추가/수정"):
            new_prefix = st.text_input("접두사 (예: t1, c, d)", key="dom_pfx")
            new_name = st.text_input("영역명 (예: 신기술, Cost)", key="dom_name")
            new_desc = st.text_input("설명", key="dom_desc",
                                       value="해당 영역에 대한 고객 평가")
            new_color = st.color_picker("색상", value="#6366f1", key="dom_color")
            if st.button("저장", key="dom_save") and new_prefix and new_name:
                st.session_state["custom_domain_map"][new_prefix] = {
                    "name": new_name, "color": new_color, "desc": new_desc,
                }
                st.rerun()

    st.markdown("**지수 정의**")
    for k, v in INDEX_DEFINITIONS.items():
        st.markdown(f"- **{k}**: {v}")

    # 영역별 Top Features 그룹화
    st.subheader("🏷 Top Features의 영역별 분포")
    feat_source_df = None
    if "imp_builtin_df" in st.session_state:
        feat_source_df = st.session_state["imp_builtin_df"].rename(columns={"importance": "score"})
    elif "shap_cache" in st.session_state and st.session_state["shap_cache"].get("shap_3d") is not None:
        from utils.shap_utils import shap_summary_df
        s3d = st.session_state["shap_cache"]["shap_3d"]
        feat_source_df = shap_summary_df(s3d, st.session_state["fe_feature_names"])
        feat_source_df = feat_source_df.rename(columns={"mean_abs_shap": "score"})

    if feat_source_df is not None:
        feat_source_df = feat_source_df.head(20).copy()

        def _detect(f):
            d = detect_domain(f, st.session_state["custom_domain_map"]) or {}
            return d.get("name", "기타"), d.get("color", "#94a3b8")

        feat_source_df[["영역", "_color"]] = feat_source_df["feature"].apply(
            lambda f: pd.Series(_detect(f))
        )

        col_t, col_g = st.columns([1, 1])
        with col_t:
            st.dataframe(feat_source_df[["feature", "영역", "score"]],
                          use_container_width=True, hide_index=True)
        with col_g:
            # 영역별 막대 차트 — 도메인 색상 적용
            agg_dom = (
                feat_source_df.groupby(["영역", "_color"])["score"].sum()
                .reset_index()
                .sort_values("score", ascending=True)
            )
            fig_dom = go.Figure(go.Bar(
                x=agg_dom["score"], y=agg_dom["영역"], orientation="h",
                marker_color=agg_dom["_color"],
                text=[f"{v:.4f}" for v in agg_dom["score"]],
                textposition="outside",
            ))
            fig_dom.update_layout(
                title="영역별 Top-20 기여도 합산",
                template="plotly_white", height=380,
                xaxis_title="기여도 합", margin=dict(l=120, r=80, t=60, b=40),
            )
            st.plotly_chart(fig_dom, use_container_width=True)

        # 도넛 차트 — 영역 비중
        st.markdown("**🍩 영역별 기여도 비중 (Donut)**")
        donut_data = feat_source_df.groupby("영역", as_index=False).agg(
            score=("score", "sum"), color=("_color", "first")
        ).sort_values("score", ascending=False)
        fig_donut = go.Figure(go.Pie(
            labels=donut_data["영역"],
            values=donut_data["score"],
            marker_colors=donut_data["color"],
            hole=0.55,
            textinfo="label+percent",
            insidetextorientation="radial",
        ))
        total_score = float(donut_data["score"].sum())
        fig_donut.update_layout(
            template="plotly_white", height=380,
            annotations=[dict(text=f"<b>{total_score:.3f}</b><br><span style='font-size:11px;'>Total</span>",
                              x=0.5, y=0.5, font_size=18, showarrow=False)],
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # 영역별 색상 적용한 Top-20 변수 막대
        st.markdown("**🏷 Top-20 변수 — 영역별 색상**")
        feat_plot = feat_source_df.iloc[::-1].copy()
        fig_feats = go.Figure(go.Bar(
            x=feat_plot["score"], y=feat_plot["feature"], orientation="h",
            marker_color=feat_plot["_color"],
            text=[f"{v:.4f}" for v in feat_plot["score"]],
            textposition="outside",
        ))
        fig_feats.update_layout(
            title="중요도 Top-20 (영역별 색상 구분)",
            template="plotly_white",
            height=max(420, 22 * len(feat_plot)),
            xaxis_title="기여도 점수",
            margin=dict(l=180, r=80, t=60, b=40),
        )
        st.plotly_chart(fig_feats, use_container_width=True)
    else:
        st.info("Page 6에서 중요도 분석을 먼저 실행하면 영역별 분포가 표시됩니다.")

# ── Tab 6: 추가 과제 제안 ─────────────────────────────────────────
with tabs[6]:
    st.markdown(
        "### 🔬 추가 과제 제안 — 4개 카테고리\n"
        "각 카테고리는 우선순위 (🔴 High / 🟡 Mid / 🟢 Low)로 분류된 후속 작업을 제안합니다."
    )

    cat_tabs = st.tabs([
        f"{ADDITIONAL_TASKS['eda']['icon']} EDA",
        f"{ADDITIONAL_TASKS['fe']['icon']} Feature Engineering",
        f"{ADDITIONAL_TASKS['model']['icon']} 모델 비교",
        f"{ADDITIONAL_TASKS['fi']['icon']} Feature Importance & SHAP",
    ])

    def render_tasks_streamlit(category: str):
        cat = ADDITIONAL_TASKS[category]
        st.subheader(f"{cat['icon']} {cat['title']}")
        for level, color, label in [
            ("high", "#dc2626", "🔴 High Priority — 즉시 착수"),
            ("mid",  "#d97706", "🟡 Mid Priority — 1~2개월"),
            ("low",  "#16a34a", "🟢 Low Priority / 탐색적"),
        ]:
            if not cat.get(level):
                continue
            st.markdown(f"<h4 style='color:{color};'>{label}</h4>",
                          unsafe_allow_html=True)
            df_tasks = pd.DataFrame(cat[level]).rename(columns={
                "name": "과제", "background": "배경",
                "method": "방법", "outcome": "기대 효과",
            })
            st.dataframe(df_tasks, use_container_width=True, hide_index=True)
        if "checklist" in cat:
            st.markdown("**☑ 운영 모델 배포 체크리스트**")
            for it in cat["checklist"]:
                st.markdown(f"- ☐ {it}")

    with cat_tabs[0]:
        render_tasks_streamlit("eda")
    with cat_tabs[1]:
        render_tasks_streamlit("fe")
    with cat_tabs[2]:
        render_tasks_streamlit("model")
    with cat_tabs[3]:
        render_tasks_streamlit("fi")

st.divider()

# ════════════════════════════════════════════════════════════════════
# 종합 HTML 리포트 생성 함수
# ════════════════════════════════════════════════════════════════════

def build_full_html() -> str:
    """모든 탭 콘텐츠를 포함한 종합 HTML 리포트."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    summary_html = summary_table.to_html(index=False, classes="report-table", border=0)
    perf_html    = table_df.to_html(index=False, classes="report-table", border=0)

    # EDA 섹션
    eda_parts = []
    if df is not None and target:
        cls = df[target].value_counts().sort_index()
        eda_parts.append(
            "<h3>타겟 분포</h3>"
            + cls.rename("count").to_frame().to_html(classes="report-table", border=0)
        )
    if "eda_kw_df" in st.session_state:
        kw_df = st.session_state["eda_kw_df"].head(20)
        eda_parts.append(
            "<h3>Kruskal-Wallis 단변량 검정 (Top 20)</h3>"
            + kw_df.to_html(index=False, classes="report-table", border=0)
        )
    eda_parts.append(insight_to_html("eda"))
    eda_html = "\n".join(eda_parts)

    # FE 섹션
    fe_parts = ["<h3>적용 전략</h3>",
                 pd.DataFrame([
                     {"항목": "전략", "값": str(strategy)},
                     {"항목": "변환 후 피처 수", "값": str(len(fe_feat))},
                 ]).to_html(index=False, classes="report-table", border=0)]
    if fe_feat:
        fe_parts.append(
            f"<p><strong>사용 피처:</strong> {', '.join(fe_feat[:30])}"
            + ("..." if len(fe_feat) > 30 else "") + "</p>"
        )
    fe_parts.append(insight_to_html("fe"))
    fe_html = "\n".join(fe_parts)

    # 모델 비교 섹션
    mc_parts = [
        "<h3>모델 성능 비교</h3>", perf_html,
    ]
    # Friedman
    fri_rows = []
    for k in [k for k in METRIC_KEYS if k != "accuracy"]:
        f = friedman_test(raw, models, seeds, k)
        fri_rows.append({
            "지표": METRIC_LABELS[k],
            "통계량": f.get("statistic"),
            "p-value": f.get("pvalue"),
            "유의": ("★" if f.get("pvalue") and f["pvalue"] < 0.05 else ""),
        })
    mc_parts.append("<h3>Friedman Test</h3>"
                     + pd.DataFrame(fri_rows).to_html(index=False, classes="report-table", border=0))
    # Wilcoxon (Macro F1 기준)
    wil_df = pairwise_wilcoxon(raw, models, seeds, "macro_f1")
    if not wil_df.empty:
        wil_df = wil_df.copy()
        wil_df["유의"] = wil_df["pvalue_bonf"].apply(
            lambda p: "★" if p is not None and p < 0.05 else ""
        )
        mc_parts.append("<h3>Pairwise Wilcoxon (Macro F1, Bonferroni)</h3>"
                         + wil_df.to_html(index=False, classes="report-table", border=0))
    mc_parts.append(insight_to_html("model"))
    mc_html = "\n".join(mc_parts)

    # Feature Importance 섹션
    fi_parts = []
    if "imp_builtin_df" in st.session_state:
        fi_parts.append("<h3>내장 중요도 Top-20</h3>"
                         + st.session_state["imp_builtin_df"].head(20)
                         .to_html(index=False, classes="report-table", border=0))
    if "imp_perm_df" in st.session_state:
        fi_parts.append("<h3>Permutation Importance Top-20</h3>"
                         + st.session_state["imp_perm_df"].head(20)
                         .to_html(index=False, classes="report-table", border=0))
    if "shap_cache" in st.session_state and st.session_state["shap_cache"].get("shap_3d") is not None:
        from utils.shap_utils import shap_summary_df
        s3d = st.session_state["shap_cache"]["shap_3d"]
        shap_df = shap_summary_df(s3d, st.session_state["fe_feature_names"]).head(20)
        fi_parts.append("<h3>SHAP Top-20</h3>"
                         + shap_df.to_html(index=False, classes="report-table", border=0))
    fi_parts.append(insight_to_html("fi"))
    fi_html = "\n".join(fi_parts) if fi_parts else "<p>중요도 분석 미실행</p>"

    # 도메인 섹션
    domain_html = domain_legend_html(st.session_state.get("custom_domain_map", DEFAULT_DOMAIN_MAP))

    # 추가 과제
    tasks_html = (
        tasks_to_html("eda")   + tasks_to_html("fe")
        + tasks_to_html("model") + tasks_to_html("fi")
    )

    top_features_text = st.session_state.get("_report_top_features_text", "—")

    return f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="UTF-8"><title>LGD CSAT — ML 종합 리포트</title>
<style>
  body {{ font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;
         background:#f1f5f9; color:#1e293b; max-width:1200px;
         margin:30px auto; padding:0 24px; }}
  .header {{ background:linear-gradient(135deg,#4f46e5,#9333ea,#ec4899); color:white;
              padding:28px 34px; border-radius:14px; margin-bottom:24px;
              box-shadow:0 6px 20px rgba(79,70,229,.25); }}
  .header .brand {{ font-size:11px; font-weight:700; letter-spacing:2px;
                     text-transform:uppercase; color:#e0e7ff; margin-bottom:6px; }}
  .header h1 {{ margin:0 0 6px; font-size:26px; }}
  .header p  {{ margin:0; opacity:.92; font-size:13px; color:#ddd6fe; }}
  .card {{ background:white; padding:22px 28px; border-radius:12px;
           margin-bottom:20px; box-shadow:0 1px 6px rgba(0,0,0,.06); }}
  h2 {{ font-size:18px; color:#1e293b; border-bottom:2px solid #e2e8f0;
       padding-bottom:8px; margin:0 0 14px; }}
  h3 {{ font-size:14px; color:#475569; margin:14px 0 8px; }}
  table.report-table {{ width:100%; border-collapse:collapse; font-size:13px;
                         margin-bottom:8px; }}
  table.report-table th, table.report-table td {{
    padding:8px 10px; border-bottom:1px solid #e2e8f0; text-align:left;
  }}
  table.report-table th {{ background:#f8fafc; font-weight:700; }}
  .kpi {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:8px 0 14px; }}
  .kpi div {{ background:#f8fafc; border-radius:8px; padding:14px 16px;
              border-left:4px solid #6366f1; }}
  .kpi b {{ font-size:22px; color:#4f46e5; display:block; }}
  ul {{ margin-left:18px; }}
</style></head>
<body>
<div class="header">
  <div class="brand">📊 LG Display</div>
  <h1>Customer Satisfaction Survey Analysis Platform</h1>
  <p>EDA · Feature Engineering · Model Comparison · Feature Importance · 도메인 해석 · 추가 과제 제안 통합</p>
  <p>생성일: {timestamp}</p>
</div>

<div class="card">
  <h2>핵심 KPI</h2>
  <div class="kpi">
    <div><b>{best_f1_val:.4f}</b><small>최고 Macro F1 ({best_f1_model})</small></div>
    <div><b>{best_au_val:.4f}</b><small>최고 Macro AUROC ({best_au_model})</small></div>
    <div><b>{len(models)}</b><small>학습 모델</small></div>
    <div><b>{len(seeds)}</b><small>반복 시드</small></div>
  </div>
  <h3>분석 설정</h3>
  {summary_html}
  <h3>Top Features</h3>
  <p>{top_features_text}</p>
</div>

<div class="card">
  <h2>🔍 1. EDA 결과</h2>
  {eda_html}
</div>

<div class="card">
  <h2>🛠 2. Feature Engineering 결과</h2>
  {fe_html}
</div>

<div class="card">
  <h2>🤖 3. 모델 비교 결과</h2>
  {mc_html}
</div>

<div class="card">
  <h2>🎯 4. Feature Importance & SHAP 결과</h2>
  {fi_html}
</div>

<div class="card">
  <h2>🏭 5. 도메인 관점 해석</h2>
  {domain_html}
</div>

<div class="card" style="border-left:5px solid #f43f5e;">
  <h2>🔬 6. 추가 과제 제안</h2>
  <p style="font-size:12px; color:#64748b;">
    분석 결과를 기반으로 후속 진행할 작업을 우선순위별로 제안합니다 (🔴 High / 🟡 Mid / 🟢 Low).
  </p>
  {tasks_html}
</div>

</body></html>"""


# ════════════════════════════════════════════════════════════════════
# 다운로드 버튼 (HTML / Excel / PPTx)
# ════════════════════════════════════════════════════════════════════
st.subheader("💾 리포트 다운로드")
st.caption(
    "동일한 리포트 내용을 HTML(웹) · Excel(다중 시트) · PPTx(프레젠테이션) "
    "3가지 포맷으로 출력할 수 있습니다."
)
col_a, col_b, col_c = st.columns(3)

with col_a:
    html_full = build_full_html()
    st.download_button(
        "📥 HTML 리포트 다운로드",
        data=html_full.encode("utf-8"),
        file_name=f"ml_full_report_{datetime.now():%Y%m%d_%H%M}.html",
        mime="text/html",
        type="primary",
        use_container_width=True,
        help="모든 섹션 + 추가 과제 통합 HTML",
    )

with col_b:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_table.to_excel(writer, sheet_name="설정", index=False)
        table_df.to_excel(writer, sheet_name="모델성능", index=False)
        # 추가 과제 시트
        all_tasks_rows = []
        for cat_key, cat in ADDITIONAL_TASKS.items():
            for level in ["high", "mid", "low"]:
                for it in cat.get(level, []):
                    all_tasks_rows.append({
                        "카테고리": cat["title"],
                        "우선순위": level.upper(),
                        "과제": it["name"],
                        "배경": it["background"],
                        "방법": it["method"],
                        "기대 효과": it["outcome"],
                    })
        pd.DataFrame(all_tasks_rows).to_excel(writer, sheet_name="추가과제", index=False)

        if "imp_builtin_df" in st.session_state:
            st.session_state["imp_builtin_df"].to_excel(
                writer, sheet_name="Builtin_Importance", index=False
            )
        if "imp_perm_df" in st.session_state:
            st.session_state["imp_perm_df"].to_excel(
                writer, sheet_name="Permutation_Importance", index=False
            )
        if "shap_cache" in st.session_state and st.session_state["shap_cache"].get("shap_3d") is not None:
            from utils.shap_utils import shap_summary_df
            s3d = st.session_state["shap_cache"]["shap_3d"]
            shap_summary_df(s3d, st.session_state["fe_feature_names"]).to_excel(
                writer, sheet_name="SHAP_Summary", index=False
            )
        if "eda_kw_df" in st.session_state:
            st.session_state["eda_kw_df"].to_excel(writer, sheet_name="KW_Test", index=False)
    output.seek(0)
    st.download_button(
        "📥 Excel 리포트 다운로드",
        data=output.getvalue(),
        file_name=f"ml_full_report_{datetime.now():%Y%m%d_%H%M}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        help="다중 시트 (설정/성능/추가과제/중요도/SHAP/KW)",
    )

with col_c:
    # ── PPTx 빌더에 넘길 데이터 준비 ────────────────────────────
    kpi_pptx = {
        "best_f1_val": best_f1_val,
        "best_f1_model": best_f1_model,
        "best_au_val": best_au_val,
        "best_au_model": best_au_model,
        "n_models": len(models),
        "n_seeds": len(seeds),
    }

    # FE info 표
    fe_info_df_pptx = pd.DataFrame([
        {"항목": "전략", "값": str(strategy)},
        {"항목": "변환 후 피처 수", "값": str(len(fe_feat))},
        {"항목": "Train shape",
         "값": str(st.session_state["fe_X_train"].shape) if "fe_X_train" in st.session_state else "—"},
        {"항목": "Test shape",
         "값": str(st.session_state["fe_X_test"].shape) if "fe_X_test" in st.session_state else "—"},
    ])

    # Friedman 표 (PPT용)
    fri_pptx_rows = []
    for k in [k for k in METRIC_KEYS if k != "accuracy"]:
        f = friedman_test(raw, models, seeds, k)
        pval = f.get("pvalue")
        fri_pptx_rows.append({
            "지표": METRIC_LABELS[k],
            "통계량": f"{f['statistic']:.4f}" if f.get("statistic") is not None else "—",
            "p-value": f"{pval:.4f}" if pval is not None else "—",
            "결과": ("★ 유의 (p<0.05)" if pval is not None and pval < 0.05
                       else ("비유의" if pval is not None else "—")),
        })
    friedman_df_pptx = pd.DataFrame(fri_pptx_rows)

    # Wilcoxon (Macro F1 기준)
    wil_full = pairwise_wilcoxon(raw, models, seeds, "macro_f1")
    if not wil_full.empty:
        wil_pptx = wil_full.copy()
        # 숫자형 round
        for c in ["statistic", "pvalue", "pvalue_bonf"]:
            if c in wil_pptx.columns:
                wil_pptx[c] = wil_pptx[c].apply(
                    lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "—"
                )
        wil_pptx["유의"] = wil_full["pvalue_bonf"].apply(
            lambda p: "★" if isinstance(p, (int, float)) and p is not None and p < 0.05 else ""
        )
    else:
        wil_pptx = None

    # Importance / SHAP
    importance_df_pptx = None
    shap_df_pptx = None
    if "imp_builtin_df" in st.session_state:
        importance_df_pptx = st.session_state["imp_builtin_df"].head(15).copy()
        if "importance" in importance_df_pptx.columns:
            importance_df_pptx["importance"] = importance_df_pptx["importance"].apply(
                lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v
            )
    if "shap_cache" in st.session_state and st.session_state["shap_cache"].get("shap_3d") is not None:
        from utils.shap_utils import shap_summary_df
        s3d = st.session_state["shap_cache"]["shap_3d"]
        shap_df_pptx = shap_summary_df(s3d, st.session_state["fe_feature_names"]).head(15).copy()
        for c in ["mean_abs_shap", "mean_shap"]:
            if c in shap_df_pptx.columns:
                shap_df_pptx[c] = shap_df_pptx[c].apply(
                    lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v
                )

    # KW (EDA)
    kw_for_pptx = None
    if "eda_kw_df" in st.session_state:
        kw_for_pptx = st.session_state["eda_kw_df"].head(15).copy()
        for c in ["statistic", "p-value", "eta_squared"]:
            if c in kw_for_pptx.columns:
                kw_for_pptx[c] = kw_for_pptx[c].apply(
                    lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v
                )

    top_features_text = st.session_state.get("_report_top_features_text", "—")

    try:
        pptx_bytes = build_pptx_report(
            kpi=kpi_pptx,
            settings_df=summary_table,
            perf_df=table_df,
            eda_kw_df=kw_for_pptx,
            fe_info_df=fe_info_df_pptx,
            friedman_df=friedman_df_pptx,
            wilcoxon_df=wil_pptx,
            importance_df=importance_df_pptx,
            shap_df=shap_df_pptx,
            insights=INSIGHT_GUIDES,
            additional_tasks=ADDITIONAL_TASKS,
            top_features_text=top_features_text,
        )
        st.download_button(
            "📥 PPTx 리포트 다운로드",
            data=pptx_bytes,
            file_name=f"ml_full_report_{datetime.now():%Y%m%d_%H%M}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True,
            help="16:9 프레젠테이션 — 타이틀/KPI/EDA/FE/모델/FI/도메인/추가과제 슬라이드",
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"PPTx 생성 실패: {e}")
        st.caption("python-pptx 라이브러리가 정상 설치되었는지 확인해주세요 (pip install python-pptx).")

# ════════════════════════════════════════════════════════════════════
# 배치 예측
# ════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📦 배치 예측 — 신규 데이터로 예측 결과 다운로드")
st.caption(
    "현재 학습된 모델 중 하나를 선택해 동일 컬럼 구조의 신규 데이터에 적용합니다. "
    "(PCA 전략은 별도 PCA 변환이 필요해 미지원)"
)

batch_file = st.file_uploader(
    "예측할 신규 데이터 (CSV / Excel)",
    type=["csv", "xlsx", "xls"], key="batch_upload",
)
batch_model_name = st.selectbox("예측 모델", options=models, key="batch_model")

if batch_file is not None:
    try:
        new_df = load_uploaded_file(batch_file)
        st.write(f"파일 로드 완료 — {new_df.shape}")
        st.dataframe(new_df.head(), use_container_width=True)

        original_features = st.session_state.get("feature_cols", [])
        missing = [c for c in original_features if c not in new_df.columns]
        if missing:
            st.error(f"필수 컬럼 누락: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        else:
            cat_cols = st.session_state.get("cat_cols", [])
            new_df_proc = new_df[original_features].copy()
            if cat_cols:
                new_df_proc, _ = encode_categorical(new_df_proc, cat_cols, method="label")

            strategy_now = st.session_state.get("fe_strategy")
            X_new = None
            if strategy_now == "pca":
                st.warning("PCA 전략은 학습 시 PCA 객체로 변환이 필요합니다 (현재 미지원).")
            elif strategy_now in ("selection", "manual"):
                fe_feat_names = st.session_state.get("fe_feature_names", [])
                if all(f in new_df_proc.columns for f in fe_feat_names):
                    X_new = new_df_proc[fe_feat_names].values
                else:
                    st.warning("학습 시 사용한 피처 일부가 누락됐습니다.")
            else:
                X_new = new_df_proc.values

            if X_new is not None:
                model = res["last_models"][batch_model_name]
                pred = model.predict(X_new)
                out = new_df.copy()
                out["predicted_label"] = pred
                try:
                    proba = model.predict_proba(X_new)
                    for ci, c in enumerate(classes):
                        if ci < proba.shape[1]:
                            out[f"prob_{c}"] = proba[:, ci]
                except Exception:
                    pass
                st.success(f"✅ 예측 완료 — {len(out)}건")
                st.dataframe(out.head(20), use_container_width=True)
                csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 예측 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"predictions_{datetime.now():%Y%m%d_%H%M}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    except Exception as e:
        st.error(f"예측 실패: {e}")
