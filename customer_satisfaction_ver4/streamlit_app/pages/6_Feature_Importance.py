"""Page 6 — Feature Importance & SHAP 분석 + 위험 고객 시뮬레이터."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.insight_utils import render_fi_insights
from utils.progress_utils import render_step_progress
from utils.shap_utils import (
    SHAP_AVAILABLE,
    compute_shap,
    shap_bar_chart,
    shap_beeswarm,
    shap_class_heatmap,
    shap_dependence,
    shap_summary_df,
    shap_waterfall,
)
from utils.viz_utils import importance_bar

st.set_page_config(page_title="LGD CSAT — 6. Feature Importance", page_icon="🎯", layout="wide")
render_step_progress(current_step=6)
st.title("🎯 6. Feature Importance & SHAP")

if "training_result" not in st.session_state:
    st.warning("⬅ 먼저 **4_Model_Training** 페이지에서 모델을 학습해주세요.")
    st.stop()

res          = st.session_state["training_result"]
last_models  = res["last_models"]
models       = res["model_names"]
classes      = res["all_classes"]
feature_names = st.session_state["fe_feature_names"]
X_train      = st.session_state["fe_X_train"]
X_test       = st.session_state["fe_X_test"]
y_test       = st.session_state["fe_y_test"]

# ── 분석 방법 / 모델 선택 ────────────────────────────────────────
st.subheader("분석 방법 및 모델 선택")
c1, c2 = st.columns([1, 2])
with c1:
    method = st.radio(
        "중요도 방법",
        options=["builtin", "permutation", "shap"],
        format_func=lambda x: {
            "builtin": "내장 중요도 (모델 자체)",
            "permutation": "Permutation Importance",
            "shap": "SHAP Value",
        }[x],
    )
with c2:
    model_for_imp = st.selectbox("분석할 모델", options=models, key="imp_model")

mdl = last_models[model_for_imp]

# ── 내장 중요도 ──────────────────────────────────────────────────
if method == "builtin":
    st.subheader(f"📊 내장 중요도 — {model_for_imp}")
    if hasattr(mdl, "feature_importances_"):
        imp = mdl.feature_importances_
    elif hasattr(mdl, "coef_"):
        imp = np.abs(mdl.coef_).mean(axis=0) if mdl.coef_.ndim > 1 else np.abs(mdl.coef_)
    elif hasattr(mdl, "booster_"):
        imp = mdl.booster_.feature_importance(importance_type="gain")
    else:
        st.error("이 모델은 내장 중요도를 지원하지 않습니다.")
        st.stop()
    df_imp = pd.DataFrame({"feature": feature_names, "importance": imp})
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    top_n = st.slider("Top-N", 5, len(df_imp), min(20, len(df_imp)), key="builtin_n")
    st.plotly_chart(
        importance_bar(df_imp.head(top_n), "feature", "importance",
                        f"Top-{top_n} Built-in Importance ({model_for_imp})"),
        use_container_width=True,
    )
    st.dataframe(df_imp.head(top_n), use_container_width=True)
    st.session_state["imp_builtin_df"] = df_imp

# ── Permutation 중요도 ───────────────────────────────────────────
elif method == "permutation":
    st.subheader(f"🔀 Permutation Importance — {model_for_imp}")
    n_repeats = st.slider("반복 횟수", 5, 30, 10)
    if st.button("Permutation 계산 실행", type="primary"):
        with st.spinner("Permutation Importance 계산 중..."):
            r = permutation_importance(mdl, X_test, y_test,
                                          n_repeats=n_repeats, random_state=42,
                                          scoring="f1_macro", n_jobs=-1)
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": r.importances_mean,
            "std": r.importances_std,
        })
        df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
        st.session_state["imp_perm_df"] = df_imp

    if "imp_perm_df" in st.session_state:
        df_imp = st.session_state["imp_perm_df"]
        top_n = st.slider("Top-N", 5, len(df_imp), min(20, len(df_imp)), key="perm_n")
        st.plotly_chart(
            importance_bar(df_imp.head(top_n), "feature", "importance",
                            f"Top-{top_n} Permutation Importance ({model_for_imp})",
                            err_col="std"),
            use_container_width=True,
        )
        st.dataframe(df_imp.head(top_n), use_container_width=True)
        st.caption("※ 음수값 = 셔플 후 성능 향상 → 노이즈 변수일 가능성")

# ── SHAP 분석 ───────────────────────────────────────────────────
elif method == "shap":
    st.subheader(f"🔮 SHAP 분석 — {model_for_imp}")
    if not SHAP_AVAILABLE:
        st.error("`shap` 패키지가 설치되지 않았습니다. `pip install shap`")
        st.stop()

    cache = st.session_state.get("shap_cache", {})
    cache_key = (model_for_imp,)

    if cache.get("key") != cache_key:
        if st.button("SHAP 값 계산 실행", type="primary"):
            with st.spinner("SHAP 계산 중 (트리 모델은 빠름, 선형은 매우 빠름)..."):
                # background 샘플 — train 일부 사용 (속도 / 정확도 균형)
                bg = X_train[:200] if len(X_train) > 200 else X_train
                shap_3d = compute_shap(mdl, X_test, bg, model_for_imp)
            st.session_state["shap_cache"] = {
                "key": cache_key,
                "shap_3d": shap_3d,
            }
            st.rerun()
    else:
        shap_3d = cache["shap_3d"]
        n_classes = shap_3d.shape[0]
        summary_df = shap_summary_df(shap_3d, feature_names)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Summary Bar", "Beeswarm", "Class 방향", "Dependence", "Waterfall (개별)",
        ])

        with tab1:
            top_n = st.slider("Top-N", 5, min(40, len(feature_names)), 20, key="shap_bar_n")
            signed_view = st.toggle("부호 보존 보기 (🔵 양성=파랑 / 🔴 음성=빨강)", value=False)
            st.plotly_chart(
                shap_bar_chart(summary_df, top_n=top_n, signed=signed_view),
                use_container_width=True,
            )
            if signed_view:
                st.caption(
                    "🔵 **파란 막대 (양성, +)** — 해당 피처가 클래스 예측 확률을 **올리는 방향**으로 기여 / "
                    "🔴 **빨간 막대 (음성, −)** — 예측 확률을 **내리는 방향**으로 기여"
                )
            st.dataframe(summary_df.head(top_n), use_container_width=True)

        with tab2:
            top_n_bw = st.slider("Top-N", 5, min(30, len(feature_names)), 15, key="bw_n")
            st.plotly_chart(
                shap_beeswarm(shap_3d, X_test, feature_names, top_n=top_n_bw),
                use_container_width=True,
            )
            st.caption(
                "ℹ️ Beeswarm은 색이 **피처 값의 크기**를 의미합니다(파랑=낮음, 빨강=높음). "
                "SHAP 부호 색상은 **Summary Bar / Class 방향 / Waterfall** 탭을 참고하세요 — "
                "이 탭들은 🔵 양성=파랑, 🔴 음성=빨강으로 통일되어 있습니다."
            )

        with tab3:
            if n_classes < 2:
                st.info("이진 분류 또는 회귀라 클래스별 방향성을 표시할 수 없습니다.")
            else:
                top_n_h = st.slider("Top-N", 5, min(25, len(feature_names)), 12, key="heat_n")
                st.plotly_chart(
                    shap_class_heatmap(shap_3d, feature_names, classes, top_n=top_n_h),
                    use_container_width=True,
                )
                st.caption(
                    "🔵 **파란 셀 (양성, +)** — 해당 피처가 그 클래스의 예측 확률을 **증가**시키는 기여 / "
                    "🔴 **빨간 셀 (음성, −)** — **감소**시키는 기여"
                )

        with tab4:
            feat_for_dep = st.selectbox("피처 선택", options=summary_df["feature"].tolist(),
                                          key="dep_feat")
            st.plotly_chart(
                shap_dependence(shap_3d, X_test, feature_names, feat_for_dep, classes),
                use_container_width=True,
            )

        with tab5:
            sample_idx = st.number_input("샘플 인덱스 (Test set)", min_value=0,
                                            max_value=len(X_test) - 1, value=0)
            cls_idx = st.selectbox(
                "클래스 인덱스",
                options=list(range(n_classes)),
                format_func=lambda i: f"#{i} (label={classes[i] if i < len(classes) else i})",
            )
            top_n_w = st.slider("Top-N", 5, 20, 10, key="wf_n")
            st.plotly_chart(
                shap_waterfall(shap_3d, sample_idx=int(sample_idx),
                                class_idx=int(cls_idx),
                                feature_names=feature_names,
                                feature_values=X_test[int(sample_idx)],
                                top_n=top_n_w),
                use_container_width=True,
            )
            actual = y_test[int(sample_idx)] if sample_idx < len(y_test) else "?"
            st.caption(f"실제 클래스: **{actual}** | 예측 클래스 인덱스: **{cls_idx}**")

# ════════════════════════════════════════════════════════════════════
# 💡 가장 중요한 결과 정리 (비전공자 친화)
# ════════════════════════════════════════════════════════════════════
_imp_df_for_insight = st.session_state.get("imp_builtin_df")
if method == "permutation" and "imp_perm_df" in st.session_state:
    _imp_df_for_insight = st.session_state["imp_perm_df"]
_shap_for_insight = None
if (
    "shap_cache" in st.session_state
    and st.session_state["shap_cache"].get("shap_3d") is not None
):
    s3d = st.session_state["shap_cache"]["shap_3d"]
    _shap_for_insight = shap_summary_df(s3d, feature_names)

if _imp_df_for_insight is not None or _shap_for_insight is not None:
    render_fi_insights(
        method=method,
        model_name=model_for_imp,
        imp_df=_imp_df_for_insight,
        shap_summary=_shap_for_insight,
        n_features=len(feature_names),
        classes=classes,
    )

# ── 위험 고객 탐지 시뮬레이터 ────────────────────────────────────
st.divider()
st.subheader("🎯 위험 고객 탐지 시뮬레이터 (실시간 예측)")
st.caption("아래 슬라이더로 피처 값을 조정하면 학습된 모델들의 예측 확률이 즉시 업데이트됩니다.")

with st.expander("시뮬레이터 열기"):
    sim_n_features = st.slider(
        "조정할 피처 수 (중요도 상위)",
        2, min(15, len(feature_names)),
        min(6, len(feature_names)),
    )
    # 중요도 기준 상위 피처 식별
    if "imp_builtin_df" in st.session_state:
        order_features = st.session_state["imp_builtin_df"]["feature"].tolist()
    elif "shap_cache" in st.session_state and st.session_state["shap_cache"].get("shap_3d") is not None:
        s3d = st.session_state["shap_cache"]["shap_3d"]
        order_features = (
            shap_summary_df(s3d, feature_names)["feature"].tolist()
        )
    else:
        order_features = list(feature_names)
    sim_features = order_features[:sim_n_features]
    sim_features_idx = [feature_names.index(f) for f in sim_features]

    # 슬라이더 — 각 피처의 min/max 범위
    base_sample = X_test[0].astype(float).copy()
    sim_input = base_sample.copy()
    cols = st.columns(min(3, sim_n_features))
    for i, feat in enumerate(sim_features):
        idx = sim_features_idx[i]
        col = cols[i % len(cols)]
        with col:
            mn = float(np.min(X_train[:, idx]))
            mx = float(np.max(X_train[:, idx]))
            cur = float(np.mean(X_train[:, idx]))
            step = max((mx - mn) / 100, 1e-3)
            sim_input[idx] = st.slider(feat, mn, mx, cur, step, key=f"sim_{feat}")

    sim_X = sim_input.reshape(1, -1)
    rows = []
    for name, m in last_models.items():
        try:
            prob = m.predict_proba(sim_X)[0]
            pred = m.predict(sim_X)[0]
            row = {"모델": name, "예측 클래스": str(pred)}
            for ci, c in enumerate(classes):
                row[f"P({c})"] = round(float(prob[ci]) if ci < len(prob) else 0, 4)
            rows.append(row)
        except Exception as e:
            rows.append({"모델": name, "예측 클래스": "오류", "에러": str(e)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
