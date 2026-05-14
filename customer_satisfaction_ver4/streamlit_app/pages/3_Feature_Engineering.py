"""Page 3 — Feature Engineering 전략 선택."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.data_utils import split_train_test
from utils.insight_utils import render_fe_insights
from utils.progress_utils import render_step_progress
from utils.viz_utils import scree_plot

st.set_page_config(page_title="LGD CSAT — 3. Feature Engineering", page_icon="🛠", layout="wide")
render_step_progress(current_step=3)
st.title("🛠 3. Feature Engineering — 전략 선택")

if "df" not in st.session_state:
    st.warning("⬅ 먼저 **1_Upload** 페이지에서 데이터를 업로드해주세요.")
    st.stop()

df       = st.session_state["df"]
target   = st.session_state["target"]
features = st.session_state["feature_cols"]

# ── Train/Test 분할 옵션 ─────────────────────────────────────────
st.subheader("Train / Test 분할")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    test_size = st.slider("Test 비율", 0.1, 0.5, 0.2, 0.05)
with col2:
    split_method = st.radio("분할 방식", ["random", "time"],
                              format_func=lambda x: "랜덤" if x == "random" else "시계열 기준")
with col3:
    time_col = None
    stratify = True
    if split_method == "time":
        time_candidates = [c for c in df.columns if c not in features]
        if time_candidates:
            time_col = st.selectbox("정렬 기준 컬럼", options=time_candidates)
        else:
            st.warning("시계열 분할에 사용할 컬럼이 없습니다. (예: year)")
    else:
        stratify = st.checkbox("Stratify (클래스 비율 유지)", value=True)

random_state = st.number_input("랜덤 시드 (분할용)", value=42, min_value=0, step=1)

# ── 전략 선택 ────────────────────────────────────────────────────
st.subheader("피처 엔지니어링 전략")
strategy = st.radio(
    "전략 선택",
    options=["all", "selection", "pca", "manual"],
    format_func=lambda x: {
        "all":       "Exp 1 — 전체 피처 사용",
        "selection": "Exp 2 — Mutual Information + Pearson 자동 선택",
        "pca":       "Exp 3 — PCA 차원 축소",
        "manual":    "Exp 4 — 사용자 직접 선택",
    }[x],
    horizontal=False,
)

apply_scaler = st.toggle(
    "StandardScaler 적용 (Logistic Regression 권장)", value=True,
    help="all/selection/manual 전략에서 적용. PCA는 내부적으로 적용됩니다.",
)

selected_features = features.copy()
pca_obj = None
explained_var = None
fe_extra_info: dict = {}

if strategy == "selection":
    cc1, cc2 = st.columns(2)
    with cc1:
        mi_top = st.slider("MI 상위 N개 선택", 5, len(features), min(30, len(features)))
    with cc2:
        corr_thr = st.slider("Pearson |r| 컷 (다중공선성)", 0.7, 0.99, 0.95, 0.01)
    st.caption("Mutual Information 상위 N개 → 그 중 Pearson |r| ≥ 컷으로 묶이는 변수 중 1개만 잔존")

elif strategy == "pca":
    pca_mode = st.radio("PCA 컴포넌트 결정 방식",
                          ["분산 설명률 기준", "고정 컴포넌트 수"], horizontal=True)
    if pca_mode == "분산 설명률 기준":
        var_ratio = st.slider("누적 분산 설명률", 0.5, 0.99, 0.95, 0.01)
        n_components = var_ratio
    else:
        n_components = st.slider("Components 수", 2, min(50, len(features)),
                                   min(25, len(features)))

elif strategy == "manual":
    selected_features = st.multiselect(
        "포함할 피처 선택", options=features,
        default=features[: min(15, len(features))],
    )

# ── 적용 버튼 ────────────────────────────────────────────────────
if st.button("✅ 전략 적용 & 학습 데이터 준비", type="primary", use_container_width=True):
    train_df, test_df = split_train_test(
        df, target=target, test_size=test_size, method=split_method,
        time_col=time_col, stratify=stratify, random_state=int(random_state),
    )
    X_train_raw = train_df[features].values
    X_test_raw  = test_df[features].values
    y_train     = train_df[target].values
    y_test      = test_df[target].values

    if strategy == "all":
        feat_names = features
        if apply_scaler:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_train_raw)
            X_te = sc.transform(X_test_raw)
        else:
            X_tr, X_te = X_train_raw, X_test_raw

    elif strategy == "selection":
        # 1) MI Top-N
        mi_scores = mutual_info_classif(X_train_raw, y_train, random_state=int(random_state))
        mi_order = np.argsort(mi_scores)[::-1]
        mi_top_idx = mi_order[:mi_top]
        # 2) 다중공선성 제거
        sub_df = pd.DataFrame(X_train_raw[:, mi_top_idx],
                                columns=[features[i] for i in mi_top_idx])
        keep_idx = []
        for i in range(len(mi_top_idx)):
            keep = True
            for j in keep_idx:
                if abs(sub_df.iloc[:, i].corr(sub_df.iloc[:, j])) >= corr_thr:
                    keep = False
                    break
            if keep:
                keep_idx.append(i)
        sel_idx = [mi_top_idx[i] for i in keep_idx]
        feat_names = [features[i] for i in sel_idx]
        X_tr = X_train_raw[:, sel_idx]
        X_te = X_test_raw[:, sel_idx]
        if apply_scaler:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        fe_extra_info["mi_scores"] = pd.DataFrame({
            "feature": [features[i] for i in mi_order],
            "mi_score": [mi_scores[i] for i in mi_order],
        })

    elif strategy == "pca":
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_train_raw)
        X_te_sc = sc.transform(X_test_raw)
        pca_obj = PCA(n_components=n_components, random_state=int(random_state))
        X_tr = pca_obj.fit_transform(X_sc)
        X_te = pca_obj.transform(X_te_sc)
        feat_names = [f"PC{i + 1}" for i in range(X_tr.shape[1])]
        explained_var = pca_obj.explained_variance_ratio_

    else:  # manual
        if not selected_features:
            st.error("최소 1개 이상 피처를 선택해주세요.")
            st.stop()
        sel_idx = [features.index(f) for f in selected_features]
        X_tr = X_train_raw[:, sel_idx]
        X_te = X_test_raw[:, sel_idx]
        feat_names = list(selected_features)
        if apply_scaler:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

    st.session_state["fe_strategy"]      = strategy
    st.session_state["fe_X_train"]       = X_tr
    st.session_state["fe_X_test"]        = X_te
    st.session_state["fe_y_train"]       = y_train
    st.session_state["fe_y_test"]        = y_test
    st.session_state["fe_feature_names"] = feat_names
    st.session_state["fe_pca_obj"]       = pca_obj
    st.session_state["fe_explained_var"] = explained_var
    st.session_state["fe_train_df"]      = train_df
    st.session_state["fe_test_df"]       = test_df
    # 학습 결과 캐시 무효화
    for k in ("training_result", "training_result_uw", "shap_cache"):
        st.session_state.pop(k, None)

    st.success(
        f"✅ 적용 완료 — Train: {X_tr.shape}, Test: {X_te.shape} | 피처 수: {len(feat_names)}"
    )

# ── 결과 미리보기 ─────────────────────────────────────────────────
if "fe_X_train" in st.session_state:
    st.divider()
    st.subheader("적용 결과 미리보기")
    X_tr = st.session_state["fe_X_train"]
    feat_names = st.session_state["fe_feature_names"]
    cur_strategy = st.session_state["fe_strategy"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Train shape", str(X_tr.shape))
    c2.metric("Test shape", str(st.session_state["fe_X_test"].shape))
    c3.metric("적용 전략", cur_strategy.upper())

    if cur_strategy == "pca" and st.session_state["fe_explained_var"] is not None:
        st.markdown("**PCA Scree Plot**")
        st.plotly_chart(scree_plot(st.session_state["fe_explained_var"]),
                          use_container_width=True)
        cum = np.cumsum(st.session_state["fe_explained_var"])
        st.caption(f"누적 분산 설명률: {cum[-1]:.2%} (총 {len(cum)}개 컴포넌트)")
    elif cur_strategy == "selection":
        st.markdown(f"**선택된 피처 ({len(feat_names)}개):** {', '.join(feat_names[:30])}"
                     + ("..." if len(feat_names) > 30 else ""))
    else:
        st.markdown(f"**사용 피처 ({len(feat_names)}개):** {', '.join(feat_names[:30])}"
                     + ("..." if len(feat_names) > 30 else ""))
    preview = pd.DataFrame(X_tr[:5], columns=feat_names)
    st.dataframe(preview, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # 💡 가장 중요한 결과 정리 (비전공자 친화)
    # ════════════════════════════════════════════════════════════════
    render_fe_insights(
        strategy=cur_strategy,
        n_original=len(features),
        n_after=len(feat_names),
        train_shape=X_tr.shape,
        test_shape=st.session_state["fe_X_test"].shape,
        apply_scaler=apply_scaler,
        explained_var=st.session_state.get("fe_explained_var"),
        selected_features=feat_names,
    )
