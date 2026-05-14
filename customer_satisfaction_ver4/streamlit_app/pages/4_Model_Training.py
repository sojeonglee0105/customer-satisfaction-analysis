"""Page 4 — 모델 선택, 하이퍼파라미터 설정, 학습 실행."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.insight_utils import render_training_insights
from utils.model_utils import (
    DEFAULT_HPARAMS,
    LGB_AVAILABLE,
    MODEL_LIST,
    XGB_AVAILABLE,
    run_repeated,
)
from utils.progress_utils import render_step_progress

st.set_page_config(page_title="LGD CSAT — 4. 모델 학습", page_icon="🤖", layout="wide")
render_step_progress(current_step=4)
st.title("🤖 4. 모델 학습 & 설정")

if "fe_X_train" not in st.session_state:
    st.warning("⬅ 먼저 **3_Feature_Engineering** 페이지에서 학습 데이터를 준비해주세요.")
    st.stop()

X_train = st.session_state["fe_X_train"]
X_test  = st.session_state["fe_X_test"]
y_train = st.session_state["fe_y_train"]
y_test  = st.session_state["fe_y_test"]

c1, c2, c3 = st.columns(3)
c1.metric("Train", f"{X_train.shape[0]:,}")
c2.metric("Test", f"{X_test.shape[0]:,}")
c3.metric("Features", f"{X_train.shape[1]:,}")

# ── 모델 선택 ────────────────────────────────────────────────────
st.subheader("학습할 모델 선택 (복수 선택 가능)")
default_models = [m for m in ["Logistic Regression", "LightGBM", "Random Forest", "Decision Tree"]
                   if m in MODEL_LIST]
model_names = st.multiselect("모델", options=MODEL_LIST, default=default_models)

if not model_names:
    st.error("모델을 1개 이상 선택해주세요.")
    st.stop()

if "LightGBM" in model_names and not LGB_AVAILABLE:
    st.warning("LightGBM이 설치되지 않았습니다. `pip install lightgbm`")
if "XGBoost" in model_names and not XGB_AVAILABLE:
    st.warning("XGBoost가 설치되지 않았습니다. `pip install xgboost`")

# ── 공통 옵션 ────────────────────────────────────────────────────
st.subheader("공통 옵션")
o1, o2, o3 = st.columns(3)
with o1:
    use_class_weight = st.toggle("class_weight = 'balanced' 적용", value=True)
with o2:
    n_seeds = st.number_input("반복 시드 수", min_value=1, max_value=20, value=5)
with o3:
    base_seed = st.number_input("시작 시드", min_value=0, value=42)

compare_with_unweighted = st.checkbox(
    "보정 vs 미보정 비교 실행 (학습 시간 2배 증가)",
    value=False,
    help="True 시 동일 설정으로 unweighted/weighted 두 조건 모두 학습합니다.",
)

# ── 하이퍼파라미터 ───────────────────────────────────────────────
st.subheader("모델별 하이퍼파라미터")
hparams_map: dict[str, dict] = {}
for name in model_names:
    with st.expander(f"⚙️ {name}", expanded=False):
        defaults = DEFAULT_HPARAMS.get(name, {})
        new_p = {}
        if name == "Logistic Regression":
            new_p["C"] = st.number_input(f"{name} — C (정규화 역수)",
                                           value=float(defaults.get("C", 1.0)),
                                           step=0.1, key=f"{name}_C")
            new_p["max_iter"] = st.number_input(f"{name} — max_iter",
                                                  value=int(defaults.get("max_iter", 2000)),
                                                  step=100, key=f"{name}_iter")
            new_p["solver"] = st.selectbox(f"{name} — solver",
                                             ["lbfgs", "liblinear", "saga"],
                                             index=0, key=f"{name}_solver")
        elif name == "Decision Tree":
            new_p["max_depth"] = st.number_input(f"{name} — max_depth",
                                                   value=int(defaults.get("max_depth", 8)),
                                                   step=1, key=f"{name}_depth")
            new_p["min_samples_split"] = st.number_input(f"{name} — min_samples_split",
                                                          value=int(defaults.get("min_samples_split", 2)),
                                                          step=1, key=f"{name}_minss")
        elif name == "Random Forest":
            new_p["n_estimators"] = st.number_input(f"{name} — n_estimators",
                                                      value=int(defaults.get("n_estimators", 200)),
                                                      step=50, key=f"{name}_n")
            md = defaults.get("max_depth")
            md_val = st.number_input(f"{name} — max_depth (0 = None)",
                                       value=int(md) if md else 0,
                                       step=1, key=f"{name}_rfmd")
            new_p["max_depth"] = None if md_val == 0 else int(md_val)
        elif name == "LightGBM":
            new_p["n_estimators"] = st.number_input(f"{name} — n_estimators",
                                                      value=int(defaults.get("n_estimators", 200)),
                                                      step=50, key=f"{name}_lgbn")
            new_p["learning_rate"] = st.number_input(f"{name} — learning_rate",
                                                       value=float(defaults.get("learning_rate", 0.05)),
                                                       step=0.01, format="%.3f", key=f"{name}_lgblr")
            new_p["num_leaves"] = st.number_input(f"{name} — num_leaves",
                                                     value=int(defaults.get("num_leaves", 31)),
                                                     step=1, key=f"{name}_lgbl")
        elif name == "XGBoost":
            new_p["n_estimators"] = st.number_input(f"{name} — n_estimators",
                                                      value=int(defaults.get("n_estimators", 200)),
                                                      step=50, key=f"{name}_xgbn")
            new_p["learning_rate"] = st.number_input(f"{name} — learning_rate",
                                                       value=float(defaults.get("learning_rate", 0.1)),
                                                       step=0.01, format="%.3f", key=f"{name}_xgblr")
            new_p["max_depth"] = st.number_input(f"{name} — max_depth",
                                                   value=int(defaults.get("max_depth", 6)),
                                                   step=1, key=f"{name}_xgbmd")
        hparams_map[name] = new_p

# ── 학습 실행 ────────────────────────────────────────────────────
st.divider()
if st.button("🚀 학습 실행", type="primary", use_container_width=True):
    seeds = [int(base_seed) + i * 100 for i in range(int(n_seeds))]
    progress = st.progress(0.0, text="학습 준비 중...")

    def cb(frac, msg):
        progress.progress(min(frac, 1.0), text=msg)

    cw = "balanced" if use_class_weight else None
    with st.spinner("모델 학습 중..."):
        if compare_with_unweighted:
            res_uw = run_repeated(
                model_names, X_train, y_train, X_test, y_test,
                class_weight=None, hparams_map=hparams_map,
                seeds=seeds, progress_cb=cb,
            )
            res_w = run_repeated(
                model_names, X_train, y_train, X_test, y_test,
                class_weight="balanced", hparams_map=hparams_map,
                seeds=seeds, progress_cb=cb,
            )
            st.session_state["training_result_uw"] = res_uw
            st.session_state["training_result"]    = res_w
            st.success("✅ 보정/미보정 비교 학습 완료")
        else:
            res = run_repeated(
                model_names, X_train, y_train, X_test, y_test,
                class_weight=cw, hparams_map=hparams_map,
                seeds=seeds, progress_cb=cb,
            )
            st.session_state["training_result"]    = res
            st.session_state.pop("training_result_uw", None)
            st.success(f"✅ 학습 완료 — class_weight={cw}, {len(seeds)} seeds × {len(model_names)} 모델")

    progress.empty()
    st.markdown("⬅ 사이드바에서 **5_Model_Comparison**으로 이동해 결과를 비교하세요.")

# ── 직전 학습 결과 요약 ───────────────────────────────────────────
if "training_result" in st.session_state:
    st.subheader("최근 학습 결과 요약")
    res = st.session_state["training_result"]
    raw = res["raw"]
    seeds = res["seeds"]
    rows = []
    for name in res["model_names"]:
        f1s = [raw[s][name].get("macro_f1", np.nan) for s in seeds]
        au  = [raw[s][name].get("macro_auroc", np.nan) for s in seeds]
        rows.append({
            "모델": name,
            "Macro F1 평균": round(float(np.nanmean(f1s)), 4),
            "Macro AUROC 평균": round(float(np.nanmean(au)), 4) if not all(np.isnan(au)) else None,
            "평균 학습시간(초)": round(float(np.mean([raw[s][name]["train_sec"] for s in seeds])), 3),
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # 💡 가장 중요한 결과 정리 (비전공자 친화)
    # ════════════════════════════════════════════════════════════════
    render_training_insights(
        summary_rows=rows,
        class_weight=res.get("class_weight"),
        n_seeds=len(seeds),
        n_features=X_train.shape[1],
    )
