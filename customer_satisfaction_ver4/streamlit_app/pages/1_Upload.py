"""Page 1 — 데이터 업로드 및 변수 선택."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.data_utils import (
    detect_column_types,
    encode_categorical,
    impute_missing,
    list_excel_sheets,
    load_uploaded_file,
    missing_summary,
)
from utils.progress_utils import render_step_progress

st.set_page_config(page_title="LGD CSAT — 1. 데이터 업로드", page_icon="📁", layout="wide")
render_step_progress(current_step=1)
st.title("📁 1. 데이터 업로드 & 변수 선택")

# ── 사이드바: 샘플 데이터 옵션 ────────────────────────────────────
with st.sidebar:
    st.markdown("### 데이터 소스")
    use_sample = st.toggle("샘플 데이터 사용 (Customer Satisfaction)", value=False)

# ── 파일 업로드 ──────────────────────────────────────────────────
if use_sample:
    sample_path = Path(__file__).resolve().parents[2] / "data" / "customer_satisfaction_train.csv"
    if not sample_path.exists():
        sample_path = Path(__file__).resolve().parents[2] / "data" / "train.csv"
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        st.success(f"샘플 데이터 로드: `{sample_path.name}` — {df.shape[0]}행 × {df.shape[1]}열")
    else:
        st.error("샘플 데이터를 찾을 수 없습니다. CSV/Excel 파일을 직접 업로드해주세요.")
        df = None
else:
    uploaded = st.file_uploader(
        "CSV / Excel / TSV 파일을 업로드하세요",
        type=["csv", "xlsx", "xls", "tsv"],
        help="구분자: CSV(,) / TSV(\\t) / Excel(.xlsx)",
    )
    df = None
    if uploaded is not None:
        sheets = list_excel_sheets(uploaded)
        sheet_name = None
        if sheets:
            sheet_name = st.selectbox("시트 선택", sheets, key="upload_sheet")
        try:
            df = load_uploaded_file(uploaded, sheet_name=sheet_name)
            st.success(f"업로드 완료 — {df.shape[0]}행 × {df.shape[1]}열")
        except Exception as e:
            st.error(f"파일 로드 실패: {e}")
            df = None

if df is None:
    st.info("⬆ 파일을 업로드하거나 사이드바에서 샘플 데이터를 선택하세요.")
    st.stop()

# ── 데이터 미리보기 ──────────────────────────────────────────────
st.subheader("데이터 미리보기")
c1, c2, c3, c4 = st.columns(4)
c1.metric("행 수", f"{df.shape[0]:,}")
c2.metric("열 수", f"{df.shape[1]:,}")
c3.metric("결측 셀", f"{int(df.isna().sum().sum()):,}")
c4.metric("중복 행", f"{int(df.duplicated().sum()):,}")

st.dataframe(df.head(10), use_container_width=True)

# ── 컬럼 정보 ────────────────────────────────────────────────────
with st.expander("컬럼 / 결측 / 타입 상세 보기"):
    miss_df = missing_summary(df)
    st.dataframe(miss_df, use_container_width=True)

types = detect_column_types(df)

# ── 변수 선택 ────────────────────────────────────────────────────
st.subheader("종속변수 / 독립변수 선택")
cc1, cc2 = st.columns([1, 2])
with cc1:
    target_default = st.session_state.get("target")
    target_index = list(df.columns).index(target_default) if target_default in df.columns else 0
    target = st.selectbox("종속변수 (Target)", options=list(df.columns), index=target_index)
    st.markdown(f"**클래스 분포 (`{target}`):**")
    target_counts = df[target].value_counts(dropna=False).sort_index()
    st.dataframe(target_counts.rename("count").to_frame(), use_container_width=True)

with cc2:
    default_features = [c for c in df.columns if c != target]
    feature_cols = st.multiselect(
        "독립변수 (Features) — 종속변수는 자동 제외",
        options=[c for c in df.columns if c != target],
        default=default_features,
    )

if not feature_cols:
    st.warning("독립변수를 1개 이상 선택해주세요.")
    st.stop()

# ── 컬럼 타입 / 인코딩 ────────────────────────────────────────────
st.subheader("범주형 / 결측치 처리")
auto_cat = [c for c in feature_cols if c in types["categorical"]]
cat_cols = st.multiselect(
    "범주형 컬럼 (자동 감지된 항목 수정 가능)",
    options=feature_cols, default=auto_cat,
)
encode_method = st.radio(
    "범주형 인코딩 방식",
    options=["label", "onehot"],
    format_func=lambda x: {"label": "Label Encoding (권장)", "onehot": "One-Hot Encoding"}[x],
    horizontal=True,
)

cm1, cm2 = st.columns(2)
with cm1:
    impute_strategy = st.radio(
        "결측치 처리 전략",
        options=["median", "mean", "most_frequent", "knn", "drop"],
        format_func=lambda x: {
            "median": "중앙값 대체",
            "mean": "평균 대체",
            "most_frequent": "최빈값 대체",
            "knn": "KNN Imputation",
            "drop": "결측 행 제거",
        }[x],
        horizontal=False,
    )
with cm2:
    missing_total = int(df[feature_cols + [target]].isna().sum().sum())
    if missing_total > 0:
        st.info(f"전체 결측 셀: {missing_total} — 위 전략을 적용해 처리합니다.")
    else:
        st.success("결측 셀이 없습니다 ✓")

# ── 적용 버튼 ────────────────────────────────────────────────────
if st.button("✅ 변수 / 전처리 설정 적용", type="primary", use_container_width=True):
    used_cols = list(set(feature_cols + [target]))
    df_used = df[used_cols].copy()

    numeric_in_features = [c for c in feature_cols if c in types["numeric"]]
    df_used = impute_missing(
        df_used, strategy=impute_strategy,
        numeric_cols=numeric_in_features,
        categorical_cols=cat_cols,
    )

    if cat_cols:
        df_used, encoders = encode_categorical(df_used, cat_cols, method=encode_method)
    else:
        encoders = {"method": "none"}

    # one-hot 시 feature_cols 재정의
    if encode_method == "onehot" and cat_cols:
        new_feature_cols = [c for c in df_used.columns if c != target]
    else:
        new_feature_cols = feature_cols

    st.session_state["df_raw"] = df.copy()
    st.session_state["df"] = df_used
    st.session_state["target"] = target
    st.session_state["feature_cols"] = new_feature_cols
    st.session_state["cat_cols"] = cat_cols
    st.session_state["impute_strategy"] = impute_strategy
    st.session_state["encoders"] = encoders

    # 후속 페이지의 캐시 무효화
    for k in ("fe_X_train", "fe_X_test", "fe_y_train", "fe_y_test",
              "fe_feature_names", "fe_strategy", "training_result",
              "training_result_uw", "shap_cache"):
        st.session_state.pop(k, None)

    st.success(f"✅ 설정 완료 — 종속변수: `{target}` / 독립변수: {len(new_feature_cols)}개")
    st.markdown("⬅ 사이드바에서 다음 단계인 **2_EDA** 페이지로 이동하세요.")
