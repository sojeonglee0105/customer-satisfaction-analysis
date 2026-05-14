"""LGD's Customer Satisfaction Survey Analysis Platform — Home 페이지."""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from utils.progress_utils import render_home_progress

APP_TITLE = "LGD's Customer Satisfaction Survey Analysis Platform"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""<div style="
      background:linear-gradient(135deg,#4f46e5 0%,#9333ea 50%,#ec4899 100%);
      padding:28px 32px;border-radius:14px;margin-bottom:18px;
      box-shadow:0 6px 20px rgba(79,70,229,.25);">
      <div style="color:#e0e7ff;font-size:11px;font-weight:700;
                   letter-spacing:2px;text-transform:uppercase;">
        📊 LG Display
      </div>
      <div style="color:white;font-size:30px;font-weight:800;
                   letter-spacing:-0.5px;line-height:1.2;margin-top:6px;">
        {APP_TITLE}
      </div>
      <div style="color:#ddd6fe;font-size:13px;margin-top:8px;">
        EDA · Feature Engineering · Model Training · Comparison · Feature Importance · Report
      </div>
    </div>""",
    unsafe_allow_html=True,
)
st.markdown(
    "CSV/Excel 파일을 업로드하면 **EDA → 피처 엔지니어링 → 모델 학습 → 비교 → 중요도/SHAP → 리포트**까지 "
    "전체 워크플로우를 대화형으로 진행할 수 있습니다."
)

# ── 진행 상태 (모든 페이지와 동일한 stepper) ─────────────────────
ss = st.session_state
render_home_progress()

st.divider()

# ── 워크플로우 다이어그램 (Mermaid) ─────────────────────────────────
st.subheader("워크플로우")
st.markdown(
    """
1. **데이터 업로드** — CSV/Excel을 업로드하고 종속/독립 변수를 선택합니다.
2. **EDA** — 분포, 상관관계, 다중공선성, Kruskal-Wallis 단변량 검정으로 데이터를 탐색합니다.
3. **피처 엔지니어링** — 전체 / Feature Selection / PCA / 직접 선택 4가지 전략을 비교합니다.
4. **모델 학습** — 4종 이상의 모델을 선택해 5-seed 반복 실험으로 학습합니다.
5. **모델 비교** — Radar / Box / 통계 검정 / Confusion Matrix / ROC를 통해 성능을 비교합니다.
6. **Feature Importance** — 내장 / Permutation / SHAP 분석으로 중요도와 방향성을 확인합니다.
7. **리포트** — KPI 카드와 다운로드 가능한 종합 리포트를 생성합니다.
"""
)

st.divider()

# ── 추가 기능 안내 ────────────────────────────────────────────────
st.subheader("추가 기능")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        "**🎯 위험 고객 시뮬레이터**\n\n"
        "Page 6에서 슬라이더로 피처 값을 입력하면 각 모델의 RPI 예측 확률이 실시간으로 표시됩니다."
    )
with c2:
    st.markdown(
        "**📦 배치 예측**\n\n"
        "Page 7에서 신규 CSV를 업로드하면 학습된 모델로 예측 결과를 다운로드할 수 있습니다."
    )
with c3:
    st.markdown(
        "**🔧 결측치 / 이상치 처리**\n\n"
        "Page 1에서 Median / Mean / KNN 결측치 대체 전략을 선택할 수 있습니다."
    )

st.divider()

# ── 시작 안내 ──────────────────────────────────────────────────────
if "df" not in ss:
    st.warning("⬅ 왼쪽 사이드바에서 **1_Upload** 페이지로 이동해 데이터를 업로드하세요.")
else:
    st.success(f"✅ 데이터 준비 완료 — {ss['df'].shape[0]}행 × {ss['df'].shape[1]}열")
    if "target" in ss:
        st.info(f"종속변수: **{ss['target']}**  |  독립변수: **{len(ss.get('feature_cols', []))}개**")
