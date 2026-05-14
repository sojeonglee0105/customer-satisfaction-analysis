"""ML 비전공자도 이해 가능한 인사이트 카드 렌더러.

각 페이지(FE / Model Training / Model Comparison / Feature Importance)
마지막에 호출되며, 화면 가장 아래에 다음 4가지를 일관된 스타일로 보여줍니다:

  ① 한 줄 결론 — 큰 글씨 + 색상
  ② 쉽게 이해하기 — 일상적 비유
  ③ 핵심 숫자 카드 — 한눈에 보는 핵심 지표
  ④ 다음 단계 — 다음 페이지/액션 안내
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

# ── 공용 스타일 ──────────────────────────────────────────────────────
_CARD_OPEN = """
<div style="
  background:linear-gradient(135deg,#eef2ff 0%,#fdf2f8 100%);
  border-left:6px solid #4f46e5;
  border-radius:14px;
  padding:22px 28px;
  margin:18px 0 8px;
  box-shadow:0 2px 12px rgba(79,70,229,.08);
">
"""
_CARD_CLOSE = "</div>"


def _section_banner(title: str, subtitle: str = ""):
    """페이지 마지막 인사이트 섹션의 큰 헤더."""
    st.markdown(
        f"""<div style="
          background:linear-gradient(90deg,#4f46e5,#9333ea);
          color:white;border-radius:12px;padding:18px 24px;margin-top:24px;
          box-shadow:0 4px 14px rgba(79,70,229,.25);">
          <div style="font-size:20px;font-weight:800;">{title}</div>
          <div style="font-size:13px;opacity:.92;margin-top:4px;">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _conclusion_box(text: str, color: str = "#4f46e5"):
    """한 줄 결론 — 매우 크고 강조된 텍스트."""
    st.markdown(
        f"""<div style="
          background:white;border:2px solid {color};border-radius:14px;
          padding:22px 28px;margin:14px 0;
          box-shadow:0 2px 8px rgba(0,0,0,.04);">
          <div style="font-size:11px;color:#64748b;letter-spacing:1px;
                      font-weight:700;margin-bottom:6px;">📢 한 줄 결론</div>
          <div style="font-size:20px;font-weight:700;color:{color};
                      line-height:1.5;">{text}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _kpi_grid(items: list[tuple[str, str, str]], color: str = "#4f46e5"):
    """KPI 카드 그리드 (label, value, subtitle)."""
    cards = "".join(
        f"""<div style="background:white;border-radius:10px;padding:16px 18px;
                       border-top:4px solid {color};
                       box-shadow:0 1px 6px rgba(0,0,0,.05);">
              <div style="font-size:11px;color:#64748b;font-weight:700;
                          letter-spacing:.5px;text-transform:uppercase;">{label}</div>
              <div style="font-size:24px;font-weight:800;color:{color};
                          margin:6px 0 4px;">{value}</div>
              <div style="font-size:12px;color:#475569;">{sub}</div>
            </div>"""
        for label, value, sub in items
    )
    st.markdown(
        f"""<div style="display:grid;grid-template-columns:repeat({len(items)},1fr);
                       gap:12px;margin:8px 0 14px;">{cards}</div>""",
        unsafe_allow_html=True,
    )


def _analogy_card(title: str, body: str):
    """일상 비유로 풀어 쓰는 설명 카드."""
    st.markdown(
        f"""<div style="background:#fefce8;border-left:4px solid #ca8a04;
                       border-radius:10px;padding:16px 20px;margin:10px 0;">
          <div style="font-weight:700;color:#854d0e;margin-bottom:6px;">
            🧩 쉽게 이해하기 — {title}
          </div>
          <div style="color:#1f2937;font-size:14px;line-height:1.7;">{body}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _qa_cards(qa_items: list[tuple[str, str]]):
    """용어 풀이 Q&A — Streamlit expander 사용."""
    with st.expander("🔍 용어가 어려우신가요? — 핵심 용어 풀이", expanded=False):
        for q, a in qa_items:
            st.markdown(f"**Q. {q}**")
            st.markdown(a)
            st.markdown("")


def _next_steps(items: list[str]):
    bullets = "".join(f"<li style='margin:6px 0;'>{x}</li>" for x in items)
    st.markdown(
        f"""<div style="background:#f0fdf4;border-left:4px solid #16a34a;
                       border-radius:10px;padding:14px 22px;margin:12px 0;">
          <div style="font-weight:700;color:#15803d;margin-bottom:6px;">
            ✅ 이제 다음 단계로
          </div>
          <ul style="margin:0;padding-left:18px;color:#1f2937;font-size:14px;
                      line-height:1.7;">{bullets}</ul>
        </div>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# Page 2 — EDA 인사이트
# ════════════════════════════════════════════════════════════════════
def render_eda_insights(*, n_samples: int, n_features: int, n_classes: int,
                          target_name: str,
                          class_counts: dict | pd.Series,
                          imbalance_ratio: float,
                          kw_df: pd.DataFrame | None = None,
                          multi_df: pd.DataFrame | None = None,
                          numeric_n: int = 0,
                          missing_total: int = 0):
    _section_banner(
        "💡 EDA — 가장 중요한 결과 정리",
        "데이터 탐색에서 발견한 핵심 사실을 비전공자도 한눈에 이해할 수 있도록 정리했습니다.",
    )

    # ── 데이터 품질 종합 판정 ────────────────────────────────────
    if imbalance_ratio > 10:
        balance_label = "🔴 심한 불균형"
        balance_color = "#dc2626"
    elif imbalance_ratio > 3:
        balance_label = "🟡 다소 불균형"
        balance_color = "#d97706"
    else:
        balance_label = "🟢 균형 양호"
        balance_color = "#16a34a"

    # KW 검정 결과 분석
    n_sig = 0
    top_features: list[tuple[str, float]] = []
    weak_features: list[str] = []
    if kw_df is not None and not kw_df.empty:
        n_sig = int((kw_df["p-value"] < 0.05).sum())
        sorted_kw = kw_df.sort_values("η²", ascending=False).reset_index(drop=True)
        top_features = [
            (str(r["변수"]), float(r["η²"])) for _, r in sorted_kw.head(5).iterrows()
        ]
        # 클래스 분리 능력 거의 없는 변수 (η² < 0.01 & p-value 비유의)
        weak_features = sorted_kw[
            (sorted_kw["η²"] < 0.01) & (sorted_kw["p-value"] >= 0.05)
        ]["변수"].astype(str).tolist()

    sig_pct = (n_sig / len(kw_df) * 100) if (kw_df is not None and len(kw_df) > 0) else 0

    n_multi = len(multi_df) if multi_df is not None else 0

    # ① 한 줄 결론
    if top_features and n_sig > 0:
        top1, top1_eta = top_features[0]
        conclusion = (
            f"<b>{n_sig}개({sig_pct:.0f}%)</b> 변수가 클래스 구분에 통계적으로 유의하며, "
            f"그 중 <b>{top1}</b>이 가장 강력한 예측 신호(η²={top1_eta:.3f})입니다."
        )
        color = "#4f46e5"
    else:
        conclusion = (
            f"<b>{target_name}</b>의 클래스 분포 비율이 {imbalance_ratio:.1f}:1이며, "
            f"모델 학습 전 분포/상관관계를 점검했습니다."
        )
        color = balance_color
    _conclusion_box(conclusion, color=color)

    # ② KPI
    _kpi_grid([
        ("데이터 크기", f"{n_samples:,} 행", f"독립변수 {n_features}개"),
        ("타겟 클래스", f"{n_classes}개", f"불균형 {imbalance_ratio:.1f}:1"),
        ("클래스 균형", balance_label, "★ 학습 시 보정 권장" if imbalance_ratio > 3 else "추가 보정 불필요"),
        ("유의 변수 비율",
         f"{sig_pct:.0f}%" if kw_df is not None else "—",
         f"전체 {len(kw_df) if kw_df is not None else 0}개 중 {n_sig}개"),
    ], color=color)

    # ③ Top 3 강력한 변수
    if top_features:
        top_html_items = "".join(
            f"""<div style="background:white;border-radius:10px;padding:14px 18px;
                          border-left:4px solid #4f46e5;
                          box-shadow:0 1px 4px rgba(0,0,0,.05);">
                  <div style="font-size:11px;color:#64748b;font-weight:700;
                              letter-spacing:.5px;">#{i+1} 강력한 신호 변수</div>
                  <div style="font-size:18px;font-weight:800;color:#4f46e5;
                              margin:4px 0 2px;">{name}</div>
                  <div style="font-size:12px;color:#475569;">η² = <b>{eta:.4f}</b></div>
                </div>"""
            for i, (name, eta) in enumerate(top_features[:5])
        )
        st.markdown(
            f"""<div style="margin:10px 0 14px;">
              <div style="font-weight:700;color:#1e293b;margin-bottom:8px;">
                🏆 클래스 분리 능력 Top 5 (Kruskal-Wallis η² 기준)
              </div>
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
                          gap:10px;">{top_html_items}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ④ 비유 카드
    _analogy_card(
        "EDA는 '병원에서 진찰 전에 받는 기본 검사'와 같습니다",
        "<b>분포 그래프</b>는 환자의 외형 검사, <b>상관관계 히트맵</b>은 혈액검사처럼 변수들의 "
        "관련성을 보는 것이고, <b>Kruskal-Wallis 검정</b>은 '어떤 검사 항목이 환자 그룹을 "
        "구별하는데 가장 도움이 되는지'를 정량적으로 알려줍니다. "
        + (
            f"<br><b>{top_features[0][0]}</b>이 가장 큰 신호를 보이므로, 비즈니스적으로도 "
            "이 변수의 측정/관리가 핵심임을 시사합니다."
            if top_features else
            "분석 결과 통계적으로 강한 신호를 가진 변수가 많지 않아, 추가 변수 수집이나 "
            "피처 생성(feature engineering)이 필요할 수 있습니다."
        ),
    )

    # ⑤ 데이터 품질 경고/안내
    warnings_html: list[str] = []

    if imbalance_ratio > 10:
        warnings_html.append(
            f"""<li><b>⚠ 심한 클래스 불균형 ({imbalance_ratio:.1f}:1)</b> — 모델이 다수 클래스만 학습할 위험이 큽니다.
                      Page 4에서 <b>class_weight='balanced'</b> 옵션을 켜고 평가는 <b>Macro F1</b>으로 보세요.</li>"""
        )
    elif imbalance_ratio > 3:
        warnings_html.append(
            f"""<li><b>🟡 다소 불균형 ({imbalance_ratio:.1f}:1)</b> — Macro F1과 Micro F1을 함께 보고,
                       소수 클래스 성능이 낮다면 보정을 적용하세요.</li>"""
        )

    if n_multi > 0:
        warnings_html.append(
            f"""<li><b>🔁 다중공선성 의심 쌍 {n_multi}개</b> — 비슷한 정보가 중복되면 해석이 어렵고
                      회귀 계수가 불안정해집니다. <b>FE에서 'Selection' 또는 'PCA' 전략</b>을 추천합니다.</li>"""
        )

    if weak_features and len(weak_features) >= 3:
        sample = weak_features[:5]
        more = f" 외 {len(weak_features) - 5}개" if len(weak_features) > 5 else ""
        warnings_html.append(
            f"""<li><b>🪶 신호가 약한 변수 {len(weak_features)}개 발견</b> ({', '.join(sample)}{more}) —
                      모델 학습에 거의 도움되지 않을 가능성이 큽니다. 제거 또는 변환을 검토하세요.</li>"""
        )

    if missing_total > 0:
        warnings_html.append(
            f"""<li><b>🩹 결측치 {missing_total:,}개</b> — Page 1에서 결측 처리 방식을 확인했는지 점검하세요.</li>"""
        )

    if not warnings_html:
        warnings_html.append(
            """<li><b>✅ 데이터 품질이 전반적으로 양호</b>합니다 — 큰 결함이 없으니 다음 단계로 진행해도 좋습니다.</li>"""
        )

    st.markdown(
        f"""<div style="background:#fef2f2;border-left:4px solid #f43f5e;
                       border-radius:10px;padding:14px 22px;margin:10px 0;">
          <div style="font-weight:700;color:#9f1239;margin-bottom:6px;">
            🚦 데이터 품질 점검 결과
          </div>
          <ul style="margin:0;padding-left:18px;color:#1f2937;font-size:14px;
                      line-height:1.8;">{''.join(warnings_html)}</ul>
        </div>""",
        unsafe_allow_html=True,
    )

    # ⑥ Q&A
    qa = [
        ("p-value < 0.05가 무슨 의미인가요?",
         "**'관찰된 차이가 우연일 확률이 5% 미만'**이라는 뜻입니다. "
         "즉 그 변수가 클래스를 구분하는데 **진짜 영향이 있다**고 통계적으로 자신할 수 있습니다."),
        ("η² (eta-squared)가 뭔가요?",
         "**효과 크기(effect size)**입니다. p-value가 '진짜 차이가 있는가?'를 답한다면, "
         "**η²는 '그 차이가 얼마나 큰가?'**를 답합니다. "
         "보통 **η² > 0.06이면 중간**, **0.14 이상이면 큰 효과**로 봅니다."),
        ("클래스 불균형은 왜 문제가 되나요?",
         "예: 90% A 클래스, 10% B 클래스라면 모델이 모두 A로 예측해도 **정확도 90%**가 나옵니다. "
         "하지만 B 클래스(소수)는 전혀 못 맞히는 모델입니다. "
         "**Macro F1**과 **class_weight='balanced'** 옵션이 이를 보정합니다."),
        ("상관관계(|r|)가 0.9 이상이면 왜 문제인가요?",
         "거의 같은 정보를 가진 두 변수는 모델 입력으로 **중복**되며, "
         "선형 모델에서는 **계수가 비현실적으로 변동**합니다(다중공선성). "
         "둘 중 하나만 남기거나 PCA로 묶어주는 것이 안전합니다."),
        ("Violin Plot은 어떻게 읽나요?",
         "**가로(좌우) 두께가 그 값에서의 데이터 빈도**입니다. 클래스별로 violin의 형태가 "
         "**확연히 다르면** 그 변수가 클래스 구분에 유용한 신호를 가집니다."),
    ]
    _qa_cards(qa)

    # ⑦ 다음 단계 — 데이터 특성에 맞춤 추천
    next_items: list[str] = []
    if n_multi > 0:
        next_items.append(
            "<b>3. Feature Engineering</b>에서 <b>'Selection' 또는 'PCA' 전략</b>을 추천합니다 (다중공선성 제거)."
        )
    elif weak_features and len(weak_features) >= 3:
        next_items.append(
            "<b>3. Feature Engineering</b>에서 <b>'Selection' 전략</b>을 추천합니다 (약한 변수 자동 제거)."
        )
    else:
        next_items.append(
            "<b>3. Feature Engineering</b>에서 'All Features' 또는 'Selection' 전략 모두 시도해볼 만합니다."
        )

    if imbalance_ratio > 3:
        next_items.append(
            "<b>4. Model Training</b>에서 반드시 <b>class_weight='balanced'</b>를 켜고 학습하세요."
        )
    if top_features:
        names = ", ".join([f for f, _ in top_features[:3]])
        next_items.append(
            f"가장 강력한 신호 변수 <b>{names}</b>는 모델 해석/도메인 검토 시 우선 고려 대상입니다."
        )
    _next_steps(next_items)


# ════════════════════════════════════════════════════════════════════
# Page 3 — Feature Engineering 인사이트
# ════════════════════════════════════════════════════════════════════
def render_fe_insights(*, strategy: str, n_original: int, n_after: int,
                         train_shape: tuple, test_shape: tuple,
                         apply_scaler: bool,
                         explained_var: np.ndarray | None = None,
                         selected_features: list[str] | None = None):
    _section_banner(
        "💡 Feature Engineering — 가장 중요한 결과 정리",
        "ML 기초가 없는 분도 한 번에 이해할 수 있도록 핵심만 정리했습니다.",
    )

    reduction = max(0, n_original - n_after)
    pct = (n_after / n_original * 100) if n_original > 0 else 0

    strategy_label = {
        "all":       "전체 피처를 그대로 사용",
        "selection": "통계적 점수로 자동 선별",
        "pca":       "PCA로 차원 압축",
        "manual":    "사용자가 직접 선택",
    }.get(strategy, strategy)

    # ① 한 줄 결론
    if strategy == "all":
        conclusion = (
            f"<b>{n_original}개</b>의 모든 피처를 <b>그대로</b> 학습에 사용합니다. "
            "(축소 없이 모든 정보를 모델에 전달)"
        )
    elif strategy == "pca":
        cum_var = float(np.sum(explained_var)) if explained_var is not None else 0
        conclusion = (
            f"<b>{n_original}개</b> 피처를 <b>{n_after}개</b>의 PCA 컴포넌트로 압축했습니다. "
            f"(원본 정보의 <b>{cum_var:.1%}</b>를 유지)"
        )
    else:
        conclusion = (
            f"<b>{n_original}개</b> 피처 중 <b>{n_after}개</b>를 사용 — "
            f"<b>{reduction}개({(reduction/n_original*100 if n_original else 0):.0f}%)</b>를 제거하여 "
            "더 깔끔한 모델이 가능합니다."
        )
    _conclusion_box(conclusion)

    # ② 핵심 KPI
    _kpi_grid([
        ("적용 전략",       strategy_label, strategy.upper()),
        ("원본 피처",       f"{n_original}개", "변환 전"),
        ("변환 후 피처",    f"{n_after}개", f"보존율 {pct:.1f}%"),
        ("Train / Test",    f"{train_shape[0]:,} / {test_shape[0]:,}",
                             f"피처 차원 {train_shape[1]}"),
    ])

    # ③ 비유 설명
    analogy_map = {
        "all": (
            "전체 정보 사용",
            "재료를 하나도 빼지 않고 요리하는 방식입니다. 정보 손실은 없지만, "
            "불필요한 재료까지 들어가서 오히려 맛이 흐려질 수 있습니다."
        ),
        "selection": (
            "꼭 필요한 정보만 선별",
            "<b>맛에 영향이 큰 재료(MI 점수 상위)</b>만 골라낸 뒤, 거의 똑같은 재료끼리는 "
            "1개만 남깁니다(다중공선성 제거). 결과적으로 <b>모델이 더 빠르고 안정적</b>이며 "
            "해석도 쉽습니다."
        ),
        "pca": (
            "여러 정보를 한 번에 압축",
            "비슷한 재료들을 합쳐 새로운 '복합 재료'로 만든 것과 같습니다. "
            "<b>차원이 줄어 빠르게 학습</b>되지만, 새로 만들어진 PC1, PC2 등은 원본 변수와 "
            "달라 <b>해석이 어려워지는</b> 단점이 있습니다."
        ),
        "manual": (
            "직접 고른 재료로 요리",
            "도메인 지식이 있는 분이 핵심 재료를 선별한 방식입니다. "
            "비즈니스 맥락이 반영되지만, <b>중요한 변수를 놓칠 가능성</b>이 있습니다."
        ),
    }
    if strategy in analogy_map:
        title, body = analogy_map[strategy]
        _analogy_card(title, body)

    # ④ 핵심 용어 Q&A
    qa = [
        ("StandardScaler가 뭔가요?",
         f"각 피처의 평균을 0, 표준편차를 1로 맞춰주는 작업입니다. "
         f"키(170cm)와 나이(30살)처럼 단위가 다른 변수를 같은 기준으로 비교 가능하게 합니다. "
         f"현재 적용: **{'적용 ✓' if apply_scaler else '미적용 ✗'}**"),
        ("왜 피처를 줄여야 하나요?",
         "변수가 너무 많으면 ① 학습이 느려지고 ② 노이즈에 모델이 흔들리며 "
         "③ 해석이 어렵습니다. **'필요한 것만 남기기'가 좋은 모델의 첫걸음**입니다."),
        ("Train / Test로 나누는 이유는?",
         "Train으로 모델을 가르치고 Test로 시험을 봅니다. "
         "**같은 데이터로 학습+평가하면 부풀려진 점수**가 나오므로 반드시 분리해야 합니다."),
    ]
    if strategy == "pca":
        qa.append((
            "PCA Scree Plot은 어떻게 봐야 하나요?",
            "막대(개별 분산)가 급격히 작아지는 '팔꿈치(elbow)' 지점이 적정 컴포넌트 수입니다. "
            "누적 분산이 80~95%면 충분한 정보를 보존한 것으로 봅니다."
        ))
    _qa_cards(qa)

    # ⑤ 다음 단계
    _next_steps([
        "사이드바에서 <b>4. Model Training</b>으로 이동하여 여러 모델을 학습해보세요.",
        "<b>class_weight='balanced'</b> 옵션을 켜면 클래스 불균형이 자동 보정됩니다.",
        "여러 시드(5회 이상)로 반복 학습하면 결과가 더 안정적입니다.",
    ])


# ════════════════════════════════════════════════════════════════════
# Page 4 — Model Training 인사이트
# ════════════════════════════════════════════════════════════════════
def render_training_insights(*, summary_rows: list[dict],
                                class_weight: str | None,
                                n_seeds: int, n_features: int):
    _section_banner(
        "💡 Model Training — 가장 중요한 결과 정리",
        "방금 학습된 여러 모델의 핵심 성능을 한눈에 정리했습니다.",
    )

    if not summary_rows:
        st.info("학습된 모델이 없어 인사이트를 표시할 수 없습니다.")
        return

    df = pd.DataFrame(summary_rows)
    if "Macro F1 평균" not in df.columns:
        st.info("Macro F1 결과가 없어 인사이트를 표시할 수 없습니다.")
        return

    df_sorted = df.sort_values("Macro F1 평균", ascending=False).reset_index(drop=True)
    best_row    = df_sorted.iloc[0]
    worst_row   = df_sorted.iloc[-1]
    fastest_row = df_sorted.sort_values("평균 학습시간(초)").iloc[0]

    # ① 한 줄 결론
    conclusion = (
        f"<b>{best_row['모델']}</b>이 가장 좋은 성능을 냈습니다 — "
        f"Macro F1 <b>{best_row['Macro F1 평균']:.4f}</b>"
    )
    _conclusion_box(conclusion, color="#16a34a")

    # ② KPI
    _kpi_grid([
        ("최고 성능 모델", str(best_row["모델"]),
         f"F1 {best_row['Macro F1 평균']:.4f}"),
        ("최저 성능 모델", str(worst_row["모델"]),
         f"F1 {worst_row['Macro F1 평균']:.4f}"),
        ("가장 빠른 모델", str(fastest_row["모델"]),
         f"{fastest_row['평균 학습시간(초)']:.2f}초"),
        ("학습 조건", f"{n_seeds}회 반복",
         f"피처 {n_features}, cw={class_weight or 'None'}"),
    ], color="#16a34a")

    # ③ 비유 — 모델 학습이 무엇인지 설명
    _analogy_card(
        "모델 학습은 '시험 공부'와 같습니다",
        "여러 명의 학생(모델)에게 같은 교재(Train 데이터)로 공부시킨 후, "
        "같은 시험(Test 데이터)을 보게 하는 것과 같습니다. "
        f"이번에는 <b>{len(df)}명의 학생</b>이 <b>{n_seeds}번씩 시험</b>을 봤고, "
        f"<b>{best_row['모델']}</b>이 가장 일관되게 좋은 점수를 얻었습니다. "
        "여러 번 시험을 보는 이유는 '운으로 잘 본 것'이 아닌 '실력으로 잘 본 것'을 가리기 위함입니다.",
    )

    # ④ Q&A
    qa = [
        ("Macro F1이 뭔가요?",
         "**모든 클래스를 똑같이 중요하게 본 평균 점수**입니다 (0~1, 높을수록 좋음). "
         "예를 들어 'A/B/C 3개 클래스'가 있으면 각 클래스의 F1을 단순 평균합니다. "
         "**클래스 불균형이 있을 때 Accuracy(정확도)보다 공정한 지표**입니다."),
        ("class_weight='balanced'는 무슨 의미인가요?",
         "데이터에서 적게 나타나는 클래스(소수 클래스)에 **자동으로 더 큰 가중치**를 부여하여 "
         "모델이 다수 클래스만 무시하지 않게 만듭니다. "
         f"현재 적용: **{class_weight or 'None'}**"),
        ("왜 시드(seed)를 여러 번 바꾸나요?",
         "ML 모델은 학습 시 약간의 무작위성을 갖습니다. "
         "**같은 데이터/모델이라도 시드가 다르면 점수가 살짝 다릅니다.** "
         "여러 시드로 반복하여 평균을 내면 '진짜 실력'을 더 정확히 알 수 있습니다."),
        ("학습 시간이 짧은 모델 vs 정확한 모델, 무엇이 좋나요?",
         "**상황에 따라 다릅니다.** 실시간 예측이 중요하면 빠른 모델, "
         "정확도가 우선이면 느려도 정확한 모델을 선택합니다. "
         "보통 <b>속도와 성능의 균형이 맞는 모델</b>(예: LightGBM)이 실무에서 선호됩니다."),
    ]
    _qa_cards(qa)

    # ⑤ 다음 단계
    _next_steps([
        "사이드바 <b>5. Model Comparison</b>으로 이동해 더 자세한 비교(통계 검정 포함)를 확인하세요.",
        "<b>6. Feature Importance</b>에서 어떤 변수가 예측에 가장 큰 영향을 줬는지 확인할 수 있습니다.",
        "성능이 부족하다면 ① 다른 FE 전략 ② 하이퍼파라미터 튜닝 ③ 모델 앙상블을 시도해보세요.",
    ])


# ════════════════════════════════════════════════════════════════════
# Page 5 — Model Comparison 인사이트
# ════════════════════════════════════════════════════════════════════
def render_comparison_insights(*, agg_df: pd.DataFrame, models: list[str],
                                  n_seeds: int,
                                  friedman_results: list[dict],
                                  cw_uw_compare: bool = False):
    _section_banner(
        "💡 Model Comparison — 가장 중요한 결과 정리",
        "단순 점수가 아닌 '진짜 더 좋은 모델인지'를 통계적으로 판단합니다.",
    )

    # 최고/최저 모델 식별 (macro_f1 기준)
    sub = agg_df[agg_df["metric"] == "macro_f1"].copy()
    if sub.empty or sub["mean"].isna().all():
        st.info("Macro F1 결과가 없어 인사이트를 표시할 수 없습니다.")
        return

    sub = sub.sort_values("mean", ascending=False).reset_index(drop=True)
    best        = sub.iloc[0]
    worst       = sub.iloc[-1]
    best_model  = str(best["model"])
    best_mean   = float(best["mean"])
    best_std    = float(best["std"])
    delta       = float(best["mean"]) - float(worst["mean"])

    # 평균 std (안정성)
    mean_std = float(sub["std"].mean())

    # Friedman — Macro F1 결과 가져오기
    fri_macro_f1 = next(
        (r for r in friedman_results if r.get("metric") == "macro_f1"),
        None,
    )
    fri_p = fri_macro_f1.get("pvalue") if fri_macro_f1 else None
    is_significant = fri_p is not None and fri_p < 0.05

    # ① 한 줄 결론
    if is_significant:
        conclusion = (
            f"<b>{best_model}</b>이 다른 모델보다 통계적으로 유의미하게 더 좋습니다 "
            f"(Macro F1 <b>{best_mean:.4f} ± {best_std:.4f}</b>, "
            f"Friedman p={fri_p:.4f} &lt; 0.05)"
        )
        color = "#16a34a"
    elif fri_p is not None:
        conclusion = (
            f"<b>{best_model}</b>이 가장 높은 점수지만, "
            f"다른 모델과의 차이가 통계적으로 유의하지 않습니다 "
            f"(Macro F1 <b>{best_mean:.4f}</b>, Friedman p={fri_p:.4f} ≥ 0.05)"
        )
        color = "#d97706"
    else:
        conclusion = (
            f"<b>{best_model}</b>이 가장 높은 Macro F1을 기록했습니다 "
            f"(<b>{best_mean:.4f}</b>)"
        )
        color = "#4f46e5"

    _conclusion_box(conclusion, color=color)

    # ② KPI
    stability = "🟢 안정적" if mean_std < 0.02 else ("🟡 다소 변동" if mean_std < 0.05 else "🔴 변동 큼")
    _kpi_grid([
        ("최고 모델", best_model, f"F1 {best_mean:.4f}"),
        ("Top↔Bottom 차이", f"{delta:.4f}",
         "값이 클수록 모델 선택 중요"),
        ("결과 안정성", stability,
         f"평균 σ {mean_std:.4f}"),
        ("통계적 유의성",
         "★ 유의" if is_significant else ("비유의" if fri_p is not None else "—"),
         f"Friedman p={fri_p:.4f}" if fri_p is not None else "—"),
    ], color=color)

    # ③ 비유
    _analogy_card(
        "모델 비교는 '시험 점수의 평균 + 통계 검정'입니다",
        f"학생 5명({n_seeds}회 시드 반복)의 시험 점수만 보면 누가 잘했는지 헷갈릴 수 있습니다. "
        f"<b>Friedman 검정</b>은 '점수 차이가 우연인지, 진짜 실력 차이인지'를 확인하는 방법입니다. "
        + (
            f"<br>이번 결과는 <b>p={fri_p:.4f}</b>로 0.05보다 작으므로, "
            f"<b>'{best_model}이 진짜로 더 좋다'</b>고 자신있게 말할 수 있습니다."
            if is_significant
            else
            f"<br>이번 결과는 <b>p={fri_p:.4f}</b>로 0.05보다 크기 때문에, "
            "현재 데이터로는 모델 간 차이를 단정하기 어렵습니다. "
            "더 많은 데이터/시드로 재실험하거나, 점수가 비슷하면 <b>해석이 쉬운 모델</b>을 선택하는 것도 좋은 방법입니다."
            if fri_p is not None
            else "<br>통계 검정 결과가 없어 추가 비교는 통계 검정 섹션을 확인해주세요."
        ),
    )

    # ④ Q&A
    qa = [
        ("p-value가 뭔가요?",
         "**'관찰된 차이가 우연일 확률'**입니다. **0.05보다 작으면 '우연이 아닐 가능성이 높다(=유의)'**고 봅니다. "
         "예: p=0.001 → 1000번 중 1번만 우연. 따라서 진짜 차이로 판단."),
        ("Confusion Matrix는 어떻게 읽어요?",
         "**대각선(왼쪽 위→오른쪽 아래)의 숫자가 클수록 좋은 모델**입니다. "
         "대각선은 '실제와 예측이 일치한 케이스 수'이고, 그 외는 모두 오류입니다. "
         "어느 클래스에서 오류가 많은지 확인하면 모델의 약점이 보입니다."),
        ("ROC Curve의 면적(AUROC)이 의미하는 건?",
         "**랜덤 추출한 양성 샘플이 음성 샘플보다 높은 점수를 받을 확률**입니다. "
         "0.5는 동전 던지기 수준, 1.0은 완벽. 보통 0.8 이상이면 실용적이라고 봅니다."),
        ("Wilcoxon은 Friedman과 어떻게 다른가요?",
         "**Friedman은 '여러 모델 중 하나라도 다른지'**, "
         "**Wilcoxon은 '두 모델끼리 직접 비교'**합니다. "
         "Bonferroni 보정은 여러 번 비교할 때 거짓 발견을 줄이기 위한 안전장치입니다."),
    ]
    if cw_uw_compare:
        qa.append((
            "보정/미보정 비교에서 '미보정이 더 높을 때'는?",
            "**Macro F1이 떨어지면서 Accuracy/Micro F1이 오른다면, "
            "다수 클래스만 잘 맞히는 편향된 모델**일 가능성이 큽니다. "
            "비즈니스에서 소수 클래스(예: 이탈 고객)도 중요하다면 보정 모델을 선택하세요."
        ))
    _qa_cards(qa)

    # ⑤ 다음 단계
    _next_steps([
        f"<b>6. Feature Importance</b>에서 <b>{best_model}</b>이 어떤 변수를 보고 결정했는지 확인해보세요.",
        "Confusion Matrix에서 자주 헷갈리는 클래스가 있다면, 추가 피처/룰 기반 보정을 검토하세요.",
        "성능이 만족스러우면 <b>7. Report</b>에서 PPT/HTML 리포트를 즉시 생성할 수 있습니다.",
    ])


# ════════════════════════════════════════════════════════════════════
# Page 6 — Feature Importance & SHAP 인사이트
# ════════════════════════════════════════════════════════════════════
def render_fi_insights(*, method: str, model_name: str,
                          imp_df: pd.DataFrame | None = None,
                          shap_summary: pd.DataFrame | None = None,
                          n_features: int = 0,
                          classes: Iterable | None = None):
    _section_banner(
        "💡 Feature Importance — 가장 중요한 결과 정리",
        "어떤 변수가 예측에 가장 큰 영향을 줬는지, 비전공자도 이해할 수 있도록 정리했습니다.",
    )

    method_label = {
        "builtin":     "내장 중요도 (Built-in)",
        "permutation": "Permutation Importance",
        "shap":        "SHAP Value",
    }.get(method, method)

    # SHAP 우선 → 없으면 imp_df
    use_df: pd.DataFrame | None = None
    score_col = None
    direction_col = None
    if shap_summary is not None and not shap_summary.empty:
        use_df = shap_summary.copy()
        score_col = "mean_abs_shap"
        direction_col = "mean_shap" if "mean_shap" in use_df.columns else None
    elif imp_df is not None and not imp_df.empty:
        use_df = imp_df.copy()
        score_col = "importance"

    if use_df is None or score_col is None:
        st.info("중요도 계산을 먼저 실행해주세요.")
        return

    use_df = use_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    top3 = use_df.head(3)
    top1_feat = str(top3.iloc[0]["feature"])
    top1_val  = float(top3.iloc[0][score_col])

    # 상위/하위 기여도 비율 — 80/20 법칙 확인
    n_total_score = float(use_df[score_col].sum())
    if n_total_score > 0:
        top10_pct = use_df.head(10)[score_col].sum() / n_total_score * 100
    else:
        top10_pct = 0.0

    # ① 한 줄 결론
    conclusion = (
        f"<b>{top1_feat}</b>가 예측에 가장 큰 영향을 미칩니다 — "
        f"({method_label} 기준 점수 <b>{top1_val:.4f}</b>)"
    )
    _conclusion_box(conclusion, color="#dc2626")

    # ② KPI — Top 3
    items = []
    for i, row in top3.iterrows():
        feat = str(row["feature"])
        score = float(row[score_col])
        if direction_col is not None and direction_col in row:
            mean_signed = float(row[direction_col])
            sign_emoji = "📈" if mean_signed > 0 else ("📉" if mean_signed < 0 else "↔")
            sub = f"{sign_emoji} 평균 SHAP {mean_signed:+.4f}"
        else:
            sub = "중요도 점수"
        items.append((f"#{i+1} 영향력 변수", feat, sub))
    items.append(("Top 10 누적 기여",
                   f"{top10_pct:.1f}%",
                   "값이 클수록 변수 집중도 높음"))
    _kpi_grid(items, color="#dc2626")

    # ③ 비유
    if method == "shap":
        _analogy_card(
            "SHAP은 '주가 움직임을 종목별로 분해'하는 것과 같습니다",
            f"오늘 KOSPI가 +30포인트 올랐다면, '삼성전자 +12, 현대차 +8, 네이버 +10'처럼 "
            f"<b>각 종목의 기여도</b>를 분해하는 것이 SHAP입니다. "
            f"여기서는 모델이 어떤 예측을 했을 때 <b>{top1_feat}</b>이 가장 크게 기여했음을 의미합니다. "
            "<b>SHAP의 부호(+/-)</b>는 <b>예측을 올렸는지/내렸는지</b>의 방향을 나타내며, "
            "값의 크기가 영향의 세기입니다.",
        )
    elif method == "permutation":
        _analogy_card(
            "Permutation은 '재료를 무작위로 섞어보는 실험'과 같습니다",
            "특정 재료(피처)의 값을 무작위로 섞었을 때 음식 맛이 얼마나 떨어지는지 확인하는 것과 같습니다. "
            "<b>맛이 크게 떨어지면 그 재료가 중요</b>한 것이고, "
            "<b>변화가 거의 없으면 그 재료는 별 영향이 없는 것</b>입니다. "
            "단, 재료끼리 강하게 연결되어 있으면(다중공선성) 이 방법이 과소평가될 수 있습니다.",
        )
    else:  # builtin
        _analogy_card(
            "내장 중요도는 '모델 내부의 자체 평가'입니다",
            "트리 모델은 <b>어떤 변수를 자주 분기 기준으로 썼는지</b>, "
            "선형 모델은 <b>계수의 절댓값</b>으로 중요도를 매깁니다. "
            "계산이 빠르지만 <b>모델 종류마다 의미가 달라</b> 모델 간 비교 시 주의가 필요합니다. "
            "더 신뢰할 수 있는 비교를 원하면 <b>SHAP</b>을 사용하세요.",
        )

    # ④ 80/20 인사이트
    if top10_pct >= 80:
        st.markdown(
            f"""<div style="background:#f0f9ff;border-left:4px solid #2563eb;
                          border-radius:10px;padding:14px 22px;margin:10px 0;">
              <div style="font-weight:700;color:#1d4ed8;margin-bottom:6px;">
                📐 파레토 법칙 발견 (80/20)
              </div>
              <div style="color:#1f2937;font-size:14px;line-height:1.7;">
                전체 변수의 상위 10개가 <b>{top10_pct:.1f}%</b>의 기여를 차지합니다. <br>
                즉 <b>나머지 변수의 영향력은 미미</b>하므로, 이 Top 10만으로
                간소화된 모델을 만들 수 있습니다 (해석 + 운영 효율 모두 향상).
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ⑤ Q&A
    qa = [
        ("같은 변수인데 모델마다 중요도가 다른 이유?",
         "**모델마다 학습 방식이 다르기 때문**입니다. 트리 모델은 분기 횟수, 선형 모델은 계수 크기로 평가합니다. "
         "**여러 모델에서 공통으로 상위에 나오는 변수**가 진짜 중요한 변수입니다."),
        ("SHAP에서 양수/음수는 뭘 의미하나요?",
         "**양수 = 해당 클래스의 예측 확률을 올리는 방향**, **음수 = 내리는 방향**으로 기여했다는 뜻입니다. "
         "예: SHAP=+0.3이면 그 변수 때문에 '해당 클래스일 확률'이 0.3만큼 증가했다는 의미."),
        ("중요도가 0에 가까운 변수는 빼도 되나요?",
         "**보통 빼도 됩니다**. 단, ① 도메인 상 중요한 변수일 수 있고 ② 다른 변수와 상호작용이 있을 수 있으므로 "
         "한 번에 다 빼지 말고 **하나씩 검증하면서 제거**하는 것이 안전합니다."),
        ("위험 고객 시뮬레이터는 어떻게 활용하나요?",
         "**'특정 변수가 일정 임계치를 넘으면 위험 등급으로 바뀌는지' 확인**할 수 있습니다. "
         "비즈니스 룰을 만들기 전 **시각적 검증 도구**로 사용하세요."),
    ]
    _qa_cards(qa)

    # ⑥ 다음 단계
    _next_steps([
        f"<b>{top1_feat}</b>의 측정/관리 방법을 비즈니스 측면에서 점검하세요 (가장 큰 영향력).",
        f"상위 3개({', '.join(top3['feature'].tolist())})를 중심으로 <b>도메인 전문가 검토</b>를 받으면 신뢰도가 높아집니다.",
        "<b>7. Report</b>에서 이 결과를 PPT/HTML로 즉시 출력해 의사결정자에게 공유할 수 있습니다.",
    ])
