"""리포트 정적 콘텐츠 — 도메인 정의, 추가 과제 제안, 인사이트 가이드.

기존 HTML 리포트의 핵심 내용을 코드로 옮긴 모듈. Page 7과 HTML 다운로드 리포트가
동일한 콘텐츠를 공유합니다.
"""
from __future__ import annotations

# ════════════════════════════════════════════════════════════════════
# 1. 도메인 정의 (기본값) — 변수 접두사 → 영역 / 의미
# ════════════════════════════════════════════════════════════════════
DEFAULT_DOMAIN_MAP: dict[str, dict] = {
    "t1": {"name": "신기술", "color": "#2563eb",
            "desc": "LG의 혁신 기술 역량에 대한 고객 평가"},
    "t2": {"name": "개발",   "color": "#16a34a",
            "desc": "제품 개발 프로세스·협력에 대한 고객 평가"},
    "c":  {"name": "Cost",   "color": "#ea580c",
            "desc": "비용·가격 경쟁력에 대한 고객 평가"},
    "d":  {"name": "공급",   "color": "#9333ea",
            "desc": "납기·공급 안정성에 대한 고객 평가"},
    "q1": {"name": "품질",   "color": "#e11d48",
            "desc": "제품·서비스 품질에 대한 고객 평가"},
    "q2": {"name": "서비스", "color": "#ca8a04",
            "desc": "사후 서비스·지원에 대한 고객 평가"},
}

INDEX_DEFINITIONS = {
    "CSI": "Customer Satisfaction Index — 영역에 대한 만족도 지수 (예: t1_csi_total)",
    "CCI": "Customer Credibility Index — 영역에 대한 신뢰도 지수 (예: c_cci_total)",
    "_total": "세부 항목(_res 대응성, _core 핵심역량, _comm 커뮤니케이션)의 종합 지수",
}


def detect_domain(feature: str, domain_map: dict[str, dict] | None = None) -> dict | None:
    """피처명에서 영역 접두사를 추출."""
    domain_map = domain_map or DEFAULT_DOMAIN_MAP
    for prefix in sorted(domain_map.keys(), key=len, reverse=True):
        if feature == prefix or feature.startswith(prefix + "_"):
            return {"prefix": prefix, **domain_map[prefix]}
    return None


# ════════════════════════════════════════════════════════════════════
# 2. 추가 과제 제안 — 4개 카테고리
# ════════════════════════════════════════════════════════════════════

ADDITIONAL_TASKS = {
    "eda": {
        "title": "EDA 후속 분석 로드맵",
        "icon": "🔍",
        "high": [
            {
                "name": "_total 12변수 전용 단순화 모델 구축",
                "background": "Top 중요도 변수 8/10이 _total 계열, 세부 항목은 r≥0.9 다중공선성",
                "method": "CSI total 6 + CCI total 6으로 모델 재학습 후 성능 비교",
                "outcome": "변수 51→12 압축 + 해석 가능한 경량 운영 모델",
            },
            {
                "name": "고객사별 RPI 추세 분석 (패널 데이터)",
                "background": "다년간 시계열 구조 + 고객사별 RPI 분포 편차",
                "method": "고객사 × 연도 패널 → 선형혼합모형 또는 고객사 고정효과 모델",
                "outcome": "고객사별 RPI 변화 트렌드 + 악화 조기경보 기준",
            },
            {
                "name": "소수 클래스(위험 고객) 심층 프로파일링",
                "background": "RPI 4~5 = 관계 위험 고객사, 예측 어려움",
                "method": "위험 고객사 특성 탐색 + 이진 분류(위험/정상)",
                "outcome": "위험 고객 조기 탐지 이진 분류기 + 특성 프로필",
            },
        ],
        "mid": [
            {
                "name": "CSI ↔ CCI 인과 구조 분석",
                "background": "CSI(만족)와 CCI(신뢰) 간 상관 높음 + RPI 기여 방향 상이",
                "method": "구조방정식모형(SEM) 또는 매개효과 분석 (만족→신뢰→재구매)",
                "outcome": "CSI-CCI-RPI 인과 다이어그램 + 개입 포인트 식별",
            },
            {
                "name": "측정 시점(t1/t2/c/q1/q2) 효과 검증",
                "background": "t1 CCI total 1위 — 특정 시점 측정값 예측력 차이 가능",
                "method": "시점별 모델 독립 학습 → AUC 비교, 정보 감소 분석",
                "outcome": "최적 측정 시점 가이드 + 불필요 측정 주기 제거",
            },
            {
                "name": "제품군 / 지역 서브그룹 모델 비교",
                "background": "product·area 변수 자체는 중요도 낮으나 그룹별 패턴 차이 가능성",
                "method": "그룹별 분리 학습 → 성능·중요도 비교",
                "outcome": "그룹별 맞춤 예측 모델 타당성 보고서",
            },
        ],
        "low": [
            {
                "name": "외부 데이터 연계 분석",
                "background": "고객사 산업 경기·실적이 RPI에 영향 가능",
                "method": "고객사 재무·공시 데이터 크롤링 → RPI 상관 분석",
                "outcome": "거시경제·고객사 실적과 RPI 연관성 보고서",
            },
            {
                "name": "이상치 고객사 케이스 스터디",
                "background": "박스플롯에서 극단 이상치로 탐지된 고객사 존재",
                "method": "이상치 고객사 선별 → CSI/CCI 세부 항목 원인 분석",
                "outcome": "이상 고객사 대응 매뉴얼 + 케이스 스터디 문서",
            },
        ],
    },

    "fe": {
        "title": "Feature Engineering 후속 로드맵",
        "icon": "🛠",
        "high": [
            {
                "name": "_total 12변수 단독 실험",
                "background": "Feature Selection이 48개 선택 → 사실상 전체와 유사, 세부항목 중복",
                "method": "CSI total 6 + CCI total 6 = 12변수만으로 동일 모델 재실험",
                "outcome": "변수 76% 감소 + 성능 유지 여부 확인",
            },
            {
                "name": "PCA 컴포넌트 수 민감도 분석",
                "background": "95% 분산 기준 25 PC → 최적값 미검증",
                "method": "n_components = 5/10/15/20/25/30/95% 별 성능 곡선",
                "outcome": "최소 PC 수 대비 성능 trade-off → 운영 복잡도 최소화",
            },
            {
                "name": "Incremental Feature Addition 실험",
                "background": "어떤 변수가 추가될 때 성능이 뛰는지 미확인",
                "method": "앙상블 순위 1위부터 순차 추가하며 Macro AUROC 변화 추적",
                "outcome": "최소 필요 변수 집합(Elbow Point) 식별",
            },
        ],
        "mid": [
            {
                "name": "PCA 대비 UMAP / Kernel PCA 비교",
                "background": "PCA(선형) 최고 성능 → 비선형 압축 미탐색",
                "method": "UMAP / Kernel PCA(rbf) 동일 모델 비교",
                "outcome": "비선형 구조 존재 시 추가 성능 향상",
            },
            {
                "name": "Domain-Knowledge 파생 변수 실험",
                "background": "CCI − CSI 차이(신뢰 대비 만족 갭) 등 비즈니스 의미 변수 미시도",
                "method": "CCI_total − CSI_total, 분기 변화율(Δ) 등 파생 변수",
                "outcome": "원본보다 압축된 도메인 특화 시그널 발굴",
            },
            {
                "name": "Target Encoding (고객사 레이블)",
                "background": "client 변수 중요도 높음 — 단순 Label Encoding 한계",
                "method": "client별 RPI 평균 기반 Target Encoding + LOO CV",
                "outcome": "고객사 히스토리 정보 반영 → Macro F1 향상",
            },
        ],
        "low": [
            {
                "name": "AutoML 기반 자동 피처 탐색",
                "background": "수동 FS + PCA 외 탐색 공간 미개척",
                "method": "FLAML / Optuna + RFECV 자동 서브셋 탐색",
                "outcome": "인간 직관 넘는 최적 피처 조합 발견",
            },
            {
                "name": "시계열 피처 (Lag / Rolling 통계)",
                "background": "다년간 데이터 — 직전 분기 CCI가 예측에 유용 가능",
                "method": "고객사별 Lag-1/Lag-2, Rolling Mean 변수 생성",
                "outcome": "시계열 패턴 활용으로 예측 정확도 향상",
            },
        ],
    },

    "model": {
        "title": "모델 비교 실험 후속 로드맵",
        "icon": "🤖",
        "high": [
            {
                "name": "Threshold Optimization (분류 임계값 조정)",
                "background": "보정 후에도 RPI 4~5 재현율 개선 여지",
                "method": "ROC 기반 클래스별 임계값 최적화 (Youden's J / F2-score)",
                "outcome": "위험 고객 재현율 향상 + 실무 허용 FP율 조절",
            },
            {
                "name": "Ordinal Classification 실험",
                "background": "RPI 1→5는 서수형 — 일반 다중분류는 순서 무시",
                "method": "OrdinalClassifier (mord) 또는 누적 로짓 모델 비교",
                "outcome": "RPI 2→4 혼동 감소, 인접 클래스 페널티 반영",
            },
            {
                "name": "Calibration 검증 (확률 보정)",
                "background": "Macro AUROC 높음 — 출력 확률의 실제 비율 일치 미검증",
                "method": "Reliability Diagram + ECE 측정, 필요시 Isotonic / Platt",
                "outcome": "확률 기반 위험 순위화 신뢰도 확보",
            },
        ],
        "mid": [
            {
                "name": "5-class 완전 평가 셋 구축",
                "background": "현재 Test에 RPI 1 클래스 없음 → 4-class 기준 지표만",
                "method": "RPI 1 포함 균형 Test 셋 또는 k-fold 전체 클래스 보장",
                "outcome": "운영 환경 반영 5-class 완전 성능 평가",
            },
            {
                "name": "모델 앙상블 (Stacking / Voting)",
                "background": "LR(안정성)+LGB(AUROC)+RF(다양성) 강점 상이",
                "method": "Soft Voting 또는 LR 메타 분류기 Stacking",
                "outcome": "단일 모델 대비 Macro F1 추가 향상",
            },
            {
                "name": "하이퍼파라미터 최적화",
                "background": "기본값 사용 — 최적값 미탐색",
                "method": "Optuna Bayesian Optimization + Stratified K-Fold CV",
                "outcome": "각 모델 1~2% 추가 성능 향상",
            },
            {
                "name": "SMOTE vs class_weight 정량 비교",
                "background": "현재는 class_weight만 실험 — SMOTE 미비교",
                "method": "SMOTE / ADASYN / BorderlineSMOTE 5-seed 비교",
                "outcome": "불균형 처리 전략별 효과 문서화",
            },
        ],
        "low": [
            {
                "name": "온라인 학습 / 점진적 업데이트",
                "background": "연간 신규 데이터 누적 — 매년 재학습 필요",
                "method": "Incremental Learning(SGD/River) 또는 연간 자동 재학습",
                "outcome": "데이터 드리프트 대응 + 운영 지속성",
            },
            {
                "name": "모델 공정성 (Fairness) 검증",
                "background": "고객사·제품군별 예측 성능 차이 가능성",
                "method": "서브그룹별 Macro F1 분해 분석",
                "outcome": "특정 그룹 불이익 탐지 + 공정 서비스 보장",
            },
        ],
        "checklist": [
            "Threshold 최적화 완료",
            "Calibration 검증 (ECE < 0.05)",
            "5-class 완전 평가",
            "하이퍼파라미터 최적화",
            "연간 재학습 파이프라인",
            "서브그룹 공정성 검증",
        ],
    },

    "fi": {
        "title": "Feature Importance & SHAP 후속 로드맵",
        "icon": "🎯",
        "high": [
            {
                "name": "SHAP Value 개별 예측 설명 (대시보드 통합)",
                "background": "내장·Permutation은 전역(global) — 개별 고객 설명 불가",
                "method": "SHAP TreeExplainer + Waterfall/Force plot",
                "outcome": "특정 고객의 RPI 4 예측 사유 영업팀 전달 가능",
            },
            {
                "name": "Top-15 단독 모델 재학습 검증",
                "background": "앙상블 Top-15 이후 점수 급락 (Elbow 추정)",
                "method": "Top-5/10/15/25/전체 별 Macro F1·AUROC 비교",
                "outcome": "최소 피처 수 + 운영 설문 문항 축소 근거",
            },
            {
                "name": "비중요 Bottom-N 제거 효과 검증",
                "background": "Permutation ≤ 0 + KW η²≈0 이중 기준 충족",
                "method": "Bottom 제거 후 5-seed 재실험 → 성능 변화 측정",
                "outcome": "노이즈 제거 → 학습 속도/과적합 위험 감소",
            },
        ],
        "mid": [
            {
                "name": "SHAP Interaction Value 분석",
                "background": "CCI total × CSI total 상호작용이 RPI에 미치는 효과 미확인",
                "method": "SHAP Interaction Values + Partial Dependence Plot",
                "outcome": "복합 규칙(예: t1_cci 낮고 c_csi 낮으면 RPI 5) 발견",
            },
            {
                "name": "측정 시점별 중요도 변화 추적",
                "background": "t1 CCI 1위 — 시점별 중요 변수 변화 미확인",
                "method": "시점 데이터 단독 학습 → 시점별 Top-5 비교",
                "outcome": "최적 측정 시점 + 불필요 시점 제거 근거",
            },
            {
                "name": "피처 중요도 기반 가중 만족도 지수(WCI) 설계",
                "background": "앙상블 점수가 설문 문항의 비즈니스 우선순위와 연계",
                "method": "앙상블 점수 → 문항 가중치 → 가중 만족도 지수",
                "outcome": "단순 평균 대신 RPI 예측력 반영 KPI",
            },
        ],
        "low": [
            {
                "name": "피처 중요도 시계열 안정성 (Longitudinal)",
                "background": "다년 데이터 — 연도별 중요 피처 변화 미확인",
                "method": "연도별 분리 학습 → Top-N Jaccard 유사도",
                "outcome": "중요도 드리프트 감지 + 재학습 트리거",
            },
            {
                "name": "고객사 군집 기반 중요도 비교",
                "background": "고객사 유형별로 다른 변수가 중요할 가능성",
                "method": "k-means on CCI/CSI profile → 군집별 중요 피처 비교",
                "outcome": "군집별 맞춤 CS 전략",
            },
            {
                "name": "피처 중요도 모니터링 대시보드 자동화",
                "background": "현재 정적 분석 — 신규 데이터 자동 갱신 불가",
                "method": "Streamlit / Dash 기반 자동 재분석 파이프라인",
                "outcome": "데이터 갱신 시 자동 알림 + 지속 운영 자산",
            },
        ],
    },
}


# ════════════════════════════════════════════════════════════════════
# 3. 인사이트 가이드 — 사용자 데이터 결과를 해석하는 일반 원칙
# ════════════════════════════════════════════════════════════════════

INSIGHT_GUIDES = {
    "eda": {
        "title": "EDA 핵심 인사이트",
        "items": [
            "**클래스 분포**: 불균형 비율 4:1 이상이면 `class_weight='balanced'` 또는 SMOTE 필수",
            "**상관관계**: |r| ≥ 0.9 쌍이 다수면 PCA 또는 변수 선택으로 다중공선성 해소 권장",
            "**KW η²**: 0.1 이상이면 단변량 수준에서도 강한 클래스 분리 능력",
            "**박스플롯 중첩**: 클래스 간 박스가 겹치면 해당 변수의 변별력 낮음",
        ],
    },
    "fe": {
        "title": "Feature Engineering 핵심 인사이트",
        "items": [
            "**전체 사용**: 다중공선성 미해소 → 트리 모델은 영향 적으나 LR은 불안정",
            "**Feature Selection**: MI + Pearson 결합이 일관 신호 + 다중공선성 해소",
            "**PCA**: 분산 95% 기준 컴포넌트 수가 적정 — 과도하면 정보 손실, 부족하면 다중공선성",
            "**비교 권장**: 동일 모델·시드로 4 전략 모두 학습 후 Macro AUROC 기준 선택",
        ],
    },
    "model": {
        "title": "모델 비교 핵심 인사이트",
        "items": [
            "**LR**: 정규화 후 안정성·해석력·속도 우수 — 운영 1순위 후보",
            "**LGB**: Macro AUROC 최고 — 확률 기반 위험 순위화에 적합",
            "**RF**: 보정 적용 시 Std 급감 — 분산 큰 데이터에 강건성 향상",
            "**class_weight 보정**: Macro 지표 향상, Micro 소폭 하락 가능 — trade-off 인지",
            "**Friedman p<0.05**: 모델 간 통계적으로 유의한 차이 → Wilcoxon으로 쌍별 비교",
        ],
    },
    "fi": {
        "title": "Feature Importance & SHAP 핵심 인사이트",
        "items": [
            "**3모델 일치 (Spearman ρ > 0.7)**: 방법론 독립적으로 신뢰할 수 있는 핵심 피처",
            "**SHAP 양수 기여**: 피처 값↑ → 클래스 확률↑ (긍정 신호)",
            "**SHAP 음수 기여**: 피처 값↑ → 클래스 확률↓ (위험 신호 가능)",
            "**Beeswarm 색상 분리**: 빨강(고값)이 +SHAP, 파랑(저값)이 −SHAP이면 강한 단조 관계",
            "**Waterfall**: 개별 고객의 예측 근거를 원인 변수 단위로 설명 → CRM 활용",
        ],
    },
}


# ════════════════════════════════════════════════════════════════════
# 4. HTML 렌더링 헬퍼
# ════════════════════════════════════════════════════════════════════

def tasks_to_html(category: str) -> str:
    """추가 과제 카테고리를 HTML 카드로 렌더링."""
    cat = ADDITIONAL_TASKS[category]

    def section(level: str, color: str, label: str, items: list[dict]) -> str:
        if not items:
            return ""
        rows = "\n".join(
            f"<tr><td><strong>{it['name']}</strong></td>"
            f"<td>{it['background']}</td>"
            f"<td>{it['method']}</td>"
            f"<td>{it['outcome']}</td></tr>"
            for it in items
        )
        return f"""
<h3 style="color:{color}; margin:14px 0 8px;">{label}</h3>
<table>
  <thead><tr><th>과제</th><th>배경</th><th>방법</th><th>기대 효과</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""

    parts = [
        f"<h2>{cat['icon']} {cat['title']}</h2>",
        section("high", "#dc2626", "🔴 High Priority — 즉시 착수", cat.get("high", [])),
        section("mid",  "#d97706", "🟡 Mid Priority — 1~2개월", cat.get("mid", [])),
        section("low",  "#16a34a", "🟢 Low Priority / 탐색적", cat.get("low", [])),
    ]
    if "checklist" in cat:
        items = " &nbsp;|&nbsp; ".join(f"☐ {x}" for x in cat["checklist"])
        parts.append(
            f"<div style='background:#ecfeff; border-radius:8px; padding:12px 16px; "
            f"margin-top:12px; font-size:12px; border-left:4px solid #0891b2;'>"
            f"<strong>운영 모델 배포 체크리스트</strong><br>{items}</div>"
        )
    return "\n".join(p for p in parts if p)


def insight_to_html(category: str) -> str:
    g = INSIGHT_GUIDES[category]
    items = "".join(f"<li>{x}</li>" for x in g["items"])
    return f"""
<div style="background:#f0f9ff; border-radius:8px; padding:14px 18px;
     border-left:4px solid #0891b2; font-size:13px; line-height:1.9;">
  <strong>{g['title']}</strong>
  <ul style="margin-top:6px; margin-left:18px;">{items}</ul>
</div>"""


def domain_legend_html(domain_map: dict[str, dict] | None = None) -> str:
    """도메인 정의 카드 HTML."""
    domain_map = domain_map or DEFAULT_DOMAIN_MAP
    cards = []
    for prefix, info in domain_map.items():
        cards.append(
            f"<div style='background:white; border-left:4px solid {info['color']}; "
            f"border-radius:8px; padding:10px 14px;'>"
            f"<strong style='color:{info['color']};'>{prefix}</strong> — {info['name']}<br>"
            f"<span style='font-size:11px; color:#475569;'>{info['desc']}</span>"
            f"</div>"
        )
    grid = "<div style='display:grid; grid-template-columns:repeat(3,1fr); gap:10px;'>" \
           + "".join(cards) + "</div>"
    defs = "".join(
        f"<li><strong>{k}</strong>: {v}</li>" for k, v in INDEX_DEFINITIONS.items()
    )
    return f"""
<h3>📐 영역(도메인) 정의</h3>
{grid}
<ul style="margin-top:12px; font-size:12px; line-height:1.8; color:#334155;">
  {defs}
</ul>"""
