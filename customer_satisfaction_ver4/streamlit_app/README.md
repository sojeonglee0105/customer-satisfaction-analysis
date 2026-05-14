# Customer Satisfaction ML Dashboard

End-to-end Streamlit 대시보드 — 사용자가 데이터 파일을 업로드하고 EDA → 피처 엔지니어링 → 모델 학습/비교 → SHAP 분석 → 종합 리포트까지 대화형으로 수행할 수 있습니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 구조

```
streamlit_app/
├── app.py                       # 홈 (워크플로우 안내)
├── pages/
│   ├── 1_Upload.py              # 데이터 업로드 + 변수 선택
│   ├── 2_EDA.py                 # 탐색적 데이터 분석
│   ├── 3_Feature_Engineering.py # 피처 엔지니어링 전략
│   ├── 4_Model_Training.py      # 모델 학습 및 하이퍼파라미터
│   ├── 5_Model_Comparison.py    # 모델 성능 비교 + 통계 검정
│   ├── 6_Feature_Importance.py  # 중요도 + SHAP 분석
│   └── 7_Report.py              # 종합 리포트 + 다운로드
└── utils/
    ├── data_utils.py            # 파일 로드, 인코딩, 분할
    ├── model_utils.py           # 모델 팩토리, 학습, 평가, 통계
    ├── viz_utils.py             # Plotly 차트 팩토리
    └── shap_utils.py            # SHAP explainer / 시각화 래퍼
```

## 워크플로우

1. **Upload** → CSV/Excel 업로드 → 종속/독립 변수 선택
2. **EDA** → 분포, 상관관계, 다중공선성, Kruskal-Wallis 단변량 검정
3. **Feature Engineering** → All / Selection / PCA / 직접선택
4. **Model Training** → 4종 모델 + class_weight + 하이퍼파라미터
5. **Model Comparison** → 5-seed Mean±Std + Friedman/Wilcoxon 통계 검정
6. **Feature Importance** → 내장 / Permutation / SHAP 통합 분석
7. **Report** → KPI 대시보드 + HTML/Excel 다운로드

## 추가 기능

- **Risk Simulator** (Page 6): 슬라이더로 피처 값 입력 → 실시간 RPI 예측
- **Batch Prediction** (Page 7): 신규 CSV 업로드 → 예측 결과 다운로드
- **결측치 대체 옵션** (Page 1): Median / Mean / KNN 선택
