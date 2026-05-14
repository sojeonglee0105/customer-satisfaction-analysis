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

## Streamlit Cloud 배포 가이드

### 1. 사전 준비 (이미 완료됨)
- ✅ Public GitHub repository (`sojeonglee0105/customer-satisfaction-analysis`)
- ✅ `requirements.txt` (의존성 명시)
- ✅ `runtime.txt` (Python 3.11 고정)
- ✅ `.streamlit/config.toml` (테마/업로드 한도)

### 2. 배포 단계

1. **[share.streamlit.io](https://share.streamlit.io)** 접속 → GitHub 계정으로 로그인
2. 우측 상단 **`New app`** 클릭
3. 아래 정보 입력:

   | 필드 | 값 |
   |------|-----|
   | **Repository** | `sojeonglee0105/customer-satisfaction-analysis` |
   | **Branch** | `main` |
   | **Main file path** | `customer_satisfaction_ver4/streamlit_app/app.py` |
   | **App URL** *(선택)* | 예: `lgd-csat-platform` |

4. **`Advanced settings`** → Python version: `3.11` 확인 (runtime.txt가 자동 반영됨)
5. **`Deploy!`** 클릭 → 약 3~5분 후 빌드 완료
6. 발급된 URL(`https://<app>.streamlit.app`)로 접속

### 3. 배포 후 운영 팁

- **자동 재배포**: `main` 브랜치에 push하면 즉시 자동 재배포됩니다.
- **로그 확인**: 앱 우측 하단 `Manage app` → `Logs`에서 빌드/런타임 로그 확인.
- **리소스 한도** (Free tier): RAM 1GB, Storage 1GB. 본 앱(샘플 데이터 1500행)은 충분히 동작합니다.
- **앱 sleep**: 7일 미사용 시 hibernate. 다시 접속하면 ~30초 내 wake-up.
- **재부팅이 필요할 때**: `Manage app` → `Reboot app`.

### 4. 트러블슈팅

| 증상 | 해결 |
|------|------|
| `ModuleNotFoundError` | `requirements.txt`에 패키지 추가 후 commit/push |
| `kaleido` 관련 PPTx 이미지 export 실패 | 일시적 문제 — Reboot 또는 잠시 후 재시도 |
| 메모리 초과 | 데이터 크기 축소 또는 `@st.cache_data` 적용 확인 |
| 한글 깨짐 | 폰트 자동 fallback 사용 중. 필요 시 `packages.txt`로 `fonts-nanum` 추가 |

