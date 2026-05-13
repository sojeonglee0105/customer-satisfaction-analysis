# 고객만족도 분석 대시보드

LG Vibe B2B 고객만족도 데이터 분석 프로젝트입니다.

## 주요 지표

| 지표 | 설명 | 범위 |
|------|------|------|
| **CSI** | 고객만족지수 (Customer Satisfaction Index) | 0 ~ 10점 (높을수록 만족) |
| **CCI** | 경쟁력지수 (Customer Competitive Index) | -5 ~ +5점 (높을수록 경쟁력 우위) |
| **RPI** | 관계성과지수 (Relationship Performance Index) | 1 ~ 5등급 (낮을수록 우수) |

## 파일 구성

```
├── customer_satisfaction_full_columns.xlsx  # 원본 데이터
├── build_dashboard.py                       # 대시보드 생성 스크립트
├── html_to_pdf.py                           # PDF 변환 스크립트
├── reports/
│   ├── customer_satisfaction_dashboard.html # 인터랙티브 대시보드
│   ├── customer_satisfaction_report.pdf     # PDF 보고서
│   ├── EDA_report.html                      # EDA 보고서
│   └── EDA_report.md                        # EDA 요약
```

## 핵심 인사이트

- **CSI/CCI ↑ → RPI ↓** : 만족도·경쟁력이 높을수록 관계 등급이 우수해지는 음의 상관관계
- **클레임(C) 영역 역설** : 클레임 처리가 잘 된 고객군이 오히려 가장 우수한 RPI를 기록 (서비스 회복 효과)
- **품질(Q) 영역** : CCI가 가장 높아 경쟁력 우위 핵심 영역

## 실행 방법

```bash
pip install pandas openpyxl numpy playwright
python -m playwright install chromium
python build_dashboard.py        # HTML 대시보드 생성
python html_to_pdf.py            # PDF 변환
```

## GitHub Pages

[대시보드 바로 보기](https://sojeonglee0105.github.io/customer-satisfaction-analysis/reports/customer_satisfaction_dashboard.html)
