# 고객 만족도 EDA 종합 보고서

- **데이터 파일**: `customer_satisfaction_ver2/customer_satisfaction_full_columns.xlsx`
- **시트**: `synthetic_data`
- **타겟 컬럼 (Y_Class = RPI)**: `rpi`
- **폰트 설정**: `Malgun Gothic`

> 문서에서 **Y_Class**라고 하면 본 데이터에서는 **`rpi`** 컬럼(고객 만족도 RPI 등급)을 가리킵니다.

## 데이터 구조 (도메인 정의)

- **조사 년도** (`year`): 수치형 — 2023, 2024, 2025
- **평가영역** (`area`): 범주형 — t1, t2, d, c, q1, q2
- **제품군** (`product`): 범주형 — 모니터, 노트북, TV, 스마트폰, 자동차
- **고객사명** (`client`): 범주형 — a ~ e
- **RPI / Y_Class** (`rpi`): 범주형 — **1(관계 매우 우수) ~ 5(관계 매우 불량)**
- **CSI**: 수치형 — 0(매우 낮음) ~ 10(매우 높음)
- **CCI**: 수치형 — -5(매우 낮음) ~ +5(매우 높음)

## 1. 데이터 요약

```
1500 x 53
```

### 컬럼 목록

- `year`
- `area`
- `product`
- `client`
- `rpi`
- `t1_csi_res`
- `t1_csi_core`
- `t1_csi_comm`
- `t1_csi_total`
- `t1_cci_res`
- `t1_cci_core`
- `t1_cci_comm`
- `t1_cci_total`
- `t2_csi_res`
- `t2_csi_core`
- `t2_csi_comm`
- `t2_csi_total`
- `t2_cci_res`
- `t2_cci_core`
- `t2_cci_comm`
- `t2_cci_total`
- `d_csi_res`
- `d_csi_core`
- `d_csi_comm`
- `d_csi_total`
- `d_cci_res`
- `d_cci_core`
- `d_cci_comm`
- `d_cci_total`
- `c_csi_res`
- `c_csi_core`
- `c_csi_comm`
- `c_csi_total`
- `c_cci_res`
- `c_cci_core`
- `c_cci_comm`
- `c_cci_total`
- `q1_csi_res`
- `q1_csi_core`
- `q1_csi_comm`
- `q1_csi_total`
- `q1_cci_res`
- `q1_cci_core`
- `q1_cci_comm`
- `q1_cci_total`
- `q2_csi_res`
- `q2_csi_core`
- `q2_csi_comm`
- `q2_csi_total`
- `q2_cci_res`
- `q2_cci_core`
- `q2_cci_comm`
- `q2_cci_total`

### 기본 통계 (수치형 처음 30개)

→ `outputs/tables/numeric_describe_first30.csv` 참고

## 2. 타겟(Y_Class = RPI / `rpi` 컬럼) 분석

```
     count  percent
rpi                
1      520    34.67
2      430    28.67
3      260    17.33
4      180    12.00
5      110     7.33
```

### 시각화

- **RPI 분포(1=우수~5=불량)**: [outputs/plots/01_rpi_class_bar_pie.png](../outputs/plots/01_rpi_class_bar_pie.png)
- **고객사별 RPI (낮을수록 우수)**: [outputs/plots/02_rpi_by_client_boxplot.png](../outputs/plots/02_rpi_by_client_boxplot.png)

## 3. 수치형 변수

- **변수 범위 상위 20 (원 척도)**: [outputs/plots/03_numeric_range_top20.png](../outputs/plots/03_numeric_range_top20.png)
- 상수(또는 유일값)에 가까운 변수: `outputs/tables/constant_like_columns.json`
  - 탐지된 상수형 변수: `없음`
- CSI·CCI 열 Z-score 적용 내역: `outputs/tables/csi_cci_zscore_summary.json`

## 4. 상관관계 및 다중공선성

> **RPI 해석**: 1은 관계가 매우 우수, 5는 매우 불량입니다.
> **상관·히트맵**은 CSI·CCI에 **열별 Z-score**를 적용한 뒤 산출했으며,
> 그 외 수치형은 **원 스케일**을 유지합니다.
> 박스플롯은 해석을 위해 **CSI·CCI 원 척도**를 사용합니다.

### RPI와 상관 상위 20개

- `t1_cci_total` : r = -0.9404
- `t1_csi_total` : r = -0.9382
- `d_csi_total` : r = -0.9357
- `c_cci_total` : r = -0.9356
- `d_cci_total` : r = -0.9351
- `q1_csi_total` : r = -0.9338
- `q2_csi_total` : r = -0.9335
- `q1_cci_total` : r = -0.9333
- `c_csi_total` : r = -0.9330
- `t2_csi_total` : r = -0.9321
- `q2_cci_total` : r = -0.9304
- `t2_cci_total` : r = -0.9292
- `q2_csi_comm` : r = -0.8952
- `q1_csi_comm` : r = -0.8923
- `c_csi_core` : r = -0.8922
- `t1_csi_core` : r = -0.8913
- `q1_csi_core` : r = -0.8891
- `t1_csi_comm` : r = -0.8889
- `d_csi_res` : r = -0.8889
- `q1_csi_res` : r = -0.8886

### 시각화

- **RPI 상관 상위 20 (CSI·CCI Z)**: [outputs/plots/04_corr_with_target_top20.png](../outputs/plots/04_corr_with_target_top20.png)
- **상위 30 히트맵 (CSI·CCI Z)**: [outputs/plots/05_heatmap_top30.png](../outputs/plots/05_heatmap_top30.png)

### 다중공선성 체크 (|r| ≥ 0.9)

- `t1_csi_res` & `t1_csi_total` : r = 0.933
- `t1_csi_core` & `t1_csi_total` : r = 0.9344
- `t1_csi_comm` & `t1_csi_total` : r = 0.9301
- `t1_csi_total` & `t1_cci_total` : r = 0.9074
- `t1_csi_total` & `t2_csi_total` : r = 0.9194
- `t1_csi_total` & `d_csi_total` : r = 0.9186
- `t1_csi_total` & `d_cci_total` : r = 0.9056
- `t1_csi_total` & `c_csi_total` : r = 0.92
- `t1_csi_total` & `c_cci_total` : r = 0.9013
- `t1_csi_total` & `q1_csi_total` : r = 0.9159

→ 전체 목록: `outputs/tables/multicollinearity_pairs.csv`

## 5. 클래스별 분포 비교

- **박스플롯 상위 6 (CSI·CCI 원 척도)**: [outputs/plots/06_boxplot_top6_by_class.png](../outputs/plots/06_boxplot_top6_by_class.png)

## 6. 인사이트

- **Y_Class**(타겟)는 엑셀 **`rpi`** 열이며, **1=관계 우수 · 5=관계 불량**으로 해석합니다.
- RPI 등급 종류 수: 5개 — 각 클래스 **300개씩 균형 분포** (이전 실험에서 불균형 개선 완료).
- 수치형 설명변수 개수 (타겟 제외): 48개 (CSI 24개 + CCI 24개).
- 상수형 변수: 0개 — 제거 불필요.
- 다중공선성 쌍 (|r| ≥ 0.9): 72쌍.
- RPI와 가장 강한 상관: `t1_cci_total` (r = -0.9404) — CSI 높을수록 RPI 낮음(우수 관계) 방향.

## 7. 다음 단계 제안

- 모델링 시 CSI·CCI에 `StandardScaler`(또는 열별 Z-score) 파이프라인을 포함합니다.
- RPI는 낮을수록 우호적 관계이므로, 회귀·랭킹 해석 시 계수 방향을 도메인 척도에 맞게 설명합니다.
- 범주형 변수(area, product, client)는 Label Encoding 또는 One-Hot Encoding 전략을 확정합니다.
- 5-class 균형 데이터이므로 Ordinal 회귀 / 다중 분류 모두 적용 가능합니다.
- 다중공선성이 72쌍으로 발견됨 — PCA 또는 VIF 기반 변수 제거 권장합니다.
