# 고객 만족도 EDA 종합 보고서

- **데이터 파일**: `C:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver2\customer_satisfaction_full_columns.xlsx`
- **시트**: `synthetic_data_full`
- **타겟 컬럼 (Y_Class = RPI)**: `rpi`
- **폰트 설정**: `Malgun Gothic`

> 문서에서 **Y_Class**라고 하면 본 데이터에서는 **`rpi`** 컬럼(고객 만족도 RPI 등급)을 가리킵니다.

## 데이터 구조 (도메인 정의)

- **조사 년도**: 수치형
- **평가영역**(예: `area`): 범주형 — t1, t2, d, c, q1, q2
- **제품군**(예: `product`): 범주형 — 모니터, 노트북, TV, 스마트폰, 자동차 등
- **고객사명**(예: `client`): 범주형 — a ~ e
- **RPI / Y_Class**(열 이름 `rpi`): 범주형 — **1(관계 매우 우수) ~ 5(관계 매우 불량)**
- **CSI**: 수치형 — 0(매우 낮음) ~ 10(매우 높음); 상관분석 전 **표준화 권장**
- **CCI**: 수치형 — -5(매우 낮음) ~ +5(매우 높음); 상관분석 전 **표준화 권장**

## 1. 데이터 요약

```
1500 × 53
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

### 기본 통계 (수치형 일부)

```
            year  t1_csi_res  t1_csi_core  t1_csi_comm  t1_csi_total  t1_cci_res  t1_cci_core  t1_cci_comm  t1_cci_total  t2_csi_res  t2_csi_core  t2_csi_comm  t2_csi_total  t2_cci_res  t2_cci_core  t2_cci_comm  t2_cci_total  d_csi_res  d_csi_core  d_csi_comm  d_csi_total  d_cci_res  d_cci_core  d_cci_comm  d_cci_total  c_csi_res  c_csi_core  c_csi_comm  c_csi_total  c_cci_res
count  1500.0000   1500.0000    1500.0000    1500.0000     1500.0000   1500.0000    1500.0000    1500.0000     1500.0000   1500.0000    1500.0000    1500.0000     1500.0000   1500.0000    1500.0000    1500.0000     1500.0000  1500.0000   1500.0000   1500.0000    1500.0000  1500.0000   1500.0000   1500.0000    1500.0000  1500.0000   1500.0000   1500.0000    1500.0000  1500.0000
mean   2023.9747      7.4381       7.5342       7.4979        7.4900      0.0225      -0.0277      -0.0008       -0.0019      7.4441       7.4778       7.5371        7.4864      0.0666       0.1060       0.0307        0.0679     7.4503      7.5239      7.4502       7.4747    -0.0318     -0.0205      0.0388      -0.0047     7.4973      7.4528      7.4324       7.4608    -0.1004
std       0.8090      1.4481       1.4127       1.3913        0.8241      1.9653       1.9042       1.8819        1.0954      1.4310       1.4187       1.4012        0.8006      2.0045       1.9518       2.0249        1.1208     1.4400      1.4243      1.4201       0.8258     2.0170      1.9283      1.9379       1.1332     1.4430      1.4437      1.5235       0.8636     2.0655
min    2023.0000      2.4800       3.0200       2.7000        4.2200     -5.0000      -5.0000      -5.0000       -3.5300      2.5100       3.2900       3.1900        4.8000     -5.0000      -5.0000      -5.0000       -3.9400     2.3300      2.8000      3.0400       4.8600    -5.0000     -5.0000     -5.0000      -3.5500     2.5600      2.7000      1.6600       4.6200    -5.0000
25%    2023.0000      6.4200       6.5500       6.5700        6.9400     -1.3225      -1.3925      -1.3200       -0.7425      6.4700       6.5100       6.5600        6.9600     -1.2700      -1.1625      -1.2800       -0.6800     6.4575      6.5200      6.4800       6.8800    -1.4525     -1.3700     -1.2400      -0.8100     6.5475      6.4475      6.4000       6.9275    -1.5400
50%    2024.0000      7.5300       7.5500       7.5150        7.5100      0.0450       0.0100       0.0000       -0.0050      7.4600       7.5050       7.5750        7.5100      0.0800       0.1200       0.0150        0.0700     7.4700      7.5500      7.4800       7.4850     0.0100      0.0450     -0.0350      -0.0200     7.5700      7.4600      7.4450       7.4800    -0.1200
75%    2025.0000      8.5100       8.5225       8.4925        8.0700      1.3100       1.2725       1.3500        0.7000      8.4800       8.5425       8.5700        8.0100      1.4300       1.4300       1.4200        0.8300     8.5000      8.5700      8.4825       8.0700     1.4125      1.2500      1.3500       0.8200     8.5200      8.5100      8.5125       8.0600     1.3300
max    2025.0000     10.0000      10.0000      10.0000        9.9600      5.0000       5.0000       5.0000        3.4200     10.0000      10.0000      10.0000        9.9100      5.0000       5.0000       5.0000        3.3900    10.0000     10.0000     10.0000       9.6700     5.0000      5.0000      5.0000       3.7600    10.0000     10.0000     10.0000       9.7200     5.0000
```

## 2. 타겟(Y_Class = RPI / `rpi` 컬럼) 분석

```
     count  percent
rpi                
3       10     0.67
4      723    48.20
5      767    51.13
```

### 시각화

- **RPI 분포(1=우수~5=불량)**: [../outputs/plots/01_rpi_class_bar_pie.png](../outputs/plots/01_rpi_class_bar_pie.png)
- **고객사별 RPI (낮을수록 우수)**: [../outputs/plots/02_rpi_by_client_boxplot.png](../outputs/plots/02_rpi_by_client_boxplot.png)
- **수치형 범위 상위 20 (원 척도)**: [../outputs/plots/03_numeric_range_top20.png](../outputs/plots/03_numeric_range_top20.png)
- **RPI 상관 상위 20(CSI·CCI Z)**: [../outputs/plots/04_corr_with_target_top20.png](../outputs/plots/04_corr_with_target_top20.png)
- **상위 30 히트맵(CSI·CCI Z)**: [../outputs/plots/05_heatmap_top30.png](../outputs/plots/05_heatmap_top30.png)
- **박스플롯 상위 6 (CSI·CCI 원 척도)**: [../outputs/plots/06_boxplot_top6_by_class.png](../outputs/plots/06_boxplot_top6_by_class.png)

## 3. 수치형 변수

- 상수(또는 유일값)에 가까운 변수는 `outputs/tables/constant_like_columns.json` 참고.
- CSI·CCI 열 표준화 적용 내역·적용 열 목록: `outputs/tables/csi_cci_zscore_summary.json`

## 4. 상관관계 및 다중공선성

> **RPI 해석**: 1은 관계가 매우 우수, 5는 매우 불량입니다. **상관·히트맵**은 CSI·CCI에 **열별 Z-score**를 적용한 뒤 산출했으며, 조사 년도 등 그 외 수치형은 **원 스케일**을 유지합니다. 박스플롯은 해석을 위해 **CSI·CCI 원 척도**(CSI 0~10, CCI -5~5)를 사용합니다.

- RPI와의 상관 상위 20개: `outputs/plots/04_corr_with_target_top20.png`
- 히트맵: `outputs/plots/05_heatmap_top30.png`
- |r|≥0.9 쌍: `outputs/tables/multicollinearity_pairs.csv`

## 5. 클래스별 분포 비교

- 박스플롯: `outputs/plots/06_boxplot_top6_by_class.png`

## 6. 인사이트

- **Y_Class**(타겟)는 엑셀 **`rpi`** 열이며, **1=관계 우수 · 5=관계 불량**으로 해석합니다.
- RPI 등급 종류 수: 3 (분포는 plots·표 참고).
- 수치형 설명변수 개수(타겟 제외): 49; 그중 상관분석 시 Z-score 적용 CSI·CCI 열: **48**개.
- 명백한 상수형 변수는 없습니다.
- 선택한 수치형 변수들 사이에 |r|≥0.9인 강한 선형 공선은 드뭅니다.

## 7. 다음 단계 제안

- 모델링 시 CSI·CCI에 `StandardScaler`(또는 본 EDA와 동일한 열별 Z-score) 파이프라인을 포함합니다.
- RPI는 낮을수록 우호적 관계이므로, 회귀·랭킹 해석 시 계수 방향을 도메인 척도에 맞게 설명합니다.
- 결측·이상치 처리(연도·고객사·평가영역별)와 범주형 인코딩 전략을 확정합니다.
- 서수형 RPI에 맞는 Ordinal 회귀 등과 같은 모델 후보를 검토합니다.
