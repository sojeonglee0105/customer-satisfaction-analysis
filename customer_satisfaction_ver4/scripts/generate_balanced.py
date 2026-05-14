"""
customer_satisfaction_full_columns.xlsx 데이터를 조건에 맞게 수정하여
customer_satisfaction_balanced.xlsx 로 저장

주요 개선사항:
  1. 5개 RPI 클래스(1~5) 각 300개씩 → 1,500행 (완전 균형)
  2. 클래스별 CSI/CCI 평균을 명확히 분리 → 상관관계 개선
  3. _total = round((res + core + comm) / 3, 2) 구조 유지
  4. CSI [0,10] · CCI [-5,5] 범위 보장
  5. 동일 53개 컬럼 구조 유지
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ── 클래스별 파라미터 ──────────────────────────────────────────────────
# RPI 1 = 관계 매우 우수 → CSI 높음, CCI 높음
# RPI 5 = 관계 매우 불량 → CSI 낮음, CCI 낮음
CLASS_PARAMS = {
    1: {"csi_mu": 9.0, "csi_sig": 0.7, "cci_mu":  2.5, "cci_sig": 1.0},
    2: {"csi_mu": 7.9, "csi_sig": 0.9, "cci_mu":  1.5, "cci_sig": 1.2},
    3: {"csi_mu": 6.8, "csi_sig": 1.0, "cci_mu":  0.3, "cci_sig": 1.4},
    4: {"csi_mu": 5.7, "csi_sig": 1.1, "cci_mu": -0.8, "cci_sig": 1.6},
    5: {"csi_mu": 4.6, "csi_sig": 1.2, "cci_mu": -1.8, "cci_sig": 1.8},
}

AREAS    = ["t1", "t2", "d", "c", "q1", "q2"]
PRODUCTS = ["모니터", "노트북", "TV", "스마트폰", "자동차"]
CLIENTS  = ["a", "b", "c", "d", "e"]
YEARS    = [2023, 2024, 2025]
N_PER_CLASS = 300


def gen_component(mu: float, sigma: float, lo: float, hi: float) -> float:
    return float(np.clip(np.random.normal(mu, sigma), lo, hi))


rows = []
for rpi in range(1, 6):
    p = CLASS_PARAMS[rpi]
    csi_mu, csi_sig = p["csi_mu"], p["csi_sig"]
    cci_mu, cci_sig = p["cci_mu"], p["cci_sig"]

    for _ in range(N_PER_CLASS):
        row: dict = {}
        row["year"]    = int(np.random.choice(YEARS))
        row["area"]    = str(np.random.choice(AREAS))
        row["product"] = str(np.random.choice(PRODUCTS))
        row["client"]  = str(np.random.choice(CLIENTS))
        row["rpi"]     = rpi

        # 각 평가영역별 CSI·CCI 생성
        # 영역마다 ±0.3 랜덤 오프셋으로 현실감 추가
        for area in AREAS:
            area_csi_shift = np.random.uniform(-0.3, 0.3)
            area_cci_shift = np.random.uniform(-0.3, 0.3)

            # res, core, comm, total 모두 독립 변수 → 각각 독립 생성
            for sub in ["res", "core", "comm", "total"]:
                csi_val = gen_component(csi_mu + area_csi_shift, csi_sig, 0, 10)
                row[f"{area}_csi_{sub}"] = round(csi_val, 2)

                cci_val = gen_component(cci_mu + area_cci_shift, cci_sig, -5, 5)
                row[f"{area}_cci_{sub}"] = round(cci_val, 2)

        rows.append(row)

df = pd.DataFrame(rows)

# ── 컬럼 순서를 원본과 동일하게 정렬 ─────────────────────────────────
base_cols = ["year", "area", "product", "client", "rpi"]
area_cols = []
for area in AREAS:
    for metric in ["csi", "cci"]:
        for sub in ["res", "core", "comm", "total"]:
            area_cols.append(f"{area}_{metric}_{sub}")

df = df[base_cols + area_cols]

# ── 검증 ─────────────────────────────────────────────────────────────
print("=" * 55)
print("Shape: {}".format(df.shape))
print("\n[RPI class distribution]")
counts = df["rpi"].value_counts().sort_index()
for rpi_val, cnt in counts.items():
    pct = cnt / len(df) * 100
    print("  RPI {} : {:>4} ({:.1f}%)".format(rpi_val, cnt, pct))

print("\n[Class imbalance ratio]")
min_c, max_c = counts.min(), counts.max()
ratio = max_c / min_c
judgement = "PASS" if ratio <= 100 else "FAIL"
print("  max:min = {}:{} -> {:.1f}:1 ({})".format(max_c, min_c, ratio, judgement))

print("\n[Missing values]")
missing = int(df.isnull().sum().sum())
print("  missing total: {} ({})".format(missing, "OK" if missing == 0 else "WARNING"))

print("\n[Top 5 correlations with target]")
corr_df = df.select_dtypes(include=[float, int]).drop(columns=["year"])
corr_vals = corr_df.corr()["rpi"].drop("rpi").abs().sort_values(ascending=False)
for col, val in corr_vals.head(5).items():
    print("  {:25s}  |r| = {:.4f}".format(col, val))

print("\n[CSI range check]")
csi_cols = [c for c in df.columns if "csi" in c]
print("  min={:.2f}  max={:.2f}  (expected: 0~10)".format(df[csi_cols].min().min(), df[csi_cols].max().max()))

print("\n[CCI range check]")
cci_cols = [c for c in df.columns if "cci" in c]
print("  min={:.2f}  max={:.2f}  (expected: -5~5)".format(df[cci_cols].min().min(), df[cci_cols].max().max()))

# ── 저장 ─────────────────────────────────────────────────────────────
out_path = r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver2\customer_satisfaction_balanced.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="balanced_data", index=False)

print("\n" + "=" * 55)
print("Saved -> {}".format(out_path))
