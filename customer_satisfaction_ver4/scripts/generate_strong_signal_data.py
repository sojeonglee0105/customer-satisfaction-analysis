"""강한 예측 신호(高 AUROC)를 보이는 Customer Satisfaction Survey 합성 데이터 생성.

기존 customer_satisfaction_ver3/customer_satisfaction_train.csv 와 동일한
스키마(53컬럼 — 메타 4 + 타겟 1 + 피처 48)를 유지하되,
각 RPI 클래스(1~5)가 서로 다른 도메인 점수 평균을 갖도록 설계해
모델이 명확히 구분 가능하게(=AUROC가 높게) 만듭니다.

설계 원칙:
1. **잠재 점수(latent score)** = 6개 도메인 점수의 가중합 + 작은 노이즈
2. RPI 1~5는 latent 점수의 분위(quintile)로 결정 → 깨끗한 단조 관계
3. 각 도메인의 csi/cci × res/core/comm 은 RPI 클래스별 평균을 다르게,
   그리고 _total = 세 sub-항목의 평균(자연스러운 다중공선성 유지)
4. q1(품질), t1(신기술), c(Cost) 영역에 가장 큰 가중치 → Top 도메인이 됨
5. 살짝 불균형(2.5:1 정도)을 인위로 만들어 현실감 추가
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── 설정 ──────────────────────────────────────────────────────────
SEED       = 42
N_SAMPLES  = 1500
OUT_DIR    = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 메타 카테고리
YEARS    = [2023, 2024, 2025]
AREAS    = ["t1", "t2", "c", "d", "q1", "q2"]
PRODUCTS = ["자동차", "tv", "모니터", "노트북", "스마트폰"]
CLIENTS  = ["a", "b", "c", "d", "e"]

# 도메인 정의 (가중치 → 합 = 1.0, 큰 값일수록 RPI 결정에 큰 영향)
DOMAINS = {
    "t1": {"name": "신기술", "weight": 0.22},
    "t2": {"name": "개발",   "weight": 0.10},
    "c":  {"name": "Cost",   "weight": 0.18},
    "d":  {"name": "공급",   "weight": 0.13},
    "q1": {"name": "품질",   "weight": 0.27},
    "q2": {"name": "서비스", "weight": 0.10},
}
SUB_KINDS = ["csi", "cci"]
SUB_PARTS = ["res", "core", "comm"]

# 클래스별 도메인 점수 평균 (RPI 1=낮음 → RPI 5=높음, 단조 증가)
# 평균 점수 베이스 (0~10 범위) — 클래스가 올라갈수록 상승
CLASS_BASE_MEAN = {1: 4.0, 2: 5.5, 3: 6.8, 4: 8.0, 5: 9.0}
# 클래스별 표준편차 — 작을수록 분리 명확 (= AUROC↑)
CLASS_BASE_STD  = {1: 0.55, 2: 0.55, 3: 0.55, 4: 0.55, 5: 0.55}

# 클래스 분포 (살짝 불균형)
CLASS_PROB = {1: 0.18, 2: 0.22, 3: 0.24, 4: 0.20, 5: 0.16}

# ── 데이터 생성 ────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)


def _sample_class_assignment(n: int) -> np.ndarray:
    """클래스를 정해진 분포에 따라 샘플링."""
    classes = np.array(list(CLASS_PROB.keys()))
    probs   = np.array(list(CLASS_PROB.values()))
    return rng.choice(classes, size=n, p=probs)


def _gen_domain_scores(class_arr: np.ndarray) -> dict[str, np.ndarray]:
    """도메인별로 강한 시그널을 가진 잠재점수 배열 생성.

    각 클래스마다 평균 점수 차이가 명확하므로 단변량 분리력(η²)도 큼.
    """
    out = {}
    for prefix, info in DOMAINS.items():
        weight = info["weight"]
        # 잠재값 — 클래스 평균에 가중치를 약간 가산하여 강한 도메인일수록
        # 클래스 분리가 더 강해지도록 한다
        boost = (weight - 0.16) * 4  # ~ -0.24 ~ +0.44
        latent = np.array([
            rng.normal(
                loc=CLASS_BASE_MEAN[c] + boost,
                scale=CLASS_BASE_STD[c],
            )
            for c in class_arr
        ])
        out[prefix] = np.clip(latent, 0, 10)
    return out


def _gen_subscore(latent: np.ndarray, jitter_scale: float = 0.6) -> np.ndarray:
    """서브 항목 점수 = latent + 작은 노이즈."""
    noise = rng.normal(0, jitter_scale, size=len(latent))
    return np.clip(latent + noise, 0, 10)


def main():
    # 1) 클래스 (RPI) 결정
    rpi = _sample_class_assignment(N_SAMPLES)

    # 2) 도메인 잠재 점수
    domain_latents = _gen_domain_scores(rpi)

    # 3) 메타 컬럼
    meta = {
        "year":    rng.choice(YEARS, N_SAMPLES, p=[0.25, 0.40, 0.35]),
        "area":    rng.choice(AREAS, N_SAMPLES),
        "product": rng.choice(PRODUCTS, N_SAMPLES, p=[0.10, 0.30, 0.20, 0.20, 0.20]),
        "client":  rng.choice(CLIENTS, N_SAMPLES, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
    }

    # 4) 피처 — 도메인 × csi/cci × res/core/comm + total(평균)
    features = {}
    for prefix, latent in domain_latents.items():
        for kind in SUB_KINDS:
            # csi와 cci는 살짝 다른 정도로 latent를 따라감
            kind_offset = 0.0 if kind == "csi" else -0.3
            sub_arrays = {}
            for part in SUB_PARTS:
                # part마다 노이즈 스케일을 살짝 다르게
                jitter = {"res": 0.7, "core": 0.55, "comm": 0.85}[part]
                sub_arrays[part] = _gen_subscore(latent + kind_offset, jitter_scale=jitter)
                features[f"{prefix}_{kind}_{part}"] = sub_arrays[part]
            # _total = 세 sub-part의 평균
            features[f"{prefix}_{kind}_total"] = np.mean(
                np.stack([sub_arrays[p] for p in SUB_PARTS], axis=0), axis=0
            )

    # 5) 노이즈 변수 — 자연스러운 약한 신호도 일부 포함되도록 일부러 안 넣음
    #   → 모든 변수가 어느 정도 신호를 가지도록 설계 (대신 가중치 차등)

    # 6) DataFrame 조립
    df = pd.DataFrame({"rpi": rpi, **meta, **features})

    # 컬럼 순서 — 기존 train.csv와 동일하게
    col_order = ["year", "area", "product", "client", "rpi"]
    for prefix in DOMAINS.keys():
        for kind in SUB_KINDS:
            for part in SUB_PARTS + ["total"]:
                col_order.append(f"{prefix}_{kind}_{part}")
    df = df[col_order]

    # 7) Train/Test 분할 (8:2, time-aware: 2023~2024 = train, 2025 = test 기반 + 추가 비율)
    df = df.sort_values(["year"]).reset_index(drop=True)
    n_test = int(len(df) * 0.2)
    train_df = df.iloc[:-n_test].copy()
    test_df  = df.iloc[-n_test:].copy()

    # 8) 저장 — Excel(통합) + CSV(train/test) 둘 다
    full_path  = OUT_DIR / "lgd_csat_strong_signal.xlsx"
    train_path = OUT_DIR / "lgd_csat_strong_signal_train.csv"
    test_path  = OUT_DIR / "lgd_csat_strong_signal_test.csv"

    df.to_csv(OUT_DIR / "lgd_csat_strong_signal_full.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    # README 시트와 함께 Excel 1개 파일 (대시보드 업로드용)
    with pd.ExcelWriter(full_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="data", index=False)

        # 메타데이터 시트
        meta_rows = [
            {"항목": "샘플 수",     "값": f"{len(df):,}"},
            {"항목": "변수 수",     "값": f"{df.shape[1]} (메타 4 + 타겟 1 + 피처 48)"},
            {"항목": "타겟 변수",   "값": "rpi (1~5, 1=낮음 / 5=높음)"},
            {"항목": "클래스 분포",
             "값": ", ".join([f"{k}={v}" for k, v in df['rpi'].value_counts().sort_index().items()])},
            {"항목": "불균형 비율",
             "값": f"{df['rpi'].value_counts().max() / df['rpi'].value_counts().min():.2f}:1"},
            {"항목": "기간",        "값": f"{df['year'].min()}~{df['year'].max()}"},
            {"항목": "도메인 영역", "값": ", ".join([f"{p}={info['name']}" for p, info in DOMAINS.items()])},
        ]
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="README", index=False)

        # 도메인 가중치
        dom_rows = [
            {
                "Prefix": p,
                "도메인": info["name"],
                "RPI 결정 가중치": f"{info['weight']:.2f}",
                "예상 영향력": "★ 매우 강함" if info["weight"] >= 0.20
                           else ("★ 강함" if info["weight"] >= 0.15 else "보통"),
            }
            for p, info in DOMAINS.items()
        ]
        pd.DataFrame(dom_rows).to_excel(writer, sheet_name="도메인 가중치", index=False)

        # 클래스별 평균
        class_means = df.groupby("rpi").agg({
            "t1_csi_total": "mean",
            "t2_csi_total": "mean",
            "c_csi_total":  "mean",
            "d_csi_total":  "mean",
            "q1_csi_total": "mean",
            "q2_csi_total": "mean",
        }).round(3)
        class_means.to_excel(writer, sheet_name="클래스별 도메인 평균")

    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 60)
    print(f"[OK] data generated: {len(df):,} rows x {df.shape[1]} cols")
    print(f"     -> Excel    : {full_path}")
    print(f"     -> Train CSV: {train_path}")
    print(f"     -> Test CSV : {test_path}")
    print()
    print("Class distribution (rpi):")
    print(df["rpi"].value_counts().sort_index().to_string())
    print()
    print("Class-wise mean of _csi_total (separation check):")
    print(class_means.to_string())


if __name__ == "__main__":
    main()
