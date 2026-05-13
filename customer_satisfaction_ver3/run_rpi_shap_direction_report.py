"""
LightGBM 다중 클래스(RPI 1~5)에 대해 SHAP(TreeExplainer)로
‘등급 5(최상) 클래스 로짓’ 기준 평균 부호·크기를 보고 방향(긍·부정)을 요약한 HTML 리포트 생성.
total 12개 제외·전처리는 run_feature_importance_eda_report.py 와 동일 전제.
"""
from __future__ import annotations

import base64
import html
import io
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

plt.rcParams["axes.unicode_minus"] = False
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "customer_satisfaction_ver3"
OUT_DIR = Path(__file__).resolve().parent
OUT_HTML = OUT_DIR / "customer_satisfaction_shap_direction_report.html"

TARGET = "rpi"
CAT_COLS = ["year", "area", "product", "client"]
RANDOM_STATE = 42
BG_SAMPLES = 400
EXPLAIN_MAX = 450

AREA_PREFIXES = ("t1", "t2", "c", "d", "q1", "q2")
EXCLUDE_COLS: tuple[str, ...] = tuple(
    f"{p}_{m}_total" for p in AREA_PREFIXES for m in ("csi", "cci")
)

LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=5,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=RANDOM_STATE,
    verbosity=-1,
    force_col_wise=True,
)


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
        ],
        remainder="drop",
    )


def as_lgbm_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(X, dtype=np.float32),
        columns=[f"f{i}" for i in range(X.shape[1])],
    )


def fig_to_base64(fig: Figure, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def main() -> None:
    train_path = DATA_DIR / "customer_satisfaction_train.csv"
    test_path = DATA_DIR / "customer_satisfaction_test.csv"
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    feature_cols = [
        c for c in df_tr.columns if c != TARGET and c not in EXCLUDE_COLS
    ]
    numeric_cols = [c for c in feature_cols if c not in CAT_COLS]

    X_train = df_tr[feature_cols]
    X_test = df_te[feature_cols]
    le = LabelEncoder()
    y_train = le.fit_transform(df_tr[TARGET].astype(int).values)
    y_test = le.transform(df_te[TARGET].astype(int).values)
    class_rpi = [int(x) for x in le.classes_]

    pre = build_preprocessor(numeric_cols)
    Xtr = pre.fit_transform(X_train, y_train)
    Xte = pre.transform(X_test)
    feat_names = list(pre.get_feature_names_out())
    n_feat = len(feat_names)

    clf = LGBMClassifier(**LGBM_PARAMS)
    rng = np.random.default_rng(RANDOM_STATE)
    n_bg = min(BG_SAMPLES, Xtr.shape[0])
    bg_idx = rng.choice(Xtr.shape[0], size=n_bg, replace=False)
    X_bg = np.asarray(Xtr[bg_idx], dtype=np.float32)
    X_bg_df = as_lgbm_frame(X_bg)

    n_ex = min(EXPLAIN_MAX, Xte.shape[0])
    X_ex = np.asarray(Xte[:n_ex], dtype=np.float32)
    X_ex_df = as_lgbm_frame(X_ex)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(as_lgbm_frame(np.asarray(Xtr, dtype=np.float32)), y_train)

    explainer = shap.TreeExplainer(clf, data=X_bg_df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(X_ex_df, check_additivity=False)

    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)

    if sv.ndim != 3:
        raise RuntimeError(f"Unexpected shap shape {sv.shape}")

    idx_high = int(np.argmax(class_rpi))
    idx_low = int(np.argmin(class_rpi))

    shap_high = sv[:, :, idx_high]
    mean_abs_high = np.mean(np.abs(shap_high), axis=0)
    mean_signed_high = np.mean(shap_high, axis=0)

    order = np.argsort(-mean_abs_high)[:35]
    rows_html = []
    pos = mean_abs_high[mean_abs_high > 0]
    thr = (
        float(np.percentile(pos, 75) * 0.05 + 1e-12) if pos.size else 1e-12
    )
    for j in order:
        name = feat_names[j]
        ma = float(mean_abs_high[j])
        sg = float(mean_signed_high[j])
        if ma < thr:
            direction = "효과 미약"
        elif sg > 0:
            direction = f"등급 {class_rpi[idx_high]} 로짓↑(평균)"
        else:
            direction = f"등급 {class_rpi[idx_high]} 로짓↓(평균)"

        rows_html.append(
            "<tr>"
            f"<td>{html.escape(name[:56])}</td>"
            f"<td class='num'>{ma:.5f}</td>"
            f"<td class='num'>{sg:+.5f}</td>"
            f"<td>{html.escape(direction)}</td>"
            "</tr>"
        )

    top20 = order[:20]
    labels = [feat_names[j][:34] for j in top20]
    colors = ["#15803d" if mean_signed_high[j] >= 0 else "#b91c1c" for j in top20]
    fig, ax = plt.subplots(figsize=(9, 7))
    ypos = np.arange(len(top20))
    ax.barh(ypos, mean_signed_high[top20], color=colors, alpha=0.85)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#334155", linewidth=0.8)
    ax.set_xlabel(
        f"평균 SHAP (클래스 RPI={class_rpi[idx_high]} raw margin 기여)"
    )
    ax.set_title(
        f"상위 20 특성 — RPI {class_rpi[idx_high]} 등급 로짓에 대한 평균 SHAP 부호"
    )
    ax.invert_yaxis()
    plt.tight_layout()
    img1 = fig_to_base64(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    ax2.barh(
        ypos,
        mean_abs_high[top20],
        color="steelblue",
        alpha=0.85,
    )
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("mean |SHAP|")
    ax2.set_title(
        f"상위 20 특성 — RPI {class_rpi[idx_high]} 로짓에 대한 평균 |SHAP| (중요도)"
    )
    ax2.invert_yaxis()
    plt.tight_layout()
    img2 = fig_to_base64(fig2)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "class_index_high_rpi": idx_high,
        "class_index_low_rpi": idx_low,
        "rpi_label_high": class_rpi[idx_high],
        "rpi_label_low": class_rpi[idx_low],
        "bg_samples": n_bg,
        "explain_rows": n_ex,
        "n_features": n_feat,
    }

    doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RPI SHAP 방향 요약 (LightGBM)</title>
  <style>
    body {{ font-family: 'Malgun Gothic', 'Segoe UI', system-ui, sans-serif; margin: 24px; max-width: 960px; color: #111; line-height: 1.55; }}
    h1 {{ font-size: 1.35rem; border-bottom: 2px solid #0f766e; padding-bottom: 8px; }}
    h2 {{ font-size: 1.05rem; color: #134e4a; margin-top: 1.5rem; }}
    p.meta {{ color: #555; font-size: 0.9rem; }}
    .note {{ background: #f8fafc; border-left: 4px solid #64748b; padding: 12px 16px; margin: 14px 0; font-size: 0.9rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; margin: 12px 0; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; }}
    th {{ background: #ecfdf5; text-align: left; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    pre {{ background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; font-size: 0.78rem; overflow: auto; }}
  </style>
</head>
<body>
  <h1>RPI 영향 요인의 방향 (SHAP · LightGBM)</h1>
  <p class="meta">
    학습: <code>{html.escape(str(train_path.relative_to(ROOT)))}</code><br/>
    설명 대상: 테스트 앞쪽 <strong>{n_ex}</strong>행 · 배경 데이터: 학습에서 <strong>{n_bg}</strong>행 샘플<br/>
    모델: LightGBM multiclass (동일 하이퍼파라미터 as 특성중요도 리포트)<br/>
    생성(UTC): {html.escape(meta["generated_utc"])}
  </p>

  <div class="note">
    <strong>해석 범위</strong><br/>
    SHAP은 <strong>이 모델·이 전처리·이 데이터 분포</strong> 안에서의 국소 기여입니다. 인과나 정책 효과가 아닙니다.<br/>
    다중 클래스에서 TreeExplainer는 클래스별 raw margin(로짓에 해당)에 대한 SHAP을 줍니다.
    본 리포트는 <strong>RPI 등급 {class_rpi[idx_high]} (가장 높은 등급 클래스)</strong> 출력에 대한
    평균 SHAP 부호로 “그 로짓을 평균적으로 올리는지/내리는지”를 봅니다.
    <strong>양수(녹색 막대)</strong>: 평균적으로 해당 등급의 로짓을 올리는 기여,
    <strong>음수(붉은 막대)</strong>: 내리는 기여입니다. (표본 평균이라 개별 행에서는 반대일 수 있음)<br/>
    <code>cat__</code> 원-핫 특성은 기준 범주 대비 상대 효과이므로, 원시 범주의 “높다/낮다”와 직접 대응하지 않습니다.<br/>
    LightGBM 다중 클래스 조합에서 SHAP 라이브러리의 엄밀 가법성 검사가 실패할 수 있어
    <code>check_additivity=False</code>로 계산했습니다. 수치는 <strong>방향·상대 크기</strong> 참고용으로 보시고,
    인과·목표 설정은 도메인 지식과 함께 검증하세요.
  </div>

  <h2>1. 평균 SHAP 부호 (상위 20)</h2>
  <p><img src="data:image/png;base64,{img1}" alt="mean SHAP class high"/></p>

  <h2>2. 평균 |SHAP| (상위 20, 중요도)</h2>
  <p><img src="data:image/png;base64,{img2}" alt="mean abs SHAP"/></p>

  <h2>3. 상위 35 특성 표 (등급 {class_rpi[idx_high]} 기준)</h2>
  <table>
    <thead><tr><th>특징</th><th class="num">mean |SHAP|</th><th class="num">mean SHAP</th><th>방향 요약</th></tr></thead>
    <tbody>{chr(10).join(rows_html)}</tbody>
  </table>

  <h2>4. 메타 JSON</h2>
  <pre>{html.escape(json.dumps(meta, indent=2, ensure_ascii=False))}</pre>
</body>
</html>
"""

    OUT_HTML.write_text(doc, encoding="utf-8")
    print("Wrote", OUT_HTML)


if __name__ == "__main__":
    main()
