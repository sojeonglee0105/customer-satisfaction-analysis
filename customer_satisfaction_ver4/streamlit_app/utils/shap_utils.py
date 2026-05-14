"""SHAP 분석 래퍼 — explainer, beeswarm, waterfall, dependence Plotly 차트."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def get_explainer(model: Any, X_background: np.ndarray, model_name: str):
    """모델 종류에 맞춰 적절한 explainer 반환."""
    if not SHAP_AVAILABLE:
        raise RuntimeError("shap 라이브러리가 설치되지 않았습니다. requirements.txt 참고")
    if model_name in {"LightGBM", "Random Forest", "Decision Tree", "XGBoost"}:
        return shap.TreeExplainer(model)
    if model_name == "Logistic Regression":
        return shap.LinearExplainer(model, X_background)
    return shap.Explainer(model, X_background)


def compute_shap(model: Any, X: np.ndarray, X_background: np.ndarray,
                  model_name: str) -> np.ndarray:
    """SHAP 값을 (n_classes, n_samples, n_features) 형태로 정규화하여 반환."""
    explainer = get_explainer(model, X_background, model_name)
    raw = explainer.shap_values(X)
    arr = np.array(raw)
    # 새 SHAP API: (n_samples, n_features, n_classes) → 변환
    if arr.ndim == 3:
        if arr.shape[0] == X.shape[0] and arr.shape[1] == X.shape[1]:
            arr = arr.transpose(2, 0, 1)
    elif arr.ndim == 2:
        # 이진 또는 회귀 — (n_samples, n_features)
        arr = arr[np.newaxis, ...]
    return arr


def shap_summary_df(shap_3d: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| + Mean SHAP(부호 보존) 요약 DataFrame."""
    mean_abs = np.abs(shap_3d).mean(axis=(0, 1))
    mean_sgn = shap_3d.mean(axis=0).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_sgn,
    })
    df["direction"] = np.where(df["mean_signed_shap"] >= 0, "▲ 양수 (RPI 상승)", "▼ 음수 (RPI 하강)")
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def shap_bar_chart(summary: pd.DataFrame, top_n: int = 20, signed: bool = False):
    """SHAP 바차트. signed=True면 부호 보존 (🔵 양수=파랑, 🔴 음수=빨강)."""
    sub = summary.head(top_n).iloc[::-1]
    if signed:
        # 양성(positive) = 파랑, 음성(negative) = 빨강
        colors = ["#2563eb" if v >= 0 else "#dc2626" for v in sub["mean_signed_shap"]]
        x_vals = sub["mean_signed_shap"]
        title = "Mean SHAP Value (부호 보존 — 🔵 양수 / 🔴 음수)"
    else:
        colors = "#2563eb"
        x_vals = sub["mean_abs_shap"]
        title = "Mean |SHAP Value|"
    fig = go.Figure(go.Bar(x=x_vals, y=sub["feature"], orientation="h",
                             marker_color=colors,
                             text=[f"{v:+.4f}" for v in sub["mean_signed_shap"]],
                             textposition="outside"))
    fig.update_layout(title=f"SHAP Top-{len(sub)} — {title}",
                       template="plotly_white",
                       height=max(380, 25 * len(sub)),
                       margin=dict(l=170, r=80, t=60, b=40))
    if signed:
        fig.add_vline(x=0, line=dict(color="black", width=1))
    return fig


def shap_beeswarm(shap_3d: np.ndarray, X: np.ndarray, feature_names: list[str],
                   top_n: int = 15):
    """SHAP Beeswarm 유사 dot plot (Plotly)."""
    summed = shap_3d.sum(axis=0)  # (n_samples, n_features)
    mean_abs = np.abs(summed).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:]
    fig = go.Figure()
    rng = np.random.default_rng(42)
    for yi, fi in enumerate(top_idx):
        sv = summed[:, fi]
        fv = X[:, fi]
        fv_n = (fv - np.min(fv)) / (np.max(fv) - np.min(fv) + 1e-12)
        jitter = rng.uniform(-0.32, 0.32, size=len(sv))
        fig.add_trace(go.Scatter(
            x=sv, y=np.full_like(sv, yi, dtype=float) + jitter, mode="markers",
            marker=dict(size=6, color=fv_n, colorscale="RdBu_r", showscale=(yi == 0),
                        colorbar=dict(title="피처 값", tickvals=[0, 0.5, 1],
                                      ticktext=["낮음", "중간", "높음"])
                        if yi == 0 else None,
                        opacity=0.65, line=dict(width=0)),
            hovertemplate=f"{feature_names[fi]}<br>SHAP=%{{x:.4f}}<br>값=%{{customdata:.2f}}<extra></extra>",
            customdata=fv,
            showlegend=False,
        ))
    fig.add_vline(x=0, line=dict(color="black", dash="dash", width=1))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(top_idx))),
                      ticktext=[feature_names[i] for i in top_idx])
    fig.update_layout(title=f"SHAP Beeswarm — Top {len(top_idx)} (양수=RPI↑, 음수=RPI↓)",
                       template="plotly_white",
                       height=max(420, 30 * len(top_idx)),
                       xaxis_title="SHAP Value",
                       margin=dict(l=170, r=60, t=60, b=40))
    return fig


def shap_class_heatmap(shap_3d: np.ndarray, feature_names: list[str],
                        class_labels: list, top_n: int = 12):
    """클래스 × Top-N 피처 SHAP 부호 히트맵 (🔵 양수=파랑 / 🔴 음수=빨강)."""
    mean_abs = np.abs(shap_3d).mean(axis=(0, 1))
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    mat = np.zeros((shap_3d.shape[0], len(top_idx)))
    for ci in range(shap_3d.shape[0]):
        mat[ci, :] = shap_3d[ci, :, :][:, top_idx].mean(axis=0)
    feats = [feature_names[i] for i in top_idx]
    cls_names = [str(c) for c in class_labels[:shap_3d.shape[0]]]
    bound = max(0.001, float(np.abs(mat).max()))
    import plotly.express as px
    # 양성(positive) = 파랑, 음성(negative) = 빨강 → "RdBu" (역전 안 함)
    fig = px.imshow(mat, x=feats, y=cls_names, color_continuous_scale="RdBu",
                     zmin=-bound, zmax=bound, aspect="auto", text_auto=".3f",
                     labels=dict(x="Feature", y="Class", color="Mean SHAP"))
    fig.update_layout(title="클래스 × Top Feature SHAP 방향 히트맵 (🔵 양수 / 🔴 음수)",
                       template="plotly_white", height=420)
    return fig


def shap_waterfall(shap_3d: np.ndarray, sample_idx: int, class_idx: int,
                    feature_names: list[str], feature_values: np.ndarray,
                    top_n: int = 10):
    """단일 샘플·단일 클래스 Waterfall (🔵 양수=파랑 / 🔴 음수=빨강)."""
    sv = shap_3d[class_idx, sample_idx, :]
    order = np.argsort(np.abs(sv))[-top_n:][::-1]
    feats = [feature_names[i] for i in order]
    vals = sv[order]
    fvals = feature_values[order]
    # 양성(positive) = 파랑, 음성(negative) = 빨강
    colors = ["#2563eb" if v >= 0 else "#dc2626" for v in vals]
    labels = [f"{f}<br>(val={v:.2f})" for f, v in zip(feats, fvals)]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                             marker_color=colors,
                             text=[f"{v:+.4f}" for v in vals],
                             textposition="outside"))
    fig.add_vline(x=0, line=dict(color="black", width=1))
    fig.update_layout(title=f"SHAP Waterfall — sample #{sample_idx}, class #{class_idx} "
                              "(🔵 양수=클래스↑ / 🔴 음수=클래스↓)",
                       template="plotly_white",
                       height=max(380, 30 * len(feats)),
                       xaxis_title="SHAP Value (🔵 양수 → 해당 클래스 예측↑)",
                       margin=dict(l=200, r=60, t=60, b=40))
    fig.update_yaxes(autorange="reversed")
    return fig


def shap_dependence(shap_3d: np.ndarray, X: np.ndarray, feature_names: list[str],
                     feature: str, class_labels: list):
    """선택 피처의 dependence plot (모든 클래스 색상 구분)."""
    fi = feature_names.index(feature)
    fig = go.Figure()
    palette = ["#3b82f6", "#22c55e", "#eab308", "#f97316", "#ef4444", "#a855f7"]
    for ci, cls in enumerate(class_labels[:shap_3d.shape[0]]):
        fig.add_trace(go.Scatter(
            x=X[:, fi], y=shap_3d[ci, :, fi],
            mode="markers",
            marker=dict(size=7, color=palette[ci % len(palette)], opacity=0.55),
            name=f"Class {cls}",
        ))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(title=f"Dependence Plot — {feature}",
                       xaxis_title=feature, yaxis_title="SHAP Value",
                       template="plotly_white", height=420)
    return fig
