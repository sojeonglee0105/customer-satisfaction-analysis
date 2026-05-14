"""Plotly 기반 시각화 팩토리."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve

MODEL_COLORS = {
    "Logistic Regression": "#6366f1",
    "LightGBM":            "#10b981",
    "Random Forest":       "#f59e0b",
    "Decision Tree":       "#ef4444",
    "XGBoost":             "#0ea5e9",
}


# ── EDA 차트 ──────────────────────────────────────────────────────
def target_distribution_chart(y: pd.Series, title: str = "타겟 분포"):
    counts = y.value_counts().sort_index()
    fig = make_subplots(rows=1, cols=2,
                         specs=[[{"type": "bar"}, {"type": "domain"}]],
                         subplot_titles=("클래스별 개수", "비율"))
    fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values,
                          marker_color="#6366f1", name="count",
                          text=counts.values, textposition="outside"),
                   row=1, col=1)
    fig.add_trace(go.Pie(labels=counts.index.astype(str), values=counts.values,
                          hole=0.35, textinfo="label+percent"), row=1, col=2)
    fig.update_layout(title=title, height=380, showlegend=False, template="plotly_white")
    return fig


def histogram_grid(df: pd.DataFrame, cols: list[str], target: str | None = None,
                    n_per_row: int = 3, height_per_row: int = 240):
    n = len(cols)
    rows = (n + n_per_row - 1) // n_per_row
    fig = make_subplots(rows=rows, cols=n_per_row, subplot_titles=cols)
    for i, c in enumerate(cols):
        r, k = i // n_per_row + 1, i % n_per_row + 1
        if target and target in df.columns:
            for cls in sorted(df[target].dropna().unique()):
                fig.add_trace(go.Histogram(x=df.loc[df[target] == cls, c],
                                             name=str(cls),
                                             opacity=0.55,
                                             showlegend=(i == 0),
                                             legendgroup=str(cls)),
                               row=r, col=k)
        else:
            fig.add_trace(go.Histogram(x=df[c], marker_color="#6366f1",
                                         showlegend=False), row=r, col=k)
    fig.update_layout(barmode="overlay", height=rows * height_per_row,
                       template="plotly_white", title="수치형 변수 히스토그램")
    return fig


def correlation_heatmap(df: pd.DataFrame, top_n: int = 30, target: str | None = None):
    num_df = df.select_dtypes(include=[np.number])
    if target and target in num_df.columns:
        order = num_df.corr()[target].abs().sort_values(ascending=False).index.tolist()
        cols = order[:top_n]
    else:
        cols = num_df.columns[:top_n].tolist()
    corr = num_df[cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, aspect="auto",
                     title=f"상관관계 히트맵 (Top {len(cols)})")
    fig.update_layout(height=max(450, 18 * len(cols)), template="plotly_white")
    return fig


def multicollinearity_pairs(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr().abs()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if v >= threshold and not np.isnan(v):
                pairs.append({"변수 1": cols[i], "변수 2": cols[j],
                               "|r|": round(float(v), 4)})
    return pd.DataFrame(pairs).sort_values("|r|", ascending=False).reset_index(drop=True)


def violin_by_class(df: pd.DataFrame, feature: str, target: str):
    fig = px.violin(df, x=target, y=feature, color=target, box=True, points="outliers",
                     title=f"{feature} 분포 — {target} 클래스별")
    fig.update_layout(template="plotly_white", height=400, showlegend=False)
    return fig


# ── Feature Engineering 차트 ──────────────────────────────────────
def scree_plot(explained_var_ratio: np.ndarray):
    cum = np.cumsum(explained_var_ratio)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=list(range(1, len(explained_var_ratio) + 1)),
                          y=explained_var_ratio, name="개별 분산", marker_color="#6366f1"),
                   secondary_y=False)
    fig.add_trace(go.Scatter(x=list(range(1, len(explained_var_ratio) + 1)),
                               y=cum, name="누적 분산", line=dict(color="#dc2626", width=2),
                               mode="lines+markers"), secondary_y=True)
    fig.add_hline(y=0.95, line=dict(color="green", dash="dash"),
                   annotation_text="95% 분산", secondary_y=True)
    fig.update_layout(title="PCA Scree Plot", template="plotly_white", height=400)
    fig.update_yaxes(title_text="개별 분산 설명률", secondary_y=False)
    fig.update_yaxes(title_text="누적 분산 설명률", range=[0, 1.05], secondary_y=True)
    fig.update_xaxes(title_text="Principal Component")
    return fig


# ── 모델 비교 차트 ─────────────────────────────────────────────────
def radar_chart(pivot: pd.DataFrame, metrics: list[str], title: str = "모델 성능 Radar"):
    fig = go.Figure()
    for model in pivot.index:
        vals = [pivot.loc[model, m] for m in metrics if m in pivot.columns]
        vals_full = vals + vals[:1]
        labels = [m for m in metrics if m in pivot.columns]
        labels_full = labels + labels[:1]
        fig.add_trace(go.Scatterpolar(
            r=vals_full, theta=labels_full, fill="toself",
            name=model, line=dict(color=MODEL_COLORS.get(model, "#64748b"), width=2),
            opacity=0.45,
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                       title=title, template="plotly_white", height=440)
    return fig


def metric_box_plot(agg_df: pd.DataFrame, metric: str):
    sub = agg_df[agg_df["metric"] == metric]
    rows = []
    for _, r in sub.iterrows():
        for v in r["values"]:
            rows.append({"model": r["model"], "value": v})
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return go.Figure()
    fig = px.box(long_df, x="model", y="value", color="model",
                  color_discrete_map=MODEL_COLORS, points="all",
                  title=f"Seed별 분포 — {metric}")
    fig.update_layout(template="plotly_white", showlegend=False, height=400)
    return fig


def confusion_matrix_chart(cm: np.ndarray, labels: list, title: str = "Confusion Matrix"):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                     aspect="auto",
                     labels=dict(x="Predicted", y="True", color="Count"),
                     x=[str(l) for l in labels], y=[str(l) for l in labels],
                     title=title)
    fig.update_layout(template="plotly_white", height=420)
    return fig


def roc_curves_chart(y_true: np.ndarray, prob_dict: dict[str, np.ndarray],
                      classes: list, title: str = "ROC Curves (Macro Average)"):
    """모델별 macro-average ROC overlay."""
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)
    fig = go.Figure()
    for name, prob in prob_dict.items():
        if prob is None:
            continue
        # 클래스별 ROC를 계산 후 평균
        fprs, tprs = [], []
        common = np.linspace(0, 1, 101)
        for i, _ in enumerate(classes):
            if y_bin.shape[1] <= i:
                continue
            yi = y_bin[:, i]
            if yi.sum() == 0:
                continue
            try:
                fpr, tpr, _ = roc_curve(yi, prob[:, i])
                interp = np.interp(common, fpr, tpr)
                fprs.append(common); tprs.append(interp)
            except Exception:
                pass
        if not tprs:
            continue
        mean_tpr = np.mean(tprs, axis=0)
        fig.add_trace(go.Scatter(x=common, y=mean_tpr, mode="lines",
                                   name=name,
                                   line=dict(color=MODEL_COLORS.get(name, "#64748b"), width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                               line=dict(dash="dash", color="gray"),
                               showlegend=False))
    fig.update_layout(title=title, xaxis_title="FPR", yaxis_title="TPR",
                       template="plotly_white", height=440)
    return fig


def metric_delta_bar(df_uw: pd.DataFrame, df_w: pd.DataFrame, metric: str):
    """class_weight 보정 전·후 변화 바차트."""
    pivot_uw = df_uw[df_uw["metric"] == metric].set_index("model")["mean"]
    pivot_w  = df_w[df_w["metric"] == metric].set_index("model")["mean"]
    common = sorted(set(pivot_uw.index) & set(pivot_w.index))
    delta = (pivot_w.loc[common] - pivot_uw.loc[common])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=common, y=pivot_uw.loc[common], name="미보정",
                          marker_color="#94a3b8"))
    fig.add_trace(go.Bar(x=common, y=pivot_w.loc[common], name="보정",
                          marker_color="#16a34a"))
    fig.update_layout(barmode="group", template="plotly_white",
                       title=f"보정 전·후 비교 — {metric}", height=400)
    return fig


# ── 중요도 차트 ────────────────────────────────────────────────────
def importance_bar(df: pd.DataFrame, label_col: str, val_col: str,
                    title: str, color: str = "#6366f1", err_col: str | None = None):
    df_sorted = df.sort_values(val_col, ascending=True)
    err = df_sorted[err_col] if err_col else None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted[val_col], y=df_sorted[label_col],
        orientation="h",
        marker_color=color,
        error_x=dict(type="data", array=err, visible=err is not None),
    ))
    fig.update_layout(title=title, template="plotly_white",
                       height=max(360, 22 * len(df_sorted)),
                       xaxis_title=val_col, yaxis_title=label_col,
                       margin=dict(l=160, r=40, t=60, b=40))
    return fig
