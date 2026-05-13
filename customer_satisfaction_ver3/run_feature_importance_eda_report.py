"""
total 12개 제외 버전 데이터로 상위 3모델(LGBM, Logistic Regression, Ordinal LogisticAT)
의 특성 중요도·퍼뮤테이션 중요도·EDA·통계 검정 후 HTML 리포트 생성.
"""
from __future__ import annotations

import base64
import html
import io
import json
import platform
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from matplotlib.figure import Figure
from mord import LogisticAT
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

matplotlib.use("Agg")
plt.rcParams["axes.unicode_minus"] = False
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
sns.set_theme(style="whitegrid", font_scale=0.9)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "customer_satisfaction_ver3"
OUT_DIR = Path(__file__).resolve().parent
OUT_HTML = OUT_DIR / "customer_satisfaction_feature_importance_eda_report.html"

TARGET = "rpi"
CAT_COLS = ["year", "area", "product", "client"]
RANDOM_STATE = 42
TOP_K_DISPLAY = 20
TOP_K_EDA = 30
PERM_REPEATS = 10

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
LR_PARAMS = dict(
    max_iter=5000,
    solver="lbfgs",
    C=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
ORDINAL_PARAMS = dict(alpha=1.0)


def build_insights_future_sections(
    *,
    w_val: float,
    chi2_w: float,
    p_w: float,
    spear_rows: list[dict],
    fried_sig_count: int,
    m_tests: int,
    perm_repeats: int,
    top_k_eda: int,
    n_eda_rows: int,
    cv_imp_mean: float | None,
    cv_un_mean: float | None,
    p_cv_g: float | None,
    p_out_g: float | None,
    p_miss_g: float | None,
    n_excluded: int,
) -> str:
    """리포트 하단에 삽입할 인사이트·향후 과제 HTML."""
    if np.isnan(p_w):
        k_line = (
            f"<li><strong>Kendall W</strong>≈{w_val:.3f} (χ²≈{chi2_w:.1f}), "
            "근사 p는 계산 불가/비유의로 해석에 주의하세요.</li>"
        )
    elif p_w < 0.05:
        k_line = (
            f"<li><strong>Kendall W</strong>={w_val:.3f} (χ²≈{chi2_w:.1f}, p={p_w:.3g})로, "
            "세 모델이 퍼뮤테이션 중요도 순위에서 통계적으로 유의한 수준의 조화를 보입니다.</li>"
        )
    else:
        k_line = (
            f"<li><strong>Kendall W</strong>={w_val:.3f} (p={p_w:.3g})는 "
            "순위 일치가 ‘우연 수준’과 구분되기 어렵다는 뜻일 수 있으나, "
            "Spearman·상위 특징 막대와 함께 정성적으로 교차 확인하는 것이 좋습니다.</li>"
        )

    spear_items = "\n    ".join(
        "<li>"
        f"{html.escape(str(r['a']))} vs {html.escape(str(r['b']))}: "
        f"Spearman ρ={r['rho']:.3f} (p={r['p']:.3g})"
        "</li>"
        for r in spear_rows
    )

    fried_line = (
        f"<li>상위 {m_tests}개 특징에 대해 Friedman 검정 후 Bonferroni(×{m_tests}) 보정 시, "
        f"모델 간 퍼뮤테이션 점수 분포 차이가 유의(p_adj&lt;0.05)한 특징은 "
        f"<strong>{fried_sig_count}</strong>개입니다. "
        "다수가 비유의라면 ‘세 모델이 비슷하게 의존’한다고 보는 편이 자연스럽습니다.</li>"
    )

    def _fmt_p(v: float | None) -> str:
        if v is None:
            return "N/A"
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return "N/A"
        if np.isnan(vf) or np.isinf(vf):
            return "N/A"
        return f"{vf:.3g}"

    if (
        cv_imp_mean is not None
        and cv_un_mean is not None
        and p_cv_g is not None
        and p_out_g is not None
    ):
        eda_grp = (
            "<li>합의 상위에 매핑된 <strong>원시 수치열</strong>의 평균 CV는 "
            f"{cv_imp_mean:.3f}, 비교군(샘플링된 기타 수치형)은 {cv_un_mean:.3f}이며, "
            f"Welch t 검정 p={_fmt_p(p_cv_g)}입니다. "
            f"이상치 비율·결측률에 대한 p는 각각 {_fmt_p(p_out_g)}, {_fmt_p(p_miss_g)}입니다. "
            "비교군이 하위 합의에서만 뽑혀 편향될 수 있으니, 해석은 보조적 근거로 두는 것이 좋습니다.</li>"
        )
    else:
        eda_grp = (
            "<li>중요·비중요 원시열 그룹 Welch 비교는 데이터 부족으로 생략되었습니다.</li>"
        )

    return f"""  <h2>6. 핵심 인사이트</h2>
  <ul class="insights-list">
    {k_line}
    {spear_items}
    {fried_line}
    {eda_grp}
    <li>퍼뮤테이션 중요도는 <strong>고정된 테스트 분할</strong>·macro OvR AUROC·반복 {perm_repeats}회 조건에서의
        ‘해당 모델·해당 전처리’에 대한 설명력입니다. 일반화나 인과로 바로 확장하지 않습니다.</li>
    <li>원시열 EDA 표는 수치형에 한정되며, 전처리 특징명→원시열 매핑은 휴리스틱입니다.
        범주형 원시열은 별도 빈도·교차표 점검이 필요합니다.</li>
    <li>Native importance(LGBM 등)와 퍼뮤테이션 중요도가 어긋나는 특징이 있으면,
        스케일·다중공선성·모델 구조 차이를 의심하고 추가 검증(예: SHAP, 부분의존)을 권장합니다.</li>
  </ul>

  <h2>7. 향후 추가 과제</h2>
  <ul class="insights-list">
    <li><strong>교차 검증 기반 퍼뮤테이션</strong>: K-fold 또는 반복 홀드아웃으로 중요도 분산·신뢰구간을 추정합니다.</li>
    <li><strong>SHAP / 부분의존(PDP)</strong>: 모델별 국소 설명과 비선형·상호작용 패턴을 보완합니다.</li>
    <li><strong>다중검정·효과크기</strong>: Friedman 외에 효과크기(예: Kendall’s W per feature)와 FDR 등 대안 보정을 비교합니다.</li>
    <li><strong>누수·정합성 점검</strong>: 연도·지역·제품 등으로 그룹 외삽 시 성능·중요도 변화를 점검합니다.</li>
    <li><strong>범주형·서열 변수 EDA</strong>: 원-핫 전 특성에 대한 분포·교차표·타깃 조건부 평균을 리포트에 포함합니다.</li>
    <li><strong>퍼뮤 반복·시드</strong>: repeats·random_state를 늘려 안정성을 보고, 알고리즘 비교 리포트와 동일한 다시드 프로토콜을 맞춥니다.</li>
    <li><strong>제외 특성 민감도</strong>: 영역 total {n_excluded}개 제외 전·후 중요도/성능 변화를 정리합니다.</li>
    <li><strong>대시보드 연계</strong>: 상위 {top_k_eda}개(현재 표 {n_eda_rows}행)와 알고리즘 비교·FE 벤치마크 지표를 한 화면에서 추적합니다.</li>
  </ul>
"""


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


def macro_ovr_auc(y_true, y_proba) -> float:
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2 or y_proba.shape[1] < 2:
        return 0.0
    return float(
        roc_auc_score(
            y_true,
            y_proba,
            labels=np.arange(y_proba.shape[1]),
            multi_class="ovr",
            average="macro",
        )
    )


def perm_score_macro_auc(est, X, y, **kwargs) -> float:
    _ = kwargs
    return macro_ovr_auc(y, est.predict_proba(X))


def fig_to_base64(fig: Figure, dpi: int = 110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def native_importance_vector(name: str, est, n_feat: int) -> np.ndarray:
    if name == "LightGBM":
        v = np.asarray(est.feature_importances_, dtype=float)
    elif name == "Logistic Regression":
        v = np.abs(est.coef_).mean(axis=0)
    elif name == "Ordinal LogisticAT":
        v = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
        if v.size != n_feat:
            v = np.abs(np.asarray(est.coef_, dtype=float)).reshape(-1)[:n_feat]
    else:
        raise ValueError(name)
    if v.shape[0] != n_feat:
        raise RuntimeError(f"{name} importance len {v.shape[0]} != {n_feat}")
    return v / (v.sum() + 1e-12)


def kendall_w(rank_matrix: np.ndarray) -> tuple[float, float, float]:
    """
    rank_matrix: (n_objects, k_raters) 각 열은 1..n_objects 순열(또는 동률 평균순위).
    반환: W, chi2, p_value (근사 chi2 분포, df=n-1)
    """
    n, k = rank_matrix.shape
    R = rank_matrix.sum(axis=1)
    R_mean = R.mean()
    s = float(np.sum((R - R_mean) ** 2))
    w = 12 * s / (k**2 * (n**3 - n) + 1e-15)
    chi2_stat = k * (n - 1) * w
    p = float(1 - stats.chi2.cdf(chi2_stat, df=n - 1))
    return float(w), float(chi2_stat), p


def column_stats_series(s: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce")
    n = len(s)
    miss = float(s.isna().mean())
    s2 = s.dropna()
    if len(s2) < 5:
        return {
            "missing_rate": miss,
            "cv": np.nan,
            "iqr": np.nan,
            "outlier_rate": np.nan,
            "skew": np.nan,
        }
    q1, q3 = s2.quantile(0.25), s2.quantile(0.75)
    iqr = float(q3 - q1) if q3 > q1 else float(s2.std() + 1e-9)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_rate = float(((s2 < lo) | (s2 > hi)).mean())
    mu = float(s2.mean())
    sd = float(s2.std()) + 1e-9
    cv = float(sd / (abs(mu) + 1e-6))
    skew = float(stats.skew(s2, bias=False)) if len(s2) > 2 else 0.0
    return {
        "missing_rate": miss,
        "cv": cv,
        "iqr": float(iqr),
        "outlier_rate": outlier_rate,
        "skew": skew,
    }


def main() -> None:
    train_path = DATA_DIR / "customer_satisfaction_train.csv"
    test_path = DATA_DIR / "customer_satisfaction_test.csv"
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)
    df_all = pd.concat([df_tr, df_te], axis=0, ignore_index=True)

    feature_cols = [
        c for c in df_tr.columns if c != TARGET and c not in EXCLUDE_COLS
    ]
    numeric_cols = [c for c in feature_cols if c not in CAT_COLS]

    X_train = df_tr[feature_cols]
    X_test = df_te[feature_cols]
    le = LabelEncoder()
    y_train = le.fit_transform(df_tr[TARGET].astype(int).values)
    y_test = le.transform(df_te[TARGET].astype(int).values)

    pre = build_preprocessor(numeric_cols)
    Xtr = pre.fit_transform(X_train, y_train)
    Xte = pre.transform(X_test)
    feat_names = list(pre.get_feature_names_out())
    n_feat = len(feat_names)

    def feat_to_raw(fname: str) -> str | None:
        if fname.startswith("num__"):
            return fname.replace("num__", "", 1)
        return None

    auc_scorer = perm_score_macro_auc

    models: dict[str, object] = {
        "LightGBM": LGBMClassifier(**LGBM_PARAMS),
        "Logistic Regression": LogisticRegression(**LR_PARAMS),
        "Ordinal LogisticAT": LogisticAT(**ORDINAL_PARAMS),
    }

    fitted: dict[str, object] = {}
    for name, clf in models.items():
        if name == "LightGBM":
            tr, te = as_lgbm_frame(Xtr), as_lgbm_frame(Xte)
        else:
            tr, te = np.asarray(Xtr), np.asarray(Xte)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(tr, y_train)
        fitted[name] = (clf, te)

    # --- native importance ---
    native: dict[str, np.ndarray] = {}
    for name, (clf, _) in fitted.items():
        native[name] = native_importance_vector(name, clf, n_feat)

    # --- permutation importance (test): 1회/모델, 전체 importances 저장 ---
    perm_full: dict[str, np.ndarray] = {}
    perm_means: dict[str, np.ndarray] = {}
    perm_std: dict[str, np.ndarray] = {}
    for name, (clf, te) in fitted.items():
        r = permutation_importance(
            clf,
            te,
            y_test,
            scoring=auc_scorer,
            n_repeats=PERM_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        perm_full[name] = r.importances
        perm_means[name] = np.asarray(r.importances_mean, dtype=float)
        perm_std[name] = np.asarray(r.importances_std, dtype=float)

    # 합의 상위 30: 퍼뮤테이션 평균을 모델별 정규화 후 평균 점수
    def norm(v: np.ndarray) -> np.ndarray:
        return v / (v.sum() + 1e-12)

    consensus = (
        norm(perm_means["LightGBM"])
        + norm(perm_means["Logistic Regression"])
        + norm(perm_means["Ordinal LogisticAT"])
    ) / 3.0
    top_idx_list = np.argsort(-consensus)[:TOP_K_EDA]

    # --- Kendall W on ranks of permutation means ---
    rank_mat = np.column_stack(
        [
            stats.rankdata(-perm_means["LightGBM"]),
            stats.rankdata(-perm_means["Logistic Regression"]),
            stats.rankdata(-perm_means["Ordinal LogisticAT"]),
        ]
    )
    w_val, chi2_w, p_w = kendall_w(rank_mat)

    spear_rows = []
    pairs = [
        ("LightGBM", "Logistic Regression"),
        ("LightGBM", "Ordinal LogisticAT"),
        ("Logistic Regression", "Ordinal LogisticAT"),
    ]
    for a, b in pairs:
        rho, p = stats.spearmanr(perm_means[a], perm_means[b])
        spear_rows.append({"a": a, "b": b, "rho": float(rho), "p": float(p)})

    # Friedman: 상위 30 특징 × 3모델 × 반복 (저장된 perm_full 사용)
    fried_rows_html = []
    fried_sig_count = 0
    m_tests = len(top_idx_list)
    for rank, j in enumerate(top_idx_list, start=1):
        samples = [
            perm_full["LightGBM"][j],
            perm_full["Logistic Regression"][j],
            perm_full["Ordinal LogisticAT"][j],
        ]
        try:
            _, p = stats.friedmanchisquare(*samples)
            p_adj = min(1.0, p * m_tests)
        except Exception:
            p, p_adj = float("nan"), float("nan")
        if not np.isnan(p_adj) and p_adj < 0.05:
            fried_sig_count += 1
        fried_rows_html.append(
            "<tr>"
            f"<td class='num'>{rank}</td>"
            f"<td>{html.escape(feat_names[j])}</td>"
            f"<td class='num'>{p:.4g}</td>"
            f"<td class='num'>{p_adj:.4g}</td>"
            "</tr>"
        )

    # --- Plots: top 20 bar per model (native) ---
    plot_html_parts: list[str] = []
    for name in ["LightGBM", "Logistic Regression", "Ordinal LogisticAT"]:
        v = native[name]
        order = np.argsort(-v)[:TOP_K_DISPLAY]
        labels = [feat_names[i][:40] for i in order]
        vals = v[order]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(labels[::-1], vals[::-1], color="steelblue")
        ax.set_title(f"{name} — native importance (top {TOP_K_DISPLAY})")
        ax.set_xlabel("normalized (sum=1)")
        plot_html_parts.append(
            f'<h3>{html.escape(name)}</h3><p><img src="data:image/png;base64,{fig_to_base64(fig)}" alt=""/></p>'
        )

    # Heatmap: Spearman correlation between models' permutation vectors
    M = np.column_stack(
        [
            perm_means["LightGBM"],
            perm_means["Logistic Regression"],
            perm_means["Ordinal LogisticAT"],
        ]
    )
    corr = pd.DataFrame(M, columns=["LGBM", "LR", "Ord"]).corr(method="spearman").values
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    labs = ["LGBM", "LR", "Ord"]
    ax.set_xticklabels(labs)
    ax.set_yticklabels(labs)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax)
    ax.set_title("퍼뮤테이션 중요도 벡터 — Spearman 상관")
    plot_html_parts.append(
        f'<h3>모델 간 중요도 벡터 상관</h3><p><img src="data:image/png;base64,{fig_to_base64(fig)}" alt=""/></p>'
    )

    # Grouped bar top 15 features by consensus name
    top15 = np.argsort(-consensus)[:15]
    x = np.arange(len(top15))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width,
        perm_means["LightGBM"][top15],
        width,
        label="LGBM perm",
    )
    ax.bar(x, perm_means["Logistic Regression"][top15], width, label="LR perm")
    ax.bar(
        x + width,
        perm_means["Ordinal LogisticAT"][top15],
        width,
        label="Ord perm",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([feat_names[i][:18] for i in top15], rotation=45, ha="right")
    ax.legend()
    ax.set_title("상위 15 특징 — 퍼뮤테이션 AUROC 하락폭 (세 모델)")
    ax.set_ylabel("mean drop (macro OvR AUROC)")
    plot_html_parts.append(
        f'<h3>상위 특징 퍼뮤테이션 중요도 비교</h3><p><img src="data:image/png;base64,{fig_to_base64(fig)}" alt=""/></p>'
    )

    # EDA: map feature names to raw columns
    eda_records = []
    for j in top_idx_list:
        fn = feat_names[j]
        raw = feat_to_raw(fn)
        if raw is None or raw not in df_all.columns:
            continue
        st = column_stats_series(df_all[raw])
        eda_records.append(
            {
                "feature": fn,
                "raw_column": raw,
                "consensus": float(consensus[j]),
                **st,
            }
        )

    eda_df = pd.DataFrame(eda_records)
    if len(eda_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for ax, col, ttl in zip(
            axes.ravel(),
            ["missing_rate", "cv", "outlier_rate", "skew"],
            ["Missing rate", "CV", "Outlier rate (IQR)", "Skewness"],
        ):
            sns.boxplot(data=eda_df, y=col, ax=ax, color="lightblue")
            ax.set_title(ttl + " (top consensus)")
        plt.tight_layout()
        plot_html_parts.append(
            f'<h3>상위 특징 EDA 분포</h3><p><img src="data:image/png;base64,{fig_to_base64(fig)}" alt=""/></p>'
        )

        # Scatter: consensus vs CV per raw column
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(eda_df["consensus"], eda_df["cv"], alpha=0.75, c="teal")
        for _, row in eda_df.nlargest(8, "consensus").iterrows():
            ax.annotate(
                str(row["raw_column"])[:14],
                (row["consensus"], row["cv"]),
                fontsize=7,
                alpha=0.9,
            )
        ax.set_xlabel("consensus importance")
        ax.set_ylabel("CV (raw column)")
        ax.set_title("Consensus vs variability")
        plt.tight_layout()
        plot_html_parts.append(
            f'<h3>합의 중요도 vs 변동성(CV)</h3>'
            f'<p class="meta">중요할수록 분산이 작다기보다는, 세부 축별 스케일 차이를 함께 봅니다.</p>'
            f'<p><img src="data:image/png;base64,{fig_to_base64(fig)}" alt=""/></p>'
        )

        # Histograms: top 6 raw columns by consensus
        top6 = eda_df.sort_values("consensus", ascending=False).head(6)
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for ax, (_, row) in zip(axes.ravel(), top6.iterrows()):
            col = row["raw_column"]
            sns.histplot(df_all[col].dropna(), kde=True, ax=ax, color="steelblue")
            ax.set_title(str(col)[:22], fontsize=9)
        plt.suptitle("Top-6 raw columns (train+test) distribution", y=1.02)
        plt.tight_layout()
        plot_html_parts.append(
            f'<h3>상위 6개 원시 특성 분포</h3><p><img src="data:image/png;base64,{fig_to_base64(fig)}" alt=""/></p>'
        )

    # Important vs unimportant: pool numeric columns from feature_cols
    imp_raw = {feat_to_raw(feat_names[j]) for j in top_idx_list if feat_to_raw(feat_names[j])}
    imp_raw.discard(None)
    bot_idx_list = np.argsort(consensus)[:TOP_K_EDA]
    bot_raw = {feat_to_raw(feat_names[j]) for j in bot_idx_list if feat_to_raw(feat_names[j])}
    all_raw_num = set(numeric_cols)
    unimp_raw = (all_raw_num - imp_raw) & bot_raw
    if not unimp_raw:
        unimp_raw = all_raw_num - imp_raw

    pool_imp = []
    pool_un = []
    for c in sorted(imp_raw):
        if c in df_all.columns:
            pool_imp.append(column_stats_series(df_all[c]))
    take_n = max(len(pool_imp), 30)
    for c in sorted(unimp_raw)[:take_n]:
        if c in df_all.columns:
            pool_un.append(column_stats_series(df_all[c]))

    cv_imp_mean = cv_un_mean = None
    p_cv_g = p_out_g = p_miss_g = None

    if pool_imp and pool_un:
        df_imp = pd.DataFrame(pool_imp)
        df_un = pd.DataFrame(pool_un)
        df_imp["group"] = "상위합의(근사)"
        df_un["group"] = "기타 수치형"
        cmp_df = pd.concat([df_imp, df_un], ignore_index=True)
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        for ax, col, ttl in zip(
            axes,
            ["missing_rate", "cv", "outlier_rate"],
            ["Missing rate", "CV", "Outlier rate"],
        ):
            sns.violinplot(data=cmp_df, x="group", y=col, ax=ax, cut=0)
            ax.set_title(ttl)
            ax.set_xlabel("")
        plt.tight_layout()
        plot_html_parts.append(
            f'<h3>중요(합의 상위) vs 기타 수치형 — 분포 비교</h3>'
            f'<p class="meta">합의 상위에 등장한 원시 수치 컬럼 {len(imp_raw)}개 vs '
            f'합의 하위에서 매핑된 기타 수치형 {len(pool_un)}개(최대 {take_n}개 샘플).</p>'
            f'<p><img src="data:image/png;base64,{fig_to_base64(fig)}" alt=""/></p>'
        )

        # Welch t-test on CV and outlier_rate
        t_cv, p_cv = stats.ttest_ind(
            df_imp["cv"].dropna(), df_un["cv"].dropna(), equal_var=False, nan_policy="omit"
        )
        t_out, p_out = stats.ttest_ind(
            df_imp["outlier_rate"].dropna(),
            df_un["outlier_rate"].dropna(),
            equal_var=False,
            nan_policy="omit",
        )
        t_miss, p_miss = stats.ttest_ind(
            df_imp["missing_rate"].dropna(),
            df_un["missing_rate"].dropna(),
            equal_var=False,
            nan_policy="omit",
        )
        cv_imp_mean = float(df_imp["cv"].dropna().mean())
        cv_un_mean = float(df_un["cv"].dropna().mean())
        p_cv_g, p_out_g, p_miss_g = float(p_cv), float(p_out), float(p_miss)
        group_test_html = (
            "<table><thead><tr><th>지표</th><th>Welch t (상위−기타)</th><th>p-value</th></tr></thead><tbody>"
            f"<tr><td>CV</td><td class='num'>{t_cv:.3f}</td><td class='num'>{p_cv:.4g}</td></tr>"
            f"<tr><td>이상치 비율</td><td class='num'>{t_out:.3f}</td><td class='num'>{p_out:.4g}</td></tr>"
            f"<tr><td>결측률</td><td class='num'>{t_miss:.3f}</td><td class='num'>{p_miss:.4g}</td></tr>"
            "</tbody></table>"
        )
    else:
        group_test_html = "<p>EDA 그룹 비교 데이터 부족</p>"

    # EDA table HTML
    if len(eda_df) > 0:
        eda_rows = []
        for _, row in eda_df.iterrows():
            eda_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row['feature']))}</td>"
                f"<td>{html.escape(str(row['raw_column']))}</td>"
                f"<td class='num'>{row['consensus']:.5f}</td>"
                f"<td class='num'>{row['missing_rate']:.4f}</td>"
                f"<td class='num'>{row['cv']:.4f}</td>"
                f"<td class='num'>{row['outlier_rate']:.4f}</td>"
                f"<td class='num'>{row['skew']:.3f}</td>"
                "</tr>"
            )
        eda_table = (
            "<table><thead><tr><th>특징(전처리명)</th><th>원시열</th><th>합의점수</th>"
            "<th>결측률</th><th>CV</th><th>이상치비율</th><th>왜도</th></tr></thead><tbody>"
            + "".join(eda_rows)
            + "</tbody></table>"
        )
    else:
        eda_table = "<p>수치형 매핑 가능한 상위 특징이 없습니다.</p>"

    spear_table = (
        "<table><thead><tr><th>비교</th><th>Spearman ρ</th><th>p</th></tr></thead><tbody>"
        + "".join(
            f"<tr><td>{html.escape(r['a'])} vs {html.escape(r['b'])}</td>"
            f"<td class='num'>{r['rho']:.4f}</td><td class='num'>{r['p']:.4g}</td></tr>"
            for r in spear_rows
        )
        + "</tbody></table>"
    )

    params_json = html.escape(
        json.dumps(
            {
                "excluded": list(EXCLUDE_COLS),
                "n_features_dense": n_feat,
                "perm_repeats": PERM_REPEATS,
                "top_k_eda": TOP_K_EDA,
                "kendall_W": w_val,
                "kendall_chi2": chi2_w,
                "kendall_p": p_w,
                "friedman_top30_bonferroni_sig_count": fried_sig_count,
            },
            indent=2,
        )
    )

    insights_and_future = build_insights_future_sections(
        w_val=w_val,
        chi2_w=chi2_w,
        p_w=p_w,
        spear_rows=spear_rows,
        fried_sig_count=fried_sig_count,
        m_tests=m_tests,
        perm_repeats=PERM_REPEATS,
        top_k_eda=TOP_K_EDA,
        n_eda_rows=len(eda_df),
        cv_imp_mean=cv_imp_mean,
        cv_un_mean=cv_un_mean,
        p_cv_g=p_cv_g,
        p_out_g=p_out_g,
        p_miss_g=p_miss_g,
        n_excluded=len(EXCLUDE_COLS),
    )

    doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <title>특성 중요도 및 EDA 리포트 (total 제외)</title>
  <style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 24px; max-width: 1100px; color: #111; line-height: 1.5; }}
    h1 {{ font-size: 1.35rem; border-bottom: 2px solid #1d4ed8; padding-bottom: 6px; }}
    h2 {{ font-size: 1.1rem; color: #1e3a8a; margin-top: 1.4rem; }}
    h3 {{ font-size: 0.98rem; color: #334155; margin-top: 1rem; }}
    p.meta {{ color: #555; font-size: 0.88rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; margin: 10px 0; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; }}
    th {{ background: #eff6ff; text-align: left; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .note {{ background: #f1f5f9; border-left: 4px solid #64748b; padding: 12px; margin: 12px 0; }}
    ul.insights-list {{ padding-left: 1.25rem; margin: 0.4rem 0 1rem; }}
    ul.insights-list li {{ margin: 0.45rem 0; }}
  </style>
</head>
<body>
  <h1>특성 중요도 · EDA · 통계 검정 (영역별 total 12개 제외)</h1>
  <p class="meta">데이터: train/test 합산 EDA, 모델 학습은 train만. 퍼뮤테이션 중요도는 <strong>테스트셋</strong>·
     scoring=<strong>macro OvR AUROC</strong>·반복 <strong>{PERM_REPEATS}</strong>회.</p>

  <div class="note">
    <strong>Kendall 조화계수 W</strong> (3모델의 퍼뮤테이션 중요도 순위 일치도): W=<strong>{w_val:.4f}</strong>,
    χ²≈<strong>{chi2_w:.2f}</strong>, 근사 p=<strong>{p_w:.4g}</strong>.
    W가 1에 가까울수록 모델들이 비슷한 순위로 특징을 평가합니다.
  </div>

  <h2>1. 모델 간 퍼뮤테이션 중요도 벡터 상관 (Spearman)</h2>
  {spear_table}

  <h2>2. 시각화</h2>
  {chr(10).join(plot_html_parts)}

  <h2>3. 상위 {TOP_K_EDA} 특징 합의 점수 기반 EDA (수치형 원시열)</h2>
  <p class="meta">합의 점수 = 세 모델 퍼뮤테이션 평균을 정규화한 뒤 산술평균.</p>
  {eda_table}

  <h2>4. 중요 vs 비중요 그룹 검정 (Welch t)</h2>
  {group_test_html}

  <h2>5. 상위 30 특징 — 모델 간 퍼뮤테이션 점수 Friedman 검정</h2>
  <p class="meta">각 특징마다 3모델 × {PERM_REPEATS}반복 점수에 Friedman 검정. p에 Bonferroni(×{m_tests}) 보정.</p>
  <table>
    <thead><tr><th>순위</th><th>특징</th><th>p (Friedman)</th><th>p×{m_tests} (Bonferroni)</th></tr></thead>
    <tbody>{chr(10).join(fried_rows_html)}</tbody>
  </table>

{insights_and_future}

  <h2>8. 설정 JSON</h2>
  <pre style="background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;font-size:0.78rem;overflow:auto;">{params_json}</pre>
</body>
</html>
"""

    OUT_HTML.write_text(doc, encoding="utf-8")
    print("Wrote", OUT_HTML)


if __name__ == "__main__":
    main()
