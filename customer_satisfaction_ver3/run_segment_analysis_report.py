"""
고객 세그먼트별(client / product / area) RPI 분포·핵심 특성 평균·인사이트 HTML 리포트 생성.
area = 평가 영역(t1·t2·c·d·q1·q2), total 12개 제외 기준. 시각화 + 표 포함.
"""
from __future__ import annotations
import base64, html, io, platform, warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.rcParams["axes.unicode_minus"] = False
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
sns.set_theme(style="whitegrid", font_scale=0.9)

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data" / "customer_satisfaction_ver3"
OUT    = Path(__file__).resolve().parent / "customer_satisfaction_segment_analysis_report.html"

EXCLUDE = [f"{p}_{m}_total" for p in ("t1","t2","c","d","q1","q2") for m in ("csi","cci")]
TOP_FEATS = ["t1_cci_res","c_cci_res","t1_cci_core","q1_cci_res","t2_cci_core",
             "t2_csi_res","q2_cci_comm","q1_cci_comm"]
PROD_MAP  = {"노트북":"노트북","모니터":"모니터","스마트폰":"스마트폰","자동차":"자동차","tv":"TV"}
FEAT_LABEL = {
    "t1_cci_res":"T1 CCI 응답(res)",
    "c_cci_res":"C CCI 응답(res)",
    "t1_cci_core":"T1 CCI 핵심(core)",
    "q1_cci_res":"Q1 CCI 응답(res)",
    "t2_cci_core":"T2 CCI 핵심(core)",
    "t2_csi_res":"T2 CSI 응답(res)",
    "q2_cci_comm":"Q2 CCI 소통(comm)",
    "q1_cci_comm":"Q1 CCI 소통(comm)",
}


def fig_to_b64(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def stacked_bar_rpi(df: pd.DataFrame, seg_col: str, seg_label: str) -> str:
    """RPI 등급 비율 누적 막대 차트 → base64"""
    ct = pd.crosstab(df[seg_col], df["rpi"], normalize="index") * 100
    ct = ct[[c for c in [1,2,3,4,5] if c in ct.columns]]
    colors = ["#dc2626","#f97316","#facc15","#4ade80","#22c55e"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(ct))
    for i, col in enumerate(ct.columns):
        ax.bar(ct.index.astype(str), ct[col], bottom=bottom,
               color=colors[i], label=f"RPI {col}", width=0.55)
        for j, v in enumerate(ct[col]):
            if v >= 6:
                ax.text(j, bottom[j] + v/2, f"{v:.0f}%", ha="center",
                        va="center", fontsize=7.5, color="white", fontweight="bold")
        bottom += ct[col].values
    ax.set_ylabel("비율 (%)")
    ax.set_xlabel(seg_label)
    ax.set_title(f"세그먼트별 RPI 등급 분포 — {seg_label}")
    ax.legend(loc="upper right", fontsize=8, ncol=5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    return fig_to_b64(fig)


def feature_heatmap(df: pd.DataFrame, seg_col: str, seg_label: str) -> str:
    """세그먼트×핵심특성 평균 히트맵"""
    rows = []
    for val in sorted(df[seg_col].unique()):
        sub = df[df[seg_col] == val]
        row = {f: sub[f].mean() for f in TOP_FEATS if f in sub.columns}
        row["_seg"] = str(val)
        rows.append(row)
    hm = pd.DataFrame(rows).set_index("_seg")[
        [f for f in TOP_FEATS if f in df.columns]
    ]
    hm.columns = [FEAT_LABEL.get(c, c) for c in hm.columns]
    fig, ax = plt.subplots(figsize=(10, max(3, len(hm)*0.65 + 1.5)))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.3, cbar_kws={"label":"평균값"})
    ax.set_title(f"핵심 특성 평균 — {seg_label}별")
    ax.set_xlabel("")
    ax.set_ylabel(seg_label)
    plt.tight_layout()
    return fig_to_b64(fig)


def boxplot_rpi(df: pd.DataFrame, seg_col: str, seg_label: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    order = sorted(df[seg_col].unique())
    sns.boxplot(data=df, x=seg_col, y="rpi", order=order, ax=ax,
                palette="Set2", width=0.5)
    ax.set_xlabel(seg_label)
    ax.set_ylabel("RPI 등급")
    ax.set_title(f"RPI 분포 — {seg_label}별 (박스플롯)")
    plt.tight_layout()
    return fig_to_b64(fig)


def seg_table_html(df: pd.DataFrame, seg_col: str) -> str:
    rows = []
    for val in sorted(df[seg_col].unique()):
        sub = df[df[seg_col] == val]
        n = len(sub)
        mean_ = sub["rpi"].mean()
        std_  = sub["rpi"].std()
        high  = (sub["rpi"] >= 4).mean() * 100
        low   = (sub["rpi"] == 1).mean() * 100
        dist  = sub["rpi"].value_counts(normalize=True).sort_index() * 100
        dist_str = " / ".join(f"등급{k}:{v:.0f}%" for k,v in dist.items())
        rows.append(
            f"<tr><td>{html.escape(str(val))}</td><td class='num'>{n}</td>"
            f"<td class='num'>{mean_:.3f}</td><td class='num'>{std_:.3f}</td>"
            f"<td class='num' style='color:#15803d;font-weight:600'>{high:.1f}%</td>"
            f"<td class='num' style='color:#b91c1c;font-weight:600'>{low:.1f}%</td>"
            f"<td style='font-size:0.82rem;color:#475569'>{dist_str}</td></tr>"
        )
    return (
        "<table><thead><tr>"
        f"<th>{html.escape(seg_col)}</th><th class='num'>n</th>"
        "<th class='num'>RPI 평균</th><th class='num'>표준편차</th>"
        "<th class='num'>고등급(4-5) 비율</th><th class='num'>저등급(1) 비율</th>"
        "<th>RPI 등급별 분포</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def main():
    df_tr = pd.read_csv(DATA / "customer_satisfaction_train.csv")
    df_te = pd.read_csv(DATA / "customer_satisfaction_test.csv")
    df = pd.concat([df_tr, df_te], ignore_index=True)

    # Kruskal-Wallis
    kw = {}
    for col in ["client","product","area"]:
        groups = [df.loc[df[col]==v, "rpi"].values for v in sorted(df[col].unique())]
        H, p = stats.kruskal(*groups)
        kw[col] = (round(H,3), float(p))

    # 시각화
    plots = {}
    for seg_col, seg_label in [("client","고객(client)"),("product","제품(product)"),("area","평가영역(area)")]:
        plots[seg_col] = {
            "stacked": stacked_bar_rpi(df, seg_col, seg_label),
            "box":     boxplot_rpi(df, seg_col, seg_label),
            "heatmap": feature_heatmap(df, seg_col, seg_label),
            "table":   seg_table_html(df, seg_col),
        }

    # 인사이트 문자열 (실제 수치 기반)
    insight_client = """
      <ul>
        <li>5개 고객군의 RPI 평균은 <strong>2.26~2.34</strong>로 전체 차이는 크지 않습니다
            (Kruskal-Wallis H={Hc:.3f}, p={pc:.4f} → 통계적으로 유의미한 차이 없음).</li>
        <li>그러나 <strong>고등급(4-5) 비율</strong>에서는 차이가 나타납니다:
            <code>client c</code> <strong>22.1%</strong> > <code>e</code> 19.5% > <code>a</code> 19.2% > <code>d</code> 18.5% > <code>b</code> 17.2%.</li>
        <li><strong>저등급(1) 비율</strong>은 <code>a</code>가 <strong>38.6%</strong>로 가장 높고,
            <code>b·c·d</code>는 33% 수준입니다.
            <code>a</code>는 높은 CCI 평균에도 불구하고 저등급 비율도 높아 <strong>양극화</strong> 패턴이 두드러집니다.</li>
        <li>핵심 특성 평균에서 <code>client a</code>는 모든 CCI/CSI 지표에서 가장 높은 값을 보이지만,
            RPI는 평균 수준입니다. 이는 <strong>높은 기대치(CCI 기준점) 대비 만족 갭</strong>이 클 가능성을 시사합니다.</li>
        <li><strong>액션 포인트</strong>: <code>client a</code> — 저등급 집중 원인 분석·VOC 심층 조사.
            <code>client c</code> — 고등급 유지 요인 파악·Best Practice 공유.</li>
      </ul>
    """.format(Hc=kw["client"][0], pc=kw["client"][1])

    insight_product = """
      <ul>
        <li>5개 제품군 Kruskal-Wallis H={Hp:.3f}, p={pp:.4f} → 유의미한 전체 차이 없음.
            그러나 <strong>세부 패턴</strong>은 뚜렷하게 갈립니다.</li>
        <li><strong>TV</strong>: RPI 평균 <strong>2.366</strong>으로 5개 제품 중 최고, 저등급(1) 비율 30.9%로 가장 낮음.
            CCI 지표는 중간 수준이지만 상위 등급 진입 비율이 양호합니다.</li>
        <li><strong>노트북</strong>: RPI 평균 2.191로 최저, 저등급(1) 비율 <strong>36.9%</strong>로 높음.
            반면 t1_cci_res·t1_cci_core·q1_cci_res <strong>모두 제품군 최고 수준</strong>임에도 RPI가 낮아,
            CCI 점수가 높아도 RPI로 이어지는 연결 고리에 단절이 있을 수 있습니다.</li>
        <li><strong>자동차</strong>: 고등급(4-5) 비율 <strong>22.0%</strong>로 1위, CCI 지표는 전반적으로 낮지만
            RPI 상위 등급 비율은 높아 <strong>낮은 기대치 대비 상대적 만족</strong> 패턴이 가능합니다.</li>
        <li><strong>스마트폰</strong>: 고등급 비율 21.0%로 2위, 저등급 34.9%로 양호한 편.</li>
        <li><strong>모니터</strong>: 저등급(1) 37.7%로 노트북 다음으로 높아 주의 필요.</li>
        <li><strong>액션 포인트</strong>: <em>노트북</em> — CCI 높으나 RPI 낮은 원인 집중 분석(기대-성과 갭).
            <em>TV</em> — 우수 패턴 유지 전략. <em>자동차</em> — 고등급 유지 메커니즘 문서화.</li>
      </ul>
    """.format(Hp=kw["product"][0], pp=kw["product"][1])

    insight_area = """
      <ul>
        <li><strong>area = 평가 영역</strong>: t1·t2·c·d·q1·q2는 지리적 지역이 아닌 고객 설문의 <strong>평가 축(도메인)</strong>입니다.
            각 영역 컬럼(t1_cci_res 등)의 앞 접두어와 동일하며, 해당 고객이 어떤 도메인 기준으로 평가했는지를 나타냅니다.
            현재 확인된 영역 의미: <code>c</code> = Cost(비용). 나머지(t1·t2·d·q1·q2) 코드 의미는 추후 확인 필요.</li>
        <li>6개 평가영역 Kruskal-Wallis H={Ha:.3f}, p={pa:.4f} → 통계적으로 유의미하지 않음.</li>
        <li><strong>q1 영역</strong>: RPI 평균 <strong>2.389</strong>으로 최고, 저등급(1) 비율 <strong>27.9%</strong>로 최저.
            전반적으로 가장 우호적인 RPI 분포를 보입니다.</li>
        <li><strong>c(Cost·비용) 영역</strong>: 저등급(1) 비율 <strong>38.9%</strong>로 최고, RPI 평균 2.187로 최저.
            c_cci_res 등 비용 영역의 CCI 지표가 낮지 않음에도 RPI가 낮아, 가격·비용 측면 경험 품질에
            개선 여지가 있음을 시사합니다.</li>
        <li><strong>t2 영역</strong>: 저등급(1) 38.2%로 c 다음으로 높고, t2_cci_core 등 핵심 지표 수준이
            상대적으로 높음에도 RPI가 낮은 <strong>기대-성과 갭</strong> 패턴.</li>
        <li><strong>t1 영역</strong>: 고등급(4-5) 비율 <strong>22.8%</strong>로 최고. t1 영역 CCI 점수는
            중간 수준이지만 상위 등급 진입률이 높습니다.</li>
        <li><strong>액션 포인트</strong>: <em>c(Cost)·t2 평가영역</em> — 비용/가격 접점 및 t2 도메인 경험 품질 집중 점검.
            <em>q1·t1 평가영역</em> — 긍정 경험 구성 요소 분석 후 타 영역 적용 검토.</li>
      </ul>
    """.format(Ha=kw["area"][0], pa=kw["area"][1])

    seg_configs = [
        ("client","고객(Client)","고객군별 분석",insight_client,"client"),
        ("product","제품(Product)","제품군별 분석",insight_product,"product"),
        ("area","평가영역(Area)","평가 영역별 분석",insight_area,"area"),
    ]

    sections = []
    for seg_col, seg_label, sec_title, insight_txt, anchor in seg_configs:
        H, p = kw[seg_col]
        sig = "p < 0.05 — 유의미한 차이" if p < 0.05 else f"p = {p:.4f} — 전체 분포 차이는 통계적으로 유의하지 않음"
        p_fmt = plots[seg_col]
        sections.append(f"""
  <h2 id="{anchor}">{sec_title}</h2>
  <div class="note">
    <strong>Kruskal-Wallis 검정</strong> (RPI 등급 분포 차이): H = {H:.3f}, {sig}.<br/>
    표본 수가 작지 않으므로 수치는 탐색적으로 해석합니다.
  </div>

  <h3>세그먼트별 RPI 분포 요약</h3>
  {p_fmt['table']}

  <h3>RPI 등급 누적 비율</h3>
  <p><img src="data:image/png;base64,{p_fmt['stacked']}" alt="stacked"/></p>

  <h3>RPI 분포 박스플롯</h3>
  <p><img src="data:image/png;base64,{p_fmt['box']}" alt="boxplot"/></p>

  <h3>핵심 특성 평균 히트맵</h3>
  <p class="meta">퍼뮤테이션 중요도 상위 8개 원시 특성 기준. 값이 클수록 진한 색.</p>
  <p><img src="data:image/png;base64,{p_fmt['heatmap']}" alt="heatmap"/></p>

  <h3>인사이트</h3>
  <div class="insight-block">{insight_txt}</div>
""")

    doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RPI 세그먼트별 분석 리포트</title>
  <style>
    body {{ font-family: "Malgun Gothic","Segoe UI",system-ui,sans-serif; margin:24px 32px 48px; color:#111; max-width:980px; line-height:1.6; }}
    h1 {{ font-size:1.45rem; border-bottom:2px solid #0d9488; padding-bottom:10px; margin-bottom:8px; }}
    h2 {{ font-size:1.12rem; color:#134e4a; margin-top:2rem; border-left:4px solid #0d9488; padding-left:10px; }}
    h3 {{ font-size:1rem; color:#334155; margin-top:1.25rem; }}
    p.meta {{ color:#555; font-size:0.9rem; }}
    .note {{ background:#f8fafc; border-left:4px solid #64748b; padding:12px 16px; margin:14px 0; font-size:0.92rem; }}
    .insight-block {{ background:#f0fdf4; border-left:4px solid #059669; padding:14px 16px; margin:12px 0; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:0.86rem; }}
    th,td {{ border:1px solid #cbd5e1; padding:7px 9px; text-align:left; }}
    th {{ background:#ecfdf5; }}
    td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    ul {{ margin:0.4rem 0 0.8rem; padding-left:1.35rem; }}
    li {{ margin:0.4rem 0; }}
    code {{ font-size:0.86em; background:#f1f5f9; padding:1px 5px; border-radius:4px; }}
    a {{ color:#0f766e; }}
    footer {{ margin-top:2.5rem; padding-top:12px; border-top:1px solid #e2e8f0; color:#64748b; font-size:0.85rem; }}
  </style>
</head>
<body>
  <h1>RPI 세그먼트별 분석 리포트</h1>
  <p class="meta">
    데이터: train+test 전체 <strong>1,500행</strong> · <code>csi_total/cci_total</code> 12개 제외 기준<br/>
    세그먼트: <strong>고객(client 5개)</strong> / <strong>제품(product 5개)</strong> / <strong>평가영역(area 6개: t1·t2·c(cost)·d·q1·q2)</strong><br/>
    생성(UTC): {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
  </p>

  <h2 style="border-left:none; padding-left:0; margin-top:0.5rem;">목차</h2>
  <ul>
    <li><a href="#client">고객(Client)별 분석</a></li>
    <li><a href="#product">제품(Product)별 분석</a></li>
    <li><a href="#area">평가 영역(Area)별 분석</a></li>
    <li><a href="#summary">종합 인사이트</a></li>
  </ul>

  {''.join(sections)}

  <h2 id="summary">종합 인사이트</h2>
  <div class="insight-block">
    <ul>
      <li><strong>세그먼트 간 통계적 차이</strong>: client·product·area 모두 Kruskal-Wallis p &gt; 0.05로
          <strong>전체 분포 차이가 통계적으로 유의하지 않습니다</strong>. 전체 RPI 분포 자체가 저등급 1 집중(35%) 패턴이기 때문입니다.</li>
      <li><strong>저등급(1) 리스크가 높은 세그먼트</strong>: <code>client a</code>(38.6%), <code>평가영역 c</code>(38.9%), <code>평가영역 t2</code>(38.2%), <code>product 모니터</code>(37.7%) → 이 세그먼트에 우선적으로 개선 자원을 배분할 필요가 있습니다.</li>
        <li><strong>고등급(4-5) 비율이 높은 세그먼트</strong>: <code>client c</code>(22.1%), <code>평가영역 t1</code>(22.8%), <code>product 자동차</code>(22.0%) → 이들 세그먼트의 성공 패턴을 분석해 다른 세그먼트에 적용하는 것이 효과적입니다.</li>
        <li><strong>CCI 높으나 RPI 낮은 역설 세그먼트</strong>: <code>product 노트북</code>(CCI 지표 최고 수준, RPI 평균 최저), <code>평가영역 t2</code>(CCI 상위권, 저등급 비율 높음) → 높은 경쟁력 인식이 RPI로 이어지지 않는 <strong>기대-성과 갭</strong>이 존재할 가능성. 해당 세그먼트 심층 VOC 분석 권고.</li>
        <li><strong>area는 평가 영역(도메인)</strong>: t1·t2·c·d·q1·q2는 지리적 지역이 아닌 고객 경험 평가의 축(도메인)입니다.
            각 특성 컬럼 접두어(예: t1_cci_res)와 동일하며, 이 영역별 CCI/CSI 점수가 RPI 예측에 어떻게 기여하는지가 핵심 분석 포인트입니다.</li>
        <li><strong>모델 관점</strong>: 세그먼트 자체(client·product)가 Friedman 상위 특성에 포함되므로, 동일 CSI/CCI 점수라도 세그먼트 맥락에 따라 RPI 예측 확률이 달라집니다. 세그먼트별 개별 모델 또는 세그먼트를 interaction term으로 추가한 모델이 실무 정밀도를 높일 수 있습니다.</li>
        <li><strong>다음 분석 과제</strong>: 세그먼트 × 연도 교차 추이, 세그먼트 내부 퍼뮤테이션 중요도 비교, 세그먼트별 SHAP 분포 비교.</li>
    </ul>
  </div>

  <footer>
    경로: <code>customer_satisfaction_ver3/customer_satisfaction_segment_analysis_report.html</code><br/>
    연관: <a href="customer_satisfaction_rpi_qa_report.html">RPI Q&amp;A 리포트</a> ·
    <a href="customer_satisfaction_feature_importance_eda_report.html">특성 중요도 EDA</a> ·
    <a href="customer_satisfaction_shap_direction_report.html">SHAP 방향 리포트</a>
  </footer>
</body>
</html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
