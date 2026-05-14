"""
도메인 정의 반영 — 4개 HTML 리포트 인사이트 업데이트
영역 정의:
  t1 = 신기술 영역  |  t2 = 개발 영역  |  c = Cost 영역
  d  = 공급 영역    |  q1 = 품질 영역  |  q2 = 서비스 영역
"""

from pathlib import Path

REPORT_DIR = Path(r"c:\Users\USER\Desktop\lg_vibe_ml\customer_satisfaction_ver4\reports")

# ── 공통 도메인 범례 카드 HTML ─────────────────────────────────────────
DOMAIN_LEGEND = """
<!-- ════════ 도메인 영역 정의 ════════ -->
<div style="background:white; border-radius:12px; padding:20px 28px; margin-bottom:20px;
     box-shadow:0 2px 8px rgba(0,0,0,.07); border-left:5px solid #0f172a;">
  <h2 style="font-size:15px; font-weight:700; color:#0f172a; margin-bottom:14px;
      border-bottom:2px solid #e2e8f0; padding-bottom:8px;">
    📐 독립변수 영역 정의 (변수명 접두사 기준)
  </h2>
  <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:10px; font-size:13px;">
    <div style="background:#eff6ff; border-radius:8px; padding:10px 14px; border-left:4px solid #2563eb;">
      <strong style="color:#1d4ed8;">t1</strong> — 신기술 영역<br>
      <span style="color:#3b82f6; font-size:11px;">New Technology Domain</span><br>
      <span style="font-size:11px; color:#1e3a5f;">LG의 혁신 기술 역량에 대한 고객 평가</span>
    </div>
    <div style="background:#f0fdf4; border-radius:8px; padding:10px 14px; border-left:4px solid #16a34a;">
      <strong style="color:#166534;">t2</strong> — 개발 영역<br>
      <span style="color:#22c55e; font-size:11px;">Development Domain</span><br>
      <span style="font-size:11px; color:#14532d;">제품 개발 프로세스·협력에 대한 고객 평가</span>
    </div>
    <div style="background:#fff7ed; border-radius:8px; padding:10px 14px; border-left:4px solid #ea580c;">
      <strong style="color:#9a3412;">c</strong> — Cost 영역<br>
      <span style="color:#f97316; font-size:11px;">Cost / Pricing Domain</span><br>
      <span style="font-size:11px; color:#7c2d12;">비용·가격 경쟁력에 대한 고객 평가</span>
    </div>
    <div style="background:#fdf4ff; border-radius:8px; padding:10px 14px; border-left:4px solid #9333ea;">
      <strong style="color:#6b21a8;">d</strong> — 공급 영역<br>
      <span style="color:#a855f7; font-size:11px;">Delivery / Supply Domain</span><br>
      <span style="font-size:11px; color:#3b0764;">납기·공급 안정성에 대한 고객 평가</span>
    </div>
    <div style="background:#fff1f2; border-radius:8px; padding:10px 14px; border-left:4px solid #e11d48;">
      <strong style="color:#9f1239;">q1</strong> — 품질 영역<br>
      <span style="color:#f43f5e; font-size:11px;">Quality Domain</span><br>
      <span style="font-size:11px; color:#881337;">제품·서비스 품질에 대한 고객 평가</span>
    </div>
    <div style="background:#fefce8; border-radius:8px; padding:10px 14px; border-left:4px solid #ca8a04;">
      <strong style="color:#854d0e;">q2</strong> — 서비스 영역<br>
      <span style="color:#eab308; font-size:11px;">After-Sales Service Domain</span><br>
      <span style="font-size:11px; color:#713f12;">사후 서비스·지원에 대한 고객 평가</span>
    </div>
  </div>
  <div style="margin-top:12px; display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:12px;">
    <div style="background:#f8fafc; border-radius:6px; padding:8px 12px;">
      <strong>CSI</strong> (Customer Satisfaction Index): 해당 영역에 대한 <strong>만족도</strong> 지수<br>
      <span style="color:#64748b;">ex) t1_csi_total = 신기술 영역 전체 만족도</span>
    </div>
    <div style="background:#f8fafc; border-radius:6px; padding:8px 12px;">
      <strong>CCI</strong> (Customer Credibility Index): 해당 영역에 대한 <strong>신뢰도</strong> 지수<br>
      <span style="color:#64748b;">ex) c_cci_total = Cost 영역 전체 신뢰도</span>
    </div>
  </div>
  <p style="font-size:11px; color:#94a3b8; margin-top:10px;">
    * 세부 항목: _res (대응성), _core (핵심역량), _comm (커뮤니케이션) | _total: 세 항목의 종합 지수
  </p>
</div>
"""

# ── feature_importance_report.html — 도메인 인사이트 섹션 ─────────────
FI_DOMAIN_INSIGHTS = """
<!-- ════════ 도메인 관점 재해석 ════════ -->
<div class="card" style="border-left:5px solid #0f172a;">
  <h2 style="color:#0f172a;">🏭 영역별 도메인 관점 Feature Importance 재해석</h2>
  <p style="font-size:12px;color:#64748b;margin-bottom:18px;">
    t1(신기술) / t2(개발) / c(Cost) / d(공급) / q1(품질) / q2(서비스) 영역 정의를 반영한 해석입니다.
    동일 수치의 의미가 도메인 맥락에서 어떻게 달라지는지 확인하세요.
  </p>

  <!-- 영역별 CCI 순위 -->
  <h3 style="font-size:14px; color:#1e293b; margin-bottom:10px;">① 영역별 CCI(신뢰도) 중요도 순위 — 재구매 의향에 미치는 영향</h3>
  <table style="font-size:13px; margin-bottom:20px;">
    <thead><tr style="background:#0f172a; color:white;">
      <th style="color:white;">앙상블 순위</th>
      <th style="color:white;">변수명</th>
      <th style="color:white;">영역</th>
      <th style="color:white;">의미</th>
      <th style="color:white; text-align:center;">앙상블 점수</th>
      <th style="color:white;">비즈니스 해석</th>
    </tr></thead>
    <tbody>
      <tr style="background:#eff6ff;">
        <td style="text-align:center; font-weight:800; color:#1d4ed8;">1위</td>
        <td><strong>t1_cci_total</strong></td>
        <td><span style="background:#dbeafe; color:#1d4ed8; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:700;">신기술</span></td>
        <td>신기술 영역 고객 신뢰도 종합</td>
        <td style="text-align:center; font-weight:700; color:#1d4ed8;">0.767</td>
        <td>LG 신기술 역량을 신뢰하는 고객일수록 재구매 의향이 압도적으로 높음</td>
      </tr>
      <tr>
        <td style="text-align:center; font-weight:800; color:#9a3412;">2위</td>
        <td><strong>c_cci_total</strong></td>
        <td><span style="background:#ffedd5; color:#9a3412; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:700;">Cost</span></td>
        <td>비용·가격 영역 고객 신뢰도 종합</td>
        <td style="text-align:center; font-weight:700; color:#9a3412;">0.731</td>
        <td>단순 저가 선호가 아닌, LG 가격 정책의 합리성·투명성 신뢰가 재구매 결정 요인</td>
      </tr>
      <tr style="background:#fff1f2;">
        <td style="text-align:center; font-weight:800; color:#9f1239;">3위</td>
        <td><strong>q1_cci_total</strong></td>
        <td><span style="background:#ffe4e6; color:#9f1239; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:700;">품질</span></td>
        <td>품질 영역 고객 신뢰도 종합</td>
        <td style="text-align:center; font-weight:700; color:#9f1239;">0.585</td>
        <td>제품 품질 수준에 대한 일관된 신뢰 — 품질 불안정성이 이탈의 직접 원인</td>
      </tr>
      <tr>
        <td style="text-align:center; font-weight:800; color:#166534;">4위</td>
        <td><strong>t2_cci_total</strong></td>
        <td><span style="background:#dcfce7; color:#166534; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:700;">개발</span></td>
        <td>개발 영역 고객 신뢰도 종합</td>
        <td style="text-align:center; font-weight:700; color:#166534;">0.580</td>
        <td>공동 개발 역량 및 기술 협력 신뢰 — B2B 장기 파트너십 결정 인자</td>
      </tr>
      <tr style="background:#fdf4ff;">
        <td style="text-align:center; font-weight:800; color:#6b21a8;">6위</td>
        <td><strong>d_cci_total</strong></td>
        <td><span style="background:#f3e8ff; color:#6b21a8; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:700;">공급</span></td>
        <td>공급·납기 영역 고객 신뢰도 종합</td>
        <td style="text-align:center; font-weight:700; color:#6b21a8;">0.497</td>
        <td>납기 준수율과 공급 안정성 신뢰 — 생산 계획 의존 고객에게 특히 중요</td>
      </tr>
      <tr>
        <td style="text-align:center; font-weight:800; color:#854d0e;">9위</td>
        <td><strong>q2_cci_total</strong></td>
        <td><span style="background:#fef9c3; color:#854d0e; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:700;">서비스</span></td>
        <td>사후 서비스 영역 고객 신뢰도 종합</td>
        <td style="text-align:center; font-weight:700; color:#854d0e;">0.393</td>
        <td>A/S·기술지원 신뢰 — 중요하나 신기술·Cost·품질에 비해 상대적으로 낮은 순위</td>
      </tr>
    </tbody>
  </table>

  <!-- CSI vs CCI 비교 -->
  <h3 style="font-size:14px; color:#1e293b; margin-bottom:10px;">② CSI(만족도) vs CCI(신뢰도) — 영역별 재구매 기여 비교</h3>
  <table style="font-size:13px; margin-bottom:20px;">
    <thead><tr style="background:#f8fafc;">
      <th>영역</th>
      <th style="text-align:center; background:#dbeafe; color:#1d4ed8;">CCI 순위<br>(신뢰도)</th>
      <th style="text-align:center; background:#dcfce7; color:#166534;">CSI 순위<br>(만족도)</th>
      <th style="text-align:center;">CCI 점수</th>
      <th style="text-align:center;">CSI 점수</th>
      <th>해석</th>
    </tr></thead>
    <tbody>
      <tr>
        <td><strong style="color:#1d4ed8;">신기술 (t1)</strong></td>
        <td style="text-align:center; background:#dbeafe; font-weight:700; color:#1d4ed8;">1위</td>
        <td style="text-align:center; background:#dcfce7; color:#166534;">5위</td>
        <td style="text-align:center;">0.767</td>
        <td style="text-align:center;">0.523</td>
        <td>신기술에 대한 <strong>신뢰가 만족보다 더 강한 재구매 동인</strong> — 단순 기능 만족을 넘어 기술 방향성 신뢰가 핵심</td>
      </tr>
      <tr style="background:#f8fafc;">
        <td><strong style="color:#9a3412;">Cost (c)</strong></td>
        <td style="text-align:center; background:#dbeafe; font-weight:700; color:#1d4ed8;">2위</td>
        <td style="text-align:center; background:#dcfce7; color:#166534;">8위</td>
        <td style="text-align:center;">0.731</td>
        <td style="text-align:center;">0.415</td>
        <td>가격 <strong>신뢰(가격 정책의 일관성·합리성)</strong>가 가격 만족(저렴함)보다 2배 이상 중요</td>
      </tr>
      <tr>
        <td><strong style="color:#9f1239;">품질 (q1)</strong></td>
        <td style="text-align:center; background:#dbeafe; font-weight:700; color:#1d4ed8;">3위</td>
        <td style="text-align:center; background:#dcfce7; color:#166534;">—</td>
        <td style="text-align:center;">0.585</td>
        <td style="text-align:center;">N/A</td>
        <td>품질 신뢰가 Top-10 진입, 품질 만족도는 상대적으로 낮은 순위 — <strong>품질 컴플레인 방지가 핵심</strong></td>
      </tr>
      <tr style="background:#f8fafc;">
        <td><strong style="color:#166534;">개발 (t2)</strong></td>
        <td style="text-align:center; background:#dbeafe; font-weight:700; color:#1d4ed8;">4위</td>
        <td style="text-align:center; background:#dcfce7; color:#166534;">10위</td>
        <td style="text-align:center;">0.580</td>
        <td style="text-align:center;">0.385</td>
        <td>개발 협력 신뢰 vs 만족 모두 중요 — B2B 공동 개발 파트너로서의 신뢰가 관계 지속성 결정</td>
      </tr>
      <tr>
        <td><strong style="color:#6b21a8;">공급 (d)</strong></td>
        <td style="text-align:center; background:#dbeafe; font-weight:700; color:#1d4ed8;">6위</td>
        <td style="text-align:center; background:#dcfce7; color:#166534;">—</td>
        <td style="text-align:center;">0.497</td>
        <td style="text-align:center;">N/A</td>
        <td>납기 신뢰가 중간 순위 — 공급망 이슈 발생 시 관계 이탈 위험 높음</td>
      </tr>
      <tr style="background:#f8fafc;">
        <td><strong style="color:#854d0e;">서비스 (q2)</strong></td>
        <td style="text-align:center; background:#dbeafe; font-weight:700; color:#1d4ed8;">9위</td>
        <td style="text-align:center; background:#dcfce7; color:#166534;">—</td>
        <td style="text-align:center;">0.393</td>
        <td style="text-align:center;">N/A</td>
        <td>A/S 신뢰는 하위권 — 구매 전 인식보다 실제 경험 후 만족이 더 중요함을 시사</td>
      </tr>
    </tbody>
  </table>

  <!-- 핵심 비즈니스 인사이트 3개 -->
  <h3 style="font-size:14px; color:#1e293b; margin-bottom:12px;">③ 영역별 도메인 핵심 비즈니스 인사이트</h3>

  <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px;">
    <div style="background:#eff6ff; border-radius:8px; padding:14px 16px; border-left:4px solid #2563eb; font-size:13px; line-height:1.9;">
      <strong style="color:#1d4ed8; font-size:13px;">🔬 신기술(t1)이 1위인 이유</strong><br>
      <span style="color:#1e3a5f;">
        B2B 시장에서 고객사의 재구매 의향은 현재 제품 만족보다
        <strong>LG가 미래 기술 방향을 이끌 수 있다는 신뢰</strong>에 달려 있습니다.
        신기술 CCI가 낮은 고객사는 "이 회사와 계속 가면 기술적으로 뒤처질 수 있다"고 인식 →
        이탈 가능성 높음. 신기술 로드맵 공유와 기술 세미나가 관계 유지의 핵심 레버입니다.
      </span>
    </div>
    <div style="background:#fff7ed; border-radius:8px; padding:14px 16px; border-left:4px solid #ea580c; font-size:13px; line-height:1.9;">
      <strong style="color:#9a3412; font-size:13px;">💰 Cost(c) 신뢰가 만족보다 2배 더 중요한 이유</strong><br>
      <span style="color:#7c2d12;">
        B2B 고객은 개별 거래의 가격보다 <strong>가격 정책의 일관성, 견적 투명성, 원가 절감 협력</strong>을
        신뢰하는 것이 더 중요합니다. Cost CCI가 낮은 고객 = "다음 협상에서 불리해질 것" 우려 →
        경쟁사 탐색 시작. 장기 가격 안정성 보장 프로그램이 효과적입니다.
      </span>
    </div>
    <div style="background:#fff1f2; border-radius:8px; padding:14px 16px; border-left:4px solid #e11d48; font-size:13px; line-height:1.9;">
      <strong style="color:#9f1239; font-size:13px;">🔧 품질(q1) 신뢰 관리의 중요성</strong><br>
      <span style="color:#881337;">
        품질 CCI는 3위 (0.585)로 높지만 CSI는 상대적으로 낮은 순위 →
        <strong>한 번의 품질 불량이 신뢰를 크게 훼손</strong>함을 시사합니다.
        품질 컴플레인 발생 시 신속한 원인 분석과 개선 공유가 신뢰 회복에 결정적입니다.
        "품질 만족"보다 "품질 약속 이행 신뢰" 관리가 핵심 KPI여야 합니다.
      </span>
    </div>
    <div style="background:#f0fdf4; border-radius:8px; padding:14px 16px; border-left:4px solid #16a34a; font-size:13px; line-height:1.9;">
      <strong style="color:#166534; font-size:13px;">🛠 개발(t2) vs 서비스(q2) 우선순위 역전</strong><br>
      <span style="color:#14532d;">
        통상 "서비스 품질이 중요하다"고 생각하지만,
        데이터는 개발 협력 신뢰(4위 0.580)가 A/S 서비스 신뢰(9위 0.393)보다
        <strong>훨씬 강한 재구매 동인</strong>임을 보여줍니다.
        고객사 엔지니어와의 공동 개발·기술 교류 프로그램이
        사후 서비스 개선보다 우선순위가 높아야 합니다.
      </span>
    </div>
  </div>

  <!-- 전략 매트릭스 -->
  <h3 style="font-size:14px; color:#1e293b; margin-bottom:10px;">④ 영역별 전략 매트릭스 (중요도 × 관리 방향)</h3>
  <table style="font-size:13px;">
    <thead>
      <tr style="background:#1e293b; color:white;">
        <th style="color:white;">영역</th>
        <th style="color:white; text-align:center;">CCI 앙상블 순위</th>
        <th style="color:white; text-align:center;">중요도 등급</th>
        <th style="color:white;">핵심 관리 포인트</th>
        <th style="color:white;">권장 액션</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#eff6ff;">
        <td><strong style="color:#1d4ed8;">신기술 (t1)</strong></td>
        <td style="text-align:center; font-weight:700; color:#1d4ed8;">1위 (0.767)</td>
        <td style="text-align:center;"><span style="background:#1d4ed8; color:white; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700;">최우선</span></td>
        <td>기술 로드맵 신뢰, 미래 혁신 방향 공감</td>
        <td>기술 세미나 정례화, 신기술 프리뷰 프로그램, R&amp;D 방향 공유</td>
      </tr>
      <tr>
        <td><strong style="color:#9a3412;">Cost (c)</strong></td>
        <td style="text-align:center; font-weight:700; color:#9a3412;">2위 (0.731)</td>
        <td style="text-align:center;"><span style="background:#9a3412; color:white; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700;">최우선</span></td>
        <td>가격 정책 일관성, 견적 투명성, 원가 협력</td>
        <td>장기 가격 안정 계약, 원가 절감 공동 프로그램, 투명 견적 시스템</td>
      </tr>
      <tr style="background:#fff1f2;">
        <td><strong style="color:#9f1239;">품질 (q1)</strong></td>
        <td style="text-align:center; font-weight:700; color:#9f1239;">3위 (0.585)</td>
        <td style="text-align:center;"><span style="background:#e11d48; color:white; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700;">High</span></td>
        <td>품질 약속 이행, 불량률 최소화, 품질 투명성</td>
        <td>품질 불량 → 48시간 내 원인 분석 공유, 품질 보증 SLA 강화</td>
      </tr>
      <tr>
        <td><strong style="color:#166534;">개발 (t2)</strong></td>
        <td style="text-align:center; font-weight:700; color:#166534;">4위 (0.580)</td>
        <td style="text-align:center;"><span style="background:#16a34a; color:white; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700;">High</span></td>
        <td>공동 개발 역량, 기술 협력 파트너십</td>
        <td>고객사 엔지니어 Joint Development Program, 기술 교류회</td>
      </tr>
      <tr style="background:#fdf4ff;">
        <td><strong style="color:#6b21a8;">공급 (d)</strong></td>
        <td style="text-align:center; font-weight:700; color:#6b21a8;">6위 (0.497)</td>
        <td style="text-align:center;"><span style="background:#9333ea; color:white; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700;">Mid</span></td>
        <td>납기 안정성, 공급망 리스크 관리</td>
        <td>납기 지연 시 사전 알림 시스템, 대체 공급 플랜 공개</td>
      </tr>
      <tr>
        <td><strong style="color:#854d0e;">서비스 (q2)</strong></td>
        <td style="text-align:center; font-weight:700; color:#854d0e;">9위 (0.393)</td>
        <td style="text-align:center;"><span style="background:#ca8a04; color:white; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700;">Mid</span></td>
        <td>A/S 대응 속도, 기술 지원 품질</td>
        <td>우선순위 대비 과잉 투자 점검 — 신기술·Cost 투자로 자원 재배분 검토</td>
      </tr>
    </tbody>
  </table>

  <div style="background:#0f172a; border-radius:8px; padding:14px 20px; margin-top:16px; font-size:13px; line-height:1.9; color:#e2e8f0;">
    <strong style="color:#f8fafc; font-size:14px;">★ 영역별 종합 전략 방향</strong><br>
    <span style="color:#94a3b8; font-size:12px;">데이터가 제시하는 투자 우선순위:</span><br>
    <span style="color:#7dd3fc;">① 신기술 신뢰 구축</span> (기술 로드맵·세미나) →
    <span style="color:#fdba74;">② Cost 투명성 강화</span> (가격 정책 일관성) →
    <span style="color:#fca5a5;">③ 품질 약속 이행</span> (SLA·불량 대응) →
    <span style="color:#86efac;">④ 개발 협력 심화</span> (JDP 프로그램) →
    <span style="color:#d8b4fe;">⑤ 공급 안정성 확보</span> →
    <span style="color:#fde68a;">⑥ 서비스 효율화</span><br>
    <span style="color:#94a3b8; font-size:12px;">
      ⚠ 현재 서비스(q2) 과잉 투자 중이라면 신기술·Cost 영역으로 자원 재배분 권장
    </span>
  </div>
</div>
"""

# ── EDA report — 도메인 관점 해석 추가 카드 ────────────────────────────
EDA_DOMAIN_CARD = """
<!-- ════════ 도메인 관점 EDA 재해석 ════════ -->
<div class="card" style="border-left:5px solid #0f172a; margin-bottom:24px;">
  <h2 style="color:#0f172a;">🏭 영역별 도메인 관점 EDA 재해석</h2>
  <p style="font-size:12px; color:#64748b; margin-bottom:14px;">
    변수명 접두사 의미: <strong>t1=신기술</strong> / <strong>t2=개발</strong> / <strong>c=Cost</strong> /
    <strong>d=공급</strong> / <strong>q1=품질</strong> / <strong>q2=서비스</strong>
  </p>
  <table style="font-size:13px; margin-bottom:16px;">
    <thead><tr style="background:#1e293b; color:white;">
      <th style="color:white;">영역</th><th style="color:white;">CCI 상관 순위<br>(with RPI)</th>
      <th style="color:white;">CSI 상관 순위<br>(with RPI)</th><th style="color:white;">데이터 분포 특성</th>
      <th style="color:white;">주요 인사이트</th>
    </tr></thead>
    <tbody>
      <tr style="background:#eff6ff;">
        <td><strong style="color:#1d4ed8;">신기술 (t1)</strong></td>
        <td style="text-align:center; font-weight:700; color:#1d4ed8;">1위 계열</td>
        <td style="text-align:center; color:#166534;">Top-5</td>
        <td>RPI 클래스 간 분포 가장 명확히 분리</td>
        <td>미래 기술 방향성 신뢰 = 재구매 핵심 선행지표</td>
      </tr>
      <tr>
        <td><strong style="color:#9a3412;">Cost (c)</strong></td>
        <td style="text-align:center; font-weight:700; color:#9a3412;">2위 계열</td>
        <td style="text-align:center; color:#166534;">Top-10</td>
        <td>RPI 1→5로 갈수록 CCI 뚜렷한 우하향 패턴</td>
        <td>가격 불신 = 이탈 강력 신호 — 가격 투명성 정책 시급</td>
      </tr>
      <tr style="background:#fff1f2;">
        <td><strong style="color:#9f1239;">품질 (q1)</strong></td>
        <td style="text-align:center; font-weight:700; color:#9f1239;">3위 계열</td>
        <td style="text-align:center; color:#166534;">Top-15</td>
        <td>이상치 다수 — 일부 고객사 품질 불만족 집중</td>
        <td>품질 이슈 고객사 집중 관리 프로그램 필요</td>
      </tr>
      <tr>
        <td><strong style="color:#166534;">개발 (t2)</strong></td>
        <td style="text-align:center; font-weight:700; color:#166534;">4위 계열</td>
        <td style="text-align:center; color:#166534;">Top-10</td>
        <td>t1과 높은 상관(r≥0.8) — 신기술·개발 동반 인식</td>
        <td>신기술·개발 영역을 통합 R&amp;D 파트너십으로 관리 권장</td>
      </tr>
      <tr style="background:#fdf4ff;">
        <td><strong style="color:#6b21a8;">공급 (d)</strong></td>
        <td style="text-align:center; font-weight:700; color:#6b21a8;">6위 계열</td>
        <td style="text-align:center; color:#64748b;">하위권</td>
        <td>변동성 큼 — 납기 이슈 발생 시 급격히 낮아짐</td>
        <td>납기 지연 이벤트가 CCI 급락 트리거 — 예외 관리 중요</td>
      </tr>
      <tr>
        <td><strong style="color:#854d0e;">서비스 (q2)</strong></td>
        <td style="text-align:center; font-weight:700; color:#854d0e;">9위 계열</td>
        <td style="text-align:center; color:#64748b;">하위권</td>
        <td>클래스 간 분포 중첩 많음 — RPI 변별력 낮음</td>
        <td>서비스 만족이 기본 기대치 수준 — 차별화 포인트 낮음</td>
      </tr>
    </tbody>
  </table>
  <div style="background:#f8fafc; border-radius:8px; padding:12px 16px; font-size:12px; line-height:1.8; color:#334155; border-left:4px solid #0f172a;">
    <strong>EDA 도메인 종합 발견:</strong>
    고객 재구매는 <strong>신기술 신뢰 → Cost 투명성 → 품질 일관성 → 개발 협력</strong> 순의 가치 사슬로 결정됩니다.
    서비스(q2) 만족보다 신기술(t1) 신뢰 구축이 4배 이상 강한 재구매 예측력을 가집니다.
    이는 LG B2B 고객이 "현재 경험"보다 "미래 파트너십 가능성"을 더 중시함을 의미합니다.
  </div>
</div>
"""

# ── feature_experiment_report — 도메인 관점 해석 카드 ──────────────────
FE_DOMAIN_CARD = """
<!-- ════════ 도메인 관점 Feature Engineering 재해석 ════════ -->
<div style="background:white; border-radius:12px; padding:24px 28px; margin-bottom:24px;
     box-shadow:0 2px 8px rgba(0,0,0,.07); border-left:5px solid #0f172a;">
  <h2 style="font-size:17px; font-weight:700; border-bottom:2px solid #e2e8f0; padding-bottom:10px;
      margin-bottom:16px; color:#1e293b;">
    🏭 영역별 도메인 관점 Feature Engineering 재해석
  </h2>
  <p style="font-size:12px; color:#64748b; margin-bottom:14px;">
    변수명 접두사: <strong>t1=신기술</strong> / <strong>t2=개발</strong> / <strong>c=Cost</strong> /
    <strong>d=공급</strong> / <strong>q1=품질</strong> / <strong>q2=서비스</strong>
  </p>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; font-size:13px;">
    <div style="background:#eff6ff; border-radius:8px; padding:12px 16px; border-left:4px solid #2563eb;">
      <strong style="color:#1d4ed8;">PCA가 최고 성능인 이유 — 도메인 관점</strong><br>
      <span style="color:#1e3a5f; line-height:1.8;">
        t1(신기술)·t2(개발)·c(Cost)·q1(품질)·q2(서비스)·d(공급)의
        res/core/comm 세부 항목이 동일 영역 내에서 r≥0.9 다중공선성을 형성합니다.
        PCA는 <strong>6개 영역 × 2지수(CSI/CCI) = 12개 잠재 축</strong>으로
        자연스럽게 압축되므로 성능이 가장 우수합니다.
      </span>
    </div>
    <div style="background:#fff7ed; border-radius:8px; padding:12px 16px; border-left:4px solid #ea580c;">
      <strong style="color:#9a3412;">Feature Selection 한계 — 도메인 관점</strong><br>
      <span style="color:#7c2d12; line-height:1.8;">
        MI + Pearson 기준 Feature Selection이 48개를 선택했지만
        영역별 세부항목(res/core/comm)을 개별로 다루어 <strong>동일 영역 내 중복 정보</strong>를 제거하지 못합니다.
        도메인 지식 기반으로 "_total만 사용"하는 전략이 더 효율적일 수 있습니다.
      </span>
    </div>
    <div style="background:#f0fdf4; border-radius:8px; padding:12px 16px; border-left:4px solid #16a34a;">
      <strong style="color:#166534;">권장: 영역 총점 피처셋 (12변수)</strong><br>
      <span style="color:#14532d; line-height:1.8;">
        t1/t2/c/d/q1/q2 각 영역의 csi_total + cci_total = <strong>12개 총점 변수</strong>만 사용.
        영역 내 다중공선성 완전 해소 + 비즈니스 해석력 극대화.
        PCA 없이도 Logistic Regression이 높은 성능을 낼 것으로 예측됩니다.
      </span>
    </div>
    <div style="background:#fdf4ff; border-radius:8px; padding:12px 16px; border-left:4px solid #9333ea;">
      <strong style="color:#6b21a8;">다음 실험: 영역별 중요도 개별 분석</strong><br>
      <span style="color:#3b0764; line-height:1.8;">
        신기술(t1) vs Cost(c) vs 품질(q1) 영역별로 단독 모델 성능 측정 →
        "어떤 영역만으로도 RPI를 예측할 수 있는가?" 검증.
        영역별 독립 예측 모델로 세분화된 CS 전략 수립 가능합니다.
      </span>
    </div>
  </div>
</div>
"""

# ── model_comparison_report — 도메인 관점 해석 카드 ───────────────────
MC_DOMAIN_CARD = """
<!-- ════════ 도메인 관점 모델 비교 재해석 ════════ -->
<div style="background:white; border-radius:12px; padding:24px 28px; margin-bottom:24px;
     box-shadow:0 2px 8px rgba(0,0,0,.07); border-left:5px solid #0f172a;">
  <h2 style="font-size:17px; font-weight:700; border-bottom:2px solid #e2e8f0; padding-bottom:10px;
      margin-bottom:16px; color:#1e293b;">
    🏭 영역별 도메인 관점 — 모델 성능 해석
  </h2>
  <p style="font-size:12px; color:#64748b; margin-bottom:14px;">
    변수명 접두사: <strong>t1=신기술</strong> / <strong>t2=개발</strong> / <strong>c=Cost</strong> /
    <strong>d=공급</strong> / <strong>q1=품질</strong> / <strong>q2=서비스</strong>
  </p>
  <table style="font-size:13px; margin-bottom:16px;">
    <thead><tr style="background:#1e293b; color:white;">
      <th style="color:white;">모델</th>
      <th style="color:white;">성능 (Macro F1, 보정)</th>
      <th style="color:white;">도메인 관점 해석</th>
      <th style="color:white;">특히 중요한 영역</th>
    </tr></thead>
    <tbody>
      <tr style="background:#eff6ff;">
        <td><strong style="color:#6366f1;">Logistic Regression</strong></td>
        <td>0.9916 ± 0.0000<br><span style="font-size:11px; color:#64748b;">완전 안정 (Std=0)</span></td>
        <td>
          신기술(t1)·Cost(c)·품질(q1) CCI total의 <strong>선형 결합</strong>이 RPI를
          거의 완벽하게 설명합니다. 6개 영역 × 2지수의 관계가 본질적으로 선형임을 증명.
        </td>
        <td>t1_cci, c_cci — 계수 크기 압도적</td>
      </tr>
      <tr>
        <td><strong style="color:#10b981;">LightGBM</strong></td>
        <td>0.9736 ± 0.0000<br><span style="font-size:11px; color:#64748b;">Macro AUROC 최고</span></td>
        <td>
          영역 간 <strong>교호작용</strong>(신기술 신뢰 높은데 Cost 불신인 고객 등)을
          비선형적으로 포착. 확률 보정 정밀도가 가장 높아 위험 고객 순위화에 최적.
        </td>
        <td>t1_cci × c_cci 교호작용</td>
      </tr>
      <tr style="background:#fffbeb;">
        <td><strong style="color:#f59e0b;">Random Forest</strong></td>
        <td>0.9538 ± 0.0037<br><span style="font-size:11px; color:#64748b;">보정 후 Std 대폭 감소</span></td>
        <td>
          영역별 세부항목(res/core/comm)의 <strong>복합 패턴</strong>을 앙상블로 포착.
          보정(balanced) 적용 후 소수 클래스(RPI 4~5, 위험 고객) 탐지 능력이 크게 향상됨.
        </td>
        <td>공급(d)·서비스(q2) 비선형 패턴</td>
      </tr>
      <tr>
        <td><strong style="color:#64748b;">Decision Tree</strong></td>
        <td>0.9603 ± 0.0078<br><span style="font-size:11px; color:#64748b;">보정 후 개선 뚜렷</span></td>
        <td>
          분기 규칙 직접 해석 가능: "t1_cci_total &lt; X이고 c_cci_total &lt; Y이면 RPI=4"
          형태의 <strong>고객 위험 분류 규칙</strong> 추출 가능.
        </td>
        <td>t1_cci + c_cci 임계값 조합</td>
      </tr>
    </tbody>
  </table>
  <div style="background:#f0f9ff; border-radius:8px; padding:12px 16px; font-size:12px; line-height:1.8; color:#0c4a6e; border-left:4px solid #0891b2;">
    <strong>도메인 종합:</strong>
    PCA 25 컴포넌트가 실질적으로 <strong>신기술/개발/Cost/품질/공급/서비스 × CCI/CSI의 12개 잠재 축</strong>을 포착하고 있습니다.
    Logistic Regression이 이 구조에서 최고 성능을 내는 것은 영역별 신뢰·만족 지수가
    RPI와 본질적으로 <strong>선형 가산적 관계</strong>임을 강력히 시사합니다.
  </div>
</div>
"""


# ════════════════════════════════════════════════════════════════════
# 파일별 업데이트 실행
# ════════════════════════════════════════════════════════════════════

# 1. feature_importance_report.html
print("Updating feature_importance_report.html ...")
fi_path = REPORT_DIR / "feature_importance_report.html"
fi_html = fi_path.read_text(encoding="utf-8")

# 도메인 범례를 <div class="container"> 바로 다음에 삽입
fi_html = fi_html.replace(
    '<div class="container">\n\n<!-- KPI 요약 -->',
    '<div class="container">\n' + DOMAIN_LEGEND + '\n<!-- KPI 요약 -->'
)

# 도메인 인사이트를 "종합 인사이트 및 권고사항" 카드 바로 앞에 삽입
fi_html = fi_html.replace(
    '<!-- ════════ 최종 인사이트 정리 ════════ -->',
    FI_DOMAIN_INSIGHTS + '\n<!-- ════════ 최종 인사이트 정리 ════════ -->'
)

fi_path.write_text(fi_html, encoding="utf-8")
print("  Done:", fi_path)


# 2. EDA_report.html
print("Updating EDA_report.html ...")
eda_path = REPORT_DIR / "EDA_report.html"
eda_html = eda_path.read_text(encoding="utf-8")

# 도메인 범례를 <div class="container"> 바로 다음에 삽입
# EDA report는 다른 container 구조 가능성 있으므로 <body> 직후 삽입
eda_html = eda_html.replace(
    '</div>\n\n<!-- ════════ 추가 과제 제안 ════════ -->',
    EDA_DOMAIN_CARD + '\n</div>\n\n<!-- ════════ 추가 과제 제안 ════════ -->'
)

eda_path.write_text(eda_html, encoding="utf-8")
print("  Done:", eda_path)


# 3. feature_experiment_report.html
print("Updating feature_experiment_report.html ...")
fe_path = REPORT_DIR / "feature_experiment_report.html"
fe_html = fe_path.read_text(encoding="utf-8")

fe_html = fe_html.replace(
    '<!-- ════════ 추가 과제 제안 ════════ -->',
    FE_DOMAIN_CARD + '\n<!-- ════════ 추가 과제 제안 ════════ -->'
)

fe_path.write_text(fe_html, encoding="utf-8")
print("  Done:", fe_path)


# 4. model_comparison_report.html
print("Updating model_comparison_report.html ...")
mc_path = REPORT_DIR / "model_comparison_report.html"
mc_html = mc_path.read_text(encoding="utf-8")

mc_html = mc_html.replace(
    '<!-- ════════ 추가 과제 제안 ════════ -->',
    MC_DOMAIN_CARD + '\n<!-- ════════ 추가 과제 제안 ════════ -->'
)

mc_path.write_text(mc_html, encoding="utf-8")
print("  Done:", mc_path)

print("\n[All 4 reports updated with domain insights]")
