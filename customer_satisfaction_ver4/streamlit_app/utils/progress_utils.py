"""모든 페이지 상단에 표시되는 진행 단계(Stepper) 인디케이터.

각 페이지 최상단에서 `render_step_progress(current_step=N)`을 호출하면
7단계 중 현재 위치/완료/미진행 상태를 일관된 디자인으로 보여줍니다.

세션 상태(`st.session_state`)를 자동으로 검사해 단계별 완료 여부를 판정합니다.
"""
from __future__ import annotations

import streamlit as st

# (step_number, short_label, icon, completion_state_key)
_STEPS = [
    (1, "Upload",      "📁", "df"),
    (2, "EDA",         "🔍", "df"),               # df가 있으면 EDA 가능
    (3, "FE",          "🛠", "fe_X_train"),
    (4, "Training",    "🤖", "training_result"),
    (5, "Compare",     "📈", "training_result"),
    (6, "Importance",  "🎯", "training_result"),
    (7, "Report",      "📑", "training_result"),
]


def _is_complete(state_key: str) -> bool:
    return state_key in st.session_state


def render_step_progress(current_step: int):
    """페이지 상단에 7단계 stepper 표시.

    - 완료된 단계 (현재보다 작은 step): 초록 ✓
    - 현재 단계: 보라색 강조 + 펄스
    - 미진행 단계: 회색
    """
    n_steps = len(_STEPS)
    nodes_html: list[str] = []
    lines_html: list[str] = []

    for idx, (num, label, icon, state_key) in enumerate(_STEPS):
        is_current = num == current_step
        is_done    = num < current_step or (
            num != current_step and _is_complete(state_key) and num <= max(current_step, 7)
        )
        # 우선순위: current > done > pending
        if is_current:
            bg, fg, border, ring = "#4f46e5", "white", "#4f46e5", "0 0 0 4px rgba(79,70,229,.15)"
            label_color = "#4f46e5"
            label_weight = "800"
            num_text = f"{icon}"
        elif is_done:
            bg, fg, border, ring = "#16a34a", "white", "#16a34a", "none"
            label_color = "#15803d"
            label_weight = "700"
            num_text = "✓"
        else:
            bg, fg, border, ring = "#f1f5f9", "#94a3b8", "#cbd5e1", "none"
            label_color = "#64748b"
            label_weight = "500"
            num_text = f"{num}"

        # 노드(원)
        nodes_html.append(
            f"""<div style="display:flex;flex-direction:column;align-items:center;
                            min-width:0;flex:1;text-align:center;">
                  <div style="width:40px;height:40px;border-radius:50%;
                              background:{bg};color:{fg};
                              border:2px solid {border};box-shadow:{ring};
                              display:flex;align-items:center;justify-content:center;
                              font-size:16px;font-weight:800;
                              transition:all .25s;">
                    {num_text}
                  </div>
                  <div style="margin-top:6px;font-size:11px;font-weight:{label_weight};
                              color:{label_color};letter-spacing:.3px;
                              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                              max-width:90px;">
                    {num}. {label}
                  </div>
                </div>"""
        )

        # 연결선 (마지막 노드 뒤에는 안 그림)
        if idx < n_steps - 1:
            next_num = _STEPS[idx + 1][0]
            line_done = num < current_step or (
                _is_complete(state_key) and _is_complete(_STEPS[idx + 1][3])
            )
            line_color = "#16a34a" if line_done else "#cbd5e1"
            lines_html.append(
                f"""<div style="flex:0 0 auto;height:2px;width:24px;
                                 background:{line_color};
                                 margin:20px 4px 0;border-radius:2px;
                                 transition:background .25s;"></div>"""
            )

    # 노드 + 연결선 인터리빙
    pieces: list[str] = []
    for i, node in enumerate(nodes_html):
        pieces.append(node)
        if i < len(lines_html):
            pieces.append(lines_html[i])

    cur = _STEPS[current_step - 1] if 1 <= current_step <= n_steps else None
    cur_label = (
        f"현재 단계: <b>{cur[0]}. {cur[1]}</b> {cur[2]}" if cur else ""
    )

    progress_pct = int((current_step / n_steps) * 100)

    st.markdown(
        f"""<div style="background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 100%);
                       border:1px solid #e2e8f0;border-radius:12px;
                       padding:14px 20px 10px;margin-bottom:16px;
                       box-shadow:0 1px 4px rgba(0,0,0,.04);">
          <div style="display:flex;justify-content:space-between;
                       align-items:center;margin-bottom:6px;">
            <span style="font-size:11px;font-weight:700;letter-spacing:1px;
                          color:#64748b;text-transform:uppercase;">
              진행 단계 (Workflow Progress)
            </span>
            <span style="font-size:11px;color:#475569;">
              {cur_label} &nbsp;·&nbsp; <b>{progress_pct}%</b> 진행
            </span>
          </div>
          <div style="display:flex;align-items:flex-start;justify-content:space-between;
                      gap:0;overflow-x:auto;padding-bottom:4px;">
            {''.join(pieces)}
          </div>
          <div style="height:4px;background:#e2e8f0;border-radius:2px;
                      margin-top:10px;overflow:hidden;">
            <div style="width:{progress_pct}%;height:100%;
                        background:linear-gradient(90deg,#4f46e5,#9333ea);
                        border-radius:2px;transition:width .3s;"></div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_home_progress():
    """app.py 홈 페이지 — 동일한 디자인 톤으로 진행 상태를 보여줌.

    홈에서는 '현재 단계'가 없으므로, 완료된 단계 수를 기반으로
    다음 추천 단계를 자동 산출하여 강조합니다.
    """
    # 다음 단계 자동 추정
    if "training_result" in st.session_state:
        next_step = 7
    elif "fe_X_train" in st.session_state:
        next_step = 4
    elif "df" in st.session_state:
        next_step = 3
    else:
        next_step = 1

    render_step_progress(current_step=next_step)
