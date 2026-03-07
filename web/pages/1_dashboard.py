"""
Dashboard — 리스크 게이지, 성질 기여도, 상태 진단, 일별 추이
"""

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "data" / "outputs"
THRESHOLD = 18.52

PROP_NAMES = {
    "Property_001": "거시경제 지표의 급변",
    "Property_013": "금융기관 간 신뢰 붕괴",
    "Property_023": "자산 가격의 급격한 변동",
    "Property_024": "외부 요인에 의한 시장 반응",
    "Property_030": "정부 개입과 시장 반응",
    "Property_049": "글로벌 금융 시스템의 상호 연결성",
    "Property_055": "수요와 공급의 불균형",
    "Property_068": "투자자 심리의 급격한 변화",
    "Property_070": "금융 규제와 정책 변화",
    "Property_078": "구조적 불안정성의 확산",
    "Property_110": "시장 참여자의 과도한 레버리지",
    "Property_113": "위기 확산의 비선형적 특성",
}

PROP_SHORT = {
    "Property_001": "거시경제",
    "Property_013": "신뢰붕괴",
    "Property_023": "자산변동",
    "Property_024": "외부충격",
    "Property_030": "정부개입",
    "Property_049": "글로벌연결",
    "Property_055": "수급불균형",
    "Property_068": "투자심리",
    "Property_070": "금융규제",
    "Property_078": "구조불안",
    "Property_110": "레버리지",
    "Property_113": "위기확산",
}

PROP_ORDER = [
    "Property_078", "Property_055", "Property_070", "Property_024",
    "Property_013", "Property_049", "Property_001", "Property_023",
    "Property_113", "Property_068", "Property_030", "Property_110",
]

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# ── 데이터 로드 ──
daily_csv = OUTPUTS / "daily_scores.csv"
if not daily_csv.exists():
    st.warning("daily_scores.csv가 없습니다. 파이프라인을 먼저 실행하세요.")
    st.stop()

df = pd.read_csv(daily_csv)
if len(df) == 0:
    st.warning("스코어 데이터가 비어 있습니다.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# details 파싱
def parse_all_details(dataframe):
    rows = []
    for _, row in dataframe.iterrows():
        details = json.loads(row["details_json"]) if isinstance(row["details_json"], str) else {}
        entry = {"date": row["date"], "risk_score": row["risk_score"], "status": row["status"]}
        for pid in PROP_ORDER:
            d = details.get(pid, {})
            entry[f"{pid}_ratio"] = d.get("ratio", 0.0)
            entry[f"{pid}_contrib"] = d.get("contribution", 0.0)
        rows.append(entry)
    return pd.DataFrame(rows)

detail_df = parse_all_details(df)

latest = df.iloc[-1]
score = latest["risk_score"]
status = latest["status"]
date_str = latest["date"].strftime("%Y-%m-%d")
details = json.loads(latest["details_json"]) if isinstance(latest["details_json"], str) else {}

delta = round(score - df.iloc[-2]["risk_score"], 2) if len(df) >= 2 else None

# WARNING 연속 일수
warn_streak = 0
for i in range(len(df) - 1, -1, -1):
    if df.iloc[i]["status"] == "WARNING":
        warn_streak += 1
    else:
        break

# 고점
peak_idx = df["risk_score"].idxmax()
peak = df.loc[peak_idx]

active_count = sum(1 for p in details.values() if p.get("ratio", 0) > 1.0)
extreme_count = sum(1 for p in details.values() if p.get("ratio", 0) > 1.5)

# ════════════════════════════════════════
# 헤더: 현재 상태 요약
# ════════════════════════════════════════
st.header("BlackEvent Dashboard")

cols = st.columns([1.5, 1, 1, 1, 1, 1])
with cols[0]:
    color = "🔴" if score >= 50 else "🟡" if score >= THRESHOLD else "🟢"
    st.metric(
        f"{color} Risk Score",
        f"{score:.1f}",
        delta=f"{delta:+.1f}" if delta is not None else None,
        delta_color="inverse",
    )
with cols[1]:
    st.metric("Status", status)
with cols[2]:
    st.metric("WARNING 연속", f"{warn_streak}일" if warn_streak > 0 else "-")
with cols[3]:
    st.metric("Active", f"{active_count}/12")
with cols[4]:
    st.metric("Extreme (>1.5x)", f"{extreme_count}/12")
with cols[5]:
    st.metric("Peak", f"{peak['risk_score']:.1f} ({peak['date'].strftime('%m/%d')})")

if status == "WARNING":
    if score >= 50:
        st.error(f"**EXTREME WARNING** — 스코어 {score:.1f}, 단기 내 고강도 이벤트 가능성")
    else:
        st.warning(f"**WARNING** — 스코어 {score:.1f} > 임계값 {THRESHOLD}")

st.divider()

# ════════════════════════════════════════
# 탭 구성
# ════════════════════════════════════════
tab_overview, tab_heatmap, tab_peak, tab_decomp, tab_compare, tab_diagnosis = st.tabs([
    "Overview", "Property Heatmap", "Peak Signal", "Score 분해", "위기 비교", "상태 진단",
])

# ════════════════════════════════════════
# Tab 1: Overview
# ════════════════════════════════════════
with tab_overview:
    # ── 게이지 + 기여도 ──
    col_gauge, col_bar = st.columns([1, 2])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": THRESHOLD, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "darkred" if score >= THRESHOLD else "darkgreen"},
                "steps": [
                    {"range": [0, THRESHOLD], "color": "#e8f5e9"},
                    {"range": [THRESHOLD, 50], "color": "#fff3e0"},
                    {"range": [50, 100], "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": THRESHOLD,
                },
            },
            title={"text": f"Risk Score ({date_str})"},
        ))
        fig_gauge.update_layout(height=350, margin=dict(t=60, b=20, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_bar:
        if details:
            props = []
            for pid, vals in details.items():
                props.append({
                    "pid": pid, "name": PROP_SHORT.get(pid, pid),
                    "ratio": vals.get("ratio", 0), "contribution": vals.get("contribution", 0),
                })
            prop_df = pd.DataFrame(props).sort_values("contribution", ascending=True)

            colors = [
                "#ef5350" if r >= 1.5 else "#ffa726" if r >= 1.0 else "#66bb6a"
                for r in prop_df["ratio"]
            ]
            fig_bar = go.Figure(go.Bar(
                x=prop_df["contribution"],
                y=[f"{row['name']} ({row['ratio']:.2f}x)" for _, row in prop_df.iterrows()],
                orientation="h",
                marker_color=colors,
                text=[f"{c:.4f}" for c in prop_df["contribution"]],
                textposition="outside",
                hovertemplate="%{y}<br>Contribution: %{x:.4f}<extra></extra>",
            ))
            fig_bar.update_layout(
                title="성질별 기여도 & 빈도비",
                xaxis_title="Contribution",
                height=350,
                margin=dict(l=150, r=60, t=40, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── 스코어 타임라인 ──
    st.subheader("Risk Score Timeline")

    fig_line = go.Figure()

    # WARNING 배경
    for _, row in df[df["status"] == "WARNING"].iterrows():
        fig_line.add_vrect(
            x0=row["date"] - pd.Timedelta(hours=12),
            x1=row["date"] + pd.Timedelta(hours=12),
            fillcolor="rgba(255,82,82,0.08)", layer="below", line_width=0,
        )

    # 스코어 라인 — 색상 구간별
    for i in range(len(df) - 1):
        s = df.iloc[i]["risk_score"]
        color = "#ef5350" if s >= 50 else "#ffa726" if s >= THRESHOLD else "#66bb6a"
        fig_line.add_trace(go.Scatter(
            x=[df.iloc[i]["date"], df.iloc[i+1]["date"]],
            y=[df.iloc[i]["risk_score"], df.iloc[i+1]["risk_score"]],
            mode="lines", line=dict(color=color, width=3),
            showlegend=False, hoverinfo="skip",
        ))

    # 포인트 + 호버
    marker_colors = [
        "#ef5350" if s >= 50 else "#ffa726" if s >= THRESHOLD else "#66bb6a"
        for s in df["risk_score"]
    ]
    fig_line.add_trace(go.Scatter(
        x=df["date"], y=df["risk_score"],
        mode="markers", marker=dict(size=8, color=marker_colors, line=dict(width=1, color="white")),
        name="Risk Score",
        customdata=np.column_stack([df["status"], df["n_articles"]]),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.1f}<br>Status: %{customdata[0]}<br>Articles: %{customdata[1]:,}<extra></extra>",
    ))

    # Threshold + Peak
    fig_line.add_hline(y=THRESHOLD, line_dash="dash", line_color="rgba(255,152,0,0.5)",
                       annotation_text=f"WARNING ({THRESHOLD})")
    fig_line.add_annotation(
        x=peak["date"], y=peak["risk_score"],
        text=f"Peak {peak['risk_score']:.1f}", showarrow=True,
        arrowhead=2, arrowcolor="#ef5350", font=dict(color="#ef5350", size=12),
        ax=0, ay=-30,
    )

    fig_line.update_layout(
        height=400, hovermode="x unified",
        margin=dict(t=30, b=40),
        xaxis_title="", yaxis_title="Risk Score",
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ════════════════════════════════════════
# Tab 2: Property Heatmap
# ════════════════════════════════════════
with tab_heatmap:
    st.subheader("Property Activation Heatmap")
    st.caption("빈도비(ratio): 1.0 = 평상시 수준, 1.5+ = 극강 신호")

    # 날짜 범위 선택
    warn_df = detail_df[detail_df["risk_score"] >= THRESHOLD * 0.5]
    use_df = warn_df if len(warn_df) >= 3 else detail_df.tail(10)

    n_dates = len(use_df)
    date_labels = [d.strftime("%m/%d") for d in use_df["date"]]
    score_labels = [f"{s:.0f}" for s in use_df["risk_score"]]

    # ratio 행렬
    z_data = []
    hover_data = []
    for pid in PROP_ORDER:
        row_z = use_df[f"{pid}_ratio"].values.tolist()
        row_hover = [
            f"{PROP_NAMES[pid]}<br>Ratio: {r:.3f}<br>Score: {s:.1f}"
            for r, s in zip(row_z, use_df["risk_score"])
        ]
        z_data.append(row_z)
        hover_data.append(row_hover)

    fig_heat = go.Figure(go.Heatmap(
        z=z_data,
        x=date_labels,
        y=[PROP_SHORT[p] for p in PROP_ORDER],
        colorscale=[
            [0.0, "#1a1a2e"], [0.3, "#2d3748"], [0.5, "#4a5568"],
            [0.6, "#d69e2e"], [0.75, "#dd6b20"], [1.0, "#e53e3e"],
        ],
        zmin=0, zmax=2.0,
        text=[[f"{v:.2f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        customdata=hover_data,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="Ratio", tickvals=[0, 0.5, 1.0, 1.5, 2.0]),
    ))

    # 하단에 스코어 표시
    fig_heat.add_trace(go.Scatter(
        x=date_labels, y=["Score"] * n_dates,
        mode="text", text=score_labels,
        textfont=dict(
            size=14,
            color=["#ef5350" if s >= 50 else "#ffa726" if s >= THRESHOLD else "#66bb6a"
                   for s in use_df["risk_score"]],
        ),
        showlegend=False, hoverinfo="skip",
    ))

    fig_heat.update_layout(
        height=500,
        margin=dict(l=100, r=20, t=30, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # 선택한 날짜의 상세
    selected_date = st.select_slider(
        "날짜 선택 (상세 보기)",
        options=use_df["date"].tolist(),
        value=use_df["date"].iloc[-1],
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )
    sel_row = use_df[use_df["date"] == selected_date].iloc[0]

    scol1, scol2 = st.columns([2, 1])
    with scol1:
        # 레이더 차트
        vals = [sel_row[f"{p}_ratio"] for p in PROP_ORDER]
        names = [PROP_SHORT[p] for p in PROP_ORDER]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=names + [names[0]],
            fill="toself", fillcolor="rgba(239,83,80,0.15)",
            line=dict(color="#ef5350", width=2),
            name=selected_date.strftime("%m/%d"),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[1.0] * (len(PROP_ORDER) + 1), theta=names + [names[0]],
            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
            name="Baseline (1.0)", fill=None,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, max(vals) * 1.15])),
            height=400, margin=dict(t=30, b=30),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with scol2:
        st.markdown(f"**{selected_date.strftime('%Y-%m-%d')}**")
        st.metric("Score", f"{sel_row['risk_score']:.1f}")
        st.markdown("---")
        for pid in PROP_ORDER:
            r = sel_row[f"{pid}_ratio"]
            icon = "🔴" if r >= 1.5 else "🟡" if r >= 1.0 else "⚪"
            st.markdown(f"{icon} **{PROP_SHORT[pid]}**: {r:.3f}x")

# ════════════════════════════════════════
# Tab 3: Peak Signal
# ════════════════════════════════════════
with tab_peak:
    st.subheader("Peak = Last Exit Signal")
    st.markdown("""
    > 스코어 고점의 높이가 후속 충격의 강도와 시급성을 예고한다.
    > 고점에서 하락하는 구간은 안전 신호가 아니라 **트리거 대기 구간**이다.
    """)

    # WARNING 기간 포커스
    warn_rows = df[df["status"] == "WARNING"]
    if len(warn_rows) > 0:
        focus_start = warn_rows.iloc[0]["date"] - pd.Timedelta(days=7)
        focus_df = df[df["date"] >= focus_start].copy().reset_index(drop=True)
    else:
        focus_df = df.tail(10).copy().reset_index(drop=True)

    f_peak_idx = focus_df["risk_score"].idxmax()
    f_peak = focus_df.loc[f_peak_idx]

    fig_peak = go.Figure()

    # 상승 구간 (고점 전)
    pre = focus_df.loc[:f_peak_idx]
    fig_peak.add_trace(go.Scatter(
        x=pre["date"], y=pre["risk_score"],
        mode="lines+markers", name="상승 (취약성 축적)",
        line=dict(color="#ef5350", width=3),
        marker=dict(size=8, color="#ef5350"),
    ))

    # 하락 구간 (고점 후)
    post = focus_df.loc[f_peak_idx:]
    if len(post) > 1:
        fig_peak.add_trace(go.Scatter(
            x=post["date"], y=post["risk_score"],
            mode="lines+markers", name="하락 (뉴스 피로, NOT 안전)",
            line=dict(color="#ffa726", width=3, dash="dash"),
            marker=dict(size=8, color="#ffa726"),
            fill="tozeroy", fillcolor="rgba(255,167,38,0.08)",
        ))

    # 고점 마커
    fig_peak.add_trace(go.Scatter(
        x=[f_peak["date"]], y=[f_peak["risk_score"]],
        mode="markers+text", name="PEAK (Last Exit)",
        marker=dict(size=18, color="#ef5350", symbol="triangle-down",
                    line=dict(width=2, color="white")),
        text=[f"PEAK {f_peak['risk_score']:.1f}"],
        textposition="top center", textfont=dict(size=14, color="#ef5350"),
    ))

    fig_peak.add_hline(y=THRESHOLD, line_dash="dash", line_color="rgba(255,152,0,0.4)")

    fig_peak.update_layout(
        height=450, hovermode="x unified",
        xaxis_title="", yaxis_title="Risk Score",
        margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_peak, use_container_width=True)

    # 역대 비교
    st.markdown("### 역대 위기 패턴 비교")
    compare_data = [
        {"위기": "2026 이란", "사전 고점": 63.7, "고점→충격": "20일", "최대 낙폭": "-18.4%", "유형": "복합 시스템"},
        {"위기": "2011 유럽", "사전 고점": 37.3, "고점→충격": "124일", "최대 낙폭": "-17.1%", "유형": "구조적"},
        {"위기": "2025 하반기", "사전 고점": 37.1, "고점→충격": "116일", "최대 낙폭": "-7.6%", "유형": "복합"},
        {"위기": "2020 COVID", "사전 고점": 12.8, "고점→충격": "(미감지)", "최대 낙폭": "-30.1%", "유형": "신종"},
        {"위기": "2024-08 급락", "사전 고점": 11.4, "고점→충격": "(미감지)", "최대 낙폭": "-12.1%", "유형": "기술적"},
    ]
    st.dataframe(pd.DataFrame(compare_data), hide_index=True, use_container_width=True)

    st.info("""
    **패턴**: 고점 ~37 → 4개월 여유, -7~17% 낙폭 | 고점 ~64 → 3주 여유, -18%+ 낙폭
    **핵심**: 스코어 하락 = 뉴스 피로, 구조적 취약성은 그대로. 하락 구간이 가장 위험.
    """)

# ════════════════════════════════════════
# Tab 4: Score 분해
# ════════════════════════════════════════
with tab_decomp:
    st.subheader("스코어 변화 분해")
    st.caption("두 날짜 사이에 어떤 성질이 스코어를 올렸고/내렸는지")

    if len(df) >= 2:
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            date_from = st.selectbox("From", df["date"].tolist(), index=max(0, len(df)-2),
                                     format_func=lambda d: d.strftime("%Y-%m-%d"))
        with dcol2:
            date_to = st.selectbox("To", df["date"].tolist(), index=len(df)-1,
                                   format_func=lambda d: d.strftime("%Y-%m-%d"))

        row_from = df[df["date"] == date_from].iloc[0]
        row_to = df[df["date"] == date_to].iloc[0]
        det_from = json.loads(row_from["details_json"]) if isinstance(row_from["details_json"], str) else {}
        det_to = json.loads(row_to["details_json"]) if isinstance(row_to["details_json"], str) else {}

        score_delta = row_to["risk_score"] - row_from["risk_score"]
        st.metric(
            "Score 변화",
            f"{row_to['risk_score']:.1f}",
            delta=f"{score_delta:+.1f} (from {row_from['risk_score']:.1f})",
            delta_color="inverse",
        )

        # 성질별 변화
        changes = []
        for pid in PROP_ORDER:
            r_from = det_from.get(pid, {}).get("ratio", 0)
            r_to = det_to.get(pid, {}).get("ratio", 0)
            c_from = det_from.get(pid, {}).get("contribution", 0)
            c_to = det_to.get(pid, {}).get("contribution", 0)
            changes.append({
                "name": PROP_SHORT[pid],
                "ratio_from": r_from, "ratio_to": r_to,
                "ratio_delta": r_to - r_from,
                "contrib_from": c_from, "contrib_to": c_to,
                "contrib_delta": c_to - c_from,
            })
        change_df = pd.DataFrame(changes).sort_values("contrib_delta")

        # 워터폴 차트
        fig_waterfall = go.Figure(go.Waterfall(
            x=[row["name"] for _, row in change_df.iterrows()],
            y=[row["contrib_delta"] for _, row in change_df.iterrows()],
            measure=["relative"] * len(change_df),
            increasing=dict(marker_color="#ef5350"),
            decreasing=dict(marker_color="#66bb6a"),
            text=[f"{d:+.4f}" for d in change_df["contrib_delta"]],
            textposition="outside",
            textfont=dict(size=10),
        ))
        fig_waterfall.update_layout(
            title=f"Contribution 변화 ({date_from.strftime('%m/%d')} → {date_to.strftime('%m/%d')})",
            yaxis_title="Δ Contribution",
            height=400,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Ratio 변화 비교
        fig_ratio = make_subplots(rows=1, cols=1)
        fig_ratio.add_trace(go.Bar(
            x=[PROP_SHORT[p] for p in PROP_ORDER],
            y=[det_from.get(p, {}).get("ratio", 0) for p in PROP_ORDER],
            name=date_from.strftime("%m/%d"),
            marker_color="rgba(66,165,245,0.6)",
        ))
        fig_ratio.add_trace(go.Bar(
            x=[PROP_SHORT[p] for p in PROP_ORDER],
            y=[det_to.get(p, {}).get("ratio", 0) for p in PROP_ORDER],
            name=date_to.strftime("%m/%d"),
            marker_color="rgba(239,83,80,0.6)",
        ))
        fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                            annotation_text="Baseline")
        fig_ratio.update_layout(
            barmode="group", height=350, margin=dict(t=30, b=40),
            yaxis_title="Ratio", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_ratio, use_container_width=True)
    else:
        st.info("비교할 데이터가 부족합니다.")

# ════════════════════════════════════════
# Tab 5: 위기 비교
# ════════════════════════════════════════
with tab_compare:
    st.subheader("현재 vs 과거 위기 비교")

    risk_csv = OUTPUTS / "risk_scores.csv"
    if risk_csv.exists() and details:
        rs = pd.read_csv(risk_csv)
        rs["date"] = pd.to_datetime(rs["date"])

        # 주요 위기 클러스터의 대표 날짜
        crisis_dates = {
            "2008 리먼 (10/06)": "2008-10-06",
            "2008 리먼 최고 (11/06)": "2008-11-06",
            "2011 유럽 (08/19)": "2011-08-19",
            "2020 COVID (02/24)": "2020-02-24",
            "2020 COVID (04/01)": "2020-04-01",
            "2024-08 급락": "2024-08-05",
        }

        selected_crisis = st.selectbox("비교할 위기", list(crisis_dates.keys()))
        crisis_date = crisis_dates[selected_crisis]

        crisis_row = rs[rs["date"] == crisis_date]
        if len(crisis_row) > 0:
            crisis_row = crisis_row.iloc[0]

            # 현재 vs 과거 레이더 비교
            # risk_scores에는 details_json이 없으므로 property별 ratio 계산이 어려움
            # 대신 be_profiles 사용
            be_profiles_path = OUTPUTS / "be_profiles.json"
            if be_profiles_path.exists():
                with open(be_profiles_path, encoding="utf-8") as f:
                    be_data = json.load(f)

                # 가장 가까운 이벤트 찾기
                event_id = str(crisis_row.get("event_id", ""))
                if event_id in be_data["profiles"]:
                    profile = be_data["profiles"][event_id]
                    offsets = profile["offsets"]
                    # D-0 우선
                    ref_ratios = offsets.get("D-0", offsets.get(list(offsets.keys())[0], {}))

                    cur_vals = [details.get(p, {}).get("ratio", 0) for p in PROP_ORDER]
                    ref_vals = [ref_ratios.get(p, 0) for p in PROP_ORDER]
                    names = [PROP_SHORT[p] for p in PROP_ORDER]

                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Scatterpolar(
                        r=cur_vals + [cur_vals[0]], theta=names + [names[0]],
                        fill="toself", name=f"현재 ({date_str})",
                        fillcolor="rgba(21,101,192,0.2)",
                        line=dict(color="#42a5f5", width=2),
                    ))
                    fig_cmp.add_trace(go.Scatterpolar(
                        r=ref_vals + [ref_vals[0]], theta=names + [names[0]],
                        fill="toself", name=f"{selected_crisis}",
                        fillcolor="rgba(239,83,80,0.2)",
                        line=dict(color="#ef5350", width=2),
                    ))
                    r_max = max(max(cur_vals), max(ref_vals)) * 1.15
                    fig_cmp.update_layout(
                        polar=dict(radialaxis=dict(range=[0, r_max])),
                        height=500, showlegend=True,
                        margin=dict(t=40, b=40, l=80, r=80),
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    # 수치 비교
                    comp_rows = []
                    for pid in PROP_ORDER:
                        comp_rows.append({
                            "성질": PROP_SHORT[pid],
                            f"현재": details.get(pid, {}).get("ratio", 0),
                            f"{selected_crisis}": ref_ratios.get(pid, 0),
                            "차이": details.get(pid, {}).get("ratio", 0) - ref_ratios.get(pid, 0),
                        })
                    comp_df = pd.DataFrame(comp_rows)
                    st.dataframe(
                        comp_df, hide_index=True, use_container_width=True,
                        column_config={
                            "현재": st.column_config.NumberColumn(format="%.3f"),
                            f"{selected_crisis}": st.column_config.NumberColumn(format="%.3f"),
                            "차이": st.column_config.NumberColumn(format="%+.3f"),
                        },
                    )
                else:
                    st.info(f"이벤트 {event_id}의 프로필이 없습니다.")
            else:
                st.info("be_profiles.json이 없습니다.")
        else:
            st.info(f"{crisis_date}의 스코어 데이터가 없습니다.")
    else:
        st.info("risk_scores.csv 또는 현재 성질 데이터가 없습니다.")

# ════════════════════════════════════════
# Tab 6: 상태 진단
# ════════════════════════════════════════
with tab_diagnosis:
    st.subheader("현재 상태 진단")

    _be_profiles_path = OUTPUTS / "be_profiles.json"
    _risk_scores_path = OUTPUTS / "risk_scores.csv"

    if not _be_profiles_path.exists():
        st.info("be_profiles.json이 없습니다.")
    elif not details:
        st.info("성질 상세 데이터가 없습니다.")
    else:
        from src.diagnosis.profile_matcher import find_similar_events, compute_score_percentile

        current_ratios = {pid: vals.get("ratio", 0) for pid, vals in details.items()}
        matches = find_similar_events(current_ratios, _be_profiles_path, top_n=20)

        _seen = set()
        unique_matches = []
        for m in matches:
            if m["event_id"] not in _seen:
                _seen.add(m["event_id"])
                unique_matches.append(m)

        if unique_matches:
            top = unique_matches[0]
            _offset_desc = {"D-0": "당일", "D-7": "7일 전", "D-14": "14일 전", "D-21": "21일 전"}

            st.info(
                f"현재 뉴스 환경은 **{top['first_shock_date']}** 폭락 "
                f"**{_offset_desc.get(top['offset'], top['offset'])}** 상황과 가장 유사합니다.\n\n"
                f"코사인 유사도: **{top['similarity']:.1%}** | "
                f"당시 결과: **{top['max_cumulative_drop']}%** 하락, "
                f"{top['duration_days']}일간 {top['shock_count']}회 충격"
            )

            if _risk_scores_path.exists():
                pct = compute_score_percentile(score, _risk_scores_path)
                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    st.metric("전체 퍼센타일", f"상위 {100 - pct['overall_percentile']:.1f}%")
                with pcol2:
                    st.metric("BlackEvent 평균", f"{pct['be_mean']:.1f}")
                with pcol3:
                    st.metric("대조군 평균", f"{pct['ctrl_mean']:.1f}")

            # 유사 사례 테이블
            st.markdown("**유사 과거 사례 (상위 10건)**")
            table_rows = []
            for m in unique_matches[:10]:
                table_rows.append({
                    "사건": f"BE_{m['event_id']}",
                    "날짜": m["first_shock_date"],
                    "매칭": _offset_desc.get(m["offset"], m["offset"]),
                    "유사도": m["similarity"],
                    "최대 하락": f"{m['max_cumulative_drop']}%",
                    "기간(일)": m["duration_days"],
                    "충격": m["shock_count"],
                })
            st.dataframe(
                pd.DataFrame(table_rows), hide_index=True, use_container_width=True,
                column_config={
                    "유사도": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                },
            )

            # 모멘텀
            if len(df) >= 2 and delta is not None:
                st.markdown("---")
                prev_row = df.iloc[-2]
                prev_det = json.loads(prev_row["details_json"]) if isinstance(prev_row["details_json"], str) else {}

                rising, falling = [], []
                for pid in details:
                    cur_r = details[pid].get("ratio", 0)
                    prev_r = prev_det.get(pid, {}).get("ratio", 0)
                    name = PROP_SHORT.get(pid, pid)
                    d = cur_r - prev_r
                    if d > 0.01:
                        rising.append((name, d))
                    elif d < -0.01:
                        falling.append((name, d))

                rising.sort(key=lambda x: -x[1])
                falling.sort(key=lambda x: x[1])

                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.markdown(f"**📈 상승 ({len(rising)}개)**")
                    for name, d in rising:
                        st.markdown(f"- {name} (+{d:.3f})")
                with mcol2:
                    st.markdown(f"**📉 하락 ({len(falling)}개)**")
                    for name, d in falling:
                        st.markdown(f"- {name} ({d:.3f})")

# ── 하단 자동 갱신 안내 ──
st.divider()
st.caption("💡 파이프라인 실행 후 페이지를 새로고침하면 최신 데이터가 반영됩니다.")
