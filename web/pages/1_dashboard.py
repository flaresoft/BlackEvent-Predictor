"""
Dashboard — 리스크 게이지, 성질 기여도, 상태 진단, 시뮬레이션
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

# ── 스코어링 모델 파라미터 (파이프라인과 동일 공식) ──
def _load_model_params():
    sig_path = OUTPUTS / "significant_properties.csv"
    rs_path = OUTPUTS / "risk_scores.csv"
    if not sig_path.exists() or not rs_path.exists():
        return None
    sig = pd.read_csv(sig_path)
    raw_w = -np.log10(sig["p_adjusted"].clip(lower=1e-300))
    norm_w = raw_w / raw_w.max()
    weights = dict(zip(sig["property_id"], norm_w))
    ctrl_means = dict(zip(sig["property_id"], sig["ctrl_freq_mean"]))
    rs = pd.read_csv(rs_path)
    empirical_max = rs.loc[rs["actual_label"] == 1, "model_b_raw_score"].max()
    return {"weights": weights, "ctrl_means": ctrl_means, "empirical_max": empirical_max}


def simulate_score(ratios: dict, params: dict) -> dict:
    """ratio dict로부터 스코어를 계산한다 (파이프라인 공식 재현)."""
    raw = 0.0
    details = {}
    for pid, w in params["weights"].items():
        r = ratios.get(pid, 0.0)
        excess = max(r - 1.0, 0.0)
        contrib = w * excess
        raw += contrib
        details[pid] = {"ratio": r, "contribution": round(contrib, 4), "weight": round(w, 4)}
    em = params["empirical_max"]
    risk_score = min(raw / em * 100, 100.0) if em > 0 else 0.0
    return {"risk_score": round(risk_score, 2), "raw_score": round(raw, 4), "details": details}


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

MODEL_PARAMS = _load_model_params()


def parse_details_for_date(row):
    return json.loads(row["details_json"]) if isinstance(row["details_json"], str) else {}


def parse_all_details(dataframe):
    rows = []
    for _, row in dataframe.iterrows():
        det = parse_details_for_date(row)
        entry = {"date": row["date"], "risk_score": row["risk_score"], "status": row["status"]}
        for pid in PROP_ORDER:
            d = det.get(pid, {})
            entry[f"{pid}_ratio"] = d.get("ratio", 0.0)
            entry[f"{pid}_contrib"] = d.get("contribution", 0.0)
        rows.append(entry)
    return pd.DataFrame(rows)


detail_df = parse_all_details(df)

latest = df.iloc[-1]
score = latest["risk_score"]
status = latest["status"]
date_str = latest["date"].strftime("%Y-%m-%d")
details = parse_details_for_date(latest)

delta = round(score - df.iloc[-2]["risk_score"], 2) if len(df) >= 2 else None

warn_streak = 0
for i in range(len(df) - 1, -1, -1):
    if df.iloc[i]["status"] == "WARNING":
        warn_streak += 1
    else:
        break

peak_idx = df["risk_score"].idxmax()
peak = df.loc[peak_idx]

active_count = sum(1 for p in details.values() if p.get("ratio", 0) > 1.0)
extreme_count = sum(1 for p in details.values() if p.get("ratio", 0) > 1.5)

# ════════════════════════════════════════
# 헤더
# ════════════════════════════════════════
st.header("BlackEvent Dashboard")

cols = st.columns([1.5, 1, 1, 1, 1, 1])
with cols[0]:
    color = "🔴" if score >= 50 else "🟡" if score >= THRESHOLD else "🟢"
    st.metric(f"{color} Risk Score", f"{score:.1f}",
              delta=f"{delta:+.1f}" if delta is not None else None, delta_color="inverse")
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
# 탭
# ════════════════════════════════════════
tab_names = [
    "Overview", "Property Heatmap", "Peak Signal", "Score 분해",
    "시뮬레이터", "타임머신", "민감도",
    "위기 비교", "상태 진단",
]
(tab_overview, tab_heatmap, tab_peak, tab_decomp,
 tab_sim, tab_timemachine, tab_sensitivity,
 tab_compare, tab_diagnosis) = st.tabs(tab_names)

# ════════════════════════════════════════
# Tab 1: Overview
# ════════════════════════════════════════
with tab_overview:
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
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": THRESHOLD},
            },
            title={"text": f"Risk Score ({date_str})"},
        ))
        fig_gauge.update_layout(height=350, margin=dict(t=60, b=20, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_bar:
        if details:
            props = []
            for pid, vals in details.items():
                props.append({"pid": pid, "name": PROP_SHORT.get(pid, pid),
                              "ratio": vals.get("ratio", 0), "contribution": vals.get("contribution", 0)})
            prop_df = pd.DataFrame(props).sort_values("contribution", ascending=True)
            colors = ["#ef5350" if r >= 1.5 else "#ffa726" if r >= 1.0 else "#66bb6a" for r in prop_df["ratio"]]
            fig_bar = go.Figure(go.Bar(
                x=prop_df["contribution"],
                y=[f"{row['name']} ({row['ratio']:.2f}x)" for _, row in prop_df.iterrows()],
                orientation="h", marker_color=colors,
                text=[f"{c:.4f}" for c in prop_df["contribution"]], textposition="outside",
                hovertemplate="%{y}<br>Contribution: %{x:.4f}<extra></extra>",
            ))
            fig_bar.update_layout(title="성질별 기여도 & 빈도비", xaxis_title="Contribution",
                                  height=350, margin=dict(l=150, r=60, t=40, b=40))
            st.plotly_chart(fig_bar, use_container_width=True)

    # 타임라인 — 날짜 범위 선택 가능
    st.subheader("Risk Score Timeline")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        range_opt = st.radio("기간", ["전체", "최근 10일", "WARNING 구간"], horizontal=True, key="tl_range")
    if range_opt == "최근 10일":
        plot_df = df.tail(10)
    elif range_opt == "WARNING 구간":
        warn_rows = df[df["status"] == "WARNING"]
        if len(warn_rows) > 0:
            start = warn_rows.iloc[0]["date"] - pd.Timedelta(days=7)
            plot_df = df[df["date"] >= start]
        else:
            plot_df = df
    else:
        plot_df = df

    fig_line = go.Figure()
    for _, row in plot_df[plot_df["status"] == "WARNING"].iterrows():
        fig_line.add_vrect(x0=row["date"] - pd.Timedelta(hours=12),
                           x1=row["date"] + pd.Timedelta(hours=12),
                           fillcolor="rgba(255,82,82,0.08)", layer="below", line_width=0)
    for i in range(len(plot_df) - 1):
        s = plot_df.iloc[i]["risk_score"]
        c = "#ef5350" if s >= 50 else "#ffa726" if s >= THRESHOLD else "#66bb6a"
        fig_line.add_trace(go.Scatter(
            x=[plot_df.iloc[i]["date"], plot_df.iloc[i+1]["date"]],
            y=[plot_df.iloc[i]["risk_score"], plot_df.iloc[i+1]["risk_score"]],
            mode="lines", line=dict(color=c, width=3), showlegend=False, hoverinfo="skip"))
    mcols = ["#ef5350" if s >= 50 else "#ffa726" if s >= THRESHOLD else "#66bb6a" for s in plot_df["risk_score"]]
    fig_line.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["risk_score"],
        mode="markers", marker=dict(size=8, color=mcols, line=dict(width=1, color="white")),
        name="Risk Score",
        customdata=np.column_stack([plot_df["status"], plot_df["n_articles"]]),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.1f}<br>Status: %{customdata[0]}<br>Articles: %{customdata[1]:,}<extra></extra>",
    ))
    fig_line.add_hline(y=THRESHOLD, line_dash="dash", line_color="rgba(255,152,0,0.5)",
                       annotation_text=f"WARNING ({THRESHOLD})")
    p_in_range = plot_df.loc[plot_df["risk_score"].idxmax()]
    fig_line.add_annotation(x=p_in_range["date"], y=p_in_range["risk_score"],
                            text=f"Peak {p_in_range['risk_score']:.1f}", showarrow=True,
                            arrowhead=2, arrowcolor="#ef5350", font=dict(color="#ef5350", size=12), ax=0, ay=-30)
    fig_line.update_layout(height=400, hovermode="x unified", margin=dict(t=30, b=40),
                           xaxis_title="", yaxis_title="Risk Score")
    st.plotly_chart(fig_line, use_container_width=True)

# ════════════════════════════════════════
# Tab 2: Property Heatmap
# ════════════════════════════════════════
with tab_heatmap:
    st.subheader("Property Activation Heatmap")
    st.caption("빈도비(ratio): 1.0 = 평상시 수준, 1.5+ = 극강 신호")

    warn_df = detail_df[detail_df["risk_score"] >= THRESHOLD * 0.5]
    use_df = warn_df if len(warn_df) >= 3 else detail_df.tail(10)

    n_dates = len(use_df)
    date_labels = [d.strftime("%m/%d") for d in use_df["date"]]
    score_labels = [f"{s:.0f}" for s in use_df["risk_score"]]

    z_data, hover_data = [], []
    for pid in PROP_ORDER:
        row_z = use_df[f"{pid}_ratio"].values.tolist()
        row_hover = [f"{PROP_NAMES[pid]}<br>Ratio: {r:.3f}<br>Score: {s:.1f}"
                     for r, s in zip(row_z, use_df["risk_score"])]
        z_data.append(row_z)
        hover_data.append(row_hover)

    fig_heat = go.Figure(go.Heatmap(
        z=z_data, x=date_labels, y=[PROP_SHORT[p] for p in PROP_ORDER],
        colorscale=[[0.0, "#1a1a2e"], [0.3, "#2d3748"], [0.5, "#4a5568"],
                    [0.6, "#d69e2e"], [0.75, "#dd6b20"], [1.0, "#e53e3e"]],
        zmin=0, zmax=2.0,
        text=[[f"{v:.2f}" for v in row] for row in z_data], texttemplate="%{text}",
        textfont={"size": 10}, customdata=hover_data,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="Ratio", tickvals=[0, 0.5, 1.0, 1.5, 2.0]),
    ))
    fig_heat.add_trace(go.Scatter(
        x=date_labels, y=["Score"] * n_dates, mode="text", text=score_labels,
        textfont=dict(size=14, color=["#ef5350" if s >= 50 else "#ffa726" if s >= THRESHOLD else "#66bb6a"
                                       for s in use_df["risk_score"]]),
        showlegend=False, hoverinfo="skip",
    ))
    fig_heat.update_layout(height=500, margin=dict(l=100, r=20, t=30, b=40), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_heat, use_container_width=True)

    selected_date = st.select_slider("날짜 선택 (상세 보기)", options=use_df["date"].tolist(),
                                     value=use_df["date"].iloc[-1],
                                     format_func=lambda d: d.strftime("%Y-%m-%d"))
    sel_row = use_df[use_df["date"] == selected_date].iloc[0]

    scol1, scol2 = st.columns([2, 1])
    with scol1:
        vals = [sel_row[f"{p}_ratio"] for p in PROP_ORDER]
        names = [PROP_SHORT[p] for p in PROP_ORDER]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=names + [names[0]],
            fill="toself", fillcolor="rgba(239,83,80,0.15)",
            line=dict(color="#ef5350", width=2), name=selected_date.strftime("%m/%d"),
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[1.0] * (len(PROP_ORDER) + 1), theta=names + [names[0]],
            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
            name="Baseline (1.0)", fill=None,
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, max(max(vals), 1.5) * 1.15])),
                                height=400, margin=dict(t=30, b=30), showlegend=True)
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
    st.markdown("> 스코어 고점의 높이가 후속 충격의 강도와 시급성을 예고한다.\n"
                "> 고점에서 하락하는 구간은 안전 신호가 아니라 **트리거 대기 구간**이다.")

    warn_rows = df[df["status"] == "WARNING"]
    if len(warn_rows) > 0:
        focus_start = warn_rows.iloc[0]["date"] - pd.Timedelta(days=7)
        focus_df = df[df["date"] >= focus_start].copy().reset_index(drop=True)
    else:
        focus_df = df.tail(10).copy().reset_index(drop=True)

    f_peak_idx = focus_df["risk_score"].idxmax()
    f_peak = focus_df.loc[f_peak_idx]
    fig_peak = go.Figure()
    pre = focus_df.loc[:f_peak_idx]
    fig_peak.add_trace(go.Scatter(x=pre["date"], y=pre["risk_score"],
                                  mode="lines+markers", name="상승 (취약성 축적)",
                                  line=dict(color="#ef5350", width=3), marker=dict(size=8, color="#ef5350")))
    post = focus_df.loc[f_peak_idx:]
    if len(post) > 1:
        fig_peak.add_trace(go.Scatter(x=post["date"], y=post["risk_score"],
                                      mode="lines+markers", name="하락 (뉴스 피로, NOT 안전)",
                                      line=dict(color="#ffa726", width=3, dash="dash"),
                                      marker=dict(size=8, color="#ffa726"),
                                      fill="tozeroy", fillcolor="rgba(255,167,38,0.08)"))
    fig_peak.add_trace(go.Scatter(
        x=[f_peak["date"]], y=[f_peak["risk_score"]],
        mode="markers+text", name="PEAK (Last Exit)",
        marker=dict(size=18, color="#ef5350", symbol="triangle-down", line=dict(width=2, color="white")),
        text=[f"PEAK {f_peak['risk_score']:.1f}"], textposition="top center",
        textfont=dict(size=14, color="#ef5350"),
    ))
    fig_peak.add_hline(y=THRESHOLD, line_dash="dash", line_color="rgba(255,152,0,0.4)")
    fig_peak.update_layout(height=450, hovermode="x unified", xaxis_title="", yaxis_title="Risk Score",
                           margin=dict(t=30, b=40), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_peak, use_container_width=True)

    st.markdown("### 역대 위기 패턴 비교")
    compare_data = [
        {"위기": "2026 이란", "사전 고점": 63.7, "고점→충격": "20일", "최대 낙폭": "-18.4%", "유형": "복합 시스템"},
        {"위기": "2011 유럽", "사전 고점": 37.3, "고점→충격": "124일", "최대 낙폭": "-17.1%", "유형": "구조적"},
        {"위기": "2025 하반기", "사전 고점": 37.1, "고점→충격": "116일", "최대 낙폭": "-7.6%", "유형": "복합"},
        {"위기": "2020 COVID", "사전 고점": 12.8, "고점→충격": "(미감지)", "최대 낙폭": "-30.1%", "유형": "신종"},
        {"위기": "2024-08 급락", "사전 고점": 11.4, "고점→충격": "(미감지)", "최대 낙폭": "-12.1%", "유형": "기술적"},
    ]
    st.dataframe(pd.DataFrame(compare_data), hide_index=True, use_container_width=True)
    st.info("**패턴**: 고점 ~37 → 4개월 여유, -7~17% 낙폭 | 고점 ~64 → 3주 여유, -18%+ 낙폭\n\n"
            "**핵심**: 스코어 하락 = 뉴스 피로, 구조적 취약성은 그대로. 하락 구간이 가장 위험.")

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

        det_from = parse_details_for_date(df[df["date"] == date_from].iloc[0])
        det_to = parse_details_for_date(df[df["date"] == date_to].iloc[0])
        row_from = df[df["date"] == date_from].iloc[0]
        row_to = df[df["date"] == date_to].iloc[0]

        score_delta = row_to["risk_score"] - row_from["risk_score"]
        st.metric("Score 변화", f"{row_to['risk_score']:.1f}",
                  delta=f"{score_delta:+.1f} (from {row_from['risk_score']:.1f})", delta_color="inverse")

        changes = []
        for pid in PROP_ORDER:
            c_from = det_from.get(pid, {}).get("contribution", 0)
            c_to = det_to.get(pid, {}).get("contribution", 0)
            r_from = det_from.get(pid, {}).get("ratio", 0)
            r_to = det_to.get(pid, {}).get("ratio", 0)
            changes.append({"name": PROP_SHORT[pid], "ratio_from": r_from, "ratio_to": r_to,
                            "contrib_delta": c_to - c_from})
        change_df = pd.DataFrame(changes).sort_values("contrib_delta")

        fig_waterfall = go.Figure(go.Waterfall(
            x=[r["name"] for _, r in change_df.iterrows()],
            y=[r["contrib_delta"] for _, r in change_df.iterrows()],
            measure=["relative"] * len(change_df),
            increasing=dict(marker_color="#ef5350"), decreasing=dict(marker_color="#66bb6a"),
            text=[f"{d:+.4f}" for d in change_df["contrib_delta"]], textposition="outside", textfont=dict(size=10),
        ))
        fig_waterfall.update_layout(
            title=f"Contribution 변화 ({date_from.strftime('%m/%d')} → {date_to.strftime('%m/%d')})",
            yaxis_title="Δ Contribution", height=400, margin=dict(t=50, b=40))
        st.plotly_chart(fig_waterfall, use_container_width=True)

        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Bar(
            x=[PROP_SHORT[p] for p in PROP_ORDER],
            y=[det_from.get(p, {}).get("ratio", 0) for p in PROP_ORDER],
            name=date_from.strftime("%m/%d"), marker_color="rgba(66,165,245,0.6)"))
        fig_ratio.add_trace(go.Bar(
            x=[PROP_SHORT[p] for p in PROP_ORDER],
            y=[det_to.get(p, {}).get("ratio", 0) for p in PROP_ORDER],
            name=date_to.strftime("%m/%d"), marker_color="rgba(239,83,80,0.6)"))
        fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Baseline")
        fig_ratio.update_layout(barmode="group", height=350, margin=dict(t=30, b=40),
                                yaxis_title="Ratio", legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_ratio, use_container_width=True)
    else:
        st.info("비교할 데이터가 부족합니다.")

# ════════════════════════════════════════
# Tab 5: 시뮬레이터 (What-If)
# ════════════════════════════════════════
with tab_sim:
    st.subheader("What-If 시뮬레이터")
    st.caption("각 성질의 빈도비(ratio)를 조절하고 스코어가 어떻게 변하는지 실시간으로 확인하세요.")

    if MODEL_PARAMS is None:
        st.error("모델 파라미터를 로드할 수 없습니다 (significant_properties.csv / risk_scores.csv 필요)")
    else:
        # 프리셋 선택
        preset_col1, preset_col2 = st.columns([2, 1])
        with preset_col1:
            preset = st.selectbox("프리셋 로드", [
                "(직접 조절)",
                f"현재 상태 ({date_str})",
            ] + [d.strftime("%Y-%m-%d") + f" (Score {r:.0f})" for d, r in zip(df["date"], df["risk_score"])],
                key="sim_preset")
        with preset_col2:
            scale_pct = st.slider("전체 스케일 (%)", 10, 300, 100, 5, key="sim_scale",
                                  help="모든 ratio를 일괄 스케일링합니다")

        # 프리셋에서 초기값 로드
        if preset.startswith("현재") or preset.startswith("(직접"):
            base_ratios = {pid: details.get(pid, {}).get("ratio", 0.0) for pid in PROP_ORDER}
        else:
            preset_date_str = preset.split(" ")[0]
            preset_row = df[df["date"] == pd.Timestamp(preset_date_str)]
            if len(preset_row) > 0:
                preset_det = parse_details_for_date(preset_row.iloc[0])
                base_ratios = {pid: preset_det.get(pid, {}).get("ratio", 0.0) for pid in PROP_ORDER}
            else:
                base_ratios = {pid: 0.0 for pid in PROP_ORDER}

        # 프리셋 변경 감지 → 슬라이더 session_state 갱신
        _prev_preset = st.session_state.get("_sim_preset_prev")
        _prev_scale = st.session_state.get("_sim_scale_prev")
        _needs_rerun = False

        if _prev_preset != preset:
            for pid in PROP_ORDER:
                st.session_state[f"sim_{pid}"] = base_ratios.get(pid, 0.0)
            st.session_state["_sim_preset_prev"] = preset
            st.session_state["_sim_scale_prev"] = scale_pct
            _needs_rerun = True

        if _prev_scale != scale_pct and _prev_scale is not None and not _needs_rerun:
            # 스케일 변경: 기존 슬라이더 값에 비례 적용
            ratio = scale_pct / _prev_scale if _prev_scale > 0 else 1.0
            for pid in PROP_ORDER:
                old_val = st.session_state.get(f"sim_{pid}", base_ratios.get(pid, 0.0))
                st.session_state[f"sim_{pid}"] = round(old_val * ratio, 3)
            st.session_state["_sim_scale_prev"] = scale_pct
            _needs_rerun = True

        if _prev_scale is None:
            st.session_state["_sim_scale_prev"] = scale_pct
        if _prev_preset is None:
            st.session_state["_sim_preset_prev"] = preset

        if _needs_rerun:
            st.rerun()

        st.markdown("---")

        # 슬라이더 + 결과 나란히
        sim_left, sim_right = st.columns([3, 2])

        with sim_left:
            st.markdown("**성질별 빈도비 조절**")
            sim_ratios = {}
            slider_cols = st.columns(3)
            for idx, pid in enumerate(PROP_ORDER):
                col = slider_cols[idx % 3]
                base_val = base_ratios.get(pid, 0.0)
                cur_val = st.session_state.get(f"sim_{pid}", base_val)
                max_val = max(3.0, base_val * 3, cur_val * 1.5)
                with col:
                    val = st.slider(
                        PROP_SHORT[pid],
                        0.0, round(max_val, 2), cur_val,
                        0.01,
                        key=f"sim_{pid}",
                        help=f"{PROP_NAMES[pid]}\n가중치: {MODEL_PARAMS['weights'].get(pid, 0):.3f}",
                    )
                    sim_ratios[pid] = val

        # 시뮬레이션 실행
        sim_result = simulate_score(sim_ratios, MODEL_PARAMS)
        sim_score = sim_result["risk_score"]
        sim_status = "WARNING" if sim_score >= THRESHOLD else "NORMAL"
        score_diff = sim_score - score

        with sim_right:
            # 게이지
            fig_sim_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sim_score,
                delta={"reference": score, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if sim_score >= THRESHOLD else "darkgreen"},
                    "steps": [
                        {"range": [0, THRESHOLD], "color": "#e8f5e9"},
                        {"range": [THRESHOLD, 50], "color": "#fff3e0"},
                        {"range": [50, 100], "color": "#ffebee"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": THRESHOLD},
                },
                title={"text": "시뮬레이션 스코어"},
            ))
            fig_sim_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
            st.plotly_chart(fig_sim_gauge, use_container_width=True)

            # 상태 표시
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("시뮬 Score", f"{sim_score:.1f}",
                          delta=f"{score_diff:+.1f} vs 현재", delta_color="inverse")
            with mc2:
                st.metric("Status", sim_status)

            # 현재 vs 시뮬 레이더 비교
            cur_vals = [details.get(p, {}).get("ratio", 0) for p in PROP_ORDER]
            sim_vals = [sim_ratios[p] for p in PROP_ORDER]
            names = [PROP_SHORT[p] for p in PROP_ORDER]

            fig_sim_radar = go.Figure()
            fig_sim_radar.add_trace(go.Scatterpolar(
                r=sim_vals + [sim_vals[0]], theta=names + [names[0]],
                fill="toself", name="시뮬레이션",
                fillcolor="rgba(21,101,192,0.2)", line=dict(color="#42a5f5", width=2),
            ))
            fig_sim_radar.add_trace(go.Scatterpolar(
                r=cur_vals + [cur_vals[0]], theta=names + [names[0]],
                fill="toself", name=f"현재 ({date_str})",
                fillcolor="rgba(239,83,80,0.15)", line=dict(color="#ef5350", width=1, dash="dash"),
            ))
            fig_sim_radar.add_trace(go.Scatterpolar(
                r=[1.0] * (len(PROP_ORDER) + 1), theta=names + [names[0]],
                line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                name="Baseline", fill=None,
            ))
            r_max = max(max(sim_vals), max(cur_vals), 1.5) * 1.15
            fig_sim_radar.update_layout(polar=dict(radialaxis=dict(range=[0, r_max])),
                                        height=350, margin=dict(t=20, b=20), showlegend=True)
            st.plotly_chart(fig_sim_radar, use_container_width=True)

        # 기여도 변화 테이블
        st.markdown("**성질별 기여도 상세**")
        sim_table = []
        for pid in PROP_ORDER:
            sd = sim_result["details"][pid]
            cur_c = details.get(pid, {}).get("contribution", 0)
            sim_table.append({
                "성질": PROP_SHORT[pid],
                "현재 ratio": details.get(pid, {}).get("ratio", 0),
                "시뮬 ratio": sd["ratio"],
                "가중치": sd["weight"],
                "현재 기여": cur_c,
                "시뮬 기여": sd["contribution"],
                "변화": sd["contribution"] - cur_c,
            })
        st.dataframe(pd.DataFrame(sim_table), hide_index=True, use_container_width=True,
                     column_config={
                         "현재 ratio": st.column_config.NumberColumn(format="%.3f"),
                         "시뮬 ratio": st.column_config.NumberColumn(format="%.3f"),
                         "가중치": st.column_config.NumberColumn(format="%.3f"),
                         "현재 기여": st.column_config.NumberColumn(format="%.4f"),
                         "시뮬 기여": st.column_config.NumberColumn(format="%.4f"),
                         "변화": st.column_config.NumberColumn(format="%+.4f"),
                     })

        # 역대 분포에서 시뮬 스코어 위치
        rs_path = OUTPUTS / "risk_scores.csv"
        if rs_path.exists():
            rs = pd.read_csv(rs_path)
            all_scores = rs["model_b_risk_score"].values
            be_scores = rs.loc[rs["actual_label"] == 1, "model_b_risk_score"].values
            ctrl_scores = rs.loc[rs["actual_label"] == 0, "model_b_risk_score"].values

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=ctrl_scores, nbinsx=50, name="대조군",
                                            marker_color="rgba(66,165,245,0.4)", opacity=0.7))
            fig_dist.add_trace(go.Histogram(x=be_scores, nbinsx=50, name="BlackEvent",
                                            marker_color="rgba(239,83,80,0.4)", opacity=0.7))
            fig_dist.add_vline(x=sim_score, line_dash="solid", line_color="#42a5f5", line_width=3,
                               annotation_text=f"시뮬 {sim_score:.1f}", annotation_font_color="#42a5f5")
            fig_dist.add_vline(x=score, line_dash="dash", line_color="#ef5350", line_width=2,
                               annotation_text=f"현재 {score:.1f}", annotation_font_color="#ef5350")
            fig_dist.add_vline(x=THRESHOLD, line_dash="dot", line_color="orange",
                               annotation_text="WARNING", annotation_font_color="orange")
            fig_dist.update_layout(barmode="overlay", height=300, margin=dict(t=30, b=40),
                                   xaxis_title="Risk Score", yaxis_title="빈도",
                                   title="역대 스코어 분포 내 시뮬레이션 위치")
            st.plotly_chart(fig_dist, use_container_width=True)

            pct = float(np.mean(all_scores <= sim_score) * 100)
            st.caption(f"시뮬레이션 스코어 {sim_score:.1f}은 전체 이력 상위 **{100-pct:.1f}%** 수준입니다.")

# ════════════════════════════════════════
# Tab 6: 타임머신
# ════════════════════════════════════════
with tab_timemachine:
    st.subheader("타임머신")
    st.caption("과거 날짜를 선택하면 그날의 상태를 '오늘'인 것처럼 탐색합니다.")

    # 날짜 선택 (슬라이더 + 멀티 선택)
    tm_mode = st.radio("모드", ["단일 날짜 탐색", "다중 날짜 비교"], horizontal=True, key="tm_mode")

    if tm_mode == "단일 날짜 탐색":
        tm_date = st.select_slider("날짜", options=df["date"].tolist(),
                                   value=df["date"].iloc[-1],
                                   format_func=lambda d: d.strftime("%Y-%m-%d"),
                                   key="tm_date")
        tm_row = df[df["date"] == tm_date].iloc[0]
        tm_det = parse_details_for_date(tm_row)
        tm_score = tm_row["risk_score"]
        tm_status = tm_row["status"]

        # 헤더
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            c = "🔴" if tm_score >= 50 else "🟡" if tm_score >= THRESHOLD else "🟢"
            st.metric(f"{c} Score", f"{tm_score:.1f}")
        with tc2:
            st.metric("Status", tm_status)
        with tc3:
            st.metric("기사 수", f"{tm_row['n_articles']:,}")
        with tc4:
            tm_active = sum(1 for p in tm_det.values() if p.get("ratio", 0) > 1.0)
            st.metric("Active", f"{tm_active}/12")

        # 게이지 + 레이더 + 바
        tg1, tg2 = st.columns([1, 2])
        with tg1:
            fig_tm_g = go.Figure(go.Indicator(
                mode="gauge+number", value=tm_score,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if tm_score >= THRESHOLD else "darkgreen"},
                    "steps": [
                        {"range": [0, THRESHOLD], "color": "#e8f5e9"},
                        {"range": [THRESHOLD, 50], "color": "#fff3e0"},
                        {"range": [50, 100], "color": "#ffebee"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": THRESHOLD},
                },
                title={"text": tm_date.strftime("%Y-%m-%d")},
            ))
            fig_tm_g.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20))
            st.plotly_chart(fig_tm_g, use_container_width=True)

        with tg2:
            # 레이더
            tm_vals = [tm_det.get(p, {}).get("ratio", 0) for p in PROP_ORDER]
            names = [PROP_SHORT[p] for p in PROP_ORDER]
            fig_tm_r = go.Figure()
            fig_tm_r.add_trace(go.Scatterpolar(
                r=tm_vals + [tm_vals[0]], theta=names + [names[0]],
                fill="toself", fillcolor="rgba(239,83,80,0.15)",
                line=dict(color="#ef5350", width=2), name=tm_date.strftime("%m/%d"),
            ))
            fig_tm_r.add_trace(go.Scatterpolar(
                r=[1.0] * (len(PROP_ORDER) + 1), theta=names + [names[0]],
                line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
                name="Baseline", fill=None,
            ))
            fig_tm_r.update_layout(polar=dict(radialaxis=dict(range=[0, max(max(tm_vals), 1.5) * 1.15])),
                                   height=350, margin=dict(t=20, b=20), showlegend=True)
            st.plotly_chart(fig_tm_r, use_container_width=True)

        # 기여도 바
        if tm_det:
            tm_props = []
            for pid, vals in tm_det.items():
                tm_props.append({"name": PROP_SHORT.get(pid, pid),
                                 "ratio": vals.get("ratio", 0), "contribution": vals.get("contribution", 0)})
            tm_pf = pd.DataFrame(tm_props).sort_values("contribution", ascending=True)
            tc = ["#ef5350" if r >= 1.5 else "#ffa726" if r >= 1.0 else "#66bb6a" for r in tm_pf["ratio"]]
            fig_tm_bar = go.Figure(go.Bar(
                x=tm_pf["contribution"],
                y=[f"{row['name']} ({row['ratio']:.2f}x)" for _, row in tm_pf.iterrows()],
                orientation="h", marker_color=tc,
                text=[f"{c:.4f}" for c in tm_pf["contribution"]], textposition="outside",
            ))
            fig_tm_bar.update_layout(title="성질별 기여도", xaxis_title="Contribution",
                                     height=350, margin=dict(l=150, r=60, t=40, b=40))
            st.plotly_chart(fig_tm_bar, use_container_width=True)

        # 유사 사례 매칭
        be_profiles_path = OUTPUTS / "be_profiles.json"
        if be_profiles_path.exists() and tm_det:
            from src.diagnosis.profile_matcher import find_similar_events
            tm_ratios = {pid: vals.get("ratio", 0) for pid, vals in tm_det.items()}
            tm_matches = find_similar_events(tm_ratios, be_profiles_path, top_n=20)
            seen = set()
            uniq = []
            for m in tm_matches:
                if m["event_id"] not in seen:
                    seen.add(m["event_id"])
                    uniq.append(m)
            if uniq:
                top = uniq[0]
                od = {"D-0": "당일", "D-7": "7일 전", "D-14": "14일 전", "D-21": "21일 전"}
                st.info(f"이 날의 뉴스 환경은 **{top['first_shock_date']}** 폭락 "
                        f"**{od.get(top['offset'], top['offset'])}** 상황과 가장 유사합니다. "
                        f"(유사도 {top['similarity']:.1%}, 당시 {top['max_cumulative_drop']}% 하락)")

    else:
        # 다중 날짜 비교
        multi_dates = st.multiselect(
            "비교할 날짜 선택 (최대 5개)",
            options=df["date"].tolist(),
            default=[df["date"].iloc[-1]],
            format_func=lambda d: d.strftime("%Y-%m-%d"),
            max_selections=5,
            key="tm_multi",
        )

        if multi_dates:
            # 레이더 오버레이
            names = [PROP_SHORT[p] for p in PROP_ORDER]
            colors = ["#ef5350", "#42a5f5", "#66bb6a", "#ffa726", "#ab47bc"]
            fig_multi = go.Figure()
            all_max = 1.5
            for i, md in enumerate(sorted(multi_dates)):
                md_row = df[df["date"] == md].iloc[0]
                md_det = parse_details_for_date(md_row)
                md_vals = [md_det.get(p, {}).get("ratio", 0) for p in PROP_ORDER]
                all_max = max(all_max, max(md_vals))
                fig_multi.add_trace(go.Scatterpolar(
                    r=md_vals + [md_vals[0]], theta=names + [names[0]],
                    fill="toself", fillcolor=f"rgba({','.join(str(int(colors[i][j:j+2], 16)) for j in (1,3,5))},0.1)",
                    line=dict(color=colors[i], width=2),
                    name=f"{md.strftime('%m/%d')} (Score {md_row['risk_score']:.0f})",
                ))
            fig_multi.add_trace(go.Scatterpolar(
                r=[1.0] * (len(PROP_ORDER) + 1), theta=names + [names[0]],
                line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
                name="Baseline", fill=None,
            ))
            fig_multi.update_layout(polar=dict(radialaxis=dict(range=[0, all_max * 1.15])),
                                    height=500, margin=dict(t=30, b=30))
            st.plotly_chart(fig_multi, use_container_width=True)

            # 비교 테이블
            comp_rows = []
            for md in sorted(multi_dates):
                md_row = df[df["date"] == md].iloc[0]
                md_det = parse_details_for_date(md_row)
                entry = {"날짜": md.strftime("%Y-%m-%d"), "Score": md_row["risk_score"],
                         "Status": md_row["status"], "기사수": md_row["n_articles"]}
                for pid in PROP_ORDER:
                    entry[PROP_SHORT[pid]] = md_det.get(pid, {}).get("ratio", 0)
                comp_rows.append(entry)
            comp_df = pd.DataFrame(comp_rows)
            col_config = {PROP_SHORT[p]: st.column_config.NumberColumn(format="%.3f") for p in PROP_ORDER}
            col_config["Score"] = st.column_config.NumberColumn(format="%.1f")
            st.dataframe(comp_df, hide_index=True, use_container_width=True, column_config=col_config)

# ════════════════════════════════════════
# Tab 7: 민감도 분석
# ════════════════════════════════════════
with tab_sensitivity:
    st.subheader("민감도 분석")
    st.caption("각 성질이 스코어에 미치는 영향력을 분석합니다.")

    if MODEL_PARAMS is None:
        st.error("모델 파라미터 로드 실패")
    else:
        # 기준 날짜 선택
        sens_date = st.select_slider("기준 날짜", options=df["date"].tolist(),
                                     value=df["date"].iloc[-1],
                                     format_func=lambda d: d.strftime("%Y-%m-%d"),
                                     key="sens_date")
        sens_row = df[df["date"] == sens_date].iloc[0]
        sens_det = parse_details_for_date(sens_row)
        base_ratios = {pid: sens_det.get(pid, {}).get("ratio", 0.0) for pid in PROP_ORDER}
        base_result = simulate_score(base_ratios, MODEL_PARAMS)
        base_score = base_result["risk_score"]

        st.markdown(f"**기준 스코어: {base_score:.1f}** ({sens_date.strftime('%Y-%m-%d')})")

        # ── 토네이도 차트: 각 성질 +50% / -50% 시 스코어 변화 ──
        st.markdown("### 토네이도 차트")
        st.caption("각 성질의 ratio를 ±50% 변동시켰을 때 스코어 영향")

        pct_change = st.slider("변동폭 (%)", 10, 200, 50, 10, key="sens_pct")

        tornado_data = []
        for pid in PROP_ORDER:
            r_base = base_ratios[pid]
            # +변동
            up_ratios = dict(base_ratios)
            up_ratios[pid] = r_base * (1 + pct_change / 100)
            up_score = simulate_score(up_ratios, MODEL_PARAMS)["risk_score"]
            # -변동
            dn_ratios = dict(base_ratios)
            dn_ratios[pid] = max(0, r_base * (1 - pct_change / 100))
            dn_score = simulate_score(dn_ratios, MODEL_PARAMS)["risk_score"]
            tornado_data.append({
                "pid": pid, "name": PROP_SHORT[pid],
                "up": up_score - base_score, "down": dn_score - base_score,
                "range": abs(up_score - dn_score),
            })

        tornado_df = pd.DataFrame(tornado_data).sort_values("range")

        fig_tornado = go.Figure()
        fig_tornado.add_trace(go.Bar(
            y=tornado_df["name"], x=tornado_df["down"],
            orientation="h", name=f"-{pct_change}%",
            marker_color="#66bb6a", text=[f"{v:+.2f}" for v in tornado_df["down"]],
            textposition="outside", textfont=dict(size=10),
        ))
        fig_tornado.add_trace(go.Bar(
            y=tornado_df["name"], x=tornado_df["up"],
            orientation="h", name=f"+{pct_change}%",
            marker_color="#ef5350", text=[f"{v:+.2f}" for v in tornado_df["up"]],
            textposition="outside", textfont=dict(size=10),
        ))
        fig_tornado.update_layout(barmode="overlay", height=450, margin=dict(l=100, r=80, t=30, b=40),
                                  xaxis_title="Score 변화", title="성질별 민감도 (토네이도)")
        st.plotly_chart(fig_tornado, use_container_width=True)

        # ── 한계 기여도: ratio가 0→2로 갈 때 스코어 곡선 ──
        st.markdown("### 개별 성질 응답 곡선")
        st.caption("선택한 성질의 ratio를 0에서 3까지 변화시켰을 때 전체 스코어가 어떻게 변하는지")

        sel_pid = st.selectbox("성질 선택", PROP_ORDER,
                               format_func=lambda p: f"{PROP_SHORT[p]} — {PROP_NAMES[p]}",
                               key="sens_sel")

        test_ratios = np.linspace(0, 3.0, 100)
        test_scores = []
        for tr in test_ratios:
            test_r = dict(base_ratios)
            test_r[sel_pid] = tr
            test_scores.append(simulate_score(test_r, MODEL_PARAMS)["risk_score"])

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=test_ratios, y=test_scores, mode="lines",
            line=dict(color="#42a5f5", width=3), name="Score",
        ))
        # 현재 위치 마커
        cur_ratio = base_ratios[sel_pid]
        fig_curve.add_trace(go.Scatter(
            x=[cur_ratio], y=[base_score], mode="markers+text",
            marker=dict(size=14, color="#ef5350", line=dict(width=2, color="white")),
            text=[f"현재 ({cur_ratio:.2f})"], textposition="top center",
            textfont=dict(color="#ef5350", size=12), name="현재",
        ))
        fig_curve.add_hline(y=THRESHOLD, line_dash="dash", line_color="orange",
                            annotation_text="WARNING")
        fig_curve.add_vline(x=1.0, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                            annotation_text="Baseline")
        fig_curve.update_layout(
            height=400, margin=dict(t=30, b=40),
            xaxis_title=f"{PROP_SHORT[sel_pid]} ratio",
            yaxis_title="Risk Score",
            title=f"{PROP_SHORT[sel_pid]} 응답 곡선 (가중치: {MODEL_PARAMS['weights'].get(sel_pid, 0):.3f})",
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # ── 임계 ratio 계산 ──
        st.markdown("### WARNING 도달까지 필요한 ratio")
        if base_score < THRESHOLD:
            thresh_data = []
            for pid in PROP_ORDER:
                w = MODEL_PARAMS["weights"].get(pid, 0)
                if w <= 0:
                    continue
                cur_r = base_ratios[pid]
                cur_excess = max(cur_r - 1.0, 0.0)
                cur_contrib = w * cur_excess
                need_raw = (THRESHOLD / 100) * MODEL_PARAMS["empirical_max"] - base_result["raw_score"] + cur_contrib
                if need_raw <= 0:
                    need_ratio = cur_r
                else:
                    need_ratio = (need_raw / w) + 1.0
                thresh_data.append({
                    "성질": PROP_SHORT[pid],
                    "현재 ratio": cur_r,
                    "WARNING ratio": round(need_ratio, 3),
                    "필요 증가": round(need_ratio - cur_r, 3),
                    "현재 대비": f"+{(need_ratio/cur_r - 1)*100:.0f}%" if cur_r > 0 else "N/A",
                })
            thresh_df = pd.DataFrame(thresh_data).sort_values("필요 증가")
            st.dataframe(thresh_df, hide_index=True, use_container_width=True,
                         column_config={
                             "현재 ratio": st.column_config.NumberColumn(format="%.3f"),
                             "WARNING ratio": st.column_config.NumberColumn(format="%.3f"),
                             "필요 증가": st.column_config.NumberColumn(format="%.3f"),
                         })
            st.caption("단일 성질만 변경한다고 가정했을 때의 계산입니다.")
        else:
            st.success(f"현재 이미 WARNING 상태입니다 (Score {base_score:.1f})")
            # NORMAL로 돌아가려면?
            excess_raw = base_result["raw_score"] - (THRESHOLD / 100) * MODEL_PARAMS["empirical_max"]
            normal_data = []
            can_alone_count = 0
            for pid in PROP_ORDER:
                w = MODEL_PARAMS["weights"].get(pid, 0)
                if w <= 0:
                    continue
                cur_r = base_ratios[pid]
                cur_excess = max(cur_r - 1.0, 0.0)
                cur_contrib = w * cur_excess
                if cur_contrib >= excess_raw:
                    need_excess = (cur_contrib - excess_raw) / w
                    need_ratio = need_excess + 1.0
                    can_alone_count += 1
                    normal_data.append({
                        "성질": PROP_SHORT[pid],
                        "현재 ratio": cur_r,
                        "NORMAL ratio": round(need_ratio, 3),
                        "필요 감소": round(cur_r - need_ratio, 3),
                    })
                else:
                    normal_data.append({
                        "성질": PROP_SHORT[pid],
                        "현재 ratio": cur_r,
                        "NORMAL ratio": None,
                        "필요 감소": None,
                    })
            if normal_data:
                st.markdown("**NORMAL 복귀에 필요한 ratio 감소:**")
                nd_df = pd.DataFrame(normal_data)
                nd_df["NORMAL ratio"] = nd_df["NORMAL ratio"].apply(
                    lambda x: f"{x:.3f}" if x is not None else "단일로 불가")
                nd_df["필요 감소"] = nd_df["필요 감소"].apply(
                    lambda x: f"{x:.3f}" if x is not None else "-")
                st.dataframe(nd_df, hide_index=True, use_container_width=True,
                             column_config={"현재 ratio": st.column_config.NumberColumn(format="%.3f")})
                if can_alone_count == 0:
                    # 최소 조합 계산: 기여도 큰 순으로 누적
                    sorted_contribs = sorted(
                        [(pid, MODEL_PARAMS["weights"].get(pid, 0) * max(base_ratios[pid] - 1.0, 0.0))
                         for pid in PROP_ORDER],
                        key=lambda x: -x[1])
                    cumsum = 0.0
                    needed = []
                    for pid, c in sorted_contribs:
                        if c <= 0:
                            continue
                        cumsum += c
                        needed.append(PROP_SHORT[pid])
                        if cumsum >= excess_raw:
                            break
                    st.warning(f"단일 성질로는 NORMAL 복귀 불가. "
                               f"최소 **{len(needed)}개** 성질({', '.join(needed)})을 "
                               f"동시에 baseline(1.0)까지 낮춰야 합니다.")

# ════════════════════════════════════════
# Tab 8: 위기 비교
# ════════════════════════════════════════
with tab_compare:
    st.subheader("현재 vs 과거 위기 비교")

    risk_csv = OUTPUTS / "risk_scores.csv"
    if risk_csv.exists() and details:
        rs = pd.read_csv(risk_csv)
        rs["date"] = pd.to_datetime(rs["date"])

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
            be_profiles_path = OUTPUTS / "be_profiles.json"
            if be_profiles_path.exists():
                with open(be_profiles_path, encoding="utf-8") as f:
                    be_data = json.load(f)

                event_id = str(crisis_row.get("event_id", ""))
                if event_id in be_data["profiles"]:
                    profile = be_data["profiles"][event_id]
                    offsets = profile["offsets"]
                    ref_ratios = offsets.get("D-0", offsets.get(list(offsets.keys())[0], {}))

                    cur_vals = [details.get(p, {}).get("ratio", 0) for p in PROP_ORDER]
                    ref_vals = [ref_ratios.get(p, 0) for p in PROP_ORDER]
                    names = [PROP_SHORT[p] for p in PROP_ORDER]

                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Scatterpolar(
                        r=cur_vals + [cur_vals[0]], theta=names + [names[0]],
                        fill="toself", name=f"현재 ({date_str})",
                        fillcolor="rgba(21,101,192,0.2)", line=dict(color="#42a5f5", width=2),
                    ))
                    fig_cmp.add_trace(go.Scatterpolar(
                        r=ref_vals + [ref_vals[0]], theta=names + [names[0]],
                        fill="toself", name=f"{selected_crisis}",
                        fillcolor="rgba(239,83,80,0.2)", line=dict(color="#ef5350", width=2),
                    ))
                    r_max = max(max(cur_vals), max(ref_vals)) * 1.15
                    fig_cmp.update_layout(polar=dict(radialaxis=dict(range=[0, r_max])),
                                          height=500, showlegend=True, margin=dict(t=40, b=40, l=80, r=80))
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    comp_rows = []
                    for pid in PROP_ORDER:
                        comp_rows.append({
                            "성질": PROP_SHORT[pid],
                            "현재": details.get(pid, {}).get("ratio", 0),
                            f"{selected_crisis}": ref_ratios.get(pid, 0),
                            "차이": details.get(pid, {}).get("ratio", 0) - ref_ratios.get(pid, 0),
                        })
                    comp_df = pd.DataFrame(comp_rows)
                    st.dataframe(comp_df, hide_index=True, use_container_width=True,
                                 column_config={
                                     "현재": st.column_config.NumberColumn(format="%.3f"),
                                     f"{selected_crisis}": st.column_config.NumberColumn(format="%.3f"),
                                     "차이": st.column_config.NumberColumn(format="%+.3f"),
                                 })
                else:
                    st.info(f"이벤트 {event_id}의 프로필이 없습니다.")
            else:
                st.info("be_profiles.json이 없습니다.")
        else:
            st.info(f"{crisis_date}의 스코어 데이터가 없습니다.")
    else:
        st.info("risk_scores.csv 또는 현재 성질 데이터가 없습니다.")

# ════════════════════════════════════════
# Tab 9: 상태 진단
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

            st.markdown("**유사 과거 사례 (상위 10건)**")
            table_rows = []
            for m in unique_matches[:10]:
                table_rows.append({
                    "사건": f"BE_{m['event_id']}", "날짜": m["first_shock_date"],
                    "매칭": _offset_desc.get(m["offset"], m["offset"]),
                    "유사도": m["similarity"],
                    "최대 하락": f"{m['max_cumulative_drop']}%",
                    "기간(일)": m["duration_days"], "충격": m["shock_count"],
                })
            st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True,
                         column_config={"유사도": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})

            if len(df) >= 2 and delta is not None:
                st.markdown("---")
                prev_row = df.iloc[-2]
                prev_det = parse_details_for_date(prev_row)

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

# ── 하단 ──
st.divider()
st.caption("💡 파이프라인 실행 후 페이지를 새로고침하면 최신 데이터가 반영됩니다.")
