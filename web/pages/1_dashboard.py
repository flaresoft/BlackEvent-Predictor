"""
Dashboard — 리스크 게이지, 성질 기여도, 일별 추이
"""

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.header("Dashboard")

# ── 데이터 로드 ──
daily_csv = OUTPUTS / "daily_scores.csv"
if not daily_csv.exists():
    st.warning("daily_scores.csv가 없습니다. 파이프라인을 먼저 실행하세요.")
    st.stop()

df = pd.read_csv(daily_csv)
if len(df) == 0:
    st.warning("스코어 데이터가 비어 있습니다.")
    st.stop()

latest = df.iloc[-1]
score = latest["risk_score"]
status = latest["status"]
date_str = latest["date"]

# 이전 스코어와의 delta
delta = None
if len(df) >= 2:
    prev_score = df.iloc[-2]["risk_score"]
    delta = round(score - prev_score, 2)

# ── 1. 리스크 게이지 ──
st.subheader("현재 리스크 스코어")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Risk Score",
        f"{score:.2f}",
        delta=f"{delta:+.2f}" if delta is not None else None,
        delta_color="inverse",
    )
with col2:
    st.metric("Status", status)
with col3:
    st.metric("Threshold", f"{THRESHOLD}")
with col4:
    st.metric("기준일", date_str)

# 게이지 차트
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
fig_gauge.update_layout(height=300, margin=dict(t=60, b=20, l=40, r=40))
st.plotly_chart(fig_gauge, width="stretch")

# ── 2. 성질 기여도 바 차트 ──
st.subheader("성질별 기여도")

details_json = latest.get("details_json", "{}")
if isinstance(details_json, str):
    details = json.loads(details_json)
else:
    details = {}

if details:
    props = []
    for pid, vals in details.items():
        props.append({
            "property_id": pid,
            "name": PROP_NAMES.get(pid, pid),
            "ratio": vals.get("ratio", 0),
            "contribution": vals.get("contribution", 0),
        })
    prop_df = pd.DataFrame(props).sort_values("contribution", ascending=True)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        colors = ["#ef5350" if r > 1.0 else "#66bb6a" for r in prop_df["ratio"]]
        fig_bar = go.Figure(go.Bar(
            x=prop_df["contribution"],
            y=[f"{row['property_id']}\n{row['name']}" for _, row in prop_df.iterrows()],
            orientation="h",
            marker_color=colors,
            text=[f"{c:.4f}" for c in prop_df["contribution"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="성질별 기여도 (contribution)",
            xaxis_title="Contribution",
            yaxis_title="",
            height=450,
            margin=dict(l=200, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_bar, width="stretch")

    with col_table:
        display_df = prop_df[["property_id", "name", "ratio", "contribution"]].sort_values(
            "contribution", ascending=False
        )
        display_df.columns = ["ID", "성질", "Ratio", "Contribution"]
        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Ratio": st.column_config.NumberColumn(format="%.3f"),
                "Contribution": st.column_config.NumberColumn(format="%.4f"),
            },
        )
else:
    st.info("성질 상세 데이터가 없습니다.")

# ── 3. 일별 추이 ──
st.subheader("일별 리스크 스코어 추이")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

fig_line = go.Figure()

# WARNING 구간 배경
warning_df = df[df["status"] == "WARNING"]
if len(warning_df) > 0:
    for _, row in warning_df.iterrows():
        fig_line.add_vrect(
            x0=row["date"] - pd.Timedelta(days=0.5),
            x1=row["date"] + pd.Timedelta(days=0.5),
            fillcolor="rgba(255,0,0,0.08)",
            layer="below",
            line_width=0,
        )

# 스코어 라인
fig_line.add_trace(go.Scatter(
    x=df["date"],
    y=df["risk_score"],
    mode="lines+markers",
    name="Risk Score",
    line=dict(color="#1565c0", width=2),
    marker=dict(
        size=8,
        color=["#ef5350" if s >= THRESHOLD else "#1565c0" for s in df["risk_score"]],
    ),
))

# Threshold 선
fig_line.add_hline(
    y=THRESHOLD,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Threshold ({THRESHOLD})",
    annotation_position="top right",
)

fig_line.update_layout(
    xaxis_title="날짜",
    yaxis_title="Risk Score",
    height=400,
    hovermode="x unified",
    margin=dict(t=20, b=40),
)
st.plotly_chart(fig_line, width="stretch")

# ── 4. 상세 테이블 ──
st.subheader("전체 스코어 이력")
show_df = df[["date", "risk_score", "status", "n_articles", "top_contributor"]].sort_values(
    "date", ascending=False
)
show_df.columns = ["날짜", "Risk Score", "상태", "기사 수", "Top Contributor"]
st.dataframe(show_df, hide_index=True, width="stretch")
