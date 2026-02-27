"""
Dashboard — 리스크 게이지, 성질 기여도, 상태 진단, 일별 추이
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

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

latest = df.iloc[-1]
score = latest["risk_score"]
status = latest["status"]
date_str = latest["date"].strftime("%Y-%m-%d")

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

# ── 3. 현재 상태 진단 ──
st.subheader("현재 상태 진단")

_be_profiles_path = OUTPUTS / "be_profiles.json"
_risk_scores_path = OUTPUTS / "risk_scores.csv"

if not _be_profiles_path.exists():
    st.info("be_profiles.json이 없습니다. 프로필 사전 계산을 실행하세요.")
elif not details:
    st.info("성질 상세 데이터가 없어 진단을 수행할 수 없습니다.")
else:
    from src.diagnosis.profile_matcher import find_similar_events, compute_score_percentile

    current_ratios = {pid: vals.get("ratio", 0) for pid, vals in details.items()}

    matches = find_similar_events(current_ratios, _be_profiles_path, top_n=20)

    # 이벤트별 최고 유사도만 추출
    _seen = set()
    unique_matches = []
    for m in matches:
        if m["event_id"] not in _seen:
            _seen.add(m["event_id"])
            unique_matches.append(m)

    if unique_matches:
        top = unique_matches[0]

        _offset_desc = {
            "D-0": "당일",
            "D-7": "7일 전",
            "D-14": "14일 전",
            "D-21": "21일 전",
        }
        _offset_label = _offset_desc.get(top["offset"], top["offset"])

        # 요약 배너
        st.info(
            f"현재 뉴스 환경은 **{top['first_shock_date']}** 폭락 "
            f"**{_offset_label}** 상황과 가장 유사합니다.\n\n"
            f"코사인 유사도: **{top['similarity']:.1%}** | "
            f"당시 결과: **{top['max_cumulative_drop']}%** 하락, "
            f"{top['duration_days']}일간 {top['shock_count']}회 충격"
        )

        # 퍼센타일 메트릭
        if _risk_scores_path.exists():
            pct = compute_score_percentile(score, _risk_scores_path)
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                top_pct = 100 - pct["overall_percentile"]
                st.metric("전체 퍼센타일", f"상위 {top_pct:.1f}%")
            with pcol2:
                st.metric("BlackEvent 평균", f"{pct['be_mean']:.1f}")
            with pcol3:
                st.metric("대조군 평균", f"{pct['ctrl_mean']:.1f}")

        # 레이더 차트 + 유사 사례 테이블
        col_radar, col_similar = st.columns([3, 2])

        with col_radar:
            with open(_be_profiles_path, encoding="utf-8") as f:
                be_data = json.load(f)

            property_ids = be_data["property_ids"]
            top_profile = be_data["profiles"][str(top["event_id"])]
            top_ratios = top_profile["offsets"][top["offset"]]

            short_names = [PROP_SHORT.get(p, p) for p in property_ids]

            cur_vals = [current_ratios.get(p, 0) for p in property_ids]
            ref_vals = [top_ratios.get(p, 0) for p in property_ids]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=cur_vals + [cur_vals[0]],
                theta=short_names + [short_names[0]],
                fill="toself",
                name=f"현재 ({date_str})",
                fillcolor="rgba(21,101,192,0.15)",
                line=dict(color="#1565c0", width=2),
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=ref_vals + [ref_vals[0]],
                theta=short_names + [short_names[0]],
                fill="toself",
                name=f"BE_{top['event_id']} {top['offset']}",
                fillcolor="rgba(239,83,80,0.15)",
                line=dict(color="#ef5350", width=2),
            ))
            r_max = max(max(cur_vals), max(ref_vals)) * 1.1
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, r_max])),
                showlegend=True,
                height=420,
                margin=dict(t=40, b=40, l=80, r=80),
                title="성질 빈도비 프로필 비교",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_similar:
            st.markdown("**유사 과거 사례 (상위 5건)**")
            table_rows = []
            for m in unique_matches[:5]:
                table_rows.append({
                    "사건": f"BE_{m['event_id']}",
                    "날짜": m["first_shock_date"],
                    "매칭": _offset_desc.get(m["offset"], m["offset"]),
                    "유사도": m["similarity"],
                    "최대 하락": f"{m['max_cumulative_drop']}%",
                    "기간(일)": m["duration_days"],
                    "충격": m["shock_count"],
                })
            table_df = pd.DataFrame(table_rows)
            st.dataframe(
                table_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "유사도": st.column_config.ProgressColumn(
                        min_value=0, max_value=1, format="%.3f",
                    ),
                },
            )

        # 모멘텀
        if len(df) >= 2 and delta is not None:
            st.markdown("---")
            prev_row = df.iloc[-2]
            prev_date_str = prev_row["date"].strftime("%m/%d")
            cur_date_str = latest["date"].strftime("%m/%d")
            delta_sign = "+" if delta >= 0 else ""
            st.markdown(
                f"**모멘텀**: 스코어 변화 **{delta_sign}{delta:.2f}** "
                f"({prev_date_str} → {cur_date_str})"
            )

            prev_details_str = prev_row.get("details_json", "{}")
            prev_det = json.loads(prev_details_str) if isinstance(prev_details_str, str) else {}

            if prev_det and details:
                rising = []
                falling = []
                for pid in details:
                    cur_r = details[pid].get("ratio", 0)
                    prev_r = prev_det.get(pid, {}).get("ratio", 0)
                    if cur_r > prev_r:
                        rising.append(PROP_NAMES.get(pid, pid))
                    elif cur_r < prev_r:
                        falling.append(PROP_NAMES.get(pid, pid))

                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    st.markdown(f"**상승 ({len(rising)}개)**")
                    for name in rising:
                        st.markdown(f"- {name}")
                with mcol2:
                    st.markdown(f"**하락 ({len(falling)}개)**")
                    for name in falling:
                        st.markdown(f"- {name}")

# ── 4. 일별 추이 ──
st.subheader("일별 리스크 스코어 추이")

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

# ── 5. 상세 테이블 ──
st.subheader("전체 스코어 이력")
show_df = df[["date", "risk_score", "status", "n_articles", "top_contributor"]].sort_values(
    "date", ascending=False
)
show_df.columns = ["날짜", "Risk Score", "상태", "기사 수", "Top Contributor"]
st.dataframe(show_df, hide_index=True, width="stretch")
