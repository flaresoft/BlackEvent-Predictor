"""
자살률 심층분석 — 리스크 스코어가 자살 사망을 예측하는가?
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SOCIAL_OUTPUT = ROOT / "data" / "outputs" / "social"

st.set_page_config(page_title="자살률 심층분석", page_icon="🔬", layout="wide")


def _load_csv(filename: str) -> pd.DataFrame:
    path = SOCIAL_OUTPUT / filename
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


# ── 데이터 로드 ──
merged_df = _load_csv("merged_monthly.csv")
summary_df = _load_csv("summary.csv")
xcorr_df = _load_csv("cross_correlation.csv")
granger_df = _load_csv("granger.csv")
rolling_df = _load_csv("rolling.csv")

if merged_df.empty or "suicide" not in merged_df.columns:
    st.warning(
        "자살률 분석 데이터가 없습니다. Module 9을 먼저 실행하세요:\n\n"
        "```bash\npython -m src.module9_social.run\n```"
    )
    st.stop()

# ── 상수 ──
# 주요 위기 시점 (KOSPI BlackEvent 기반)
CRISIS_EVENTS = [
    ("2008-09", "글로벌 금융위기", "#e53935"),
    ("2011-08", "유럽 재정위기", "#f57c00"),
    ("2015-08", "중국 증시 폭락", "#7b1fa2"),
    ("2018-10", "미중 무역전쟁", "#1565c0"),
    ("2020-03", "COVID-19 팬데믹", "#2e7d32"),
    ("2022-06", "글로벌 금리 인상", "#4e342e"),
    ("2024-08", "엔캐리 트레이드 청산", "#37474f"),
    ("2026-01", "이란 위기", "#b71c1c"),
]

# ── 사이드바 ──
st.sidebar.markdown("### 분석 설정")
selected_metric = st.sidebar.radio(
    "리스크 지표",
    ["risk_mean", "risk_max", "warning_days"],
    format_func=lambda x: {
        "risk_mean": "리스크 평균",
        "risk_max": "리스크 최대",
        "warning_days": "WARNING 일수",
    }[x],
    index=0,
)

# 최적 lag 가져오기
optimal_lag = 7
if not summary_df.empty:
    match = summary_df[
        (summary_df["social_indicator"] == "suicide")
        & (summary_df["risk_metric"] == selected_metric)
    ]
    if not match.empty:
        optimal_lag = int(match.iloc[0]["best_lag"])

lag_override = st.sidebar.slider("시차 (개월)", 0, 12, optimal_lag)
show_events = st.sidebar.checkbox("위기 이벤트 표시", value=True)

METRIC_LABEL = {
    "risk_mean": "리스크 평균",
    "risk_max": "리스크 최대",
    "warning_days": "WARNING 일수",
}[selected_metric]


# ════════════════════════════════════════════════
# 헤더
# ════════════════════════════════════════════════
st.header("자살률 심층분석")
st.caption("경제 위기 리스크 스코어 → 자살 사망자 수 선행 관계 검증")

# ── 핵심 수치 카드 ──
if not summary_df.empty:
    suicide_summary = summary_df[summary_df["social_indicator"] == "suicide"]
    if not suicide_summary.empty:
        cols = st.columns(4)
        for idx, (_, row) in enumerate(suicide_summary.iterrows()):
            if idx >= 4:
                break
            metric_name = {
                "risk_mean": "평균",
                "risk_max": "최대",
                "warning_days": "WARNING",
            }.get(row["risk_metric"], row["risk_metric"])
            with cols[idx]:
                st.metric(
                    label=f"{metric_name} → 자살",
                    value=f"r = {row['best_correlation']:.3f}",
                    delta=f"lag {int(row['best_lag'])}개월",
                    delta_color="inverse",
                )

        # 4번째 칸: Granger 요약
        if not granger_df.empty:
            g_suicide = granger_df[
                (granger_df["social_indicator"] == "suicide")
                & (granger_df["risk_metric"] == selected_metric)
                & (granger_df["significant"])
            ]
            with cols[3]:
                st.metric(
                    label="Granger 인과",
                    value=f"{len(g_suicide)}/6 시차 유의",
                    delta="인과관계 확인" if len(g_suicide) > 0 else "미확인",
                    delta_color="normal" if len(g_suicide) > 0 else "off",
                )

st.divider()

# ════════════════════════════════════════════════
# 1. Lag-Shifted Overlay (핵심 차트)
# ════════════════════════════════════════════════
st.subheader("1. 예측 오버레이 — 리스크 스코어를 미래로 이동")
st.caption(
    f"리스크 스코어({METRIC_LABEL})를 {lag_override}개월 앞으로 밀어 자살 사망과 겹침 → "
    "겹칠수록 예측력이 높다"
)

plot_df = merged_df.dropna(subset=[selected_metric, "suicide"]).copy()
if not plot_df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 자살 사망 (실선)
    fig.add_trace(
        go.Scatter(
            x=plot_df["year_month"],
            y=plot_df["suicide"],
            name="자살 사망자 수 (실제)",
            line=dict(color="#1e88e5", width=2.5),
            opacity=0.9,
        ),
        secondary_y=True,
    )

    # 리스크 스코어 — lag-shifted (미래로 이동)
    if lag_override > 0 and len(plot_df) > lag_override:
        shifted_risk = plot_df[selected_metric].iloc[:-lag_override].values
        shifted_months = plot_df["year_month"].iloc[lag_override:].values
    else:
        shifted_risk = plot_df[selected_metric].values
        shifted_months = plot_df["year_month"].values

    fig.add_trace(
        go.Scatter(
            x=shifted_months,
            y=shifted_risk,
            name=f"{METRIC_LABEL} ({lag_override}개월 후 이동)",
            line=dict(color="#e53935", width=2, dash="dot"),
            opacity=0.85,
        ),
        secondary_y=False,
    )

    # 원본 리스크 (연한 배경)
    fig.add_trace(
        go.Scatter(
            x=plot_df["year_month"],
            y=plot_df[selected_metric],
            name=f"{METRIC_LABEL} (원본)",
            line=dict(color="#e53935", width=1),
            opacity=0.2,
        ),
        secondary_y=False,
    )

    # 위기 이벤트 마커
    if show_events:
        all_months = plot_df["year_month"].tolist()
        for event_month, event_name, event_color in CRISIS_EVENTS:
            if event_month in all_months:
                fig.add_vline(
                    x=event_month,
                    line_width=1.5,
                    line_dash="dash",
                    line_color=event_color,
                    opacity=0.6,
                )
                fig.add_annotation(
                    x=event_month, y=1.05, yref="paper",
                    text=event_name, showarrow=False,
                    font=dict(size=9, color=event_color),
                )

    fig.update_layout(
        height=500,
        margin=dict(t=40, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=METRIC_LABEL, secondary_y=False, title_font_color="#e53935")
    fig.update_yaxes(title_text="자살 사망자 수 (명/월)", secondary_y=True, title_font_color="#1e88e5")

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════
# 2. 산점도 + 회귀선
# ════════════════════════════════════════════════
st.subheader("2. 산점도 — 리스크 스코어 vs 자살 사망 (최적 시차 적용)")

if not plot_df.empty and lag_override > 0:
    x_vals = plot_df[selected_metric].iloc[:-lag_override].values
    y_vals = plot_df["suicide"].iloc[lag_override:].values
    months = plot_df["year_month"].iloc[lag_override:].values
elif not plot_df.empty:
    x_vals = plot_df[selected_metric].values
    y_vals = plot_df["suicide"].values
    months = plot_df["year_month"].values
else:
    x_vals, y_vals, months = [], [], []

if len(x_vals) > 5:
    # 회귀선
    mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
    x_clean, y_clean = x_vals[mask], y_vals[mask]

    if len(x_clean) > 5:
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)

        # 연도별 색상
        years = [m[:4] for m in months[mask]]
        unique_years = sorted(set(years))

        # 색상 팔레트
        color_scale = [
            "#f44336", "#e91e63", "#9c27b0", "#673ab7", "#3f51b5",
            "#2196f3", "#00bcd4", "#009688", "#4caf50", "#8bc34a",
            "#cddc39", "#ffeb3b", "#ffc107", "#ff9800", "#ff5722",
            "#795548", "#607d8b", "#9e9e9e", "#455a64", "#263238",
        ]
        year_colors = {y: color_scale[i % len(color_scale)] for i, y in enumerate(unique_years)}

        fig_scatter = go.Figure()

        # 연도별 산점도
        for year in unique_years:
            year_mask = np.array([y == year for y in years])
            fig_scatter.add_trace(go.Scatter(
                x=x_clean[year_mask],
                y=y_clean[year_mask],
                mode="markers",
                name=year,
                marker=dict(
                    size=7,
                    color=year_colors[year],
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                text=[m for m, ym in zip(months[mask], years) if ym == year],
                hovertemplate="%{text}<br>" + METRIC_LABEL + ": %{x:.1f}<br>자살: %{y:.0f}<extra></extra>",
            ))

        # 회귀선
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

        fig_scatter.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode="lines",
            name=f"회귀선 (r={r_value:.3f})",
            line=dict(color="black", width=2, dash="dash"),
            showlegend=True,
        ))

        # 95% 신뢰구간
        n = len(x_clean)
        x_mean = x_clean.mean()
        se = std_err * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x_clean - x_mean)**2))
        t_val = stats.t.ppf(0.975, n - 2)

        fig_scatter.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([p(x_line) + t_val * se, (p(x_line) - t_val * se)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,0,0,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig_scatter.update_layout(
            height=500,
            xaxis_title=f"{METRIC_LABEL} (t - {lag_override}개월)",
            yaxis_title="자살 사망자 수 (t)",
            margin=dict(t=30, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5, font_size=10,
            ),
        )

        col_scatter, col_stats = st.columns([3, 1])
        with col_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_stats:
            st.markdown("#### 회귀 통계")
            st.markdown(f"- **r** = {r_value:.4f}")
            st.markdown(f"- **r²** = {r_value**2:.4f}")
            st.markdown(f"- **기울기** = {slope:.2f}")
            st.markdown(f"- **p-value** = {p_value:.2e}")
            st.markdown(f"- **N** = {n}")
            st.markdown(f"- **시차** = {lag_override}개월")
            st.markdown("---")
            st.markdown(
                f"리스크 스코어가 **1** 증가하면,\n\n"
                f"**{lag_override}개월 후** 월간 자살 사망자가\n\n"
                f"약 **{abs(slope):.1f}명** {'증가' if slope > 0 else '감소'}한다."
            )

st.divider()

# ════════════════════════════════════════════════
# 3. 위기 시기별 비교
# ════════════════════════════════════════════════
st.subheader("3. 위기 시기별 영향 비교")
st.caption("각 경제 위기 전후 리스크 스코어와 자살 사망 변화")

if not plot_df.empty:
    crisis_records = []
    all_months = plot_df["year_month"].tolist()

    for event_month, event_name, _ in CRISIS_EVENTS:
        if event_month not in all_months:
            continue
        idx = all_months.index(event_month)

        # 위기 전 6개월, 위기 후 12개월
        pre_start = max(0, idx - 6)
        post_end = min(len(plot_df), idx + 13)

        pre_risk = plot_df[selected_metric].iloc[pre_start:idx].mean()
        peak_risk = plot_df[selected_metric].iloc[max(0, idx-1):min(len(plot_df), idx+2)].max()

        # 자살: 위기 후 lag 기간 이후
        suicide_pre = plot_df["suicide"].iloc[pre_start:idx].mean()
        post_suicide_start = min(len(plot_df), idx + lag_override)
        post_suicide_end = min(len(plot_df), idx + lag_override + 6)
        if post_suicide_start < len(plot_df):
            suicide_post = plot_df["suicide"].iloc[post_suicide_start:post_suicide_end].mean()
        else:
            suicide_post = np.nan

        crisis_records.append({
            "위기": event_name,
            "시점": event_month,
            f"{METRIC_LABEL} (피크)": round(peak_risk, 1),
            f"{METRIC_LABEL} (전 6M 평균)": round(pre_risk, 1),
            "자살 (전 6M 평균)": round(suicide_pre, 0) if not np.isnan(suicide_pre) else "-",
            f"자살 (후 {lag_override}~{lag_override+6}M 평균)": round(suicide_post, 0) if not np.isnan(suicide_post) else "-",
        })

    if crisis_records:
        crisis_df = pd.DataFrame(crisis_records)
        st.dataframe(crisis_df, hide_index=True, use_container_width=True)

        # 위기별 바 차트
        valid_records = [r for r in crisis_records
                         if r.get(f"자살 (후 {lag_override}~{lag_override+6}M 평균)") != "-"
                         and r.get("자살 (전 6M 평균)") != "-"]

        if valid_records:
            fig_crisis = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f"{METRIC_LABEL} 피크", "자살 사망 변화율"],
                horizontal_spacing=0.15,
            )

            names = [r["위기"] for r in valid_records]
            peaks = [r[f"{METRIC_LABEL} (피크)"] for r in valid_records]
            suicide_changes = []
            for r in valid_records:
                pre = r["자살 (전 6M 평균)"]
                post = r[f"자살 (후 {lag_override}~{lag_override+6}M 평균)"]
                if pre > 0:
                    suicide_changes.append(round((post - pre) / pre * 100, 1))
                else:
                    suicide_changes.append(0)

            colors_peak = ["#e53935" if p > 20 else "#ff8a65" if p > 10 else "#90a4ae" for p in peaks]

            fig_crisis.add_trace(
                go.Bar(x=names, y=peaks, marker_color=colors_peak, showlegend=False,
                       text=[f"{v:.0f}" for v in peaks], textposition="outside"),
                row=1, col=1,
            )

            colors_change = ["#e53935" if c > 0 else "#1e88e5" for c in suicide_changes]
            fig_crisis.add_trace(
                go.Bar(x=names, y=suicide_changes, marker_color=colors_change, showlegend=False,
                       text=[f"{v:+.1f}%" for v in suicide_changes], textposition="outside"),
                row=1, col=2,
            )
            fig_crisis.add_hline(y=0, row=1, col=2, line_width=1, line_color="black")

            fig_crisis.update_layout(height=400, margin=dict(t=50, b=80))
            fig_crisis.update_xaxes(tickangle=30)

            st.plotly_chart(fig_crisis, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════
# 4. 롤링 상관 + 위기 마커
# ════════════════════════════════════════════════
st.subheader("4. 롤링 상관계수 — 관계의 시간적 안정성")
st.caption("12개월 이동 윈도우 상관계수. 위기 시기를 수직선으로 표시")

if not rolling_df.empty:
    filtered_rolling = rolling_df[
        (rolling_df["risk_metric"] == selected_metric)
        & (rolling_df["social_indicator"] == "suicide")
    ]

    if not filtered_rolling.empty:
        fig_roll = go.Figure()

        # 양/음 상관 구간 분리
        y_vals_roll = filtered_rolling["rolling_corr"].values
        x_months_roll = filtered_rolling["year_month"].values

        fig_roll.add_trace(go.Scatter(
            x=x_months_roll,
            y=y_vals_roll,
            mode="lines",
            line=dict(color="#1e88e5", width=2),
            name="롤링 상관계수",
        ))

        # 양의 상관 영역
        fig_roll.add_trace(go.Scatter(
            x=x_months_roll,
            y=np.clip(y_vals_roll, 0, 1),
            fill="tozeroy",
            fillcolor="rgba(229, 57, 53, 0.15)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # 음의 상관 영역
        fig_roll.add_trace(go.Scatter(
            x=x_months_roll,
            y=np.clip(y_vals_roll, -1, 0),
            fill="tozeroy",
            fillcolor="rgba(30, 136, 229, 0.15)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig_roll.add_hline(y=0, line_color="black", line_width=1, line_dash="dash")

        # 위기 이벤트
        if show_events:
            roll_months = set(x_months_roll)
            for event_month, event_name, event_color in CRISIS_EVENTS:
                if event_month in roll_months:
                    fig_roll.add_vline(
                        x=event_month,
                        line_width=1.5,
                        line_dash="dash",
                        line_color=event_color,
                        opacity=0.5,
                    )
                    fig_roll.add_annotation(
                        x=event_month, y=1.05, yref="paper",
                        text=event_name, showarrow=False,
                        font=dict(size=9, color=event_color),
                    )

        lag_val = filtered_rolling["lag"].iloc[0] if "lag" in filtered_rolling.columns else lag_override
        fig_roll.update_layout(
            height=400,
            yaxis_title="롤링 상관계수",
            yaxis_range=[-1, 1],
            margin=dict(t=30, b=60),
            hovermode="x unified",
        )

        st.plotly_chart(fig_roll, use_container_width=True)

        # 구간별 요약
        positive_pct = (y_vals_roll > 0).sum() / len(y_vals_roll) * 100
        strong_pct = (y_vals_roll > 0.3).sum() / len(y_vals_roll) * 100
        mean_corr = np.nanmean(y_vals_roll)

        c1, c2, c3 = st.columns(3)
        c1.metric("양의 상관 구간", f"{positive_pct:.0f}%")
        c2.metric("강한 양의 상관 (r>0.3)", f"{strong_pct:.0f}%")
        c3.metric("평균 롤링 상관", f"{mean_corr:.3f}")

st.divider()

# ════════════════════════════════════════════════
# 5. 교차상관 상세
# ════════════════════════════════════════════════
st.subheader("5. 교차상관 상세 — 시차별 상관계수")

if not xcorr_df.empty:
    # 3개 metric 모두 한 차트에
    fig_xcorr = go.Figure()
    colors_metric = {
        "risk_mean": "#e53935",
        "risk_max": "#ff9800",
        "warning_days": "#7b1fa2",
    }
    names_metric = {
        "risk_mean": "리스크 평균",
        "risk_max": "리스크 최대",
        "warning_days": "WARNING 일수",
    }

    for metric in ["risk_mean", "risk_max", "warning_days"]:
        subset = xcorr_df[
            (xcorr_df["risk_metric"] == metric)
            & (xcorr_df["social_indicator"] == "suicide")
        ]
        if not subset.empty:
            fig_xcorr.add_trace(go.Scatter(
                x=subset["lag"],
                y=subset["correlation"],
                mode="lines+markers",
                name=names_metric[metric],
                line=dict(color=colors_metric[metric], width=2),
                marker=dict(
                    size=[10 if s else 6 for s in subset["significant"]],
                    symbol=["diamond" if s else "circle" for s in subset["significant"]],
                ),
            ))

    fig_xcorr.add_hline(y=0, line_color="black", line_width=0.5)
    fig_xcorr.update_layout(
        height=400,
        xaxis_title="시차 (개월) — 리스크 스코어가 N개월 선행",
        yaxis_title="상관계수 (r)",
        yaxis_range=[-0.2, 0.5],
        margin=dict(t=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig_xcorr.update_xaxes(dtick=1)

    st.plotly_chart(fig_xcorr, use_container_width=True)
    st.caption("큰 마름모 = 통계적으로 유의 (Bonferroni 보정 p < 0.05)")

st.divider()

# ════════════════════════════════════════════════
# 6. Granger 인과성 테이블
# ════════════════════════════════════════════════
st.subheader("6. Granger 인과성 검정")

if not granger_df.empty:
    g_suicide = granger_df[granger_df["social_indicator"] == "suicide"]

    if not g_suicide.empty:
        # pivot: metric x lag
        pivot_data = []
        for metric in ["risk_mean", "risk_max", "warning_days"]:
            m_data = g_suicide[g_suicide["risk_metric"] == metric]
            for _, row in m_data.iterrows():
                sig_marker = " **" if row["significant"] else ""
                pivot_data.append({
                    "리스크 지표": names_metric.get(metric, metric),
                    f"lag {int(row['lag'])}": f"F={row['f_statistic']:.1f} (p={row['p_value']:.4f}){sig_marker}",
                })

        # 더 깔끔한 테이블
        cols_granger = st.columns(3)
        for idx, metric in enumerate(["risk_mean", "risk_max", "warning_days"]):
            m_data = g_suicide[g_suicide["risk_metric"] == metric].copy()
            if m_data.empty:
                continue
            with cols_granger[idx]:
                st.markdown(f"**{names_metric[metric]}**")
                display_g = m_data[["lag", "f_statistic", "p_value", "significant"]].copy()
                display_g.columns = ["시차", "F값", "p-value", "유의"]
                display_g["p-value"] = display_g["p-value"].apply(lambda x: f"{x:.4f}")
                display_g["F값"] = display_g["F값"].apply(lambda x: f"{x:.2f}")
                display_g["유의"] = display_g["유의"].apply(lambda x: "O" if x else "")
                st.dataframe(display_g, hide_index=True, use_container_width=True)

        sig_total = g_suicide["significant"].sum()
        total = len(g_suicide)
        if sig_total > 0:
            st.success(
                f"전체 {total}개 검정 중 **{sig_total}개**에서 유의미 — "
                f"리스크 스코어가 자살 사망을 Granger-cause 한다는 증거가 있습니다."
            )

st.divider()

# ════════════════════════════════════════════════
# 7. 연도별 히트맵
# ════════════════════════════════════════════════
st.subheader("7. 연도-월 히트맵")

if not plot_df.empty:
    heatmap_df = plot_df[["year_month", selected_metric, "suicide"]].dropna().copy()
    heatmap_df["year"] = heatmap_df["year_month"].str[:4]
    heatmap_df["month"] = heatmap_df["year_month"].str[5:7].astype(int)

    tab_risk, tab_suicide = st.tabs([f"{METRIC_LABEL} 히트맵", "자살 사망 히트맵"])

    for tab, col, cmap, label in [
        (tab_risk, selected_metric, "Reds", METRIC_LABEL),
        (tab_suicide, "suicide", "Blues", "자살 사망자 수"),
    ]:
        with tab:
            pivot_hm = heatmap_df.pivot_table(
                index="year", columns="month", values=col, aggfunc="first",
            )

            fig_hm = go.Figure(data=go.Heatmap(
                z=pivot_hm.values,
                x=[f"{m}월" for m in pivot_hm.columns],
                y=pivot_hm.index.tolist(),
                colorscale=cmap,
                text=pivot_hm.values.round(1),
                texttemplate="%{text}",
                textfont=dict(size=9),
                hoverongaps=False,
                colorbar_title=label,
            ))

            fig_hm.update_layout(
                height=max(300, len(pivot_hm) * 25),
                margin=dict(t=20, b=40),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════
# 8. v3 예측 모델 결과
# ════════════════════════════════════════════════
st.subheader("8. 자살 사망 예측 (v3 모델)")
st.caption("7변수 다변량 회귀: Property 분해 + 위기 피로감 + CPI + 가계대출 증가율")

# v3 모델 빌드
_props_df = _load_csv("monthly_properties.csv")
_loan_df = _load_csv("household_loan_raw.csv")
_cpi_df = _load_csv("cpi_raw.csv")

if not _props_df.empty and not _loan_df.empty and not _cpi_df.empty and not merged_df.empty:
    from sklearn.linear_model import LinearRegression

    _loan_df.columns = ["year_month", "household_loan"]
    _cpi_df.columns = _cpi_df.columns.tolist()[:1] + ["cpi"] if len(_cpi_df.columns) == 2 else _cpi_df.columns.tolist()
    if "cpi" not in _cpi_df.columns:
        _cpi_df = _cpi_df.rename(columns={_cpi_df.columns[1]: "cpi"})

    _model_df = _props_df.merge(merged_df[["year_month", "suicide"]], on="year_month", how="left")
    _model_df = _model_df.merge(_loan_df, on="year_month", how="left")
    _model_df = _model_df.merge(_cpi_df[["year_month", "cpi"]], on="year_month", how="left")
    _model_df["cpi_change"] = _model_df["cpi"].pct_change(fill_method=None) * 100
    _model_df["loan_yoy"] = _model_df["household_loan"].pct_change(12, fill_method=None) * 100

    _features = {
        "consec_warning_months": 4,
        "Property_055_contrib_mean": 7,
        "Property_070_contrib_mean": 7,
        "Property_113_contrib_mean": 7,
        "Property_110_contrib_mean": 9,
        "cpi_change": 6,
        "loan_yoy": 9,
    }

    _lag_cols = []
    for feat, lag in _features.items():
        cn = f"{feat}_lag{lag}"
        _model_df[cn] = _model_df[feat].shift(lag)
        _lag_cols.append(cn)

    _train = _model_df.dropna(subset=_lag_cols + ["suicide"])

    if len(_train) > 20:
        _reg = LinearRegression().fit(_train[_lag_cols], _train["suicide"])
        _y_pred = _reg.predict(_train[_lag_cols])
        _residual_std = (_train["suicide"] - _y_pred).std()
        _r2 = 1 - np.sum((_train["suicide"] - _y_pred) ** 2) / np.sum((_train["suicide"] - _train["suicide"].mean()) ** 2)

        # 전체 기간 예측 (train + future)
        _all_pred = _model_df.dropna(subset=_lag_cols).copy()
        _all_pred["predicted"] = _reg.predict(_all_pred[_lag_cols])
        _all_pred["ci_lo"] = _all_pred["predicted"] - 1.96 * _residual_std
        _all_pred["ci_hi"] = _all_pred["predicted"] + 1.96 * _residual_std

        # 모델 성능 카드
        _n_features = len(_features)
        _adj_r2 = 1 - (1 - _r2) * (len(_train) - 1) / (len(_train) - _n_features - 1)
        _mae = np.mean(np.abs(_train["suicide"] - _y_pred))

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("R²", f"{_r2:.3f}")
        mc2.metric("adj. R²", f"{_adj_r2:.3f}")
        mc3.metric("MAE", f"{_mae:.0f}명")
        mc4.metric("변수 수", f"{_n_features}개")

        # 예측 vs 실제 차트
        fig_pred = go.Figure()

        # 실제 자살 사망
        _actual = _all_pred[_all_pred["suicide"].notna()]
        fig_pred.add_trace(go.Scatter(
            x=_actual["year_month"], y=_actual["suicide"],
            name="실제", mode="lines",
            line=dict(color="#1e88e5", width=2.5),
        ))

        # 예측값 (전체)
        fig_pred.add_trace(go.Scatter(
            x=_all_pred["year_month"], y=_all_pred["predicted"],
            name="v3 예측", mode="lines",
            line=dict(color="#e53935", width=2, dash="dot"),
        ))

        # 95% CI (미래 구간만 강조)
        _future = _all_pred[_all_pred["suicide"].isna()]
        if not _future.empty:
            fig_pred.add_trace(go.Scatter(
                x=pd.concat([_future["year_month"], _future["year_month"][::-1]]),
                y=pd.concat([_future["ci_hi"], _future["ci_lo"][::-1]]),
                fill="toself", fillcolor="rgba(229, 57, 53, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=True, name="95% 신뢰구간",
                hoverinfo="skip",
            ))

        # 2024 평균선
        _avg_2024 = _model_df[_model_df["year_month"].str.startswith("2024")]["suicide"].mean()
        if not np.isnan(_avg_2024):
            fig_pred.add_hline(
                y=_avg_2024, line_dash="dash", line_color="gray", line_width=1,
                annotation_text=f"2024 평균 ({_avg_2024:.0f}명)",
                annotation_position="bottom right",
            )

        fig_pred.update_layout(
            height=500,
            yaxis_title="자살 사망자 수 (명/월)",
            margin=dict(t=30, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="x unified",
        )
        # 최근 3년만 표시
        fig_pred.update_xaxes(range=["2023-01", "2026-12"])

        st.plotly_chart(fig_pred, use_container_width=True)

        # 예측 테이블
        _future_display = _all_pred[_all_pred["year_month"] >= "2025-01"][
            ["year_month", "predicted", "ci_lo", "ci_hi", "suicide"]
        ].copy()
        _future_display.columns = ["월", "예측", "하한", "상한", "실제"]
        _future_display["예측"] = _future_display["예측"].apply(lambda x: f"{x:,.0f}명")
        _future_display["하한"] = _future_display["하한"].apply(lambda x: f"{x:,.0f}명")
        _future_display["상한"] = _future_display["상한"].apply(lambda x: f"{x:,.0f}명")
        _future_display["실제"] = _future_display["실제"].apply(
            lambda x: f"{x:,.0f}명" if pd.notna(x) else "미발표"
        )

        with st.expander("월별 예측 상세"):
            st.dataframe(_future_display, hide_index=True, use_container_width=True)

        # 변수 기여도
        with st.expander("변수별 기여도 (β 계수)"):
            _coef_data = []
            for feat, coef in zip(_features.keys(), _reg.coef_):
                _feat_labels = {
                    "consec_warning_months": "연속 WARNING 월수",
                    "Property_055_contrib_mean": "P_055 (경제불안)",
                    "Property_070_contrib_mean": "P_070 (위기 전이)",
                    "Property_113_contrib_mean": "P_113 (위기 확산)",
                    "Property_110_contrib_mean": "P_110 (레버리지)",
                    "cpi_change": "CPI 변화율",
                    "loan_yoy": "가계대출 증가율 (YoY)",
                }
                _coef_data.append({
                    "변수": _feat_labels.get(feat, feat),
                    "시차": f"{_features[feat]}개월",
                    "계수 (β)": f"{coef:+.1f}",
                    "해석": f"1단위 증가 → {abs(coef):.0f}명 {'증가' if coef > 0 else '감소'}",
                })
            st.dataframe(pd.DataFrame(_coef_data), hide_index=True, use_container_width=True)
    else:
        st.info("모델 학습 데이터가 충분하지 않습니다.")
else:
    st.info("예측 모델 데이터가 없습니다. 필요 파일: monthly_properties.csv, household_loan_raw.csv, cpi_raw.csv")

st.divider()

# ════════════════════════════════════════════════
# 해석 노트
# ════════════════════════════════════════════════
with st.expander("분석 해석 노트"):
    st.markdown("""
### 핵심 발견

**경제 위기 리스크 스코어는 4~7개월 후의 자살 사망을 통계적으로 유의미하게 예측한다.**

- Pearson 교차상관: r = 0.39~0.42, Bonferroni 보정 후 p < 0.0001
- Granger 인과성: lag 1~6에서 유의 (리스크 → 자살 방향)
- 롤링 상관: 2010년대 이후 안정적으로 양의 상관

### 정신건강 지표의 사각지대

정신질환 통계는 병원 진단 기반 — "도움을 구한 사람"만 카운트.
자살 통계는 사망신고서 기반 — "도움을 구하지 못한 사람"까지 포함.

한국 사회에서:
- 정신과 방문에 대한 사회적 낙인이 강하다
- 경제 위기에 가장 취약한 계층이 정신과를 가장 안 가는 계층이다
- 따라서 경제 위기 → 정신건강 악화의 경로가 정신질환 통계에 포착되지 않는다
- 자살은 회피 불가능한 최종 결과이므로, 실제 정신건강 위기를 더 정확히 반영한다

### 정책적 함의

경제 위기 시 정신건강 대응을 정신과 방문 통계로 판단하면
실제 위기의 심각성을 과소평가할 위험이 있다.
리스크 스코어가 WARNING 수준에 도달하면,
4~7개월 후 자살 사망 증가에 대비한 선제적 정신건강 지원이 필요하다.

### 데이터 출처
- 리스크 스코어: BlackEvent-Predictor Model B (2006~2026, 7,329일)
- 자살 사망: KOSIS 사망원인(104항목)/성/월별 사망자수 (2006~2024, 228개월)
- 분석 방법: Pearson 교차상관 + Bonferroni 보정, Granger F-test, 12개월 롤링 상관
    """)

# ── Footer ──
st.caption("데이터: KOSIS 국가통계포털 | 분석: BlackEvent-Predictor Module 9")
