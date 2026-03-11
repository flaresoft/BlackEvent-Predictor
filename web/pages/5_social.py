"""
Social Impact — 리스크 스코어 vs 사회 지표 (자살률, 범죄율, 실업률)
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SOCIAL_OUTPUT = ROOT / "data" / "outputs" / "social"

st.set_page_config(page_title="Social Impact", page_icon="🔬", layout="wide")
st.header("Social Impact Analysis")
st.caption("경제 위기 리스크 스코어와 사회 지표의 상관관계 분석")

INDICATOR_LABELS = {
    "suicide": "자살 사망자 수",
    "mental_disorder": "정신행동장애 사망자 수",
    "substance_disorder": "정신활성물질 장애 사망자 수",
    "crime": "범죄 발생건수",
    "unemployment": "실업률 (%)",
}

METRIC_LABELS = {
    "risk_mean": "리스크 평균",
    "risk_max": "리스크 최대",
    "warning_days": "WARNING 일수",
}


def _load_csv(filename: str) -> pd.DataFrame:
    path = SOCIAL_OUTPUT / filename
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


# 데이터 로드
merged_df = _load_csv("merged_monthly.csv")
summary_df = _load_csv("summary.csv")
xcorr_df = _load_csv("cross_correlation.csv")
granger_df = _load_csv("granger.csv")
rolling_df = _load_csv("rolling.csv")

if merged_df.empty:
    st.warning(
        "분석 데이터가 없습니다. Module 9을 먼저 실행하세요:\n\n"
        "```bash\npython -m src.module9_social.run\n```"
    )
    st.stop()

# 사회 지표 감지
social_cols = [c for c in merged_df.columns
               if c not in ["year_month", "risk_mean", "risk_max",
                            "risk_std", "warning_days", "n_days"]]

if not social_cols:
    st.warning("병합된 데이터에 사회 지표 컬럼이 없습니다.")
    st.stop()

# ── 컨트롤 ──
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    selected_indicator = st.selectbox(
        "사회 지표",
        social_cols,
        format_func=lambda x: INDICATOR_LABELS.get(x, x),
    )
with col_ctrl2:
    selected_metric = st.selectbox(
        "리스크 지표",
        ["risk_mean", "risk_max", "warning_days"],
        format_func=lambda x: METRIC_LABELS.get(x, x),
    )

st.divider()

# ── 1. 요약 카드 ──
if not summary_df.empty:
    st.subheader("핵심 발견")

    sig_results = summary_df[summary_df["significant"]]
    if not sig_results.empty:
        cols = st.columns(min(len(sig_results), 3))
        for idx, (_, row) in enumerate(sig_results.iterrows()):
            with cols[idx % 3]:
                metric_label = METRIC_LABELS.get(row["risk_metric"], row["risk_metric"])
                indicator_label = INDICATOR_LABELS.get(row["social_indicator"], row["social_indicator"])
                direction = "양의" if row["best_correlation"] > 0 else "음의"
                st.metric(
                    label=f"{metric_label} → {indicator_label}",
                    value=f"r = {row['best_correlation']:.3f}",
                    delta=f"lag {int(row['best_lag'])}개월, {direction} 상관",
                )
    else:
        st.info("유의미한 상관관계가 발견되지 않았습니다. 데이터 기간이 짧을 수 있습니다.")

    st.divider()

# ── 2. 시계열 비교 ──
st.subheader("시계열 비교")

plot_df = merged_df.dropna(subset=[selected_metric, selected_indicator])

if not plot_df.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    metric_label = METRIC_LABELS.get(selected_metric, selected_metric)
    indicator_label = INDICATOR_LABELS.get(selected_indicator, selected_indicator)

    fig.add_trace(
        go.Scatter(
            x=plot_df["year_month"], y=plot_df[selected_metric],
            name=metric_label, line=dict(color="#e53935", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["year_month"], y=plot_df[selected_indicator],
            name=indicator_label, line=dict(color="#1e88e5", width=2),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        height=400,
        margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text=metric_label, secondary_y=False)
    fig.update_yaxes(title_text=indicator_label, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("표시할 데이터가 없습니다.")

st.divider()

# ── 3. 교차상관 ──
st.subheader("교차상관 (시차별 상관계수)")

if not xcorr_df.empty:
    filtered_xcorr = xcorr_df[
        (xcorr_df["risk_metric"] == selected_metric)
        & (xcorr_df["social_indicator"] == selected_indicator)
    ]

    if not filtered_xcorr.empty:
        fig_xcorr = go.Figure()
        colors = [
            "#e53935" if s else "#90a4ae"
            for s in filtered_xcorr["significant"]
        ]
        fig_xcorr.add_trace(go.Bar(
            x=filtered_xcorr["lag"],
            y=filtered_xcorr["correlation"],
            marker_color=colors,
            text=[f"{r:.3f}" for r in filtered_xcorr["correlation"]],
            textposition="outside",
        ))
        fig_xcorr.add_hline(y=0, line_color="black", line_width=0.5)
        fig_xcorr.update_layout(
            xaxis_title="시차 (개월) — 리스크 스코어가 N개월 선행",
            yaxis_title="상관계수 (r)",
            yaxis_range=[-1, 1],
            height=350,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_xcorr, use_container_width=True)
        st.caption("빨간 막대 = 통계적으로 유의미 (Bonferroni 보정 p < 0.05)")
    else:
        st.info("선택한 조합의 교차상관 데이터가 없습니다.")
else:
    st.info("교차상관 분석 결과가 없습니다.")

st.divider()

# ── 4. Granger 인과성 ──
st.subheader("Granger 인과성 검정")

if not granger_df.empty:
    filtered_granger = granger_df[
        (granger_df["risk_metric"] == selected_metric)
        & (granger_df["social_indicator"] == selected_indicator)
    ]

    if not filtered_granger.empty:
        display_cols = ["lag", "f_statistic", "p_value", "significant"]
        display_df = filtered_granger[display_cols].copy()
        display_df.columns = ["시차 (개월)", "F-통계량", "p-value", "유의미"]
        display_df["p-value"] = display_df["p-value"].apply(lambda x: f"{x:.4f}")
        display_df["F-통계량"] = display_df["F-통계량"].apply(lambda x: f"{x:.2f}")
        display_df["유의미"] = display_df["유의미"].apply(lambda x: "O" if x else "")

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        sig_count = filtered_granger["significant"].sum()
        if sig_count > 0:
            st.success(
                f"리스크 스코어가 {INDICATOR_LABELS.get(selected_indicator, selected_indicator)}를 "
                f"Granger-cause 한다는 증거가 {sig_count}개 시차에서 발견되었습니다."
            )
        else:
            st.info("Granger 인과성이 유의미하지 않습니다.")

        if filtered_granger["cause_diffed"].any():
            st.caption("비정상 시계열에 1차 차분이 적용되었습니다.")
    else:
        st.info("선택한 조합의 Granger 검정 데이터가 없습니다.")
else:
    st.info("Granger 검정 결과가 없습니다.")

st.divider()

# ── 5. 롤링 상관 ──
st.subheader("롤링 상관계수 (관계의 시간적 안정성)")

if not rolling_df.empty:
    filtered_rolling = rolling_df[
        (rolling_df["risk_metric"] == selected_metric)
        & (rolling_df["social_indicator"] == selected_indicator)
    ]

    if not filtered_rolling.empty:
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=filtered_rolling["year_month"],
            y=filtered_rolling["rolling_corr"],
            mode="lines",
            line=dict(color="#1e88e5", width=2),
            fill="tozeroy",
            fillcolor="rgba(30, 136, 229, 0.2)",
        ))
        fig_roll.add_hline(y=0, line_color="black", line_width=0.5, line_dash="dash")
        window = filtered_rolling["window"].iloc[0] if "window" in filtered_rolling.columns else "?"
        lag = filtered_rolling["lag"].iloc[0] if "lag" in filtered_rolling.columns else 0
        fig_roll.update_layout(
            xaxis_title="기간",
            yaxis_title="롤링 상관계수",
            yaxis_range=[-1, 1],
            height=350,
            margin=dict(t=20, b=40),
            title_text=f"윈도우 {window}개월, lag {lag}개월",
        )
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("선택한 조합의 롤링 상관 데이터가 없습니다.")
else:
    st.info("롤링 상관 분석 결과가 없습니다.")

st.divider()

# ── 6. 전체 요약 테이블 ──
st.subheader("전체 분석 요약")

if not summary_df.empty:
    display = summary_df.copy()
    display["risk_metric"] = display["risk_metric"].map(
        lambda x: METRIC_LABELS.get(x, x)
    )
    display["social_indicator"] = display["social_indicator"].map(
        lambda x: INDICATOR_LABELS.get(x, x)
    )
    display.columns = [
        "리스크 지표", "사회 지표", "최적 시차 (개월)",
        "상관계수", "p-value (보정)", "유의미",
    ]
    display["상관계수"] = display["상관계수"].apply(lambda x: f"{x:.3f}")
    display["p-value (보정)"] = display["p-value (보정)"].apply(lambda x: f"{x:.4f}")
    display["유의미"] = display["유의미"].apply(lambda x: "O" if x else "")

    st.dataframe(display, hide_index=True, use_container_width=True)

# ── 데이터 정보 ──
with st.expander("데이터 정보"):
    st.write(f"- 병합 데이터: {len(merged_df)}개월")
    st.write(f"- 사회 지표: {', '.join(INDICATOR_LABELS.get(c, c) for c in social_cols)}")
    for col in social_cols:
        n_valid = merged_df[col].notna().sum()
        st.write(f"  - {INDICATOR_LABELS.get(col, col)}: {n_valid}개월 데이터")

    st.markdown("---")
    st.markdown(
        "**데이터 출처**: [KOSIS 국가통계포털](https://kosis.kr/)\n\n"
        "**분석 방법**:\n"
        "- 교차상관: Pearson 상관 + Bonferroni 보정\n"
        "- Granger 인과성: F-test (ADF 정상성 검정 후 필요 시 차분)\n"
        "- 롤링 상관: 12개월 이동 윈도우\n\n"
        "**실행**: `python -m src.module9_social.run`"
    )
