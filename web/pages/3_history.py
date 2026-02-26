"""
History — 과거 BlackEvent 분석, 모델 성능, Lead Time
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "data" / "outputs"

st.set_page_config(page_title="History", page_icon="📈", layout="wide")
st.header("History & Model Performance")

# ── 1. BlackEvent vs Control 분포 ──
st.subheader("과거 BlackEvent vs Control 스코어 분포")

risk_csv = OUTPUTS / "risk_scores.csv"
if risk_csv.exists():
    risk_df = pd.read_csv(risk_csv)

    be_df = risk_df[risk_df["actual_label"] == 1]
    ctrl_df = risk_df[risk_df["actual_label"] == 0]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=ctrl_df["model_b_risk_score"],
        name="Control",
        marker_color="#66bb6a",
        opacity=0.7,
        nbinsx=40,
    ))
    fig_hist.add_trace(go.Histogram(
        x=be_df["model_b_risk_score"],
        name="BlackEvent",
        marker_color="#ef5350",
        opacity=0.7,
        nbinsx=40,
    ))
    fig_hist.add_vline(
        x=18.52,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold (18.52)",
    )
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Risk Score (Model B)",
        yaxis_title="Count",
        height=400,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_hist, width="stretch")

    # 통계
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**BlackEvent 스코어 통계**")
        st.write(f"- 건수: {len(be_df)}")
        st.write(f"- 평균: {be_df['model_b_risk_score'].mean():.2f}")
        st.write(f"- 중앙값: {be_df['model_b_risk_score'].median():.2f}")
        st.write(f"- 최대: {be_df['model_b_risk_score'].max():.2f}")
        detected = (be_df["model_b_risk_score"] >= 18.52).sum()
        st.write(f"- 탐지율 (≥18.52): {detected}/{len(be_df)} ({detected/len(be_df)*100:.1f}%)")
    with col2:
        st.markdown("**Control 스코어 통계**")
        st.write(f"- 건수: {len(ctrl_df)}")
        st.write(f"- 평균: {ctrl_df['model_b_risk_score'].mean():.2f}")
        st.write(f"- 중앙값: {ctrl_df['model_b_risk_score'].median():.2f}")
        st.write(f"- 최대: {ctrl_df['model_b_risk_score'].max():.2f}")
        false_alarm = (ctrl_df["model_b_risk_score"] >= 18.52).sum()
        st.write(f"- 오경보율 (≥18.52): {false_alarm}/{len(ctrl_df)} ({false_alarm/len(ctrl_df)*100:.1f}%)")
else:
    st.info("risk_scores.csv가 없습니다.")

st.divider()

# ── 2. 모델 성능 ──
st.subheader("Model B 성능 지표")

col_train, col_test = st.columns(2)

with col_train:
    st.markdown("**학습 데이터**")
    train_metrics = {
        "AUC-ROC": 0.6855,
        "Best F1": 0.4973,
        "Best Threshold": 18.52,
        "Precision": 0.5169,
        "Recall": 0.4792,
    }
    st.dataframe(
        pd.DataFrame(train_metrics.items(), columns=["지표", "값"]),
        hide_index=True,
        width="stretch",
    )

with col_test:
    st.markdown("**테스트 데이터**")
    test_metrics = {
        "AUC-ROC": 0.9072,
        "Threshold": 18.52,
        "Precision": 1.0000,
        "Recall": 0.5778,
        "F1 Score": 0.7324,
        "FPR": 0.0000,
    }
    st.dataframe(
        pd.DataFrame(test_metrics.items(), columns=["지표", "값"]),
        hide_index=True,
        width="stretch",
    )

st.divider()

# ── 3. Lead Time 분석 ──
st.subheader("Lead Time 분석")

lead_csv = OUTPUTS / "lead_time_analysis.csv"
if lead_csv.exists():
    lead_df = pd.read_csv(lead_csv)

    # 이벤트별 detected 비율
    pivot = lead_df.pivot_table(
        index="event_id", columns="lead_days", values="detected", aggfunc="first"
    )
    lead_days_sorted = sorted(lead_df["lead_days"].unique())

    # 각 lead_days별 detection rate
    detection_rates = []
    for ld in lead_days_sorted:
        subset = lead_df[lead_df["lead_days"] == ld]
        rate = subset["detected"].mean() * 100
        detection_rates.append({"Lead Days": ld, "Detection Rate (%)": rate})

    dr_df = pd.DataFrame(detection_rates)

    fig_lead = go.Figure(go.Bar(
        x=dr_df["Lead Days"],
        y=dr_df["Detection Rate (%)"],
        marker_color=["#ef5350" if d == 0 else "#42a5f5" for d in dr_df["Lead Days"]],
        text=[f"{r:.0f}%" for r in dr_df["Detection Rate (%)"]],
        textposition="outside",
    ))
    fig_lead.update_layout(
        xaxis_title="Lead Days (이벤트 전 일수)",
        yaxis_title="Detection Rate (%)",
        yaxis_range=[0, 100],
        height=350,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_lead, width="stretch")

    # 이벤트별 스코어 변화 (최근 10개)
    recent_events = sorted(lead_df["event_id"].unique())[-10:]
    recent_df = lead_df[lead_df["event_id"].isin(recent_events)]

    fig_event = px.line(
        recent_df,
        x="lead_days",
        y="risk_score",
        color="event_id",
        markers=True,
        title="최근 이벤트별 Lead Time 스코어",
    )
    fig_event.add_hline(y=18.52, line_dash="dash", line_color="red")
    fig_event.update_layout(
        xaxis_title="Lead Days",
        yaxis_title="Risk Score",
        height=400,
        xaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_event, width="stretch")
else:
    st.info("lead_time_analysis.csv가 없습니다.")

st.divider()

# ── 4. 기존 시각화 이미지 ──
st.subheader("모델 시각화")

images = {
    "Confusion Matrix (Model A)": OUTPUTS / "confusion_matrix_model_a.png",
    "Confusion Matrix (Model B)": OUTPUTS / "confusion_matrix_model_b.png",
    "ROC Curve (Model B)": OUTPUTS / "roc_curve_model_b.png",
    "Precision-Recall (Model B)": OUTPUTS / "precision_recall_model_b.png",
}

cols = st.columns(2)
idx = 0
for title, img_path in images.items():
    if img_path.exists():
        with cols[idx % 2]:
            st.markdown(f"**{title}**")
            st.image(str(img_path), width="stretch")
        idx += 1

if idx == 0:
    st.info("시각화 이미지가 없습니다.")
