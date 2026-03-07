"""
BlackEvent Predictor — Web Dashboard
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── 페이지 설정 ──
st.set_page_config(
    page_title="BlackEvent Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUTS = ROOT / "data" / "outputs"
COOKIE_FILE = ROOT / "config" / "bigkinds_cookie.txt"

# ── 사이드바: 쿠키 상태 ──
with st.sidebar:
    st.title("BlackEvent Predictor")
    st.divider()

    # 쿠키 상태
    if COOKIE_FILE.exists() and COOKIE_FILE.read_text(encoding="utf-8").strip():
        cookie_text = COOKIE_FILE.read_text(encoding="utf-8").strip()
        if len(cookie_text) > 50:
            st.success("Cookie: 설정됨")
        else:
            st.warning("Cookie: 내용 부족")
    else:
        st.error("Cookie: 미설정")

    st.caption("BigKinds 쿠키는 Collect 페이지에서 설정")

# ── 메인 홈 ──
st.header("BlackEvent Predictor")
st.markdown("한국 금융시장 Black Event(급락) 예측 시스템")

st.divider()

# 최신 스코어 표시
daily_csv = OUTPUTS / "daily_scores.csv"
if daily_csv.exists():
    df = pd.read_csv(daily_csv)
    if len(df) > 0:
        latest = df.iloc[-1]
        score = latest["risk_score"]
        status = latest["status"]
        date = latest["date"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("최신 리스크 스코어", f"{score:.2f}/100")
        with col2:
            st.metric("상태", status)
        with col3:
            st.metric("기준일", date)

        if status == "WARNING":
            st.warning(
                f"**WARNING**: 리스크 스코어 {score:.2f}이 임계값(18.52)을 초과했습니다."
            )
        else:
            st.info(f"NORMAL: 리스크 스코어 {score:.2f} — 임계값(18.52) 이하")
else:
    st.info("아직 산출된 스코어가 없습니다. Collect 페이지에서 파이프라인을 실행하세요.")

st.divider()

st.subheader("프로젝트 요약")
st.markdown("""
- **목표**: 뉴스 텍스트에서 금융시장 급락(BlackEvent) 신호를 감지
- **방법**: 12개 핵심 성질의 빈도 변화를 추적하여 리스크 스코어 산출
- **모델**: Model B (가중 스코어링) — 테스트 AUC 0.9072, Precision 1.0
- **임계값**: 18.52 이상 시 WARNING

**파이프라인**: BigKinds 뉴스 수집 → 임베딩 태깅 → 리스크 스코어링
""")

st.subheader("페이지 안내")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Dashboard**\n\n리스크 게이지, 성질 기여도, 일별 추이 차트")
with col2:
    st.markdown("**Collect**\n\nBigKinds 쿠키 설정, 뉴스 수집·태깅·스코어링 실행")
with col3:
    st.markdown("**History**\n\n과거 BlackEvent 분석, 모델 성능, Lead Time")
with col4:
    st.markdown("**Evaluator**\n\nLLM 대화형 평가자 — 태깅/스코어링 평가, 인사이트 축적")
