"""
Evaluator — LLM 기반 대화형 평가자

태깅과 스코어링을 평가하고, 대화를 통해 인사이트를 축적한다.
평가자는 기계의 언어(Property 번호, 비율, 패턴)로 사고한다.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluator.context import build_context
from src.evaluator.store import (
    load_memory,
    save_memory,
    load_insights,
    save_insight,
    save_judgment,
    load_judgments,
    save_session,
    EVALUATOR_DIR,
)
from src.utils import load_config, get_path


# ── 페이지 설정 ──
st.set_page_config(page_title="Evaluator", page_icon="@", layout="wide")

# ── 모바일 최적화 CSS ──
st.markdown("""
<style>
/* 모바일: 사이드바 기본 접힘, 메인 패딩 축소 */
@media (max-width: 768px) {
    .stMainBlockContainer { padding: 0.5rem 0.8rem !important; }
    .stChatMessage { padding: 0.4rem 0.6rem !important; }
    .stMetric { padding: 0.3rem !important; }
    .stMetric label { font-size: 0.75rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    h1 { font-size: 1.4rem !important; }
    .stChatInput { position: sticky; bottom: 0; z-index: 999; }
    section[data-testid="stSidebar"] { min-width: 260px !important; max-width: 260px !important; }
    .stTextArea textarea { font-size: 0.85rem !important; }
}

/* 공통: 채팅 영역 스크롤 */
.stChatMessage p { margin-bottom: 0.3rem; }

/* 상태 바 compact */
div[data-testid="stMetric"] { background: rgba(28, 131, 225, 0.05); border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


DEFAULT_ROLE = """You are the Evaluator of BlackEvent Predictor.
Your role: evaluate the quality of property tagging and risk scoring.
You think in MACHINE LANGUAGE: Property numbers, ratios, patterns.
You do NOT categorize by human topics (economy, war, real estate).
You observe the "texture" (결) of how properties move.

CRITICAL RULES:
- Use Property IDs (P_083, P_055, etc.), not human labels
- When you discover a pattern, state your confidence (0-1)
- When past experience contradicts new data, revise openly
- You can simulate: "if this pattern continues, based on [historical analog]..."
- You accumulate experience. Your memory persists across sessions.

Respond in Korean. Be concise. Think in patterns, not narratives."""


def get_api_key():
    """API 키를 .env 또는 환경변수에서 로드한다."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").strip().split("\n"):
                if line.startswith("ANTHROPIC_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return key


def build_system_prompt(context: dict, memory: str, insights: list, custom_role: str = None) -> str:
    """평가자의 시스템 프롬프트를 구성한다."""
    summary = context["summary"]
    rising = summary.get("rising_properties", [])
    falling = summary.get("falling_properties", [])

    rising_str = "\n".join(
        f"  {p['property']}: {p['current']}% (v={p['velocity']:+.2f})"
        for p in rising[:10]
    )
    falling_str = "\n".join(
        f"  {p['property']}: {p['current']}% (v={p['velocity']:+.2f})"
        for p in falling[:10]
    )

    scores = context.get("score_history", [])
    score_str = "\n".join(
        f"  {s['date']}: score={s['risk_score']:.1f} [{s['status']}] top={s['top_contributor']}"
        for s in scores[-7:]
    )

    cross = context.get("cross_patterns", [])
    cross_str = "\n".join(
        f"  {c['pair'][0]} <-> {c['pair'][1]}: {c['correlation']:+.3f} ({c['type']})"
        for c in cross[:5]
    )

    recent_insights = insights[-10:] if insights else []
    insights_str = "\n".join(
        f"  [{i.get('type','?')}] {i.get('observation','')[:120]}"
        for i in recent_insights
    )

    role_block = custom_role if custom_role else "(no role defined)"

    dates = summary.get("dates_covered", ["?", "?"])
    date_range = f"{dates[0]} ~ {dates[-1]}" if dates else "no data"

    return f"""{role_block}

=== CURRENT DATA ({date_range}) ===

Rising properties:
{rising_str or '  (none)'}

Falling properties:
{falling_str or '  (none)'}

High density dates (sig15+>30%): {summary.get('high_density_dates', [])}

Recent scores:
{score_str or '  (none)'}

Cross patterns:
{cross_str or '  (none)'}

=== YOUR ACCUMULATED MEMORY ===
{memory if memory else '(first session — no prior experience)'}

=== RECENT INSIGHTS ===
{insights_str if insights_str else '(none yet)'}"""


@st.cache_resource
def load_evaluator_context():
    """평가자 컨텍스트를 로드한다 (캐싱)."""
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")
    daily_dir = processed_path / "corpus_daily"
    return build_context(processed_path, outputs_path, daily_dir)


def call_claude(messages: list, system: str, api_key: str) -> str:
    """Claude API를 호출한다."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system,
        messages=messages,
    )
    return response.content[0].text


# ── 사이드바 ──
with st.sidebar:
    st.title("Evaluator")
    st.caption("LLM 대화형 평가자")
    st.divider()

    # Status compact
    memory = load_memory()
    all_insights = load_insights()
    all_judgments = load_judgments()

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Mem", f"{len(memory.split(chr(10)))}L" if memory else "0")
    with sc2:
        st.metric("Ins", len(all_insights))
    with sc3:
        st.metric("Jdg", len(all_judgments))

    st.divider()

    # Actions
    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button("Reload", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    with ac2:
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if st.button("Save Session", use_container_width=True):
        if "messages" in st.session_state and st.session_state.messages:
            date_str = datetime.now().strftime("%Y-%m-%d")
            session_content = f"# Evaluator Session {date_str}\n\n"
            for msg in st.session_state.messages:
                role = "USER" if msg["role"] == "user" else "EVALUATOR"
                session_content += f"## {role}\n{msg['content']}\n\n"
            save_session(date_str, session_content)
            st.success(f"Saved: {date_str}")

    st.divider()

    # System prompt - in expander for mobile
    with st.expander("System Prompt", expanded=False):
        if "custom_role" not in st.session_state:
            st.session_state.custom_role = DEFAULT_ROLE

        custom_role = st.text_area(
            "Role & Rules",
            value=st.session_state.custom_role,
            height=200,
            key="role_input",
            label_visibility="collapsed",
        )
        st.session_state.custom_role = custom_role

        if st.button("Reset Default", use_container_width=True):
            st.session_state.custom_role = DEFAULT_ROLE
            st.rerun()

    # Insight quick-save - in expander for mobile
    with st.expander("Quick Insight", expanded=False):
        insight_type = st.selectbox("Type", ["pattern", "anomaly", "threshold", "correlation", "revision"])
        insight_props = st.text_input("Properties", placeholder="Property_083, Property_055")
        insight_obs = st.text_area("Observation", placeholder="P_083 crossed 10%...", height=80)
        insight_conf = st.slider("Confidence", 0.0, 1.0, 0.5, 0.1)

        if st.button("Save Insight", use_container_width=True):
            if insight_obs:
                props = [p.strip() for p in insight_props.split(",") if p.strip()]
                save_insight({
                    "type": insight_type,
                    "properties": props,
                    "observation": insight_obs,
                    "confidence": insight_conf,
                    "basis": "manual input via evaluator UI",
                })
                st.success("Saved!")


# ── 메인 ──
api_key = get_api_key()
if not api_key:
    st.error("ANTHROPIC_API_KEY not found. Set it in .env file.")
    st.stop()

# Load context
context = load_evaluator_context()
memory = load_memory()
insights = load_insights()
custom_role = st.session_state.get("custom_role", DEFAULT_ROLE)
system_prompt = build_system_prompt(context, memory, insights, custom_role=custom_role)

# Summary bar (responsive: 4 cols desktop, 2 cols mobile via CSS)
summary = context["summary"]
col1, col2, col3, col4 = st.columns(4)
with col1:
    dates = summary["dates_covered"]
    if dates:
        st.metric("Range", f"{dates[0][-5:]}~{dates[-1][-5:]}")
    else:
        st.metric("Range", "N/A")
with col2:
    st.metric("Props", summary["total_properties_observed"])
with col3:
    st.metric("Rising", len(summary["rising_properties"]))
with col4:
    scores = context.get("score_history", [])
    if scores:
        latest = scores[-1]
        st.metric("Score", f"{latest['risk_score']:.1f}")
    else:
        st.metric("Score", "N/A")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("평가자에게 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("..."):
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            response = call_claude(api_messages, system_prompt, api_key)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
