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

TOOLS — you have tools to persist knowledge and search the web:
- append_memory: Save patterns, assessments, questions to your memory. This survives across sessions.
  Use when: you discover something worth remembering, revise a belief, or identify an open question.
- save_insight: Save a structured, verifiable insight with confidence level.
  Use when: you identify a specific pattern with clear conditions.
- web_search: Search the internet for current news, market data, or any real-time information.
  Use when: you need to verify current events, check market status, or find recent developments.
- Save automatically when you find something meaningful. Do NOT ask permission.
- Write in machine language. No human categories.

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


def build_system_prompt(context: dict, memory: str, insights: list, custom_role: str = None) -> list:
    """평가자의 시스템 프롬프트를 캐싱 가능한 블록 리스트로 구성한다.

    Anthropic prompt caching: 동일 프리픽스 재전송 시 입력 토큰 비용 ~90% 절감.
    - 블록 1: 역할 정의 (세션 내 불변)
    - 블록 2: 데이터 + 메모리 + 인사이트 (세션 내 불변, cache_control 설정)
    Tool-use 루프에서 동일 시스템 프롬프트가 반복 전송되므로 캐싱 효과가 크다.
    """
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
    # 2026년 스코어 전체 (compact: 한 줄에 여러 날짜) + 과거 참고용 최근 3개
    scores_2026 = [s for s in scores if s["date"] >= "2026"]
    scores_older = [s for s in scores if s["date"] < "2026"][-3:]

    # 과거 참고용
    older_lines = [
        f"  {s['date']}: {s['risk_score']:.1f} [{s['status']}]"
        for s in scores_older
    ]
    # 2026 compact: 날짜(MM-DD) score 형태로 한 줄에 5개씩
    chunks = []
    for i in range(0, len(scores_2026), 5):
        chunk = scores_2026[i:i+5]
        line = "  " + " | ".join(
            f"{s['date'][5:]}: {s['risk_score']:.1f}"
            for s in chunk
        )
        chunks.append(line)
    score_str = "\n".join(older_lines + (["  --- 2026 ---"] if older_lines else []) + chunks)

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

    # 날짜별 sig density + 기사 수 (compact)
    daily_stats = context.get("daily_stats", {})
    daily_lines = []
    for d in sorted(daily_stats.keys()):
        s = daily_stats[d]
        sig = s.get("sig_density", {})
        daily_lines.append(
            f"  {d[5:]}: {s['total_articles']:,}건 sig7={sig.get('sig7',0)}% sig15={sig.get('sig15',0)}%"
        )
    daily_str = "\n".join(daily_lines)

    role_text = custom_role if custom_role else "(no role defined)"

    dates = summary.get("dates_covered", ["?", "?"])
    date_range = f"{dates[0]} ~ {dates[-1]}" if dates else "no data"

    data_text = f"""=== CURRENT DATA ({date_range}, {len(dates)} days) ===

Daily overview:
{daily_str or '  (none)'}

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

    return [
        {"type": "text", "text": role_text},
        {
            "type": "text",
            "text": data_text,
            "cache_control": {"type": "ephemeral"},
        },
    ]


@st.cache_resource
def load_evaluator_context():
    """평가자 컨텍스트를 로드한다 (캐싱)."""
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")
    daily_dir = processed_path / "corpus_daily"
    return build_context(processed_path, outputs_path, daily_dir)


EVALUATOR_TOOLS = [
    {
        "name": "append_memory",
        "description": (
            "Append a new entry to your persistent memory. "
            "Use this when you discover a pattern, revise a belief, or want to remember something for future sessions. "
            "Write in machine language (Property IDs, ratios, patterns). "
            "Each entry should be a concise markdown snippet (a heading + bullet points)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Memory section: 'observed patterns', 'scoring assessments', 'open questions', or 'retrospective evaluation'",
                },
                "heading": {
                    "type": "string",
                    "description": "Short heading for this entry (e.g., 'P_083 acceleration pattern')",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown bullet points with the observation. Include confidence (0-1).",
                },
            },
            "required": ["section", "heading", "content"],
        },
    },
    {
        "name": "save_insight",
        "description": (
            "Save a structured insight for future reference. "
            "Use this when you identify a specific, verifiable pattern with clear conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["pattern", "anomaly", "threshold", "correlation", "revision"],
                },
                "properties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Property IDs involved (e.g., ['Property_083', 'Property_055'])",
                },
                "observation": {
                    "type": "string",
                    "description": "What was observed, in machine language",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level 0.0-1.0",
                },
            },
            "required": ["type", "observation", "confidence"],
        },
    },
]


def _handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """평가자의 도구 호출을 처리한다."""
    if tool_name == "append_memory":
        section = tool_input.get("section", "observed patterns")
        heading = tool_input.get("heading", "untitled")
        content = tool_input.get("content", "")

        current = load_memory()
        section_marker = f"## {section}"

        entry = f"\n### {heading}\n{content}\n"

        if section_marker in current:
            # 섹션 끝에 추가
            parts = current.split(section_marker, 1)
            # 다음 ## 섹션 찾기
            rest = parts[1]
            next_section = rest.find("\n## ")
            if next_section != -1:
                updated = parts[0] + section_marker + rest[:next_section] + entry + rest[next_section:]
            else:
                updated = parts[0] + section_marker + rest + entry
        else:
            # 새 섹션 추가
            updated = current.rstrip() + f"\n\n{section_marker}\n{entry}"

        save_memory(updated)
        return f"Memory updated: [{section}] {heading}"

    elif tool_name == "save_insight":
        insight = {
            "type": tool_input.get("type", "pattern"),
            "properties": tool_input.get("properties", []),
            "observation": tool_input.get("observation", ""),
            "confidence": tool_input.get("confidence", 0.5),
            "basis": "evaluator auto-save",
        }
        from src.evaluator.store import save_insight as _save_insight
        _save_insight(insight)
        return f"Insight saved: [{insight['type']}] {insight['observation'][:80]}"

    return f"Unknown tool: {tool_name}"


def call_claude(messages: list, system: list, api_key: str) -> tuple[str, list[str]]:
    """Claude API를 호출한다 (tool use 지원, prompt caching 적용).

    Args:
        system: cache_control이 포함된 시스템 프롬프트 블록 리스트.
    Returns:
        (응답 텍스트, 도구 사용 로그 리스트)
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key, max_retries=3)
    tool_logs = []

    all_tools = EVALUATOR_TOOLS + [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
        },
    ]

    def _stream_collect(**kwargs) -> anthropic.types.Message:
        """streaming으로 호출하고 최종 Message 객체를 반환한다."""
        with client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    call_kwargs = dict(
        model="claude-opus-4-20250514",
        max_tokens=16384,
        system=system,
        messages=messages,
        tools=all_tools,
    )

    response = _stream_collect(**call_kwargs)
    total_input = 0
    total_cache_read = 0
    total_cache_create = 0
    call_count = 0

    def _accumulate_usage(resp):
        nonlocal total_input, total_cache_read, total_cache_create, call_count
        call_count += 1
        u = resp.usage
        total_input += getattr(u, "input_tokens", 0)
        total_cache_read += getattr(u, "cache_read_input_tokens", 0)
        total_cache_create += getattr(u, "cache_creation_input_tokens", 0)

    _accumulate_usage(response)

    # tool use 루프: 커스텀 도구 호출이 있으면 처리 후 재호출
    while response.stop_reason == "tool_use":
        tool_results = []
        assistant_content = []
        for block in response.content:
            if block.type == "tool_use":
                result = _handle_tool_call(block.name, block.input)
                tool_logs.append(result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "text":
                assistant_content.append({
                    "type": "text",
                    "text": block.text,
                })

        messages = messages + [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_results},
        ]
        call_kwargs["messages"] = messages
        response = _stream_collect(**call_kwargs)
        _accumulate_usage(response)

    # 캐시 사용량 로그
    if total_cache_read > 0 or total_cache_create > 0:
        saved_pct = (total_cache_read / (total_input + total_cache_read + total_cache_create) * 100) if (total_input + total_cache_read + total_cache_create) > 0 else 0
        tool_logs.insert(0, f"Cache: {total_cache_read:,} read / {total_cache_create:,} created ({saved_pct:.0f}% cached, {call_count} calls)")

    # 텍스트 응답 추출
    text_parts = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(text_parts), tool_logs


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
            try:
                response, tool_logs = call_claude(api_messages, system_prompt, api_key)
            except Exception as e:
                err_msg = str(e)
                if "overloaded" in err_msg.lower() or "529" in err_msg:
                    response = "서버가 지금 혼잡합니다. 잠시 후 다시 시도해주세요."
                elif "rate" in err_msg.lower() and "limit" in err_msg.lower():
                    response = "API 호출 한도에 도달했습니다. 잠시 후 다시 시도해주세요."
                elif "authentication" in err_msg.lower() or "401" in err_msg:
                    response = "API 키가 유효하지 않습니다. .env 파일을 확인해주세요."
                else:
                    response = f"오류가 발생했습니다: {type(e).__name__}: {err_msg[:200]}"
                tool_logs = []
                st.warning(response)
                st.session_state.messages.pop()  # 실패한 user 메시지 제거
                st.stop()

        st.markdown(response)
        if tool_logs:
            with st.expander(f"Auto-saved ({len(tool_logs)})"):
                for log in tool_logs:
                    st.caption(log)

    st.session_state.messages.append({"role": "assistant", "content": response})
