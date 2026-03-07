"""
Evaluator Store

평가자의 경험을 축적하고 로딩하는 모듈.
- memory.md: 평가자의 현재 믿음/패턴 인식 (기계의 언어)
- insights/*.jsonl: 발견된 패턴 (날짜별, 구조화, 검증 가능)
- sessions/*.md: 대화 전체 기록
- judgments.jsonl: 태깅/스코어링 판단 이력
"""

import json
from datetime import datetime
from pathlib import Path


EVALUATOR_DIR = Path("data/evaluator")
MEMORY_PATH = EVALUATOR_DIR / "memory.md"
INSIGHTS_DIR = EVALUATOR_DIR / "insights"
SESSIONS_DIR = EVALUATOR_DIR / "sessions"
JUDGMENTS_PATH = EVALUATOR_DIR / "judgments.jsonl"


def ensure_dirs():
    """디렉토리 구조를 보장한다."""
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_memory() -> str:
    """평가자의 축적된 경험을 로드한다."""
    if MEMORY_PATH.exists():
        return MEMORY_PATH.read_text(encoding="utf-8")
    return ""


def save_memory(content: str):
    """평가자의 경험을 저장한다."""
    ensure_dirs()
    MEMORY_PATH.write_text(content, encoding="utf-8")


def load_insights(date: str = None) -> list[dict]:
    """인사이트를 로드한다. date 미지정 시 전체."""
    ensure_dirs()
    results = []

    if date:
        fp = INSIGHTS_DIR / f"{date}.jsonl"
        if fp.exists():
            for line in fp.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    results.append(json.loads(line))
    else:
        for fp in sorted(INSIGHTS_DIR.glob("*.jsonl")):
            for line in fp.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    results.append(json.loads(line))

    return results


def save_insight(insight: dict, date: str = None):
    """인사이트를 추가한다.

    insight 구조:
    {
        "timestamp": "2026-03-07T14:30:00",
        "type": "pattern" | "threshold" | "correlation" | "anomaly" | "revision",
        "properties": ["Property_083", "Property_055"],
        "observation": "P_083 rate > 10% for 2 consecutive days...",
        "conditions": {"P_083_rate": ">10", "period": "2d"},
        "confidence": 0.7,
        "basis": "2026-03-05~06 data",
        "prior_ref": null,  # 이전 인사이트 수정 시 참조
    }
    """
    ensure_dirs()
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if "timestamp" not in insight:
        insight["timestamp"] = datetime.now().isoformat()

    fp = INSIGHTS_DIR / f"{date}.jsonl"
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(insight, ensure_ascii=False) + "\n")


def save_judgment(judgment: dict):
    """태깅/스코어링 판단을 기록한다.

    judgment 구조:
    {
        "timestamp": "2026-03-07T14:30:00",
        "target_date": "2026-03-06",
        "type": "tagging" | "scoring",
        "assessment": "P_110 ratio 0.75 is understated given...",
        "properties_evaluated": ["Property_110"],
        "verdict": "underweight",  # accurate | overweight | underweight | noise | signal
        "confidence": 0.6,
        "revisable": true,
    }
    """
    ensure_dirs()
    if "timestamp" not in judgment:
        judgment["timestamp"] = datetime.now().isoformat()

    with open(JUDGMENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(judgment, ensure_ascii=False) + "\n")


def load_judgments() -> list[dict]:
    """전체 판단 이력을 로드한다."""
    if not JUDGMENTS_PATH.exists():
        return []
    results = []
    for line in JUDGMENTS_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    return results


def save_session(date: str, content: str):
    """세션 로그를 저장한다."""
    ensure_dirs()
    fp = SESSIONS_DIR / f"{date}.md"
    fp.write_text(content, encoding="utf-8")


def load_session(date: str) -> str:
    """세션 로그를 로드한다."""
    fp = SESSIONS_DIR / f"{date}.md"
    if fp.exists():
        return fp.read_text(encoding="utf-8")
    return ""
