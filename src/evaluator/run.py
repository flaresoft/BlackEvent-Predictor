"""
Evaluator Runner

평가자 세션을 실행한다:
1. 데이터에서 컨텍스트 추출
2. 이전 경험(memory) 로딩
3. 평가 수행 → 인사이트/판단 생성
4. 경험 업데이트

CLI 사용:
    python -m src.evaluator.run                    # 컨텍스트 생성 + 출력
    python -m src.evaluator.run --save-insight     # 대화형 인사이트 저장
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.evaluator.context import build_context
from src.evaluator.store import (
    load_memory,
    save_memory,
    load_insights,
    save_insight,
    save_judgment,
    load_judgments,
)
from src.utils import load_config, get_path


def generate_context_report() -> dict:
    """현재 데이터 상태에서 평가자 컨텍스트를 생성한다."""
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")
    daily_dir = processed_path / "corpus_daily"

    context = build_context(processed_path, outputs_path, daily_dir)
    return context


def print_context_summary(context: dict):
    """컨텍스트 요약을 출력한다."""
    summary = context["summary"]

    print(f"\n=== Evaluator Context ({context['generated_at'][:10]}) ===")
    print(f"Dates: {summary['dates_covered'][0]} ~ {summary['dates_covered'][-1]}")
    print(f"Properties observed: {summary['total_properties_observed']}")

    print(f"\nRising ({len(summary['rising_properties'])}):")
    for p in summary["rising_properties"][:10]:
        print(f"  {p['property']}: {p['current']}% (v={p['velocity']:+.2f})")

    print(f"\nFalling ({len(summary['falling_properties'])}):")
    for p in summary["falling_properties"][:10]:
        print(f"  {p['property']}: {p['current']}% (v={p['velocity']:+.2f})")

    if summary["high_density_dates"]:
        print(f"\nHigh density dates (sig15+>30%): {summary['high_density_dates']}")

    # Score history
    scores = context["score_history"]
    if scores:
        print(f"\nScore history ({len(scores)} entries):")
        for s in scores[-5:]:
            active = len(s.get("active_properties", {}))
            print(
                f"  {s['date']}: {s['risk_score']:.1f} "
                f"[{s['status']}] top={s['top_contributor']} "
                f"active={active}/12"
            )

    # Cross patterns
    cross = context["cross_patterns"]
    if cross:
        print(f"\nCross patterns ({len(cross)}):")
        for cp in cross[:5]:
            print(f"  {cp['pair'][0]} <-> {cp['pair'][1]}: {cp['correlation']:+.3f} ({cp['type']})")

    # Memory
    memory = load_memory()
    if memory:
        lines = memory.strip().split("\n")
        print(f"\nEvaluator memory: {len(lines)} lines loaded")
    else:
        print("\nEvaluator memory: empty (first session)")

    # Past insights
    all_insights = load_insights()
    if all_insights:
        print(f"Past insights: {len(all_insights)}")
    else:
        print("Past insights: none")

    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluator")
    parser.add_argument(
        "--json", action="store_true",
        help="Output full context as JSON"
    )
    args = parser.parse_args()

    context = generate_context_report()

    if args.json:
        print(json.dumps(context, ensure_ascii=False, indent=2, default=str))
    else:
        print_context_summary(context)


if __name__ == "__main__":
    main()
