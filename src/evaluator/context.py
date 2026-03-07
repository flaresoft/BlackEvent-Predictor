"""
Evaluator Context Builder

데이터에서 평가자용 컨텍스트를 추출한다.
인간의 카테고리가 아닌 기계의 언어(Property 번호, 비율, 패턴)로 기술한다.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_daily_property_stats(daily_dir: Path, dates: list[str] = None) -> dict:
    """일별 property 통계를 로드한다.

    Returns:
        {
            "2026-03-06": {
                "total_articles": 8324,
                "property_rates": {"Property_054": 25.3, ...},  # % of articles
                "sig_density": {"sig7": 56.4, "sig10": 44.1, "sig15": 28.5},
            }, ...
        }
    """
    results = {}

    parquet_files = sorted(daily_dir.glob("*.parquet"))
    if dates:
        parquet_files = [f for f in parquet_files if f.stem in dates]

    for fp in parquet_files:
        date = fp.stem
        df = pq.read_table(fp).to_pandas()
        total = len(df)
        if total == 0:
            continue

        prop_counts = defaultdict(int)
        sig7 = sig10 = sig15 = 0

        for props in df["properties"].values:
            if isinstance(props, np.ndarray):
                props = list(props)
            elif not isinstance(props, list):
                continue

            n = len(props)
            if n >= 7:
                sig7 += 1
            if n >= 10:
                sig10 += 1
            if n >= 15:
                sig15 += 1

            for p in props:
                prop_counts[p] += 1

        property_rates = {
            p: round(c / total * 100, 2) for p, c in prop_counts.items()
        }

        results[date] = {
            "total_articles": total,
            "property_rates": property_rates,
            "sig_density": {
                "sig7": round(sig7 / total * 100, 1),
                "sig10": round(sig10 / total * 100, 1),
                "sig15": round(sig15 / total * 100, 1),
            },
        }

    return results


def compute_property_dynamics(daily_stats: dict) -> list[dict]:
    """Property 간 동역학을 계산한다.

    - 각 property의 추세 (상승/하락/안정)
    - property 간 상관관계 변화
    - 급변 감지

    Returns:
        [
            {
                "property": "Property_083",
                "trend": "rising",
                "rate_sequence": [5.4, 8.8, 10.2],
                "velocity": +2.4,  # 일평균 변화율
            }, ...
        ]
    """
    dates = sorted(daily_stats.keys())
    if len(dates) < 2:
        return []

    all_props = set()
    for d in dates:
        all_props.update(daily_stats[d]["property_rates"].keys())

    dynamics = []
    for prop in sorted(all_props):
        rates = [daily_stats[d]["property_rates"].get(prop, 0) for d in dates]
        if max(rates) < 0.5:
            continue

        # 추세 판정
        if len(rates) >= 3:
            recent_3 = rates[-3:]
            if recent_3[-1] > recent_3[0] * 1.15:
                trend = "rising"
            elif recent_3[-1] < recent_3[0] * 0.85:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        # 속도 (최근 3일 평균 변화)
        if len(rates) >= 2:
            diffs = [rates[i] - rates[i - 1] for i in range(1, len(rates))]
            velocity = round(np.mean(diffs[-3:]), 3)
        else:
            velocity = 0

        # 급변 감지
        spike = False
        if len(rates) >= 2:
            last_change = abs(rates[-1] - rates[-2])
            avg_rate = np.mean(rates)
            if avg_rate > 0 and last_change / avg_rate > 0.3:
                spike = True

        dynamics.append({
            "property": prop,
            "trend": trend,
            "rate_sequence": [round(r, 2) for r in rates],
            "dates": dates,
            "velocity": velocity,
            "spike": spike,
            "current_rate": round(rates[-1], 2),
        })

    return dynamics


def compute_cross_patterns(daily_stats: dict) -> list[dict]:
    """Property 간 교차 패턴을 찾는다.

    - 동시 상승/하락 쌍
    - 역상관 쌍 (하나 오르면 하나 내림)
    """
    dates = sorted(daily_stats.keys())
    if len(dates) < 4:
        return []

    all_props = set()
    for d in dates:
        all_props.update(daily_stats[d]["property_rates"].keys())

    # 의미 있는 property만
    meaningful = []
    for prop in sorted(all_props):
        rates = [daily_stats[d]["property_rates"].get(prop, 0) for d in dates]
        if np.mean(rates) > 1.0:
            meaningful.append(prop)

    patterns = []
    for i, p1 in enumerate(meaningful):
        r1 = np.array([daily_stats[d]["property_rates"].get(p1, 0) for d in dates])
        for p2 in meaningful[i + 1:]:
            r2 = np.array([daily_stats[d]["property_rates"].get(p2, 0) for d in dates])

            if np.std(r1) < 0.1 or np.std(r2) < 0.1:
                continue

            corr = np.corrcoef(r1, r2)[0, 1]
            if abs(corr) > 0.7:
                patterns.append({
                    "pair": [p1, p2],
                    "correlation": round(corr, 3),
                    "type": "co-moving" if corr > 0 else "inverse",
                })

    patterns.sort(key=lambda x: -abs(x["correlation"]))
    return patterns[:30]


def load_score_history(outputs_path: Path) -> list[dict]:
    """daily_scores.csv에서 스코어 이력을 로드한다."""
    csv_path = outputs_path / "daily_scores.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    df = df.sort_values("date")

    history = []
    for _, row in df.iterrows():
        entry = {
            "date": row["date"],
            "risk_score": row["risk_score"],
            "status": row["status"],
            "n_articles": row.get("n_articles", 0),
            "top_contributor": row.get("top_contributor", ""),
        }

        if "details_json" in row and pd.notna(row["details_json"]):
            details = json.loads(row["details_json"])
            # ratio > 1인 것만 추출
            active = {
                pid: {"ratio": d["ratio"], "contribution": d["contribution"]}
                for pid, d in details.items()
                if d["ratio"] > 1.0
            }
            entry["active_properties"] = active

        history.append(entry)

    return history


def build_context(
    processed_path: Path,
    outputs_path: Path,
    daily_dir: Path = None,
    n_recent_days: int = 14,
) -> dict:
    """평가자용 전체 컨텍스트를 구성한다.

    Returns:
        {
            "generated_at": "2026-03-07T...",
            "daily_stats": {...},
            "property_dynamics": [...],
            "cross_patterns": [...],
            "score_history": [...],
            "summary": {
                "dates_covered": ["2026-02-23", ...],
                "total_properties_observed": 115,
                "rising_properties": [...],
                "falling_properties": [...],
                "high_density_dates": [...],
            }
        }
    """
    from datetime import datetime

    if daily_dir is None:
        daily_dir = processed_path / "corpus_daily"

    # 일별 통계
    daily_stats = load_daily_property_stats(daily_dir)

    # 최근 N일만
    all_dates = sorted(daily_stats.keys())
    recent_dates = all_dates[-n_recent_days:] if len(all_dates) > n_recent_days else all_dates
    recent_stats = {d: daily_stats[d] for d in recent_dates}

    # 동역학
    dynamics = compute_property_dynamics(recent_stats)

    # 교차 패턴
    cross = compute_cross_patterns(recent_stats)

    # 스코어 이력
    scores = load_score_history(outputs_path)

    # 요약
    rising = [d for d in dynamics if d["trend"] == "rising" and d["current_rate"] > 2]
    falling = [d for d in dynamics if d["trend"] == "falling" and d["current_rate"] > 2]
    high_density = [
        d for d in recent_dates
        if recent_stats[d]["sig_density"]["sig15"] > 30
    ]

    context = {
        "generated_at": datetime.now().isoformat(),
        "daily_stats": recent_stats,
        "property_dynamics": dynamics,
        "cross_patterns": cross,
        "score_history": scores,
        "summary": {
            "dates_covered": recent_dates,
            "total_properties_observed": len(
                set().union(*(s["property_rates"].keys() for s in recent_stats.values()))
            ),
            "rising_properties": [
                {"property": d["property"], "velocity": d["velocity"], "current": d["current_rate"]}
                for d in rising
            ],
            "falling_properties": [
                {"property": d["property"], "velocity": d["velocity"], "current": d["current_rate"]}
                for d in falling
            ],
            "high_density_dates": high_density,
        },
    }

    return context
