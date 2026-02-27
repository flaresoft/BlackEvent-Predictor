"""
Daily Risk Scorer — 매일 자동으로 현재 리스크 스코어를 산출한다.

Usage:
    python -m src.daily_scorer.run               # 코퍼스 최신 날짜 기준
    python -m src.daily_scorer.run --date 2026-02-22  # 특정 날짜 지정
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, get_path, setup_logging
from src.module6_scoring.run import (
    load_daily_aggregation,
    compute_window_frequency,
    WeightedFrequencyModel,
)

logger = setup_logging("daily_scorer")

THRESHOLD = 18.52  # Module 7에서 결정된 최적 threshold

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


def find_latest_date(processed_path: Path) -> pd.Timestamp:
    """코퍼스에서 가장 최근 날짜를 찾는다."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(processed_path / "corpus_tagged.parquet")
    last_rg = pf.read_row_group(pf.metadata.num_row_groups - 1, columns=["date"])
    df = last_rg.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df["date"].max()


def run(target_date: str = None) -> dict:
    """단일 날짜의 리스크 스코어를 산출한다."""
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")
    scoring_config = config.get("scoring", {})
    normalization_mode = scoring_config.get("normalization_mode", "empirical")
    default_window = scoring_config.get("default_window", 60)

    # ── 기준일 결정 ──
    if target_date:
        ref_date = pd.Timestamp(target_date)
    else:
        ref_date = find_latest_date(processed_path)
    logger.info(f"기준일: {ref_date.strftime('%Y-%m-%d')}")

    # ── 유의미 성질 로드 ──
    sig_df = pd.read_csv(outputs_path / "significant_properties.csv")
    sig_props = sig_df["property_id"].tolist()
    optimal_windows: dict[str, int] = {}
    for _, r in sig_df.iterrows():
        optimal_windows[r["property_id"]] = int(r["optimal_window"])

    # ── Model B 초기화 + 캘리브레이션 ──
    model_b = WeightedFrequencyModel(sig_df, normalization_mode=normalization_mode)
    risk_scores_df = pd.read_csv(outputs_path / "risk_scores.csv")
    empirical_max = risk_scores_df.loc[
        risk_scores_df["actual_label"] == 1, "model_b_raw_score"
    ].max()
    model_b.calibrate(empirical_max)

    # ── 코퍼스 일별 집계 (필요한 범위만) ──
    max_window = max(optimal_windows.values()) if optimal_windows else default_window
    date_min = (ref_date - pd.Timedelta(days=max_window + 5)).strftime("%Y-%m-%d")
    date_max = ref_date.strftime("%Y-%m-%d")

    logger.info(f"코퍼스 로드: {date_min} ~ {date_max}")
    dates_arr, daily_counts, articles_per_day, total = load_daily_aggregation(
        processed_path, date_min=date_min, date_max=date_max
    )
    logger.info(f"로드 완료: {total:,}건 / {len(dates_arr):,}일")

    # ── 윈도우별 빈도 계산 ──
    unique_windows = set(optimal_windows.values())
    window_freqs: dict[int, dict[str, int]] = {}
    window_articles: dict[int, int] = {}
    for w in unique_windows:
        freq_map, n_art = compute_window_frequency(
            dates_arr, daily_counts, articles_per_day, ref_date, w
        )
        window_freqs[w] = freq_map
        window_articles[w] = n_art

    # ── 성질별 빈도 추출 ──
    combined_freq: dict[str, int] = {}
    for prop in sig_props:
        w = optimal_windows.get(prop, default_window)
        combined_freq[prop] = window_freqs.get(w, {}).get(prop, 0)

    n_articles = window_articles.get(max(unique_windows), 0)

    # ── 스코어 산출 ──
    result = model_b.predict(combined_freq, n_articles)
    risk_score = result["risk_score"]
    raw_score = result["raw_score"]
    details = result["details"]

    # ── 상태 판정 ──
    if risk_score >= THRESHOLD:
        status = "WARNING"
        gap_str = f"+{risk_score - THRESHOLD:.2f}"
    else:
        status = "NORMAL"
        gap_str = f"-{THRESHOLD - risk_score:.2f}"

    # ── Top 기여 성질 정렬 ──
    sorted_props = sorted(
        details.keys(), key=lambda x: details[x]["contribution"], reverse=True
    )
    top3 = []
    for pid in sorted_props[:3]:
        d = details[pid]
        if d["ratio"] > 0:
            top3.append(f"{pid}({d['ratio']:.2f}x)")

    # ── 콘솔 출력 ──
    date_str = ref_date.strftime("%Y-%m-%d")
    print()
    print(f"[{date_str}] Risk Score: {risk_score:.2f}/100 | "
          f"Threshold: {THRESHOLD} | STATUS: {status} ({gap_str})")
    print(f"  Top 3: {', '.join(top3)}")
    print()

    # 상세 출력
    print(f"  {'ID':<14s} {'Name':<26s} {'Win':>4s} {'Freq':>7s} "
          f"{'Ctrl':>7s} {'Ratio':>6s} {'Contrib':>7s}")
    print("  " + "-" * 76)
    for pid in sorted_props:
        d = details[pid]
        name = PROP_NAMES.get(pid, pid)[:24]
        w = optimal_windows.get(pid, default_window)
        arrow = " !!" if d["ratio"] > 1.2 else (" +" if d["ratio"] > 1.0 else "")
        print(f"  {pid:<14s} {name:<26s} {w:>3d}d {d['freq']:>7,d} "
              f"{d['ctrl_mean']:>7,.0f} {d['ratio']:>6.3f} {d['contribution']:>7.4f}{arrow}")

    print()

    # ── daily_scores.csv에 append ──
    details_json = json.dumps(
        {pid: {"ratio": details[pid]["ratio"], "contribution": details[pid]["contribution"]}
         for pid in sorted_props},
        ensure_ascii=False,
    )
    top_contributor = sorted_props[0] if sorted_props else ""

    row = {
        "date": date_str,
        "risk_score": round(risk_score, 2),
        "raw_score": round(raw_score, 4),
        "threshold": THRESHOLD,
        "status": status,
        "n_articles": n_articles,
        "top_contributor": top_contributor,
        "details_json": details_json,
    }

    csv_path = outputs_path / "daily_scores.csv"
    new_row = pd.DataFrame([row])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # 같은 날짜 중복 방지: 기존 행 교체
        existing = existing[existing["date"] != date_str]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(csv_path, index=False)
    logger.info(f"저장: {csv_path} ({len(combined)}건)")

    return row


def main():
    parser = argparse.ArgumentParser(description="Daily Risk Scorer")
    parser.add_argument("--date", type=str, default=None,
                        help="기준일 (YYYY-MM-DD). 미지정 시 코퍼스 최신 날짜")
    args = parser.parse_args()
    run(target_date=args.date)


if __name__ == "__main__":
    main()
