"""
과거 일별 리스크 스코어 역산 — 코퍼스 전체 기간에 대해 매일 스코어를 산출한다.

코퍼스를 한 번만 로드하고 전체 날짜를 순회하므로,
daily_scorer를 날짜별로 반복 호출하는 것보다 훨씬 효율적이다.

Usage:
    python -m src.module9_social.backfill_scores
    python -m src.module9_social.backfill_scores --start 2010-01-01 --end 2025-12-31
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, get_path, setup_logging
from src.module6_scoring.run import (
    load_daily_aggregation,
    compute_window_frequency,
    WeightedFrequencyModel,
    _safe_load_csv,
)

logger = setup_logging("backfill_scores")

THRESHOLD = 18.52


def run(start_date: str = None, end_date: str = None):
    """전체 기간 일별 리스크 스코어를 역산한다."""
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")
    scoring_config = config.get("scoring", {})
    normalization_mode = scoring_config.get("normalization_mode", "empirical")
    default_window = scoring_config.get("default_window", 60)

    # ── 유의미 성질 로드 ──
    sig_df = _safe_load_csv(outputs_path / "significant_properties.csv", "significant_properties")
    if sig_df.empty:
        logger.error("significant_properties.csv가 없습니다.")
        return

    sig_props = sig_df["property_id"].tolist()
    optimal_windows: dict[str, int] = {}
    for _, r in sig_df.iterrows():
        optimal_windows[r["property_id"]] = int(r["optimal_window"])

    max_window = max(optimal_windows.values()) if optimal_windows else default_window
    unique_windows = set(optimal_windows.values())

    # ── Model B 초기화 + 캘리브레이션 ──
    model_b = WeightedFrequencyModel(sig_df, normalization_mode=normalization_mode)
    risk_scores_df = _safe_load_csv(outputs_path / "risk_scores.csv", "risk_scores")
    if not risk_scores_df.empty:
        empirical_max = risk_scores_df.loc[
            risk_scores_df["actual_label"] == 1, "model_b_raw_score"
        ].max()
        model_b.calibrate(empirical_max)
    else:
        logger.warning("risk_scores.csv가 없어 캘리브레이션을 건너뜁니다.")

    # ── 코퍼스 전체 로드 (한 번만) ──
    logger.info("코퍼스 전체 일별 집계 로드 중... (시간이 걸립니다)")
    dates_arr, daily_counts, articles_per_day, total = load_daily_aggregation(
        processed_path, date_min=start_date, date_max=end_date,
    )
    logger.info(f"로드 완료: {total:,}건 / {len(dates_arr):,}일")

    if len(dates_arr) == 0:
        logger.error("코퍼스에 데이터가 없습니다.")
        return

    # ── 스코어링 날짜 범위 결정 ──
    corpus_start = pd.Timestamp(dates_arr[0])
    corpus_end = pd.Timestamp(dates_arr[-1])

    # max_window 이후부터 시작 (윈도우 데이터 확보)
    score_start = corpus_start + pd.Timedelta(days=max_window)
    score_end = corpus_end

    if start_date:
        req_start = pd.Timestamp(start_date) + pd.Timedelta(days=max_window)
        score_start = max(score_start, req_start)
    if end_date:
        score_end = min(score_end, pd.Timestamp(end_date))

    # 전체 날짜 목록 생성
    all_dates = pd.date_range(score_start, score_end, freq="D")
    logger.info(
        f"스코어링 기간: {score_start.strftime('%Y-%m-%d')} ~ "
        f"{score_end.strftime('%Y-%m-%d')} ({len(all_dates):,}일)"
    )

    # ── 이미 계산된 날짜 확인 ──
    csv_path = outputs_path / "daily_scores.csv"
    existing_dates = set()
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        existing_dates = set(existing_df["date"].astype(str))
        logger.info(f"기존 스코어: {len(existing_dates)}일 (건너뜀)")

    # 계산 필요한 날짜만 필터
    dates_to_score = [d for d in all_dates if d.strftime("%Y-%m-%d") not in existing_dates]
    logger.info(f"신규 계산 대상: {len(dates_to_score):,}일")

    if not dates_to_score:
        logger.info("모든 날짜가 이미 계산되어 있습니다.")
        return

    # ── 배치 스코어링 ──
    rows = []
    for i, ref_date in enumerate(dates_to_score):
        ref_ts = pd.Timestamp(ref_date)

        # 윈도우별 빈도 계산
        window_freqs: dict[int, dict[str, int]] = {}
        window_articles: dict[int, int] = {}
        for w in unique_windows:
            freq_map, n_art, actual_days = compute_window_frequency(
                dates_arr, daily_counts, articles_per_day, ref_ts, w,
            )
            # coverage 보정
            if actual_days > 0 and actual_days < w:
                coverage = actual_days / w
                if coverage < 0.8:
                    freq_map = {k: int(v / coverage) for k, v in freq_map.items()}
                    n_art = int(n_art / coverage)
            window_freqs[w] = freq_map
            window_articles[w] = n_art

        # 성질별 빈도 추출
        combined_freq: dict[str, int] = {}
        for prop in sig_props:
            w = optimal_windows.get(prop, default_window)
            combined_freq[prop] = window_freqs.get(w, {}).get(prop, 0)

        n_articles = window_articles.get(max(unique_windows), 0)

        # 스코어 산출
        result = model_b.predict(combined_freq, n_articles)
        risk_score = result["risk_score"]
        raw_score = result["raw_score"]
        details = result["details"]

        status = "WARNING" if risk_score >= THRESHOLD else "NORMAL"

        sorted_props = sorted(
            details.keys(), key=lambda x: details[x]["contribution"], reverse=True,
        )
        top_contributor = sorted_props[0] if sorted_props else ""

        details_json = json.dumps(
            {pid: {"ratio": details[pid]["ratio"], "contribution": details[pid]["contribution"]}
             for pid in sorted_props},
            ensure_ascii=False,
        )

        rows.append({
            "date": ref_ts.strftime("%Y-%m-%d"),
            "risk_score": round(risk_score, 2),
            "raw_score": round(raw_score, 4),
            "threshold": THRESHOLD,
            "status": status,
            "n_articles": n_articles,
            "top_contributor": top_contributor,
            "details_json": details_json,
        })

        # 진행 로그
        if (i + 1) % 100 == 0 or i == len(dates_to_score) - 1:
            pct = (i + 1) / len(dates_to_score) * 100
            logger.info(
                f"  [{i+1:,}/{len(dates_to_score):,}] {pct:.1f}% — "
                f"{ref_ts.strftime('%Y-%m-%d')}: {risk_score:.2f} ({status})"
            )

    # ── 저장 ──
    new_df = pd.DataFrame(rows)
    logger.info(f"신규 스코어: {len(new_df):,}일")

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)

    logger.info(f"저장 완료: {csv_path} (총 {len(combined):,}일)")

    # 요약
    warning_days = (combined["status"] == "WARNING").sum()
    logger.info(
        f"\n=== 요약 ===\n"
        f"  전체 기간: {combined['date'].iloc[0]} ~ {combined['date'].iloc[-1]}\n"
        f"  총 일수: {len(combined):,}\n"
        f"  WARNING 일수: {warning_days:,} ({warning_days/len(combined)*100:.1f}%)\n"
        f"  평균 스코어: {combined['risk_score'].mean():.2f}\n"
        f"  최대 스코어: {combined['risk_score'].max():.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="과거 일별 리스크 스코어 역산")
    parser.add_argument("--start", type=str, default=None, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="종료일 (YYYY-MM-DD)")
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end)
