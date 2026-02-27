"""
Lead Time Test — BlackEvent 며칠 전부터 모델이 감지하는지 검증한다.

각 테스트 BlackEvent에 대해 lead_days=[0,1,3,7,14,21,30]일 전 시점에서
Model B 스코어를 계산하고, threshold(18.52) 초과 여부를 판단한다.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, get_path, setup_logging
from src.module6_scoring.run import (
    load_daily_aggregation,
    compute_window_frequency,
    WeightedFrequencyModel,
)

logger = setup_logging("lead_time_test")

LEAD_DAYS = [0, 1, 3, 7, 14, 21, 30]


def run():
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")

    scoring_config = config.get("scoring", {})
    normalization_mode = scoring_config.get("normalization_mode", "empirical")
    default_window = scoring_config.get("default_window", 60)
    val_config = config["validation"]
    train_end = val_config["train_end"]

    # ── 유의미 성질 & 모델 설정 ──
    sig_df = pd.read_csv(outputs_path / "significant_properties.csv")
    sig_props = sig_df["property_id"].tolist()

    optimal_windows: dict[str, int] = {}
    for _, r in sig_df.iterrows():
        optimal_windows[r["property_id"]] = int(r["optimal_window"])

    unique_windows = set(optimal_windows.values())
    max_window = max(unique_windows) if unique_windows else default_window
    logger.info(f"유의미 성질 {len(sig_props)}개, 고유 윈도우: {sorted(unique_windows)}")

    # ── Model B 초기화 + 캘리브레이션 ──
    model_b = WeightedFrequencyModel(sig_df, normalization_mode=normalization_mode)

    risk_scores = pd.read_csv(outputs_path / "risk_scores.csv")
    empirical_max = risk_scores.loc[
        risk_scores["actual_label"] == 1, "model_b_raw_score"
    ].max()
    model_b.calibrate(empirical_max)
    logger.info(f"캘리브레이션: empirical_max = {empirical_max:.4f}")

    # Module 7에서 구한 threshold
    threshold = 18.52
    logger.info(f"검출 threshold: {threshold}")

    # ── 테스트 BlackEvent 추출 ──
    events_df = pd.read_csv(
        processed_path / "black_events.csv", parse_dates=["first_shock_date"]
    )
    test_events = events_df[
        events_df["first_shock_date"] > pd.Timestamp(train_end)
    ].reset_index(drop=True)
    logger.info(f"테스트 BlackEvent: {len(test_events)}건")

    # ── 일별 집계 로드 (테스트 기간만) ──
    max_lead = max(LEAD_DAYS)
    date_min = (
        test_events["first_shock_date"].min()
        - pd.Timedelta(days=max_window + max_lead + 5)
    ).strftime("%Y-%m-%d")
    date_max = test_events["first_shock_date"].max().strftime("%Y-%m-%d")

    logger.info(f"코퍼스 일별 집계 로드: {date_min} ~ {date_max}")
    dates_arr, daily_counts, articles_per_day, total = load_daily_aggregation(
        processed_path, date_min=date_min, date_max=date_max
    )
    logger.info(f"로드 완료: {total:,}건 → {len(dates_arr):,}일")

    # ── Lead Time 스코어링 ──
    logger.info(f"Lead Time 테스트 시작: {len(test_events)}건 × {len(LEAD_DAYS)} offsets")
    results = []

    for i, (_, event) in enumerate(test_events.iterrows()):
        be_date = event["first_shock_date"]
        event_id = event.get("event_id", i)

        for lead in LEAD_DAYS:
            ref_date = be_date - pd.Timedelta(days=lead)

            # 고유 윈도우별 빈도 계산
            window_freqs: dict[int, dict[str, int]] = {}
            window_articles: dict[int, int] = {}
            for w in unique_windows:
                freq_map, n_art = compute_window_frequency(
                    dates_arr, daily_counts, articles_per_day, ref_date, w
                )
                window_freqs[w] = freq_map
                window_articles[w] = n_art

            # 성질별 최적 윈도우에서 빈도 추출
            combined_freq: dict[str, int] = {}
            for prop in sig_props:
                w = optimal_windows.get(prop, default_window)
                combined_freq[prop] = window_freqs.get(w, {}).get(prop, 0)

            n_articles = window_articles.get(max(unique_windows), 0)
            result = model_b.predict(combined_freq, n_articles)

            results.append({
                "event_id": event_id,
                "be_date": be_date.strftime("%Y-%m-%d"),
                "lead_days": lead,
                "ref_date": ref_date.strftime("%Y-%m-%d"),
                "risk_score": result["risk_score"],
                "raw_score": result["raw_score"],
                "detected": 1 if result["risk_score"] >= threshold else 0,
            })

        if (i + 1) % 5 == 0 or (i + 1) == len(test_events):
            logger.info(f"  진행: {i+1}/{len(test_events)}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(outputs_path / "lead_time_analysis.csv", index=False)
    logger.info(f"결과 저장: {outputs_path / 'lead_time_analysis.csv'}")

    # ── 요약 ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Lead Time 분석 결과 (threshold >= %.2f)", threshold)
    logger.info("=" * 60)
    logger.info(
        f"  {'Lead':>6s}  {'감지':>5s}/{len(test_events):<3d}  "
        f"{'감지율':>7s}  {'평균스코어':>10s}  {'중앙값':>8s}"
    )
    logger.info("-" * 60)

    for lead in LEAD_DAYS:
        subset = results_df[results_df["lead_days"] == lead]
        n_detected = int(subset["detected"].sum())
        n_total = len(subset)
        rate = n_detected / n_total * 100 if n_total > 0 else 0
        mean_score = subset["risk_score"].mean()
        median_score = subset["risk_score"].median()
        logger.info(
            f"  {lead:4d}일  {n_detected:5d}/{n_total:<3d}  "
            f"{rate:6.1f}%  {mean_score:10.2f}  {median_score:8.2f}"
        )

    logger.info("=" * 60)
    logger.info("=== Lead Time 테스트 완료 ===")
    return results_df


if __name__ == "__main__":
    run()
