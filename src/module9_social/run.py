"""
Module 9 — Social Impact Analysis
경제 위기 리스크 스코어와 사회 지표(자살률, 범죄율, 실업률)의 상관관계를 분석한다.

핵심 가설: 뉴스 기반 구조적 위기 신호가 사회적 지표에 선행한다.
"""

import json
from pathlib import Path

import pandas as pd

from src.utils import load_config, get_path, setup_logging
from src.module9_social.data_collector import load_all_indicators
from src.module9_social.aggregator import (
    load_daily_scores,
    aggregate_monthly,
    merge_with_social,
)
from src.module9_social.analysis import run_full_analysis
from src.module9_social.visualizer import generate_all_plots

logger = setup_logging("module9")


def run(force_refresh: bool = False):
    """Module 9 전체 파이프라인 실행

    Args:
        force_refresh: True이면 KOSIS API에서 데이터를 다시 가져온다.
    """
    config = load_config()
    outputs_path = get_path(config, "outputs")
    social_config = config.get("social", {})

    if not social_config:
        logger.error(
            "config/settings.yaml에 'social' 섹션이 없습니다. "
            "설정을 추가한 후 다시 실행하세요."
        )
        return

    social_output = outputs_path / "social"
    social_output.mkdir(parents=True, exist_ok=True)

    threshold = social_config.get("aggregation", {}).get("threshold", 18.52)

    # ── Step 1: 사회 지표 수집 ──
    logger.info("=== Step 1: 사회 지표 데이터 수집 ===")
    social_data = load_all_indicators(
        social_config, cache_dir=social_output, force_refresh=force_refresh,
    )

    loaded = {k: len(v) for k, v in social_data.items() if not v.empty}
    if not loaded:
        logger.error(
            "사회 지표 데이터를 수집하지 못했습니다. "
            "KOSIS_API_KEY를 확인하거나, "
            "data/outputs/social/{indicator}_raw.csv를 수동으로 준비하세요.\n"
            "CSV 형식: year_month (YYYY-MM), value"
        )
        return
    logger.info(f"수집된 지표: {loaded}")

    # ── Step 2: 일별 리스크 스코어 월별 집계 ──
    logger.info("=== Step 2: 리스크 스코어 월별 집계 ===")
    daily_df = load_daily_scores(outputs_path)
    if daily_df.empty:
        logger.error("daily_scores.csv가 없습니다. daily_pipeline을 먼저 실행하세요.")
        return

    monthly_risk = aggregate_monthly(daily_df, threshold=threshold)
    monthly_risk.to_csv(social_output / "monthly_risk.csv", index=False)
    logger.info(f"월별 집계 저장: {social_output / 'monthly_risk.csv'}")

    # ── Step 3: 병합 ──
    logger.info("=== Step 3: 데이터 병합 ===")
    merged = merge_with_social(monthly_risk, social_data)
    merged.to_csv(social_output / "merged_monthly.csv", index=False)
    logger.info(f"병합 데이터 저장: {social_output / 'merged_monthly.csv'}")

    # ── Step 4: 분석 ──
    logger.info("=== Step 4: 상관분석 ===")
    results = run_full_analysis(merged, social_config)

    # 결과 저장
    for key in ["cross_correlation", "granger", "rolling", "summary"]:
        df = results.get(key, pd.DataFrame())
        if not df.empty:
            df.to_csv(social_output / f"{key}.csv", index=False)
            logger.info(f"{key} 저장: {len(df)}행")

    # ── Step 5: 시각화 ──
    logger.info("=== Step 5: 시각화 ===")
    generate_all_plots(merged, results, social_output)

    # ── 요약 출력 ──
    summary_df = results.get("summary", pd.DataFrame())
    if not summary_df.empty:
        logger.info("\n=== 분석 요약 ===")
        sig_results = summary_df[summary_df["significant"]]
        if not sig_results.empty:
            for _, row in sig_results.iterrows():
                logger.info(
                    f"  {row['risk_metric']} → {row['social_indicator']}: "
                    f"lag={row['best_lag']}개월, r={row['best_correlation']:.3f}, "
                    f"p={row['best_p_adjusted']:.4f} *"
                )
        else:
            logger.info("  유의미한 상관관계가 발견되지 않았습니다.")
            logger.info("  (데이터 기간이 짧거나 관계가 약할 수 있습니다)")

    logger.info("=== Module 9 완료 ===")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 9 — Social Impact Analysis")
    parser.add_argument(
        "--refresh", action="store_true",
        help="KOSIS API에서 데이터를 다시 가져온다",
    )
    args = parser.parse_args()
    run(force_refresh=args.refresh)
