"""
일별 리스크 스코어를 월별로 집계하고 사회 지표 데이터와 병합한다.
"""

from pathlib import Path

import pandas as pd

from src.utils import setup_logging

logger = setup_logging("module9.aggregator")


def load_daily_scores(outputs_path: Path) -> pd.DataFrame:
    """daily_scores.csv를 로드하고 날짜 파싱한다."""
    csv_path = outputs_path / "daily_scores.csv"
    if not csv_path.exists():
        logger.error(f"daily_scores.csv가 없습니다: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    logger.info(f"일별 스코어 로드: {len(df)}일 ({df['date'].min()} ~ {df['date'].max()})")
    return df


def aggregate_monthly(daily_df: pd.DataFrame, threshold: float = 18.52) -> pd.DataFrame:
    """일별 리스크 스코어를 월별로 집계한다.

    집계 지표:
    - risk_mean: 월 평균 리스크 스코어
    - risk_max: 월 최대 리스크 스코어
    - risk_std: 월 표준편차
    - warning_days: WARNING 일수 (>= threshold)
    - n_days: 데이터 존재 일수
    """
    if daily_df.empty:
        return pd.DataFrame()

    monthly = daily_df.groupby("year_month").agg(
        risk_mean=("risk_score", "mean"),
        risk_max=("risk_score", "max"),
        risk_std=("risk_score", "std"),
        n_days=("risk_score", "count"),
    ).reset_index()

    # WARNING 일수
    warning_counts = (
        daily_df[daily_df["risk_score"] >= threshold]
        .groupby("year_month")
        .size()
        .reset_index(name="warning_days")
    )
    monthly = monthly.merge(warning_counts, on="year_month", how="left")
    monthly["warning_days"] = monthly["warning_days"].fillna(0).astype(int)
    monthly["risk_std"] = monthly["risk_std"].fillna(0)

    logger.info(f"월별 집계: {len(monthly)}개월")
    return monthly


def merge_with_social(
    monthly_risk: pd.DataFrame,
    social_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """월별 리스크 집계와 사회 지표를 병합한다."""
    merged = monthly_risk.copy()

    for name, df in social_data.items():
        if df.empty:
            logger.warning(f"사회 지표 '{name}' 데이터가 비어있어 건너뜁니다.")
            continue

        indicator_df = df.rename(columns={"value": name})[["year_month", name]]
        merged = merged.merge(indicator_df, on="year_month", how="left")
        n_matched = merged[name].notna().sum()
        logger.info(f"  '{name}' 병합: {n_matched}/{len(merged)}개월 매칭")

    logger.info(f"병합 완료: {len(merged)}행, 컬럼 {list(merged.columns)}")
    return merged
