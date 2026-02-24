"""
Module 1 — BlackEvent Index
코스피 일별 데이터에서 급락 이벤트(BlackEvent)를 정의하고 추출한다.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import load_config, get_path, setup_logging
from src.module1_blackevent.validate import (
    check_output_staleness,
    validate_kospi_data,
    validate_shock_days,
    validate_black_events,
    process_validation_report,
)

logger = setup_logging("module1")


def fetch_kospi_data(ticker: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    """
    yfinance로 코스피(^KS11) 일봉 데이터를 수집한다.
    Returns: columns=['Date','Close'] (Date는 인덱스가 아닌 컬럼)
    """
    logger.info(f"코스피 데이터 수집: {ticker} ({start_date} ~ {end_date or '현재'})")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        raise ValueError("코스피 데이터를 가져올 수 없습니다. 네트워크를 확인하세요.")

    # yfinance MultiIndex 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].copy()
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info(f"수집 완료: {len(df)}일치 데이터 ({df['Date'].min()} ~ {df['Date'].max()})")
    return df


def detect_shock_days(df: pd.DataFrame, threshold: float = -0.02) -> pd.DataFrame:
    """
    전일 대비 종가 변동률이 threshold 이하인 '충격일(ShockDay)'을 추출한다.
    """
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    shock = df[df["Return"] <= threshold].copy()
    shock = shock.reset_index(drop=True)
    logger.info(f"충격일 탐지: {len(shock)}일 (threshold={threshold:.1%})")
    return shock


def cluster_black_events(shock_days: pd.DataFrame, gap_days: int = 5) -> list[list[pd.Timestamp]]:
    """
    인접 충격일(gap_days 이내)을 하나의 BlackEvent로 클러스터링한다.
    각 BlackEvent의 첫 번째 충격일만 분석 기준점으로 사용.
    Returns: 각 원소가 Timestamp 리스트인 이벤트 리스트
    """
    if shock_days.empty:
        return []

    dates = shock_days["Date"].sort_values().reset_index(drop=True)
    events = []
    cluster_dates = [dates.iloc[0]]

    for i in range(1, len(dates)):
        if (dates.iloc[i] - cluster_dates[-1]).days <= gap_days:
            cluster_dates.append(dates.iloc[i])
        else:
            events.append(cluster_dates)
            cluster_dates = [dates.iloc[i]]
    events.append(cluster_dates)

    logger.info(f"BlackEvent 클러스터링: {len(events)}개 이벤트 (gap={gap_days}일)")
    return events


def compute_metadata(events: list[list[pd.Timestamp]], kospi_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 BlackEvent에 대해 메타데이터를 산출한다.
    - first_shock_date: 첫 충격일
    - duration_days: 이벤트 지속 기간
    - shock_count: 이벤트 내 충격일 횟수
    - max_cumulative_drop: 누적 최대 낙폭 (%)
    """
    records = []
    kospi_indexed = kospi_df.set_index("Date")

    for event_id, cluster_dates in enumerate(events, start=1):
        first = min(cluster_dates)
        last = max(cluster_dates)
        duration = (last - first).days + 1

        # 누적 최대 낙폭: 첫 충격일 전일 종가 대비
        pre_date_mask = kospi_indexed.index < first
        if pre_date_mask.any():
            pre_close = kospi_indexed.loc[pre_date_mask, "Close"].iloc[-1]
        else:
            # 데이터 시작일이 곧 첫 충격일 → 전일 종가가 없어 낙폭 의미 없음
            logger.warning(
                f"BlackEvent #{event_id} ({first.date()}): "
                "전일 데이터 없음 — max_cumulative_drop을 NaN 처리"
            )
            records.append({
                "event_id": event_id,
                "first_shock_date": first,
                "last_shock_date": last,
                "duration_days": duration,
                "shock_count": len(cluster_dates),
                "max_cumulative_drop": np.nan,
            })
            continue

        event_period = kospi_indexed.loc[
            (kospi_indexed.index >= first) & (kospi_indexed.index <= last)
        ]
        if not event_period.empty:
            min_close = event_period["Close"].min()
            max_cum_drop = (min_close - pre_close) / pre_close
        else:
            max_cum_drop = 0.0

        records.append({
            "event_id": event_id,
            "first_shock_date": first,
            "last_shock_date": last,
            "duration_days": duration,
            "shock_count": len(cluster_dates),
            "max_cumulative_drop": round(max_cum_drop * 100, 2),
        })

    df = pd.DataFrame(records)
    logger.info(f"메타데이터 산출 완료: {len(df)}개 BlackEvent")
    return df


def generate_control_dates(
    events_df: pd.DataFrame,
    kospi_df: pd.DataFrame,
    exclusion_days: int = 90,
    multiplier: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    BlackEvent 전후 exclusion_days 이내가 아닌 '평상시' 날짜를 무작위 샘플링한다.
    대조군 수 = BlackEvent 수 × multiplier
    """
    all_dates = set(kospi_df["Date"])

    # BlackEvent 전후 exclusion_days 이내 날짜 제외
    excluded = set()
    for _, row in events_df.iterrows():
        center = row["first_shock_date"]
        for d in range(-exclusion_days, exclusion_days + 1):
            excluded.add(center + pd.Timedelta(days=d))

    candidates = sorted(all_dates - excluded)
    n_control = len(events_df) * multiplier

    rng = np.random.default_rng(seed)
    if len(candidates) < n_control:
        logger.warning(f"대조군 후보({len(candidates)})가 요청({n_control})보다 적습니다. 전체 사용.")
        selected = candidates
    else:
        indices = rng.choice(len(candidates), size=n_control, replace=False)
        selected = [candidates[i] for i in sorted(indices)]

    df = pd.DataFrame({
        "control_id": range(1, len(selected) + 1),
        "date": selected,
    })
    logger.info(f"대조군 생성: {len(df)}일 (BlackEvent {len(events_df)}개 × {multiplier})")
    return df


def search_optimal_gap(
    shock_days: pd.DataFrame,
    kospi_df: pd.DataFrame,
    gap_min: int = 3,
    gap_max: int = 14,
) -> pd.DataFrame:
    """
    gap_days를 gap_min~gap_max 범위로 순회하면서
    각 값별 BlackEvent 수·평균 duration·평균 낙폭 등 비교 테이블을 생성한다.
    사람이 보고 최적값을 판단할 수 있도록 CSV로도 저장.
    """
    records = []

    for gap in range(gap_min, gap_max + 1):
        events = cluster_black_events(shock_days, gap_days=gap)
        if not events:
            records.append({
                "gap_days": gap,
                "n_events": 0,
                "avg_duration": np.nan,
                "median_duration": np.nan,
                "avg_shock_count": np.nan,
                "avg_cumulative_drop": np.nan,
                "median_cumulative_drop": np.nan,
                "min_cumulative_drop": np.nan,
                "max_cumulative_drop_val": np.nan,
            })
            continue

        meta = compute_metadata(events, kospi_df)
        drops = meta["max_cumulative_drop"].dropna()

        records.append({
            "gap_days": gap,
            "n_events": len(meta),
            "avg_duration": round(meta["duration_days"].mean(), 2),
            "median_duration": round(meta["duration_days"].median(), 2),
            "avg_shock_count": round(meta["shock_count"].mean(), 2),
            "avg_cumulative_drop": round(drops.mean(), 2) if len(drops) else np.nan,
            "median_cumulative_drop": round(drops.median(), 2) if len(drops) else np.nan,
            "min_cumulative_drop": round(drops.min(), 2) if len(drops) else np.nan,
            "max_cumulative_drop_val": round(drops.min(), 2) if len(drops) else np.nan,  # 최대 낙폭 (음수가 가장 작은 값)
        })

    result = pd.DataFrame(records)

    logger.info(f"\n=== gap_days 최적값 탐색 ({gap_min}~{gap_max}) ===")
    logger.info(f"\n{result.to_string(index=False)}")
    return result


def run():
    """Module 1 전체 파이프라인 실행"""
    config = load_config()

    # ── 검증 0: 기존 출력 파일 확인 ──
    staleness_report = check_output_staleness(config)
    process_validation_report(staleness_report, logger)

    # 1. 코스피 데이터 수집
    kospi_df = fetch_kospi_data(
        ticker=config["data"]["kospi_ticker"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 원본 저장
    raw_path = get_path(config, "raw_data")
    raw_path.mkdir(parents=True, exist_ok=True)
    kospi_df.to_csv(raw_path / "kospi_daily.csv", index=False)
    logger.info(f"원본 데이터 저장: {raw_path / 'kospi_daily.csv'}")

    # ── 검증 1: 코스피 데이터 무결성 ──
    kospi_report = validate_kospi_data(kospi_df, config)
    process_validation_report(kospi_report, logger)

    # 2. 충격일 탐지
    be_config = config["blackevent"]
    shock_days = detect_shock_days(kospi_df, threshold=be_config["shock_threshold"])

    # ── 검증 2: 충격일 결과 ──
    shock_report = validate_shock_days(shock_days, kospi_df, be_config["shock_threshold"])
    process_validation_report(shock_report, logger)

    # 3. gap_days 최적값 탐색
    gap_range = be_config["cluster_gap_search_range"]
    gap_search_df = search_optimal_gap(
        shock_days, kospi_df,
        gap_min=gap_range["min"],
        gap_max=gap_range["max"],
    )

    # 탐색 결과 저장
    processed_path = get_path(config, "processed_data")
    processed_path.mkdir(parents=True, exist_ok=True)
    gap_search_df.to_csv(processed_path / "gap_days_search.csv", index=False)
    logger.info(f"gap_days 탐색 결과 저장: {processed_path / 'gap_days_search.csv'}")

    # 4. 설정된 초기값으로 최종 BlackEvent 클러스터링
    selected_gap = be_config["cluster_gap_days"]
    logger.info(f"선택된 gap_days: {selected_gap}일 (config 초기값)")

    events = cluster_black_events(shock_days, gap_days=selected_gap)

    # 5. 메타데이터 산출
    events_df = compute_metadata(events, kospi_df)

    # 6. 대조군 생성
    control_df = generate_control_dates(
        events_df,
        kospi_df,
        exclusion_days=be_config["control_exclusion_days"],
        multiplier=be_config["control_multiplier"],
    )

    # ── 검증 3: BlackEvent + 대조군 결과 ──
    event_report = validate_black_events(events_df, control_df, kospi_df, config)
    process_validation_report(event_report, logger)

    # 7. 저장
    events_df.to_csv(processed_path / "black_events.csv", index=False)
    control_df.to_csv(processed_path / "control_dates.csv", index=False)

    logger.info(f"BlackEvent 저장: {processed_path / 'black_events.csv'}")
    logger.info(f"대조군 저장: {processed_path / 'control_dates.csv'}")
    logger.info("=== Module 1 완료 ===")

    return events_df, control_df


if __name__ == "__main__":
    run()
