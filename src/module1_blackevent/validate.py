"""
Module 1 — 데이터 전처리 검증
코스피 데이터, 충격일, BlackEvent 클러스터링 결과의 무결성을 검증한다.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils import load_config, get_path, setup_logging

logger = setup_logging("module1.validate")

# ──────────────────────────────────────────────
# 심각도 상수
# ──────────────────────────────────────────────
CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"

# ──────────────────────────────────────────────
# 검증 임계값 (비즈니스 파라미터가 아닌 메타 파라미터)
# ──────────────────────────────────────────────
MAX_GAP_CALENDAR_DAYS = 5       # 주말/공휴일 고려 시 5일 초과 공백은 의심
MIN_TRADING_DAYS = 200          # 최소 거래일 수
NAN_CRITICAL_RATIO = 0.01       # NaN 1% 초과 시 CRITICAL
SHOCK_MAX_RATIO = 0.30          # 충격일 30% 초과 시 데이터 오류 의심
SHOCK_MIN_RATIO = 0.005         # 충격일 0.5% 미만 시 경고
EXTREME_DROP_THRESHOLD = -0.20  # -20% 이하 단일일 변동 경고
MAX_EVENT_DURATION_DAYS = 365   # 365일 초과 이벤트 경고
MIN_EVENTS = 3                  # 통계 분석에 필요한 최소 이벤트 수


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def _issue(severity: str, check: str, message: str, details: dict | None = None) -> dict:
    return {
        "severity": severity,
        "check": check,
        "message": message,
        "details": details or {},
    }


def _build_report(stage: str, issues: list[dict], summary: dict | None = None) -> dict:
    has_critical = any(i["severity"] == CRITICAL for i in issues)
    return {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "passed": not has_critical,
        "issues": issues,
        "summary": summary or {},
    }


# ──────────────────────────────────────────────
# Stage 0: 출력 파일 스테일 체크
# ──────────────────────────────────────────────
def check_output_staleness(config: dict) -> dict:
    """기존 출력 파일 존재 여부를 확인하고 덮어쓰기 경고를 발생시킨다."""
    issues = []
    processed = get_path(config, "processed_data")
    raw = get_path(config, "raw_data")

    targets = [
        (processed / "black_events.csv", "processed"),
        (processed / "control_dates.csv", "processed"),
        (processed / "gap_days_search.csv", "processed"),
        (raw / "kospi_daily.csv", "raw"),
    ]

    existing_files = []
    for fpath, category in targets:
        if fpath.exists():
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            existing_files.append(fpath.name)
            issues.append(_issue(
                WARNING,
                "output_file_exists",
                f"기존 파일 발견: {fpath.name} (수정: {mtime:%Y-%m-%d %H:%M}) — 덮어쓰기 예정",
                {"file": str(fpath), "modified": mtime.isoformat()},
            ))

    if not existing_files:
        issues.append(_issue(INFO, "output_files_clean", "기존 출력 파일 없음 — 첫 실행"))

    return _build_report("output_staleness", issues, {"existing_files": existing_files})


# ──────────────────────────────────────────────
# Stage 1: 코스피 데이터 무결성 검증
# ──────────────────────────────────────────────
def _check_date_range_coverage(df: pd.DataFrame, config: dict) -> list[dict]:
    issues = []
    actual_start = df["Date"].min()
    actual_end = df["Date"].max()
    config_start = pd.Timestamp(config["data"]["start_date"])
    config_end = (
        pd.Timestamp(config["data"]["end_date"])
        if config["data"]["end_date"]
        else pd.Timestamp.now()
    )

    start_diff = (actual_start - config_start).days
    end_diff = (config_end - actual_end).days

    issues.append(_issue(
        INFO,
        "date_range_info",
        f"데이터 기간: {actual_start.date()} ~ {actual_end.date()} ({len(df):,}거래일)",
        {"actual_start": str(actual_start.date()), "actual_end": str(actual_end.date())},
    ))

    if start_diff > 30:
        issues.append(_issue(
            WARNING,
            "date_range_start_gap",
            f"설정 시작일({config_start.date()}) 대비 실제 시작일이 {start_diff}일 늦음",
            {"config_start": str(config_start.date()), "actual_start": str(actual_start.date())},
        ))

    if end_diff > 7:
        issues.append(_issue(
            WARNING,
            "date_range_end_gap",
            f"설정 종료일({config_end.date()}) 대비 실제 종료일이 {end_diff}일 이전",
            {"config_end": str(config_end.date()), "actual_end": str(actual_end.date())},
        ))

    return issues


def _check_missing_trading_days(df: pd.DataFrame) -> list[dict]:
    issues = []
    dates = df["Date"].sort_values().reset_index(drop=True)
    gaps = []

    for i in range(1, len(dates)):
        diff = (dates.iloc[i] - dates.iloc[i - 1]).days
        if diff > MAX_GAP_CALENDAR_DAYS:
            gaps.append({
                "from": str(dates.iloc[i - 1].date()),
                "to": str(dates.iloc[i].date()),
                "gap_days": diff,
            })

    if gaps:
        for g in gaps:
            issues.append(_issue(
                WARNING,
                "missing_trading_days",
                f"의심스러운 거래일 갭: {g['from']} ~ {g['to']} ({g['gap_days']}일)",
                g,
            ))
    else:
        issues.append(_issue(INFO, "no_trading_gaps", "비정상 거래일 갭 없음"))

    return issues


def _check_nan_close_prices(df: pd.DataFrame) -> list[dict]:
    issues = []
    nan_count = df["Close"].isna().sum()

    if nan_count == 0:
        return issues

    nan_ratio = nan_count / len(df)
    nan_dates = df.loc[df["Close"].isna(), "Date"].dt.date.tolist()

    if nan_ratio > NAN_CRITICAL_RATIO:
        issues.append(_issue(
            CRITICAL,
            "nan_close_critical",
            f"Close 가격에 NaN 값 {nan_count}건 발견 ({nan_ratio:.1%}) — 허용 한계({NAN_CRITICAL_RATIO:.0%}) 초과",
            {"nan_count": nan_count, "nan_ratio": nan_ratio, "nan_dates": [str(d) for d in nan_dates[:10]]},
        ))
    else:
        issues.append(_issue(
            WARNING,
            "nan_close_warning",
            f"Close 가격에 NaN 값 {nan_count}건 발견 ({nan_ratio:.1%})",
            {"nan_count": nan_count, "nan_dates": [str(d) for d in nan_dates]},
        ))

    return issues


def _check_duplicate_dates(df: pd.DataFrame) -> list[dict]:
    issues = []
    dup_mask = df["Date"].duplicated(keep=False)
    dup_count = dup_mask.sum()

    if dup_count > 0:
        dup_dates = df.loc[dup_mask, "Date"].dt.date.unique().tolist()
        issues.append(_issue(
            WARNING,
            "duplicate_dates",
            f"중복 날짜 {len(dup_dates)}건 발견: {', '.join(str(d) for d in dup_dates[:5])}",
            {"duplicate_dates": [str(d) for d in dup_dates]},
        ))

    return issues


def _check_non_positive_prices(df: pd.DataFrame) -> list[dict]:
    issues = []
    bad_mask = df["Close"].dropna() <= 0

    if bad_mask.any():
        bad_count = bad_mask.sum()
        issues.append(_issue(
            CRITICAL,
            "non_positive_close",
            f"비양수 종가 {bad_count}건 발견 — 데이터 손상",
            {"count": int(bad_count)},
        ))

    return issues


def _check_minimum_data_volume(df: pd.DataFrame) -> list[dict]:
    issues = []

    if len(df) < MIN_TRADING_DAYS:
        issues.append(_issue(
            CRITICAL,
            "insufficient_data",
            f"거래일 {len(df)}일 — 최소 {MIN_TRADING_DAYS}일 필요",
            {"actual": len(df), "minimum": MIN_TRADING_DAYS},
        ))

    return issues


def validate_kospi_data(df: pd.DataFrame, config: dict) -> dict:
    """코스피 데이터 무결성을 검증한다."""
    issues = []
    issues.extend(_check_date_range_coverage(df, config))
    issues.extend(_check_missing_trading_days(df))
    issues.extend(_check_nan_close_prices(df))
    issues.extend(_check_duplicate_dates(df))
    issues.extend(_check_non_positive_prices(df))
    issues.extend(_check_minimum_data_volume(df))

    summary = {
        "total_trading_days": len(df),
        "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "nan_count": int(df["Close"].isna().sum()),
    }

    return _build_report("kospi_data", issues, summary)


# ──────────────────────────────────────────────
# Stage 2: 충격일 검증
# ──────────────────────────────────────────────
def _check_shock_day_count(shock_days: pd.DataFrame, total_days: int) -> list[dict]:
    issues = []
    n_shock = len(shock_days)

    if n_shock == 0:
        issues.append(_issue(
            CRITICAL,
            "zero_shock_days",
            "충격일 0건 — 파이프라인 진행 불가",
        ))
        return issues

    ratio = n_shock / total_days

    issues.append(_issue(
        INFO,
        "shock_day_count",
        f"충격일 {n_shock}일 탐지 (전체 {total_days:,}일 대비 {ratio:.1%})",
        {"shock_days": n_shock, "total_days": total_days, "ratio": ratio},
    ))

    if ratio > SHOCK_MAX_RATIO:
        issues.append(_issue(
            CRITICAL,
            "shock_ratio_too_high",
            f"충격일 비율 {ratio:.1%} — 전체의 {SHOCK_MAX_RATIO:.0%} 초과 (데이터 오류 의심)",
            {"ratio": ratio},
        ))
    elif ratio < SHOCK_MIN_RATIO:
        issues.append(_issue(
            WARNING,
            "shock_ratio_too_low",
            f"충격일 비율 {ratio:.1%} — 비정상적으로 적음 (임계값 또는 데이터 확인 필요)",
            {"ratio": ratio},
        ))

    return issues


def _check_extreme_drops(shock_days: pd.DataFrame) -> list[dict]:
    issues = []

    if "Return" not in shock_days.columns or shock_days.empty:
        return issues

    extreme = shock_days[shock_days["Return"] <= EXTREME_DROP_THRESHOLD]

    for _, row in extreme.iterrows():
        issues.append(_issue(
            WARNING,
            "extreme_daily_drop",
            f"극단적 일일 변동: {row['Date'].date()} ({row['Return']:.2%}) — 데이터 오류 가능성 확인 필요",
            {"date": str(row["Date"].date()), "return": round(float(row["Return"]), 4)},
        ))

    return issues


def validate_shock_days(
    shock_days: pd.DataFrame,
    kospi_df: pd.DataFrame,
    threshold: float,
) -> dict:
    """충격일 검출 결과를 검증한다."""
    issues = []
    issues.extend(_check_shock_day_count(shock_days, len(kospi_df)))
    issues.extend(_check_extreme_drops(shock_days))

    summary = {
        "shock_day_count": len(shock_days),
        "total_trading_days": len(kospi_df),
        "threshold": threshold,
    }

    return _build_report("shock_days", issues, summary)


# ──────────────────────────────────────────────
# Stage 3: BlackEvent + 대조군 검증
# ──────────────────────────────────────────────
def _check_event_count(events_df: pd.DataFrame) -> list[dict]:
    issues = []
    n = len(events_df)

    if n == 0:
        issues.append(_issue(
            CRITICAL,
            "zero_events",
            "BlackEvent 0개 — 파이프라인 진행 불가",
        ))
    elif n < MIN_EVENTS:
        issues.append(_issue(
            WARNING,
            "few_events",
            f"BlackEvent {n}개 — 통계 분석에 최소 {MIN_EVENTS}개 권장",
            {"event_count": n, "minimum": MIN_EVENTS},
        ))
    else:
        issues.append(_issue(
            INFO,
            "event_count",
            f"BlackEvent {n}개 탐지",
            {"event_count": n},
        ))

    return issues


def _check_event_duration(events_df: pd.DataFrame) -> list[dict]:
    issues = []

    if "duration_days" not in events_df.columns or events_df.empty:
        return issues

    long_events = events_df[events_df["duration_days"] > MAX_EVENT_DURATION_DAYS]

    for _, row in long_events.iterrows():
        issues.append(_issue(
            WARNING,
            "long_event_duration",
            f"BlackEvent #{int(row['event_id'])} 기간 {int(row['duration_days'])}일 "
            f"({row['first_shock_date']} ~ {row['last_shock_date']}) — gap_days 파라미터 확인 필요",
            {"event_id": int(row["event_id"]), "duration": int(row["duration_days"])},
        ))

    return issues


def _check_control_dates(
    control_df: pd.DataFrame,
    events_df: pd.DataFrame,
    multiplier: int,
) -> list[dict]:
    issues = []
    n_control = len(control_df)
    expected = len(events_df) * multiplier

    if n_control == 0:
        issues.append(_issue(
            CRITICAL,
            "zero_control_dates",
            "대조군 0개 — 파이프라인 진행 불가",
        ))
    elif n_control < expected:
        issues.append(_issue(
            WARNING,
            "insufficient_control_dates",
            f"대조군 {n_control}개 — 요청 {expected}개 대비 부족",
            {"actual": n_control, "expected": expected},
        ))
    else:
        issues.append(_issue(
            INFO,
            "control_dates_ok",
            f"대조군 {n_control}개 생성 (BlackEvent {len(events_df)}개 × {multiplier})",
            {"control_count": n_control, "event_count": len(events_df)},
        ))

    return issues


def validate_black_events(
    events_df: pd.DataFrame,
    control_df: pd.DataFrame,
    kospi_df: pd.DataFrame,
    config: dict,
) -> dict:
    """BlackEvent 클러스터링 및 대조군 생성 결과를 검증한다."""
    issues = []
    be_config = config["blackevent"]

    issues.extend(_check_event_count(events_df))
    issues.extend(_check_event_duration(events_df))
    issues.extend(_check_control_dates(control_df, events_df, be_config["control_multiplier"]))

    summary = {
        "event_count": len(events_df),
        "control_count": len(control_df),
    }

    return _build_report("black_events", issues, summary)


# ──────────────────────────────────────────────
# 보고서 처리
# ──────────────────────────────────────────────
def process_validation_report(report: dict, log: logging.Logger) -> None:
    """
    검증 보고서를 로깅하고, CRITICAL 이슈가 있으면 예외를 발생시킨다.
    """
    stage = report["stage"]
    issues = report["issues"]

    if not issues:
        log.info(f"[검증:{stage}] 모든 검사 통과 ✓")
        return

    critical_issues = []

    for issue in issues:
        severity = issue["severity"]
        msg = f"[검증:{stage}] [{severity}] {issue['message']}"

        if severity == INFO:
            log.info(msg)
        elif severity == WARNING:
            log.warning(msg)
        elif severity == CRITICAL:
            log.error(msg)
            critical_issues.append(issue)

    if not critical_issues:
        log.info(f"[검증:{stage}] CRITICAL 이슈 없음 — 계속 진행")

    if critical_issues:
        messages = "; ".join(i["message"] for i in critical_issues)
        raise ValueError(f"Module 1 검증 실패 ({stage}): {messages}")
