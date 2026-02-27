"""
Step 8-1: BlackEvent 궤적 역산

각 BlackEvent / 대조군 날짜에 대해 복수 시점(D-30, D-21, D-14, D-7, D-0)의
Risk Score를 역산하여 궤적 벡터를 생성한다.
"""

import numpy as np
import pandas as pd

from src.utils import setup_logging

logger = setup_logging("module8.builder")

# 궤적 시점 (D-day 기준 오프셋)
TRAJECTORY_OFFSETS = [30, 21, 14, 7, 0]


def compute_raw_score(
    freq_map: dict[str, int],
    property_weights: dict[str, float],
    ctrl_freq_means: dict[str, float],
) -> float:
    """Model B raw score를 산출한다 (정규화 없이).

    score = Σ(weight × max(freq / ctrl_mean - 1, 0))
    """
    raw = 0.0
    for prop, weight in property_weights.items():
        freq = freq_map.get(prop, 0)
        ctrl_mean = ctrl_freq_means.get(prop, 1)
        if ctrl_mean > 0 and freq > 0:
            excess = max(freq / ctrl_mean - 1.0, 0.0)
            raw += weight * excess
    return raw


def build_trajectories(
    ref_dates: list[pd.Timestamp],
    labels: list[str],
    dates_arr: np.ndarray,
    daily_counts: list[dict[str, int]],
    articles_per_day: list[int],
    property_weights: dict[str, float],
    ctrl_freq_means: dict[str, float],
    optimal_windows: dict[str, int],
    default_window: int = 60,
) -> pd.DataFrame:
    """기준일 목록에 대해 궤적 벡터를 역산한다.

    Args:
        ref_dates: 기준일 목록 (first_shock_date 또는 control date)
        labels: 각 기준일의 식별자 (event_id 또는 control_id)
        dates_arr: 일별 집계 날짜 배열 (sorted)
        daily_counts: [{property_id: count}, ...] 일별 성질 빈도
        articles_per_day: 일별 기사 수
        property_weights: 성질별 가중치
        ctrl_freq_means: 성질별 대조군 평균 빈도
        optimal_windows: 성질별 최적 윈도우
        default_window: 기본 윈도우 크기

    Returns:
        DataFrame: columns = [id, D-30, D-21, D-14, D-7, D-0]
    """
    from src.module6_scoring.run import compute_window_frequency

    unique_windows = set(optimal_windows.values()) or {default_window}
    rows = []

    for idx, (ref_date, label) in enumerate(zip(ref_dates, labels)):
        ref_ts = pd.Timestamp(ref_date)
        trajectory = []

        for offset in TRAJECTORY_OFFSETS:
            # 이 시점의 가상 기준일
            point_date = ref_ts - pd.Timedelta(days=offset)

            # 성질별 최적 윈도우로 빈도 계산
            window_freqs: dict[int, dict[str, int]] = {}
            for w in unique_windows:
                freq_map, _ = compute_window_frequency(
                    dates_arr, daily_counts, articles_per_day, point_date, w
                )
                window_freqs[w] = freq_map

            # 성질별 최적 윈도우에서 빈도 추출
            combined_freq: dict[str, int] = {}
            for prop in property_weights:
                w = optimal_windows.get(prop, default_window)
                combined_freq[prop] = window_freqs.get(w, {}).get(prop, 0)

            score = compute_raw_score(combined_freq, property_weights, ctrl_freq_means)
            trajectory.append(score)

        rows.append({"id": label, **{f"D-{o}": s for o, s in zip(TRAJECTORY_OFFSETS, trajectory)}})

        if (idx + 1) % 20 == 0 or idx == len(ref_dates) - 1:
            logger.info(f"  궤적 역산: {idx + 1}/{len(ref_dates)}")

    return pd.DataFrame(rows)


def normalize_trajectories(
    df: pd.DataFrame,
    method: str = "minmax",
) -> pd.DataFrame:
    """궤적 벡터를 정규화한다.

    Args:
        df: build_trajectories 결과 (id, D-30, D-21, D-14, D-7, D-0)
        method: "minmax" (궤적 내) 또는 "global" (전체 max 기준)

    Returns:
        정규화된 DataFrame (동일 구조)
    """
    score_cols = [f"D-{o}" for o in TRAJECTORY_OFFSETS]
    result = df.copy()
    vals = result[score_cols].values.astype(float)

    if method == "minmax":
        # 각 행(궤적) 내 min-max 정규화
        for i in range(len(vals)):
            row_min = vals[i].min()
            row_max = vals[i].max()
            rng = row_max - row_min
            if rng > 0:
                vals[i] = (vals[i] - row_min) / rng
            else:
                vals[i] = 0.0
    elif method == "global":
        # 전체 궤적의 글로벌 max로 나눔
        global_max = vals.max()
        if global_max > 0:
            vals = vals / global_max
    else:
        raise ValueError(f"Unknown method: {method}")

    result[score_cols] = vals
    return result
