"""
현재 상태 진단: 성질 프로필 기반 과거 BlackEvent 유사도 매칭

BlackEvent 직전 뉴스의 성질 빈도비 프로필을 사전 계산하고,
현재 뉴스 상태와 비교하여 "과거 어떤 위기 직전과 유사한가"를 판정한다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.module6_scoring.run import load_daily_aggregation, compute_window_frequency
from src.utils import get_path, load_config, setup_logging

logger = setup_logging("diagnosis.profile_matcher")

MATCH_OFFSETS = [0, 7, 14, 21]


def precompute_be_profiles(
    outputs_path: Path | None = None,
    processed_path: Path | None = None,
) -> dict:
    """모든 BlackEvent에 대해 D-0/D-7/D-14/D-21 시점의 성질 빈도비 프로필을 계산한다.

    결과를 be_profiles.json으로 저장하고, dict로도 반환한다.
    """
    if outputs_path is None or processed_path is None:
        config = load_config()
        processed_path = processed_path or get_path(config, "processed_data")
        outputs_path = outputs_path or get_path(config, "outputs")

    # 유의미 성질 로드
    sig_df = pd.read_csv(outputs_path / "significant_properties.csv")
    ctrl_means = dict(zip(sig_df["property_id"], sig_df["ctrl_freq_mean"]))
    opt_windows = dict(
        zip(sig_df["property_id"], sig_df["optimal_window"].astype(int))
    )
    property_ids = sig_df["property_id"].tolist()

    # BlackEvent 로드
    events_df = pd.read_csv(
        processed_path / "black_events.csv",
        parse_dates=["first_shock_date"],
    )

    # 코퍼스 일별 집계 로드
    max_offset = max(MATCH_OFFSETS)
    max_window = max(opt_windows.values()) if opt_windows else 60
    date_min = (
        events_df["first_shock_date"].min()
        - pd.Timedelta(days=max_offset + max_window + 1)
    ).strftime("%Y-%m-%d")
    date_max = events_df["first_shock_date"].max().strftime("%Y-%m-%d")

    logger.info(f"코퍼스 일별 집계 로드 ({date_min} ~ {date_max})...")
    dates_arr, daily_counts, articles_per_day, total = load_daily_aggregation(
        processed_path, date_min=date_min, date_max=date_max
    )
    logger.info(f"일별 집계 완료: {total:,}건, {len(dates_arr):,}일")

    unique_windows = sorted(set(opt_windows.values()))

    # 각 BE × 각 offset의 프로필 계산
    profiles = {}
    for idx, (_, row) in enumerate(events_df.iterrows()):
        event_id = str(int(row["event_id"]))
        be_date = row["first_shock_date"]

        event_profile = {
            "first_shock_date": be_date.strftime("%Y-%m-%d"),
            "max_cumulative_drop": float(row["max_cumulative_drop"]),
            "duration_days": int(row["duration_days"]),
            "shock_count": int(row["shock_count"]),
            "offsets": {},
        }

        for offset in MATCH_OFFSETS:
            ref_date = be_date - pd.Timedelta(days=offset)

            # 고유 윈도우별 빈도 한 번만 계산
            window_freqs = {}
            for w in unique_windows:
                freq_map, _ = compute_window_frequency(
                    dates_arr, daily_counts, articles_per_day, ref_date, w
                )
                window_freqs[w] = freq_map

            # 성질별 빈도비
            ratios = {}
            for prop in property_ids:
                w = opt_windows[prop]
                freq = window_freqs.get(w, {}).get(prop, 0)
                cm = ctrl_means[prop]
                if cm > 0 and freq > 0:
                    ratios[prop] = round(freq / cm, 4)
                else:
                    ratios[prop] = 0.0

            event_profile["offsets"][f"D-{offset}"] = ratios

        profiles[event_id] = event_profile

        if (idx + 1) % 20 == 0 or idx == len(events_df) - 1:
            logger.info(f"  프로필 계산: {idx + 1}/{len(events_df)}")

    result = {
        "match_offsets": MATCH_OFFSETS,
        "property_ids": property_ids,
        "profiles": profiles,
    }

    out_path = outputs_path / "be_profiles.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"BE 프로필 저장: {out_path} ({len(profiles)}건)")

    return result


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도를 계산한다."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def find_similar_events(
    current_ratios: dict[str, float],
    be_profiles_path: Path,
    top_n: int = 10,
) -> list[dict]:
    """현재 성질 프로필과 가장 유사한 과거 BlackEvent를 찾는다.

    Args:
        current_ratios: {property_id: ratio} — 현재 날짜의 성질별 빈도비
        be_profiles_path: be_profiles.json 경로
        top_n: 반환할 상위 매칭 수

    Returns:
        유사도 내림차순 정렬된 매칭 리스트
    """
    with open(be_profiles_path, encoding="utf-8") as f:
        data = json.load(f)

    property_ids = data["property_ids"]
    profiles = data["profiles"]

    # 현재 벡터 구성
    current_vec = np.array(
        [current_ratios.get(pid, 0.0) for pid in property_ids]
    )

    matches = []
    for event_id, profile in profiles.items():
        for offset_key, ratios in profile["offsets"].items():
            be_vec = np.array([ratios.get(pid, 0.0) for pid in property_ids])
            sim = _cosine_similarity(current_vec, be_vec)

            matches.append(
                {
                    "event_id": int(event_id),
                    "offset": offset_key,
                    "similarity": round(sim, 4),
                    "first_shock_date": profile["first_shock_date"],
                    "max_cumulative_drop": profile["max_cumulative_drop"],
                    "duration_days": profile["duration_days"],
                    "shock_count": profile["shock_count"],
                }
            )

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches[:top_n]


def compute_score_percentile(
    current_score: float,
    risk_scores_path: Path,
) -> dict:
    """현재 스코어의 전체 이력 내 퍼센타일을 계산한다.

    Args:
        current_score: 현재 risk_score (0~100)
        risk_scores_path: risk_scores.csv 경로

    Returns:
        퍼센타일 정보 dict
    """
    df = pd.read_csv(risk_scores_path)
    all_scores = df["model_b_risk_score"].values
    be_scores = df.loc[df["actual_label"] == 1, "model_b_risk_score"].values
    ctrl_scores = df.loc[df["actual_label"] == 0, "model_b_risk_score"].values

    overall_pct = float(np.mean(all_scores <= current_score) * 100)
    be_pct = float(np.mean(be_scores <= current_score) * 100) if len(be_scores) > 0 else 0.0
    ctrl_pct = float(np.mean(ctrl_scores <= current_score) * 100) if len(ctrl_scores) > 0 else 0.0

    return {
        "overall_percentile": round(overall_pct, 1),
        "be_percentile": round(be_pct, 1),
        "ctrl_percentile": round(ctrl_pct, 1),
        "total_samples": len(all_scores),
        "be_mean": round(float(be_scores.mean()), 2) if len(be_scores) > 0 else 0.0,
        "ctrl_mean": round(float(ctrl_scores.mean()), 2) if len(ctrl_scores) > 0 else 0.0,
    }


if __name__ == "__main__":
    precompute_be_profiles()
