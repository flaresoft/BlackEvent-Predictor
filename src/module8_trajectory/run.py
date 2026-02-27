"""
Module 8 — Crisis Trajectory Classification (위기 궤적 유형 분류)

BlackEvent 스코어 궤적 역산 → 비지도 유형 분류 → 라이브 궤적 매칭

Usage:
    python -m src.module8_trajectory.run
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, get_path, setup_logging
from src.module6_scoring.run import load_daily_aggregation
from src.module8_trajectory.trajectory_builder import (
    TRAJECTORY_OFFSETS,
    build_trajectories,
    normalize_trajectories,
)
from src.module8_trajectory.trajectory_clustering import (
    cluster_trajectories,
    build_trajectory_dictionary,
)
from src.module8_trajectory.trajectory_matcher import (
    match_trajectory,
    calibrate_thresholds,
    apply_confidence,
)
from src.module8_trajectory.trajectory_validator import (
    validate_control_separation,
    validate_leave_one_out,
    validate_time_split,
)

logger = setup_logging("module8")


def _init_model_b(sig_df: pd.DataFrame) -> tuple[dict, dict, dict, int]:
    """significant_properties.csv에서 Model B 파라미터를 추출한다.

    Returns:
        (property_weights, ctrl_freq_means, optimal_windows, default_window)
    """
    property_weights: dict[str, float] = {}
    ctrl_freq_means: dict[str, float] = {}
    optimal_windows: dict[str, int] = {}

    if "p_adjusted" in sig_df.columns:
        raw_w = -np.log10(sig_df["p_adjusted"].clip(lower=1e-300))
        max_w = raw_w.max()
        if max_w > 0:
            normed = raw_w / max_w
            for i, (_, row) in enumerate(sig_df.iterrows()):
                property_weights[row["property_id"]] = normed.iloc[i]

    if "ctrl_freq_mean" in sig_df.columns:
        for _, row in sig_df.iterrows():
            ctrl_freq_means[row["property_id"]] = row["ctrl_freq_mean"]

    if "optimal_window" in sig_df.columns:
        for _, row in sig_df.iterrows():
            optimal_windows[row["property_id"]] = int(row["optimal_window"])

    default_window = max(optimal_windows.values()) if optimal_windows else 60
    return property_weights, ctrl_freq_means, optimal_windows, default_window


def run():
    """Module 8 전체 파이프라인 실행"""
    t0 = time.time()
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")

    logger.info("=== Module 8: Crisis Trajectory Classification ===")

    # ── 입력 데이터 로드 ──
    events_df = pd.read_csv(
        processed_path / "black_events.csv",
        parse_dates=["first_shock_date"],
    )
    control_df = pd.read_csv(
        processed_path / "control_dates.csv",
        parse_dates=["date"],
    )
    sig_df = pd.read_csv(outputs_path / "significant_properties.csv")

    logger.info(
        f"입력: BlackEvent {len(events_df)}건, 대조군 {len(control_df)}건, "
        f"유의미 성질 {len(sig_df)}개"
    )

    # ── Model B 파라미터 ──
    property_weights, ctrl_freq_means, optimal_windows, default_window = _init_model_b(sig_df)
    logger.info(f"Model B: {len(property_weights)}개 성질, default_window={default_window}")

    # ── 코퍼스 일별 집계 ──
    all_dates = pd.concat([events_df["first_shock_date"], control_df["date"]])
    max_offset = max(TRAJECTORY_OFFSETS)
    max_window = max(optimal_windows.values()) if optimal_windows else default_window
    date_min = (all_dates.min() - pd.Timedelta(days=max_offset + max_window + 5)).strftime("%Y-%m-%d")
    date_max = all_dates.max().strftime("%Y-%m-%d")

    logger.info(f"코퍼스 로드: {date_min} ~ {date_max}")
    dates_arr, daily_counts, articles_per_day, total = load_daily_aggregation(
        processed_path, date_min=date_min, date_max=date_max
    )
    logger.info(f"코퍼스: {total:,}건, {len(dates_arr):,}일")

    # ════════════════════════════════════════════
    # Step 8-1: 궤적 역산
    # ════════════════════════════════════════════
    logger.info("--- Step 8-1: BlackEvent 궤적 역산 ---")

    be_ref_dates = events_df["first_shock_date"].tolist()
    be_labels = [str(eid) for eid in events_df["event_id"].tolist()]

    be_traj_raw = build_trajectories(
        be_ref_dates, be_labels,
        dates_arr, daily_counts, articles_per_day,
        property_weights, ctrl_freq_means, optimal_windows, default_window,
    )
    logger.info(f"BlackEvent 궤적: {len(be_traj_raw)}건")

    logger.info("--- Step 8-1: 대조군 궤적 역산 ---")
    ctrl_ref_dates = control_df["date"].tolist()
    ctrl_labels = [str(cid) for cid in control_df["control_id"].tolist()]

    ctrl_traj_raw = build_trajectories(
        ctrl_ref_dates, ctrl_labels,
        dates_arr, daily_counts, articles_per_day,
        property_weights, ctrl_freq_means, optimal_windows, default_window,
    )
    logger.info(f"대조군 궤적: {len(ctrl_traj_raw)}건")

    # 궤적 CSV 저장 (raw)
    be_traj_raw.to_csv(outputs_path / "trajectory_vectors.csv", index=False)
    ctrl_traj_raw.to_csv(outputs_path / "trajectory_vectors_control.csv", index=False)
    logger.info("궤적 벡터 저장 완료")

    # ── 정규화 (방법 C: 양쪽 모두) ──
    be_minmax = normalize_trajectories(be_traj_raw, method="minmax")
    be_global = normalize_trajectories(be_traj_raw, method="global")
    ctrl_minmax = normalize_trajectories(ctrl_traj_raw, method="minmax")
    ctrl_global = normalize_trajectories(ctrl_traj_raw, method="global")

    # ════════════════════════════════════════════
    # Step 8-2: 궤적 클러스터링
    # ════════════════════════════════════════════
    logger.info("--- Step 8-2: HDBSCAN 궤적 클러스터링 ---")

    results_by_method = {}
    for method_name, be_norm, ctrl_norm, alpha in [
        ("minmax", be_minmax, ctrl_minmax, 0.7),
        ("global", be_global, ctrl_global, 0.5),
    ]:
        logger.info(f"\n[{method_name} 정규화]")
        labels, best_mcs, search = cluster_trajectories(
            be_norm, min_cluster_size_range=range(5, 21)
        )
        traj_dict = build_trajectory_dictionary(be_norm, labels)

        n_clusters = len([d for d in traj_dict if d["cluster_label"] != -1])
        n_noise = sum(1 for l in labels if l == -1)
        logger.info(f"  결과: {n_clusters} 유형, {n_noise} noise")

        results_by_method[method_name] = {
            "labels": labels,
            "best_mcs": best_mcs,
            "search": search,
            "dictionary": traj_dict,
            "alpha": alpha,
            "be_norm": be_norm,
            "ctrl_norm": ctrl_norm,
        }

    # ── 최적 방법 선택 (실루엣 스코어 기준) ──
    best_method = max(
        results_by_method.keys(),
        key=lambda m: results_by_method[m]["search"][results_by_method[m]["best_mcs"]]["silhouette"],
    )
    logger.info(f"\n최적 정규화: {best_method}")

    chosen = results_by_method[best_method]
    labels = chosen["labels"]
    best_mcs = chosen["best_mcs"]
    traj_dict = chosen["dictionary"]
    alpha = chosen["alpha"]
    be_norm = chosen["be_norm"]
    ctrl_norm = chosen["ctrl_norm"]

    # 유형 사전 저장
    def _to_native(obj):
        """numpy 타입을 Python 네이티브로 변환."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(v) for v in obj]
        return obj

    dict_output = _to_native({
        "normalization_method": best_method,
        "min_cluster_size": best_mcs,
        "alpha": alpha,
        "trajectory_offsets": TRAJECTORY_OFFSETS,
        "types": traj_dict,
        "search_results": {
            method: results_by_method[method]["search"]
            for method in results_by_method
        },
    })
    with open(outputs_path / "trajectory_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(dict_output, f, ensure_ascii=False, indent=2)
    logger.info("유형 사전 저장: trajectory_dictionary.json")

    # ════════════════════════════════════════════
    # Step 8-3: 대조군 캘리브레이션 + 검증
    # ════════════════════════════════════════════
    logger.info("--- Step 8-3: 대조군 캘리브레이션 ---")

    score_cols = [f"D-{o}" for o in TRAJECTORY_OFFSETS]
    ctrl_vectors = ctrl_norm[score_cols].values.astype(float)

    cal = calibrate_thresholds(ctrl_vectors, traj_dict, alpha=alpha)
    logger.info(f"임계값: {cal['threshold']}, margin: {cal['margin']}")

    # ── 대조군 분리 검증 ──
    logger.info("--- 검증 1: 대조군 분리 ---")
    ctrl_sep = validate_control_separation(ctrl_norm, traj_dict, alpha=alpha)
    logger.info(
        f"대조군 매칭: mean={ctrl_sep['mean_similarity']:.4f}, "
        f"max={ctrl_sep['max_similarity']:.4f}, p95={ctrl_sep['p95_similarity']:.4f}"
    )

    # ── Leave-One-Out 검증 ──
    logger.info("--- 검증 2: Leave-One-Out ---")
    loo_df = validate_leave_one_out(be_norm, min_cluster_size=best_mcs, alpha=alpha)
    loo_df.to_csv(outputs_path / "trajectory_validation.csv", index=False)

    # LOO 요약
    loo_match = (loo_df["full_match_type"] == loo_df["partial_match_type"]).mean()
    loo_mean_sim = loo_df["partial_match_sim"].mean()
    logger.info(
        f"LOO 결과: 유형 일치율 {loo_match:.1%}, "
        f"부분매칭 평균 유사도 {loo_mean_sim:.4f}"
    )

    # ── 시간순 분할 검증 ──
    logger.info("--- 검증 3: 시간순 분할 ---")
    time_split = validate_time_split(
        be_norm, events_df, split_year=2020,
        min_cluster_size=best_mcs, alpha=alpha,
    )
    if not time_split.get("skipped"):
        logger.info(
            f"시간순 검증: 매칭 {time_split['matched_count']}/{time_split['n_test']}, "
            f"평균 유사도 {time_split['mean_similarity']:.4f}"
        )

    # ════════════════════════════════════════════
    # 종합 리포트
    # ════════════════════════════════════════════
    elapsed = time.time() - t0
    logger.info(f"\n=== Module 8 완료 ({elapsed:.1f}초) ===")

    # 유형별 요약
    logger.info("\n--- 궤적 유형 요약 ---")
    for entry in traj_dict:
        if entry["cluster_label"] == -1:
            logger.info(f"  {entry['trajectory_type_id']}: {entry['cluster_size']}건 (노이즈)")
        else:
            centroid = entry["centroid"]
            shape = _describe_shape(centroid)
            logger.info(
                f"  {entry['trajectory_type_id']}: {entry['cluster_size']}건, "
                f"centroid={[f'{v:.3f}' for v in centroid]}, 형태={shape}"
            )

    # 라이브 매칭용 결과 저장
    live_output = {
        "calibration": cal,
        "control_separation": ctrl_sep,
        "loo_type_accuracy": round(loo_match, 4),
        "loo_mean_similarity": round(loo_mean_sim, 4),
        "time_split": {k: v for k, v in time_split.items() if k != "test_results"},
        "normalization_method": best_method,
        "alpha": alpha,
    }
    with open(outputs_path / "live_trajectory_match.json", "w", encoding="utf-8") as f:
        json.dump(live_output, f, ensure_ascii=False, indent=2)

    logger.info("출력 파일:")
    logger.info("  trajectory_vectors.csv")
    logger.info("  trajectory_vectors_control.csv")
    logger.info("  trajectory_dictionary.json")
    logger.info("  trajectory_validation.csv")
    logger.info("  live_trajectory_match.json")

    return dict_output


def _describe_shape(centroid: list[float]) -> str:
    """centroid 모양을 간단히 기술한다 (사후 해석 보조)."""
    if len(centroid) < 3:
        return "unknown"

    vals = np.array(centroid)
    rng = vals.max() - vals.min()

    if rng < 0.1:
        return "flat"

    # 후반 급등 여부
    latter_rise = vals[-1] - vals[0]
    mid = len(vals) // 2
    early_avg = vals[:mid].mean()
    late_avg = vals[mid:].mean()

    if early_avg < 0.3 and late_avg > 0.6:
        return "late-surge"
    if early_avg > 0.4 and late_avg > 0.6:
        return "gradual-build"
    if vals.std() < 0.15:
        return "plateau"
    if vals[-1] > vals[-2] > vals[-3]:
        return "accelerating"
    return "mixed"


if __name__ == "__main__":
    run()
