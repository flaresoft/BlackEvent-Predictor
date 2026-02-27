"""
검증: Leave-One-Out + 시간순 분할 + 대조군 분리 검증
"""

import numpy as np
import pandas as pd

from src.module8_trajectory.trajectory_builder import TRAJECTORY_OFFSETS
from src.module8_trajectory.trajectory_clustering import (
    cluster_trajectories,
    build_trajectory_dictionary,
)
from src.module8_trajectory.trajectory_matcher import (
    match_trajectory,
    compute_similarity,
)
from src.utils import setup_logging

logger = setup_logging("module8.validator")


def validate_control_separation(
    ctrl_traj_df: pd.DataFrame,
    trajectory_dict: list[dict],
    alpha: float = 0.5,
) -> dict:
    """대조군 궤적이 위기 유형에 매칭되지 않는지 검증한다."""
    score_cols = [f"D-{o}" for o in TRAJECTORY_OFFSETS]
    ctrl_vectors = ctrl_traj_df[score_cols].values.astype(float)

    ctrl_sims = []
    for vec in ctrl_vectors:
        result = match_trajectory(vec, trajectory_dict, alpha)
        ctrl_sims.append(result["similarity"])

    ctrl_sims = np.array(ctrl_sims)
    return {
        "n_control": len(ctrl_sims),
        "mean_similarity": round(float(ctrl_sims.mean()), 4),
        "std_similarity": round(float(ctrl_sims.std()), 4),
        "max_similarity": round(float(ctrl_sims.max()), 4),
        "p95_similarity": round(float(np.percentile(ctrl_sims, 95)), 4),
    }


def validate_leave_one_out(
    be_traj_df: pd.DataFrame,
    min_cluster_size: int,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Leave-One-Out 검증: 1건 빼고 140건으로 유형 분류 → 빠진 1건 매칭.

    부분 궤적(D-30~D-14, 앞 3시점)으로 매칭한 결과가
    전체 궤적 분류 결과와 일치하는지 확인한다.
    """
    score_cols = [f"D-{o}" for o in TRAJECTORY_OFFSETS]
    n = len(be_traj_df)
    results = []

    # 전체 궤적으로 한 번 클러스터링 (정답 라벨)
    from hdbscan import HDBSCAN
    X_all = be_traj_df[score_cols].values.astype(float)
    full_clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size,
        metric="euclidean",
    )
    full_labels = full_clusterer.fit_predict(X_all)

    for i in range(n):
        # Leave-one-out: i번째 제외
        train_df = be_traj_df.drop(be_traj_df.index[i]).reset_index(drop=True)
        test_vec = X_all[i]
        true_label = full_labels[i]
        event_id = be_traj_df.iloc[i]["id"]

        # 나머지로 클러스터링
        X_train = train_df[score_cols].values.astype(float)
        loo_clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size,
            metric="euclidean",
        )
        loo_labels = loo_clusterer.fit_predict(X_train)
        loo_dict = build_trajectory_dictionary(train_df, loo_labels)

        # 부분 궤적 매칭 (앞 3시점: D-30, D-21, D-14)
        partial_vec = test_vec[:3]
        # 전체 벡터로도 매칭 (검증용)
        full_result = match_trajectory(test_vec, loo_dict, alpha)
        partial_result = match_trajectory(partial_vec, loo_dict, alpha)

        results.append({
            "event_id": event_id,
            "true_label": int(true_label),
            "full_match_type": full_result["best_match_type"],
            "full_match_sim": full_result["similarity"],
            "partial_match_type": partial_result["best_match_type"],
            "partial_match_phase": partial_result["best_match_phase"],
            "partial_match_sim": partial_result["similarity"],
        })

        if (i + 1) % 20 == 0 or i == n - 1:
            logger.info(f"  LOO 검증: {i + 1}/{n}")

    return pd.DataFrame(results)


def validate_time_split(
    be_traj_df: pd.DataFrame,
    events_df: pd.DataFrame,
    split_year: int = 2020,
    min_cluster_size: int = 10,
    alpha: float = 0.5,
) -> dict:
    """시간순 분할 검증: split_year 이전으로 학습, 이후로 테스트."""
    score_cols = [f"D-{o}" for o in TRAJECTORY_OFFSETS]

    # event_id로 날짜 매핑
    date_map = {}
    for _, row in events_df.iterrows():
        date_map[str(row["event_id"])] = pd.Timestamp(row["first_shock_date"])

    train_mask = []
    test_mask = []
    for _, row in be_traj_df.iterrows():
        eid = str(row["id"])
        dt = date_map.get(eid)
        if dt is not None and dt.year < split_year:
            train_mask.append(True)
            test_mask.append(False)
        else:
            train_mask.append(False)
            test_mask.append(True)

    train_df = be_traj_df[train_mask].reset_index(drop=True)
    test_df = be_traj_df[test_mask].reset_index(drop=True)

    logger.info(f"시간순 분할: 학습 {len(train_df)}건 (<{split_year}), 테스트 {len(test_df)}건")

    if len(train_df) < min_cluster_size * 2:
        logger.warning("학습 데이터 부족 — 시간순 분할 검증 스킵")
        return {"skipped": True, "reason": "insufficient_train_data"}

    # 학습 데이터로 클러스터링
    from hdbscan import HDBSCAN
    X_train = train_df[score_cols].values.astype(float)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size,
        metric="euclidean",
    )
    train_labels = clusterer.fit_predict(X_train)
    train_dict = build_trajectory_dictionary(train_df, train_labels)

    n_clusters = len(set(train_labels)) - (1 if -1 in train_labels else 0)
    if n_clusters < 2:
        logger.warning(f"학습 클러스터 {n_clusters}개 — 시간순 분할 검증 불완전")
        return {"skipped": True, "reason": f"only_{n_clusters}_clusters"}

    # 테스트 데이터 매칭
    test_results = []
    X_test = test_df[score_cols].values.astype(float)
    for i in range(len(test_df)):
        result = match_trajectory(X_test[i], train_dict, alpha)
        test_results.append({
            "event_id": test_df.iloc[i]["id"],
            "match_type": result["best_match_type"],
            "match_phase": result["best_match_phase"],
            "similarity": result["similarity"],
        })

    test_results_df = pd.DataFrame(test_results)
    mean_sim = test_results_df["similarity"].mean()
    matched_count = (test_results_df["match_type"].notna()).sum()

    logger.info(
        f"시간순 검증 결과: 매칭 {matched_count}/{len(test_df)}, "
        f"평균 유사도 {mean_sim:.4f}"
    )

    return {
        "skipped": False,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_train_clusters": n_clusters,
        "mean_similarity": round(mean_sim, 4),
        "matched_count": int(matched_count),
        "test_results": test_results_df.to_dict(orient="records"),
    }
