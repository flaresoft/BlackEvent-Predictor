"""
Step 8-2: 궤적 클러스터링 → 유형 사전 생성

HDBSCAN으로 BlackEvent 궤적을 비지도 클러스터링하고,
각 클러스터(=유형)의 centroid, std, 멤버를 기록한다.
"""

import json

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.module8_trajectory.trajectory_builder import TRAJECTORY_OFFSETS
from src.utils import setup_logging

logger = setup_logging("module8.clustering")


def cluster_trajectories(
    traj_df: pd.DataFrame,
    min_cluster_size_range: range = range(5, 21),
) -> tuple[np.ndarray, int, dict]:
    """HDBSCAN으로 궤적을 클러스터링한다.

    min_cluster_size를 탐색하여 실루엣 스코어가 최대인 설정을 선택한다.

    Args:
        traj_df: 정규화된 궤적 DataFrame (id, D-30, ..., D-0)
        min_cluster_size_range: 탐색할 min_cluster_size 범위

    Returns:
        (labels, best_min_cluster_size, search_results)
    """
    score_cols = [f"D-{o}" for o in TRAJECTORY_OFFSETS]
    X = traj_df[score_cols].values.astype(float)

    best_score = -1.0
    best_labels = np.full(len(X), -1)
    best_mcs = min_cluster_size_range.start
    search_results = {}

    for mcs in min_cluster_size_range:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=mcs,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        if n_clusters >= 2:
            sil = silhouette_score(X, labels)
        else:
            sil = -1.0

        search_results[mcs] = {
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "silhouette": round(float(sil), 4),
        }
        logger.info(
            f"  min_cluster_size={mcs}: "
            f"{n_clusters} clusters, {n_noise} noise, silhouette={sil:.4f}"
        )

        if sil > best_score:
            best_score = sil
            best_labels = labels.copy()
            best_mcs = mcs

    logger.info(
        f"최적 min_cluster_size={best_mcs}, silhouette={best_score:.4f}"
    )
    return best_labels, best_mcs, search_results


def build_trajectory_dictionary(
    traj_df: pd.DataFrame,
    labels: np.ndarray,
) -> list[dict]:
    """클러스터링 결과로 유형 사전을 생성한다.

    Args:
        traj_df: 궤적 DataFrame (id, D-30, ..., D-0)
        labels: HDBSCAN 라벨 배열

    Returns:
        유형 사전 리스트
    """
    score_cols = [f"D-{o}" for o in TRAJECTORY_OFFSETS]
    X = traj_df[score_cols].values.astype(float)
    ids = traj_df["id"].values

    unique_labels = sorted(set(labels))
    dictionary = []

    for label in unique_labels:
        mask = labels == label
        member_ids = ids[mask].tolist()
        member_vectors = X[mask]

        centroid = member_vectors.mean(axis=0).tolist()
        std = member_vectors.std(axis=0).tolist()

        if label == -1:
            type_id = "T_noise"
        else:
            type_id = f"T{label + 1:03d}"

        entry = {
            "trajectory_type_id": type_id,
            "cluster_label": int(label),
            "cluster_size": int(mask.sum()),
            "centroid": [round(v, 6) for v in centroid],
            "std": [round(v, 6) for v in std],
            "member_events": member_ids,
            "interpretation": None,
        }
        dictionary.append(entry)
        logger.info(
            f"  {type_id}: {entry['cluster_size']}건, "
            f"centroid={[f'{v:.3f}' for v in centroid]}"
        )

    return dictionary
