"""
Step 8-3: 궤적 매칭 — 슬라이딩 윈도우로 유형 + phase 판정

관측된 궤적을 각 유형의 centroid에 부분 매칭하여
"어떤 유형의 어디쯤인가"를 판정한다.
"""

import numpy as np
from scipy.spatial.distance import cosine, euclidean

from src.module8_trajectory.trajectory_builder import TRAJECTORY_OFFSETS
from src.utils import setup_logging

logger = setup_logging("module8.matcher")

# 슬라이딩 매칭 가설: "오늘이 D-X 시점이다"
# 각 가설은 centroid에서 사용할 시점 수를 의미
HYPOTHESES = [
    {"phase": "D-0", "n_points": 5},   # 전체 매칭
    {"phase": "D-7", "n_points": 4},   # 앞 4개
    {"phase": "D-14", "n_points": 3},  # 앞 3개
    {"phase": "D-21", "n_points": 2},  # 앞 2개
]


def compute_similarity(
    observed: np.ndarray,
    centroid: np.ndarray,
    alpha: float = 0.5,
) -> float:
    """코사인 유사도 + 유클리드 거리 기반 종합 유사도.

    similarity = α × cosine_sim + (1 - α) × (1 - normalized_euclidean)
    """
    if len(observed) < 2 or np.allclose(observed, 0) or np.allclose(centroid, 0):
        return 0.0

    # 코사인 유사도
    cos_dist = cosine(observed, centroid)
    cos_sim = 1.0 - cos_dist

    # 정규화 유클리드 거리 (0~1 범위로 조정)
    euc_dist = euclidean(observed, centroid)
    max_possible = np.sqrt(len(observed))  # 단위 큐브 대각선
    norm_euc = min(euc_dist / max_possible, 1.0) if max_possible > 0 else 0.0

    return alpha * cos_sim + (1 - alpha) * (1.0 - norm_euc)


def match_trajectory(
    observed_vector: np.ndarray,
    trajectory_dict: list[dict],
    alpha: float = 0.5,
) -> dict:
    """관측된 궤적을 유형 사전에 슬라이딩 매칭한다.

    Args:
        observed_vector: 5시점 관측 궤적 [T-30, T-21, T-14, T-7, T]
        trajectory_dict: 유형 사전 리스트
        alpha: 유사도 가중치 (코사인 vs 유클리드)

    Returns:
        매칭 결과 dict
    """
    all_matches = []

    for entry in trajectory_dict:
        if entry["cluster_label"] == -1:
            continue  # 노이즈 유형은 매칭 대상에서 제외

        centroid = np.array(entry["centroid"])
        type_id = entry["trajectory_type_id"]

        for hyp in HYPOTHESES:
            n = hyp["n_points"]
            if n > len(observed_vector):
                continue

            obs_slice = observed_vector[:n]
            cen_slice = centroid[:n]

            sim = compute_similarity(obs_slice, cen_slice, alpha=alpha)
            all_matches.append({
                "type": type_id,
                "phase": hyp["phase"],
                "similarity": round(sim, 4),
                "n_points": n,
            })

    # 유사도 내림차순 정렬
    all_matches.sort(key=lambda x: x["similarity"], reverse=True)

    if not all_matches:
        return {
            "best_match_type": None,
            "best_match_phase": None,
            "similarity": 0.0,
            "confidence": "unclassifiable",
            "all_matches": [],
        }

    best = all_matches[0]
    runner_up = all_matches[1] if len(all_matches) > 1 else None

    return {
        "best_match_type": best["type"],
        "best_match_phase": best["phase"],
        "similarity": best["similarity"],
        "runner_up": runner_up,
        "all_matches": all_matches[:10],
    }


def calibrate_thresholds(
    control_trajectories: np.ndarray,
    trajectory_dict: list[dict],
    alpha: float = 0.5,
    percentile: float = 95.0,
) -> dict:
    """대조군 궤적으로 매칭 신뢰도 임계값을 캘리브레이션한다.

    대조군의 best similarity 분포에서 percentile 값을 임계값으로 설정.
    이 값 이상이어야 "의미 있는 매칭"으로 판정한다.

    Args:
        control_trajectories: (N, 5) 대조군 궤적 배열
        trajectory_dict: 유형 사전
        alpha: 유사도 가중치
        percentile: 임계값 백분위수

    Returns:
        캘리브레이션 결과 dict
    """
    ctrl_best_sims = []

    for i in range(len(control_trajectories)):
        result = match_trajectory(control_trajectories[i], trajectory_dict, alpha)
        ctrl_best_sims.append(result["similarity"])

    ctrl_best_sims = np.array(ctrl_best_sims)
    threshold = float(np.percentile(ctrl_best_sims, percentile))
    margin = float(np.std(ctrl_best_sims) * 0.5)

    logger.info(
        f"대조군 캘리브레이션: mean={ctrl_best_sims.mean():.4f}, "
        f"std={ctrl_best_sims.std():.4f}, "
        f"p{percentile:.0f}={threshold:.4f}"
    )

    return {
        "threshold": round(threshold, 4),
        "margin": round(margin, 4),
        "ctrl_sim_mean": round(float(ctrl_best_sims.mean()), 4),
        "ctrl_sim_std": round(float(ctrl_best_sims.std()), 4),
        "ctrl_sim_percentile": percentile,
    }


def apply_confidence(
    match_result: dict,
    threshold: float,
    margin: float,
) -> dict:
    """매칭 결과에 신뢰도 레이블을 부여한다."""
    result = match_result.copy()
    sim = result["similarity"]

    if sim < threshold:
        result["confidence"] = "unclassifiable"
    elif result.get("runner_up") and (sim - result["runner_up"]["similarity"]) < margin:
        result["confidence"] = "uncertain"
    else:
        result["confidence"] = "confident"

    return result
