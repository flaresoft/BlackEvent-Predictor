"""
Module 6 — Risk Score Model
현재 시점의 뉴스에서 성질을 추출하고 BlackEvent 리스크 스코어를 산출한다.
Model A: 임계값 기반 (Threshold) — 유의미 성질 존재 수 기반
Model B: 가중 빈도 스코어링 (Weighted Frequency Score) — 빈도 × 가중치
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import load_config, get_path, setup_logging

logger = setup_logging("module6")


# ──────────────────────────────────────────────
# 데이터 로드 (메모리 최적화: 일별 집계)
# ──────────────────────────────────────────────
def load_daily_aggregation(processed_path, date_min=None, date_max=None):
    """코퍼스를 row group 단위로 읽으며 일별 성질 빈도로 집계한다.
    Module 5와 동일한 방식으로 메모리를 절약한다.

    Returns:
        dates_arr: 정렬된 numpy datetime64 배열
        daily_counts: [{property_id: count}, ...] 리스트
        total_articles: 총 기사 수
    """
    import pyarrow.parquet as pq

    parquet_path = processed_path / "corpus_tagged.parquet"
    pf = pq.ParquetFile(parquet_path)

    date_min_ts = pd.Timestamp(date_min) if date_min is not None else None
    date_max_ts = pd.Timestamp(date_max) if date_max is not None else None

    daily = defaultdict(lambda: defaultdict(int))
    daily_article_count = defaultdict(int)
    total_articles = 0
    num_rg = pf.metadata.num_row_groups

    for rg_idx in range(num_rg):
        table = pf.read_row_group(rg_idx, columns=["date", "properties"])
        chunk_df = table.to_pandas()
        del table

        chunk_df["date"] = pd.to_datetime(chunk_df["date"])

        if date_min_ts is not None and date_max_ts is not None:
            mask = (chunk_df["date"] >= date_min_ts) & (chunk_df["date"] <= date_max_ts)
            chunk_df = chunk_df[mask]

        total_articles += len(chunk_df)

        for date_val, props in zip(chunk_df["date"].values, chunk_df["properties"].values):
            if isinstance(props, np.ndarray):
                props_list = props.tolist()
            elif isinstance(props, str):
                props_list = json.loads(props)
            elif isinstance(props, list):
                props_list = props
            else:
                continue
            daily_article_count[date_val] += 1
            for prop in props_list:
                daily[date_val][prop] += 1

        del chunk_df
        if (rg_idx + 1) % 10 == 0 or rg_idx == num_rg - 1:
            logger.info(
                f"  일별 집계: RG {rg_idx+1}/{num_rg} "
                f"(누적 {total_articles:,}건, 고유 날짜 {len(daily):,}개)"
            )

    sorted_dates = sorted(daily.keys())
    dates_arr = np.array(sorted_dates, dtype="datetime64[ns]")
    counts_list = [dict(daily[d]) for d in sorted_dates]
    articles_per_day = [daily_article_count[d] for d in sorted_dates]

    return dates_arr, counts_list, articles_per_day, total_articles


def compute_window_frequency(
    dates_arr: np.ndarray,
    daily_counts: list[dict[str, int]],
    articles_per_day: list[int],
    ref_date: pd.Timestamp,
    window_days: int,
) -> tuple[dict[str, int], int, int]:
    """단일 기준일의 윈도우 내 성질 빈도와 총 기사 수를 계산한다.

    Returns:
        freq_map: {property_id: count}
        total_articles: 윈도우 내 총 기사 수
        actual_days: 윈도우 내 실제 데이터 존재 일수
    """
    start = ref_date - pd.Timedelta(days=window_days)
    end = ref_date - pd.Timedelta(days=1)

    start_np = np.datetime64(start)
    end_np = np.datetime64(end)
    i_start = np.searchsorted(dates_arr, start_np, side="left")
    i_end = np.searchsorted(dates_arr, end_np, side="right")

    freq_map = defaultdict(int)
    total = 0

    for i in range(i_start, i_end):
        total += articles_per_day[i]
        for prop, count in daily_counts[i].items():
            freq_map[prop] += count

    actual_days = i_end - i_start
    return dict(freq_map), total, actual_days


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────
def _safe_load_csv(path: Path, label: str) -> pd.DataFrame:
    """CSV 파일을 안전하게 로드한다."""
    if not path.exists():
        logger.warning(f"{label} 파일이 존재하지 않습니다: {path}")
        return pd.DataFrame()
    if path.stat().st_size == 0:
        logger.warning(f"{label} 파일이 비어있습니다: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning(f"{label} 파일 읽기 실패: {e}")
        return pd.DataFrame()
    if df.empty:
        logger.warning(f"{label} 파일에 데이터 행이 없습니다: {path}")
    return df


# ──────────────────────────────────────────────
# Model A — 임계값 기반
# ──────────────────────────────────────────────
class ThresholdModel:
    """유의미한 성질의 동시 존재 수에 따라 경고 수준을 산출한다.
    Level 0 (Normal) ~ Level 3 (Alert)
    """

    def __init__(self, config: dict) -> None:
        self.thresholds: dict = config["threshold_model"]

    def predict(
        self,
        present_properties: set[str],
        significant_properties: list[str],
    ) -> dict:
        """경고 수준을 산출한다."""
        sig_set = set(significant_properties)
        sig_present = [p for p in present_properties if p in sig_set]
        n_sig = len(sig_present)

        level_3_props = self.thresholds["level_3_properties"]
        level_3_combos = self.thresholds["level_3_combinations"]

        if n_sig >= level_3_props:
            level, label = 3, "Alert"
        elif self.thresholds["level_2"][0] <= n_sig <= self.thresholds["level_2"][1]:
            level, label = 2, "Warning"
        elif self.thresholds["level_1"][0] <= n_sig <= self.thresholds["level_1"][1]:
            level, label = 1, "Watch"
        else:
            level, label = 0, "Normal"

        return {
            "level": level,
            "label": label,
            "significant_count": n_sig,
            "matched_properties": sig_present,
        }


# ──────────────────────────────────────────────
# Model B — 가중 빈도 스코어링
# ──────────────────────────────────────────────
class WeightedFrequencyModel:
    """
    Risk Score = Σ(weight_i × frequency_ratio_i)

    frequency_ratio: 윈도우 내 성질 빈도 / 대조군 평균 빈도
    weight: -log10(p_adjusted)로 산출 (유의성이 높을수록 가중치 큼)

    정규화: empirical (BE 관측 최대 기준) 또는 theoretical (이론적 최대 기준)
    """

    def __init__(
        self,
        significant_df: pd.DataFrame,
        normalization_mode: str = "empirical",
    ) -> None:
        self.normalization_mode = normalization_mode
        self.property_weights: dict[str, float] = {}
        self.ctrl_freq_means: dict[str, float] = {}
        self.optimal_windows: dict[str, int] = {}
        self._empirical_max: Optional[float] = None

        if not significant_df.empty:
            # 가중치: -log10(p_adjusted)로 유의성 기반 차등 부여
            if "p_adjusted" in significant_df.columns:
                raw_weights = -np.log10(
                    significant_df["p_adjusted"].clip(lower=1e-300)
                )
                max_w = raw_weights.max()
                if max_w > 0:
                    normalized = raw_weights / max_w
                    for idx, (_, row) in enumerate(significant_df.iterrows()):
                        self.property_weights[row["property_id"]] = normalized.iloc[idx]

            # 대조군 평균 빈도 (빈도비 산출용)
            if "ctrl_freq_mean" in significant_df.columns:
                for _, row in significant_df.iterrows():
                    self.ctrl_freq_means[row["property_id"]] = row["ctrl_freq_mean"]

            # 최적 윈도우
            if "optimal_window" in significant_df.columns:
                for _, row in significant_df.iterrows():
                    self.optimal_windows[row["property_id"]] = int(row["optimal_window"])

            logger.info(
                f"Model B 초기화: {len(self.property_weights)}개 성질, "
                f"가중치 범위 {min(self.property_weights.values()):.3f}"
                f"~{max(self.property_weights.values()):.3f}"
            )

        # 이론적 최대 (모든 성질이 최대 빈도비일 때)
        self._theoretical_max = sum(self.property_weights.values()) * 2.0

    def calibrate(self, empirical_max_score: float) -> None:
        """BlackEvent 학습 데이터의 관측 최대 raw score를 설정한다."""
        if empirical_max_score > 0:
            self._empirical_max = empirical_max_score
            logger.info(f"경험적 최대 raw score 설정: {empirical_max_score:.4f}")

    def predict(self, freq_map: dict[str, int], total_articles: int) -> dict:
        """빈도 기반 리스크 스코어를 산출한다.

        Args:
            freq_map: {property_id: count} — 윈도우 내 성질별 출현 빈도
            total_articles: 윈도우 내 총 기사 수

        Returns:
            스코어링 결과 dict
        """
        raw_score = 0.0
        details = {}

        for prop, weight in self.property_weights.items():
            freq = freq_map.get(prop, 0)
            ctrl_mean = self.ctrl_freq_means.get(prop, 1)

            if ctrl_mean > 0 and freq > 0:
                # 빈도비: 현재 빈도 / 대조군 평균 빈도
                freq_ratio = freq / ctrl_mean
                # 초과분만 반영 (1.0 = baseline, >1.0 = 초과)
                excess = max(freq_ratio - 1.0, 0.0)
                contribution = weight * excess
            else:
                freq_ratio = 0.0
                excess = 0.0
                contribution = 0.0

            raw_score += contribution
            details[prop] = {
                "freq": freq, "ctrl_mean": ctrl_mean,
                "ratio": round(freq_ratio, 4), "contribution": round(contribution, 4),
            }

        # 0~100 정규화
        if (
            self.normalization_mode == "empirical"
            and self._empirical_max is not None
            and self._empirical_max > 0
        ):
            normalized = min(raw_score / self._empirical_max * 100, 100.0)
        elif self._theoretical_max > 0:
            normalized = min(raw_score / self._theoretical_max * 100, 100.0)
        else:
            normalized = 0.0

        return {
            "risk_score": round(normalized, 2),
            "raw_score": round(raw_score, 4),
            "details": details,
        }


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def run() -> pd.DataFrame:
    """Module 6 전체 파이프라인 실행"""
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")

    scoring_config: dict = config.get("scoring", {})
    normalization_mode: str = scoring_config.get("normalization_mode", "empirical")
    default_window: int = scoring_config.get("default_window", 60)

    logger.info(f"정규화 모드: {normalization_mode}, 기본 윈도우: {default_window}일")

    # ── 분석 결과 로드 ──
    significant_df = _safe_load_csv(
        outputs_path / "significant_properties.csv", "significant_properties"
    )
    if significant_df.empty:
        logger.error("유의미한 성질 데이터가 없습니다. 스코어링을 중단합니다.")
        return pd.DataFrame()

    significant_properties: list[str] = significant_df["property_id"].tolist()

    # 최적 윈도우 추출
    optimal_windows: dict[str, int] = {}
    if "optimal_window" in significant_df.columns:
        for _, r in significant_df.iterrows():
            optimal_windows[r["property_id"]] = int(r["optimal_window"])
    logger.info(f"유의미한 성질: {len(significant_properties)}개, 최적 윈도우 설정 완료")

    # 가중치 정보 로그
    if "p_adjusted" in significant_df.columns:
        for _, r in significant_df.sort_values("p_adjusted").iterrows():
            w = -np.log10(max(r["p_adjusted"], 1e-300))
            logger.info(
                f"  {r['property_id']}: p_adj={r['p_adjusted']:.2e}, "
                f"weight(raw)={w:.2f}, window={optimal_windows.get(r['property_id'], default_window)}일"
            )

    combinations_df = _safe_load_csv(
        outputs_path / "significant_combinations.csv", "significant_combinations"
    )

    # ── 이벤트/대조군 로드 ──
    events_df = pd.read_csv(
        processed_path / "black_events.csv", parse_dates=["first_shock_date"]
    )
    control_df = pd.read_csv(
        processed_path / "control_dates.csv", parse_dates=["date"]
    )
    logger.info(f"BlackEvent {len(events_df)}건, 대조군 {len(control_df)}건")

    # ── 일별 집계 로드 (메모리 최적화) ──
    all_ref_dates = pd.concat([
        events_df["first_shock_date"],
        control_df["date"],
    ])
    max_opt_window = max(optimal_windows.values()) if optimal_windows else default_window
    date_min = (all_ref_dates.min() - pd.Timedelta(days=max_opt_window)).strftime("%Y-%m-%d")
    date_max = all_ref_dates.max().strftime("%Y-%m-%d")

    logger.info(f"코퍼스 일별 집계 시작 (날짜 범위: {date_min} ~ {date_max})...")
    dates_arr, daily_counts, articles_per_day, total_articles = load_daily_aggregation(
        processed_path, date_min=date_min, date_max=date_max
    )
    logger.info(
        f"코퍼스 로드: {total_articles:,}건 → 일별 집계 {len(dates_arr):,}일"
    )

    # ── 모델 초기화 ──
    threshold_model = ThresholdModel(scoring_config)
    weighted_model = WeightedFrequencyModel(
        significant_df, normalization_mode=normalization_mode,
    )

    # ── 스코어링 함수 ──
    def score_date(ref_date: pd.Timestamp) -> dict:
        """단일 기준일 스코어링"""
        if not isinstance(ref_date, pd.Timestamp):
            ref_date = pd.Timestamp(ref_date)

        # 성질별 최적 윈도우로 빈도 계산
        # 고유 윈도우별로 한 번만 계산하고 재사용
        unique_windows = set(optimal_windows.values())
        if not unique_windows:
            unique_windows = {default_window}

        window_freqs: dict[int, dict[str, int]] = {}
        window_articles: dict[int, int] = {}
        for w in unique_windows:
            freq_map, n_articles, actual_days = compute_window_frequency(
                dates_arr, daily_counts, articles_per_day, ref_date, w
            )
            if actual_days > 0 and actual_days < w:
                coverage = actual_days / w
                if coverage < 0.8:
                    logger.warning(
                        f"  윈도우 {w}일 coverage 부족: {actual_days}/{w}일 "
                        f"({coverage:.0%}) — 빈도 보정 적용"
                    )
                freq_map = {k: int(v / coverage) for k, v in freq_map.items()}
                n_articles = int(n_articles / coverage)
            window_freqs[w] = freq_map
            window_articles[w] = n_articles

        # 성질별 최적 윈도우에서 빈도 추출
        combined_freq: dict[str, int] = {}
        present_properties: set[str] = set()
        primary_window = max(unique_windows)  # 기사 수 기준 윈도우
        n_articles_total = window_articles.get(primary_window, 0)

        for prop in significant_properties:
            w = optimal_windows.get(prop, default_window)
            freq = window_freqs.get(w, {}).get(prop, 0)
            combined_freq[prop] = freq
            if freq > 0:
                present_properties.add(prop)

        # Model A
        result_a = threshold_model.predict(present_properties, significant_properties)

        # Model B (빈도 기반)
        result_b = weighted_model.predict(combined_freq, n_articles_total)

        return {
            "n_articles": n_articles_total,
            "n_properties_present": len(present_properties),
            # Model A
            "model_a_level": result_a["level"],
            "model_a_label": result_a["label"],
            "model_a_sig_count": result_a["significant_count"],
            # Model B
            "model_b_risk_score": result_b["risk_score"],
            "model_b_raw_score": result_b["raw_score"],
        }

    # ── BlackEvent 스코어링 ──
    logger.info("BlackEvent 스코어링 중...")
    be_results = []
    for i, (_, row) in enumerate(events_df.iterrows()):
        ref_date = row["first_shock_date"]
        result = score_date(ref_date)
        result["date"] = ref_date
        result["event_id"] = row.get("event_id", f"be_{i}")
        result["actual_label"] = 1
        be_results.append(result)
        if (i + 1) % 20 == 0:
            logger.info(f"  BE 스코어링: {i+1}/{len(events_df)}")

    be_scores = pd.DataFrame(be_results)
    logger.info(f"BlackEvent 스코어링 완료: {len(be_scores)}건")

    # ── 경험적 캘리브레이션 ──
    if normalization_mode == "empirical" and not be_scores.empty:
        empirical_max = be_scores["model_b_raw_score"].max()
        weighted_model.calibrate(empirical_max)

        # BE 스코어 재정규화
        if empirical_max > 0:
            be_scores["model_b_risk_score"] = (
                (be_scores["model_b_raw_score"] / empirical_max * 100)
                .clip(upper=100)
                .round(2)
            )
            logger.info(f"경험적 재정규화 완료 (max raw={empirical_max:.4f})")

    # ── 대조군 스코어링 ──
    logger.info("대조군 스코어링 중...")
    ctrl_results = []
    for i, (_, row) in enumerate(control_df.iterrows()):
        ref_date = row["date"]
        result = score_date(ref_date)
        result["date"] = ref_date
        result["event_id"] = row.get("control_id", f"ctrl_{i}")
        result["actual_label"] = 0
        ctrl_results.append(result)
        if (i + 1) % 50 == 0:
            logger.info(f"  대조군 스코어링: {i+1}/{len(control_df)}")

    ctrl_scores = pd.DataFrame(ctrl_results)
    logger.info(f"대조군 스코어링 완료: {len(ctrl_scores)}건")

    # ── 결합 및 저장 ──
    all_scores = pd.concat([be_scores, ctrl_scores], ignore_index=True)
    all_scores.to_csv(outputs_path / "risk_scores.csv", index=False)
    logger.info(f"리스크 스코어 저장: {len(all_scores)}건 → {outputs_path / 'risk_scores.csv'}")

    # ── 요약 ──
    logger.info("--- Model A 요약 (임계값) ---")
    for level in range(4):
        labels = {0: "Normal", 1: "Watch", 2: "Warning", 3: "Alert"}
        n = (all_scores["model_a_level"] == level).sum()
        n_be = (
            (all_scores["model_a_level"] == level)
            & (all_scores["actual_label"] == 1)
        ).sum()
        n_ctrl = n - n_be
        logger.info(f"  Level {level} ({labels[level]}): {n}건 (BE: {n_be}, 대조군: {n_ctrl})")

    logger.info("--- Model B 요약 (가중 빈도 스코어) ---")
    if not all_scores.empty:
        be_mean = all_scores.loc[all_scores["actual_label"] == 1, "model_b_risk_score"].mean()
        be_std = all_scores.loc[all_scores["actual_label"] == 1, "model_b_risk_score"].std()
        ctrl_mean = all_scores.loc[all_scores["actual_label"] == 0, "model_b_risk_score"].mean()
        ctrl_std = all_scores.loc[all_scores["actual_label"] == 0, "model_b_risk_score"].std()
        logger.info(f"  BlackEvent 평균: {be_mean:.2f} (±{be_std:.2f})")
        logger.info(f"  대조군 평균: {ctrl_mean:.2f} (±{ctrl_std:.2f})")
        logger.info(f"  차이: {be_mean - ctrl_mean:.2f}")
        logger.info(f"  정규화 모드: {normalization_mode}")

    logger.info("=== Module 6 완료 ===")
    return all_scores


if __name__ == "__main__":
    run()
