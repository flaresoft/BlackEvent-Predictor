"""
Module 6 — Risk Score Model
현재 시점의 뉴스에서 성질을 추출하고 BlackEvent 리스크 스코어를 산출한다.
Model A: 임계값 기반 (Threshold)
Model B: 가중 스코어링 (Weighted Score)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import load_config, get_path, setup_logging

logger = setup_logging("module6")


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────
def cap_inf_odds_ratio(
    series: pd.Series, fallback_multiplier: float = 2.0
) -> pd.Series:
    """odds_ratio Series에서 inf를 유한 최대값의 배수로 대체한다.

    Args:
        series: odds_ratio 값이 담긴 Series
        fallback_multiplier: 유한 최대값에 곱할 배수 (기본 2.0)

    Returns:
        inf가 대체된 새 Series
    """
    result = series.copy()
    finite_mask = np.isfinite(result)

    if not finite_mask.any():
        logger.warning("odds_ratio에 유한값이 없어 1.0으로 대체합니다.")
        return result.replace([np.inf, -np.inf], 1.0)

    finite_max = result[finite_mask].max()
    cap_value = finite_max * fallback_multiplier
    inf_count = (~finite_mask).sum()

    if inf_count > 0:
        result[~finite_mask] = cap_value
        logger.debug(
            f"odds_ratio inf {inf_count}건 → {cap_value:.4f}로 대체 "
            f"(유한 최대 {finite_max:.4f} × {fallback_multiplier})"
        )

    return result


def _safe_load_csv(path: Path, label: str) -> pd.DataFrame:
    """CSV 파일을 안전하게 로드한다.

    파일이 없거나, 비어있거나, 헤더만 있는 경우 빈 DataFrame을 반환한다.
    """
    if not path.exists():
        logger.warning(f"{label} 파일이 존재하지 않습니다: {path}")
        return pd.DataFrame()

    if path.stat().st_size == 0:
        logger.warning(f"{label} 파일이 비어있습니다 (0 bytes): {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        logger.warning(f"{label} 파일에 데이터 행이 없습니다 (헤더만 존재): {path}")
    else:
        logger.debug(f"{label} 로드 완료: {len(df)}행")

    return df


def _compute_window_presence(
    articles: pd.DataFrame,
    ref_date: pd.Timestamp,
    relevant_properties: set[str],
    optimal_windows: dict[str, int],
    default_window: int,
) -> set[str]:
    """기준일로부터 각 성질의 최적 윈도우 내 존재 여부를 개별 판정한다.

    각 성질마다 해당 성질의 optimal_window 일수 이내의 기사에서만
    존재 여부를 확인한다. 즉, 성질 A(윈도우 7일)와 성질 B(윈도우 30일)는
    서로 다른 기사 범위에서 판정된다.

    Args:
        articles: 해당 이벤트에 연결된 기사 DataFrame (date, properties 컬럼 필요)
        ref_date: 기준일 (BlackEvent first_shock_date 또는 대조군 date)
        relevant_properties: 존재 여부를 확인할 성질 ID 집합
        optimal_windows: 성질별 최적 윈도우 (일수)
        default_window: optimal_windows에 없는 성질에 적용할 기본 윈도우

    Returns:
        윈도우 내 존재가 확인된 성질 ID 집합
    """
    if articles.empty:
        return set()

    present: set[str] = set()

    for _, article in articles.iterrows():
        article_date = article["date"]
        if not isinstance(article_date, pd.Timestamp):
            article_date = pd.Timestamp(article_date)

        days_before = (ref_date - article_date).days
        if days_before < 0:
            continue  # 기준일 이후 기사는 제외

        props = article["properties"]
        if not isinstance(props, (list, set)):
            continue

        for prop in props:
            if prop in relevant_properties and prop not in present:
                window = optimal_windows.get(prop, default_window)
                if days_before <= window:
                    present.add(prop)

    return present


# ──────────────────────────────────────────────
# Model A — 임계값 기반
# ──────────────────────────────────────────────
class ThresholdModel:
    """
    유의미한 성질의 동시 존재 수에 따라 경고 수준을 산출한다.
    Level 0 (Normal) ~ Level 3 (Alert)
    """

    def __init__(self, config: dict) -> None:
        self.thresholds: dict = config["threshold_model"]

    def predict(
        self,
        present_properties: set[str],
        significant_properties: list[str],
        significant_combinations: list[dict],
    ) -> dict:
        """경고 수준을 산출한다."""
        # 현재 존재하는 유의미한 성질 수
        sig_set = set(significant_properties)
        sig_present = [p for p in present_properties if p in sig_set]
        n_sig = len(sig_present)

        # 매칭된 유의미 조합 수
        n_combos = 0
        for combo in significant_combinations:
            antecedents = combo.get("antecedents", [])
            if all(p in present_properties for p in antecedents):
                n_combos += 1

        # 경고 수준 결정
        level_3_props = self.thresholds["level_3_properties"]
        level_3_combos = self.thresholds["level_3_combinations"]

        if n_sig >= level_3_props or n_combos >= level_3_combos:
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
            "combination_matches": n_combos,
            "matched_properties": sig_present,
        }


# ──────────────────────────────────────────────
# Model B — 가중 스코어링
# ──────────────────────────────────────────────
class WeightedScoreModel:
    """
    Risk Score = Σ(property_weight × presence) + Σ(combo_weight × combo_presence)

    정규화 모드:
    - "empirical": 학습 BlackEvent 케이스의 관측 최대 raw score 기준 (디폴트)
    - "theoretical": 모든 성질+조합 동시 매칭 이론적 최대 기준
    """

    def __init__(
        self,
        significant_df: pd.DataFrame,
        combinations_df: pd.DataFrame,
        normalization_mode: str = "empirical",
    ) -> None:
        self.normalization_mode: str = normalization_mode
        self.property_weights: dict[str, float] = {}
        self.combination_weights: list[dict] = []
        self.optimal_windows: dict[str, int] = {}
        self._empirical_max: Optional[float] = None

        # 성질 가중치: odds_ratio 기반 (정규화)
        if not significant_df.empty and "odds_ratio" in significant_df.columns:
            capped_or = cap_inf_odds_ratio(significant_df["odds_ratio"].copy())
            max_or = capped_or.max()
            if max_or > 0:
                for idx, row in significant_df.iterrows():
                    self.property_weights[row["property_id"]] = capped_or.at[idx] / max_or
            logger.debug(f"성질 가중치 설정: {len(self.property_weights)}개")

        # 조합 가중치: lift 기반 (정규화)
        if not combinations_df.empty and "lift" in combinations_df.columns:
            max_lift = combinations_df["lift"].max()
            if max_lift > 0:
                for _, row in combinations_df.iterrows():
                    antecedents = row["antecedents"]
                    if isinstance(antecedents, str):
                        antecedents = json.loads(antecedents)
                    self.combination_weights.append({
                        "antecedents": antecedents,
                        "weight": row["lift"] / max_lift,
                    })
            logger.debug(f"조합 가중치 설정: {len(self.combination_weights)}개")

        # 최적 윈도우
        if not significant_df.empty and "optimal_window" in significant_df.columns:
            for _, row in significant_df.iterrows():
                self.optimal_windows[row["property_id"]] = int(row["optimal_window"])
            logger.debug(f"최적 윈도우 설정: {len(self.optimal_windows)}개 성질")

        # 이론적 최대 raw score (모든 성질+조합 동시 매칭 시)
        self._theoretical_max: float = (
            sum(self.property_weights.values())
            + sum(c["weight"] for c in self.combination_weights)
        )

    def calibrate(self, empirical_max_score: float) -> None:
        """BlackEvent 학습 데이터의 관측 최대 raw score를 설정한다.

        run()에서 BlackEvent 스코어링 후 호출되어, 이후 대조군
        스코어링 및 재정규화에 사용된다.
        """
        if empirical_max_score > 0:
            self._empirical_max = empirical_max_score
            logger.info(f"경험적 최대 raw score 설정: {empirical_max_score:.4f}")
        else:
            logger.warning(
                "경험적 최대 raw score가 0 이하입니다. "
                "이론적 최대를 사용합니다."
            )

    def predict(self, present_properties: set[str]) -> dict:
        """리스크 스코어를 산출한다.

        Args:
            present_properties: 최적 윈도우 내 존재가 확인된 성질 ID 집합
        """
        # 단일 성질 스코어
        prop_score = sum(
            self.property_weights.get(p, 0) for p in present_properties
        )

        # 조합 스코어
        combo_score = 0.0
        combo_matches = 0
        for combo in self.combination_weights:
            if all(p in present_properties for p in combo["antecedents"]):
                combo_score += combo["weight"]
                combo_matches += 1

        total_score = prop_score + combo_score

        # 0~100 스케일 정규화
        if (
            self.normalization_mode == "empirical"
            and self._empirical_max is not None
            and self._empirical_max > 0
        ):
            normalized_score = min(total_score / self._empirical_max * 100, 100.0)
        elif self._theoretical_max > 0:
            normalized_score = total_score / self._theoretical_max * 100
        else:
            normalized_score = 0.0

        return {
            "risk_score": round(normalized_score, 2),
            "property_score": round(prop_score, 4),
            "combination_score": round(combo_score, 4),
            "total_raw_score": round(total_score, 4),
            "combination_matches": combo_matches,
        }


# ──────────────────────────────────────────────
# 스코어링 파이프라인
# ──────────────────────────────────────────────
def score_all_dates(
    corpus_df: pd.DataFrame,
    dates_df: pd.DataFrame,
    date_col: str,
    id_col: str,
    id_prefix: str,
    threshold_model: ThresholdModel,
    weighted_model: WeightedScoreModel,
    significant_properties: list[str],
    significant_combinations: list[dict],
    optimal_windows: dict[str, int],
    default_window: int = 60,
) -> pd.DataFrame:
    """모든 기준일에 대해 두 모델로 스코어링한다.

    Args:
        corpus_df: 태깅된 전체 기사 (date, event_id, properties 컬럼)
        dates_df: 기준일 목록 (BlackEvent 또는 대조군)
        date_col: dates_df에서 기준일 컬럼명
        id_col: dates_df에서 이벤트/대조군 ID 컬럼명 (e.g., "event_id", "control_id")
        id_prefix: corpus_df event_id의 prefix (e.g., "blackevent", "control")
        threshold_model: Model A
        weighted_model: Model B
        significant_properties: 유의미한 성질 ID 목록
        significant_combinations: 유의미한 성질 조합 목록 (antecedents가 이미 파싱됨)
        optimal_windows: 성질별 최적 윈도우 (일수)
        default_window: optimal_windows에 없는 성질에 적용할 기본 윈도우

    Returns:
        날짜별 스코어링 결과 DataFrame
    """
    # id_col 존재 확인
    if id_col not in dates_df.columns:
        logger.warning(
            f"dates_df에 '{id_col}' 컬럼이 없습니다. "
            f"사용 가능 컬럼: {list(dates_df.columns)}"
        )
        return pd.DataFrame()

    # 존재 여부를 확인할 전체 성질 집합 (유의미 성질 + 조합 내 성질)
    relevant_properties: set[str] = set(significant_properties)
    for combo in significant_combinations:
        ants = combo.get("antecedents", [])
        relevant_properties.update(ants)

    logger.debug(f"윈도우 기반 존재 판정 대상 성질: {len(relevant_properties)}개")

    results: list[dict] = []
    match_fail_count = 0

    for _, row in dates_df.iterrows():
        ref_date = row[date_col]
        if not isinstance(ref_date, pd.Timestamp):
            ref_date = pd.Timestamp(ref_date)

        # 이벤트 ID 구성 (config의 prefix 사용)
        raw_id = row[id_col]
        eid = f"{id_prefix}_{raw_id}"

        # corpus에서 매칭
        articles = corpus_df[corpus_df["event_id"] == eid]

        if articles.empty:
            match_fail_count += 1
            logger.debug(
                f"event_id='{eid}' 매칭 기사 없음 (ref_date={ref_date.date()})"
            )

        n_articles_total = len(articles)

        # 성질별 최적 윈도우 기반 존재 판정
        present_properties = _compute_window_presence(
            articles, ref_date, relevant_properties,
            optimal_windows, default_window,
        )

        logger.debug(
            f"ref_date={ref_date.date()}, event_id={eid}: "
            f"전체 기사 {n_articles_total}건, "
            f"윈도우 필터 후 존재 성질 {len(present_properties)}개"
        )

        # Model A
        result_a = threshold_model.predict(
            present_properties, significant_properties, significant_combinations
        )
        # Model B
        result_b = weighted_model.predict(present_properties)

        results.append({
            "date": ref_date,
            "event_id": eid,
            "n_articles": n_articles_total,
            "n_properties": len(present_properties),
            # Model A
            "model_a_level": result_a["level"],
            "model_a_label": result_a["label"],
            "model_a_sig_count": result_a["significant_count"],
            "model_a_combo_matches": result_a["combination_matches"],
            # Model B
            "model_b_risk_score": result_b["risk_score"],
            "model_b_prop_score": result_b["property_score"],
            "model_b_combo_score": result_b["combination_score"],
            "model_b_raw_score": result_b["total_raw_score"],
        })

    if match_fail_count > 0:
        logger.warning(
            f"event_id 매칭 실패: {match_fail_count}/{len(dates_df)}건 "
            f"(prefix='{id_prefix}', id_col='{id_col}')"
        )

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def run() -> pd.DataFrame:
    """Module 6 전체 파이프라인 실행"""
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")

    # event_id prefix (config 기반, 하드코딩 제거)
    be_prefix: str = config.get("event_id_prefix", {}).get("blackevent", "blackevent")
    ctrl_prefix: str = config.get("event_id_prefix", {}).get("control", "control")

    # scoring 설정
    scoring_config: dict = config.get("scoring", {})
    normalization_mode: str = scoring_config.get("normalization_mode", "empirical")
    default_window: int = scoring_config.get("default_window", 60)

    logger.info(f"정규화 모드: {normalization_mode}, 기본 윈도우: {default_window}일")

    # ── 데이터 로드 ──
    corpus_path = processed_path / "corpus_tagged.parquet"
    if not corpus_path.exists():
        logger.error(f"corpus_tagged.parquet 파일이 없습니다: {corpus_path}")
        return pd.DataFrame()

    corpus_df = pd.read_parquet(corpus_path)
    corpus_df["properties"] = corpus_df["properties"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    corpus_df["date"] = pd.to_datetime(corpus_df["date"])
    logger.info(f"코퍼스 로드: {len(corpus_df)}건")

    events_df = pd.read_csv(
        processed_path / "black_events.csv", parse_dates=["first_shock_date"]
    )
    control_df = pd.read_csv(
        processed_path / "control_dates.csv", parse_dates=["date"]
    )

    # ── 분석 결과 로드 (방어 로직) ──
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
    else:
        logger.warning(
            "significant_properties.csv에 optimal_window 컬럼이 없습니다. "
            f"기본 윈도우({default_window}일)를 사용합니다."
        )
    logger.info(f"최적 윈도우 설정 완료: {len(optimal_windows)}개 성질")

    combinations_df = _safe_load_csv(
        outputs_path / "significant_combinations.csv", "significant_combinations"
    )

    # 조합의 antecedents JSON 파싱 (한 번만 수행)
    significant_combinations: list[dict] = []
    if not combinations_df.empty:
        for _, r in combinations_df.iterrows():
            record = r.to_dict()
            ants = record.get("antecedents", [])
            if isinstance(ants, str):
                record["antecedents"] = json.loads(ants)
            significant_combinations.append(record)

    logger.info(
        f"유의미한 성질: {len(significant_properties)}개, "
        f"조합: {len(significant_combinations)}개"
    )

    # ── 모델 초기화 ──
    threshold_model = ThresholdModel(scoring_config)
    weighted_model = WeightedScoreModel(
        significant_df, combinations_df,
        normalization_mode=normalization_mode,
    )

    # ── BlackEvent 스코어링 (1차) ──
    be_scores = score_all_dates(
        corpus_df, events_df, "first_shock_date",
        id_col="event_id", id_prefix=be_prefix,
        threshold_model=threshold_model,
        weighted_model=weighted_model,
        significant_properties=significant_properties,
        significant_combinations=significant_combinations,
        optimal_windows=optimal_windows,
        default_window=default_window,
    )
    be_scores["actual_label"] = 1  # BlackEvent

    # ── 경험적 정규화 캘리브레이션 ──
    if normalization_mode == "empirical" and not be_scores.empty:
        empirical_max = be_scores["model_b_raw_score"].max()
        weighted_model.calibrate(empirical_max)

        # BE 스코어 재정규화 (calibrate 전에 이론적 최대로 산출된 값 교체)
        if empirical_max > 0:
            be_scores["model_b_risk_score"] = (
                (be_scores["model_b_raw_score"] / empirical_max * 100)
                .clip(upper=100)
                .round(2)
            )
            logger.info(
                f"BlackEvent 스코어 경험적 재정규화 완료 "
                f"(max raw={empirical_max:.4f})"
            )

    # ── 대조군 스코어링 (calibrate 완료 후 → 경험적 최대 반영) ──
    ctrl_scores = score_all_dates(
        corpus_df, control_df, "date",
        id_col="control_id", id_prefix=ctrl_prefix,
        threshold_model=threshold_model,
        weighted_model=weighted_model,
        significant_properties=significant_properties,
        significant_combinations=significant_combinations,
        optimal_windows=optimal_windows,
        default_window=default_window,
    )
    ctrl_scores["actual_label"] = 0  # 평상시

    # ── 결합 및 저장 ──
    all_scores = pd.concat([be_scores, ctrl_scores], ignore_index=True)
    all_scores.to_csv(outputs_path / "risk_scores.csv", index=False)

    logger.info(
        f"리스크 스코어 저장: {len(all_scores)}건 → "
        f"{outputs_path / 'risk_scores.csv'}"
    )

    # ── 요약 ──
    logger.info("--- Model A 요약 ---")
    for level in range(4):
        n = (all_scores["model_a_level"] == level).sum()
        n_be = (
            (all_scores["model_a_level"] == level)
            & (all_scores["actual_label"] == 1)
        ).sum()
        logger.info(f"  Level {level}: {n}건 (BlackEvent: {n_be}건)")

    logger.info("--- Model B 요약 ---")
    if not all_scores.empty:
        be_mean = all_scores.loc[
            all_scores["actual_label"] == 1, "model_b_risk_score"
        ].mean()
        ctrl_mean = all_scores.loc[
            all_scores["actual_label"] == 0, "model_b_risk_score"
        ].mean()
        logger.info(f"  BlackEvent 평균 스코어: {be_mean:.2f}")
        logger.info(f"  대조군 평균 스코어: {ctrl_mean:.2f}")
        logger.info(f"  정규화 모드: {normalization_mode}")

    logger.info("=== Module 6 완료 ===")
    return all_scores


if __name__ == "__main__":
    run()
