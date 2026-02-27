"""
Module 5 — Temporal Pattern Analysis
BlackEvent 직전에 유독 많이 출현하는 성질 조합 패턴을 찾는다.

분석 1: 단일 성질 존재 기반 검정 (presence)
분석 1b: 단일 성질 빈도 기반 검정 (frequency) — Mann-Whitney U
분석 2: 최적 윈도우 탐색
분석 3: 성질 조합 패턴 — 연관규칙 마이닝 (Apriori)
"""

import json
import math
import platform
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.utils import load_config, get_path, setup_logging

logger = setup_logging("module5")


# ──────────────────────────────────────────────
# 한글 폰트 설정
# ──────────────────────────────────────────────
def _setup_korean_font():
    """한글 폰트 설정. Windows → Malgun Gothic, 그 외 → NanumGothic, fallback → sans-serif"""
    system = platform.system()
    candidates = []
    if system == "Windows":
        candidates = ["Malgun Gothic", "NanumGothic"]
    elif system == "Darwin":
        candidates = ["AppleGothic", "NanumGothic"]
    else:
        candidates = ["NanumGothic", "UnDotum"]

    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}

    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            logger.info(f"한글 폰트 설정: {font_name}")
            return

    logger.warning("한글 폰트를 찾지 못했습니다. 차트에서 한글이 깨질 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False


_setup_korean_font()


# ──────────────────────────────────────────────
# 데이터 준비 (메모리 최적화: 일별 집계)
# ──────────────────────────────────────────────
def load_daily_aggregation(processed_path, date_min=None, date_max=None):
    """코퍼스를 row group 단위로 읽으며 일별 성질 빈도로 집계한다.
    46.5M행 → ~7,300일 수준으로 축소하여 메모리를 극적으로 절약한다.

    Returns:
        dates_arr: 정렬된 numpy datetime64 배열 (고유 날짜)
        daily_counts: dates_arr에 대응하는 [{property_id: count}, ...] 리스트
        total_articles: 처리된 총 기사 수
    """
    import pyarrow.parquet as pq
    import datetime as dt

    parquet_path = processed_path / "corpus_tagged.parquet"
    pf = pq.ParquetFile(parquet_path)

    # 날짜 필터 준비
    date_min_ts = None
    date_max_ts = None
    if date_min is not None:
        date_min_ts = pd.Timestamp(date_min)
    if date_max is not None:
        date_max_ts = pd.Timestamp(date_max)

    daily = defaultdict(lambda: defaultdict(int))
    total_articles = 0
    num_rg = pf.metadata.num_row_groups

    for rg_idx in range(num_rg):
        table = pf.read_row_group(rg_idx, columns=["date", "properties"])
        chunk_df = table.to_pandas()
        del table

        chunk_df["date"] = pd.to_datetime(chunk_df["date"])

        # 날짜 필터
        if date_min_ts is not None and date_max_ts is not None:
            mask = (chunk_df["date"] >= date_min_ts) & (chunk_df["date"] <= date_max_ts)
            chunk_df = chunk_df[mask]

        rg_count = len(chunk_df)
        total_articles += rg_count

        # 일별 성질 빈도 집계
        for date_val, props in zip(chunk_df["date"].values, chunk_df["properties"].values):
            if isinstance(props, np.ndarray):
                props_list = props.tolist()
            elif isinstance(props, str):
                props_list = json.loads(props)
            elif isinstance(props, list):
                props_list = props
            else:
                continue
            for prop in props_list:
                daily[date_val][prop] += 1

        del chunk_df
        if (rg_idx + 1) % 5 == 0 or rg_idx == num_rg - 1:
            logger.info(
                f"  일별 집계: RG {rg_idx+1}/{num_rg} "
                f"(누적 {total_articles:,}건, 고유 날짜 {len(daily):,}개)"
            )

    # 정렬
    sorted_dates = sorted(daily.keys())
    dates_arr = np.array(sorted_dates, dtype="datetime64[ns]")
    counts_list = [dict(daily[d]) for d in sorted_dates]

    return dates_arr, counts_list, total_articles


def prepare_event_data(processed_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BlackEvent와 대조군 데이터를 로드한다."""
    events_df = pd.read_csv(processed_path / "black_events.csv", parse_dates=["first_shock_date"])
    control_df = pd.read_csv(processed_path / "control_dates.csv", parse_dates=["date"])
    return events_df, control_df


# ──────────────────────────────────────────────
# 일별 집계 기반 윈도우 프로파일 계산
# ──────────────────────────────────────────────
def compute_window_from_daily(
    dates_arr: np.ndarray,
    daily_counts: list[dict[str, int]],
    ref_dates: list[pd.Timestamp],
    window_days: int,
) -> tuple[dict[int, set[str]], dict[int, dict[str, int]]]:
    """일별 집계 데이터에서 윈도우 프로파일을 계산한다.
    ~7,300개 일별 엔트리만 탐색하므로 46.5M행 대비 극히 빠르다.

    Returns:
        presence_map: {date_idx: set of property_ids present}
        frequency_map: {date_idx: {property_id: count}}
    """
    presence_map = {}
    frequency_map = {}

    for idx, ref_date in enumerate(ref_dates):
        start = ref_date - pd.Timedelta(days=window_days)
        end = ref_date - pd.Timedelta(days=1)

        start_np = np.datetime64(start)
        end_np = np.datetime64(end)
        i_start = np.searchsorted(dates_arr, start_np, side="left")
        i_end = np.searchsorted(dates_arr, end_np, side="right")

        present_set = set()
        freq_dict = defaultdict(int)

        for i in range(i_start, i_end):
            for prop, count in daily_counts[i].items():
                present_set.add(prop)
                freq_dict[prop] += count

        presence_map[idx] = present_set
        frequency_map[idx] = dict(freq_dict)

    return presence_map, frequency_map


# ──────────────────────────────────────────────
# 분석 1: 단일 성질 검정 (존재 + 빈도)
# ──────────────────────────────────────────────
def compute_property_frequencies(
    be_presence: dict[int, set[str]],
    be_frequency: dict[int, dict[str, int]],
    ctrl_presence: dict[int, set[str]],
    ctrl_frequency: dict[int, dict[str, int]],
    be_dates: list,
    ctrl_dates: list,
    window_days: int,
    all_properties: list[str],
) -> pd.DataFrame:
    """BlackEvent 직전 vs 대조군의 성질별 출현을 계산한다.
    사전 계산된 presence/frequency 맵을 직접 사용한다.
    """
    records = []

    for prop in all_properties:
        # ── 존재 기반 (presence) ──
        be_present = sum(1 for idx in range(len(be_dates)) if prop in be_presence[idx])
        ctrl_present = sum(1 for idx in range(len(ctrl_dates)) if prop in ctrl_presence[idx])

        # ── 빈도 기반 (frequency) ──
        be_counts = [be_frequency[idx].get(prop, 0) for idx in range(len(be_dates))]
        ctrl_counts = [ctrl_frequency[idx].get(prop, 0) for idx in range(len(ctrl_dates))]

        be_total = len(be_dates)
        ctrl_total = len(ctrl_dates)

        records.append({
            "property_id": prop,
            "window_days": window_days,
            # 존재 기반
            "be_present": be_present,
            "be_total": be_total,
            "be_rate": be_present / be_total if be_total > 0 else 0,
            "ctrl_present": ctrl_present,
            "ctrl_total": ctrl_total,
            "ctrl_rate": ctrl_present / ctrl_total if ctrl_total > 0 else 0,
            # 빈도 기반
            "be_freq_mean": np.mean(be_counts) if be_counts else 0,
            "ctrl_freq_mean": np.mean(ctrl_counts) if ctrl_counts else 0,
            "be_freq_counts": be_counts,
            "ctrl_freq_counts": ctrl_counts,
        })

    return pd.DataFrame(records)


def statistical_test(freq_df: pd.DataFrame, significance: float = 0.05) -> pd.DataFrame:
    """
    존재 기반: 카이제곱 / Fisher exact test
    빈도 기반: Mann-Whitney U test
    방향성: be_rate > ctrl_rate (단측) 일관 적용
    """
    results = []

    for _, row in freq_df.iterrows():
        base = {
            "property_id": row["property_id"],
            "window_days": row["window_days"],
            "be_present": row["be_present"],
            "be_total": row["be_total"],
            "be_rate": row["be_rate"],
            "ctrl_present": row["ctrl_present"],
            "ctrl_total": row["ctrl_total"],
            "ctrl_rate": row["ctrl_rate"],
        }

        # ── 존재 기반 (presence) ──
        table = np.array([
            [row["be_present"], row["be_total"] - row["be_present"]],
            [row["ctrl_present"], row["ctrl_total"] - row["ctrl_present"]],
        ])

        # 기대빈도 최소값: chi2_contingency의 expected 배열 사용
        try:
            chi2_val, chi2_p, dof, expected = chi2_contingency(table)
            expected_min = expected.min()
        except ValueError:
            expected_min = 0
            chi2_val, chi2_p = 0, 1.0

        if table.min() < 5 or expected_min < 5:
            try:
                odds_ratio, p_value = fisher_exact(table, alternative="greater")
            except ValueError:
                odds_ratio, p_value = 1.0, 1.0
            test_type = "fisher"
        else:
            p_value = chi2_p
            odds_ratio = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0]) \
                if table[0, 1] * table[1, 0] > 0 else float("inf")
            test_type = "chi2"
            # chi2는 양측이므로, be_rate <= ctrl_rate이면 유의미하지 않음
            if row["be_rate"] <= row["ctrl_rate"]:
                p_value = 1.0

        # Cohen's h: 효과 크기 (presence 기반)
        be_rate = row["be_rate"]
        ctrl_rate = row["ctrl_rate"]
        cohens_h = 2 * (math.asin(math.sqrt(be_rate)) - math.asin(math.sqrt(ctrl_rate)))

        results.append({
            **base,
            "analysis_type": "presence",
            "test_type": test_type,
            "p_value": p_value,
            "odds_ratio": odds_ratio,
            "effect_size": cohens_h,
            "significant": p_value < significance and be_rate > ctrl_rate,
        })

        # ── 빈도 기반 (frequency) ──
        be_counts = row["be_freq_counts"]
        ctrl_counts = row["ctrl_freq_counts"]
        be_freq_mean = row["be_freq_mean"]
        ctrl_freq_mean = row["ctrl_freq_mean"]

        try:
            if sum(be_counts) + sum(ctrl_counts) > 0:
                u_stat, u_p = mannwhitneyu(
                    be_counts, ctrl_counts, alternative="greater"
                )
            else:
                u_stat, u_p = 0, 1.0
        except ValueError:
            u_stat, u_p = 0, 1.0

        results.append({
            **base,
            "analysis_type": "frequency",
            "test_type": "mann_whitney_u",
            "p_value": u_p,
            "odds_ratio": be_freq_mean / ctrl_freq_mean if ctrl_freq_mean > 0 else float("inf"),
            "effect_size": np.nan,
            "be_freq_mean": be_freq_mean,
            "ctrl_freq_mean": ctrl_freq_mean,
            "significant": u_p < significance and be_freq_mean > ctrl_freq_mean,
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# 분석 2: 최적 윈도우 탐색
# ──────────────────────────────────────────────
def find_optimal_windows(
    dates_arr: np.ndarray,
    daily_counts: list[dict[str, int]],
    events_df: pd.DataFrame,
    control_df: pd.DataFrame,
    all_properties: list[str],
    windows: list[int],
    significance: float = 0.05,
    min_effect_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """각 성질마다 가장 유의미한 윈도우를 찾는다.
    일별 집계 데이터를 사용하여 메모리 효율적으로 계산한다.
    """
    be_dates = events_df["first_shock_date"].tolist()
    ctrl_dates = control_df["date"].tolist()

    all_results = []
    for wi, window in enumerate(windows, 1):
        logger.info(f"[{wi}/{len(windows)}] 윈도우 {window}일 분석 중...")
        be_presence, be_frequency = compute_window_from_daily(
            dates_arr, daily_counts, be_dates, window
        )
        ctrl_presence, ctrl_frequency = compute_window_from_daily(
            dates_arr, daily_counts, ctrl_dates, window
        )

        freq_df = compute_property_frequencies(
            be_presence, be_frequency,
            ctrl_presence, ctrl_frequency,
            be_dates, ctrl_dates, window, all_properties,
        )
        test_df = statistical_test(freq_df, significance)
        all_results.append(test_df)
        logger.info(f"[{wi}/{len(windows)}] 윈도우 {window}일 완료")

    combined = pd.concat(all_results, ignore_index=True)

    # ── Benjamini-Hochberg FDR 보정 ──
    combined["p_adjusted"] = np.nan
    for atype in ("presence", "frequency"):
        mask = combined["analysis_type"] == atype
        pvals = combined.loc[mask, "p_value"].values
        if len(pvals) > 0:
            _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
            combined.loc[mask, "p_adjusted"] = p_adj

    # significant 판정을 보정된 p-value + 효과 크기 기준으로 갱신
    is_presence = combined["analysis_type"] == "presence"
    combined["significant"] = False
    combined.loc[is_presence, "significant"] = (
        (combined.loc[is_presence, "p_adjusted"] < significance)
        & (combined.loc[is_presence, "be_rate"] > combined.loc[is_presence, "ctrl_rate"])
        & (combined.loc[is_presence, "effect_size"].abs() >= min_effect_size)
    )
    combined.loc[~is_presence, "significant"] = (
        (combined.loc[~is_presence, "p_adjusted"] < significance)
        & (combined.loc[~is_presence, "be_freq_mean"] > combined.loc[~is_presence, "ctrl_freq_mean"])
    )

    # be_freq_counts / ctrl_freq_counts 는 저장용에서 제외 (리스트라 CSV 비호환)
    save_cols = [c for c in combined.columns if c not in ("be_freq_counts", "ctrl_freq_counts")]
    combined_save = combined[save_cols]

    # 각 성질별 최적 윈도우 선택
    # presence가 포화 상태(be_rate==ctrl_rate==1.0)이면 frequency 기준으로 전환
    presence_df = combined[combined["analysis_type"] == "presence"]
    presence_saturated = (
        (presence_df["be_rate"] == 1.0) & (presence_df["ctrl_rate"] == 1.0)
    ).all()

    if presence_saturated:
        logger.info("Presence 분석 포화 (모든 성질 100% 출현) → frequency 기준으로 최적 윈도우 선택")
        select_df = combined[combined["analysis_type"] == "frequency"]
    else:
        select_df = presence_df

    optimal = (
        select_df
        .sort_values("p_adjusted")
        .groupby("property_id")
        .first()
        .reset_index()
    )
    optimal = optimal.rename(columns={"window_days": "optimal_window"})

    return optimal, combined_save


# ──────────────────────────────────────────────
# 분석 3: 성질 조합 패턴 (연관규칙 마이닝)
# ──────────────────────────────────────────────
def build_transaction_matrix(
    dates_arr: np.ndarray,
    daily_counts: list[dict[str, int]],
    dates_df: pd.DataFrame,
    date_col: str,
    optimal_windows: pd.DataFrame,
    all_properties: list[str],
) -> pd.DataFrame:
    """
    각 기준일에 대해 해당 윈도우 내 존재하는 성질의 원-핫 매트릭스를 생성한다.
    일별 집계 데이터를 사용하여 메모리 효율적으로 계산한다.
    """
    # 성질별 최적 윈도우 매핑
    prop_window = {}
    for _, row in optimal_windows.iterrows():
        prop_window[row["property_id"]] = int(row["optimal_window"])

    ref_dates = dates_df[date_col].tolist()
    unique_windows = set(prop_window.values()) if prop_window else {30}

    # 윈도우별 프로파일 캐시
    window_profiles = {}
    for w in unique_windows:
        wp, _ = compute_window_from_daily(dates_arr, daily_counts, ref_dates, w)
        window_profiles[w] = wp

    # 기본 윈도우 (fallback)
    max_window = max(unique_windows)
    default_profile = window_profiles[max_window]

    rows = []
    for idx in range(len(ref_dates)):
        prop_flags = {}
        for prop in all_properties:
            w = prop_window.get(prop, 30)
            profile = window_profiles.get(w, default_profile)
            prop_flags[prop] = prop in profile[idx]
        rows.append(prop_flags)

    return pd.DataFrame(rows)


def apriori_analysis(
    be_matrix: pd.DataFrame,
    ctrl_matrix: pd.DataFrame,
    min_support: float = 0.1,
    min_confidence: float = 0.5,
    min_lift: float = 1.5,
) -> pd.DataFrame:
    """
    연관규칙 마이닝으로 BlackEvent 직전에만 유독 빈번한 성질 조합을 추출한다.
    대조군 지지도는 antecedents ∪ consequents 전체 itemset 기준으로 계산.
    """
    try:
        be_freq = apriori(be_matrix, min_support=min_support, use_colnames=True)
        if be_freq.empty:
            logger.warning("BlackEvent에서 빈발 항목집합이 없습니다. min_support를 낮춰보세요.")
            return pd.DataFrame()

        be_rules = association_rules(be_freq, metric="confidence", min_threshold=min_confidence)
    except Exception as e:
        logger.error(f"Apriori 분석 오류: {e}")
        return pd.DataFrame()

    # 대조군에서의 지지도: antecedents ∪ consequents 전체 itemset 기준
    ctrl_support = {}
    for _, row in be_freq.iterrows():
        itemset = row["itemsets"]
        cols = [c for c in list(itemset) if c in ctrl_matrix.columns]
        if cols:
            mask = ctrl_matrix[cols].all(axis=1)
            ctrl_support[frozenset(itemset)] = mask.mean()
        else:
            ctrl_support[frozenset(itemset)] = 0.0

    # lift 필터
    be_rules = be_rules[be_rules["lift"] >= min_lift].copy()

    # 대조군 지지도: 규칙 전체 (antecedents ∪ consequents)
    def _ctrl_support_for_rule(row):
        full_itemset = frozenset(row["antecedents"]) | frozenset(row["consequents"])
        return ctrl_support.get(full_itemset, 0.0)

    be_rules["ctrl_support"] = [
        _ctrl_support_for_rule(row) for _, row in be_rules.iterrows()
    ]

    # frozenset → list 변환 (직렬화용)
    be_rules["antecedents"] = be_rules["antecedents"].apply(lambda x: sorted(x))
    be_rules["consequents"] = be_rules["consequents"].apply(lambda x: sorted(x))

    return be_rules.sort_values("lift", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
def plot_heatmap(combined_df: pd.DataFrame, outputs_path):
    """성질 × 윈도우 히트맵 (p-value 기반, presence만)"""
    presence = combined_df[combined_df["analysis_type"] == "presence"] \
        if "analysis_type" in combined_df.columns else combined_df

    pivot = presence.pivot_table(
        index="property_id",
        columns="window_days",
        values="p_value",
        aggfunc="min",
    )

    if pivot.empty:
        logger.warning("히트맵 생성 실패: 데이터가 없습니다.")
        return

    pivot_log = -np.log10(pivot.clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.3)))
    sns.heatmap(pivot_log, annot=False, cmap="YlOrRd", ax=ax)
    ax.set_title("성질 × 윈도우 유의성 (-log10 p-value)")
    ax.set_xlabel("윈도우 (일)")
    ax.set_ylabel("성질")
    plt.tight_layout()
    fig.savefig(outputs_path / "heatmap_significance.png", dpi=150)
    plt.close(fig)
    logger.info(f"히트맵 저장: {outputs_path / 'heatmap_significance.png'}")


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def run():
    """Module 5 전체 파이프라인 실행"""
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")
    analysis_config = config["analysis"]

    # 데이터 로드: 이벤트를 먼저 읽어 날짜 범위를 결정
    events_df, control_df = prepare_event_data(processed_path)

    max_window = max(analysis_config["windows"])  # 가장 넓은 윈도우 (기본 60일)
    all_ref_dates = pd.concat([
        events_df["first_shock_date"],
        control_df["date"],
    ])
    date_min = (all_ref_dates.min() - pd.Timedelta(days=max_window)).strftime("%Y-%m-%d")
    date_max = all_ref_dates.max().strftime("%Y-%m-%d")

    # 일별 집계로 로드 (46.5M행을 메모리에 올리지 않음)
    logger.info(f"코퍼스 일별 집계 시작 (날짜 범위: {date_min} ~ {date_max})...")
    dates_arr, daily_counts, total_articles = load_daily_aggregation(
        processed_path, date_min=date_min, date_max=date_max
    )
    logger.info(
        f"데이터 로드: 코퍼스 {total_articles:,}건 → 일별 집계 {len(dates_arr):,}일, "
        f"BlackEvent {len(events_df)}건, 대조군 {len(control_df)}건"
    )

    # 성질 사전 로드
    with open(outputs_path / "property_dictionary.json", "r", encoding="utf-8") as f:
        property_dict = json.load(f)
    all_properties = [p["property_id"] for p in property_dict]
    logger.info(f"분석 대상 성질: {len(all_properties)}개")

    # ── 분석 1 & 2: 단일 성질 검정 + 최적 윈도우 ──
    optimal_df, combined_df = find_optimal_windows(
        dates_arr, daily_counts,
        events_df, control_df,
        all_properties,
        windows=analysis_config["windows"],
        significance=analysis_config["significance_level"],
        min_effect_size=analysis_config.get("min_effect_size", 0.2),
    )

    # combined_df 전체 저장
    combined_df.to_csv(outputs_path / "all_window_results.csv", index=False)
    logger.info(f"전체 윈도우 결과 저장: {outputs_path / 'all_window_results.csv'}")

    # 유의미한 성질만 필터 (presence 기준)
    significant_df = optimal_df[optimal_df["significant"]].copy()
    logger.info(f"유의미한 단일 성질: {len(significant_df)}/{len(all_properties)}개")

    significant_df.to_csv(outputs_path / "significant_properties.csv", index=False)
    logger.info(f"저장: {outputs_path / 'significant_properties.csv'}")

    # ── 분석 3: 성질 조합 패턴 ──
    # 1차: 엄격 기준 통과 (p_adjusted < significance)
    strict_properties = significant_df["property_id"].tolist()

    # 2차: 느슨한 기준 (p_adjusted < combo_candidate_threshold) — 조합 분석 후보군
    loose_threshold = analysis_config.get("combo_candidate_threshold", 0.2)
    loose_df = optimal_df[optimal_df["p_adjusted"] < loose_threshold]
    combo_candidate_properties = loose_df["property_id"].tolist()

    logger.info(
        f"조합 분석 후보: 엄격 기준 {len(strict_properties)}개, "
        f"조합 후보군 {len(combo_candidate_properties)}개 (p_adjusted < {loose_threshold})"
    )

    if len(combo_candidate_properties) >= 2:
        be_dates_df = events_df[["first_shock_date"]].copy()
        ctrl_dates_df = control_df[["date"]].copy()

        be_matrix = build_transaction_matrix(
            dates_arr, daily_counts,
            be_dates_df, "first_shock_date", optimal_df, combo_candidate_properties
        )
        ctrl_matrix = build_transaction_matrix(
            dates_arr, daily_counts,
            ctrl_dates_df, "date", optimal_df, combo_candidate_properties
        )

        apriori_config = analysis_config["apriori"]
        combo_rules = apriori_analysis(
            be_matrix, ctrl_matrix,
            min_support=apriori_config["min_support"],
            min_confidence=apriori_config["min_confidence"],
            min_lift=apriori_config["min_lift"],
        )

        if not combo_rules.empty:
            combo_rules.to_csv(outputs_path / "significant_combinations.csv", index=False)
            logger.info(f"유의미한 조합: {len(combo_rules)}개 → {outputs_path / 'significant_combinations.csv'}")
        else:
            logger.info("유의미한 성질 조합 없음")
            pd.DataFrame().to_csv(outputs_path / "significant_combinations.csv", index=False)
    else:
        logger.info("조합 후보군이 2개 미만, 조합 분석 건너뜀")
        pd.DataFrame().to_csv(outputs_path / "significant_combinations.csv", index=False)

    # ── 시각화 ──
    plot_heatmap(combined_df, outputs_path)

    logger.info("=== Module 5 완료 ===")
    return significant_df


if __name__ == "__main__":
    run()
