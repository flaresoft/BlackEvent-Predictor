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
# 데이터 준비
# ──────────────────────────────────────────────
def load_tagged_corpus(processed_path, date_min=None, date_max=None) -> pd.DataFrame:
    """태깅된 코퍼스를 로드하고 properties를 리스트로 보장한다.
    date_min, date_max가 주어지면 해당 범위만 필터링하여 로드 (메모리 절약).
    """
    import pyarrow.parquet as pq

    parquet_path = processed_path / "corpus_tagged.parquet"

    if date_min is not None and date_max is not None:
        filters = [("date", ">=", date_min), ("date", "<=", date_max)]
        table = pq.read_table(parquet_path, filters=filters)
        df = table.to_pandas()
    else:
        df = pd.read_parquet(parquet_path)

    df["properties"] = df["properties"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def prepare_event_data(processed_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BlackEvent와 대조군 데이터를 로드한다."""
    events_df = pd.read_csv(processed_path / "black_events.csv", parse_dates=["first_shock_date"])
    control_df = pd.read_csv(processed_path / "control_dates.csv", parse_dates=["date"])
    return events_df, control_df


# ──────────────────────────────────────────────
# 윈도우 내 기사 캐시 (성능 최적화 핵심)
# ──────────────────────────────────────────────
def precompute_window_profiles(
    corpus_df: pd.DataFrame,
    ref_dates: list[pd.Timestamp],
    window_days: int,
    all_properties: list[str],
) -> tuple[dict[int, set[str]], dict[int, dict[str, int]]]:
    """
    각 기준일에 대해 윈도우 내 기사를 1번만 조회하여
    (1) 존재 기반: 어떤 성질이 존재하는지 (set)
    (2) 빈도 기반: 각 성질이 몇 번 등장하는지 (dict)
    를 사전 계산한다.

    Returns:
        presence_map: {date_idx: set of property_ids present}
        frequency_map: {date_idx: {property_id: count}}
    """
    # corpus를 날짜 정렬 후 numpy array로 변환 (binary search용)
    corpus_sorted = corpus_df.sort_values("date").reset_index(drop=True)
    dates_array = corpus_sorted["date"].values  # numpy datetime64
    props_array = corpus_sorted["properties"].values

    presence_map = {}
    frequency_map = {}

    for idx, ref_date in enumerate(ref_dates):
        start = ref_date - pd.Timedelta(days=window_days)
        end = ref_date - pd.Timedelta(days=1)

        # binary search로 윈도우 내 기사 인덱스 범위
        start_np = np.datetime64(start)
        end_np = np.datetime64(end)
        i_start = np.searchsorted(dates_array, start_np, side="left")
        i_end = np.searchsorted(dates_array, end_np, side="right")

        # 해당 윈도우 내 모든 성질을 한 번에 수집
        present_set = set()
        freq_dict = defaultdict(int)

        for i in range(i_start, i_end):
            for prop in props_array[i]:
                present_set.add(prop)
                freq_dict[prop] += 1

        presence_map[idx] = present_set
        frequency_map[idx] = dict(freq_dict)

    return presence_map, frequency_map


def precompute_window_profiles_detailed(
    corpus_df: pd.DataFrame,
    ref_dates: list[pd.Timestamp],
    window_days: int,
) -> dict[int, list[tuple[str, np.datetime64]]]:
    """
    최대 윈도우로 1회 호출하여 각 기준일의 윈도우 내 (property, date) 쌍을 수집한다.
    서브 윈도우 계산 시 날짜 필터만 적용하면 된다.

    Returns:
        detailed_map: {date_idx: [(property_id, article_date), ...]}
    """
    corpus_sorted = corpus_df.sort_values("date").reset_index(drop=True)
    dates_array = corpus_sorted["date"].values
    props_array = corpus_sorted["properties"].values

    detailed_map = {}

    for idx, ref_date in enumerate(ref_dates):
        start = ref_date - pd.Timedelta(days=window_days)
        end = ref_date - pd.Timedelta(days=1)

        start_np = np.datetime64(start)
        end_np = np.datetime64(end)
        i_start = np.searchsorted(dates_array, start_np, side="left")
        i_end = np.searchsorted(dates_array, end_np, side="right")

        records = []
        for i in range(i_start, i_end):
            article_date = dates_array[i]
            for prop in props_array[i]:
                records.append((prop, article_date))

        detailed_map[idx] = records

    return detailed_map


def _derive_from_detailed(
    detailed_map: dict[int, list[tuple[str, np.datetime64]]],
    ref_dates: list[pd.Timestamp],
    window_days: int,
) -> tuple[dict[int, set[str]], dict[int, dict[str, int]]]:
    """detailed_map에서 서브 윈도우의 presence/frequency를 날짜 필터로 산출한다."""
    presence_map = {}
    frequency_map = {}

    for idx, ref_date in enumerate(ref_dates):
        start_np = np.datetime64(ref_date - pd.Timedelta(days=window_days))
        end_np = np.datetime64(ref_date - pd.Timedelta(days=1))

        present_set = set()
        freq_dict = defaultdict(int)

        for prop, article_date in detailed_map[idx]:
            if start_np <= article_date <= end_np:
                present_set.add(prop)
                freq_dict[prop] += 1

        presence_map[idx] = present_set
        frequency_map[idx] = dict(freq_dict)

    return presence_map, frequency_map


# ──────────────────────────────────────────────
# 분석 1: 단일 성질 검정 (존재 + 빈도)
# ──────────────────────────────────────────────
def compute_property_frequencies(
    corpus_df: pd.DataFrame,
    be_dates: list[pd.Timestamp],
    ctrl_dates: list[pd.Timestamp],
    window_days: int,
    all_properties: list[str],
    be_detailed: dict = None,
    ctrl_detailed: dict = None,
) -> pd.DataFrame:
    """
    BlackEvent 직전 vs 대조군의 성질별 출현을 계산한다.
    기준일 하나당 get_articles_in_window는 1번만 호출.
    존재 기반(presence) + 빈도 기반(frequency) 동시 산출.

    be_detailed / ctrl_detailed가 제공되면, 날짜 필터로 서브 윈도우를 산출한다.
    """
    if be_detailed is not None and ctrl_detailed is not None:
        be_presence, be_frequency = _derive_from_detailed(be_detailed, be_dates, window_days)
        ctrl_presence, ctrl_frequency = _derive_from_detailed(ctrl_detailed, ctrl_dates, window_days)
    else:
        be_presence, be_frequency = precompute_window_profiles(
            corpus_df, be_dates, window_days, all_properties
        )
        ctrl_presence, ctrl_frequency = precompute_window_profiles(
            corpus_df, ctrl_dates, window_days, all_properties
        )

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
    corpus_df: pd.DataFrame,
    events_df: pd.DataFrame,
    control_df: pd.DataFrame,
    all_properties: list[str],
    windows: list[int],
    significance: float = 0.05,
    min_effect_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """각 성질마다 가장 유의미한 윈도우를 찾는다."""
    be_dates = events_df["first_shock_date"].tolist()
    ctrl_dates = control_df["date"].tolist()

    # 최대 윈도우로 1회만 사전 계산, 서브 윈도우는 날짜 필터로 산출
    max_window = max(windows)
    logger.info(f"최대 윈도우 {max_window}일로 사전 계산 중...")
    be_detailed = precompute_window_profiles_detailed(corpus_df, be_dates, max_window)
    ctrl_detailed = precompute_window_profiles_detailed(corpus_df, ctrl_dates, max_window)

    all_results = []
    for window in windows:
        logger.info(f"윈도우 {window}일 분석 중...")
        freq_df = compute_property_frequencies(
            corpus_df, be_dates, ctrl_dates, window, all_properties,
            be_detailed=be_detailed, ctrl_detailed=ctrl_detailed,
        )
        test_df = statistical_test(freq_df, significance)
        all_results.append(test_df)

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
    # presence: p_adjusted < significance AND abs(effect_size) >= min_effect_size
    # frequency: p_adjusted < significance (effect_size 없음)
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

    # 각 성질별 최적 윈도우: presence 기준 p_adjusted가 가장 작은 윈도우
    presence_df = combined[combined["analysis_type"] == "presence"]
    optimal = (
        presence_df
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
    corpus_df: pd.DataFrame,
    dates_df: pd.DataFrame,
    date_col: str,
    optimal_windows: pd.DataFrame,
    all_properties: list[str],
) -> pd.DataFrame:
    """
    각 기준일에 대해 해당 윈도우 내 존재하는 성질의 원-핫 매트릭스를 생성한다.
    dates_df: 기준 날짜가 담긴 DataFrame
    date_col: 날짜 컬럼명
    """
    # 성질별 최적 윈도우 매핑
    prop_window = {}
    for _, row in optimal_windows.iterrows():
        prop_window[row["property_id"]] = int(row["optimal_window"])

    # 각 성질의 최대 윈도우로 한 번에 기사 수집, 이후 성질별로 서브 필터
    max_window = max(prop_window.values()) if prop_window else 30

    ref_dates = dates_df[date_col].tolist()
    presence_map, _ = precompute_window_profiles(
        corpus_df, ref_dates, max_window, all_properties
    )

    # 성질별 최적 윈도우로 재검증이 필요하지만,
    # max_window로 뽑은 결과의 superset이므로 최적 윈도우별 재계산
    # → 성질별 윈도우가 다르면 각각 계산
    rows = []
    unique_windows = set(prop_window.values())

    # 윈도우별 프로파일 캐시
    window_profiles = {max_window: presence_map}
    for w in unique_windows:
        if w != max_window:
            wp, _ = precompute_window_profiles(corpus_df, ref_dates, w, all_properties)
            window_profiles[w] = wp

    for idx in range(len(ref_dates)):
        prop_flags = {}
        for prop in all_properties:
            w = prop_window.get(prop, 30)
            profile = window_profiles.get(w, presence_map)
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

    be_rules["ctrl_support"] = be_rules.apply(_ctrl_support_for_rule, axis=1)

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

    # 데이터 로드: 이벤트를 먼저 읽어 날짜 범위를 결정한 뒤 코퍼스를 필터 로드
    events_df, control_df = prepare_event_data(processed_path)

    max_window = max(analysis_config["windows"])  # 가장 넓은 윈도우 (기본 60일)
    all_ref_dates = pd.concat([
        events_df["first_shock_date"],
        control_df["date"],
    ])
    date_min = (all_ref_dates.min() - pd.Timedelta(days=max_window)).strftime("%Y-%m-%d")
    date_max = all_ref_dates.max().strftime("%Y-%m-%d")

    corpus_df = load_tagged_corpus(processed_path, date_min=date_min, date_max=date_max)
    logger.info(
        f"데이터 로드: 코퍼스 {len(corpus_df)}건 (날짜 범위: {date_min} ~ {date_max}), "
        f"BlackEvent {len(events_df)}건, 대조군 {len(control_df)}건"
    )

    # 성질 사전 로드
    with open(outputs_path / "property_dictionary.json", "r", encoding="utf-8") as f:
        property_dict = json.load(f)
    all_properties = [p["property_id"] for p in property_dict]
    logger.info(f"분석 대상 성질: {len(all_properties)}개")

    # ── 분석 1 & 2: 단일 성질 검정 + 최적 윈도우 ──
    optimal_df, combined_df = find_optimal_windows(
        corpus_df, events_df, control_df,
        all_properties,
        windows=analysis_config["windows"],
        significance=analysis_config["significance_level"],
        min_effect_size=analysis_config.get("min_effect_size", 0.2),
    )

    # combined_df 전체 저장 (#7)
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

    logger.info(f"조합 분석 후보: 엄격 기준 {len(strict_properties)}개, 조합 후보군 {len(combo_candidate_properties)}개 (p_adjusted < {loose_threshold})")

    if len(combo_candidate_properties) >= 2:
        # 시그니처 단순화: dates_df + date_col만 전달 (#6)
        be_dates_df = events_df[["first_shock_date"]].copy()
        ctrl_dates_df = control_df[["date"]].copy()

        be_matrix = build_transaction_matrix(
            corpus_df, be_dates_df, "first_shock_date", optimal_df, combo_candidate_properties
        )
        ctrl_matrix = build_transaction_matrix(
            corpus_df, ctrl_dates_df, "date", optimal_df, combo_candidate_properties
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
