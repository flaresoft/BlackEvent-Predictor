"""
Backtest — BlackEvent 리스크 스코어 기반 트레이딩 전략 백테스팅

전략:
  1. Risk-Off (임계값 18.52): 스코어 >= 18.52 → 매도, < 18.52 → 매수
  2. Risk-Off (민감, 임계값 10.0): 스코어 >= 10 → 매도, < 10 → 매수
  3. Buy-the-Dip: 피크 대비 50% 하락 시 매수, 30일 보유 후 매도
  벤치마크: Buy & Hold

실행: python -m src.backtest.run
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.utils import load_config, get_path, setup_logging
from src.module6_scoring.run import (
    load_daily_aggregation,
    compute_window_frequency,
    WeightedFrequencyModel,
)
from src.module1_blackevent.run import fetch_kospi_data

logger = setup_logging("backtest")

# ──────────────────────────────────────────────
# 1. 일별 리스크 스코어 배치 계산
# ──────────────────────────────────────────────
def compute_all_daily_scores(
    dates_arr: np.ndarray,
    daily_counts: list[dict],
    articles_per_day: list[int],
    significant_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    default_window: int = 60,
    empirical_max: float | None = None,
) -> pd.DataFrame:
    """모든 거래일에 대해 리스크 스코어를 계산한다."""

    model = WeightedFrequencyModel(significant_df, normalization_mode="empirical")
    if empirical_max and empirical_max > 0:
        model.calibrate(empirical_max)

    # 속성별 최적 윈도우
    optimal_windows: dict[str, int] = {}
    if "optimal_window" in significant_df.columns:
        for _, row in significant_df.iterrows():
            optimal_windows[row["property_id"]] = int(row["optimal_window"])

    significant_properties = significant_df["property_id"].tolist()
    unique_windows = set(optimal_windows.values()) or {default_window}

    results = []
    n = len(trading_dates)

    for i, td in enumerate(trading_dates):
        ref_date = pd.Timestamp(td)

        # 윈도우별 빈도 계산
        window_freqs: dict[int, dict] = {}
        window_articles: dict[int, int] = {}
        for w in unique_windows:
            freq_map, n_articles, actual_days = compute_window_frequency(
                dates_arr, daily_counts, articles_per_day, ref_date, w
            )
            # coverage 보정
            if actual_days > 0 and actual_days < w:
                coverage = actual_days / w
                if coverage < 0.8:
                    freq_map = {k: int(v / coverage) for k, v in freq_map.items()}
                    n_articles = int(n_articles / coverage)
            window_freqs[w] = freq_map
            window_articles[w] = n_articles

        # 성질별 최적 윈도우에서 빈도 추출
        combined_freq: dict[str, int] = {}
        for prop in significant_properties:
            w = optimal_windows.get(prop, default_window)
            combined_freq[prop] = window_freqs.get(w, {}).get(prop, 0)

        primary_window = max(unique_windows)
        n_articles_total = window_articles.get(primary_window, 0)

        result_b = model.predict(combined_freq, n_articles_total)
        results.append({
            "date": ref_date,
            "risk_score": result_b["risk_score"],
            "raw_score": result_b["raw_score"],
        })

        if (i + 1) % 500 == 0 or i == n - 1:
            logger.info(f"  스코어링: {i+1}/{n} ({ref_date.date()})")

    return pd.DataFrame(results)


def load_or_compute_daily_scores(config: dict) -> pd.DataFrame:
    """캐시된 일별 스코어를 로드하거나, 없으면 계산한다."""
    outputs_path = get_path(config, "outputs")
    cache_path = outputs_path / "backtest_daily_scores.csv"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        logger.info(f"캐시된 일별 스코어 로드: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        logger.info(f"  {len(df)}일치 스코어 로드 완료")
        return df

    # 데이터 로드
    processed_path = get_path(config, "processed_data")
    significant_df = pd.read_csv(outputs_path / "significant_properties.csv")
    scoring_config = config.get("scoring", {})
    default_window = scoring_config.get("default_window", 60)

    # empirical max 로드 (기존 risk_scores.csv에서)
    empirical_max = None
    risk_scores_path = outputs_path / "risk_scores.csv"
    if risk_scores_path.exists():
        rs = pd.read_csv(risk_scores_path)
        be_rows = rs[rs["actual_label"] == 1]
        if not be_rows.empty and "model_b_raw_score" in be_rows.columns:
            empirical_max = be_rows["model_b_raw_score"].max()
            logger.info(f"Empirical max raw score: {empirical_max:.4f}")

    # KOSPI 데이터 로드 → 거래일 추출
    logger.info("KOSPI 데이터 다운로드...")
    kospi_df = fetch_kospi_data(
        ticker=config["data"]["kospi_ticker"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    trading_dates = pd.DatetimeIndex(kospi_df["Date"])

    # 코퍼스 일별 집계 (전체 기간)
    max_window = default_window
    if "optimal_window" in significant_df.columns:
        max_window = max(max_window, int(significant_df["optimal_window"].max()))

    date_min = (trading_dates.min() - pd.Timedelta(days=max_window)).strftime("%Y-%m-%d")
    date_max = trading_dates.max().strftime("%Y-%m-%d")

    logger.info(f"코퍼스 일별 집계 시작 ({date_min} ~ {date_max})...")
    dates_arr, daily_counts, articles_per_day, total = load_daily_aggregation(
        processed_path, date_min=date_min, date_max=date_max
    )
    logger.info(f"코퍼스 로드 완료: {total:,}건, {len(dates_arr):,}일")

    # 스코어 계산
    logger.info(f"전 거래일 스코어링 시작 ({len(trading_dates):,}일)...")
    scores_df = compute_all_daily_scores(
        dates_arr, daily_counts, articles_per_day,
        significant_df, trading_dates, default_window, empirical_max,
    )

    # 캐시 저장
    scores_df.to_csv(cache_path, index=False)
    logger.info(f"일별 스코어 캐시 저장: {cache_path} ({len(scores_df)}건)")

    return scores_df


# ──────────────────────────────────────────────
# 2. 백테스팅 엔진
# ──────────────────────────────────────────────
COMMISSION_RATE = 0.00015  # 편도 수수료 0.015%


def run_strategy_risk_off(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Risk-Off 전략: 스코어 >= threshold → 현금, < threshold → 매수"""
    df = df.copy()
    df["position"] = (df["risk_score"] < threshold).astype(int)
    df["leverage"] = 1.0
    return df


def run_strategy_buy_the_dip(df: pd.DataFrame, drop_pct: float = 0.5, hold_days: int = 30, leverage: float = 1.0) -> pd.DataFrame:
    """Buy-the-Dip: 스코어가 피크 대비 drop_pct 이상 하락 → 매수, hold_days 보유 후 매도"""
    df = df.copy()
    df["position"] = 0  # 기본 현금
    df["leverage"] = 1.0

    rolling_peak = 0.0
    buy_countdown = 0

    for i in range(len(df)):
        score = df.iloc[i]["risk_score"]

        if score > rolling_peak:
            rolling_peak = score

        if buy_countdown > 0:
            df.iloc[i, df.columns.get_loc("position")] = 1
            df.iloc[i, df.columns.get_loc("leverage")] = leverage
            buy_countdown -= 1
            if buy_countdown == 0:
                rolling_peak = score  # 피크 리셋
        elif rolling_peak > 5.0 and score <= rolling_peak * (1 - drop_pct):
            # 위기 피크 후 하락 → 매수 신호
            df.iloc[i, df.columns.get_loc("position")] = 1
            df.iloc[i, df.columns.get_loc("leverage")] = leverage
            buy_countdown = hold_days - 1
            rolling_peak = score  # 피크 리셋

    return df


def run_strategy_lev_contrarian(df: pd.DataFrame, threshold: float, leverage: float = 2.0, hold_days: int = 60) -> pd.DataFrame:
    """레버리지 역발상: 스코어 >= threshold 진입 후 hold_days 보유 (2x 레버리지)
    위기 감지 → 레버리지 매수 → 반등 수익 극대화"""
    df = df.copy()
    df["position"] = 0
    df["leverage"] = 1.0

    buy_countdown = 0
    triggered = False

    for i in range(len(df)):
        score = df.iloc[i]["risk_score"]

        if buy_countdown > 0:
            df.iloc[i, df.columns.get_loc("position")] = 1
            df.iloc[i, df.columns.get_loc("leverage")] = leverage
            buy_countdown -= 1
            if buy_countdown == 0:
                triggered = False
        elif score >= threshold and not triggered:
            df.iloc[i, df.columns.get_loc("position")] = 1
            df.iloc[i, df.columns.get_loc("leverage")] = leverage
            buy_countdown = hold_days - 1
            triggered = True

    return df


def run_strategy_inverse_then_leverage(
    df: pd.DataFrame,
    threshold: float = 18.52,
    inv_leverage: float = -2.0,
    lev_leverage: float = 2.0,
    hold_days: int = 60,
    switch_drop_pct: float = 0.5,
) -> pd.DataFrame:
    """인버스→레버리지 전환: 위기 감지 시 인버스로 하락 수익, 스코어 꺾이면 레버리지로 반등 수익

    Phase 1: 스코어 >= threshold → 인버스 2x 매수 (하락장 수익)
    Phase 2: 스코어가 피크 대비 switch_drop_pct 하락 → 레버리지 2x 전환 (반등 수익)
    hold_days 후 전량 매도
    """
    df = df.copy()
    df["position"] = 0
    df["leverage"] = 1.0

    in_trade = False
    phase = 0  # 1=인버스, 2=레버리지
    score_peak = 0.0
    countdown = 0

    for i in range(len(df)):
        score = df.iloc[i]["risk_score"]

        if in_trade:
            df.iloc[i, df.columns.get_loc("position")] = 1

            if phase == 1:
                df.iloc[i, df.columns.get_loc("leverage")] = inv_leverage
                if score > score_peak:
                    score_peak = score
                # 스코어가 피크 대비 충분히 하락 → 레버리지로 전환
                if score_peak > 0 and score <= score_peak * (1 - switch_drop_pct):
                    phase = 2
                    countdown = hold_days
            elif phase == 2:
                df.iloc[i, df.columns.get_loc("leverage")] = lev_leverage
                countdown -= 1
                if countdown <= 0:
                    in_trade = False
                    phase = 0
        elif score >= threshold:
            in_trade = True
            phase = 1
            score_peak = score
            df.iloc[i, df.columns.get_loc("position")] = 1
            df.iloc[i, df.columns.get_loc("leverage")] = inv_leverage

    return df


def run_strategy_hold_plus_lev_dip(df: pd.DataFrame, drop_pct: float = 0.5, hold_days: int = 30, leverage: float = 2.0) -> pd.DataFrame:
    """평소 1x KOSPI 보유 + 위기 후 반등 구간만 2x 레버리지 전환"""
    df = df.copy()
    df["position"] = 1  # 항상 보유
    df["leverage"] = 1.0  # 기본 1x

    rolling_peak = 0.0
    lev_countdown = 0

    for i in range(len(df)):
        score = df.iloc[i]["risk_score"]

        if score > rolling_peak:
            rolling_peak = score

        if lev_countdown > 0:
            df.iloc[i, df.columns.get_loc("leverage")] = leverage
            lev_countdown -= 1
            if lev_countdown == 0:
                rolling_peak = score
        elif rolling_peak > 5.0 and score <= rolling_peak * (1 - drop_pct):
            df.iloc[i, df.columns.get_loc("leverage")] = leverage
            lev_countdown = hold_days - 1
            rolling_peak = score

    return df


def run_strategy_lev_dca_contrarian(
    df: pd.DataFrame,
    threshold: float = 18.52,
    leverage: float = 2.0,
    hold_days: int = 60,
    n_tranches: int = 4,
    dip_interval_pct: float = 0.05,
) -> pd.DataFrame:
    """물타기 레버리지 역발상: 위기 감지 시 분할매수 (DCA)

    1차: 스코어 >= threshold → 1/n_tranches 매수
    2차~: 가격이 전 매수 대비 dip_interval_pct 추가 하락할 때마다 1/n_tranches 추가 매수
    전량 매수 or hold_days 경과 후 매도
    """
    df = df.copy()
    # 이 전략은 simulate_dca에서 처리하므로 시그널만 기록
    df["position"] = 0
    df["leverage"] = 1.0
    df["signal"] = ""  # "entry", "add", "exit"

    in_trade = False
    entry_price = 0.0
    last_add_price = 0.0
    tranches_used = 0
    countdown = 0

    for i in range(len(df)):
        score = df.iloc[i]["risk_score"]
        price = df.iloc[i]["close"]

        if in_trade:
            df.iloc[i, df.columns.get_loc("position")] = 1
            df.iloc[i, df.columns.get_loc("leverage")] = leverage

            # 물타기: 추가 하락 시 추가 매수
            if tranches_used < n_tranches and price <= last_add_price * (1 - dip_interval_pct):
                df.iloc[i, df.columns.get_loc("signal")] = "add"
                last_add_price = price
                tranches_used += 1

            countdown -= 1
            if countdown <= 0:
                df.iloc[i, df.columns.get_loc("signal")] = "exit"
                in_trade = False

        elif score >= threshold:
            # 첫 진입
            df.iloc[i, df.columns.get_loc("position")] = 1
            df.iloc[i, df.columns.get_loc("leverage")] = leverage
            df.iloc[i, df.columns.get_loc("signal")] = "entry"
            in_trade = True
            entry_price = price
            last_add_price = price
            tranches_used = 1
            countdown = hold_days

    return df


def simulate_dca(df: pd.DataFrame, seed: int = 1_000_000, n_tranches: int = 4) -> pd.DataFrame:
    """물타기(DCA) 전략 전용 시뮬레이터.

    자금을 n_tranches로 나누어 signal에 따라 분할 투입.
    entry/add 시 한 트랜치씩 투입, exit 시 전량 매도.
    레버리지는 일별 수익률 × leverage 방식.
    """
    df = df.copy()
    df["daily_return"] = df["close"].pct_change().fillna(0)

    cash = float(seed)
    tranche_size = seed / n_tranches  # 초기 기준
    invested = 0.0  # 현재 투자 중 금액
    asset_values = []

    for i in range(len(df)):
        signal = df.iloc[i]["signal"]
        lev = df.iloc[i]["leverage"]
        daily_ret = df.iloc[i]["daily_return"]

        if signal == "entry":
            # 첫 트랜치 투입
            tranche_size = cash / n_tranches  # 현재 현금 기준 재계산
            amount = tranche_size
            cash -= amount * (1 + COMMISSION_RATE)
            invested += amount
        elif signal == "add":
            # 추가 트랜치
            amount = min(tranche_size, cash / (1 + COMMISSION_RATE))
            if amount > 0:
                cash -= amount * (1 + COMMISSION_RATE)
                invested += amount
        elif signal == "exit":
            # 전량 매도
            cash += invested * (1 - COMMISSION_RATE)
            invested = 0.0

        # 투자 중 금액에 레버리지 일별 수익 반영
        if invested > 0 and i > 0:
            invested *= (1 + daily_ret * lev)

        total = cash + invested
        asset_values.append(total)

    df["asset"] = asset_values
    return df


def run_strategy_buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
    """Buy & Hold: 첫날 매수, 끝까지 보유"""
    df = df.copy()
    df["position"] = 1
    df["leverage"] = 1.0
    return df


def simulate(df: pd.DataFrame, seed: int = 1_000_000) -> pd.DataFrame:
    """포지션 기반 자산 시뮬레이션 (수수료 반영, 레버리지 지원)

    leverage 컬럼이 있으면 일별 수익률에 레버리지 배수를 적용한다.
    레버리지 ETF와 동일하게 일별 리밸런싱 방식으로 시뮬레이션.
    """
    df = df.copy()
    df["daily_return"] = df["close"].pct_change().fillna(0)

    has_leverage = "leverage" in df.columns
    asset = float(seed)
    in_market = False
    prev_position = 0
    prev_leverage = 1.0
    asset_values = []

    for i in range(len(df)):
        position = df.iloc[i]["position"]
        lev = df.iloc[i]["leverage"] if has_leverage else 1.0
        daily_ret = df.iloc[i]["daily_return"]

        # 포지션 변경 또는 레버리지 변경 시 수수료
        if position != prev_position or (position == 1 and lev != prev_leverage):
            if position == 1 and prev_position == 0:
                asset *= (1 - COMMISSION_RATE)  # 매수 수수료
            elif position == 0 and prev_position == 1:
                asset *= (1 - COMMISSION_RATE)  # 매도 수수료
            elif position == 1 and lev != prev_leverage:
                asset *= (1 - COMMISSION_RATE)  # 리밸런싱 수수료

        # 수익률 반영 (인버스 포함, 자산 하한 0)
        if position == 1 and i > 0:
            asset *= max(1 + daily_ret * lev, 0.0)

        asset_values.append(asset)
        prev_position = position
        prev_leverage = lev

    df["asset"] = asset_values
    return df


# ──────────────────────────────────────────────
# 3. 성과 지표 계산
# ──────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame, seed: int = 1_000_000) -> dict:
    """성과 지표를 계산한다."""
    final_asset = df["asset"].iloc[-1]
    total_return = (final_asset - seed) / seed * 100

    # CAGR
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    years = days / 365.25
    cagr = ((final_asset / seed) ** (1 / years) - 1) * 100 if years > 0 else 0

    # MDD
    peak = df["asset"].cummax()
    drawdown = (df["asset"] - peak) / peak
    mdd = drawdown.min() * 100

    # Sharpe ratio (연간화)
    daily_returns = df["asset"].pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        annual_return = daily_returns.mean() * 252
        annual_vol = daily_returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.03) / annual_vol  # 무위험이자율 3%
    else:
        sharpe = 0.0

    # 거래 횟수 및 승률
    position_changes = df["position"].diff().fillna(0)
    buy_signals = (position_changes == 1)
    sell_signals = (position_changes == -1)
    n_trades = buy_signals.sum() + sell_signals.sum()

    # 승률: 매수→매도 구간별 수익 여부
    buy_indices = df.index[buy_signals].tolist()
    sell_indices = df.index[sell_signals].tolist()
    wins = 0
    total_round_trips = 0
    for bi in buy_indices:
        matching_sells = [si for si in sell_indices if si > bi]
        if matching_sells:
            si = matching_sells[0]
            if df.loc[si, "asset"] > df.loc[bi, "asset"]:
                wins += 1
            total_round_trips += 1

    win_rate = (wins / total_round_trips * 100) if total_round_trips > 0 else 0.0

    return {
        "최종자산": f"₩{final_asset:,.0f}",
        "총수익률": f"{total_return:.1f}%",
        "CAGR": f"{cagr:.2f}%",
        "MDD": f"{mdd:.1f}%",
        "Sharpe": f"{sharpe:.2f}",
        "거래횟수": int(n_trades),
        "승률": f"{win_rate:.1f}%",
        "_final_asset": final_asset,
        "_total_return": total_return,
        "_cagr": cagr,
        "_mdd": mdd,
        "_sharpe": sharpe,
    }


# ──────────────────────────────────────────────
# 4. 리포트 & 차트 생성
# ──────────────────────────────────────────────
def generate_report(results: dict[str, pd.DataFrame], metrics: dict[str, dict], outputs_path: Path):
    """리포트 출력 및 차트 저장"""

    # 콘솔 리포트
    print("\n" + "=" * 70)
    print("  BlackEvent Predictor — 백테스팅 결과")
    print("=" * 70)
    print(f"  기간: {list(results.values())[0]['date'].iloc[0].date()} ~ "
          f"{list(results.values())[0]['date'].iloc[-1].date()}")
    print(f"  시드머니: ₩1,000,000")
    print(f"  수수료: 편도 0.015%")
    print("-" * 70)

    header = f"{'전략':<24} {'총수익률':>10} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'거래':>6} {'승률':>8}"
    print(header)
    print("-" * 78)
    for name, m in metrics.items():
        line = f"{name:<24} {m['총수익률']:>10} {m['CAGR']:>8} {m['MDD']:>8} {m['Sharpe']:>8} {m['거래횟수']:>6} {m['승률']:>8}"
        print(line)
    print("=" * 78)
    print("\n※ look-ahead bias 존재: significant_properties는 전체 기간 학습 결과")
    print("  실제 투자 성과를 보장하지 않으며, 시스템 유효성 참고 지표로만 활용\n")

    # ── 차트 생성 ──
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    colors = {
        "Buy & Hold": "#888888",
        "Risk-Off (18.52)": "#2196F3",
        "Risk-Off (10.0)": "#FF9800",
        "Buy-the-Dip": "#4CAF50",
        "Lev2x Dip Buy": "#E91E63",
        "Lev2x Contrarian": "#9C27B0",
        "Hold+Lev2x Dip": "#00BCD4",
        "Inv1x (30일)": "#607D8B",
        "Inv2x (30일)": "#F44336",
        "Inv2x (60일)": "#B71C1C",
        "Inv2x→Lev2x": "#FFD700",
        "Lev2x DCA (4회분할)": "#FF5722",
        "Lev2x DCA (보수적)": "#795548",
    }

    # 차트 1: 자산 곡선
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, df in results.items():
        ax.plot(df["date"], df["asset"], label=name, color=colors.get(name, "#333"), linewidth=1.2)
    ax.set_title("자산 곡선 비교 (시드 ₩1,000,000)", fontsize=14)
    ax.set_ylabel("자산 (₩)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputs_path / "backtest_equity_curve.png", dpi=150)
    plt.close(fig)
    logger.info(f"차트 저장: backtest_equity_curve.png")

    # 차트 2: 리스크 스코어 + 매매 시그널
    ref_df = results.get("Risk-Off (18.52)", list(results.values())[0])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 2]})

    ax1.plot(ref_df["date"], ref_df["risk_score"], color="#E53935", linewidth=0.6, alpha=0.8)
    ax1.axhline(y=18.52, color="#2196F3", linestyle="--", alpha=0.7, label="Threshold 18.52")
    ax1.axhline(y=10.0, color="#FF9800", linestyle="--", alpha=0.7, label="Threshold 10.0")
    ax1.set_ylabel("Risk Score")
    ax1.set_title("리스크 스코어 타임라인 & 매매 시그널")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ref_df["date"], ref_df["close"], color="#333", linewidth=0.8, label="KOSPI")
    # 매수/매도 시그널
    pos_change = ref_df["position"].diff().fillna(0)
    buys = ref_df[pos_change == 1]
    sells = ref_df[pos_change == -1]
    ax2.scatter(buys["date"], buys["close"], marker="^", color="#4CAF50", s=20, zorder=5, label="매수")
    ax2.scatter(sells["date"], sells["close"], marker="v", color="#E53935", s=20, zorder=5, label="매도")
    ax2.set_ylabel("KOSPI")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(outputs_path / "backtest_signals.png", dpi=150)
    plt.close(fig)
    logger.info(f"차트 저장: backtest_signals.png")

    # 차트 3: 연도별 수익률 비교
    fig, ax = plt.subplots(figsize=(14, 6))
    yearly_data = {}
    for name, df in results.items():
        df_y = df.copy()
        df_y["year"] = df_y["date"].dt.year
        yearly_returns = df_y.groupby("year")["asset"].agg(["first", "last"])
        yearly_returns["return"] = (yearly_returns["last"] / yearly_returns["first"] - 1) * 100
        yearly_data[name] = yearly_returns["return"]

    years = sorted(set().union(*[d.index for d in yearly_data.values()]))
    x = np.arange(len(years))
    n_strategies = len(yearly_data)
    width = 0.8 / n_strategies
    for i, (name, returns) in enumerate(yearly_data.items()):
        vals = [returns.get(y, 0) for y in years]
        ax.bar(x + i * width, vals, width, label=name, color=colors.get(name, "#333"), alpha=0.8)

    ax.set_xticks(x + width * n_strategies / 2)
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("수익률 (%)")
    ax.set_title("연도별 수익률 비교")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outputs_path / "backtest_yearly.png", dpi=150)
    plt.close(fig)
    logger.info(f"차트 저장: backtest_yearly.png")

    # 차트 4: 주요 위기 구간 확대
    crisis_periods = [
        ("2008 금융위기", "2008-01-01", "2009-06-30"),
        ("2020 코로나", "2020-01-01", "2020-12-31"),
        ("2022 긴축", "2022-01-01", "2022-12-31"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, start, end) in zip(axes, crisis_periods):
        for name, df in results.items():
            mask = (df["date"] >= start) & (df["date"] <= end)
            sub = df[mask]
            if len(sub) > 0:
                normalized = sub["asset"] / sub["asset"].iloc[0] * 100
                ax.plot(sub["date"], normalized, label=name, color=colors.get(name, "#333"), linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("자산 (기준=100)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    fig.suptitle("주요 위기 구간 자산 변화", fontsize=13)
    fig.tight_layout()
    fig.savefig(outputs_path / "backtest_crisis_zoom.png", dpi=150)
    plt.close(fig)
    logger.info(f"차트 저장: backtest_crisis_zoom.png")


# ──────────────────────────────────────────────
# 5. 메인
# ──────────────────────────────────────────────
def run():
    config = load_config()
    outputs_path = get_path(config, "outputs")
    outputs_path.mkdir(parents=True, exist_ok=True)
    seed = 1_000_000

    # ── 일별 리스크 스코어 ──
    scores_df = load_or_compute_daily_scores(config)

    # ── KOSPI 데이터 ──
    logger.info("KOSPI 데이터 로드...")
    kospi_df = fetch_kospi_data(
        ticker=config["data"]["kospi_ticker"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 병합
    kospi_df["date"] = pd.to_datetime(kospi_df["Date"])
    scores_df["date"] = pd.to_datetime(scores_df["date"])
    merged = pd.merge(kospi_df[["date", "Close"]], scores_df[["date", "risk_score"]], on="date", how="inner")
    merged = merged.rename(columns={"Close": "close"}).sort_values("date").reset_index(drop=True)
    logger.info(f"병합 완료: {len(merged)}일 ({merged['date'].iloc[0].date()} ~ {merged['date'].iloc[-1].date()})")

    # 스코어 없는 날은 0으로 채움
    merged["risk_score"] = merged["risk_score"].fillna(0)

    # ── 전략 실행 ──
    strategies = {
        "Buy & Hold": run_strategy_buy_and_hold,
        "Risk-Off (18.52)": lambda df: run_strategy_risk_off(df, threshold=18.52),
        "Risk-Off (10.0)": lambda df: run_strategy_risk_off(df, threshold=10.0),
        "Buy-the-Dip": run_strategy_buy_the_dip,
        "Lev2x Dip Buy": lambda df: run_strategy_buy_the_dip(df, leverage=2.0),
        "Lev2x Contrarian": lambda df: run_strategy_lev_contrarian(df, threshold=18.52, leverage=2.0, hold_days=60),
        "Hold+Lev2x Dip": lambda df: run_strategy_hold_plus_lev_dip(df, leverage=2.0),
        "Inv1x (30일)": lambda df: run_strategy_lev_contrarian(df, threshold=18.52, leverage=-1.0, hold_days=30),
        "Inv2x (30일)": lambda df: run_strategy_lev_contrarian(df, threshold=18.52, leverage=-2.0, hold_days=30),
        "Inv2x (60일)": lambda df: run_strategy_lev_contrarian(df, threshold=18.52, leverage=-2.0, hold_days=60),
        "Inv2x→Lev2x": lambda df: run_strategy_inverse_then_leverage(df, threshold=18.52),
    }

    # DCA (물타기) 전략 — 별도 시뮬레이터 사용
    dca_strategies = {
        "Lev2x DCA (4회분할)": lambda df: run_strategy_lev_dca_contrarian(df, threshold=18.52, leverage=2.0, hold_days=60, n_tranches=4, dip_interval_pct=0.05),
        "Lev2x DCA (보수적)": lambda df: run_strategy_lev_dca_contrarian(df, threshold=10.0, leverage=2.0, hold_days=90, n_tranches=5, dip_interval_pct=0.03),
    }

    results = {}
    all_metrics = {}

    for name, strategy_fn in strategies.items():
        logger.info(f"전략 실행: {name}")
        df_with_pos = strategy_fn(merged.copy())
        df_sim = simulate(df_with_pos, seed=seed)
        results[name] = df_sim
        all_metrics[name] = compute_metrics(df_sim, seed=seed)

    for name, strategy_fn in dca_strategies.items():
        logger.info(f"전략 실행: {name}")
        df_with_pos = strategy_fn(merged.copy())
        n_tr = 4 if "4회" in name else 5
        df_sim = simulate_dca(df_with_pos, seed=seed, n_tranches=n_tr)
        results[name] = df_sim
        all_metrics[name] = compute_metrics(df_sim, seed=seed)

    # ── 리포트 ──
    generate_report(results, all_metrics, outputs_path)

    # 결과 CSV 저장
    summary_rows = []
    for name, m in all_metrics.items():
        summary_rows.append({
            "strategy": name,
            "final_asset": m["_final_asset"],
            "total_return_pct": m["_total_return"],
            "cagr_pct": m["_cagr"],
            "mdd_pct": m["_mdd"],
            "sharpe": m["_sharpe"],
            "n_trades": m["거래횟수"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outputs_path / "backtest_summary.csv", index=False)
    logger.info(f"요약 저장: {outputs_path / 'backtest_summary.csv'}")

    logger.info("=== 백테스팅 완료 ===")


if __name__ == "__main__":
    run()
