"""
Module 9 시각화 — 정적 차트 (matplotlib)
"""

import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import setup_logging

logger = setup_logging("module9.visualizer")

INDICATOR_LABELS = {
    "suicide": "자살 사망자 수",
    "mental_disorder": "정신행동장애 사망자 수",
    "substance_disorder": "정신활성물질 장애 사망자 수",
    "crime": "범죄 발생건수",
    "unemployment": "실업률 (%)",
}

METRIC_LABELS = {
    "risk_mean": "리스크 평균",
    "risk_max": "리스크 최대",
    "warning_days": "WARNING 일수",
}


def _setup_korean_font():
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
            return
    plt.rcParams["axes.unicode_minus"] = False


_setup_korean_font()


def plot_cross_correlation_heatmap(results: dict, output_path: Path):
    """교차상관 히트맵 — lag x (metric, indicator) 조합"""
    xcorr_df = results.get("cross_correlation", pd.DataFrame())
    if xcorr_df.empty:
        logger.warning("교차상관 데이터 없음, 히트맵 건너뜀")
        return

    xcorr_df["pair"] = xcorr_df["risk_metric"] + " → " + xcorr_df["social_indicator"]
    pivot = xcorr_df.pivot_table(
        index="pair", columns="lag", values="correlation", aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax, linewidths=0.5,
    )
    ax.set_title("교차상관 히트맵 (리스크 스코어 → 사회 지표)")
    ax.set_xlabel("시차 (개월)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_path / "cross_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info(f"저장: {output_path / 'cross_correlation_heatmap.png'}")


def plot_time_series_dual_axis(
    merged_df: pd.DataFrame,
    risk_col: str,
    social_col: str,
    output_path: Path,
    best_lag: int = 0,
):
    """이중 축 시계열 차트."""
    df = merged_df.dropna(subset=[risk_col, social_col]).copy()
    if df.empty:
        return

    risk_label = METRIC_LABELS.get(risk_col, risk_col)
    social_label = INDICATOR_LABELS.get(social_col, social_col)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    color1 = "#e53935"
    color2 = "#1e88e5"

    x = range(len(df))
    ax1.plot(x, df[risk_col].values, color=color1, linewidth=1.5, label=risk_label)
    ax1.set_ylabel(risk_label, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(x, df[social_col].values, color=color2, linewidth=1.5, label=social_label)
    ax2.set_ylabel(social_label, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # x축 라벨 (매 6개월)
    tick_positions = list(range(0, len(df), 6))
    tick_labels = [df["year_month"].iloc[i] for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    lag_text = f" (lag={best_lag}M)" if best_lag > 0 else ""
    ax1.set_title(f"{risk_label} vs {social_label}{lag_text}")
    fig.tight_layout()

    filename = f"timeseries_{risk_col}_{social_col}.png"
    fig.savefig(output_path / filename, dpi=150)
    plt.close(fig)
    logger.info(f"저장: {output_path / filename}")


def plot_lag_profile(xcorr_df: pd.DataFrame, indicator: str, output_path: Path):
    """특정 지표의 lag별 상관계수 바 차트."""
    if xcorr_df.empty:
        return

    subset = xcorr_df[xcorr_df["social_indicator"] == indicator]
    if subset.empty:
        return

    metrics = subset["risk_metric"].unique()
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    social_label = INDICATOR_LABELS.get(indicator, indicator)

    for ax, metric in zip(axes, metrics):
        data = subset[subset["risk_metric"] == metric]
        colors = ["#e53935" if s else "#90a4ae" for s in data["significant"]]
        ax.bar(data["lag"], data["correlation"], color=colors, width=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("시차 (개월)")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}")
        ax.set_xticks(data["lag"])

    axes[0].set_ylabel("상관계수 (r)")
    fig.suptitle(f"시차별 상관: 리스크 스코어 → {social_label}", fontsize=13)
    fig.tight_layout()

    filename = f"lag_profile_{indicator}.png"
    fig.savefig(output_path / filename, dpi=150)
    plt.close(fig)
    logger.info(f"저장: {output_path / filename}")


def plot_rolling_correlation(results: dict, output_path: Path):
    """롤링 상관계수 시계열."""
    rolling_df = results.get("rolling", pd.DataFrame())
    if rolling_df.empty:
        logger.warning("롤링 상관 데이터 없음, 건너뜀")
        return

    pairs = rolling_df.groupby(["risk_metric", "social_indicator"])
    n_pairs = len(pairs)

    fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 4 * n_pairs), squeeze=False)

    for idx, ((metric, indicator), group) in enumerate(pairs):
        ax = axes[idx, 0]
        ax.plot(range(len(group)), group["rolling_corr"].values,
                color="#1e88e5", linewidth=1.5)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.fill_between(range(len(group)), group["rolling_corr"].values,
                        alpha=0.3, color="#1e88e5")

        metric_label = METRIC_LABELS.get(metric, metric)
        social_label = INDICATOR_LABELS.get(indicator, indicator)
        lag = group["lag"].iloc[0] if "lag" in group.columns else 0
        ax.set_title(f"{metric_label} → {social_label} (lag={lag}M)")
        ax.set_ylabel("롤링 상관계수")
        ax.set_ylim(-1, 1)

        # x축 라벨
        tick_positions = list(range(0, len(group), 6))
        tick_labels = [group["year_month"].iloc[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path / "rolling_correlation.png", dpi=150)
    plt.close(fig)
    logger.info(f"저장: {output_path / 'rolling_correlation.png'}")


def generate_all_plots(merged_df: pd.DataFrame, results: dict, output_path: Path):
    """모든 시각화를 생성한다."""
    output_path.mkdir(parents=True, exist_ok=True)

    # 교차상관 히트맵
    plot_cross_correlation_heatmap(results, output_path)

    # 시계열 차트
    social_cols = [c for c in merged_df.columns
                   if c not in ["year_month", "risk_mean", "risk_max",
                                "risk_std", "warning_days", "n_days"]]
    summary_df = results.get("summary", pd.DataFrame())

    for social_col in social_cols:
        best_lag = 0
        if not summary_df.empty:
            match = summary_df[
                (summary_df["social_indicator"] == social_col)
                & (summary_df["risk_metric"] == "risk_mean")
            ]
            if not match.empty:
                best_lag = int(match.iloc[0]["best_lag"])

        plot_time_series_dual_axis(
            merged_df, "risk_mean", social_col, output_path, best_lag=best_lag,
        )

    # lag 프로필
    xcorr_df = results.get("cross_correlation", pd.DataFrame())
    for indicator in social_cols:
        plot_lag_profile(xcorr_df, indicator, output_path)

    # 롤링 상관
    plot_rolling_correlation(results, output_path)

    logger.info("=== 시각화 완료 ===")
