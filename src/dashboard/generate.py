"""
GitHub 정적 대시보드 이미지 생성

Usage:
    python -m src.dashboard.generate
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("data/outputs/dashboard")
SCORES_CSV = Path("data/outputs/daily_scores.csv")
THRESHOLD = 18.52

PROP_NAMES = {
    "Property_001": "거시경제 급변",
    "Property_013": "금융기관 신뢰 붕괴",
    "Property_023": "자산 가격 급변",
    "Property_024": "외부 요인 시장 반응",
    "Property_030": "정부 개입",
    "Property_049": "글로벌 상호 연결성",
    "Property_055": "수급 불균형",
    "Property_068": "투자 심리 급변",
    "Property_070": "금융 규제/정책 변화",
    "Property_078": "구조적 불안정 확산",
    "Property_110": "과도한 레버리지",
    "Property_113": "비선형 위기 확산",
}
PROP_ORDER = [
    "Property_078", "Property_055", "Property_070", "Property_024",
    "Property_013", "Property_049", "Property_001", "Property_023",
    "Property_113", "Property_068", "Property_030", "Property_110",
]

# ── 스타일 ──
BG = "#0d1117"
BG2 = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#7d8590"
GRID = "#21262d"
GREEN = "#3fb950"
YELLOW = "#d29922"
RED = "#f85149"
ORANGE = "#db6d28"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG2,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "grid.color": GRID,
        "grid.alpha": 0.5,
        "font.size": 11,
        "font.family": "Malgun Gothic",
    })


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def parse_details(df: pd.DataFrame) -> pd.DataFrame:
    """details_json을 파싱하여 property별 ratio/contribution 컬럼 추가."""
    rows = []
    for _, row in df.iterrows():
        details = json.loads(row["details_json"])
        entry = {"date": row["date"], "risk_score": row["risk_score"]}
        for pid in PROP_ORDER:
            if pid in details:
                entry[f"{pid}_ratio"] = details[pid]["ratio"]
                entry[f"{pid}_contrib"] = details[pid]["contribution"]
            else:
                entry[f"{pid}_ratio"] = 0.0
                entry[f"{pid}_contrib"] = 0.0
        rows.append(entry)
    return pd.DataFrame(rows)


# ── Chart 1: 리스크 스코어 타임라인 ──
def chart_timeline(df: pd.DataFrame):
    # WARNING 기간 포커스: WARNING 첫 날짜 2주 전부터
    warn_rows = df[df["status"] == "WARNING"]
    if len(warn_rows) > 0:
        focus_start = warn_rows.iloc[0]["date"] - pd.Timedelta(days=14)
    else:
        focus_start = df["date"].iloc[-1] - pd.Timedelta(days=60)

    focus_df = df[df["date"] >= focus_start].copy().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    dates = focus_df["date"]
    scores = focus_df["risk_score"]

    # 배경 영역
    ax.axhspan(0, THRESHOLD, color=GREEN, alpha=0.05)
    ax.axhspan(THRESHOLD, 50, color=YELLOW, alpha=0.05)
    ax.axhspan(50, 100, color=RED, alpha=0.05)

    # Threshold 선
    ax.axhline(y=THRESHOLD, color=YELLOW, linewidth=1, linestyle="--", alpha=0.7)
    ax.text(dates.iloc[0] + pd.Timedelta(days=1), THRESHOLD + 1.5,
            f"WARNING ({THRESHOLD})", color=YELLOW, fontsize=9, alpha=0.8)

    # 스코어 라인
    colors = [RED if s >= 50 else YELLOW if s >= THRESHOLD else GREEN for s in scores]
    for i in range(len(dates) - 1):
        ax.plot(dates.iloc[i:i+2], scores.iloc[i:i+2],
                color=colors[i], linewidth=2.5, solid_capstyle="round")

    # 포인트
    ax.scatter(dates, scores, c=colors, s=50, zorder=5, edgecolors="none")

    # 각 포인트에 스코어 표시
    for i, (d, s) in enumerate(zip(dates, scores)):
        offset_y = 8 if i % 2 == 0 else -14
        c = RED if s >= 50 else YELLOW if s >= THRESHOLD else GREEN
        ax.annotate(f"{s:.0f}", xy=(d, s), xytext=(0, offset_y),
                    textcoords="offset points", fontsize=8, color=c,
                    ha="center", fontweight="bold")

    # 최신 값 강조
    latest = focus_df.iloc[-1]
    color = RED if latest["risk_score"] >= 50 else YELLOW if latest["risk_score"] >= THRESHOLD else GREEN
    ax.annotate(
        f'{latest["risk_score"]:.1f}',
        xy=(latest["date"], latest["risk_score"]),
        xytext=(20, 15), textcoords="offset points",
        fontsize=18, fontweight="bold", color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=1.5),
    )

    # 고점 표시
    peak_idx = focus_df["risk_score"].idxmax()
    peak = focus_df.loc[peak_idx]
    if peak_idx != focus_df.index[-1]:
        ax.annotate(
            f'Peak {peak["risk_score"]:.1f}',
            xy=(peak["date"], peak["risk_score"]),
            xytext=(-30, 18), textcoords="offset points",
            fontsize=11, color=RED, ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
        )

    ax.set_xlim(dates.iloc[0] - pd.Timedelta(days=2), dates.iloc[-1] + pd.Timedelta(days=5))
    ax.set_ylim(0, max(scores.max() * 1.2, 25))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    ax.set_ylabel("Risk Score")
    ax.set_title("BlackEvent Risk Score Timeline", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="y", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  timeline.png")


# ── Chart 2: Property 히트맵 ──
def chart_heatmap(detail_df: pd.DataFrame):
    # WARNING 이상인 날짜만
    warn_df = detail_df[detail_df["risk_score"] >= THRESHOLD * 0.5].copy()
    if len(warn_df) == 0:
        warn_df = detail_df.tail(5)

    n_dates = len(warn_df)
    n_props = len(PROP_ORDER)

    ratio_matrix = np.zeros((n_props, n_dates))
    for j, (_, row) in enumerate(warn_df.iterrows()):
        for i, pid in enumerate(PROP_ORDER):
            ratio_matrix[i, j] = row.get(f"{pid}_ratio", 0.0)

    fig, ax = plt.subplots(figsize=(max(10, n_dates * 0.8 + 3), 6))

    # 커스텀 컬러맵: 1.0 미만=회색, 1.0~1.2=노랑, 1.2~1.5=주황, 1.5+=빨강
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("risk", [
        (0.0, "#1a1a2e"),    # < 0.5: 어둡게
        (0.4, "#2d3748"),    # ~0.8
        (0.5, "#4a5568"),    # ~1.0: 중립
        (0.6, "#d69e2e"),    # ~1.2: 노랑
        (0.75, "#dd6b20"),   # ~1.5: 주황
        (1.0, "#e53e3e"),    # 2.0+: 빨강
    ])

    im = ax.imshow(ratio_matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=2.0,
                   interpolation="nearest")

    # 값 표시
    for i in range(n_props):
        for j in range(n_dates):
            val = ratio_matrix[i, j]
            if val >= 1.0:
                txt_color = "white" if val >= 1.5 else "#fefce8"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold" if val >= 1.5 else "normal")
            elif val > 0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="#718096")

    # 축 레이블
    date_labels = [d.strftime("%m/%d") for d in warn_df["date"]]
    ax.set_xticks(range(n_dates))
    ax.set_xticklabels(date_labels, fontsize=9)
    ax.set_yticks(range(n_props))
    ax.set_yticklabels([PROP_NAMES[p] for p in PROP_ORDER], fontsize=9)

    # 스코어 행 (하단)
    for j, (_, row) in enumerate(warn_df.iterrows()):
        score = row["risk_score"]
        color = RED if score >= 50 else YELLOW if score >= THRESHOLD else GREEN
        ax.text(j, n_props + 0.3, f"{score:.0f}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")
    ax.text(-0.5, n_props + 0.3, "Score", ha="right", va="center",
            fontsize=9, color=TEXT_DIM)

    ax.set_xlim(-0.5, n_dates - 0.5)
    ax.set_ylim(n_props + 0.8, -0.5)
    ax.set_title("Property Activation Heatmap (ratio to baseline)", fontsize=13,
                 fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Frequency Ratio", color=TEXT_DIM, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_DIM)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  heatmap.png")


# ── Chart 3: 현재 상태 카드 ──
def chart_status(df: pd.DataFrame, detail_df: pd.DataFrame):
    latest = df.iloc[-1]
    score = latest["risk_score"]
    date_str = latest["date"].strftime("%Y-%m-%d")
    status = latest["status"]

    # WARNING 연속 일수
    consecutive = 0
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]["status"] == "WARNING":
            consecutive += 1
        else:
            break

    # 고점 정보
    peak_idx = df["risk_score"].idxmax()
    peak = df.loc[peak_idx]

    # 최신 property details
    details = json.loads(latest["details_json"])
    sorted_props = sorted(details.keys(), key=lambda x: details[x]["contribution"], reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.2, 1.3]})

    # ── 패널 1: 스코어 게이지 ──
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    color = RED if score >= 50 else YELLOW if score >= THRESHOLD else GREEN
    status_text = "WARNING" if status == "WARNING" else "NORMAL"

    ax.text(5, 8.5, date_str, fontsize=12, ha="center", color=TEXT_DIM)
    ax.text(5, 5.5, f"{score:.1f}", fontsize=48, ha="center", va="center",
            fontweight="bold", color=color)
    ax.text(5, 2.8, status_text, fontsize=18, ha="center", fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15,
                      edgecolor=color, linewidth=1.5))

    if consecutive > 0:
        ax.text(5, 1.0, f"WARNING {consecutive}일 연속",
                fontsize=10, ha="center", color=ORANGE)

    # ── 패널 2: 핵심 지표 ──
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    metrics = [
        ("Threshold", f"{THRESHOLD}", TEXT_DIM),
        ("Peak", f'{peak["risk_score"]:.1f}  ({peak["date"].strftime("%m/%d")})', RED),
        ("Articles", f'{latest["n_articles"]:,}', TEXT_DIM),
        ("Active (>1.0)", f'{sum(1 for p in details.values() if p["ratio"] > 1.0)}/12', YELLOW),
        ("Extreme (>1.5)", f'{sum(1 for p in details.values() if p["ratio"] > 1.5)}/12',
         RED if sum(1 for p in details.values() if p["ratio"] > 1.5) >= 5 else ORANGE),
    ]

    ax.text(5, 9.3, "Key Metrics", fontsize=12, ha="center", fontweight="bold", color=TEXT)
    for i, (label, value, c) in enumerate(metrics):
        y = 7.5 - i * 1.5
        ax.text(1, y, label, fontsize=10, color=TEXT_DIM, va="center")
        ax.text(9, y, value, fontsize=11, color=c, va="center", ha="right", fontweight="bold")

    # ── 패널 3: Top 기여 성질 ──
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.3, "Top Contributors", fontsize=12, ha="center", fontweight="bold", color=TEXT)

    for i, pid in enumerate(sorted_props[:6]):
        d = details[pid]
        name = PROP_NAMES.get(pid, pid)
        ratio = d["ratio"]
        contrib = d["contribution"]
        y = 7.8 - i * 1.3

        if ratio >= 1.5:
            c = RED
            marker = "!!"
        elif ratio >= 1.0:
            c = YELLOW
            marker = "+"
        else:
            c = TEXT_DIM
            marker = ""

        ax.text(0.3, y, name, fontsize=9, color=c, va="center")
        bar_len = min(ratio / 2.0 * 3, 3.0)
        ax.barh(y, bar_len, height=0.4, left=6.5, color=c, alpha=0.4)
        ax.text(9.8, y, f"{ratio:.2f}x {marker}", fontsize=9, color=c,
                va="center", ha="right", fontweight="bold")

    fig.suptitle("BlackEvent Predictor — Live Status", fontsize=15,
                 fontweight="bold", color=TEXT, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "status.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  status.png")


# ── Chart 4: 스코어 고점 경고 패턴 ──
def chart_peak_pattern(df: pd.DataFrame):
    """고점 = 마지막 탈출 신호 패턴을 시각화. WARNING 기간 포커스."""
    warn_rows = df[df["status"] == "WARNING"]
    if len(warn_rows) == 0:
        return

    # WARNING 시작 1주 전부터
    focus_start = warn_rows.iloc[0]["date"] - pd.Timedelta(days=7)
    focus_df = df[df["date"] >= focus_start].copy().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))

    dates = focus_df["date"]
    scores = focus_df["risk_score"]

    # WARNING 배경
    warn_mask = focus_df["status"] == "WARNING"
    ax.fill_between(dates, 0, scores, where=warn_mask, color=RED, alpha=0.08)

    # 고점 찾기
    peak_idx = focus_df["risk_score"].idxmax()
    peak = focus_df.loc[peak_idx]

    # 고점 전 상승 구간
    pre_peak = focus_df.loc[:peak_idx]
    ax.plot(pre_peak["date"], pre_peak["risk_score"],
            color=RED, linewidth=3, solid_capstyle="round")
    ax.scatter(pre_peak["date"], pre_peak["risk_score"],
               c=RED, s=40, zorder=5, edgecolors="none")

    # 고점 이후 하락 구간
    post_peak = focus_df.loc[peak_idx:]
    if len(post_peak) > 1:
        ax.fill_between(post_peak["date"], 0, post_peak["risk_score"],
                        color=ORANGE, alpha=0.12)
        ax.plot(post_peak["date"], post_peak["risk_score"],
                color=ORANGE, linewidth=3, linestyle="--", solid_capstyle="round")
        ax.scatter(post_peak["date"], post_peak["risk_score"],
                   c=ORANGE, s=40, zorder=5, edgecolors="none")

    # 고점 전 NORMAL 포인트
    normal_pre = focus_df[(focus_df.index < focus_df[warn_mask].index.min()) if warn_mask.any() else []]
    if len(normal_pre) > 0:
        ax.plot(normal_pre["date"], normal_pre["risk_score"],
                color=GREEN, linewidth=2, solid_capstyle="round")
        ax.scatter(normal_pre["date"], normal_pre["risk_score"],
                   c=GREEN, s=40, zorder=5, edgecolors="none")

    # 고점 마커
    ax.scatter([peak["date"]], [peak["risk_score"]], s=250, color=RED,
               zorder=10, marker="v", edgecolors="white", linewidth=2)
    ax.annotate(
        f'PEAK {peak["risk_score"]:.1f}\nLast Exit Window',
        xy=(peak["date"], peak["risk_score"]),
        xytext=(-50, 20), textcoords="offset points",
        fontsize=12, color=RED, ha="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
    )

    # Threshold
    ax.axhline(y=THRESHOLD, color=YELLOW, linewidth=1, linestyle="--", alpha=0.5)
    ax.text(dates.iloc[0] + pd.Timedelta(days=1), THRESHOLD + 1,
            f"Threshold ({THRESHOLD})", color=YELLOW, fontsize=8, alpha=0.7)

    # 하락 구간 주석
    latest = focus_df.iloc[-1]
    if peak["risk_score"] > latest["risk_score"] and len(post_peak) > 2:
        mid_idx = peak_idx + len(post_peak) // 2
        if mid_idx < len(focus_df):
            mid_row = focus_df.iloc[mid_idx]
            drop = peak["risk_score"] - latest["risk_score"]
            ax.annotate(
                f'Score dropping ({drop:.0f})\n= News fatigue, NOT safety',
                xy=(mid_row["date"], mid_row["risk_score"]),
                xytext=(30, -25), textcoords="offset points",
                fontsize=10, color=ORANGE, ha="left", style="italic",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=ORANGE, alpha=0.12,
                          edgecolor=ORANGE, linewidth=0.5),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1),
            )

    ax.set_xlim(dates.iloc[0] - pd.Timedelta(days=2), dates.iloc[-1] + pd.Timedelta(days=5))
    ax.set_ylim(0, peak["risk_score"] * 1.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.set_ylabel("Risk Score")
    ax.set_title('Peak = Last Exit Signal  ("Score drop ≠ Safety")',
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, axis="y", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "peak_pattern.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  peak_pattern.png")


def generate_dashboard_md(df: pd.DataFrame):
    """DASHBOARD.md 생성."""
    latest = df.iloc[-1]
    score = latest["risk_score"]
    date_str = latest["date"].strftime("%Y-%m-%d")
    status = latest["status"]

    # WARNING 연속 일수
    consecutive = 0
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]["status"] == "WARNING":
            consecutive += 1
        else:
            break

    # 고점
    peak_idx = df["risk_score"].idxmax()
    peak = df.loc[peak_idx]

    # Property details
    details = json.loads(latest["details_json"])
    active = sum(1 for p in details.values() if p["ratio"] > 1.0)
    extreme = sum(1 for p in details.values() if p["ratio"] > 1.5)

    status_emoji = "🔴" if score >= 50 else "🟡" if score >= THRESHOLD else "🟢"

    md = f"""# BlackEvent Predictor — Dashboard

> Last updated: **{date_str}** | Auto-generated by `python -m src.dashboard.generate`

---

## Current Status

![Status](data/outputs/dashboard/status.png)

---

## Risk Score Timeline

![Timeline](data/outputs/dashboard/timeline.png)

---

## Peak = Last Exit Signal

> 스코어 고점의 높이가 후속 충격의 강도와 시급성을 예고한다.
> 고점에서 하락하는 구간은 안전 신호가 아니라 **트리거 대기 구간**이다.

![Peak Pattern](data/outputs/dashboard/peak_pattern.png)

---

## Property Activation Heatmap

> 12개 유의미 성질의 빈도 비율. 1.0 = baseline, 1.5+ = 극강 신호.

![Heatmap](data/outputs/dashboard/heatmap.png)

---

## Reading Guide

| Score Range | Status | Meaning |
|:-----------:|:------:|---------|
| 0 ~ 18.5 | NORMAL | Baseline risk level |
| 18.5 ~ 40 | WARNING | Elevated structural stress |
| 40 ~ 60 | WARNING | High risk — event likely within months |
| 60+ | WARNING | Extreme — event likely within weeks |

| Peak → Drop Pattern | Interpretation |
|---------------------|----------------|
| Score rising | Structural vulnerability accumulating |
| **Score at peak** | **Last exit window** |
| Score dropping | News fatigue — NOT improvement |
| Score dropping + still above threshold | Trigger imminent |

---

> This dashboard is auto-generated from model outputs. Not financial advice.
>
> See [`insight_2026-03-04_iran_crisis.md`](data/outputs/insight_2026-03-04_iran_crisis.md) and
> [`insight_2026-03-07_peak_exit_signal.md`](data/outputs/insight_2026-03-07_peak_exit_signal.md)
> for detailed analysis.
"""
    Path("DASHBOARD.md").write_text(md, encoding="utf-8")
    print("  DASHBOARD.md")


def main():
    _setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scores...")
    df = load_scores()
    detail_df = parse_details(df)

    print("Generating charts...")
    chart_status(df, detail_df)
    chart_timeline(df)
    chart_heatmap(detail_df)
    chart_peak_pattern(df)

    print("Generating DASHBOARD.md...")
    generate_dashboard_md(df)

    print("Done.")


if __name__ == "__main__":
    main()
