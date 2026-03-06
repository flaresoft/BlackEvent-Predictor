"""
US Futures Market Monitor — 미국 선물시장 실시간 상태를 추적한다.

선물 가격 자체가 시장 참여자들의 포지션과 공포를 직접 보여준다.
뉴스 해석이 아닌, 가격·변동성·스프레드로 시장의 성질을 읽는다.

Usage:
    python -m src.futures_monitor.run              # 전체 선물 현황
    python -m src.futures_monitor.run --watch       # 반복 모니터링 (60초 간격)
    python -m src.futures_monitor.run --json        # JSON 출력
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from src.utils import setup_logging

logger = setup_logging("futures_monitor")

# ── 추적 대상 선물 ──
FUTURES = {
    "ES=F": {"name": "S&P 500 E-mini", "short": "ES"},
    "NQ=F": {"name": "NASDAQ 100 E-mini", "short": "NQ"},
    "YM=F": {"name": "Dow Jones E-mini", "short": "YM"},
    "RTY=F": {"name": "Russell 2000 E-mini", "short": "RTY"},
    "^VIX": {"name": "VIX (공포지수)", "short": "VIX"},
    "ZN=F": {"name": "10Y T-Note", "short": "ZN"},
    "GC=F": {"name": "Gold", "short": "Gold"},
    "CL=F": {"name": "Crude Oil WTI", "short": "Oil"},
}

# ── 경보 기준 ──
ALERT_THRESHOLDS = {
    "equity_drop": -1.0,     # 주가지수 선물 -1% 이하
    "equity_crash": -2.0,    # 주가지수 선물 -2% 이하 (급락)
    "vix_elevated": 20.0,    # VIX 20 이상
    "vix_spike": 30.0,       # VIX 30 이상 (공포)
    "vix_change": 10.0,      # VIX 일간 변동 +10% 이상
}

EQUITY_TICKERS = {"ES=F", "NQ=F", "YM=F", "RTY=F"}


def fetch_futures_data() -> dict:
    """선물 시세를 가져온다. 가격, 변동폭, 전일 종가."""
    results = {}

    for ticker, meta in FUTURES.items():
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d")

            if hist.empty or len(hist) < 1:
                results[ticker] = {**meta, "error": "데이터 없음"}
                continue

            current = hist.iloc[-1]
            price = current["Close"]

            if len(hist) >= 2:
                prev_close = hist.iloc[-2]["Close"]
                change = price - prev_close
                change_pct = (change / prev_close) * 100
            else:
                prev_close = price
                change = 0.0
                change_pct = 0.0

            # 5일 고가/저가 범위
            high_5d = hist["High"].max()
            low_5d = hist["Low"].min()
            range_5d = ((high_5d - low_5d) / low_5d) * 100 if low_5d > 0 else 0

            results[ticker] = {
                **meta,
                "price": round(price, 2),
                "prev_close": round(prev_close, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "high_5d": round(high_5d, 2),
                "low_5d": round(low_5d, 2),
                "range_5d_pct": round(range_5d, 2),
                "last_date": str(hist.index[-1].date()),
            }

        except Exception as e:
            results[ticker] = {**meta, "error": str(e)}

    return results


def evaluate_alerts(data: dict) -> list[dict]:
    """경보 조건을 평가한다."""
    alerts = []

    for ticker, d in data.items():
        if "error" in d:
            continue

        # 주가지수 선물 급락
        if ticker in EQUITY_TICKERS:
            if d["change_pct"] <= ALERT_THRESHOLDS["equity_crash"]:
                alerts.append({
                    "level": "CRITICAL",
                    "ticker": d["short"],
                    "message": f"{d['name']} {d['change_pct']:+.2f}% 급락",
                    "value": d["change_pct"],
                })
            elif d["change_pct"] <= ALERT_THRESHOLDS["equity_drop"]:
                alerts.append({
                    "level": "WARNING",
                    "ticker": d["short"],
                    "message": f"{d['name']} {d['change_pct']:+.2f}% 하락",
                    "value": d["change_pct"],
                })

        # VIX 수준
        if ticker == "^VIX":
            if d["price"] >= ALERT_THRESHOLDS["vix_spike"]:
                alerts.append({
                    "level": "CRITICAL",
                    "ticker": "VIX",
                    "message": f"VIX {d['price']:.1f} — 극단적 공포 구간",
                    "value": d["price"],
                })
            elif d["price"] >= ALERT_THRESHOLDS["vix_elevated"]:
                alerts.append({
                    "level": "WARNING",
                    "ticker": "VIX",
                    "message": f"VIX {d['price']:.1f} — 불안 구간",
                    "value": d["price"],
                })

            # VIX 급등
            if d["change_pct"] >= ALERT_THRESHOLDS["vix_change"]:
                alerts.append({
                    "level": "WARNING",
                    "ticker": "VIX",
                    "message": f"VIX {d['change_pct']:+.2f}% 급등",
                    "value": d["change_pct"],
                })

    # 동반 하락 감지: 주요 지수 3개 이상 동시 -1%
    dropping = [t for t in EQUITY_TICKERS
                if t in data and "error" not in data[t]
                and data[t]["change_pct"] <= ALERT_THRESHOLDS["equity_drop"]]
    if len(dropping) >= 3:
        alerts.append({
            "level": "CRITICAL",
            "ticker": "ALL",
            "message": f"주요 지수 {len(dropping)}개 동반 하락 — 시장 전반 약세",
            "value": len(dropping),
        })

    return alerts


def compute_kospi_impact(data: dict) -> dict:
    """미국 선물 움직임으로부터 KOSPI 영향을 추정한다.

    단순 경험적 가중치 기반. 통계 모델이 아닌 참고 수준.
    """
    weights = {"ES=F": 0.45, "NQ=F": 0.35, "^VIX": -0.10, "CL=F": 0.10}
    impact = 0.0
    available = 0

    for ticker, w in weights.items():
        if ticker in data and "error" not in data[ticker]:
            if ticker == "^VIX":
                # VIX는 역방향: VIX 상승 → KOSPI 하락 압력
                impact += w * data[ticker]["change_pct"]
            else:
                impact += w * data[ticker]["change_pct"]
            available += 1

    if available == 0:
        return {"estimated_impact_pct": 0.0, "confidence": "N/A"}

    confidence = "참고용" if available >= 3 else "불확실"
    return {
        "estimated_impact_pct": round(impact, 2),
        "confidence": confidence,
    }


def print_report(data: dict, alerts: list[dict], impact: dict):
    """터미널에 리포트를 출력한다."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(f"{'=' * 72}")
    print(f"  US Futures Monitor | {now}")
    print(f"{'=' * 72}")

    # 경보
    if alerts:
        print()
        for a in alerts:
            marker = "***" if a["level"] == "CRITICAL" else " ! "
            print(f"  {marker} [{a['level']}] {a['message']}")
        print()

    # 시세 테이블
    print(f"  {'':6s} {'Name':<22s} {'Price':>10s} {'Chg':>8s} {'Chg%':>7s}  {'5D Range':>8s}")
    print(f"  {'-' * 66}")

    for ticker, d in data.items():
        if "error" in d:
            print(f"  {d['short']:6s} {d['name']:<22s} {'(error)':>10s}")
            continue

        chg_str = f"{d['change']:+.2f}"
        pct_str = f"{d['change_pct']:+.2f}%"
        rng_str = f"{d['range_5d_pct']:.1f}%"

        # 변동 방향 표시
        if d["change_pct"] <= -2:
            indicator = " <<"
        elif d["change_pct"] <= -1:
            indicator = "  <"
        elif d["change_pct"] >= 2:
            indicator = " >>"
        elif d["change_pct"] >= 1:
            indicator = "  >"
        else:
            indicator = "   "

        print(f"  {d['short']:6s} {d['name']:<22s} {d['price']:>10,.2f} "
              f"{chg_str:>8s} {pct_str:>7s}{indicator} {rng_str:>8s}")

    # KOSPI 영향 추정
    print()
    print(f"  KOSPI 예상 영향: {impact['estimated_impact_pct']:+.2f}% ({impact['confidence']})")
    print(f"{'=' * 72}")
    print()


def run(watch: bool = False, interval: int = 60, as_json: bool = False) -> Optional[dict]:
    """메인 실행."""
    while True:
        logger.info("선물 데이터 수집 중...")
        data = fetch_futures_data()
        alerts = evaluate_alerts(data)
        impact = compute_kospi_impact(data)

        if as_json:
            output = {
                "timestamp": datetime.now().isoformat(),
                "futures": data,
                "alerts": alerts,
                "kospi_impact": impact,
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return output

        print_report(data, alerts, impact)

        if not watch:
            return {"futures": data, "alerts": alerts, "kospi_impact": impact}

        logger.info(f"{interval}초 후 갱신...")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="US Futures Market Monitor")
    parser.add_argument("--watch", action="store_true",
                        help="반복 모니터링 모드")
    parser.add_argument("--interval", type=int, default=60,
                        help="모니터링 갱신 주기 (초, 기본 60)")
    parser.add_argument("--json", action="store_true",
                        help="JSON 형식 출력")
    args = parser.parse_args()
    run(watch=args.watch, interval=args.interval, as_json=args.json)


if __name__ == "__main__":
    main()
