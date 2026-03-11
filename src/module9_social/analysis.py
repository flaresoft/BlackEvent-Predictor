"""
사회 지표 상관분석 — 교차상관, Granger 인과성, 롤링 상관
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import setup_logging

logger = setup_logging("module9.analysis")


def cross_correlation(
    series_x: pd.Series,
    series_y: pd.Series,
    max_lag: int = 12,
) -> pd.DataFrame:
    """시차 교차상관분석. x가 y를 lag개월 선행한다고 가정.

    lag > 0: x가 y보다 lag개월 먼저 (x → y 인과 가설)
    lag = 0: 동시 상관

    Returns:
        DataFrame [lag, correlation, p_value, significant]
    """
    x = series_x.dropna().values
    y = series_y.dropna().values

    # 공통 인덱스 기반으로 정렬
    common = series_x.dropna().index.intersection(series_y.dropna().index)
    if len(common) < 10:
        logger.warning(f"공통 데이터 부족: {len(common)}개월 (최소 10개월 필요)")
        return pd.DataFrame()

    x_aligned = series_x.loc[common].values
    y_aligned = series_y.loc[common].values

    records = []
    n = len(common)

    for lag in range(0, max_lag + 1):
        if lag >= n - 5:
            break

        # x[:-lag] vs y[lag:] (x가 lag개월 선행)
        if lag == 0:
            x_lagged = x_aligned
            y_lagged = y_aligned
        else:
            x_lagged = x_aligned[:-lag]
            y_lagged = y_aligned[lag:]

        if len(x_lagged) < 5:
            break

        r, p = stats.pearsonr(x_lagged, y_lagged)
        records.append({
            "lag": lag,
            "correlation": round(r, 4),
            "p_value": p,
            "n_obs": len(x_lagged),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        # Bonferroni 보정
        df["p_adjusted"] = (df["p_value"] * len(df)).clip(upper=1.0)
        df["significant"] = df["p_adjusted"] < 0.05

    return df


def granger_causality_test(
    merged_df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 6,
) -> pd.DataFrame:
    """Granger 인과성 검정. cause_col이 effect_col을 Granger-cause하는지.

    ADF 검정으로 정상성 확인 후 필요 시 차분 적용.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    except ImportError:
        logger.error("statsmodels가 필요합니다: pip install statsmodels")
        return pd.DataFrame()

    df = merged_df[[cause_col, effect_col]].dropna()
    if len(df) < max_lag + 10:
        logger.warning(
            f"Granger 검정에 데이터 부족: {len(df)}행 "
            f"(최소 {max_lag + 10}행 필요)"
        )
        return pd.DataFrame()

    # 정상성 검정 및 차분
    diff_applied = {}
    for col in [cause_col, effect_col]:
        try:
            adf_result = adfuller(df[col].values, maxlag=max_lag)
            if adf_result[1] > 0.05:
                df[col] = df[col].diff()
                diff_applied[col] = True
                logger.info(f"  {col}: 비정상 → 1차 차분 적용 (ADF p={adf_result[1]:.4f})")
            else:
                diff_applied[col] = False
        except Exception as e:
            logger.warning(f"  ADF 검정 실패 ({col}): {e}")
            diff_applied[col] = False

    df = df.dropna()
    if len(df) < max_lag + 5:
        logger.warning(f"차분 후 데이터 부족: {len(df)}행")
        return pd.DataFrame()

    # Granger 검정
    try:
        test_data = df[[effect_col, cause_col]].values
        results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
    except Exception as e:
        logger.error(f"Granger 검정 실패: {e}")
        return pd.DataFrame()

    records = []
    for lag in range(1, max_lag + 1):
        if lag in results:
            test_result = results[lag]
            f_stat = test_result[0]["ssr_ftest"][0]
            p_value = test_result[0]["ssr_ftest"][1]
            records.append({
                "lag": lag,
                "f_statistic": round(f_stat, 4),
                "p_value": p_value,
                "significant": p_value < 0.05,
                "cause_diffed": diff_applied.get(cause_col, False),
                "effect_diffed": diff_applied.get(effect_col, False),
            })

    return pd.DataFrame(records)


def rolling_correlation(
    series_x: pd.Series,
    series_y: pd.Series,
    window: int = 12,
    best_lag: int = 0,
) -> pd.DataFrame:
    """롤링 상관계수. 관계의 시간적 안정성을 본다."""
    common = series_x.dropna().index.intersection(series_y.dropna().index)
    if len(common) < window + best_lag:
        logger.warning(f"롤링 상관 데이터 부족: {len(common)}개월")
        return pd.DataFrame()

    x = series_x.loc[common]
    y = series_y.loc[common]

    if best_lag > 0:
        x = x.iloc[:-best_lag].reset_index(drop=True)
        y = y.iloc[best_lag:].reset_index(drop=True)
        index_labels = common[best_lag:]
    else:
        index_labels = common

    rolling_corr = x.rolling(window=window).corr(y)

    records = []
    for i, idx in enumerate(index_labels):
        if i < window - 1:
            continue
        val = rolling_corr.iloc[i] if i < len(rolling_corr) else None
        if val is not None and not np.isnan(val):
            records.append({
                "year_month": idx,
                "rolling_corr": round(val, 4),
                "window": window,
                "lag": best_lag,
            })

    return pd.DataFrame(records)


def run_full_analysis(
    merged_df: pd.DataFrame,
    social_config: dict,
) -> dict:
    """모든 사회 지표에 대해 전체 분석을 실행한다."""
    corr_config = social_config.get("correlation", {})
    granger_config = social_config.get("granger", {})
    max_lag = corr_config.get("max_lag_months", 12)
    rolling_window = corr_config.get("rolling_window_months", 12)
    granger_max_lag = granger_config.get("max_lag", 6)

    risk_metrics = ["risk_mean", "risk_max", "warning_days"]
    social_indicators = [c for c in merged_df.columns
                         if c not in ["year_month", "risk_mean", "risk_max",
                                      "risk_std", "warning_days", "n_days"]]

    if not social_indicators:
        logger.warning("사회 지표 컬럼이 없습니다.")
        return {}

    # 인덱스 설정
    df = merged_df.set_index("year_month").sort_index()

    results = {
        "cross_correlation": [],
        "granger": [],
        "rolling": [],
        "summary": [],
    }

    for indicator in social_indicators:
        for metric in risk_metrics:
            if metric not in df.columns or indicator not in df.columns:
                continue

            pair_label = f"{metric} → {indicator}"
            logger.info(f"분석: {pair_label}")

            # 교차상관
            xcorr = cross_correlation(df[metric], df[indicator], max_lag=max_lag)
            if not xcorr.empty:
                xcorr["risk_metric"] = metric
                xcorr["social_indicator"] = indicator
                results["cross_correlation"].append(xcorr)

                # 최적 lag 찾기
                best_row = xcorr.loc[xcorr["correlation"].abs().idxmax()]
                best_lag = int(best_row["lag"])
                best_r = best_row["correlation"]
                best_p = best_row["p_adjusted"]

                results["summary"].append({
                    "risk_metric": metric,
                    "social_indicator": indicator,
                    "best_lag": best_lag,
                    "best_correlation": best_r,
                    "best_p_adjusted": best_p,
                    "significant": best_p < 0.05,
                })

                logger.info(
                    f"  최적 lag={best_lag}개월, r={best_r:.3f}, "
                    f"p_adj={best_p:.4f} {'*' if best_p < 0.05 else ''}"
                )

                # 롤링 상관 (최적 lag 사용)
                roll = rolling_correlation(
                    df[metric], df[indicator],
                    window=rolling_window, best_lag=best_lag,
                )
                if not roll.empty:
                    roll["risk_metric"] = metric
                    roll["social_indicator"] = indicator
                    results["rolling"].append(roll)

            # Granger 검정
            granger = granger_causality_test(
                df.reset_index(), metric, indicator, max_lag=granger_max_lag,
            )
            if not granger.empty:
                granger["risk_metric"] = metric
                granger["social_indicator"] = indicator
                results["granger"].append(granger)

                sig_lags = granger[granger["significant"]]["lag"].tolist()
                if sig_lags:
                    logger.info(f"  Granger 유의: lag {sig_lags}")

    # 결합
    for key in ["cross_correlation", "granger", "rolling"]:
        if results[key]:
            results[key] = pd.concat(results[key], ignore_index=True)
        else:
            results[key] = pd.DataFrame()

    # summary는 dict 리스트 → DataFrame
    if results["summary"]:
        results["summary"] = pd.DataFrame(results["summary"])
    else:
        results["summary"] = pd.DataFrame()

    return results
