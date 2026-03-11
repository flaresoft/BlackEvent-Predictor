"""
KOSIS Open API 클라이언트 — 사회 지표 데이터 수집
자살 사망자 수, 범죄 발생건수, 실업률 (월별)

API 키 발급: https://kosis.kr/openapi/ (즉시 발급)
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils import load_env, setup_logging

logger = setup_logging("module9.collector")

KOSIS_BASE_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


def _get_api_key() -> str:
    """KOSIS API 키를 환경변수에서 로드한다."""
    load_env()
    key = os.environ.get("KOSIS_API_KEY", "")
    if not key:
        raise ValueError(
            "KOSIS_API_KEY가 설정되지 않았습니다. "
            "https://kosis.kr/openapi/ 에서 발급 후 .env에 추가하세요."
        )
    return key


def test_api_key() -> bool:
    """API 키가 유효한지 테스트한다."""
    try:
        api_key = _get_api_key()
    except ValueError as e:
        logger.error(str(e))
        return False

    # 간단한 테이블 조회로 키 테스트
    params = {
        "method": "getList",
        "apiKey": api_key,
        "itmId": "ALL",
        "objL1": "ALL",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": "202501",
        "endPrdDe": "202501",
        "orgId": "101",
        "tblId": "DT_2KAA335",
    }
    try:
        resp = requests.get(KOSIS_BASE_URL, params=params, timeout=15)
        data = resp.json()
        if isinstance(data, dict) and data.get("err"):
            logger.error(f"API 키 오류: {data.get('err', '')}")
            return False
        if isinstance(data, list) and len(data) > 0:
            logger.info("KOSIS API 키 검증 성공")
            return True
        logger.warning("KOSIS API 응답이 비어있습니다. 키는 유효하지만 데이터가 없을 수 있습니다.")
        return True
    except Exception as e:
        logger.error(f"API 테스트 실패: {e}")
        return False


def explore_table(org_id: str, tbl_id: str) -> pd.DataFrame:
    """테이블의 구조(항목, 분류)를 탐색한다. 올바른 itmId/objL1 확인용."""
    api_key = _get_api_key()
    params = {
        "method": "getList",
        "apiKey": api_key,
        "itmId": "ALL",
        "objL1": "ALL",
        "objL2": "ALL",
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": "202401",
        "endPrdDe": "202401",
        "orgId": org_id,
        "tblId": tbl_id,
    }

    logger.info(f"테이블 탐색: orgId={org_id}, tblId={tbl_id}")
    try:
        resp = requests.get(KOSIS_BASE_URL, params=params, timeout=30)
        data = resp.json()
    except Exception as e:
        logger.error(f"탐색 실패: {e}")
        return pd.DataFrame()

    if isinstance(data, dict) and data.get("err"):
        logger.error(f"오류: {data.get('err', '')}")
        return pd.DataFrame()

    if not isinstance(data, list):
        logger.warning("응답이 리스트가 아닙니다.")
        return pd.DataFrame()

    # 응답 구조 분석
    df = pd.DataFrame(data)
    if not df.empty:
        key_cols = [c for c in df.columns if c in [
            "TBL_NM", "ITM_ID", "ITM_NM", "ITM_NM_ENG",
            "OBJ_L1_ID", "OBJ_L1_NM", "OBJ_L2_ID", "OBJ_L2_NM",
            "PRD_DE", "DT", "UNIT_NM",
        ]]
        logger.info(f"  응답: {len(df)}행, 컬럼: {key_cols}")
        if key_cols:
            unique_items = df[key_cols].drop_duplicates()
            for _, row in unique_items.head(20).iterrows():
                parts = [f"{c}={row[c]}" for c in key_cols if pd.notna(row[c])]
                logger.info(f"    {', '.join(parts)}")

    return df


def fetch_kosis_monthly(
    org_id: str,
    tbl_id: str,
    item_id: str,
    start_year: int,
    end_year: int,
    obj_l1: str = "ALL",
    obj_l2: str = "",
) -> pd.DataFrame:
    """KOSIS Open API에서 월별 통계 데이터를 가져온다.

    API가 여러 항목/분류를 반환할 경우, 월별로 합산한다.

    Returns:
        DataFrame with columns: [year_month, value]
    """
    api_key = _get_api_key()

    start_period = f"{start_year}01"
    end_period = f"{end_year}12"

    params = {
        "method": "getList",
        "apiKey": api_key,
        "itmId": item_id,
        "objL1": obj_l1,
        "objL2": obj_l2,
        "format": "json",
        "jsonVD": "Y",
        "prdSe": "M",
        "startPrdDe": start_period,
        "endPrdDe": end_period,
        "orgId": org_id,
        "tblId": tbl_id,
    }

    # 빈 파라미터 제거
    params = {k: v for k, v in params.items() if v}

    logger.info(f"KOSIS API 호출: orgId={org_id}, tblId={tbl_id}, {start_period}~{end_period}")

    try:
        resp = requests.get(KOSIS_BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error(f"KOSIS API 요청 실패: {e}")
        return pd.DataFrame()

    if isinstance(data, dict) and data.get("err"):
        err_msg = data.get("err", "")
        logger.error(f"KOSIS API 오류: {err_msg}")
        if "30" in str(err_msg):
            logger.info("  힌트: 테이블 ID가 잘못되었을 수 있습니다. explore_table()로 확인하세요.")
        return pd.DataFrame()

    if not isinstance(data, list) or len(data) == 0:
        logger.warning(f"KOSIS API 응답이 비어있습니다 (orgId={org_id}, tblId={tbl_id})")
        return pd.DataFrame()

    # 응답 구조 로깅
    sample = data[0]
    logger.info(
        f"  응답 {len(data)}건, "
        f"테이블: {sample.get('TBL_NM', '?')}, "
        f"항목: {sample.get('ITM_NM', '?')}, "
        f"단위: {sample.get('UNIT_NM', '?')}"
    )

    # 월별 데이터 추출 — 여러 분류가 있으면 합산
    monthly_values: dict[str, float] = {}
    for item in data:
        prd = item.get("PRD_DE", "")
        val_str = item.get("DT", "")
        if len(prd) < 6 or not val_str or val_str == "-":
            continue

        year_month = f"{prd[:4]}-{prd[4:6]}"
        try:
            value = float(val_str.replace(",", ""))
        except ValueError:
            continue

        # 합산 (여러 분류/항목이 반환될 경우)
        monthly_values[year_month] = monthly_values.get(year_month, 0) + value

    if not monthly_values:
        logger.warning("  파싱된 데이터가 없습니다.")
        return pd.DataFrame()

    records = [{"year_month": k, "value": v} for k, v in sorted(monthly_values.items())]
    df = pd.DataFrame(records)
    logger.info(f"  수집 완료: {len(df)}개월 ({df['year_month'].iloc[0]} ~ {df['year_month'].iloc[-1]})")

    return df


def load_or_fetch(
    indicator_name: str,
    indicator_config: dict,
    cache_dir: Path,
    start_year: int = 2006,
    end_year: int = 2026,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """캐시된 데이터가 있으면 로드하고, 없으면 API에서 가져온다."""
    cache_path = cache_dir / f"{indicator_name}_raw.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path)
        logger.info(f"캐시 로드: {indicator_name} ({len(df)}개월) — {cache_path}")
        return df

    try:
        df = fetch_kosis_monthly(
            org_id=indicator_config["org_id"],
            tbl_id=indicator_config["tbl_id"],
            item_id=indicator_config.get("item_id", "ALL"),
            start_year=start_year,
            end_year=end_year,
            obj_l1=indicator_config.get("obj_l1", "ALL"),
            obj_l2=indicator_config.get("obj_l2", ""),
        )
    except ValueError as e:
        logger.error(str(e))
        # API 키 없으면 캐시 확인
        if cache_path.exists():
            logger.info(f"  API 키 없음, 기존 캐시 사용: {cache_path}")
            return pd.read_csv(cache_path)
        return pd.DataFrame()

    if not df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"캐시 저장: {cache_path}")

    return df


def load_all_indicators(
    social_config: dict,
    cache_dir: Path,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """모든 사회 지표 데이터를 로드/수집한다."""
    indicators = social_config.get("indicators", {})
    result = {}

    for name, config in indicators.items():
        df = load_or_fetch(
            indicator_name=name,
            indicator_config=config,
            cache_dir=cache_dir,
            start_year=social_config.get("start_year", 2006),
            end_year=social_config.get("end_year", 2026),
            force_refresh=force_refresh,
        )
        result[name] = df
        time.sleep(0.5)  # API 예의

    return result


def load_csv_manual(csv_path: Path, indicator_name: str) -> pd.DataFrame:
    """KOSIS에서 수동 다운로드한 CSV를 로드한다.
    수동 CSV는 cache_dir/{indicator_name}_raw.csv에 넣으면 된다.
    컬럼은 year_month (YYYY-MM), value 두 개.
    """
    if not csv_path.exists():
        logger.warning(f"수동 CSV 없음: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if "year_month" not in df.columns or "value" not in df.columns:
        logger.error(
            f"수동 CSV 형식 오류: {csv_path} — "
            "컬럼은 year_month (YYYY-MM), value 여야 합니다."
        )
        return pd.DataFrame()

    logger.info(f"수동 CSV 로드: {indicator_name} ({len(df)}개월)")
    return df
