"""
Daily Pipeline — 뉴스 수집 → 태깅 → 스코어링 전체 자동화

매일 실행하면:
  1. BigKinds에서 새 뉴스 다운로드
  2. 임베딩 + 성질 태깅 → corpus_daily/에 저장
  3. 기존 코퍼스 + daily 파일 합쳐서 리스크 스코어 산출

Usage:
    python -m src.daily_pipeline.run                  # 어제 날짜 뉴스 수집 + 스코어
    python -m src.daily_pipeline.run --date 2026-02-23  # 특정 날짜
    python -m src.daily_pipeline.run --score-only       # 수집/태깅 스킵, 스코어만
"""

import argparse
import gc
import json
import os
import re
import time
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils import load_config, get_path, setup_logging

logger = setup_logging("daily_pipeline")

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
THRESHOLD = 18.52
COOKIE_FILE = Path("config/bigkinds_cookie.txt")

BIGKINDS_BASE = "https://www.bigkinds.or.kr"
CATEGORY_CODES = [
    "001000000", "001005000", "001001000", "001004000",
    "001003000", "001007000", "001002000", "001006000",
    "002000000", "002004000", "002010000", "002008000",
    "002014000", "002011000", "002009000", "002005000",
    "002001000", "002012000", "002006000", "002002000",
    "002007000", "002003000", "002013000",
    "003000000", "003007000", "003003000", "003006000",
    "003002000", "003009000", "003001000", "003005000",
    "003010000", "003008000", "003004000",
]

PROP_NAMES = {
    "Property_001": "거시경제 지표의 급변",
    "Property_013": "금융기관 간 신뢰 붕괴",
    "Property_023": "자산 가격의 급격한 변동",
    "Property_024": "외부 요인에 의한 시장 반응",
    "Property_030": "정부 개입과 시장 반응",
    "Property_049": "글로벌 금융 시스템의 상호 연결성",
    "Property_055": "수요와 공급의 불균형",
    "Property_068": "투자자 심리의 급격한 변화",
    "Property_070": "금융 규제와 정책 변화",
    "Property_078": "구조적 불안정성의 확산",
    "Property_110": "시장 참여자의 과도한 레버리지",
    "Property_113": "위기 확산의 비선형적 특성",
}


# ──────────────────────────────────────────────
# Step 1: BigKinds 뉴스 수집
# ──────────────────────────────────────────────
def _read_cookie() -> str:
    """config/bigkinds_cookie.txt에서 쿠키를 읽는다."""
    if not COOKIE_FILE.exists():
        return ""
    return COOKIE_FILE.read_text(encoding="utf-8").strip()


def _build_bigkinds_param(start_date: str, end_date: str) -> dict:
    return {
        "indexName": "news",
        "searchKey": "  ",
        "searchKeys": [{}],
        "byLine": "",
        "searchFilterType": "1",
        "searchScopeType": "1",
        "searchSortType": "date",
        "sortMethod": "date",
        "mainTodayPersonYn": "",
        "startDate": start_date,
        "endDate": end_date,
        "newsIds": [],
        "categoryCodes": CATEGORY_CODES,
        "providerCodes": [],
        "incidentCodes": [],
        "networkNodeType": "",
        "topicOrigin": "",
        "dateCodes": [],
        "editorialIs": False,
        "startNo": 1,
        "resultNumber": 10,
        "isTmUsable": False,
        "isNotTmUsable": False,
    }


def collect_news(target_date: str, download_dir: Path) -> Path | None:
    """BigKinds에서 하루치 뉴스를 다운로드한다.

    Returns:
        저장된 xlsx 파일 경로, 실패 시 None
    """
    import requests

    cookie = _read_cookie()
    if not cookie:
        logger.warning(
            "BigKinds 쿠키 없음. config/bigkinds_cookie.txt에 쿠키를 넣어주세요."
        )
        return None

    filename = f"bigkinds_{target_date}_{target_date}.xlsx"
    save_path = download_dir / filename

    if save_path.exists() and os.path.getsize(save_path) > 1024:
        logger.info(f"이미 다운로드됨: {filename}")
        return save_path

    headers_json = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/json;charset=UTF-8",
        "Origin": BIGKINDS_BASE,
        "Referer": f"{BIGKINDS_BASE}/v2/news/search.do",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Cookie": cookie,
    }
    headers_download = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": BIGKINDS_BASE,
        "Referer": f"{BIGKINDS_BASE}/v2/news/search.do",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "Cookie": cookie,
    }

    session = requests.Session()

    # 건수 확인
    try:
        param = _build_bigkinds_param(target_date, target_date)
        param["resultNumber"] = 1
        resp = session.post(
            f"{BIGKINDS_BASE}/api/news/search.do",
            headers=headers_json,
            json=param,
            timeout=30,
        )
        resp.raise_for_status()
        count = resp.json().get("totalCount", 0)
    except Exception as e:
        logger.error(f"BigKinds 건수 조회 실패: {e}")
        return None

    if count == 0:
        logger.info(f"{target_date}: 기사 0건 (주말/공휴일?)")
        return None

    logger.info(f"{target_date}: {count:,}건 다운로드 중...")

    # 다운로드
    try:
        param = _build_bigkinds_param(target_date, target_date)
        param["sectionDiv"] = "1000"
        param["realURI"] = "/api/news/previewData.do"
        param["interval"] = 2

        form_data = "jsonSearchParam=" + urllib.parse.quote(
            json.dumps(param, ensure_ascii=False)
        )
        resp = session.post(
            f"{BIGKINDS_BASE}/api/news/download.do",
            headers=headers_download,
            data=form_data.encode("utf-8"),
            timeout=180,
            stream=True,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type.lower() and len(resp.content) < 5000:
            logger.warning("BigKinds 세션 만료! 쿠키를 갱신해주세요.")
            return None

        download_dir.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = os.path.getsize(save_path) / 1024
        logger.info(f"저장: {filename} ({size_kb:.0f} KB, {count:,}건)")
        return save_path

    except Exception as e:
        logger.error(f"다운로드 실패: {e}")
        if save_path.exists():
            save_path.unlink()
        return None


# ──────────────────────────────────────────────
# Step 2: Excel → 태깅된 Parquet
# ──────────────────────────────────────────────
def _extract_lead(text: str, max_chars: int = 200) -> str:
    """본문에서 리드문(1~2문장)을 추출한다."""
    if not isinstance(text, str) or not text.strip():
        return ""
    snippet = text[:max_chars]
    # 마지막 문장 부호에서 자르기
    for sep in [".", "다.", "요.", "음."]:
        idx = snippet.rfind(sep)
        if idx > 50:
            return snippet[: idx + len(sep)]
    return snippet


def process_and_tag(xlsx_path: Path, daily_dir: Path, config: dict) -> Path | None:
    """Excel 파일을 읽고, 임베딩+태깅하여 parquet로 저장한다.

    Returns:
        저장된 parquet 경로, 실패 시 None
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from src.module4_tagging.run import build_property_vectors, tag_chunk

    outputs_path = get_path(config, "outputs")
    dict_config = config.get("dictionary", {})
    model_name = dict_config.get("embedding_models", {}).get("main", "nlpai-lab/KURE-v1")

    # 날짜 추출
    stem = xlsx_path.stem
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    if not date_match:
        logger.error(f"파일명에서 날짜를 추출할 수 없습니다: {stem}")
        return None
    date_str = date_match.group(1)

    out_path = daily_dir / f"{date_str}.parquet"
    if out_path.exists():
        logger.info(f"이미 태깅됨: {out_path.name}")
        return out_path

    # ── Excel 읽기 ──
    logger.info(f"Excel 읽기: {xlsx_path.name}")
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception:
        try:
            for enc in ["utf-8", "cp949", "euc-kr"]:
                try:
                    df = pd.read_csv(xlsx_path, encoding=enc)
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            else:
                logger.error(f"파일 읽기 실패: {xlsx_path}")
                return None
        except Exception as e:
            logger.error(f"파일 읽기 실패: {e}")
            return None

    # 컬럼 매핑
    col_map = {}
    for col in df.columns:
        cl = col.strip()
        if "식별자" in cl:
            col_map[col] = "article_id"
        elif cl == "일자":
            col_map[col] = "date"
        elif cl == "제목":
            col_map[col] = "title"
        elif cl == "본문":
            col_map[col] = "body"

    df = df.rename(columns=col_map)

    required = ["date", "title"]
    for r in required:
        if r not in df.columns:
            logger.error(f"필수 컬럼 누락: {r}")
            return None

    # 리드문 추출
    if "body" in df.columns:
        df["lead"] = df["body"].apply(_extract_lead)
        df = df.drop(columns=["body"])
    elif "lead" not in df.columns:
        df["lead"] = ""

    # article_id 보정
    if "article_id" not in df.columns:
        df["article_id"] = [f"daily_{date_str}_{i}" for i in range(len(df))]
    df["article_id"] = df["article_id"].astype(str)

    # 날짜 형식 통일
    df["date"] = df["date"].astype(str).str.replace(r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", regex=True)

    # 분석제외 제거
    if "분석제외 여부" in df.columns:
        df = df[df["분석제외 여부"].isna()].copy()

    # 중복 제거
    if "article_id" in df.columns:
        df = df.drop_duplicates(subset=["article_id"])

    logger.info(f"전처리 완료: {len(df):,}건")

    if len(df) == 0:
        logger.warning("기사 0건, 태깅 스킵")
        return None

    # ── 임베딩 모델 로드 ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"임베딩 모델 로드: {model_name} ({device})")
    embed_model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        embed_model.half()

    # ── 성질 사전 & 벡터 ──
    dict_path = outputs_path / "property_dictionary.json"
    with open(dict_path, "r", encoding="utf-8") as f:
        property_dict = json.load(f)
    property_ids = [p["property_id"] for p in property_dict]

    property_vectors = build_property_vectors(property_dict, embed_model)

    # ── 임계값 ──
    cal_path = outputs_path / "threshold_calibration.json"
    if cal_path.exists():
        with open(cal_path, "r", encoding="utf-8") as f:
            cal = json.load(f)
        per_prop = cal.get("per_property_thresholds", {})
        global_th = cal.get("global_threshold", 0.4)
        threshold_vec = np.array(
            [per_prop.get(pid, global_th) for pid in property_ids]
        )
    else:
        threshold_vec = np.full(len(property_ids), 0.4)

    # ── 임베딩 ──
    texts = (df["title"].fillna("") + " " + df["lead"].fillna("")).tolist()
    logger.info(f"임베딩: {len(texts):,}건...")
    embeddings = embed_model.encode(
        texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True,
    )

    # ── 태깅 ──
    tags = tag_chunk(embeddings, property_vectors, property_ids, threshold_vec)
    df["properties"] = tags

    n_tagged = sum(1 for t in tags if len(t) > 0)
    avg_tags = np.mean([len(t) for t in tags])
    logger.info(f"태깅 완료: {n_tagged:,}/{len(df):,}건 (평균 {avg_tags:.1f}개/기사)")

    # ── 메모리 정리 ──
    del embed_model, embeddings, property_vectors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Parquet 저장 ──
    save_cols = ["article_id", "date", "title", "lead", "properties"]
    existing_cols = [c for c in save_cols if c in df.columns]
    out_df = df[existing_cols]

    table = pa.Table.from_pandas(out_df, preserve_index=False)
    daily_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    logger.info(f"저장: {out_path} ({len(out_df):,}건)")

    return out_path


# ──────────────────────────────────────────────
# Step 3: 스코어링 (corpus_tagged + daily 합산)
# ──────────────────────────────────────────────
def load_daily_aggregation_combined(
    processed_path: Path,
    daily_dir: Path,
    date_min: str,
    date_max: str,
) -> tuple[np.ndarray, list[dict], list[int], int]:
    """메인 코퍼스 + daily 파일에서 일별 집계를 로드한다.

    메인 코퍼스에서 날짜 범위에 해당하는 데이터를 먼저 로드하고,
    daily 파일에서 추가 날짜를 보충한다.
    """
    daily = defaultdict(lambda: defaultdict(int))
    daily_article_count = defaultdict(int)
    total_articles = 0

    date_min_ts = pd.Timestamp(date_min)
    date_max_ts = pd.Timestamp(date_max)

    # ── 메인 코퍼스 먼저 (기준 데이터) ──
    corpus_dates_covered = set()
    corpus_path = processed_path / "corpus_tagged.parquet"
    if corpus_path.exists():
        pf = pq.ParquetFile(corpus_path)
        num_rg = pf.metadata.num_row_groups

        for rg_idx in range(num_rg):
            table = pf.read_row_group(rg_idx, columns=["date", "properties"])
            chunk = table.to_pandas()
            del table

            chunk["date"] = pd.to_datetime(chunk["date"])
            mask = (chunk["date"] >= date_min_ts) & (chunk["date"] <= date_max_ts)
            chunk = chunk[mask]

            for date_val, props in zip(chunk["date"].values, chunk["properties"].values):
                if isinstance(props, np.ndarray):
                    props_list = props.tolist()
                elif isinstance(props, str):
                    props_list = json.loads(props)
                elif isinstance(props, list):
                    props_list = props
                else:
                    continue
                daily_article_count[date_val] += 1
                total_articles += 1
                corpus_dates_covered.add(pd.Timestamp(date_val).strftime("%Y-%m-%d"))
                for prop in props_list:
                    daily[date_val][prop] += 1

            del chunk
            if (rg_idx + 1) % 50 == 0 or rg_idx == num_rg - 1:
                logger.info(f"  메인 코퍼스: RG {rg_idx+1}/{num_rg}")

    logger.info(f"메인 코퍼스: {len(corpus_dates_covered)}일 로드")

    # ── daily 파일 (메인 코퍼스에 없는 날짜만 보충) ──
    daily_dates_added = set()
    if daily_dir.exists():
        for fp in sorted(daily_dir.glob("*.parquet")):
            try:
                tbl = pq.read_table(fp)
                chunk = tbl.to_pandas()
                del tbl
            except Exception:
                continue

            chunk["date"] = pd.to_datetime(chunk["date"])
            mask = (chunk["date"] >= date_min_ts) & (chunk["date"] <= date_max_ts)
            chunk = chunk[mask]

            # 메인 코퍼스에 이미 있는 날짜는 스킵
            if corpus_dates_covered:
                chunk_dates = chunk["date"].dt.strftime("%Y-%m-%d")
                skip_mask = chunk_dates.isin(corpus_dates_covered)
                chunk = chunk[~skip_mask]

            for date_val, props in zip(chunk["date"].values, chunk["properties"].values):
                if isinstance(props, np.ndarray):
                    props_list = props.tolist()
                elif isinstance(props, str):
                    props_list = json.loads(props)
                elif isinstance(props, list):
                    props_list = props
                else:
                    continue
                daily_article_count[date_val] += 1
                total_articles += 1
                daily_dates_added.add(pd.Timestamp(date_val).strftime("%Y-%m-%d"))
                for prop in props_list:
                    daily[date_val][prop] += 1

            del chunk

    logger.info(f"Daily 보충: {len(daily_dates_added)}일 추가")

    sorted_dates = sorted(daily.keys())
    dates_arr = np.array(sorted_dates, dtype="datetime64[ns]")
    counts_list = [dict(daily[d]) for d in sorted_dates]
    articles_per_day = [daily_article_count[d] for d in sorted_dates]

    logger.info(f"합산 완료: {total_articles:,}건 / {len(dates_arr):,}일")
    return dates_arr, counts_list, articles_per_day, total_articles


def score(
    processed_path: Path,
    outputs_path: Path,
    daily_dir: Path,
    ref_date: pd.Timestamp,
    config: dict,
) -> dict:
    """리스크 스코어를 산출한다."""
    from src.module6_scoring.run import (
        compute_window_frequency,
        WeightedFrequencyModel,
    )

    scoring_config = config.get("scoring", {})
    normalization_mode = scoring_config.get("normalization_mode", "empirical")
    default_window = scoring_config.get("default_window", 60)

    # 유의미 성질
    sig_df = pd.read_csv(outputs_path / "significant_properties.csv")
    sig_props = sig_df["property_id"].tolist()
    optimal_windows: dict[str, int] = {}
    for _, r in sig_df.iterrows():
        optimal_windows[r["property_id"]] = int(r["optimal_window"])

    # Model B
    model_b = WeightedFrequencyModel(sig_df, normalization_mode=normalization_mode)
    risk_scores_df = pd.read_csv(outputs_path / "risk_scores.csv")
    empirical_max = risk_scores_df.loc[
        risk_scores_df["actual_label"] == 1, "model_b_raw_score"
    ].max()
    model_b.calibrate(empirical_max)

    # 데이터 로드
    max_window = max(optimal_windows.values()) if optimal_windows else default_window
    date_min = (ref_date - pd.Timedelta(days=max_window + 5)).strftime("%Y-%m-%d")
    date_max = ref_date.strftime("%Y-%m-%d")

    dates_arr, daily_counts, articles_per_day, total = load_daily_aggregation_combined(
        processed_path, daily_dir, date_min, date_max
    )

    # 윈도우별 빈도
    unique_windows = set(optimal_windows.values())
    window_freqs: dict[int, dict] = {}
    window_articles: dict[int, int] = {}
    for w in unique_windows:
        freq_map, n_art = compute_window_frequency(
            dates_arr, daily_counts, articles_per_day, ref_date, w
        )
        window_freqs[w] = freq_map
        window_articles[w] = n_art

    combined_freq: dict[str, int] = {}
    for prop in sig_props:
        w = optimal_windows.get(prop, default_window)
        combined_freq[prop] = window_freqs.get(w, {}).get(prop, 0)

    n_articles = window_articles.get(max(unique_windows), 0)
    result = model_b.predict(combined_freq, n_articles)
    risk_score = result["risk_score"]
    details = result["details"]

    # 상태 판정
    status = "WARNING" if risk_score >= THRESHOLD else "NORMAL"
    gap = risk_score - THRESHOLD if risk_score >= THRESHOLD else THRESHOLD - risk_score
    gap_str = f"+{gap:.2f}" if risk_score >= THRESHOLD else f"-{gap:.2f}"

    # 기여도 정렬
    sorted_props = sorted(
        details.keys(), key=lambda x: details[x]["contribution"], reverse=True
    )
    top3 = []
    for pid in sorted_props[:3]:
        d = details[pid]
        if d["ratio"] > 0:
            top3.append(f"{pid}({d['ratio']:.2f}x)")

    # 출력
    date_str = ref_date.strftime("%Y-%m-%d")
    print()
    print(
        f"[{date_str}] Risk Score: {risk_score:.2f}/100 | "
        f"Threshold: {THRESHOLD} | STATUS: {status} ({gap_str})"
    )
    print(f"  Top 3: {', '.join(top3)}")
    print()

    print(
        f"  {'ID':<14s} {'Name':<26s} {'Win':>4s} {'Freq':>7s} "
        f"{'Ctrl':>7s} {'Ratio':>6s} {'Contrib':>7s}"
    )
    print("  " + "-" * 76)
    for pid in sorted_props:
        d = details[pid]
        name = PROP_NAMES.get(pid, pid)[:24]
        w = optimal_windows.get(pid, default_window)
        arrow = " !!" if d["ratio"] > 1.2 else (" +" if d["ratio"] > 1.0 else "")
        print(
            f"  {pid:<14s} {name:<26s} {w:>3d}d {d['freq']:>7,d} "
            f"{d['ctrl_mean']:>7,.0f} {d['ratio']:>6.3f} {d['contribution']:>7.4f}{arrow}"
        )
    print()

    # CSV 저장
    details_json = json.dumps(
        {pid: {"ratio": details[pid]["ratio"], "contribution": details[pid]["contribution"]}
         for pid in sorted_props},
        ensure_ascii=False,
    )

    row = {
        "date": date_str,
        "risk_score": round(risk_score, 2),
        "raw_score": round(result["raw_score"], 4),
        "threshold": THRESHOLD,
        "status": status,
        "n_articles": n_articles,
        "top_contributor": sorted_props[0] if sorted_props else "",
        "details_json": details_json,
    }

    csv_path = outputs_path / "daily_scores.csv"
    new_row = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["date"] != date_str]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(csv_path, index=False)
    logger.info(f"저장: {csv_path}")

    return row


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def run(target_date: str = None, score_only: bool = False):
    config = load_config()
    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")
    download_dir = Path("data/raw/bigkinds")
    daily_dir = processed_path / "corpus_daily"

    # 기준일
    if target_date:
        date_str = target_date
    else:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    ref_date = pd.Timestamp(date_str)
    logger.info(f"=== Daily Pipeline: {date_str} ===")

    if not score_only:
        # Step 1: 뉴스 수집
        logger.info("--- Step 1: 뉴스 수집 ---")
        xlsx_path = collect_news(date_str, download_dir)

        # Step 2: 태깅
        if xlsx_path is not None:
            logger.info("--- Step 2: 임베딩 + 태깅 ---")
            process_and_tag(xlsx_path, daily_dir, config)
        else:
            logger.info("새 뉴스 없음, 태깅 스킵")

    # Step 3: 스코어링
    logger.info("--- Step 3: 스코어링 ---")

    # 스코어링 기준일: daily 파일 중 가장 최근 날짜, 또는 코퍼스 최신
    if daily_dir.exists() and list(daily_dir.glob("*.parquet")):
        latest_daily = sorted(daily_dir.glob("*.parquet"))[-1]
        latest_date_str = latest_daily.stem
        scoring_date = pd.Timestamp(latest_date_str)
    else:
        # daily 파일 없으면 코퍼스 최신일 사용
        pf = pq.ParquetFile(processed_path / "corpus_tagged.parquet")
        last_rg = pf.read_row_group(pf.metadata.num_row_groups - 1, columns=["date"])
        last_df = last_rg.to_pandas()
        last_df["date"] = pd.to_datetime(last_df["date"])
        scoring_date = last_df["date"].max()

    logger.info(f"스코어링 기준일: {scoring_date.strftime('%Y-%m-%d')}")
    score(processed_path, outputs_path, daily_dir, scoring_date, config)

    logger.info("=== Daily Pipeline 완료 ===")


def main():
    parser = argparse.ArgumentParser(description="Daily Pipeline")
    parser.add_argument("--date", type=str, default=None,
                        help="수집할 날짜 (YYYY-MM-DD). 미지정 시 어제")
    parser.add_argument("--score-only", action="store_true",
                        help="수집/태깅 스킵, 스코어링만 실행")
    args = parser.parse_args()
    run(target_date=args.date, score_only=args.score_only)


if __name__ == "__main__":
    main()
