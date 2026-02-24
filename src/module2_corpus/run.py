"""
Module 2 — News Corpus
BlackEvent 및 대조군 날짜 기준으로 직전 뉴스 기사를 매칭한다.
빅카인즈(bigkinds.or.kr)에서 수동 다운로드한 엑셀/CSV 파일을 로드하여 사용.
"""

import hashlib
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import load_config, get_path, setup_logging

logger = setup_logging("module2")


# ──────────────────────────────────────────────
# 빅카인즈 로컬 파일 로더
# ──────────────────────────────────────────────
def load_bigkinds_files(bigkinds_dir: Path) -> pd.DataFrame:
    """
    data/raw/bigkinds/ 폴더의 .xlsx / .csv 파일을 모두 읽어 하나의 DataFrame으로 합친다.
    빅카인즈 엑셀 컬럼명을 내부 스키마로 매핑한다.
    """
    files = list(bigkinds_dir.glob("*.xlsx")) + list(bigkinds_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"빅카인즈 파일이 없습니다: {bigkinds_dir}\n"
            "bigkinds.or.kr에서 뉴스를 검색·다운로드한 뒤 이 폴더에 넣어주세요."
        )

    dfs = []
    for f in files:
        logger.info(f"파일 로드: {f.name}")
        if f.suffix == ".xlsx":
            df = pd.read_excel(f, engine="openpyxl")
        else:
            # csv — 인코딩 자동 탐색
            for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
                try:
                    df = pd.read_csv(f, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                logger.warning(f"인코딩을 판별할 수 없어 건너뜀: {f.name}")
                continue
        dfs.append(df)

    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"로드 완료: {len(files)}개 파일, 총 {len(raw_df)}건")
    return raw_df


def map_bigkinds_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    빅카인즈 엑셀 컬럼 → 내부 스키마 매핑
      일자       → date  (YYYYMMDD → YYYY-MM-DD)
      제목       → title
      본문       → lead  (첫 200자에서 리드문 1~2문장 추출)
      언론사     → source
      뉴스 식별자 → news_id
    """
    # 컬럼명 확인
    col_map = {
        "일자": "date",
        "제목": "title",
        "본문": "body",
        "언론사": "source",
        "뉴스 식별자": "news_id",
    }

    missing = [k for k in col_map if k not in raw_df.columns]
    if missing:
        logger.warning(f"빅카인즈 파일에 없는 컬럼: {missing}")
        # 유사 컬럼명 탐색 (빅카인즈 버전별 차이 대응)
        for orig in list(missing):
            for col in raw_df.columns:
                if orig.replace(" ", "") in col.replace(" ", ""):
                    col_map[col] = col_map.pop(orig)
                    missing.remove(orig)
                    logger.info(f"  '{orig}' → '{col}' 자동 매핑")
                    break

    df = raw_df.rename(columns=col_map).copy()

    # 날짜 변환: YYYYMMDD 또는 YYYY-MM-DD 등 다양한 형식 대응
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="mixed", dayfirst=False)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # 리드문 추출: 본문 첫 200자에서 1~2문장
    df["lead"] = df.get("body", pd.Series([""] * len(df), dtype=str)).fillna("").apply(extract_lead)

    # news_id 중복 제거
    if "news_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="news_id", keep="first")
        dupes = before - len(df)
        if dupes:
            logger.info(f"뉴스 식별자 기준 중복 제거: {dupes}건")

    # 필요 컬럼만 유지
    keep_cols = ["date", "title", "lead", "source"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[keep_cols].copy()

    # 빈 제목 제거
    df = df[df["title"].fillna("").str.strip() != ""].reset_index(drop=True)

    logger.info(f"컬럼 매핑 완료: {len(df)}건")
    return df


def extract_lead(body: str, max_chars: int = 200, max_sentences: int = 2) -> str:
    """본문에서 리드문(첫 200자 범위 내 1~2문장)을 추출한다."""
    if not body or not isinstance(body, str):
        return ""
    snippet = body[:max_chars].strip()
    sentences = re.split(r"(?<=[.!?。다])\s+", snippet)
    lead = " ".join(sentences[:max_sentences])
    return lead.strip()


# ──────────────────────────────────────────────
# 전처리 함수
# ──────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """텍스트 정규화: 특수문자, 공백 정리"""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s가-힣.,!?·\-()\"']", "", text)
    return text.strip()


def deduplicate(articles: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    제목 유사도 기반 중복 제거.
    제목의 해시를 기준으로 정확히 같은 제목 제거 +
    제목의 처음 20자 기준 유사 기사 제거.
    """
    seen_hashes = set()
    seen_prefixes = {}
    unique = []

    for article in articles:
        title = article["title"]
        title_hash = hashlib.md5(title.encode()).hexdigest()

        if title_hash in seen_hashes:
            continue
        seen_hashes.add(title_hash)

        # 제목 앞부분 기반 유사도 간이 체크
        prefix = title[:20]
        if prefix in seen_prefixes:
            continue
        seen_prefixes[prefix] = True

        unique.append(article)

    return unique


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 단위 텍스트 정규화 및 빈 리드문 처리"""
    df = df.copy()
    df["title"] = df["title"].fillna("").apply(normalize_text)
    df["lead"] = df["lead"].fillna("").apply(normalize_text)

    # 빈 리드문이면 제목으로 대체
    mask = df["lead"].str.strip() == ""
    df.loc[mask, "lead"] = df.loc[mask, "title"]

    # 빈 제목 제거
    df = df[df["title"].str.strip() != ""].reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# 날짜 기준 매칭
# ──────────────────────────────────────────────
def match_articles_to_dates(
    news_df: pd.DataFrame,
    dates: list,
    event_ids: list,
    event_type: str,
    window_days: int,
) -> list[dict]:
    """
    각 기준일 직전 window_days일 범위의 기사를 필터링하여 event_id를 부착한다.
    """
    news_df = news_df.copy()
    news_df["date_dt"] = pd.to_datetime(news_df["date"])

    all_articles = []

    for ref_date, event_id in tqdm(
        zip(dates, event_ids),
        total=len(dates),
        desc=f"뉴스 매칭 ({event_type})",
    ):
        ref_dt = pd.to_datetime(ref_date)
        start_dt = ref_dt - pd.Timedelta(days=window_days)
        end_dt = ref_dt - pd.Timedelta(days=1)

        mask = (news_df["date_dt"] >= start_dt) & (news_df["date_dt"] <= end_dt)
        matched = news_df[mask]

        for _, row in matched.iterrows():
            all_articles.append({
                "date": row["date"],
                "title": row["title"],
                "lead": row["lead"],
                "source": row["source"],
                "event_id": f"{event_type}_{event_id}",
            })

    return all_articles


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def run():
    """Module 2 전체 파이프라인 실행"""
    config = load_config()

    raw_path = get_path(config, "raw_data")
    processed_path = get_path(config, "processed_data")
    bigkinds_dir = raw_path / "bigkinds"

    # 1. 빅카인즈 로컬 파일 로드
    logger.info("=== 빅카인즈 파일 로드 ===")
    raw_df = load_bigkinds_files(bigkinds_dir)
    news_df = map_bigkinds_columns(raw_df)
    news_df = preprocess_df(news_df)
    logger.info(f"전처리 후: {len(news_df)}건")

    # 2. BlackEvent & 대조군 날짜 로드
    events_df = pd.read_csv(processed_path / "black_events.csv", parse_dates=["first_shock_date"])
    control_df = pd.read_csv(processed_path / "control_dates.csv", parse_dates=["date"])

    corpus_config = config["corpus"]
    window_days = corpus_config["news_window_days"]

    # 3. BlackEvent 직전 뉴스 매칭
    logger.info("=== BlackEvent 직전 뉴스 매칭 ===")
    be_articles = match_articles_to_dates(
        news_df,
        dates=events_df["first_shock_date"].tolist(),
        event_ids=events_df["event_id"].tolist(),
        event_type="blackevent",
        window_days=window_days,
    )
    logger.info(f"BlackEvent 매칭: {len(be_articles)}건")

    # 4. 대조군 직전 뉴스 매칭
    logger.info("=== 대조군 직전 뉴스 매칭 ===")
    ctrl_articles = match_articles_to_dates(
        news_df,
        dates=control_df["date"].tolist(),
        event_ids=control_df["control_id"].tolist(),
        event_type="control",
        window_days=window_days,
    )
    logger.info(f"대조군 매칭: {len(ctrl_articles)}건")

    # 5. 합치기 + 중복 제거
    all_articles = be_articles + ctrl_articles
    all_articles = deduplicate(all_articles, threshold=corpus_config["dedup_similarity_threshold"])

    # 6. article_id 부여
    for i, article in enumerate(all_articles, start=1):
        article["article_id"] = i

    # 7. DataFrame 변환 및 저장
    corpus_df = pd.DataFrame(all_articles)
    corpus_df = corpus_df[["article_id", "date", "title", "lead", "event_id", "source"]]

    processed_path.mkdir(parents=True, exist_ok=True)
    corpus_df.to_parquet(processed_path / "news_corpus.parquet", index=False)
    logger.info(f"뉴스 코퍼스 저장: {len(corpus_df)}건 → {processed_path / 'news_corpus.parquet'}")
    logger.info("=== Module 2 완료 ===")

    return corpus_df


if __name__ == "__main__":
    run()
