"""
Module 3 — Property Dictionary Generation
기사의 '구조적 성질'을 비지도로 추출하여 성질 사전(Property Dictionary)을 생성한다.

Anti-Bias Rule (절대 준수):
1. LLM에게 기사 원문만 제공. 예시 태그 절대 금지.
2. '다음 중 선택하시오' 형태 절대 금지.
3. 프로젝트 목적(주가 예측) 비공개.
4. 이전 기사 태깅 결과를 컨텍스트에 포함 금지.
5. 자유 서술 → 후처리로 구조화.
"""

import gc
import json
import os
import time
from collections import defaultdict

import anthropic
import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as l2_normalize
from tqdm import tqdm

from src.utils import load_config, load_env, get_path, setup_logging

logger = setup_logging("module3")

# ──────────────────────────────────────────────
# Anti-Bias 준수 프롬프트 (예시 없음, 목적 비공개)
# ──────────────────────────────────────────────
OPEN_TAGGING_PROMPT = """다음은 뉴스 기사의 제목과 첫 문장입니다.

제목: {title}
본문: {lead}

이 기사가 묘사하는 사건 또는 현상의 구조적 특성을 3~5개의 짧은 구(phrase)로 서술하세요.
- 사건의 "주제"가 아니라 "사건이 가진 성질"을 서술하세요
- 예시를 참고하지 마세요. 자유롭게 서술하세요

각 구를 줄바꿈으로 구분하여 출력하세요."""


# ──────────────────────────────────────────────
# Step 3-1: 샘플 오픈형 태깅
# ──────────────────────────────────────────────
def sample_articles(
    corpus_df: pd.DataFrame,
    n: int = 5000,
    seed: int = 42,
    event_type_col: str = "event_type",
) -> pd.DataFrame:
    """
    전체 코퍼스에서 n건 층화 샘플링(Stratified Sampling).

    축 1 — 시기별 균등: 설정의 periods 구간별로 동일 수량 배분
    축 2 — BlackEvent 직전 / 평상시 균등: 각 시기 내 50:50

    특정 구간에서 해당 유형 기사가 부족하면 인접 구간에서 보충하되
    BlackEvent/평상시 비율 50:50은 유지.
    """
    config = load_config()
    be_prefix = config.get("event_id_prefix", {}).get("blackevent", "blackevent")
    strat_config = config.get("dictionary", {}).get("stratified_sampling", {})

    # 기본 시기 구간
    default_periods = [
        ("2006-01-01", "2009-12-31"),
        ("2010-01-01", "2013-12-31"),
        ("2014-01-01", "2017-12-31"),
        ("2018-01-01", "2021-12-31"),
        ("2022-01-01", None),
    ]
    periods = []
    for p in strat_config.get("periods", default_periods):
        start = p[0]
        end = p[1] if p[1] is not None else pd.Timestamp.now().strftime("%Y-%m-%d")
        periods.append((start, end))

    n_periods = len(periods)
    per_period = n // n_periods           # 구간당 총 건수
    half_per_period = per_period // 2     # 구간당 BlackEvent / 평상시 각 건수

    # BlackEvent / 대조군 분류
    if event_type_col in corpus_df.columns:
        be_mask = corpus_df[event_type_col] == "blackevent"
    else:
        be_mask = corpus_df["event_id"].str.startswith(be_prefix)

    # date 컬럼을 datetime으로 변환 (비교용)
    dates = pd.to_datetime(corpus_df["date"])

    # 구간별 기사 인덱스 분류
    rng = np.random.RandomState(seed)
    period_be_pools = []    # 구간별 BlackEvent 기사 인덱스
    period_ctrl_pools = []  # 구간별 대조군 기사 인덱스

    for start, end in periods:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        in_period = (dates >= start_dt) & (dates <= end_dt)

        be_idx = corpus_df.index[in_period & be_mask].tolist()
        ctrl_idx = corpus_df.index[in_period & ~be_mask].tolist()
        period_be_pools.append(be_idx)
        period_ctrl_pools.append(ctrl_idx)

        logger.info(
            f"구간 {start}~{end}: "
            f"BlackEvent {len(be_idx)}건, 대조군 {len(ctrl_idx)}건"
        )

    # 1차 샘플링: 각 구간에서 가능한 만큼 추출
    sampled_be = []       # 구간별 실제 샘플 인덱스
    sampled_ctrl = []
    be_shortage = []      # (구간 인덱스, 부족 건수)
    ctrl_shortage = []

    for i in range(n_periods):
        # BlackEvent
        available_be = len(period_be_pools[i])
        take_be = min(half_per_period, available_be)
        if take_be > 0:
            chosen = rng.choice(period_be_pools[i], size=take_be, replace=False).tolist()
            sampled_be.extend(chosen)
        if take_be < half_per_period:
            be_shortage.append((i, half_per_period - take_be))

        # 대조군
        available_ctrl = len(period_ctrl_pools[i])
        take_ctrl = min(half_per_period, available_ctrl)
        if take_ctrl > 0:
            chosen = rng.choice(period_ctrl_pools[i], size=take_ctrl, replace=False).tolist()
            sampled_ctrl.extend(chosen)
        if take_ctrl < half_per_period:
            ctrl_shortage.append((i, half_per_period - take_ctrl))

    # 2차 보충: 부족분을 인접 구간에서 보충
    already_sampled = set(sampled_be + sampled_ctrl)

    def fill_shortage(shortage_list, pools, already_used):
        """인접 구간에서 부족분 보충. 가까운 구간부터 탐색."""
        extra = []
        for orig_idx, deficit in shortage_list:
            filled = 0
            # 인접 구간 순서: 거리 1, 2, ... 순으로 양쪽 탐색
            for dist in range(1, n_periods):
                for neighbor_idx in [orig_idx - dist, orig_idx + dist]:
                    if neighbor_idx < 0 or neighbor_idx >= n_periods:
                        continue
                    pool = [idx for idx in pools[neighbor_idx] if idx not in already_used]
                    if not pool:
                        continue
                    take = min(deficit - filled, len(pool))
                    chosen = rng.choice(pool, size=take, replace=False).tolist()
                    extra.extend(chosen)
                    already_used.update(chosen)
                    filled += take
                    if filled >= deficit:
                        break
                if filled >= deficit:
                    break
            if filled < deficit:
                logger.warning(
                    f"구간 {orig_idx} 부족분 보충 미완: "
                    f"필요 {deficit}, 확보 {filled}"
                )
        return extra

    extra_be = fill_shortage(be_shortage, period_be_pools, already_sampled)
    extra_ctrl = fill_shortage(ctrl_shortage, period_ctrl_pools, already_sampled)

    all_indices = sampled_be + extra_be + sampled_ctrl + extra_ctrl
    sample = corpus_df.loc[all_indices].reset_index(drop=True)

    n_be = len(sampled_be) + len(extra_be)
    n_ctrl = len(sampled_ctrl) + len(extra_ctrl)
    logger.info(
        f"층화 샘플링 완료: {len(sample)}건 "
        f"(BlackEvent: {n_be}, 대조군: {n_ctrl}, "
        f"구간 수: {n_periods}, 구간당 목표: {per_period}건)"
    )
    return sample


def tag_article_open(
    client: anthropic.Anthropic,
    title: str,
    lead: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 512,
) -> list[str]:
    """
    Claude API로 기사의 구조적 성질을 자유 서술 태깅한다.
    Anti-Bias Rule 준수: 예시 없음, 목적 비공개, 이전 결과 미포함.
    """
    prompt = OPEN_TAGGING_PROMPT.format(title=title, lead=lead)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # 줄바꿈 기준으로 태그 분리, 번호/불릿 제거
    tags = []
    for line in text.split("\n"):
        line = line.strip()
        line = line.lstrip("0123456789.-·•) ")
        if line:
            tags.append(line)
    return tags[:5]  # 최대 5개


def _smart_sleep(rpm_limit: int = 50):
    """분당 요청 수 기반 throttle (초당 요청 간격)"""
    time.sleep(60.0 / rpm_limit)


def batch_open_tagging(
    sample_df: pd.DataFrame,
    client: anthropic.Anthropic,
    model: str,
    temperature: float = 0,
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
    checkpoint_path=None,
    checkpoint_interval: int = 100,
    rpm_limit: int = 50,
) -> pd.DataFrame:
    """
    샘플 기사를 건건이 동기 호출로 오픈형 태깅한다.
    - 에러 유형별 retry (RateLimitError → Retry-After, 기타 → 지수 백오프)
    - checkpoint_interval마다 중간 저장, 재실행 시 이어서 태깅
    - rpm_limit: 분당 요청 수 제한 (config에서 설정)
    """
    # 체크포인트에서 이전 결과 로드
    start_idx = 0
    results = []
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        logger.info(f"체크포인트에서 재개: {start_idx}/{len(sample_df)}건 완료")

    for i in tqdm(range(start_idx, len(sample_df)), initial=start_idx,
                  total=len(sample_df), desc="오픈형 태깅"):
        row = sample_df.iloc[i]
        tags = []

        for attempt in range(retry_attempts):
            try:
                tags = tag_article_open(client, row["title"], row["lead"], model, temperature)
                break
            except anthropic.RateLimitError as e:
                # Retry-After 헤더 존재 시 해당 시간만큼 대기
                retry_after = getattr(e, "response", None)
                if retry_after is not None:
                    wait = int(retry_after.headers.get("retry-after", 30))
                else:
                    wait = 30
                logger.warning(f"Rate limit — {wait}초 대기 (attempt {attempt + 1})")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                # 서버 에러 등 → 지수 백오프
                wait = retry_delay * (2 ** attempt)
                logger.warning(f"API 오류({e.status_code}) — {wait:.1f}초 대기 (attempt {attempt + 1})")
                time.sleep(wait)
            except Exception as e:
                wait = retry_delay * (2 ** attempt)
                logger.error(f"태깅 오류: {e} — {wait:.1f}초 대기 (attempt {attempt + 1})")
                time.sleep(wait)

        results.append({
            "article_id": row["article_id"],
            "title": row["title"],
            "lead": row["lead"],
            "event_id": row["event_id"],
            "tags": tags,
        })

        # 분당 요청 수 기반 throttle
        _smart_sleep(rpm_limit=rpm_limit)

        # 체크포인트 저장
        if checkpoint_path and (i + 1) % checkpoint_interval == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            logger.info(f"체크포인트 저장: {i + 1}/{len(sample_df)}")

    # 최종 체크포인트 저장
    if checkpoint_path:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    df = pd.DataFrame(results)
    total_tags = sum(len(t) for t in df["tags"])
    logger.info(f"오픈형 태깅 완료: {len(df)}건, 총 {total_tags}개 태그")
    return df


def _parse_tagging_response(text: str) -> list[str]:
    """LLM 응답 텍스트에서 태그를 파싱한다."""
    tags = []
    for line in text.split("\n"):
        line = line.strip()
        line = line.lstrip("0123456789.-·•) ")
        if line:
            tags.append(line)
    return tags[:5]


def batch_open_tagging_batch_api(
    sample_df: pd.DataFrame,
    client: anthropic.Anthropic,
    model: str,
    temperature: float = 0,
    max_tokens: int = 512,
    poll_interval: int = 30,
) -> pd.DataFrame:
    """
    Anthropic Batch API를 사용하여 샘플 기사를 일괄 태깅한다.
    - 50% 비용 할인
    - throttle 불필요 (API가 자체 관리)
    - 완료까지 폴링하여 대기
    """
    # 전체 요청 구성
    requests = []
    for i in range(len(sample_df)):
        row = sample_df.iloc[i]
        prompt = OPEN_TAGGING_PROMPT.format(title=row["title"], lead=row["lead"])
        requests.append({
            "custom_id": str(i),
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    logger.info(f"Batch API 요청 제출: {len(requests)}건")
    batch = client.messages.batches.create(requests=requests)
    logger.info(f"Batch ID: {batch.id}")

    # 완료 폴링
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        status = batch.processing_status

        # 진행률 로그
        counts = batch.request_counts
        logger.info(
            f"Batch 상태: {status} "
            f"(성공: {counts.succeeded}, 오류: {counts.errored}, "
            f"처리중: {counts.processing}, 취소: {counts.canceled})"
        )

        if status == "ended":
            break
        time.sleep(poll_interval)

    # 결과 수신 및 파싱
    results = [None] * len(sample_df)
    n_errors = 0

    for result in client.messages.batches.results(batch.id):
        idx = int(result.custom_id)
        row = sample_df.iloc[idx]

        if result.result.type == "succeeded":
            text = result.result.message.content[0].text.strip()
            tags = _parse_tagging_response(text)
        else:
            logger.warning(f"Batch 항목 {idx} 오류: {result.result.type}")
            tags = []
            n_errors += 1

        results[idx] = {
            "article_id": row["article_id"],
            "title": row["title"],
            "lead": row["lead"],
            "event_id": row["event_id"],
            "tags": tags,
        }

    # 누락된 항목 처리 (안전장치)
    for idx in range(len(results)):
        if results[idx] is None:
            row = sample_df.iloc[idx]
            results[idx] = {
                "article_id": row["article_id"],
                "title": row["title"],
                "lead": row["lead"],
                "event_id": row["event_id"],
                "tags": [],
            }
            n_errors += 1

    df = pd.DataFrame(results)
    total_tags = sum(len(t) for t in df["tags"])
    logger.info(
        f"Batch API 태깅 완료: {len(df)}건, 총 {total_tags}개 태그"
        f" (오류: {n_errors}건)"
    )
    return df


# ──────────────────────────────────────────────
# 태그-기사 매핑 관리
# ──────────────────────────────────────────────
def build_tag_article_map(tagged_df: pd.DataFrame) -> tuple[list[str], list, dict]:
    """
    태깅 결과에서 (tag_text, article_id) 관계를 구축한다.
    Returns:
        all_tags:      태그 텍스트 리스트 (임베딩 입력용, 중복 포함)
        tag_to_article: all_tags[i]가 속한 article_id
        article_to_tags: article_id → {tag_index, ...}
    """
    all_tags = []
    tag_to_article = []

    for _, row in tagged_df.iterrows():
        article_id = row["article_id"]
        for tag in row["tags"]:
            all_tags.append(tag)
            tag_to_article.append(article_id)

    # article_id → tag indices (역방향)
    article_to_tags: dict = defaultdict(set)
    for idx, aid in enumerate(tag_to_article):
        article_to_tags[aid].add(idx)

    logger.info(f"태그-기사 매핑: {len(all_tags)}개 태그, {len(article_to_tags)}개 기사")
    return all_tags, tag_to_article, dict(article_to_tags)


# ──────────────────────────────────────────────
# Step 3-2: 태그 클러스터링 → 성질 사전 생성
# ──────────────────────────────────────────────
def embed_tags(tags: list[str], model_name: str) -> np.ndarray:
    """
    태그를 임베딩 벡터로 변환한다.
    모델을 로드 → 인코딩 → 명시적 해제하여 메모리를 관리한다.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"임베딩 모델 로드: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(tags, show_progress_bar=True, batch_size=64)
    logger.info(f"임베딩 완료: {len(tags)}개 태그 → {embeddings.shape}")

    # 메모리 해제
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU 메모리 해제 완료")
    except ImportError:
        pass

    return embeddings


def cluster_tags(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> np.ndarray:
    """
    HDBSCAN으로 태그를 클러스터링한다.
    sentence embedding은 cosine similarity 기준으로 훈련되므로
    L2 normalize 후 euclidean distance를 사용한다 (= cosine distance와 동치).
    """
    # L2 정규화 → euclidean 거리 = cosine 거리와 단조 관계
    normed = l2_normalize(embeddings, norm="l2")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(normed)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"클러스터링 완료: {n_clusters}개 클러스터, {n_noise}개 노이즈")
    return labels


def extract_representatives(
    tags: list[str],
    tag_to_article: list[int],
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    """
    각 클러스터의 대표 태그(centroid에 가장 가까운 태그)를 추출한다.
    태그-기사 매핑을 이용해 클러스터가 커버하는 고유 기사 수를 계산한다.
    """
    clusters = []
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue

        member_indices = [i for i in range(len(tags)) if labels[i] == label]
        cluster_embeddings = embeddings[member_indices]
        cluster_tags = [tags[i] for i in member_indices]

        # 커버하는 고유 기사 수
        covered_articles = set(tag_to_article[i] for i in member_indices)

        # centroid 기반 대표 태그 선정
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, cluster_embeddings)[0]
        best_idx = sims.argmax()

        # centroid 유사도 기준 정렬 → 상위 10개를 cluster_keywords로 선택
        sorted_indices = sims.argsort()[::-1]
        sorted_tags = [cluster_tags[i] for i in sorted_indices]

        clusters.append({
            "cluster_id": int(label),
            "representative_tag": cluster_tags[best_idx],
            "cluster_keywords": sorted_tags[:10],
            "cluster_size": len(member_indices),       # 태그 수
            "article_count": len(covered_articles),    # 고유 기사 수
            "member_indices": member_indices,           # 전체 태그 인덱스 (교차검증용)
            "covered_articles": covered_articles,       # 커버 기사 ID 집합
        })

    return clusters


def cross_validate_clusters(
    tags: list[str],
    tag_to_article: list[int],
    embeddings_main: np.ndarray,
    embeddings_cv: np.ndarray,
    labels_main: np.ndarray,
    labels_cv: np.ndarray,
) -> list[dict]:
    """
    두 임베딩 모델의 클러스터링 결과를 교차 검증.
    메인 클러스터의 **전체 멤버 태그**가 CV에서 동일 라벨로 묶이는 비율을 계산.
    """
    main_clusters = extract_representatives(tags, tag_to_article, embeddings_main, labels_main)

    for cluster in main_clusters:
        member_indices = cluster["member_indices"]

        # 메인 클러스터 멤버들의 CV 라벨 수집
        cv_labels_for_members = [labels_cv[i] for i in member_indices]

        # 노이즈(-1) 제외한 유효 CV 라벨
        valid_cv_labels = [l for l in cv_labels_for_members if l != -1]

        if not valid_cv_labels:
            cluster["cross_validation_confidence"] = 0.0
            continue

        # CV에서 가장 빈번한 라벨 = 대응 클러스터
        from collections import Counter
        label_counts = Counter(valid_cv_labels)
        dominant_label, dominant_count = label_counts.most_common(1)[0]

        # 전체 멤버 중 dominant CV 라벨에 속하는 비율
        overlap_ratio = dominant_count / len(member_indices)
        cluster["cross_validation_confidence"] = round(overlap_ratio, 3)

    # member_indices는 내부 처리용이므로 정리하지 않고 유지 (filter에서 사용)
    return main_clusters


# ──────────────────────────────────────────────
# Step 3-3: 성질 사전 정제
# ──────────────────────────────────────────────
def filter_clusters(
    clusters: list[dict],
    total_articles: int,
    min_articles: int = 10,
    max_coverage: float = 0.8,
    min_cv_confidence: float = 0.3,
) -> list[dict]:
    """
    너무 작거나 너무 일반적인 클러스터를 제거한다.
    기준: 클러스터가 커버하는 **고유 기사 수** (태그 수가 아님).
    - min_articles 미만 제거
    - 전체 기사의 max_coverage 초과 제거
    - cross_validation_confidence가 min_cv_confidence 미만 제거
    """
    max_article_count = int(total_articles * max_coverage)
    filtered = []
    removed_size = 0
    removed_coverage = 0
    removed_cv = 0

    for cluster in clusters:
        ac = cluster["article_count"]
        cv = cluster.get("cross_validation_confidence", 0.0)

        if ac < min_articles:
            removed_size += 1
            continue
        if ac > max_article_count:
            removed_coverage += 1
            continue
        if cv < min_cv_confidence:
            removed_cv += 1
            continue
        filtered.append(cluster)

    removed = len(clusters) - len(filtered)
    logger.info(
        f"클러스터 정제: {len(clusters)} → {len(filtered)} ({removed}개 제거) "
        f"[기사 수 부족: {removed_size}, 커버리지 초과: {removed_coverage}, "
        f"CV confidence 미달(<{min_cv_confidence}): {removed_cv}]"
    )
    return filtered


def build_property_dictionary(clusters: list[dict]) -> list[dict]:
    """최종 성질 사전을 구성한다. 내부 처리용 필드는 제거."""
    dictionary = []
    for i, cluster in enumerate(clusters, start=1):
        dictionary.append({
            "property_id": f"Property_{i:03d}",
            "representative_tag": cluster["representative_tag"],
            "cluster_keywords": cluster["cluster_keywords"],
            "cluster_size": cluster["cluster_size"],
            "article_count": cluster["article_count"],
            "cross_validation_confidence": cluster.get("cross_validation_confidence", 0.0),
        })
    return dictionary


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def run():
    """Module 3 전체 파이프라인 실행"""
    config = load_config()
    load_env()

    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")
    outputs_path.mkdir(parents=True, exist_ok=True)

    dict_config = config["dictionary"]
    llm_config = config["llm"]

    # 뉴스 코퍼스 로드
    corpus_df = pd.read_parquet(processed_path / "news_corpus.parquet")
    logger.info(f"뉴스 코퍼스 로드: {len(corpus_df)}건")

    # ── Step 3-1: 샘플 오픈형 태깅 ──
    sample_df = sample_articles(corpus_df, n=dict_config["sample_size"])

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    use_batch_api = llm_config.get("use_batch_api", False)

    if use_batch_api:
        logger.info("Batch API 모드로 태깅 시작 (50% 비용 할인)")
        tagged_df = batch_open_tagging_batch_api(
            sample_df,
            client=client,
            model=llm_config["model"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config.get("max_tokens", 512),
            poll_interval=llm_config.get("batch_poll_interval", 30),
        )
    else:
        logger.info("건건이 호출 모드로 태깅 시작")
        checkpoint_path = outputs_path / "sample_tagging_checkpoint.json"
        tagged_df = batch_open_tagging(
            sample_df,
            client=client,
            model=llm_config["model"],
            temperature=llm_config["temperature"],
            retry_attempts=llm_config["retry_attempts"],
            retry_delay=llm_config["retry_delay"],
            checkpoint_path=checkpoint_path,
            checkpoint_interval=100,
            rpm_limit=llm_config.get("rpm_limit", 50),
        )

    # raw 태깅 결과 저장
    save_df = tagged_df.copy()
    save_df["tags"] = save_df["tags"].apply(json.dumps, ensure_ascii=False)
    save_df.to_csv(outputs_path / "sample_tagging_raw.csv", index=False, encoding="utf-8-sig")
    logger.info(f"Raw 태깅 저장: {outputs_path / 'sample_tagging_raw.csv'}")

    # 체크포인트 정리 (완료 후, 건건이 호출 모드에서만)
    if not use_batch_api:
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    # ── 태그-기사 매핑 구축 ──
    all_tags, tag_to_article, article_to_tags = build_tag_article_map(tagged_df)

    if not all_tags:
        logger.error("태그가 없습니다. 태깅 결과를 확인하세요.")
        return None

    # ── Step 3-2: 태그 클러스터링 ──
    emb_config = dict_config["embedding_models"]
    hdb_config = dict_config["hdbscan"]

    # 메인 임베딩: KURE-v1 (완료 후 메모리 해제됨)
    embeddings_main = embed_tags(all_tags, emb_config["main"])
    labels_main = cluster_tags(
        embeddings_main,
        min_cluster_size=hdb_config["min_cluster_size"],
        min_samples=hdb_config["min_samples"],
    )

    # 교차 검증 임베딩: BGE-M3 (메인 모델 해제 후 로드)
    embeddings_cv = embed_tags(all_tags, emb_config["cross_validation"])
    labels_cv = cluster_tags(
        embeddings_cv,
        min_cluster_size=hdb_config["min_cluster_size"],
        min_samples=hdb_config["min_samples"],
    )

    # 교차 검증 수행
    clusters = cross_validate_clusters(
        all_tags, tag_to_article,
        embeddings_main, embeddings_cv,
        labels_main, labels_cv,
    )

    # 임베딩 메모리 해제
    del embeddings_main, embeddings_cv
    gc.collect()

    # ── Step 3-3: 정제 ──
    filter_config = dict_config["cluster_filter"]
    n_unique_articles = len(set(tag_to_article))
    clusters = filter_clusters(
        clusters,
        total_articles=n_unique_articles,
        min_articles=filter_config["min_articles"],
        max_coverage=filter_config["max_coverage"],
        min_cv_confidence=filter_config.get("min_cv_confidence", 0.3),
    )

    # 성질 사전 구성
    property_dict = build_property_dictionary(clusters)

    # 저장
    with open(outputs_path / "property_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(property_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"성질 사전 저장: {len(property_dict)}개 항목 → {outputs_path / 'property_dictionary.json'}")
    logger.info("=== Module 3 완료 ===")

    return property_dict


if __name__ == "__main__":
    run()
