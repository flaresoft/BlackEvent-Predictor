"""
Module 4 — Full Corpus Tagging (임베딩 유사도 매칭)
성질 사전 기반 임베딩 유사도 매칭으로 전체 기사에 구조적 성질을 태깅한다.
LLM 미사용, 로컬 처리. API 비용 $0.
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.utils import load_config, get_path, setup_logging

logger = setup_logging("module4")


# ──────────────────────────────────────────────
# 성질 벡터 생성
# ──────────────────────────────────────────────
def build_property_vectors(
    property_dict: list[dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    성질 사전의 각 항목을 임베딩하여 성질 벡터를 생성한다.
    representative_tag 하나가 아닌 cluster_keywords 전체의 평균 벡터로 구성하여
    표면적 표현 차이에 의한 의미적 연결 누락을 보완한다.

    Returns: (n_properties, embed_dim) 행렬
    """
    property_vectors = []
    for prop in property_dict:
        # cluster_keywords가 있으면 전체를 임베딩하여 평균, 없으면 representative_tag만 사용
        keywords = prop.get("cluster_keywords", [])
        if not keywords:
            keywords = [prop["representative_tag"]]
        else:
            # representative_tag도 포함하여 평균
            keywords = [prop["representative_tag"]] + keywords

        embeddings = model.encode(keywords, show_progress_bar=False)
        mean_vector = np.mean(embeddings, axis=0)
        # L2 정규화
        mean_vector = mean_vector / np.linalg.norm(mean_vector)
        property_vectors.append(mean_vector)

    return np.array(property_vectors)


# ──────────────────────────────────────────────
# 기사 임베딩 (OOM 자동 복구)
# ──────────────────────────────────────────────
def encode_articles(
    df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int = 256,
    min_batch_size: int = 16,
) -> np.ndarray:
    """
    기사의 제목+리드문을 임베딩한다.
    GPU OOM 발생 시 batch_size를 자동으로 절반씩 줄여 재시도한다.

    Returns: (n_articles, embed_dim) 행렬
    """
    texts = (df["title"].fillna("") + " " + df["lead"].fillna("")).tolist()
    logger.info(f"기사 {len(texts)}건 임베딩 시작 (batch_size={batch_size})")

    current_batch_size = batch_size
    while current_batch_size >= min_batch_size:
        try:
            embeddings = model.encode(
                texts,
                batch_size=current_batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            return embeddings
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                new_batch_size = current_batch_size // 2
                logger.warning(
                    f"GPU OOM 발생 (batch_size={current_batch_size}). "
                    f"batch_size={new_batch_size}로 재시도"
                )
                current_batch_size = new_batch_size
            else:
                raise

    # 최소 batch_size로도 실패한 경우 마지막 시도
    logger.warning(f"최소 batch_size={min_batch_size}로 최종 시도")
    return model.encode(
        texts,
        batch_size=min_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


# ──────────────────────────────────────────────
# LLM 자유 태그 → property_id 매핑
# ──────────────────────────────────────────────
def map_tags_to_properties(
    sample_df: pd.DataFrame,
    property_dict: list[dict],
    property_vectors: np.ndarray,
    model: SentenceTransformer,
    similarity_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    sample_tagging_raw.csv의 자유 서술 태그(tags 컬럼)를
    property_dictionary.json의 property_id에 매핑한다.

    각 LLM 태그를 임베딩하고, 113개 성질 벡터와 코사인 유사도를 계산하여
    가장 유사하고 threshold 이상인 성질에 매핑한다.

    Returns: sample_df에 "properties" 컬럼이 추가된 DataFrame
    """
    property_ids = [p["property_id"] for p in property_dict]

    # 1) 각 기사의 태그 목록 파싱
    all_tags_per_article = []
    for _, row in sample_df.iterrows():
        tags_raw = row.get("tags", "[]")
        if isinstance(tags_raw, str):
            try:
                tags = json.loads(tags_raw)
            except json.JSONDecodeError:
                tags = []
        elif isinstance(tags_raw, list):
            tags = tags_raw
        else:
            tags = []
        all_tags_per_article.append(tags)

    # 2) 전체 태그를 한 번에 임베딩 (배치 효율)
    flat_tags = []
    tag_offsets = [0]
    for tags in all_tags_per_article:
        flat_tags.extend(tags)
        tag_offsets.append(len(flat_tags))

    if not flat_tags:
        sample_df = sample_df.copy()
        sample_df["properties"] = [[] for _ in range(len(sample_df))]
        return sample_df

    logger.info(f"LLM 태그 {len(flat_tags)}건 임베딩 중...")
    all_tag_embeddings = model.encode(
        flat_tags, normalize_embeddings=True, show_progress_bar=False,
    )

    # (n_tags, n_properties) 유사도 행렬
    all_sim = all_tag_embeddings @ property_vectors.T

    # 3) 기사별로 태그 → property_id 매핑
    all_properties = []
    for i, tags in enumerate(all_tags_per_article):
        if not tags:
            all_properties.append([])
            continue

        start = tag_offsets[i]
        end = tag_offsets[i + 1]
        sim_chunk = all_sim[start:end]

        mapped_ids = set()
        for tag_idx in range(len(tags)):
            best_prop_idx = int(np.argmax(sim_chunk[tag_idx]))
            best_sim = sim_chunk[tag_idx, best_prop_idx]
            if best_sim >= similarity_threshold:
                mapped_ids.add(property_ids[best_prop_idx])

        all_properties.append(sorted(mapped_ids))

    sample_df = sample_df.copy()
    sample_df["properties"] = all_properties

    # 매핑 통계 로그
    n_with_props = sum(1 for p in all_properties if len(p) > 0)
    avg_props = np.mean([len(p) for p in all_properties]) if all_properties else 0
    logger.info(
        f"태그→성질 매핑: {n_with_props}/{len(sample_df)}건에 성질 부여 "
        f"(평균 {avg_props:.1f}개/기사)"
    )

    return sample_df


# ──────────────────────────────────────────────
# 임계값 캘리브레이션
# ──────────────────────────────────────────────
def calibrate_threshold(
    sample_df: pd.DataFrame,
    sample_embeddings: np.ndarray,
    property_vectors: np.ndarray,
    property_ids: list[str],
    threshold_range: tuple[float, float, float] = (0.3, 0.8, 0.05),
) -> tuple[float, dict[str, float]]:
    """
    Module 3에서 LLM으로 태깅한 1,000건 샘플을 ground truth로 사용하여
    글로벌 threshold와 성질별 threshold를 탐색한다.

    Returns: (best_global_threshold, per_property_thresholds)
    """
    start, end, step = threshold_range
    thresholds = np.arange(start, end + step / 2, step)

    n_articles = len(sample_df)
    n_properties = len(property_ids)

    # ground truth → 이진 행렬 (n_articles, n_properties)
    gt_matrix = np.zeros((n_articles, n_properties), dtype=int)
    for i, (_, row) in enumerate(sample_df.iterrows()):
        props = row.get("properties", [])
        if isinstance(props, list):
            for p in props:
                if p in property_ids:
                    j = property_ids.index(p)
                    gt_matrix[i, j] = 1

    # 코사인 유사도 행렬 (n_articles, n_properties)
    sim_matrix = sample_embeddings @ property_vectors.T

    # ── 글로벌 threshold 탐색 ──
    best_global_t = 0.5
    best_global_f1 = 0.0

    for t in thresholds:
        pred_matrix = (sim_matrix >= t).astype(int)
        # 전체 성질에 대해 micro-averaged F1
        f1 = f1_score(gt_matrix.ravel(), pred_matrix.ravel(), zero_division=0)
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_t = float(t)

    logger.info(
        f"글로벌 최적 threshold: {best_global_t:.2f} (F1={best_global_f1:.4f})"
    )

    # ── 성질별 threshold 탐색 ──
    per_property_t = {}
    for j, pid in enumerate(property_ids):
        gt_col = gt_matrix[:, j]
        sim_col = sim_matrix[:, j]

        # ground truth에 양성이 없으면 글로벌 threshold 사용
        if gt_col.sum() == 0:
            per_property_t[pid] = best_global_t
            continue

        best_t = best_global_t
        best_f1 = 0.0
        for t in thresholds:
            pred_col = (sim_col >= t).astype(int)
            f1 = f1_score(gt_col, pred_col, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        per_property_t[pid] = best_t

    # 성질별 threshold 통계 로그
    t_values = list(per_property_t.values())
    logger.info(
        f"성질별 threshold 범위: {min(t_values):.2f}~{max(t_values):.2f} "
        f"(평균 {np.mean(t_values):.2f})"
    )

    return best_global_t, per_property_t


# ──────────────────────────────────────────────
# 청크 단위 태깅
# ──────────────────────────────────────────────
def tag_chunk(
    article_embeddings: np.ndarray,
    property_vectors: np.ndarray,
    property_ids: list[str],
    threshold_vec: np.ndarray,
) -> list[list[str]]:
    """
    기사 임베딩 chunk에 대해 유사도 기반 태깅을 수행한다.
    np.where 벡터화로 Python for 루프 대비 대폭 가속.
    """
    sim_matrix = article_embeddings @ property_vectors.T
    matches = sim_matrix >= threshold_vec  # broadcasting

    pid_array = np.array(property_ids)
    chunk_tags = []
    rows, cols = np.where(matches)
    # row별로 그룹핑
    if len(rows) == 0:
        return [[] for _ in range(article_embeddings.shape[0])]

    # 각 row의 시작/끝 인덱스 계산
    n_articles = article_embeddings.shape[0]
    tag_dict: dict[int, list[str]] = {}
    for r, c in zip(rows, cols):
        tag_dict.setdefault(int(r), []).append(property_ids[int(c)])

    for i in range(n_articles):
        chunk_tags.append(tag_dict.get(i, []))

    return chunk_tags


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def run():
    """Module 4 전체 파이프라인 실행 (chunk 기반)"""
    config = load_config()

    processed_path = get_path(config, "processed_data")
    outputs_path = get_path(config, "outputs")
    tag_config = config["tagging"]
    dict_config = config["dictionary"]

    corpus_path = processed_path / "news_corpus.parquet"
    out_path = processed_path / "corpus_tagged.parquet"
    checkpoint_dir = outputs_path / "module4_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 성질 사전 로드 ──
    with open(outputs_path / "property_dictionary.json", "r", encoding="utf-8") as f:
        property_dict = json.load(f)
    logger.info(f"성질 사전 로드: {len(property_dict)}개 항목")

    property_ids = [p["property_id"] for p in property_dict]

    # ── 2. 임베딩 모델 로드 ──
    model_name = dict_config["embedding_models"]["main"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"임베딩 모델 로드: {model_name} (device={device})")
    embed_model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        embed_model.half()
        logger.info("FP16(반정밀도) 모드 활성화 — VRAM 절감 + 속도 향상")

    # ── 3. 성질 벡터 생성 ──
    logger.info("성질 벡터 생성 중...")
    property_vectors = build_property_vectors(property_dict, embed_model)
    logger.info(f"성질 벡터: {property_vectors.shape}")

    # ── 4. 임계값 캘리브레이션 ──
    # Module 3에서 LLM으로 태깅한 샘플을 ground truth로 사용
    sample_path = outputs_path / "sample_tagging_raw.csv"
    if sample_path.exists():
        logger.info("캘리브레이션용 샘플 데이터 로드...")
        sample_df = pd.read_csv(sample_path)

        # tags 컬럼(자유 서술) → property_id 매핑
        if "tags" in sample_df.columns:
            logger.info("LLM 자유 태그 → property_id 매핑 수행 중...")
            sample_df = map_tags_to_properties(
                sample_df, property_dict, property_vectors, embed_model,
            )
        elif "properties" in sample_df.columns:
            # 이미 property_id가 있는 경우 (파싱만)
            sample_df["properties"] = sample_df["properties"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
            )
        else:
            logger.warning("샘플 데이터에 tags/properties 컬럼이 없습니다. 기본 threshold 사용.")
            sample_df = None
    else:
        logger.warning(f"캘리브레이션 샘플 없음: {sample_path}. 기본 threshold 사용.")
        sample_df = None

    if sample_df is not None and len(sample_df) > 0:
        # 샘플 기사 임베딩
        sample_embeddings = encode_articles(
            sample_df, embed_model,
            batch_size=tag_config.get("encode_batch_size", 256),
        )

        threshold_range = (
            tag_config.get("threshold_min", 0.3),
            tag_config.get("threshold_max", 0.8),
            tag_config.get("threshold_step", 0.05),
        )

        global_threshold, per_property_thresholds = calibrate_threshold(
            sample_df, sample_embeddings, property_vectors,
            property_ids, threshold_range,
        )

        del sample_embeddings
        gc.collect()
    else:
        global_threshold = tag_config.get("default_threshold", 0.5)
        per_property_thresholds = {pid: global_threshold for pid in property_ids}
        logger.info(f"기본 threshold 사용: {global_threshold}")

    # 캘리브레이션 결과 저장
    calibration_result = {
        "global_threshold": global_threshold,
        "per_property_thresholds": per_property_thresholds,
        "embedding_model": model_name,
    }
    cal_path = outputs_path / "threshold_calibration.json"
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(calibration_result, f, ensure_ascii=False, indent=2)
    logger.info(f"캘리브레이션 결과 저장: {cal_path}")

    # ── 5. Threshold 벡터 준비 ──
    use_per_property = tag_config.get("use_per_property_threshold", True)
    n_properties = len(property_ids)

    if use_per_property:
        threshold_vec = np.array(
            [per_property_thresholds.get(pid, global_threshold) for pid in property_ids]
        )
    else:
        threshold_vec = np.full(n_properties, global_threshold)

    # ── 6. 전체 코퍼스 sub-chunk 기반 태깅 ──
    pf = pq.ParquetFile(corpus_path)
    total_rows = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups
    sub_chunk_size = tag_config.get("sub_chunk_size", 50_000)
    logger.info(
        f"코퍼스: {total_rows:,}건, {n_row_groups}개 row group, "
        f"sub-chunk 크기: {sub_chunk_size:,}건"
    )

    # 이전 체크포인트 확인 (crash 후 재개 지원)
    # 체크포인트 키: "rg{rg_idx}_sc{sc_idx}" (row group + sub-chunk 인덱스)
    checkpoint_meta_path = checkpoint_dir / "progress.json"
    completed_chunks: set[str] = set()
    completed_groups: set[int] = set()
    if checkpoint_meta_path.exists():
        with open(checkpoint_meta_path, "r") as f:
            progress = json.load(f)
        completed_chunks = set(progress.get("completed_chunks", []))
        completed_groups = set(progress.get("completed_groups", []))
        logger.info(
            f"체크포인트에서 재개: {len(completed_groups)}/{n_row_groups} 그룹 완료, "
            f"{len(completed_chunks)} sub-chunks 완료"
        )

    batch_size = tag_config.get("encode_batch_size", 256)
    processed_total = 0
    n_tagged_total = 0
    total_tag_count = 0
    start_time = time.time()
    global_sc_count = 0
    checkpoint_interval = 20  # 매 20 sub-chunk마다 진행 상태 저장

    save_cols = [
        "article_id", "date", "title", "lead",
        "event_id", "event_type", "properties",
    ]

    for rg_idx in tqdm(range(n_row_groups), desc="Row groups"):
        if rg_idx in completed_groups:
            rg_meta = pf.metadata.row_group(rg_idx)
            processed_total += rg_meta.num_rows
            continue

        # row group 읽기
        table = pf.read_row_group(rg_idx)
        chunk_df = table.to_pandas()
        del table

        rg_size = len(chunk_df)
        n_sub_chunks = (rg_size + sub_chunk_size - 1) // sub_chunk_size

        rg_results: list[pd.DataFrame] = []

        for sc_idx in range(n_sub_chunks):
            sc_key = f"rg{rg_idx}_sc{sc_idx}"

            # 이미 완료된 sub-chunk → 파일이 있으면 스킵
            if sc_key in completed_chunks:
                sc_start = sc_idx * sub_chunk_size
                sc_end = min(sc_start + sub_chunk_size, rg_size)
                processed_total += (sc_end - sc_start)
                global_sc_count += 1
                continue

            sc_start = sc_idx * sub_chunk_size
            sc_end = min(sc_start + sub_chunk_size, rg_size)
            sub_df = chunk_df.iloc[sc_start:sc_end].copy()

            # 인코딩
            sub_embeddings = encode_articles(
                sub_df, embed_model, batch_size=batch_size,
            )

            # 태깅
            sub_tags = tag_chunk(
                sub_embeddings, property_vectors, property_ids, threshold_vec,
            )
            sub_df["properties"] = sub_tags

            # 통계 누적
            n_tagged_total += sum(1 for tags in sub_tags if len(tags) > 0)
            total_tag_count += sum(len(tags) for tags in sub_tags)

            # 임베딩 메모리 해제
            del sub_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # sub-chunk 결과를 parquet로 저장
            existing_cols = [c for c in save_cols if c in sub_df.columns]
            sub_result = sub_df[existing_cols]
            sc_path = checkpoint_dir / f"chunk_{rg_idx:04d}_sc{sc_idx:04d}.parquet"
            sc_table = pa.Table.from_pandas(sub_result, preserve_index=False)
            pq.write_table(sc_table, sc_path, compression="zstd")

            del sub_df, sub_result, sc_table, sub_tags
            gc.collect()

            processed_total += (sc_end - sc_start)
            completed_chunks.add(sc_key)
            global_sc_count += 1

            # 진행률 로그
            elapsed = time.time() - start_time
            rate = processed_total / elapsed if elapsed > 0 else 0
            remaining = (total_rows - processed_total) / rate if rate > 0 else 0
            logger.info(
                f"[RG {rg_idx + 1}/{n_row_groups}, SC {sc_idx + 1}/{n_sub_chunks}] "
                f"{processed_total:,}/{total_rows:,}건 처리 "
                f"({processed_total / total_rows * 100:.1f}%) "
                f"| {rate:.0f}건/초 "
                f"| 경과 {elapsed / 3600:.1f}h | 잔여 ~{remaining / 3600:.1f}h"
            )

            # 체크포인트 저장 (매 N sub-chunk마다)
            if global_sc_count % checkpoint_interval == 0:
                with open(checkpoint_meta_path, "w") as f:
                    json.dump({
                        "completed_chunks": sorted(completed_chunks),
                        "completed_groups": sorted(completed_groups),
                    }, f)
                logger.info(f"체크포인트 저장: {len(completed_chunks)} sub-chunks 완료")

        # row group 완료
        del chunk_df
        gc.collect()
        completed_groups.add(rg_idx)

        # row group 끝날 때도 체크포인트 저장
        with open(checkpoint_meta_path, "w") as f:
            json.dump({
                "completed_chunks": sorted(completed_chunks),
                "completed_groups": sorted(completed_groups),
            }, f)

    # 최종 체크포인트 저장
    with open(checkpoint_meta_path, "w") as f:
        json.dump({
            "completed_chunks": sorted(completed_chunks),
            "completed_groups": sorted(completed_groups),
        }, f)

    # ── 7. 모든 sub-chunk를 하나의 parquet로 병합 ──
    logger.info("sub-chunk 병합 중...")
    chunk_files = sorted(checkpoint_dir.glob("chunk_*_sc*.parquet"))

    writer = None
    for chunk_file in tqdm(chunk_files, desc="병합"):
        chunk_table = pq.read_table(chunk_file)
        if writer is None:
            writer = pq.ParquetWriter(
                out_path, chunk_table.schema, compression="zstd",
            )
        writer.write_table(chunk_table)
        del chunk_table

    if writer is not None:
        writer.close()

    # 태깅 통계
    avg_tags = total_tag_count / total_rows if total_rows > 0 else 0
    logger.info(
        f"태깅 결과: {n_tagged_total:,}/{total_rows:,}건 태깅됨 "
        f"(평균 {avg_tags:.1f}개 성질/기사)"
    )

    elapsed_total = time.time() - start_time
    logger.info(f"태깅 코퍼스 저장: {total_rows:,}건 → {out_path}")
    logger.info(f"총 소요 시간: {elapsed_total / 3600:.1f}시간")
    logger.info("=== Module 4 완료 ===")


if __name__ == "__main__":
    run()
