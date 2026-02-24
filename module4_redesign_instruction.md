# Module 4 설계 변경 요청

## 배경
현재 Module 4(Full Corpus Tagging)가 LLM API로 전체 기사를 태깅하도록 설계되어 있다.
실제 데이터를 확인한 결과 태깅 대상 기사가 약 1,600만 건으로 추산되며,
가장 저렴한 조합(Haiku + Batch)으로도 $3,000 이상이 소요된다.

Module 4는 이미 만들어진 성질 사전을 "적용"하는 단계이므로,
LLM 대신 임베딩 유사도 매칭으로 대체해도 Anti-Bias Rule을 위반하지 않는다.
(Anti-Bias Rule은 성질을 "발견"하는 Module 3에만 해당)

## 변경 요청 사항

### 1. `BlackEvent_Predictor_Architecture.md` 수정

#### 파이프라인 개요 수정
Module 4 설명을 아래와 같이 변경:
```
Module 4: Full Corpus Tagging
  (성질 사전 기반 임베딩 유사도 매칭으로 전체 기사 태깅 — LLM 미사용, 로컬 처리)
```

#### Module 4 섹션 전체 재작성
기존 LLM 프롬프트 기반 방식을 삭제하고, 아래 내용으로 교체:

**방법:**
- Module 3에서 사용한 동일 임베딩 모델(KURE-v1 메인, BGE-M3 교차검증)을 사용
- 성질 사전의 각 항목(representative_tag + cluster_keywords)을 임베딩하여 성질 벡터 생성
- 전체 기사의 제목+리드문을 임베딩하여 기사 벡터 생성
- 기사 벡터와 각 성질 벡터 간 코사인 유사도 계산
- 유사도가 임계값(threshold) 이상인 성질을 해당 기사에 태깅
- 임계값은 Module 3의 샘플 1,000건에서 LLM 태깅 결과와 비교하여 최적값 탐색
  (LLM 태깅 결과와 F1 Score가 최대화되는 threshold 선택)

**임계값 캘리브레이션 절차:**
1. Module 3에서 LLM으로 태깅한 1,000건 샘플의 결과를 정답지(ground truth)로 사용
2. 같은 1,000건에 대해 임베딩 매칭 수행
3. threshold를 0.3~0.8 범위에서 0.05 단위로 탐색
4. LLM 태깅 결과 대비 F1 Score가 가장 높은 threshold를 채택
5. 성질별로 최적 threshold가 다를 수 있으므로, 글로벌 threshold와 성질별 threshold 모두 시도

**기사 1건당 복수 성질 태그 가능 (기존과 동일)**

**장점:**
- API 비용 $0 (로컬 GPU 또는 CPU에서 처리)
- 완벽한 일관성 (동일 입력 → 동일 결과)
- 재현 가능성 100%

**주의사항:**
- 임베딩은 표면적 표현이 다르면 의미적 연결을 놓칠 수 있음
- 이를 보완하기 위해 성질 벡터를 representative_tag 하나가 아닌 cluster_keywords 전체의 평균 벡터로 구성
- 임계값 캘리브레이션으로 품질을 LLM 태깅에 근접하도록 조정

**배치 처리:**
- sentence-transformers의 model.encode()에 batch_size 설정하여 대량 처리
- GPU 사용 시 1,600만 건 처리 예상 시간: 수 시간
- CPU만 사용 시: 1~2일 (그래도 LLM API 대비 비용 $0)

**출력 (기존과 동일):**
- `corpus_tagged.parquet`: 전체 기사 + 태깅 결과
  - columns: article_id, date, title, lead, event_id, properties (list)

#### 기술 스택 섹션 수정
LLM (성질 태깅) 항목을 아래로 변경:
- LLM: Module 3 샘플 태깅(1,000건)에만 사용. Claude API (Anthropic)
- Module 4 전체 태깅: 임베딩 유사도 매칭 (LLM 미사용, 로컬 처리)

#### 주의사항 및 리스크 섹션 수정
- 기존: "LLM 태깅의 일관성" 리스크 → 임베딩 매칭으로 해소되었음을 명시
- 추가: "임베딩 매칭의 한계: 간접적 의미 연결을 놓칠 수 있으나, 통계적 패턴 분석에서는 대량 데이터의 일관성이 더 중요하므로 수용 가능한 트레이드오프"

### 2. 변경하지 않는 것들 (그대로 유지)
- Module 3: LLM 오픈형 태깅 그대로 유지 (Anti-Bias Rule 적용 대상)
- Module 3의 Anti-Bias Rule 5개 조항: 그대로 유지
- Module 5, 6, 7: 변경 없음
- 프로젝트 철학, 핵심 원칙: 변경 없음
