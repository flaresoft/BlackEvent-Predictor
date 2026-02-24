# BlackEvent Predictor — 프로젝트 설계 문서

## 프로젝트 철학

> "인간역사는 반복된다. 겉보기에 무관한 사건들이 누적되면서 임계점을 넘어 주가 급락이 발생하는 패턴을 찾는다."

### 핵심 원칙
- **백지 상태(Zero Bias)**: 어떤 사건이 원인인지 사람이 미리 정하지 않는다. 데이터가 스스로 말하게 한다.
- **주제가 아닌 구조**: "부동산", "반도체" 같은 토픽이 아니라 "사건이 가진 구조적 성질"을 팩터로 삼는다.
- **섹션 무시**: 정치면/경제면/사회면 구분을 태깅에 사용하지 않는다. 모든 기사는 출처 구분 없이 하나의 풀.

---

## 전체 파이프라인 개요

```
[코스피 데이터] → Module 1: BlackEvent Index
                          ↓
[뉴스 데이터]  → Module 2: News Corpus
                          ↓
                  Module 3: Property Dictionary Generation
                    (샘플 → LLM 오픈형 태깅 → 임베딩 → 클러스터링 → 성질 사전)
                          ↓
                  Module 4: Full Corpus Tagging
                    (성질 사전 기반 임베딩 유사도 매칭으로 전체 기사 태깅 — LLM 미사용, 로컬 처리)
                          ↓
                  Module 5: Temporal Pattern Analysis
                    (BlackEvent 직전 vs 평상시, 성질 조합 패턴 비교)
                          ↓
                  Module 6: Risk Score Model
                    (A. 임계값 기반 + B. 가중 스코어링, 양쪽 모두)
                          ↓
                  Module 7: Validation
                    (시간순 분할 검증)
```

---

## Module 1 — BlackEvent Index

### 목적
코스피 일별 데이터에서 급락 이벤트를 정의하고 추출

### 데이터
- 소스: KRX(한국거래소) 또는 Yahoo Finance (^KS11)
- 기간: 2006년 1월 ~ 현재
- 단위: 일봉 (일별 종가)

### BlackEvent 정의

```
1차 — 충격일(ShockDay) 탐지
  - 전일 대비 종가 변동률 ≤ -2.0%인 날

2차 — 이벤트 클러스터링
  - ShockDay 간 간격이 N일 이내이면 하나의 BlackEvent로 묶음
  - N은 분석 과정에서 최적값 탐색 (초기값: 5일, 탐색 범위: 3~14일)
  
3차 — 분석 기준점
  - 각 BlackEvent의 "첫 번째 ShockDay"만 분석 대상
  - 이후 ShockDay는 "위기 진행 중"이므로 예측 대상에서 제외
```

### BlackEvent 메타데이터
각 BlackEvent에 아래 정보를 기록:
- `first_shock_date`: 첫 충격일 (= 뉴스 분석 기준점)
- `duration_days`: 이벤트 지속 기간
- `shock_count`: 이벤트 내 충격일 횟수  
- `max_cumulative_drop`: 누적 최대 낙폭 (%)

### 대조군 정의
- BlackEvent first_shock_date 기준 전후 90일 이내가 아닌 "평상시" 날짜를 무작위 샘플링
- 대조군 수는 BlackEvent 수의 2~3배

### 출력
- `black_events.csv`: BlackEvent 목록 + 메타데이터
- `control_dates.csv`: 대조군 날짜 목록

---

## Module 2 — News Corpus

### 목적
BlackEvent 및 대조군 날짜 기준으로 직전 뉴스 기사 수집

### 수집 범위
- 각 기준 날짜로부터 **직전 60일간** 기사
- 충분히 넓게 잡고, Module 5에서 최적 윈도우를 탐색

### 수집 대상 섹션
- 정치, 경제, 사회/문화 (단, 섹션 태그는 수집만 하고 분석에 사용하지 않음)

### 수집 단위
- 제목 (title)
- 리드문 (첫 1~2문장, lead)
- 발행일 (date)
- 매체명 (source) — 참고용만
- **섹션 태그 제거** — 모든 기사가 출처 구분 없이 하나의 풀

### 데이터 소스 (우선순위)
1. **빅카인즈(BIGKinds)** — 한국언론진흥재단, 무료, 날짜/섹션 검색 가능 (주력)
2. **네이버 뉴스 아카이브** — 보조 소스
3. 기타 접근 가능한 뉴스 API

### 전처리
- 중복 기사 제거 (제목 유사도 기반)
- 텍스트 정규화 (특수문자, 공백 정리)
- 빈 리드문 처리

### 출력
- `news_corpus.parquet`: 전체 기사 데이터
  - columns: `article_id`, `date`, `title`, `lead`, `event_id` (연결된 BlackEvent 또는 대조군 ID)

---

## Module 3 — Property Dictionary Generation

### 목적
기사의 "구조적 성질"을 비지도로 추출하여 성질 사전(Property Dictionary)을 생성

### ⚠️ Anti-Bias Rule (절대 준수)
```
1. LLM에게 기사 원문만 제공한다. 예시 태그를 절대 주지 않는다.
2. "다음 중 선택하시오" 형태를 절대 사용하지 않는다.
3. 프로젝트의 목적(주가 예측)을 LLM에게 알려주지 않는다.
4. 이전 기사의 태깅 결과를 다음 기사 태깅 시 컨텍스트로 넣지 않는다.
5. 자유 서술 → 후처리로 구조화한다.
```

### 3단계 프로세스

**Step 3-1: 샘플 오픈형 태깅**
- 전체 코퍼스에서 **5,000건 층화 샘플링(Stratified Sampling)**
- 층화 기준 1: 시기별 균등 (2006~2009 / 2010~2013 / 2014~2017 / 2018~2021 / 2022~현재, 각 1,000건)
- 층화 기준 2: 각 시기 내 BlackEvent 직전 / 평상시 균등 배분 (각 500건)
- 특정 구간에서 해당 유형 기사가 부족하면 인접 구간에서 보충 (비율 유지)
- LLM 프롬프트:

```
다음은 뉴스 기사의 제목과 첫 문장입니다.

제목: {title}
본문: {lead}

이 기사가 묘사하는 사건 또는 현상의 구조적 특성을 3~5개의 짧은 구(phrase)로 서술하세요.
- 사건의 "주제"가 아니라 "사건이 가진 성질"을 서술하세요
- 예시를 참고하지 마세요. 자유롭게 서술하세요
```

- 출력: 기사당 3~5개의 자유 서술 태그 → 총 15,000~25,000개 raw 태그

**Step 3-2: 태그 클러스터링 → 성질 사전 생성**
- Raw 태그들을 임베딩
  - **메인 모델: KURE-v1** (`nlpai-lab/KURE-v1`)
  - **교차 검증: BGE-M3** (`BAAI/bge-m3` 또는 `dragonkue/BGE-m3-ko`)
- HDBSCAN 클러스터링 (클러스터 수 사전 지정 안 함)
- 각 클러스터의 대표 태그(centroid에 가장 가까운 태그)를 성질 사전 항목으로 등록
- **두 임베딩 모델에서 공통으로 나타나는 클러스터 = 높은 신뢰도**

**Step 3-3: 성질 사전 정제**
- 너무 작은 클러스터(기사 N건 미만) 제거
- 너무 일반적인 클러스터(거의 모든 기사에 해당) 제거
- 최종 성질 사전: Property_001 ~ Property_N

### 출력
- `property_dictionary.json`: 성질 사전
  - 각 항목: `property_id`, `representative_tag`, `cluster_keywords`, `cluster_size`
- `sample_tagging_raw.csv`: 5,000건 샘플의 raw 태그 데이터

---

## Module 4 — Full Corpus Tagging

### 목적
성질 사전을 기반으로 전체 뉴스 코퍼스에 구조적 성질을 태깅

### 방법
- Module 3에서 사용한 동일 임베딩 모델(KURE-v1 메인, BGE-M3 교차검증)을 사용
- 성질 사전의 각 항목(representative_tag + cluster_keywords)을 임베딩하여 성질 벡터 생성
- 전체 기사의 제목+리드문을 임베딩하여 기사 벡터 생성
- 기사 벡터와 각 성질 벡터 간 코사인 유사도 계산
- 유사도가 임계값(threshold) 이상인 성질을 해당 기사에 태깅
- 임계값은 Module 3의 샘플 5,000건에서 LLM 태깅 결과와 비교하여 최적값 탐색
  (LLM 태깅 결과와 F1 Score가 최대화되는 threshold 선택)

### 임계값 캘리브레이션 절차
1. Module 3에서 LLM으로 태깅한 5,000건 샘플의 결과를 정답지(ground truth)로 사용
2. 같은 5,000건에 대해 임베딩 매칭 수행
3. threshold를 0.3~0.8 범위에서 0.05 단위로 탐색
4. LLM 태깅 결과 대비 F1 Score가 가장 높은 threshold를 채택
5. 성질별로 최적 threshold가 다를 수 있으므로, 글로벌 threshold와 성질별 threshold 모두 시도

### 기사 1건당 복수 성질 태그 가능 (기존과 동일)

### 장점
- API 비용 $0 (로컬 GPU 또는 CPU에서 처리)
- 완벽한 일관성 (동일 입력 → 동일 결과)
- 재현 가능성 100%

### 주의사항
- 임베딩은 표면적 표현이 다르면 의미적 연결을 놓칠 수 있음
- 이를 보완하기 위해 성질 벡터를 representative_tag 하나가 아닌 cluster_keywords 전체의 평균 벡터로 구성
- 임계값 캘리브레이션으로 품질을 LLM 태깅에 근접하도록 조정

### 배치 처리
- sentence-transformers의 model.encode()에 batch_size 설정하여 대량 처리
- GPU 사용 시 1,600만 건 처리 예상 시간: 수 시간
- CPU만 사용 시: 1~2일 (그래도 LLM API 대비 비용 $0)

### 출력
- `corpus_tagged.parquet`: 전체 기사 + 태깅 결과
  - columns: `article_id`, `date`, `title`, `lead`, `event_id`, `properties` (list)

---

## Module 5 — Temporal Pattern Analysis

### 목적
BlackEvent 직전에 유독 많이 출현하는 성질 조합 패턴을 찾는다

### 분석 1 — 단일 성질 빈도 검정
- 각 성질(Property)에 대해:
  - BlackEvent 직전 N일 내 출현 빈도
  - 대조군 동일 기간 출현 빈도
  - 통계 검정 (카이제곱 검정 또는 Fisher's exact test)
- **빈도 기반 + 존재 기반** 모두 수행

### 분석 2 — 최적 윈도우 탐색
- 각 성질마다 [3일, 7일, 14일, 21일, 30일, 45일, 60일] 윈도우에서 검정
- 성질별로 가장 유의미한 윈도우 = 그 성질의 "리드타임"

### 분석 3 — 성질 조합 패턴
- 단일 성질이 아닌 **성질 조합**의 동시 출현 패턴 탐색
- 방법: 연관규칙 마이닝 (Apriori) 또는 그래프 기반 동시출현 분석
- BlackEvent 직전에만 유독 빈번한 조합 추출

### 출력
- `significant_properties.csv`: 유의미한 단일 성질 목록 + p-value + 최적 윈도우
- `significant_combinations.csv`: 유의미한 성질 조합 목록 + 지지도, 신뢰도, 향상도
- 시계열 히트맵 시각화

---

## Module 6 — Risk Score Model

### 목적
현재 시점의 뉴스에서 성질을 추출하고 BlackEvent 리스크 스코어를 산출

### Model A — 임계값 기반 (Threshold)
```
유의미한 성질 중 현재 N개 이상 동시 존재 → 경고 수준 산출
- Level 0 (Normal): 유의미한 성질 0~2개
- Level 1 (Watch): 유의미한 성질 3~4개
- Level 2 (Warning): 유의미한 성질 5~6개
- Level 3 (Alert): 유의미한 성질 7개 이상 또는 유의미한 조합 2개 이상 매칭

임계값은 Module 7 Validation에서 최적화
```

### Model B — 가중 스코어링 (Weighted Score)
```
Risk Score = Σ (property_weight_i × presence_i) + Σ (combination_weight_j × combo_presence_j)

- property_weight: Module 5에서 산출된 각 성질의 BlackEvent 상관계수
- combination_weight: 성질 조합의 향상도(lift) 기반 가중치
- 각 성질의 최적 윈도우 내 존재 여부로 presence 판정
```

### 출력
- 두 모델 모두 구현
- Module 7에서 어느 쪽이 더 나은지 비교

---

## Module 7 — Validation

### 목적
모델의 실제 예측력 검증

### 방법
- **시간순 분할**: 2006~2019 학습 / 2020~현재 테스트
- 학습 데이터로 성질 사전 생성, 패턴 추출, 가중치 산출
- 테스트 데이터로 예측 수행

### 평가 지표
- **Precision**: 경고를 발령했을 때, 실제로 BlackEvent가 왔는가
- **Recall**: 실제 BlackEvent 중 몇 %를 사전에 잡아냈는가
- **Lead Time**: 경고에서 BlackEvent까지 평균 며칠 전에 잡히는가
- **False Positive Rate**: 경고 발령 중 BlackEvent가 안 온 비율

### Model A vs Model B 비교
- 동일 테스트 세트에서 두 모델의 위 지표 비교
- 가능하면 앙상블(A+B 결합)도 시도

### 출력
- `validation_report.md`: 종합 검증 리포트
- 혼동행렬, ROC 커브, Precision-Recall 커브 시각화

---

## 기술 스택 (권장)

### 언어
- Python 3.10+

### 데이터 처리
- pandas, polars
- parquet 형식 (대용량 처리)

### 임베딩 모델
- **메인**: `nlpai-lab/KURE-v1` (sentence-transformers)
- **교차 검증**: `BAAI/bge-m3`
- 설치: `pip install sentence-transformers`

### 클러스터링
- HDBSCAN (`pip install hdbscan`)
- 클러스터 수 사전 지정 안 함

### LLM
- Module 3 샘플 태깅(5,000건)에만 사용. Claude API (Anthropic)
- Module 4 전체 태깅: 임베딩 유사도 매칭 (LLM 미사용, 로컬 처리)

### 통계 검정
- scipy.stats (chi2_contingency, fisher_exact)
- mlxtend (연관규칙 마이닝, Apriori)

### 시각화
- matplotlib, seaborn
- plotly (인터랙티브)

### 뉴스 수집
- 빅카인즈 API / 웹 스크래핑
- 네이버 뉴스 API (보조)

---

## 프로젝트 구조 (권장)

```
blackevent-predictor/
├── README.md
├── config/
│   └── settings.yaml          # 전역 설정 (기간, 임계값, 모델 경로 등)
├── data/
│   ├── raw/                   # 원본 데이터
│   ├── processed/             # 전처리된 데이터
│   └── outputs/               # 분석 결과
├── src/
│   ├── module1_blackevent/    # BlackEvent Index
│   ├── module2_corpus/        # News Corpus 수집/전처리
│   ├── module3_dictionary/    # Property Dictionary 생성
│   ├── module4_tagging/       # Full Corpus Tagging
│   ├── module5_analysis/      # Temporal Pattern Analysis
│   ├── module6_scoring/       # Risk Score Model
│   └── module7_validation/    # Validation
├── notebooks/                 # 탐색적 분석 노트북
└── tests/                     # 테스트 코드
```

---

## 실행 순서

```bash
# 1. 코스피 데이터 수집 + BlackEvent 추출
python -m src.module1_blackevent.run

# 2. 뉴스 코퍼스 수집
python -m src.module2_corpus.run

# 3. 성질 사전 생성 (샘플 태깅 → 클러스터링)
python -m src.module3_dictionary.run

# 4. 전체 코퍼스 태깅
python -m src.module4_tagging.run

# 5. 패턴 분석
python -m src.module5_analysis.run

# 6. 리스크 스코어 모델 구축
python -m src.module6_scoring.run

# 7. 검증
python -m src.module7_validation.run
```

---

## 주의사항 및 리스크

### 데이터 리스크
- 빅카인즈 2006~2010년 초기 데이터 커버리지가 낮을 수 있음
- 뉴스 기사 저작권 이슈: 제목 + 리드문 수준으로 제한

### 분석 리스크
- ~~LLM 태깅의 일관성~~ → Module 4를 임베딩 매칭으로 전환하여 해소됨 (동일 입력 → 동일 결과, 재현 가능성 100%)
- 임베딩 매칭의 한계: 간접적 의미 연결을 놓칠 수 있으나, 통계적 패턴 분석에서는 대량 데이터의 일관성이 더 중요하므로 수용 가능한 트레이드오프
- 과적합: BlackEvent 자체가 20년간 수십~100건 수준이므로 샘플 수 부족 가능
- 생존자 편향: "이 패턴이 나타났는데 BlackEvent가 안 온" 케이스도 충분히 분석해야 함

### Anti-Bias Rule (재강조)
- Module 3의 LLM 프롬프트에 예시 태그를 절대 넣지 않는다
- 프로젝트 목적(주가 예측)을 태깅 LLM에게 알려주지 않는다
- 섹션(정치/경제/사회) 정보를 태깅 입력에 포함하지 않는다
