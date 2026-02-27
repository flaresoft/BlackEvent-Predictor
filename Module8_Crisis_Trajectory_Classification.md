# Module 8 — Crisis Trajectory Classification (위기 궤적 유형 분류)

## 위치

```
Module 7: Validation
  (시간순 분할 검증)
          ↓
Module 8: Crisis Trajectory Classification ← NEW
  (BlackEvent 스코어 궤적 역산 → 비지도 유형 분류 → 라이브 궤적 매칭)
```

Module 7 이후에 위치하는 **독립 모듈**이다.
기존 Module 1~7의 코드, 출력, 검증 결과를 일절 변경하지 않는다.

---

## 목적

현재 모델(Module 6 Model B)은 **"위험도가 얼마인가"**만 말한다.
Module 8은 **"이 위험이 어떤 유형이고, 지금 어디쯤인가"**를 말한다.

- 같은 Risk Score 36이라도:
  - "천천히 쌓여온 축적형 위기의 중반" → 아직 시간이 있다
  - "갑자기 치솟는 급발진형 위기의 직전" → 당장 대응해야 한다
- 이 구분은 기존 단일 스코어로는 불가능하다

---

## 핵심 원칙

### Anti-Bias Rule 확장 적용

```
1. 위기 유형의 수를 사전에 지정하지 않는다.
2. "급발진형", "축적형" 같은 인간의 유형 분류를 레이블로 사용하지 않는다.
3. 궤적 데이터를 클러스터링에 넣고, 데이터가 유형을 정하게 한다.
4. 클러스터링 결과에 대한 해석(naming)은 사후에 수행한다.
```

Module 3에서 성질 사전을 만들 때와 동일한 철학이다.
HDBSCAN이 113개 성질을 만들어낸 것처럼, 궤적 유형도 HDBSCAN이 정한다.

---

## 입력 데이터

Module 8은 기존 파이프라인의 출력만을 입력으로 사용한다.

| 입력 파일 | 출처 | 용도 |
|-----------|------|------|
| `black_events.csv` | Module 1 | BlackEvent 목록 + first_shock_date |
| `control_dates.csv` | Module 1 | 대조군 날짜 목록 |
| `corpus_tagged.parquet` | Module 4 | 전체 기사 + 성질 태깅 결과 |
| `significant_properties.csv` | Module 5 | 유의미 성질 12개 + 가중치 |
| `risk_scores.csv` | Module 6 | (있으면 활용, 없으면 자체 산출) |

---

## 3단계 프로세스

### Step 8-1: BlackEvent 궤적 역산

각 BlackEvent에 대해, 복수 시점의 Risk Score를 역산한다.

```
시점: [D-30, D-21, D-14, D-7, D-0]

각 시점에서:
  1. 해당 날짜 기준 직전 21일(최적 윈도우) 내 뉴스를 수집
  2. Module 6 Model B 공식으로 Risk Score를 산출
     score = Σ(weight × max(freq_ratio - 1, 0))
  3. 결과: BlackEvent 1건당 5개 시점의 스코어 → 궤적 벡터 [s_30, s_21, s_14, s_7, s_0]
```

**141개 BlackEvent → 141개 궤적 벡터 (각 5차원)**

#### 정규화

BlackEvent마다 스코어 규모가 다를 수 있으므로, 궤적 벡터를 정규화한다.

```
방법 A: Min-Max 정규화 (궤적 내)
  - 각 궤적의 최솟값을 0, 최댓값을 1로 스케일링
  - 장점: "모양"만 비교 가능 (크기 무시)
  - 단점: 절대적 위험도 정보를 잃음

방법 B: 글로벌 정규화
  - 전체 141개 궤적의 글로벌 max로 나눔
  - 장점: 절대적 위험도 보존
  - 단점: 큰 이벤트에 의해 작은 이벤트의 모양이 뭉개질 수 있음

방법 C: 양쪽 모두 수행 → 클러스터링 결과 비교
  - Module 3에서 KURE-v1 / BGE-M3 교차검증한 것과 같은 논리
```

**권장: 방법 C — 양쪽 모두 수행 후 공통 클러스터를 높은 신뢰도로 채택**

#### 대조군 궤적

- 대조군 423개에 대해서도 동일한 궤적 벡터를 산출
- 클러스터링에는 포함하지 않음 (BlackEvent 궤적만으로 유형 분류)
- 이후 Step 8-3에서 "대조군 궤적은 어떤 유형에도 매칭되지 않는가" 검증에 사용

---

### Step 8-2: 궤적 클러스터링 → 유형 사전 생성

141개 궤적 벡터를 HDBSCAN으로 클러스터링한다.

```
모델: HDBSCAN
입력: 141개 × 5차원 궤적 벡터
파라미터:
  - min_cluster_size: 통계적 최소 의미 기준
    → 각 유형에 최소 N개 BlackEvent가 포함되어야 함
    → 초기값: 10 (141의 약 7%)
    → 탐색 범위: 5~20
  - min_samples: HDBSCAN 기본값 사용 (= min_cluster_size)
  - metric: euclidean (5차원 공간에서 표준적)

클러스터 수: 사전 지정하지 않음 (HDBSCAN이 결정)
노이즈 포인트: HDBSCAN의 label=-1 → "분류 불가 유형"으로 별도 관리
```

#### 유형 사전 (Trajectory Dictionary)

각 클러스터에 대해 아래 정보를 기록:

```json
{
  "trajectory_type_id": "T001",
  "cluster_size": 38,
  "centroid": [0.12, 0.18, 0.35, 0.67, 1.0],
  "std": [0.05, 0.08, 0.12, 0.15, 0.10],
  "member_events": ["BE_001", "BE_015", "BE_042", ...],
  "interpretation": null
}
```

- `centroid`: 클러스터 중심 궤적 (= 이 유형의 "평균 모양")
- `std`: 각 시점의 표준편차 (= 이 유형 내 변동 폭)
- `interpretation`: 사후 해석용 필드. 클러스터링 시에는 null.

#### 사후 해석 (Post-hoc Naming)

클러스터링 완료 후, 각 유형의 centroid 모양을 보고 **사람이** 이름을 붙인다.
이름은 분석에 사용되지 않으며, 순전히 가독성을 위한 것이다.

예시 (실제 클러스터링 결과에 따라 달라짐):
- centroid가 [낮음, 낮음, 낮음, 높음, 매우높음] → "급발진형"
- centroid가 [중간, 중간, 높음, 높음, 매우높음] → "축적형"
- centroid가 [높음, 높음, 높음, 높음, 높음] → "고원형"

**이름이 분석 로직에 영향을 주어서는 안 된다.**

---

### Step 8-3: 궤적 매칭 — "지금 어디쯤인가"

라이브 진단 또는 과거 날짜 진단 시, 현재 시점의 궤적을 유형에 매칭한다.

#### 부분 궤적 매칭

라이브 진단의 핵심 난이도: **D-0이 아직 안 왔다.**
즉, 5개 시점 중 일부만 존재하는 "부분 궤적"으로 매칭해야 한다.

```
예: 오늘이 미지의 BlackEvent로부터 D-14 시점이라면
  - 관측 가능한 궤적: [s_30, s_21, s_14, ?, ?]
  - D-7, D-0은 아직 모름

하지만 우리는 "오늘이 D-몇인지"를 모른다.
따라서 아래와 같이 슬라이딩 매칭을 수행한다.
```

#### 슬라이딩 윈도우 매칭

```
오늘(T) 기준으로, 과거 여러 시점의 스코어를 산출:

  score_T      = 오늘 기준 Risk Score
  score_T-7    = 7일 전 기준 Risk Score  
  score_T-14   = 14일 전 기준 Risk Score
  score_T-21   = 21일 전 기준 Risk Score
  score_T-30   = 30일 전 기준 Risk Score

관측된 궤적: [score_T-30, score_T-21, score_T-14, score_T-7, score_T]
```

이 관측 궤적을 각 유형의 centroid와 비교하되, **"관측 궤적이 centroid의 어디까지와 매칭되는가"**를 본다.

```
가설 1: "오늘이 D-0이다"
  → 관측 궤적 [T-30, T-21, T-14, T-7, T]를 
    centroid [D-30, D-21, D-14, D-7, D-0]과 전체 비교
  → 유사도 계산

가설 2: "오늘이 D-7이다"  
  → 관측 궤적 [T-30, T-21, T-14, T-7]을
    centroid [D-30, D-21, D-14, D-7]의 앞 4개와 비교
  → 유사도 계산

가설 3: "오늘이 D-14이다"
  → 관측 궤적 [T-30, T-21, T-14]을
    centroid [D-30, D-21, D-14]의 앞 3개와 비교
  → 유사도 계산

가설 4: "오늘이 D-21이다"
  → 관측 궤적 [T-30, T-21]을
    centroid [D-30, D-21]의 앞 2개와 비교
  → 유사도 계산
```

각 (유형, 가설) 조합에 대해 유사도를 산출하면:

```
결과 = {
  "best_match_type": "T001",
  "best_match_phase": "D-14",
  "similarity": 0.87,
  "all_matches": [
    {"type": "T001", "phase": "D-14", "similarity": 0.87},
    {"type": "T002", "phase": "D-7",  "similarity": 0.72},
    {"type": "T001", "phase": "D-7",  "similarity": 0.65},
    ...
  ]
}
```

#### 유사도 측정

궤적 간 유사도는 두 가지를 동시에 본다:

```
1. 코사인 유사도 — 궤적의 "모양"이 얼마나 비슷한가
2. 유클리드 거리 — 궤적의 "절대 수준"이 얼마나 가까운가

종합 유사도 = α × cosine_sim + (1 - α) × (1 - normalized_euclidean)
α는 Step 8-1의 정규화 방법에 따라 조정
  - Min-Max 정규화 사용 시: α = 0.7 (모양 중심)
  - 글로벌 정규화 사용 시: α = 0.5 (균형)
```

#### 신뢰도 필터

매칭 결과의 신뢰도가 낮으면 "분류 불가"로 출력한다.

```
조건:
  - best_match의 similarity < threshold → "분류 불가"
  - 1위와 2위의 similarity 차이 < margin → "불확실" (복수 유형 제시)

threshold, margin은 대조군 궤적으로 캘리브레이션:
  - 대조군 423개의 매칭 유사도 분포를 구함
  - 대조군의 95 percentile 이상 = 의미 있는 매칭으로 판정
  → 이렇게 하면 "평상시에도 나올 수 있는 유사도"를 자동으로 걸러냄
```

---

## 검증

### 검증 1: 유형 분리 품질

- 실루엣 스코어: 클러스터 간 분리가 명확한가
- 대조군 검증: 대조군 궤적이 어떤 유형에도 높은 유사도로 매칭되지 않는가
  → 매칭되면 그 유형은 "위기 고유"가 아니므로 의미 없음

### 검증 2: 부분 궤적 매칭 정확도

- 과거 BlackEvent를 이용한 Leave-One-Out 검증:
  ```
  1. BlackEvent 1건을 빼고 나머지 140건으로 유형 분류
  2. 빠진 1건의 D-30~D-14 부분 궤적으로 매칭 수행
  3. 매칭된 유형이 전체 궤적으로 분류한 유형과 일치하는가
  4. 예측된 phase가 실제 phase와 일치하는가
  5. 141번 반복하여 정확도 산출
  ```

### 검증 3: 시간순 분할 검증 (Module 7 연장)

- 학습: 2006~2019 BlackEvent로 유형 분류
- 테스트: 2020~2026 BlackEvent에 대해 매칭
- 테스트 기간의 BlackEvent가 올바른 유형으로 매칭되는가
- 매칭된 phase와 실제 D-day가 일관성 있는가

---

## 출력

### 파일 출력

| 파일 | 내용 |
|------|------|
| `trajectory_vectors.csv` | 141개 BlackEvent의 5시점 스코어 궤적 |
| `trajectory_dictionary.json` | 유형 사전 (centroid, std, member_events) |
| `trajectory_validation.csv` | Leave-One-Out 검증 결과 |
| `live_trajectory_match.json` | 라이브 진단 시 매칭 결과 |

### 라이브 진단 출력 (예시)

```
=== BlackEvent Predictor — Live Diagnosis ===
Date: 2026-02-22

[Module 6] Risk Score: 36.11 / 100  ⚠️ WARNING

[Module 8] Trajectory Analysis:
  Best Match: Type T001 (축적형) — Phase D-14
  Similarity: 0.87
  Interpretation: 현재 궤적은 과거 "축적형" 위기의 약 2주 전 시점과
                  가장 유사합니다. 이 유형의 과거 사례에서는 D-14 이후
                  평균 14~21일 내에 BlackEvent가 발생했습니다.

  Runner-up: Type T002 (급발진형) — Phase D-21
  Similarity: 0.65

  Historical Examples (T001):
    - 2008-09-15 (리먼 사태): D-14 시점 score 34.2
    - 2011-08-04 (유럽 재정위기): D-14 시점 score 29.8
    - 2022-06-13 (긴축 충격): D-14 시점 score 31.5
```

---

## 기술 사항

### 의존성
- numpy, pandas (궤적 벡터 연산)
- hdbscan (클러스터링, Module 3과 동일)
- scipy.spatial.distance (유사도 계산)
- scikit-learn (실루엣 스코어, 정규화)

### 계산 비용
- 궤적 역산: 141 BlackEvent × 5 시점 = 705회 스코어 산출
  - 각 스코어 산출은 뉴스 코퍼스에서 날짜 필터 + 빈도 집계
  - 예상 소요: 수 분 (Module 5보다 훨씬 가벼움)
- 클러스터링: 141개 × 5차원 → 1초 미만
- 대조군 궤적: 423 × 5 = 2,115회 스코어 산출 → 수 분 추가
- **총 예상: 10~20분**

### 코드 구조

```
src/module8_trajectory/
├── __init__.py
├── run.py                    # 메인 실행
├── trajectory_builder.py     # Step 8-1: 궤적 역산
├── trajectory_clustering.py  # Step 8-2: HDBSCAN 클러스터링
├── trajectory_matcher.py     # Step 8-3: 슬라이딩 매칭
└── trajectory_validator.py   # 검증 (LOO, 시간순 분할)
```

---

## 주의사항 및 리스크

### 데이터 리스크
- **N이 작다**: 141개 궤적을 클러스터링하면 유형당 샘플이 적을 수 있음
  - HDBSCAN의 min_cluster_size로 최소 품질 보장
  - 노이즈 포인트(분류 불가)가 많이 나올 수 있음 → 이 자체가 정보
- **5차원의 한계**: 시점이 5개뿐이므로 궤적의 세밀한 형태를 포착하기 어려움
  - 필요 시 시점 추가 가능 (D-45, D-3 등) → 단, 차원 증가에 따른 트레이드오프

### 분석 리스크
- **부분 궤적 매칭의 불확실성**: 아직 안 온 시점을 모르는 상태에서 매칭하므로 본질적 한계
  - 슬라이딩 매칭으로 모든 가설을 평가하되, 신뢰도 필터로 불확실한 결과를 솔직하게 표시
- **과적합**: 141개로 유형을 나누고, 같은 141개로 검증하면 순환논증
  - Leave-One-Out과 시간순 분할로 방지

### Anti-Bias Rule (재강조)
- 유형 수를 사전에 정하지 않는다
- 유형 이름(해석)이 분석 로직에 영향을 주지 않는다
- centroid의 "모양"을 보고 유형을 삭제하거나 병합하지 않는다 (데이터가 정한 것을 존중)
