# BlackEvent Predictor

KOSPI 급락(BlackEvent)을 뉴스의 **구조적 성질** 빈도 변화로 예측하는 시스템.

"어떤 사건이 원인인가"를 사람이 정하지 않고, 뉴스가 가진 **구조적 패턴**을 데이터가 스스로 찾게 한다.

## 핵심 아이디어

1. KOSPI 일별 -2% 이상 급락일을 **BlackEvent**로 정의 (2006~2026, 141건)
2. 4,650만 건의 뉴스 기사에서 LLM(Claude)으로 **113개 구조적 성질**을 비지도 추출
3. BlackEvent 직전 vs 평상시의 성질 빈도를 통계 비교 → **12개 유의미 성질** 발견
4. 가중 빈도 스코어링으로 실시간 리스크 점수 산출

## 성능 (시간순 분할 검증)

| 지표 | Model B (가중 빈도) |
|------|:---:|
| **AUC-ROC** | **0.9072** |
| Precision | 1.0000 |
| Recall | 0.5778 |
| F1 Score | 0.7324 |
| False Positive Rate | 0.0000 |

> 경고 발령 시 **100% 정밀도** (오경보 0건). 전체 BlackEvent 중 약 58%를 사전 감지.

## 아키텍처

```
[KOSPI 데이터] → Module 1: BlackEvent 정의 (141건 추출)
                         ↓
[뉴스 4,650만건] → Module 2: 코퍼스 구축
                         ↓
                 Module 3: 성질 사전 생성
                   (샘플 5,000건 → Claude 오픈형 태깅 → 임베딩 클러스터링 → 113개 성질)
                         ↓
                 Module 4: 전체 코퍼스 태깅
                   (KURE 임베딩 코사인 유사도 매칭 — LLM 미사용, GPU 로컬 처리)
                         ↓
                 Module 5: 통계 분석
                   (Mann-Whitney U, FDR 보정 → 12개 유의미 성질)
                         ↓
                 Module 6: 리스크 스코어링
                   (가중 빈도 모델: score = Σ(weight × excess_ratio))
                         ↓
                 Module 7: 검증 (시간순 분할, ROC, PR, 선행시간)
                         ↓
                 Module 8: 위기 궤적 분류 (HDBSCAN 클러스터링)
```

## 유의미 성질 12개

| ID | 성질 | p (FDR 보정) |
|----|------|:---:|
| Property_078 | 구조적 불안정성의 확산 | 3.37e-10 |
| Property_055 | 수요와 공급의 불균형 | 3.55e-09 |
| Property_070 | 금융 규제와 정책 변화 | 7.58e-09 |
| Property_024 | 외부 요인에 의한 시장 반응 | 2.76e-08 |
| Property_013 | 금융기관 간 신뢰 붕괴 | 1.45e-06 |
| Property_113 | 위기 확산의 비선형적 특성 | 1.86e-05 |
| Property_049 | 글로벌 금융 시스템의 상호 연결성 | 7.25e-07 |
| Property_110 | 시장 참여자의 과도한 레버리지 | 4.43e-04 |
| Property_001 | 거시경제 지표의 급변 | 2.08e-03 |
| Property_023 | 자산 가격의 급격한 변동 | 1.89e-03 |
| Property_068 | 투자자 심리의 급격한 변화 | 1.87e-02 |
| Property_030 | 정부 개입과 시장 반응 | 4.51e-02 |

## 설치

```bash
git clone https://github.com/flaresoft/BlackEvent-Predictor.git
cd BlackEvent-Predictor
pip install -r requirements.txt
```

### 요구사항
- Python 3.11+
- CUDA GPU (태깅용, RTX 4060 Ti 16GB에서 테스트)
- BigKinds 계정 (뉴스 수집용)

### 환경변수

```bash
cp .env.example .env
# .env 파일에 API 키 입력
```

| 변수 | 용도 | 필요 시점 |
|------|------|----------|
| `ANTHROPIC_API_KEY` | Claude API (성질 사전 생성) | Module 3 실행 시에만 |
| `NAVER_CLIENT_ID/SECRET` | 네이버 뉴스 API (보조) | 선택 |

## 사용법

### 일일 파이프라인 (수집 → 태깅 → 스코어링)

```bash
# 어제 뉴스 수집 + 태깅 + 스코어링
python -m src.daily_pipeline.run

# 특정 날짜
python -m src.daily_pipeline.run --date 2026-02-27

# 스코어링만 (수집/태깅 스킵)
python -m src.daily_pipeline.run --score-only
```

### 웹 대시보드 (Streamlit)

```bash
streamlit run web/app.py
```

- **Dashboard**: 리스크 게이지, 성질 기여도, 과거 유사 사례 매칭, 일별 추이
- **Collect**: BigKinds 뉴스 수집 + 태깅/스코어링
- **History**: 스코어 이력 조회

### 개별 모듈 실행

```bash
python -m src.module1_blackevent.run    # BlackEvent 추출
python -m src.module2_corpus.run        # 코퍼스 구축
python -m src.module3_dictionary.run    # 성질 사전 생성 (Claude API 필요)
python -m src.module4_tagging.run       # 전체 코퍼스 태깅 (GPU)
python -m src.module5_analysis.run      # 통계 분석
python -m src.module6_scoring.run       # 스코어링
python -m src.module7_validation.run    # 검증
python -m src.module8_trajectory.run    # 궤적 분류
```

## 프로젝트 구조

```
├── config/
│   └── settings.yaml               # 전체 설정
├── data/
│   ├── processed/
│   │   ├── black_events.csv         # 141개 BlackEvent 목록 (포함)
│   │   └── control_dates.csv        # 423개 대조군 날짜 (포함)
│   └── outputs/
│       ├── property_dictionary.json # 113개 성질 사전
│       ├── significant_properties.csv # 유의미 성질 12개 + 가중치
│       ├── threshold_calibration.json # 태깅 임계값
│       ├── risk_scores.csv          # 전체 스코어링 결과
│       ├── be_profiles.json         # BE 성질 프로필 (진단용)
│       └── *.png                    # 검증 시각화
├── src/
│   ├── module1_blackevent/          # BlackEvent 정의
│   ├── module2_corpus/              # 코퍼스 구축
│   ├── module3_dictionary/          # 성질 사전 생성 (Claude API)
│   ├── module4_tagging/             # 전체 코퍼스 태깅 (GPU)
│   ├── module5_analysis/            # 통계 분석
│   ├── module6_scoring/             # 리스크 스코어링
│   ├── module7_validation/          # 검증
│   ├── module8_trajectory/          # 궤적 분류
│   ├── daily_pipeline/              # 일일 자동화
│   └── diagnosis/                   # 상태 진단
└── web/                             # Streamlit 대시보드
```

## 데이터 안내

### 포함된 데이터
- `black_events.csv` — KOSPI 급락 이벤트 141건 (Yahoo Finance 공개 데이터 기반)
- `control_dates.csv` — 대조군 날짜 423건
- `property_dictionary.json` — 113개 구조적 성질 사전
- `significant_properties.csv` — 유의미 성질 12개 + 통계량/가중치
- `threshold_calibration.json` — 태깅 코사인 유사도 임계값
- `risk_scores.csv` — 전체 BlackEvent/대조군 스코어링 결과
- `be_profiles.json` — BlackEvent별 성질 프로필 (상태 진단용)
- 검증 시각화 (ROC, PR, 혼동행렬)

### 포함되지 않은 데이터 (저작권)
- **뉴스 코퍼스** (`corpus_tagged.parquet`, 8.5GB) — BigKinds 뉴스 원문. 직접 수집 필요.
- **원본 xlsx** (`data/raw/`) — BigKinds 다운로드 파일

뉴스 데이터는 [BigKinds](https://www.bigkinds.or.kr/)에서 무료 계정으로 직접 수집할 수 있습니다.

## 스코어링 모델 (Model B)

```
Risk Score = Σ(weight_i × max(freq_ratio_i - 1, 0)) × 100 / empirical_max

weight_i    = -log10(p_adjusted) 정규화 (유의성 기반)
freq_ratio  = 현재 빈도 / 대조군 평균 빈도
```

- 각 성질의 **최적 관측 윈도우** (14~45일)를 개별 탐색
- 빈도비가 1.0 (baseline)을 초과하는 만큼만 기여
- BlackEvent 관측 최대치 기준으로 0~100 정규화

## 기술 스택

| 구분 | 기술 |
|------|------|
| 임베딩 | [KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1) (한국어 특화) |
| LLM | Claude API (Module 3 사전 생성에만 사용) |
| 클러스터링 | HDBSCAN |
| 통계 | Mann-Whitney U, Benjamini-Hochberg FDR |
| 프론트엔드 | Streamlit + Plotly |
| 데이터 | PyArrow Parquet (row-group 단위 메모리 최적화) |

## 라이선스

MIT License

## 저자

**flaresoft** — [GitHub](https://github.com/flaresoft)
