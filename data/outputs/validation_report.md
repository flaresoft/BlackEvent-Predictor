# BlackEvent Predictor — Validation Report

## 데이터 분할
- **학습 데이터**: 457건
- **테스트 데이터**: 107건
- **분할 기준**: 학습 ≤ train_end, 테스트 > train_end (겹침 없음)

---

## Model A — 임계값 기반 (Threshold)

### Alert Level 최적화 (학습 데이터)

학습 데이터에서 F1 기준으로 최적 alert_level을 탐색한 결과:

| Alert Level | Precision | Recall | F1 | FPR | |
|:-----------:|:---------:|:------:|:--:|:---:|---|
| 1 | 0.2101 | 1.0000 | 0.3472 | 1.0000 | **← 최적**
| 2 | 0.2101 | 1.0000 | 0.3472 | 1.0000 |
| 3 | 0.2101 | 1.0000 | 0.3472 | 1.0000 |

**최적 alert_level: 1**

### 학습 데이터 성능 (alert_level=1)
| 지표 | 값 |
|------|-----|
| Precision | 0.2101 |
| Recall | 1.0000 |
| F1 Score | 0.3472 |
| False Positive Rate | 1.0000 |

### 테스트 데이터 성능 (alert_level=1)
| 지표 | 값 |
|------|-----|
| Precision | 0.4206 |
| Recall | 1.0000 |
| F1 Score | 0.5921 |
| False Positive Rate | 1.0000 |

### 혼동행렬 (테스트)
|  | Predicted Normal | Predicted BlackEvent |
|--|-----------------|---------------------|
| Actual Normal | 0 | 62 |
| Actual BlackEvent | 0 | 45 |

---

## Model B — 가중 스코어링 (Weighted Score)

### 학습 데이터 성능 (threshold 자동 탐색)
| 지표 | 값 |
|------|-----|
| AUC-ROC | 0.6855 |
| Best F1 | 0.4973 |
| Best Threshold | 18.52 |
| Precision | 0.5169 |
| Recall | 0.4792 |

### 테스트 데이터 성능 (학습 threshold=18.52 적용)
| 지표 | 값 |
|------|-----|
| AUC-ROC | 0.9072 |
| Threshold Used | 18.52 |
| Precision | 1.0000 |
| Recall | 0.5778 |
| F1 Score | 0.7324 |
| False Positive Rate | 0.0000 |

### 혼동행렬 (테스트)
|  | Predicted Normal | Predicted BlackEvent |
|--|-----------------|---------------------|
| Actual Normal | 62 | 0 |
| Actual BlackEvent | 19 | 26 |

---

## Lead Time 분석

경고 발령일에서 가장 가까운 다음 BlackEvent까지의 일수 (테스트 데이터).

### Model A (alert_level=1)
| 지표 | 값 |
|------|-----|
| 평균 Lead Time | 75.3일 |
| 중앙값 Lead Time | 98.0일 |
| 최소 | 0일 |
| 최대 | 188일 |
| 경고 발령 수 | 107건 |
| 미래 BE 매칭 | 107건 |

### Model B (threshold=18.52)
| 지표 | 값 |
|------|-----|
| 평균 Lead Time | 0.0일 |
| 중앙값 Lead Time | 0.0일 |
| 최소 | 0일 |
| 최대 | 0일 |
| 경고 발령 수 | 26건 |
| 미래 BE 매칭 | 26건 |

---

## 앙상블 (Model A + B 결합, 테스트 데이터)

### AND 앙상블 (둘 다 경고일 때만 최종 경고)
| 지표 | 값 |
|------|-----|
| Precision | 1.0000 |
| Recall | 0.5778 |
| F1 Score | 0.7324 |
| FPR | 0.0000 |

### OR 앙상블 (하나라도 경고면 최종 경고)
| 지표 | 값 |
|------|-----|
| Precision | 0.4206 |
| Recall | 1.0000 |
| F1 Score | 0.5921 |
| FPR | 1.0000 |

---

## 종합 비교 (테스트 데이터)

| 지표 | Model A | Model B | AND 앙상블 | OR 앙상블 |
|------|---------|---------|-----------|----------|
| Precision | 0.4206 | 1.0000 | 1.0000 | 0.4206 |
| Recall | 1.0000 | 0.5778 | 0.5778 | 1.0000 |
| F1 Score | 0.5921 | 0.7324 | 0.7324 | 0.5921 |
| FPR | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| Lead Time (평균) | 75.3일 | 0.0일 | - | - |

---

## 시각화 파일
- `confusion_matrix_model_a.png`
- `confusion_matrix_model_b.png`
- `roc_curve_model_b.png`
- `precision_recall_model_b.png`
