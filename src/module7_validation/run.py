"""
Module 7 — Validation
모델의 실제 예측력을 시간순 분할 검증으로 평가한다.

주요 평가:
- Model A (Threshold): 학습에서 최적 alert_level 탐색 → 테스트 적용
- Model B (Weighted Score): 학습에서 최적 threshold 탐색 → 테스트 적용
- Lead Time: 경고에서 다음 BlackEvent까지 일수
- 앙상블 (AND/OR): Model A + B 결합
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.utils import load_config, get_path, setup_logging

logger = setup_logging("module7")


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """이진 분류 기본 지표를 계산한다."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ──────────────────────────────────────────────
# 시간순 분할
# ──────────────────────────────────────────────
def time_split(
    scores_df: pd.DataFrame, train_end: str, test_start: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시간순으로 학습/테스트 데이터를 분할한다.

    겹침 방지: test는 train_end 초과(>)로 필터한다.
    test_start 파라미터는 하위 호환을 위해 유지하나,
    실제 분할 기준은 train_end 하나만 사용한다.
    """
    scores_df["date"] = pd.to_datetime(scores_df["date"])
    train_end_ts = pd.Timestamp(train_end)

    train = scores_df[scores_df["date"] <= train_end_ts].copy()
    test = scores_df[scores_df["date"] > train_end_ts].copy()

    logger.info(
        f"데이터 분할: 학습 {len(train)}건 (≤ {train_end}) / "
        f"테스트 {len(test)}건 (> {train_end})"
    )
    return train, test


# ──────────────────────────────────────────────
# Model A 평가
# ──────────────────────────────────────────────
def evaluate_threshold_model(df: pd.DataFrame, alert_level: int = 2) -> dict:
    """Model A (임계값 기반) 평가. alert_level 이상이면 경고 발령."""
    y_true = df["actual_label"].values
    y_pred = (df["model_a_level"] >= alert_level).astype(int).values

    metrics = _compute_binary_metrics(y_true, y_pred)
    metrics["alert_level"] = alert_level
    return metrics


def find_optimal_alert_level(
    train_df: pd.DataFrame, levels: Optional[list[int]] = None
) -> tuple[int, dict[int, dict]]:
    """학습 데이터에서 F1 기준 최적 alert_level을 탐색한다.

    Args:
        train_df: 학습 데이터
        levels: 탐색할 alert_level 목록 (기본 [1, 2, 3])

    Returns:
        (best_level, {level: metrics_dict})
    """
    if levels is None:
        levels = [1, 2, 3]

    level_metrics: dict[int, dict] = {}
    for level in levels:
        level_metrics[level] = evaluate_threshold_model(train_df, alert_level=level)
        logger.debug(
            f"  alert_level={level}: F1={level_metrics[level]['f1']:.4f}, "
            f"P={level_metrics[level]['precision']:.4f}, "
            f"R={level_metrics[level]['recall']:.4f}"
        )

    best_level = max(levels, key=lambda lv: level_metrics[lv]["f1"])
    logger.info(
        f"최적 alert_level: {best_level} "
        f"(F1={level_metrics[best_level]['f1']:.4f})"
    )
    return best_level, level_metrics


# ──────────────────────────────────────────────
# Model B 평가
# ──────────────────────────────────────────────
def find_best_threshold_b(df: pd.DataFrame) -> tuple[float, dict]:
    """학습 데이터에서 F1 기준 최적 threshold를 탐색한다.

    학습 데이터의 precision-recall curve로 최적 threshold를 찾고,
    이를 테스트에 적용할 수 있도록 반환한다.

    Returns:
        (best_threshold, train_metrics_dict)
    """
    y_true = df["actual_label"].values
    scores = df["model_b_risk_score"].values

    if y_true.sum() == 0 or len(np.unique(y_true)) < 2:
        return 0.0, {
            "auc_roc": 0, "best_threshold": 0, "best_f1": 0,
            "precision": 0, "recall": 0, "fpr": 0, "f1": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        }

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.0

    # F1 기준 최적 threshold 탐색
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = f1_scores.argmax()
    best_threshold = (
        float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.0
    )
    best_f1 = float(f1_scores[best_idx])

    # 해당 threshold에서의 지표
    y_pred = (scores >= best_threshold).astype(int)
    metrics = _compute_binary_metrics(y_true, y_pred)
    metrics["auc_roc"] = round(auc, 4)
    metrics["best_threshold"] = round(best_threshold, 2)
    metrics["best_f1"] = round(best_f1, 4)

    logger.info(
        f"학습 최적 threshold: {best_threshold:.2f} "
        f"(F1={best_f1:.4f}, AUC={auc:.4f})"
    )
    return best_threshold, metrics


def evaluate_weighted_model(df: pd.DataFrame, threshold: float) -> dict:
    """주어진 threshold로 Model B를 평가한다 (테스트용, data leakage 없음).

    Args:
        df: 평가 데이터
        threshold: 학습 데이터에서 구한 최적 threshold
    """
    y_true = df["actual_label"].values
    scores = df["model_b_risk_score"].values

    try:
        auc = (
            roc_auc_score(y_true, scores)
            if len(np.unique(y_true)) > 1
            else 0.0
        )
    except ValueError:
        auc = 0.0

    y_pred = (scores >= threshold).astype(int)
    metrics = _compute_binary_metrics(y_true, y_pred)
    metrics["auc_roc"] = round(auc, 4)
    metrics["threshold_used"] = round(threshold, 2)
    return metrics


# ──────────────────────────────────────────────
# Lead Time 평가
# ──────────────────────────────────────────────
def compute_lead_time(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    all_be_dates: list[pd.Timestamp],
) -> dict:
    """경고 발령일에서 가장 가까운 다음 BlackEvent까지의 일수를 계산한다.

    Args:
        df: 평가 데이터 (date 컬럼 필요)
        y_pred: 예측 라벨 (1=경고)
        all_be_dates: 전체 BlackEvent 날짜 목록 (학습+테스트 모두 포함)

    Returns:
        lead time 통계 dict
    """
    if len(all_be_dates) == 0:
        return {
            "mean": None, "median": None,
            "n_warnings": 0, "n_with_future_be": 0,
        }

    warning_mask = y_pred == 1
    n_warnings = int(warning_mask.sum())

    if n_warnings == 0:
        return {
            "mean": None, "median": None,
            "n_warnings": 0, "n_with_future_be": 0,
        }

    warning_dates = pd.to_datetime(df.loc[warning_mask, "date"].values)
    be_sorted = sorted(pd.Timestamp(d) for d in all_be_dates)
    lead_times: list[int] = []

    for w_date in warning_dates:
        w_ts = pd.Timestamp(w_date)
        # 가장 가까운 미래(또는 당일) BlackEvent
        future_be = [d for d in be_sorted if d >= w_ts]
        if future_be:
            lead_days = (future_be[0] - w_ts).days
            lead_times.append(lead_days)

    if not lead_times:
        return {
            "mean": None, "median": None,
            "n_warnings": n_warnings, "n_with_future_be": 0,
        }

    return {
        "mean": round(float(np.mean(lead_times)), 1),
        "median": round(float(np.median(lead_times)), 1),
        "min": int(min(lead_times)),
        "max": int(max(lead_times)),
        "n_warnings": n_warnings,
        "n_with_future_be": len(lead_times),
    }


# ──────────────────────────────────────────────
# 앙상블 평가
# ──────────────────────────────────────────────
def evaluate_ensemble(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    mode: str = "and",
) -> dict:
    """Model A + B 앙상블 평가.

    Args:
        y_true: 실제 라벨
        y_pred_a: Model A 예측 (1=경고)
        y_pred_b: Model B 예측 (1=경고)
        mode: "and" (둘 다 경고일 때만) 또는 "or" (하나라도 경고면)
    """
    if mode == "and":
        y_pred = ((y_pred_a == 1) & (y_pred_b == 1)).astype(int)
    else:
        y_pred = ((y_pred_a == 1) | (y_pred_b == 1)).astype(int)

    metrics = _compute_binary_metrics(y_true, y_pred)
    metrics["mode"] = mode
    return metrics


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """혼동행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "BlackEvent"],
        yticklabels=["Normal", "BlackEvent"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """ROC 커브 시각화"""
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_precision_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Precision-Recall 커브 시각화"""
    precisions, recalls, _ = precision_recall_curve(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
# 리포트 생성
# ──────────────────────────────────────────────
def _fmt(v, fmt_str: str = ".4f") -> str:
    """리포트 출력용 포맷. None이면 '-'을 반환."""
    if v is None:
        return "-"
    return f"{v:{fmt_str}}"


def generate_report(
    # Model A
    train_metrics_a: dict,
    test_metrics_a: dict,
    optimal_alert_level: int,
    alert_level_comparison: dict[int, dict],
    # Model B
    train_metrics_b: dict,
    test_metrics_b: dict,
    train_threshold_b: float,
    # Lead time
    lead_time_a: dict,
    lead_time_b: dict,
    # Ensemble
    ensemble_and: dict,
    ensemble_or: dict,
    # General
    train_size: int,
    test_size: int,
    outputs_path: Path,
) -> None:
    """종합 검증 리포트를 생성한다."""

    # ── Alert Level 비교표 ──
    level_rows = ""
    for level in sorted(alert_level_comparison.keys()):
        m = alert_level_comparison[level]
        marker = " **← 최적**" if level == optimal_alert_level else ""
        level_rows += (
            f"| {level} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m['fpr']:.4f} |{marker}\n"
        )

    # ── Lead Time 포맷 ──
    def _lead_time_section(lt: dict, model_name: str) -> str:
        if lt.get("mean") is None:
            return f"경고 발령 없음 또는 미래 BlackEvent 없음\n"
        return (
            f"| 평균 Lead Time | {lt['mean']:.1f}일 |\n"
            f"| 중앙값 Lead Time | {lt['median']:.1f}일 |\n"
            f"| 최소 | {lt['min']}일 |\n"
            f"| 최대 | {lt['max']}일 |\n"
            f"| 경고 발령 수 | {lt['n_warnings']}건 |\n"
            f"| 미래 BE 매칭 | {lt['n_with_future_be']}건 |\n"
        )

    # ── Model B 테스트 threshold 표기 ──
    test_b_threshold_label = test_metrics_b.get(
        "threshold_used", train_threshold_b
    )

    report = f"""# BlackEvent Predictor — Validation Report

## 데이터 분할
- **학습 데이터**: {train_size}건
- **테스트 데이터**: {test_size}건
- **분할 기준**: 학습 ≤ train_end, 테스트 > train_end (겹침 없음)

---

## Model A — 임계값 기반 (Threshold)

### Alert Level 최적화 (학습 데이터)

학습 데이터에서 F1 기준으로 최적 alert_level을 탐색한 결과:

| Alert Level | Precision | Recall | F1 | FPR | |
|:-----------:|:---------:|:------:|:--:|:---:|---|
{level_rows}
**최적 alert_level: {optimal_alert_level}**

### 학습 데이터 성능 (alert_level={optimal_alert_level})
| 지표 | 값 |
|------|-----|
| Precision | {train_metrics_a['precision']:.4f} |
| Recall | {train_metrics_a['recall']:.4f} |
| F1 Score | {train_metrics_a['f1']:.4f} |
| False Positive Rate | {train_metrics_a['fpr']:.4f} |

### 테스트 데이터 성능 (alert_level={optimal_alert_level})
| 지표 | 값 |
|------|-----|
| Precision | {test_metrics_a['precision']:.4f} |
| Recall | {test_metrics_a['recall']:.4f} |
| F1 Score | {test_metrics_a['f1']:.4f} |
| False Positive Rate | {test_metrics_a['fpr']:.4f} |

### 혼동행렬 (테스트)
|  | Predicted Normal | Predicted BlackEvent |
|--|-----------------|---------------------|
| Actual Normal | {test_metrics_a.get('tn', '-')} | {test_metrics_a.get('fp', '-')} |
| Actual BlackEvent | {test_metrics_a.get('fn', '-')} | {test_metrics_a.get('tp', '-')} |

---

## Model B — 가중 스코어링 (Weighted Score)

### 학습 데이터 성능 (threshold 자동 탐색)
| 지표 | 값 |
|------|-----|
| AUC-ROC | {train_metrics_b['auc_roc']:.4f} |
| Best F1 | {train_metrics_b['best_f1']:.4f} |
| Best Threshold | {train_metrics_b['best_threshold']:.2f} |
| Precision | {train_metrics_b['precision']:.4f} |
| Recall | {train_metrics_b['recall']:.4f} |

### 테스트 데이터 성능 (학습 threshold={test_b_threshold_label} 적용)
| 지표 | 값 |
|------|-----|
| AUC-ROC | {test_metrics_b['auc_roc']:.4f} |
| Threshold Used | {test_b_threshold_label} |
| Precision | {test_metrics_b['precision']:.4f} |
| Recall | {test_metrics_b['recall']:.4f} |
| F1 Score | {test_metrics_b['f1']:.4f} |
| False Positive Rate | {test_metrics_b['fpr']:.4f} |

### 혼동행렬 (테스트)
|  | Predicted Normal | Predicted BlackEvent |
|--|-----------------|---------------------|
| Actual Normal | {test_metrics_b.get('tn', '-')} | {test_metrics_b.get('fp', '-')} |
| Actual BlackEvent | {test_metrics_b.get('fn', '-')} | {test_metrics_b.get('tp', '-')} |

---

## Lead Time 분석

경고 발령일에서 가장 가까운 다음 BlackEvent까지의 일수 (테스트 데이터).

### Model A (alert_level={optimal_alert_level})
| 지표 | 값 |
|------|-----|
{_lead_time_section(lead_time_a, "A")}
### Model B (threshold={test_b_threshold_label})
| 지표 | 값 |
|------|-----|
{_lead_time_section(lead_time_b, "B")}
---

## 앙상블 (Model A + B 결합, 테스트 데이터)

### AND 앙상블 (둘 다 경고일 때만 최종 경고)
| 지표 | 값 |
|------|-----|
| Precision | {ensemble_and['precision']:.4f} |
| Recall | {ensemble_and['recall']:.4f} |
| F1 Score | {ensemble_and['f1']:.4f} |
| FPR | {ensemble_and['fpr']:.4f} |

### OR 앙상블 (하나라도 경고면 최종 경고)
| 지표 | 값 |
|------|-----|
| Precision | {ensemble_or['precision']:.4f} |
| Recall | {ensemble_or['recall']:.4f} |
| F1 Score | {ensemble_or['f1']:.4f} |
| FPR | {ensemble_or['fpr']:.4f} |

---

## 종합 비교 (테스트 데이터)

| 지표 | Model A | Model B | AND 앙상블 | OR 앙상블 |
|------|---------|---------|-----------|----------|
| Precision | {test_metrics_a['precision']:.4f} | {test_metrics_b['precision']:.4f} | {ensemble_and['precision']:.4f} | {ensemble_or['precision']:.4f} |
| Recall | {test_metrics_a['recall']:.4f} | {test_metrics_b['recall']:.4f} | {ensemble_and['recall']:.4f} | {ensemble_or['recall']:.4f} |
| F1 Score | {test_metrics_a['f1']:.4f} | {test_metrics_b['f1']:.4f} | {ensemble_and['f1']:.4f} | {ensemble_or['f1']:.4f} |
| FPR | {test_metrics_a['fpr']:.4f} | {test_metrics_b['fpr']:.4f} | {ensemble_and['fpr']:.4f} | {ensemble_or['fpr']:.4f} |
| Lead Time (평균) | {_fmt(lead_time_a.get('mean'), '.1f')}일 | {_fmt(lead_time_b.get('mean'), '.1f')}일 | - | - |

---

## 시각화 파일
- `confusion_matrix_model_a.png`
- `confusion_matrix_model_b.png`
- `roc_curve_model_b.png`
- `precision_recall_model_b.png`
"""

    report_path = outputs_path / "validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"리포트 저장: {report_path}")


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def run() -> None:
    """Module 7 전체 파이프라인 실행"""
    config = load_config()
    outputs_path = get_path(config, "outputs")
    val_config = config["validation"]

    # 리스크 스코어 로드
    scores_df = pd.read_csv(outputs_path / "risk_scores.csv")
    logger.info(f"리스크 스코어 로드: {len(scores_df)}건")

    # 시간순 분할 (겹침 방지: test > train_end)
    train_df, test_df = time_split(
        scores_df,
        train_end=val_config["train_end"],
        test_start=val_config["test_start"],
    )

    # 전체 BlackEvent 날짜 (lead time 계산용)
    all_be_dates: list[pd.Timestamp] = (
        scores_df.loc[scores_df["actual_label"] == 1, "date"]
        .apply(pd.Timestamp)
        .tolist()
    )

    # ══════════════════════════════════════════
    # Model A 평가
    # ══════════════════════════════════════════
    logger.info("--- Model A 평가 ---")

    # 학습: alert_level 최적화
    optimal_alert_level, alert_level_comparison = find_optimal_alert_level(
        train_df, levels=[1, 2, 3]
    )
    train_metrics_a = alert_level_comparison[optimal_alert_level]

    # 테스트: 학습에서 구한 최적 level 적용
    test_metrics_a = evaluate_threshold_model(
        test_df, alert_level=optimal_alert_level
    )
    logger.info(f"  학습 (level={optimal_alert_level}): {train_metrics_a}")
    logger.info(f"  테스트 (level={optimal_alert_level}): {test_metrics_a}")

    # Model A 혼동행렬
    y_true_test = test_df["actual_label"].values
    y_pred_a = (
        (test_df["model_a_level"] >= optimal_alert_level).astype(int).values
    )
    if y_true_test.sum() > 0:
        plot_confusion_matrix(
            y_true_test, y_pred_a,
            f"Model A Confusion Matrix (Test, level≥{optimal_alert_level})",
            outputs_path / "confusion_matrix_model_a.png",
        )

    # ══════════════════════════════════════════
    # Model B 평가
    # ══════════════════════════════════════════
    logger.info("--- Model B 평가 ---")

    # 학습: threshold 탐색 (여기서만 threshold 결정)
    train_threshold_b, train_metrics_b = find_best_threshold_b(train_df)

    # 테스트: 학습 threshold 적용 (data leakage 없음)
    test_metrics_b = evaluate_weighted_model(test_df, threshold=train_threshold_b)
    logger.info(f"  학습: {train_metrics_b}")
    logger.info(f"  테스트 (threshold={train_threshold_b:.2f}): {test_metrics_b}")

    # Model B 시각화
    if y_true_test.sum() > 0 and len(np.unique(y_true_test)) > 1:
        scores_test = test_df["model_b_risk_score"].values
        y_pred_b = (scores_test >= train_threshold_b).astype(int)

        plot_confusion_matrix(
            y_true_test, y_pred_b,
            f"Model B Confusion Matrix (Test, threshold={train_threshold_b:.2f})",
            outputs_path / "confusion_matrix_model_b.png",
        )
        plot_roc_curve(
            y_true_test, scores_test,
            "Model B ROC Curve (Test)",
            outputs_path / "roc_curve_model_b.png",
        )
        plot_precision_recall(
            y_true_test, scores_test,
            "Model B Precision-Recall Curve (Test)",
            outputs_path / "precision_recall_model_b.png",
        )
    else:
        y_pred_b = np.zeros(len(y_true_test), dtype=int)

    # ══════════════════════════════════════════
    # Lead Time 평가
    # ══════════════════════════════════════════
    logger.info("--- Lead Time 평가 ---")
    lead_time_a = compute_lead_time(test_df, y_pred_a, all_be_dates)
    lead_time_b = compute_lead_time(test_df, y_pred_b, all_be_dates)
    logger.info(f"  Model A: {lead_time_a}")
    logger.info(f"  Model B: {lead_time_b}")

    # ══════════════════════════════════════════
    # 앙상블 평가
    # ══════════════════════════════════════════
    logger.info("--- 앙상블 평가 ---")
    ensemble_and = evaluate_ensemble(y_true_test, y_pred_a, y_pred_b, mode="and")
    ensemble_or = evaluate_ensemble(y_true_test, y_pred_a, y_pred_b, mode="or")
    logger.info(f"  AND 앙상블: {ensemble_and}")
    logger.info(f"  OR 앙상블: {ensemble_or}")

    # ══════════════════════════════════════════
    # 리포트 생성
    # ══════════════════════════════════════════
    generate_report(
        train_metrics_a=train_metrics_a,
        test_metrics_a=test_metrics_a,
        optimal_alert_level=optimal_alert_level,
        alert_level_comparison=alert_level_comparison,
        train_metrics_b=train_metrics_b,
        test_metrics_b=test_metrics_b,
        train_threshold_b=train_threshold_b,
        lead_time_a=lead_time_a,
        lead_time_b=lead_time_b,
        ensemble_and=ensemble_and,
        ensemble_or=ensemble_or,
        train_size=len(train_df),
        test_size=len(test_df),
        outputs_path=outputs_path,
    )

    logger.info("=== Module 7 완료 ===")


if __name__ == "__main__":
    run()
