"""Metrics written by hand so the CI environment does not need scikit-learn."""

import numpy as np

from ml.data import ACTIVITIES


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 6) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    report = {}
    for index, name in enumerate(ACTIVITIES):
        true_positive = float(matrix[index, index])
        predicted, actual = float(matrix[:, index].sum()), float(matrix[index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(actual)}
    return report


def _macro_f1(report: dict[str, dict[str, float]]) -> float:
    return float(np.mean([entry["f1"] for entry in report.values()]))


def summarize(y_true: np.ndarray, y_pred: np.ndarray, subject: np.ndarray) -> dict:
    matrix = confusion(y_true, y_pred)
    report = per_class(matrix)
    by_subject = {}
    for person in np.unique(subject).tolist():
        mask = subject == person
        subject_report = per_class(confusion(y_true[mask], y_pred[mask]))
        by_subject[str(person)] = {
            "windows": int(mask.sum()),
            "accuracy": float((y_true[mask] == y_pred[mask]).mean()),
            "macro_f1": _macro_f1(subject_report),
        }
    accuracies = np.array([entry["accuracy"] for entry in by_subject.values()])
    return {
        "windows": len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": _macro_f1(report),
        "per_class": report,
        "confusion": matrix.tolist(),
        "per_subject": by_subject,
        "subject_accuracy_mean": float(accuracies.mean()),
        "subject_accuracy_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
        "subject_accuracy_min": float(accuracies.min()),
        "subject_accuracy_max": float(accuracies.max()),
    }
