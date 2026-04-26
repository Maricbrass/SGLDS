from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def classification_metrics(y_true, y_prob, pos_label: int = 1):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)

    def tpr_at(target_fpr: float) -> float:
        valid = np.where(fpr <= target_fpr)[0]
        if len(valid) == 0:
            return 0.0
        return float(np.max(tpr[valid]))

    return {
        "roc_auc": float(auc),
        "tpr_at_fpr_1e-2": tpr_at(1e-2),
        "tpr_at_fpr_1e-3": tpr_at(1e-3),
        "tpr_at_fpr_1e-4": tpr_at(1e-4),
    }
