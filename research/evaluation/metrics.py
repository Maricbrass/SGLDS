import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_metrics(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    def tpr_at(target):
        return tpr[np.argmin(np.abs(fpr - target))]

    return {
        "AUC": auc(fpr, tpr),
        "TPR@1e-2": tpr_at(1e-2),
        "TPR@1e-3": tpr_at(1e-3),
        "TPR@1e-4": tpr_at(1e-4),
    }
