import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    auc,
    make_scorer,
    roc_auc_score,
    SCORERS as SCORERS_SKLEARN
)


def pr_auc_score(y_true, y_score):
    """  Area under the Precision-Recall curve """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


# Scorer for Area under the Precision-Recall curve
pr_auc_scorer = make_scorer(pr_auc_score, greater_is_better=True, needs_proba=True)


SCORERS = {
    'roc_auc': SCORERS_SKLEARN['roc_auc'],
    'pr_auc': pr_auc_scorer,
}


if __name__ == "__main__":
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    pred_y = np.array([0, 0.03, 0.09, 0.33, 0.05, 0, 0.25, 0.66])
    print(average_precision_score(y, pred_y))
    print(pr_auc_score(y, pred_y))
    print(roc_auc_score(y, pred_y))
