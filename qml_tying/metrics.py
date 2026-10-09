import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, confusion_matrix)
from .core import probability, loss


def classify_metrics(y, f):
    p = probability(f); pred = (p >= .5).astype(int)
    order = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(order == b) * abs(np.mean(y[order == b]) - np.mean(p[order == b]))
              for b in range(10) if np.any(order == b))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                f1_macro=f1_score(y, pred, average='macro', zero_division=0),
                roc_auc=roc_auc_score(y, p) if len(np.unique(y))>1 else None,
                average_precision=average_precision_score(y, p) if len(np.unique(y))>1 else None,
                brier=brier_score_loss(y, p), log_loss=loss(f, y), ece_10=float(ece),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
