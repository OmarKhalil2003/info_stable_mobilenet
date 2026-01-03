import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# ============================================================
# Threshold Selection
# ============================================================
def compute_threshold(
    clean_scores: torch.Tensor,
    alpha: float = 0.05,
):
    """
    Computes detection threshold τ from clean data.

    Args:
        clean_scores: tensor of shape (N,)
        alpha: false rejection rate on clean data

    Returns:
        threshold τ
    """
    scores_np = clean_scores.detach().cpu().numpy()
    tau = np.quantile(scores_np, 1.0 - alpha)
    return tau


# ============================================================
# Apply Threshold
# ============================================================
def apply_threshold(
    scores: torch.Tensor,
    threshold: float,
):
    """
    Returns binary decision: 1 = reject, 0 = accept
    """
    return (scores > threshold).long()


# ============================================================
# ROC & AUROC
# ============================================================
def compute_roc_metrics(
    clean_scores: torch.Tensor,
    other_scores: torch.Tensor,
):
    """
    Computes ROC curve and AUROC between clean and other samples.

    Args:
        clean_scores: scores for clean samples
        other_scores: scores for adversarial or OOD samples

    Returns:
        fpr, tpr, auc
    """
    y_true = torch.cat([
        torch.zeros_like(clean_scores),
        torch.ones_like(other_scores),
    ])

    y_scores = torch.cat([clean_scores, other_scores])

    y_true = y_true.cpu().numpy()
    y_scores = y_scores.cpu().numpy()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    return fpr, tpr, auc


# ============================================================
# TPR @ FPR
# ============================================================
def tpr_at_fpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_fpr: float = 0.05,
):
    """
    Returns TPR at a given FPR level.
    """
    idx = np.searchsorted(fpr, target_fpr, side="right")
    idx = min(idx, len(tpr) - 1)
    return tpr[idx]
