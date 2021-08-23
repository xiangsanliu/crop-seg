import torch.nn as nn

import torch.nn.functional as F


class LabelSmoothingCrossEntropy2d(nn.Module):
    """
    Original implementation: fast.ai
    """
    def __init__(
        self, eps: float = 0.1, reduction="mean", weight=None, ignore_index=-100
    ):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)

    def forward(self, output, target):
        num_classes = output.size(1)
        log_preds = F.log_softmax(output, dim=1)
        if self.reduction == "sum":
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1)
            if self.reduction == "mean":
                loss = loss.mean()

        return loss * self.eps / num_classes + (1 - self.eps) * F.nll_loss(
            log_preds,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
