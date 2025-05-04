import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor([1.0, 1.0])
        else:
            self.alpha = alpha.clone().detach() if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=1)
        prob = torch.exp(log_prob)

        ce_loss = F.nll_loss(log_prob, target, reduction='none')
        pt = prob.gather(1, target.unsqueeze(1)).squeeze(1)

        at = self.alpha.to(input.device)[target]
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
