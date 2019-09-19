import torch.nn.functional as F
import torch
from torch import nn


class FocalLoss(torch.nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, target):
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        neg_weights = torch.pow(1 - target, 4)
        loss = 0
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class MaskedLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self._loss = loss

    def forward(self, input, target, mask):
        mask = mask.unsqueeze(1).expand_as(input)
        return self._loss(input*mask, target*mask)


class RegL1Loss(torch.nn.Module):
    def forward(self, output, target, mask):
        mask = mask.unsqueeze(1).expand_as(target).float()
        loss = F.l1_loss(output * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss
