import torch
from torch import nn

def IoUMetric(pred, gt, softmax=False):
    if softmax:
        pred = nn.Softmax(dim=1)(pred)
        
    gt = torch.cat([(gt == i )for i in range(3)], dim=1)

    intersection = gt * pred
    union = gt + pred - intersection

    iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)
    
    return iou.mean()

class IoULoss(nn.Module):
    def __init__(self, softmax=False):
        super().__init__()
        self.softmax = softmax
    
    def forward(self, pred, gt):
        return -(IoUMetric(pred, gt, self.softmax).log())