import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fusion_weights_2 = nn.Parameter(torch.ones(2))
        self.fusion_weights_3 = nn.Parameter(torch.ones(3))

    def fusion(self, preds):
        """
        Args:
            preds(torch.Tensor): prediction from 3 channels: water_level, water_discharge, rainfall [B, L, 3]
        Returns:
            out(torch.Tensor): [B, L, 1]
        """
        assert self.fusion_weights_2.device == preds.device, "fusion_weights_2 and preds devices not match"
        assert self.fusion_weights_3.device == preds.device, "fusion_weights_3 and preds devices not match"
        if preds.shape[2] == 1:
            out = preds
        elif preds.shape[2] == 2:
            w_sm = F.softmax(self.fusion_weights_2, dim=0)
            out = (preds * w_sm.view(1, 1, 2)).sum(dim=2, keepdim=True)  # [B, L, 1]
        elif preds.shape[2] == 3:
            w_sm = F.softmax(self.fusion_weights_3, dim=0)
            out = (preds * w_sm.view(1, 1, 3)).sum(dim=2, keepdim=True)  # [B, L, 1]
        else:
            raise NotImplementedError
        return out

    def forward(self, preds):
        out = self.fusion(preds)
        return out
