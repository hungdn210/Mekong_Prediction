import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def encoder(self, x):
        x = x.permute(0, 2, 1)
        x = self.Linear(x)
        return x.permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.encoder(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
