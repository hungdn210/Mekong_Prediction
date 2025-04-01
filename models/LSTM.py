import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.name = 'LSTM'
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers
        self.hidden_size = configs.d_model
        self.dropout = configs.dropout
        self.base_model = nn.LSTM(input_size=self.seq_len, hidden_size=self.hidden_size, batch_first=True,
                                  num_layers=self.e_layers, dropout=self.dropout)
        self.project = nn.Linear(self.hidden_size, self.pred_len)

    def forward(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)

        x_out = self.base_model(x_enc)
        if not isinstance(x_out, torch.Tensor):
            x_out, *_ = x_out
            x_out = self.project(x_out)

        x_out = x_out.permute(0, 2, 1)
        return x_out
