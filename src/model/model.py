"""
model.py
Model based on Li et al. (2023)
https://github.com/TJU-MHB/ChatGPT-sepsis-prediction/blob/master/BILSTM/module/model_bilstm.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ============================================================
# MODEL (BiLSTM)
# ============================================================
class MyLSTM(nn.Module):
    def __init__(self, dim_input, bilstm_input, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(dim_input, bilstm_input)
        self.bilstm = nn.LSTM(
            bilstm_input,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output = nn.Linear(hidden_dim * 2, 2)

    def forward(self, inputs):
        seqs, lengths = inputs

        x = torch.tanh(self.embedding(seqs))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        _, (hidden, _) = self.bilstm(packed)

        # last forward + backward hidden states
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        logits = self.output(last_hidden)
        return logits