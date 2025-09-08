#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 11:24:35 2025

@author: tanjintoma
"""

# src/sequential/models_seq.py

import torch
import torch.nn as nn
import math

# ----------------------------
# 1. RNN Model
# ----------------------------
class RNNChurn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.3):
        super(RNNChurn, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        out = self.fc(h_n[-1])   # last hidden state
        return self.sigmoid(out).squeeze()


# ----------------------------
# 2. LSTM Model 
# ----------------------------
class LSTMChurn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.3):
        super(LSTMChurn, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        out = self.fc(h_n[-1])   # last hidden state
        return self.sigmoid(out).squeeze()


# ----------------------------
# 3. Transformer Model
# ----------------------------
class PositionalEncoding(nn.Module):
    """Injects positional information into the embeddings."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]



class TransformerChurn(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1,
                 nhead=2, dropout=0.5, max_len=500, noise_std=0.01):
        super(TransformerChurn, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout,
            batch_first=True, norm_first=True  # ensures LayerNorm stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # LayerNorm before classification
        self.ln = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Noise parameters
        self.noise_std = noise_std

    def forward(self, x, lengths):
        x = self.input_fc(x)
        x = self.pos_encoder(x)

        # Add Gaussian noise only during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Mask for padding
        max_len = x.size(1)
        mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]

        out = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Last valid timestep
        last_hidden = out[torch.arange(out.size(0)), lengths - 1]

        # Apply LayerNorm
        last_hidden = self.ln(last_hidden)

        out = self.fc(last_hidden)
        return self.sigmoid(out).squeeze()
