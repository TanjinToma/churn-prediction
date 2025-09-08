#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:58:02 2025

@author: tanjintoma
"""

# src/sequential/data_prep_seq.py


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

# Custom Dataset
class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.lengths = [len(seq) for seq in sequences]
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]

# Collate function for DataLoader
def collate_fn(batch):
    sequences, lengths, labels = zip(*batch)

    # Pad sequences to max length in this batch
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    return padded_seqs, torch.tensor(lengths), torch.tensor(labels, dtype=torch.float)

# Main function (kept as preprocess_seq for compatibility)
def preprocess_seq(df, batch_size=32, test_size=0.2, random_state=42):
    """
    Prepares PyTorch DataLoaders for sequential churn dataset.

    Args:
        df (pd.DataFrame): DataFrame with sequence columns and churn label
        batch_size (int): batch size for DataLoader
        test_size (float): fraction for test split
        random_state (int): random seed

    Returns:
        train_loader, test_loader, input_dim
    """
    seq_features = []
    labels = []

    for _, row in df.iterrows():
        seq = np.vstack([
            row["seq_monthly_charges"],
            row["seq_data_usage"],
            row["seq_complaints"]
        ]).T  # shape = (timesteps, 3)

        seq_features.append(torch.tensor(seq, dtype=torch.float))
        labels.append(row["churn_label"])

    labels = torch.tensor(labels, dtype=torch.float)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        seq_features, labels, test_size=test_size,
        random_state=random_state, stratify=labels
    )

    # Wrap with Dataset
    train_dataset = SeqDataset(X_train, y_train)
    test_dataset = SeqDataset(X_test, y_test)

    # Dataloaders with custom collate
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    input_dim = 3  # (monthly_charges, data_usage, complaints)
    return train_loader, test_loader, input_dim
