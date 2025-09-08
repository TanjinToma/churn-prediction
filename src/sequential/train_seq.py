#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:54:56 2025

@author: tanjintoma
"""

# src/sequential/train_seq.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.sequential.evaluate_seq import evaluate_seq
import numpy as np
import random

def set_seed(seed):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model_seq(
    model, train_loader, val_loader,
    epochs=30, lr=0.001, device="cpu", patience=5
):
    """
    Train model for sequential churn prediction with early stopping.
    """
    # Reproducibility
    set_seed(seed=5)
    
    criterion = nn.BCELoss()
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ------------------ Training ------------------
        model.train()
        running_loss = 0.0

        for X_batch, lengths, y_batch in train_loader:
            X_batch, lengths, y_batch = (
                X_batch.to(device),
                lengths.to(device),
                y_batch.to(device),
            )

            optimizer.zero_grad()
            outputs = model(X_batch, lengths)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ------------------ Validation ------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, lengths, y_batch in val_loader:
                X_batch, lengths, y_batch = (
                    X_batch.to(device),
                    lengths.to(device),
                    y_batch.to(device),
                )
                outputs = model(X_batch, lengths)
                loss = criterion(outputs, y_batch.float())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # ------------------ Early Stopping ------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Restore best model
    model.load_state_dict(best_model_state)

    # ------------------ Plot losses ------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.show()

    return model
