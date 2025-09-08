#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 14:42:46 2025

@author: tanjintoma
"""

# src/sequential/evaluate_seq.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

def evaluate_seq(model, test_loader, device, model_name="Model"):
    model.eval()
    y_true, y_pred, y_proba = [], [], []

    with torch.no_grad():
        for X_batch, lengths, y_batch in test_loader:
            X_batch, lengths, y_batch = (
                X_batch.to(device),
                lengths.to(device),
                y_batch.to(device),
            )
            outputs = model(X_batch, lengths)
            preds = (outputs > 0.5).int()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(outputs.cpu().numpy())

    # ---------------- Classification Report ----------------
    print("=" * 60)
    print(f" Classification Report for {model_name} ")
    print("=" * 60)
    print(classification_report(y_true, y_pred))

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ---------------- ROC Curve ----------------
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.show()

    return auc
