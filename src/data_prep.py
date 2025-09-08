#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 17:16:45 2025

@author: tanjintoma
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import torch
from torch.utils.data import TensorDataset, DataLoader


def preprocess_sklearn(df: pd.DataFrame):
    """
    Preprocess features for scikit-learn models (Logistic Regression, etc.).
    Steps:
    - Drop customer_id
    - Separate churn target (map Yes/No → 1/0)
    - Train/test split
    - Impute missing values (train only, apply to test)
    - One-hot encode categoricals (fit on train, align test)
    - Scale numeric features (fit on train, apply to test)
    Returns: X_train, X_test, y_train, y_test
    """
    # Drop non-predictive columns
    if "customer_id" in df.columns:
        df = df.drop("customer_id", axis=1)

    # Separate target
    y = df["churn"].map({"No": 0, "Yes": 1})
    X = df.drop("churn", axis=1)

    # Train/test split FIRST (avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split numeric & categorical
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object", "bool"]).columns

    # Impute numeric → median
    num_imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    # Impute categorical → most frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # One-hot encode categoricals (fit on train, align test)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def preprocess(df: pd.DataFrame, batch_size=32):
    """
    Preprocess features for PyTorch models.
    Same as preprocess_sklearn, but returns DataLoaders.
    Returns: train_loader, test_loader, input_dim
    """
    # Drop non-predictive columns
    if "customer_id" in df.columns:
        df = df.drop("customer_id", axis=1)

    # Separate target
    y = df["churn"].map({"No": 0, "Yes": 1})
    X = df.drop("churn", axis=1)

    # Train/test split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split numeric & categorical
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object", "bool"]).columns

    # Impute numeric → median
    num_imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    # Impute categorical → most frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # One-hot encode categoricals (fit on train, align test)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Wrap in Datasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train.shape[1]

