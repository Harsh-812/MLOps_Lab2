import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the Wine dataset and return features (X) and targets (y).
    Returns:
        X (numpy.ndarray): Wine features.
        y (numpy.ndarray): Wine class labels (0..2).
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y

def split_data(X, y):
    """
    Train/test split (stratified) for Wine dataset.
    """
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
