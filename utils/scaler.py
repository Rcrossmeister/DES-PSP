import numpy as np


class StandardScaler():
    def __init__(self, remove_invalid=False):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.std = np.std(data, axis=1, keepdims=True)
        self.std[self.std == 0] = 1

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean