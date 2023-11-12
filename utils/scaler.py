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

class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = np.min(data, axis=1, keepdims=True)
        self.max = np.max(data, axis=1, keepdims=True)

    def transform(self, data):
        # 防止除以零
        range = self.max - self.min
        range[range == 0] = 1
        return (data - self.min) / range

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min