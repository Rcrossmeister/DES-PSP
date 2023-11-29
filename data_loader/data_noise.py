import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.scaler import StandardScaler, MinMaxScaler

class Dataset_Noise(Dataset):

    def __init__(self, stock_num=3000, input_len=352, seed=88):
        np.random.seed(seed)
        # generate noise data, size = (1, 3000, 352)
        self.data = np.random.rand(1, stock_num, input_len)

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return self.data[:, :, :]
        
