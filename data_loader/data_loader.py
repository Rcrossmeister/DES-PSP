import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import init_logger, prepare_data, remove_invalid_stocks
from utils.scaler import StandardScaler

class Dataset_Stock_Price(Dataset):
    def __init__(self, root_path='df_path', data_path='All_Data.csv', 
                 start_date='2015/11/09', end_date='2016/11/08', pred_len=14,
                 remove_invalid=False, flag='train', scale=True, inverse=False):
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.start_date = start_date
        self.end_date = end_date
        self.pred_len = pred_len
        self.scale = scale
        self.inverse = inverse
        self.remove_invalid = remove_invalid
        self.root_path = root_path
        self.data_path = data_path
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_path = os.path.join(self.root_path, self.data_path)
        self.input_data, self.target_data = prepare_data(df_path, self.start_date, self.end_date, self.pred_len)
        if self.remove_invalid:
            self.input_data, self.target_data = remove_invalid_stocks(self.input_data, self.target_data)
        if self.scale:
            self.input_scaler.fit(self.input_data)
            self.target_scaler.fit(self.target_data)
            self.input_data = self.input_scaler.transform(self.input_data)
            self.target_data = self.target_scaler.transform(self.target_data)
        self.input_data = self.input_data.reshape(self.input_data.shape[0], self.input_data.shape[1], 1)
        self.target_data = self.target_data.reshape(self.target_data.shape[0], self.target_data.shape[1], 1)
        self.input_data = self.input_data.astype(np.float32)
        self.target_data = self.target_data.astype(np.float32)
    
    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        input_seq = self.input_data[index,:,:]
        target_seq = self.target_data[index,:,:]
        return input_seq, target_seq
    
if __name__ == '__main__':
    dataset_stock = Dataset_Stock_Price()
    data_loader = DataLoader(
            dataset_stock,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            drop_last=True)
    for i, (input_seq, target_seq) in enumerate(data_loader):
        print(input_seq.shape)
        print(target_seq.shape)


class Dataset_Stock_Movement(Dataset):

    def __init__(self, root_path='df_path', data_path='All_Data.csv',
                 start_date='2015/11/09', end_date='2016/11/08', pred_len=14,
                 remove_invalid=False, flag='train', scale=True, inverse=False):
        pass