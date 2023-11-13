import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import init_logger, prepare_data, remove_invalid_stocks, remove_zeros_row
from utils.scaler import StandardScaler, MinMaxScaler


class Dataset_Stock(Dataset):
    def __init__(self, root_path='df_path', data_path='All_Data.csv', target='price',
                 start_date='2015/11/09', end_date='2016/11/08', pred_len=14,
                 remove_invalid=False, flag='train', scale=True, inverse=False):
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.flag = flag
        self.set_type = type_map[flag]
        self.target = target
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
        self.input_data, self.target_data = prepare_data(df_path, self.start_date, self.end_date, self.pred_len, self.target)
        # self.input_data, self.target_data = remove_zeros_row(self.input_data, self.target_data)
        if self.remove_invalid:
            self.input_data, self.target_data = remove_invalid_stocks(self.input_data, self.target_data)
        if self.scale:
            self.input_scaler.fit(self.input_data)
            self.input_data = self.input_scaler.transform(self.input_data)
            if self.target == 'price':
                self.target_scaler.fit(self.target_data)
                self.target_data = self.target_scaler.transform(self.target_data)

        self.input_data = self.input_data.reshape(self.input_data.shape[0], self.input_data.shape[1], 1)
        self.target_data = self.target_data.reshape(self.target_data.shape[0], self.target_data.shape[1], 1)
        self.input_data = self.input_data.astype(np.float32)
        self.target_data = self.target_data.astype(np.float32)
    
    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        input_seq = self.input_data[index, :, :]   # [index, pred_len, feature=1]
        target_seq = self.target_data[index, :, :]
        return input_seq, target_seq # [batch_size, pred_len, feature=1]


class DataSet_Competitor(Dataset):
    def __init__(self, root_path='df_path', data_path='GroupC.csv', target='price',
                 start_date='2015/11/09', end_date='2016/11/08', pred_len=14,
                 remove_invalid=False, flag='train', scale=True, inverse=False):
        self.pred_len = pred_len
        self.target = target
        self.scale = scale
        self.remove_invalid = remove_invalid
        self.start_date = start_date
        self.end_date = end_date
        self.root_path = root_path
        self.data_path = data_path
        self.input_scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_path = os.path.join(self.root_path, self.data_path)
        self.input_data, self.target_data = prepare_data(df_path, self.start_date, self.end_date, self.pred_len,
                                                         self.target)
        if self.remove_invalid:
            self.input_data, self.target_data = remove_invalid_stocks(self.input_data, self.target_data)
        if self.scale:
            self.input_scaler.fit(self.input_data)
            self.input_data = self.input_scaler.transform(self.input_data)
        self.input_data = self.input_data.reshape(self.input_data.shape[0], self.input_data.shape[1], 1)
        self.input_data = self.input_data.astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        input_seq = self.input_data[:, :, :]   # [index, pred_len, feature=1]
        input_matrix = np.expand_dims(input_seq, axis=0)
        input_matrix = np.transpose(input_matrix, (0, 3, 1, 2))

        return input_matrix


if __name__ == '__main__':
    # dataset_stock = Dataset_Stock(target='movement', flag='test', data_path='GroupA.csv', start_date='2019/12/15', end_date='2020/12/14')
    dataset_stock = Dataset_Stock(target='movement', data_path='GroupB.csv')
    target_data = dataset_stock.target_data
    zeros_count = np.count_nonzero(target_data == 0)
    ones_count = np.count_nonzero(target_data == 1)
    print(zeros_count)
    print(ones_count)

    # data_loader = DataLoader(
    #         dataset_stock,
    #         batch_size=32,
    #         shuffle=True,
    #         num_workers=0,
    #         drop_last=True)
    # for i, (input_seq, target_seq) in enumerate(data_loader):
    #     print(input_seq.shape)
    #     print(target_seq.shape)