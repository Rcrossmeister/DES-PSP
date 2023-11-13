import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.scaler import StandardScaler, MinMaxScaler

class Dataset_npy(Dataset):
    def __init__(self, root_path='npy_path', data_path='all_target.npy', target='movement',
                 start_date='2015/11/09', end_date='2016/11/08', pred_len=14,
                 remove_invalid=False, flag='train', scale=True, inverse=False):
        self.all = np.load(os.path.join(root_path, data_path))
        self.input_data = self.all[:, :366]
        self.target_data = self.all[:, -14:]
        self.target = target

        self.input_data = self.input_data.reshape(self.input_data.shape[0], self.input_data.shape[1], 1)
        self.target_data = self.target_data.reshape(self.target_data.shape[0], self.target_data.shape[1], 1)
        self.input_data = self.input_data.astype(np.float32)
        self.target_data = self.target_data.astype(np.float32)

        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.input_scaler.fit(self.input_data)
        self.input_data = self.input_scaler.transform(self.input_data)
        if self.target == 'price':
            self.target_scaler.fit(self.target_data)
            self.target_data = self.target_scaler.transform(self.target_data)

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        input_seq = self.input_data[index, :, :]  # [index, pred_len, feature=1]
        target_seq = self.target_data[index, :, :]
        return input_seq, target_seq  # [batch_size, pred_len, feature=1]


if __name__ == '__main__':
    dataset = Dataset_npy()
    print(dataset.input_data.shape)
    print(dataset.target_data.shape)