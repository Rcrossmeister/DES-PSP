import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.scaler import StandardScaler, MinMaxScaler

class Dataset_Stock(Dataset):
    def __init__(self, root_path='npy_path', data_path='08', input_file='all_input.npy',
                 target_pr_file='all_target_pr.npy', target_mo_file='all_target_mo.npy',
                 target='price', pred_len=14, flag='train', scale=True, inverse=False):
        data_path = os.path.join(root_path, data_path)
        self.target = target
        self.flag = flag
        self.inverse = inverse

        self.input_data = np.load(os.path.join(data_path, input_file))
        if target == 'price':
            self.target_data = np.load(os.path.join(data_path, target_pr_file))

        else:
            self.target_data = np.load(os.path.join(data_path, target_mo_file))
        self.target_data = self.target_data[:, :pred_len]

        self.input_data = self.input_data.reshape(self.input_data.shape[0], self.input_data.shape[1], 1)
        self.target_data = self.target_data.reshape(self.target_data.shape[0], self.target_data.shape[1], 1)
        self.input_data = self.input_data.astype(np.float32)
        self.target_data = self.target_data.astype(np.float32)

        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.input_scaler.fit(self.input_data)
        self.input_data = self.input_scaler.transform(self.input_data)
        if scale:
            if self.target == 'price':
                self.target_scaler.fit(self.target_data)
                self.target_data = self.target_scaler.transform(self.target_data)

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        input_seq = self.input_data[index, :, :]  # [index, pred_len, feature=1]
        target_seq = self.target_data[index, :, :]
        return input_seq, target_seq  # [batch_size, pred_len, feature=1]

class DataSet_Competitor(Dataset):
    def __init__(self, root_path='npy_path', data_path='08', input_file='com_input.npy',
                 target_pr_file=None, target_mo_file=None,
                 target=None, pred_len=None, flag='train', scale=True, inverse=False):
        data_path = os.path.join(root_path, data_path)
        self.target = target
        self.flag = flag
        self.scale = scale
        self.inverse = inverse

        self.input_data = np.load(os.path.join(data_path, input_file))
        self.input_scaler = StandardScaler()

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
    dataset = Dataset_Stock()
    print(dataset.input_data.shape)
    print(dataset.target_data.shape)

    dataset_com = DataSet_Competitor()
    print(dataset_com[0].shape)