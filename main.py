import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import prepare_data, remove_invalid_stocks
import os

import argparse

parser = argparse.ArgumentParser(description='Stock Price Prediction')
parser.add_argument('--df_path', type=str, default='df_path/', help='data frame path')
# parser.add_argument('--df_path_trump', type=str, default='/home/hzj/NLP1/StockPricePrediction/2023-05-22/GroupB.csv', help='data frame path')
# parser.add_argument('--df_path_biden', type=str, default='/home/hzj/NLP1/StockPricePrediction/2023-05-22/GroupA.csv', help='data frame path')
parser.add_argument('--data_start_date', type=str, default='2015/11/09', help='data start date')
parser.add_argument('--data_end_date', type=str, default='2016/11/08', help='data end date')
parser.add_argument('--val_start_date', type=str, default='2019/12/15', help='validation start date')
parser.add_argument('--val_end_date', type=str, default='2020/12/14', help='validation end date')
parser.add_argument('--pred_steps', type=int, default=14, help='prediction steps')
args = parser.parse_args()

df_path = os.path.join(args.df_path, "All_Data.csv")
print(df_path)
df_path_trump = os.path.join(args.df_path, "GroupB.csv")
print(df_path_trump)
df_path_biden = os.path.join(args.df_path, "GroupA.csv")
print(df_path_biden)

# df_path = '/home/hzj/NLP1/StockPricePrediction/rc_ross/All_Data.csv'
# df_path_trump = '/home/hzj/NLP1/StockPricePrediction/2023-05-22/GroupB.csv'
# df_path_biden = '/home/hzj/NLP1/StockPricePrediction/2023-05-22/GroupA.csv'
# data_start_date = '2015/11/09'
# data_end_date = '2016/11/08'
# val_start_date = '2019/12/15'
# val_end_date = '2020/12/14'
# pred_steps = 14

train_input_data, train_target_data = prepare_data(df_path, args.data_start_date, args.data_end_date, args.pred_steps)
val_input_data, val_target_data = prepare_data(df_path_biden, args.val_start_date, args.val_end_date, args.pred_steps)
# train_input_data, train_target_data, val_input_data, val_target_data = remove_invalid_stocks(train_input_data,
#                                                                                                 train_target_data,
#                                                                                                 val_input_data,
#                                                                                                 val_target_data)
# # 准备数据
input_seq = train_input_data # 输入序列
target_seq = train_target_data  # 目标序列

# 归一化数据
input_seq_mean = np.mean(input_seq, axis=1, keepdims=True)
input_seq_std = np.std(input_seq, axis=1, keepdims=True)
input_seq_std[input_seq_std == 0] = 1  # 将为零的标准差替换为1
input_seq = (input_seq - input_seq_mean) / input_seq_std

target_seq_mean = np.mean(target_seq, axis=1, keepdims=True)
target_seq_std = np.std(target_seq, axis=1, keepdims=True)
target_seq_std[target_seq_std == 0] = 1  # 将为零的标准差替换为1
target_seq = (target_seq - target_seq_mean) / target_seq_std


# 将数据转换为PyTorch张量，并放到CUDA设备上
device = torch.device("cuda:1")
input_seq = torch.from_numpy(input_seq).float().to(device)
target_seq = torch.from_numpy(target_seq).float().to(device)

# 将NaN值替换为0
input_seq[input_seq != input_seq] = 0
target_seq[target_seq != target_seq] = 0

# 定义Seq2Seq模型并放到CUDA设备上
class Seq2Seq_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq_LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, dropout=0.2, num_layers=5)
        self.decoder = nn.LSTM(hidden_size - 1, hidden_size, dropout=0.2, num_layers=5)
        self.fc_encoder = nn.Linear(input_size, hidden_size)
        self.fc_decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        _, (hidden, cell) = self.encoder(input_seq)
        input_seq = self.fc_encoder(input_seq)
        output_seq, _ = self.decoder(input_seq[:, :-1], (hidden, cell))
        output_seq = self.fc_decoder(output_seq)
        return output_seq

class Seq2Seq_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq_GRU, self).__init__()
        self.encoder = nn.GRU(input_size, hidden_size, dropout=0.2, num_layers=5)
        self.decoder = nn.GRU(hidden_size - 1, hidden_size, dropout=0.2, num_layers=5)
        self.fc_encoder = nn.Linear(input_size, hidden_size)
        self.fc_decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        _, hidden = self.encoder(input_seq)
        input_seq = self.fc_encoder(input_seq)
        output_seq, _ = self.decoder(input_seq[:, :-1], hidden)
        output_seq = self.fc_decoder(output_seq)
        return output_seq

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_size, dropout=0.2, num_layers=5)
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=0.2, num_layers=5, bidirectional=True)
        # self.fc = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        out, (hidden, cell) = self.lstm(input_seq)
        output_seq = self.fc(out)

        return output_seq

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, dropout=0.2, num_layers=5, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, input_seq):
        out, _ = self.gru(input_seq)
        output_seq = self.fc(out)
        return output_seq

# 初始化模型并放到CUDA设备上
input_size = 366
hidden_size = 64
output_size = 14

model = Seq2Seq_LSTM(input_size, hidden_size, output_size).to(device)
# model = Seq2Seq_GRU(input_size, hidden_size, output_size).to(device)
# model = LSTM(input_size, hidden_size, output_size).to(device)
# model = GRU(input_size, hidden_size, output_size).to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
batch_size = 512

for epoch in range(num_epochs):
    for i in range(0, len(input_seq) - batch_size, batch_size):
        input_batch = input_seq[i:i+batch_size]
        target_batch = target_seq[i:i+batch_size]

        # 前向传播
        output_batch = model(input_batch)

        # 计算损失
        loss = criterion(output_batch, target_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')
# 验证
# 归一化验证数据
val_input_mean = np.mean(val_input_data, axis=1, keepdims=True)
val_input_std = np.std(val_input_data, axis=1, keepdims=True)
val_input_std[val_input_std == 0] = 1  # 将为零的标准差替换为1
val_input_seq = (val_input_data - val_input_mean) / val_input_std

val_target_mean = np.mean(val_target_data, axis=1, keepdims=True)
val_target_std = np.std(val_target_data, axis=1, keepdims=True)
val_target_std[val_target_std == 0] = 1  # 将为零的标准差替换为1
val_target_seq = (val_target_data - val_target_mean) / val_target_std


# 将验证数据转换为PyTorch张量，并放到CUDA设备上
val_input_seq = torch.from_numpy(val_input_seq).float().to(device)
val_target_seq = torch.from_numpy(val_target_seq).float().to(device)

# 将NaN值替换为0
val_input_seq[val_input_seq != val_input_seq] = 0
val_target_seq[val_target_seq != val_target_seq] = 0

# 将模型设为评估模式
model.eval()

# 禁用梯度计算
with torch.no_grad():
    # 前向传播
    val_output_seq = model(val_input_seq)

# 计算验证损失
val_loss = criterion(val_output_seq, val_target_seq)
print(f'Validation Loss: {val_loss.item()}')

# 逆归一化验证数据
# val_output_data = val_output_seq.cpu().numpy() * val_target_std + val_target_mean
val_output_data = val_output_seq.cpu().numpy()
# val_target_data = val_target_seq.cpu().numpy() * val_target_std + val_target_mean
val_target_data = val_target_seq.cpu().numpy()

# 计算MAE
mae = np.mean(np.abs(val_output_data - val_target_data))
# 计算ADE和FDE
ade = np.mean(np.linalg.norm(val_output_data - val_target_data, axis=1))
fde = np.linalg.norm(val_output_data[-1] - val_target_data[-1])
# 计算MSE
mse = np.mean((val_output_data - val_target_data)**2)

# 打印结果
print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'ADE: {ade:.4f}')
print(f'FDE: {fde:.4f}')

# save model
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.save(model.state_dict(), os.path.join(model_dir, 'seq2seq_lstm.pth'))
