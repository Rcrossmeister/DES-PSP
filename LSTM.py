import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 准备数据
stock_prices = [] #[/* 在这里插入您的8000条股票的收盘价格数据 */]

# 数据预处理
normalized_prices = (stock_prices - np.mean(stock_prices)) / np.std(stock_prices)

# 将数据划分为训练集和测试集
train_data = normalized_prices[:8000]
test_data = normalized_prices[8000:]

# 定义函数将时间序列数据转换为输入特征和目标值
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs).unsqueeze(2), torch.tensor(ys).unsqueeze(1)

# 创建训练集和测试集的序列数据
seq_length = 10  # 时间窗口大小，可根据需求调整
train_inputs, train_targets = create_sequences(train_data, seq_length)
test_inputs, test_targets = create_sequences(test_data, seq_length)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        output = self.fc(lstm_out[-1])
        return output

# 初始化模型
input_size = 1
hidden_size = 64
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, len(train_inputs) - batch_size, batch_size):
        input_batch = train_inputs[i:i+batch_size]
        target_batch = train_targets[i:i+batch_size]

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

# 测试模型
model.eval()
test_predictions = model(test_inputs)
test_loss = criterion(test_predictions, test_targets)
print(f'Test Loss: {test_loss.item()}')

# 进行未来股票价格预测
future_inputs = torch.tensor(test_data[-seq_length:]).unsqueeze(0).unsqueeze(2)
future_predictions = model(future_inputs)
predicted_prices = future_predictions.squeeze().detach().numpy()
predicted_prices = (predicted_prices * np.std(stock_prices)) + np.mean(stock_prices)
