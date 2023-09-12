import torch
import torch.nn as nn
import torch.optim as optim

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Encoding
        _, (hidden, cell) = self.encoder(x)
        # Decoding
        outputs, _ = self.decoder(hidden.repeat(x.size(1), 1, 1))
        outputs = self.fc(outputs)
        return outputs

# 定义数据
input_size = 1
hidden_size = 128
output_size = 1
seq_len = 10
batch_size = 32

# 生成随机数据
x = torch.randn(batch_size, seq_len, input_size)
y = torch.randn(batch_size, seq_len, output_size)

# 填充序列
max_len = max(seq_len, y.shape[1])
x_pad = nn.functional.pad(x, (0, 0, 0, max_len-seq_len))
y_pad = nn.functional.pad(y, (0, 0, 0, max_len-y.shape[1]))

# 初始化模型和优化器
model = Seq2Seq(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 向前传播
    outputs = model(x_pad)
    # 计算损失
    loss = nn.MSELoss()(outputs, y_pad)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 输出损失
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
