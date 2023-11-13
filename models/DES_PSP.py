import torch
import torch.nn as nn
from models.baseline_model import Encoder_LSTM, Decoder_LSTM


class Encoder_CNN(nn.Module):

    def __init__(self, num_layers, input_channel, output_channel, kernel_size, stride, padding):
        super(Encoder_CNN, self).__init__()
        CNN_layers = []
        CNN_layers.append(nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding))
        CNN_layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        for i in range(num_layers-1):
            CNN_layers.append(nn.Conv2d(output_channel, output_channel, kernel_size, stride, padding))
            CNN_layers.append(nn.ReLU(inplace=True))
        self.CNN_layers = nn.Sequential(*CNN_layers)

        # global average pooling
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
                                # x: [batch,  input_channel, height, width], e.g. [1,  1, 3000, 366]
        x = self.CNN_layers(x)  # x: [batch, output_channel, height, width], e.g. [1, 64, 3000, 366]
        x = self.GAP(x)         # x: [batch, output_channel,      1,     1], e.g. [1, 64,    1,   1]
        x = x.squeeze(-1).squeeze(-1).unsqueeze(0)  # x: [1, 1, 64]
        return x


class DES_PSP_Model(nn.Module):
    '''
    competitor concept stock encoder extract features by CNN,
    then add to former presidential concept stock LSTM encoder
    '''
    def __init__(self, input_size=1, hidden_size=64, output_size=1,
                 kernel_size=3, stride=1, padding=1,
                 lstm_num_layers=5, cnn_num_layers=32,
                 alpha=0.2, pred_steps=14, dropout=0.2):
        super(DES_PSP_Model, self).__init__()
        self.pred_steps = pred_steps
        self.lstm_num_layers = lstm_num_layers
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.encoder_lstm = Encoder_LSTM(input_size, hidden_size, output_size, lstm_num_layers, dropout)
        self.encoder_cnn = Encoder_CNN(cnn_num_layers, input_size, hidden_size, kernel_size, stride, padding)
        self.decoder = Decoder_LSTM(input_size, hidden_size, output_size, lstm_num_layers, dropout)

    def forward(self, x, y):
        # x: [batch, seq_len, input_size] former presidential concept stock
        # y: [1, channel, stock_num, seq_len] all competitor concept stock
        encoder_hidden, encoder_cell = self.encoder_lstm(x)  # hidden: [num_layers, batch, hidden_size]
        features = self.encoder_cnn(y)       # features: [1, 1, hidden_size]
        features = features.expand(self.lstm_num_layers, encoder_hidden.shape[1], self.hidden_size)
        decoder_input = x[:, -1, :].unsqueeze(1)  # decoder_input: [batch, 1, input_size]

        outputs = []
        input_hidden = encoder_hidden + self.alpha * features
        input_cell = encoder_cell

        for _ in range(self.pred_steps):
            out, input_hidden, input_cell = self.decoder(decoder_input, input_hidden, input_cell)
            outputs.append(out)
            decoder_input = out.unsqueeze(-1)

        return torch.cat(outputs, dim=1).unsqueeze(2)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_seq_1 = torch.randn(10, 366, 1).to(device)
    input_seq_2 = torch.randn(1, 3000, 366, 1).to(device)
    input_seq_2 = input_seq_2.permute(0, 3, 1, 2)

    model = DES_PSP_Model().to(device)
    # from torchinfo import summary
    #
    # summary(model)

    # output = model(input_seq_1, input_seq_2)
    # print(output.shape)