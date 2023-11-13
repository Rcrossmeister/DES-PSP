from models.baseline_model import Encoder_LSTM, Decoder_LSTM
from models.DES_PSP import Encoder_CNN

import torch
import torch.nn as nn


class learnable_alpha(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(learnable_alpha, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

    def forward(self, extracted_features, x):
        # extracted_features: [1, 1, hidden_size]
        transformed = []
        for i in range(self.num_layers):
            transformed.append(self.linear_layers[i](extracted_features))
        features = torch.cat(transformed, dim=0)
        output_features = features.repeat(1, x.shape[1], 1)

        return output_features + x

class DES_PSP_L_Model(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, output_size=1,
                 kernel_size=3, stride=1, padding=1,
                 lstm_num_layers=5, cnn_num_layers=32,
                 alpha=0.2, pred_steps=14, dropout=0.2):
        super(DES_PSP_L_Model, self).__init__()
        self.pred_steps = pred_steps
        self.lstm_num_layers = lstm_num_layers
        self.hidden_size = hidden_size
        self.encoder_lstm = Encoder_LSTM(input_size, hidden_size, output_size, lstm_num_layers, dropout)
        self.encoder_cnn = Encoder_CNN(cnn_num_layers, input_size, hidden_size, kernel_size, stride, padding)
        self.decoder = Decoder_LSTM(input_size, hidden_size, output_size, lstm_num_layers, dropout)

        self.alpha_layer1 = learnable_alpha(lstm_num_layers, hidden_size)
        self.alpha_layer2 = learnable_alpha(lstm_num_layers, hidden_size)

    def forward(self, x, y):
        # x: [batch, seq_len, input_size] former presidential concept stock
        # y: [1, channel, stock_num, seq_len] all competitor concept stock
        encoder_hidden, encoder_cell = self.encoder_lstm(x)  # hidden: [num_layers, batch, hidden_size]
        features = self.encoder_cnn(y)       # features: [1, 1, hidden_size]
        decoder_input = x[:, -1, :].unsqueeze(1)  # decoder_input: [batch, 1, input_size]

        outputs = []
        input_hidden = self.alpha_layer1(features, encoder_hidden)
        input_cell = self.alpha_layer1(features, encoder_cell)

        for _ in range(self.pred_steps):
            out, input_hidden, input_cell = self.decoder(decoder_input, input_hidden, input_cell)
            outputs.append(out)
            decoder_input = out.unsqueeze(-1)

        return torch.cat(outputs, dim=1).unsqueeze(2)


if __name__ == '__main__':
    input_map = torch.randn(1, 1, 8000, 366).to("cuda:0")
    input_seq = torch.randn(10, 366, 1).to("cuda:0")
    model = DES_PSP_L_Model().to("cuda:0")
    output = model(input_seq, input_map)
    print(output.shape)
