import torch
import torch.nn as nn


class Encoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=5, dropout=0.2):
        super(Encoder_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        out = self.fc(out[:, -1, :])
        return out, hidden, cell


class Seq2Seq_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len=14, num_layers=5, dropout=0.2):
        super(Seq2Seq_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.pred_len = pred_len

        self.encoder = Encoder_LSTM(input_size, hidden_size, output_size, num_layers, dropout)
        self.decoder = Decoder_LSTM(input_size, hidden_size, output_size, num_layers, dropout)

    def forward(self, x):
        encoder_hidden, encoder_cell = self.encoder(x)
        decoder_input = x[:, -1, :].unsqueeze(1) # [batch, 1, 1]

        outputs = []
        for _ in range(self.pred_len):
            out, encoder_hidden, encoder_cell = self.decoder(decoder_input, encoder_hidden, encoder_cell)
            outputs.append(out)
            decoder_input = out.unsqueeze(-1)

        return torch.cat(outputs, dim=1).unsqueeze(2)


class Seq2Seq_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len=14, num_layers=5, dropout=0.2):
        super(Seq2Seq_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=2 * hidden_size,  # because of BiLSTM
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )

        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        _, (h_n, c_n) = self.encoder(x)

        h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_size).transpose(1, 2).contiguous().view(
            self.num_layers, x.size(0), 2 * self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, x.size(0), self.hidden_size).transpose(1, 2).contiguous().view(
            self.num_layers, x.size(0), 2 * self.hidden_size)

        decoder_input = x[:, -1, :].unsqueeze(1)
        outputs = []

        for _ in range(self.pred_len):
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            out = self.fc(decoder_output)
            outputs.append(out)
            decoder_input = out

        return torch.cat(outputs, dim=1)


class Seq2Seq_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len=14, num_layers=5, dropout=0.2):
        super(Seq2Seq_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )

        self.decoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h_n = self.encoder(x)

        decoder_input = x[:, -1, :].unsqueeze(1)
        outputs = []

        for _ in range(self.pred_len):
            decoder_output, h_n = self.decoder(decoder_input, h_n)
            out = self.fc(decoder_output)
            outputs.append(out)
            decoder_input = out

        return torch.cat(outputs, dim=1)


class Seq2Seq_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len=14, num_layers=5, dropout=0.2):
        super(Seq2Seq_BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        self.decoder = nn.GRU(
            input_size=input_size,
            hidden_size=2 * hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        # Encoder
        _, h_n = self.encoder(x)
        h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_size).transpose(1, 2).contiguous().view(
            self.num_layers, x.size(0), 2 * self.hidden_size)

        decoder_input = x[:, -1, :].unsqueeze(1)
        outputs = []

        for _ in range(self.pred_len):
            decoder_output, h_n = self.decoder(decoder_input, h_n)
            out = self.fc(decoder_output)
            outputs.append(out)
            decoder_input = out

        return torch.cat(outputs, dim=1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len=14, num_layers=5, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            dropout=dropout,
                            num_layers=num_layers,
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, input_seq):
        out, (hidden, cell) = self.lstm(input_seq)
        output_seq = self.fc(hidden[0]).unsqueeze(-1)
        return output_seq


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len=14, num_layers=1, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            dropout=dropout,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, pred_len)

    def forward(self, input_seq):
        out, (hidden, cell) = self.lstm(input_seq)
        bi_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), 1)
        output_seq = self.fc(bi_hidden).unsqueeze(-1)
        return output_seq


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len=14, num_layers=5, dropout=0.2):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, hidden = self.gru(x)
        output_seq = self.fc(out[:, -1, :]).unsqueeze(-1)

        return output_seq


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len=14, num_layers=5, dropout=0.2):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(2 * hidden_size, pred_len)

    def forward(self, x):
        out, hidden = self.gru(x)
        out_ = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), 1)
        output_seq = self.fc(out_).unsqueeze(-1)
        return output_seq

class CNN_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, pred_len=14, num_layers=5, dropout=0.2):
        super(CNN_LSTM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=30, seq_len=24, input_size=7) ---> permute(0, 2, 1)
        # (30, 7, 24)
        # self.conv = nn.Sequential(
        #     nn.Conv1d(in_channels=args.in_channels, out_channels=args.out_channels, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=3, stride=1)
        # )

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1)

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        x, (hidden, cell) = self.lstm(x)
        x = self.fc(hidden[0]).unsqueeze(-1)

        return x




if __name__ == '__main__':
    input_seq = torch.randn(512, 366, 1).to('cuda')
    model = CNN_LSTM().to('cuda')
    output_seq = model(input_seq)
    print(output_seq.shape)