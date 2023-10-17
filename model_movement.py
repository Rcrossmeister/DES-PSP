import torch
import torch.nn as nn

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
        output_seq = torch.sigmoid(output_seq)
        return output_seq

class Seq2Seq_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq_BiLSTM, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, dropout=0.2, num_layers=5, bidirectional=True)
        self.decoder = nn.LSTM(hidden_size - 1, hidden_size, dropout=0.2, num_layers=5, bidirectional=True)
        self.fc_encoder = nn.Linear(input_size, hidden_size)
        self.fc_decoder = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        _, (hidden, cell) = self.encoder(input_seq)
        input_seq = self.fc_encoder(input_seq)
        output_seq, _ = self.decoder(input_seq[:, :-1], (hidden, cell))
        output_seq = self.fc_decoder(output_seq)
        output_seq = torch.sigmoid(output_seq)
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
        output_seq = torch.sigmoid(output_seq)
        return output_seq

class Seq2Seq_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq_BiGRU, self).__init__()
        self.encoder = nn.GRU(input_size, hidden_size, dropout=0.2, num_layers=5, bidirectional=True)
        self.decoder = nn.GRU(hidden_size - 1, hidden_size, dropout=0.2, num_layers=5, bidirectional=True)
        self.fc_encoder = nn.Linear(input_size, hidden_size)
        self.fc_decoder = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        _, hidden = self.encoder(input_seq)
        input_seq = self.fc_encoder(input_seq)
        output_seq, _ = self.decoder(input_seq[:, :-1], hidden)
        output_seq = self.fc_decoder(output_seq)
        output_seq = torch.sigmoid(output_seq)
        return output_seq

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_size, dropout=0.2, num_layers=5)
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=0.2, num_layers=5, bidirectional=False)
        # self.fc = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        out, (hidden, cell) = self.lstm(input_seq)
        output_seq = self.fc(out)
        output_seq = torch.sigmoid(output_seq)
        return output_seq
    
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        # self.lstm = nn.LSTM(input_size, hidden_size, dropout=0.2, num_layers=5)
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=0.2, num_layers=5, bidirectional=True)
        # self.fc = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        out, (hidden, cell) = self.lstm(input_seq)
        output_seq = self.fc(out)
        output_seq = torch.sigmoid(output_seq)
        return output_seq

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, dropout=0.2, num_layers=5, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        out, _ = self.gru(input_seq)
        output_seq = self.fc(out)
        output_seq = torch.sigmoid(output_seq)
        return output_seq
    
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, dropout=0.2, num_layers=5, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, input_seq):
        out, _ = self.gru(input_seq)
        output_seq = self.fc(out)
        output_seq = torch.sigmoid(output_seq)
        return output_seq
