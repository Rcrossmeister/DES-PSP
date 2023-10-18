import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils_movement import init_logger, prepare_data, remove_invalid_stocks, acc, MCC
import os

import argparse
import time
import logging


parser = argparse.ArgumentParser(description='Stock Price Prediction')
parser.add_argument('--model', type=str, default='seq2seq_lstm', help='model name')
parser.add_argument('--output_path', type=str, default='results_88/', help='log path')
parser.add_argument('--df_path', type=str, default='df_path/', help='data frame path')
# parser.add_argument('--df_path_trump', type=str, default='/home/hzj/NLP1/StockPricePrediction/2023-05-22/GroupB.csv', help='data frame path')
# parser.add_argument('--df_path_biden', type=str, default='/home/hzj/NLP1/StockPricePrediction/2023-05-22/GroupA.csv', help='data frame path')
parser.add_argument('--data_start_date', type=str, default='2015/11/09', help='data start date')
parser.add_argument('--data_end_date', type=str, default='2016/11/08', help='data end date')
parser.add_argument('--val_start_date', type=str, default='2019/12/15', help='validation start date')
parser.add_argument('--val_end_date', type=str, default='2020/12/14', help='validation end date')
parser.add_argument('--pred_steps', type=int, default=14, help='prediction steps')
parser.add_argument('--cuda', type=str, default='cuda:0', help='cuda device')
parser.add_argument('--seed', type=int, default=88, help='seed')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


class experiment(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(self.args.cuda if torch.cuda.is_available() else 'cpu')
        self.model = self._get_model()
        self.df_path = os.path.join(args.df_path, "All_Data.csv")
        self.df_path_trump = os.path.join(args.df_path, "GroupB.csv")
        self.df_path_biden = os.path.join(args.df_path, "GroupA.csv")
        current_datetime = time.strftime("%Y-%m-%d_%H-%M")
        self.output_path = f'{args.output_path}/{args.model}_{current_datetime}'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _get_model(self):
        input_size = 366
        hidden_size = 64
        output_size = self.args.pred_steps

        if self.args.model == 'seq2seq_lstm':
            from model_movement import Seq2Seq_LSTM
            model = Seq2Seq_LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        elif self.args.model == 'seq2seq_bilstm':
            from model_movement import Seq2Seq_BiLSTM
            model = Seq2Seq_BiLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        elif self.args.model == 'seq2seq_gru':
            from model_movement import Seq2Seq_GRU
            model = Seq2Seq_GRU(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        elif self.args.model == 'seq2seq_bigru':
            from model_movement import Seq2Seq_BiGRU
            model = Seq2Seq_BiGRU(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        elif self.args.model == 'lstm':
            from model_movement import LSTM
            model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        elif self.args.model == 'bilstm':
            from model_movement import BiLSTM
            model = BiLSTM(input_size, hidden_size, output_size)
        elif self.args.model == 'gru':
            from model_movement import GRU
            model = GRU(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        elif self.args.model == 'bigru':
            from model_movement import BiGRU
            model = BiGRU(input_size, hidden_size, output_size)
        else:
            raise ValueError('No such model!')

        model.to(self.device)
        return model
    
    def _scaler(self, data):
        data_mean = np.mean(data, axis=1, keepdims=True)
        data_std = np.std(data, axis=1, keepdims=True)
        data_std[data_std == 0] = 1
        data = (data - data_mean) / data_std

        data[data != data] = 0
        return data

    def _get_data(self):
        train_input_data, train_target_data = prepare_data(self.df_path, self.args.data_start_date, self.args.data_end_date, self.args.pred_steps)
        val_input_data, val_target_data = prepare_data(self.df_path, self.args.val_start_date, self.args.val_end_date, self.args.pred_steps)
        # train_input_data, train_target_data, val_input_data, val_target_data = remove_invalid_stocks(train_input_data, train_target_data, val_input_data, val_target_data)
        train_input_data = self._scaler(train_input_data)
        val_input_data = self._scaler(val_input_data)
        return train_input_data, train_target_data, val_input_data, val_target_data
    
    def train(self):
        train_input_data, train_target_data, val_input_data, val_target_data = self._get_data()
        train_input_seq = torch.from_numpy(train_input_data).float().to(self.device)
        train_target_seq = torch.from_numpy(train_target_data).float().to(self.device)
        val_input_seq = torch.from_numpy(val_input_data).float().to(self.device)
        val_target_seq = torch.from_numpy(val_target_data).float().to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        logger = init_logger(self.output_path)
        logger.info(f'''
                    model: {self.args.model}
                    data_start_date: {self.args.data_start_date}
                    data_end_date: {self.args.data_end_date}
                    val_start_date: {self.args.val_start_date}
                    val_end_date: {self.args.val_end_date}
                    pred_steps: {self.args.pred_steps}
                    cuda: {self.args.cuda}
                    seed: {self.args.seed}
                    ''')    
        logger.info('Start training...')

        num_epochs = 100
        batch_size = 512

        for epoch in range(num_epochs):
            self.model.train()
            for i in range(0, len(train_input_seq) - batch_size, batch_size):
                input_batch = train_input_seq[i:i + batch_size]
                target_batch = train_target_seq[i:i + batch_size]

                output_batch = self.model(input_batch)

                loss = criterion(output_batch, target_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                logger.info(f'Epoch: {epoch+1}, Loss: {loss.item()}')
        
        logger.info('Finish training!')
        logger.info('Start testing...')

        with torch.no_grad():
            self.model.eval()
            val_output_seq = self.model(val_input_seq)
            val_loss = criterion(val_output_seq, val_target_seq)
            logger.info('Test Loss: {}'.format(val_loss.item()))
            threshold = 0.5
            val_output_seq = (val_output_seq > threshold).float()
            accuracy = acc(val_output_seq, val_target_seq)
            mcc = MCC(val_output_seq, val_target_seq)
            logger.info(f'acc:{accuracy:.20f}, MCC:{mcc:.20f}')
        
        logger.info('Finish testing!')
        logger.info('Saving model...')
        torch.save(self.model.state_dict(), os.path.join(self.output_path, f'{self.args.model}.pth'))

    def test_only(self, model_path):
        logger = init_logger(self.output_path)
        logger.info(f'{self.args.model}, test_only, model_path:{model_path}')
        self.model.load_state_dict(torch.load(model_path))
        train_input_data, train_target_data, val_input_data, val_target_data = self._get_data()
        criterion = nn.BCELoss()
        with torch.no_grad():
            self.model.eval()
            val_input_seq = torch.from_numpy(val_input_data).float().to(self.device)
            val_target_seq = torch.from_numpy(val_target_data).float().to(self.device)
            val_output_seq = self.model(val_input_seq)
            val_loss = criterion(val_output_seq, val_target_seq)
            logger.info('Test Loss: {}'.format(val_loss.item()))
            threshold = 0.5
            val_output_seq = (val_output_seq > threshold).float()
            accuracy = acc(val_output_seq, val_target_seq)
            mcc = MCC(val_output_seq, val_target_seq)
            logger.info(f'acc:{accuracy:.20f}, MCC:{mcc:.20f}')
if __name__ == '__main__':
    exp = experiment(args)
    
    exp.train()
    # exp.test_only('/home/hjd/work/DES-PSP/results_mm/seq2seq_gru_2023-10-18_19-22/seq2seq_gru.pth')