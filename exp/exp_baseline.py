from data_loader.data_loader import Dataset_Stock
from exp.exp_basic import Exp_Basic
from utils.plotter import plot_loss
from utils.metrics import compute_metrics, acc, MCC
from utils.metrics_new import classification_metrics, regression_metrics, calculate_label_num
from utils.tools import init_logger
from models.baseline_model import Seq2Seq_LSTM, Seq2Seq_BiLSTM, Seq2Seq_GRU, Seq2Seq_BiGRU, LSTM, BiLSTM, GRU, BiGRU, CNN_LSTM

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import os
import time
import pprint
import json




class Exp_Baseline(Exp_Basic):
    def __init__(self, args):
        super(Exp_Baseline, self).__init__(args)

        current_datetime = time.strftime("%Y-%m-%d_%H-%M")
        if self.args.test_only:
            self.output_path = os.path.join(self.args.test_results_path, f'{self.args.target}_{self.args.model}_{current_datetime}')
        else:
            self.output_path = os.path.join(self.args.results_path, f'{self.args.target}_{self.args.model}_{current_datetime}')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        self.logger = init_logger(self.output_path)

    def _build_model(self):
        model_dict = {
            'seq2seq_lstm': Seq2Seq_LSTM,
            'seq2seq_bilstm': Seq2Seq_BiLSTM,
            'seq2seq_gru': Seq2Seq_GRU,
            'seq2seq_bigru': Seq2Seq_BiGRU,
            'lstm': LSTM,
            'bilstm': BiLSTM,
            'gru': GRU,
            'bigru': BiGRU,
            'cnn_lstm': CNN_LSTM
        }
        model = model_dict[self.args.model](
            self.args.input_size,
            self.args.hidden_size, 
            self.args.output_size,
            self.args.pred_steps,
            self.args.num_layers,
            self.args.dropout)
        
        return model
    
    def _get_data(self, flag):
        if flag == 'train':
            shuffle_flag = True
            drop_last = False
            batch_size = self.args.batch_size
            data_path = self.args.all_data_path
            start_date = self.args.data_start_date
            end_date = self.args.data_end_date
        elif flag == 'val':
            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_size
            data_path = self.args.biden_data_path
            start_date = self.args.val_start_date
            end_date = self.args.val_end_date
        else:
            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_size
            data_path = self.args.biden_data_path
            start_date = self.args.test_start_date
            end_date = self.args.test_end_date
        data_set = Dataset_Stock(
            root_path=self.args.root_path,
            data_path=data_path,
            target=self.args.target,
            start_date=start_date,
            end_date=end_date,
            pred_len=self.args.pred_steps,
            remove_invalid=self.args.remove_invalid,
            flag=flag,
            scale=self.args.scale,
            inverse=self.args.inverse)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last)
        
        return data_set, data_loader
    
    def _set_optim(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return optimizer
    
    def _set_criterion(self):
        if self.args.target == 'price':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        return criterion
    
    def train(self):
        arg_dict = vars(self.args)
        pp = pprint.PrettyPrinter(indent=4)
        self.logger.info(pp.pformat(arg_dict))

        self.logger.info('Start loading data...')

        train_data_set, train_data_loader = self._get_data('train')
        self.logger.info('Train data loaded successfully!')

        val_data_set, val_data_loader = self._get_data('val')
        self.logger.info('Val data loaded successfully!')

        test_data_set, test_data_loader = self._get_data('test')
        self.logger.info('Test data loaded successfully!')

        time_now = time.time()

        optim = self._set_optim()
        criterion = self._set_criterion()

        self.logger.info('Start training...')
        all_train_loss = []
        all_val_loss = []
        for epoch in range(self.args.epochs):
            self.model.train()
            train_loss = []
            for i, (input_seq, target_seq) in enumerate(train_data_loader):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                output_seq = self.model(input_seq)
                loss = criterion(output_seq, target_seq)
                train_loss.append(loss.item())
                optim.zero_grad()
                loss.backward()
                optim.step()
                self.logger.info(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item():.10f}")
                # if loss.item() > 1000:
                #     print(input_seq)
                #     print(target_seq)
                #     exit(0)
            current_train_loss = np.mean(train_loss)
            self.logger.info(f"Epoch: {epoch}, Train Loss: {current_train_loss:.10f}")
            val_loss, val_targets, val_outputs  = self.val(val_data_set, val_data_loader, criterion)
            self.logger.info(f"Epoch: {epoch}, Val Loss: {val_loss:.10f}")

            all_train_loss.append(current_train_loss)
            all_val_loss.append(val_loss)

        self.logger.info(f"Training finished, total training time: {time.time() - time_now:.4f}s")

        plot_loss('train_loss', all_train_loss, self.output_path)
        plot_loss('val_loss', all_val_loss, self.output_path)

        self.logger.info('Testing...')
        self.test(test_data_set, test_data_loader, criterion)

        self.logger.info('Start saving model...')
        torch.save(self.model.state_dict(), os.path.join(self.output_path, f'{self.args.model}.pth'))

    def val(self, val_data, val_loader, criterion):
        self.model.eval()
        val_loss = []
        all_targets = []
        all_outputs = []
        for i, (input_seq, target_seq) in enumerate(val_loader):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            output_seq = self.model(input_seq)
            all_targets.append(target_seq.detach().cpu())
            all_outputs.append(output_seq.detach().cpu())
            loss = criterion(output_seq, target_seq)
            val_loss.append(loss.item())
        return np.mean(val_loss), all_targets, all_outputs

    def test(self, test_data_set, test_data_loader, criterion):
        self.model.eval()
        test_loss, test_targets, test_outputs = self.val(test_data_set, test_data_loader, criterion)
        all_test_targets = torch.cat(test_targets, dim=0)
        all_test_outputs = torch.cat(test_outputs, dim=0)

        if self.args.target == 'price':

            mse_score, rmse_score, mae_score, ade, fde = regression_metrics(all_test_outputs, all_test_targets)
            self.logger.info(f'''
                            Metrics on Test set:
                                MSE:{mse_score},
                                RMSE:{rmse_score},
                                MAE:{mae_score},
                                ADE:{ade},
                                FDE:{fde}
            ''')
            # save as json
            metrics = {
                'Model': self.args.model,
                'target': self.args.target,
                'Metrics': {
                    'MSE': mse_score.item(),
                    'RMSE': rmse_score.item(),
                    'MAE': mae_score.item(),
                    'ADE': ade.item(),
                    'FDE': fde.item()
                },
            }

        else:
            threshold = 0.5
            all_test_outputs = torch.sigmoid(all_test_outputs)
            all_test_outputs = (all_test_outputs > threshold).float()

            target_0s, target_1s = calculate_label_num(all_test_targets)
            output_0s, output_1s = calculate_label_num(all_test_outputs)

            acc, f1, mcc = classification_metrics(all_test_outputs, all_test_targets)
            self.logger.info(f'''
                            Target 0s: {target_0s}, Target 1s: {target_1s}
                            Output 0s: {output_0s}, Output 1s: {output_1s}
                            Metrics on Test set:
                                Accuracy:{acc},
                                F1:{f1},
                                MCC:{mcc}
            ''')
            # save as json
            metrics = {
                'Model': self.args.model,
                'target': self.args.target,
                'Metrics': {
                    'Accuracy': acc.item(),
                    'F1': f1.item(),
                    'MCC': mcc.item()
                },
            }

        with open(os.path.join(self.output_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)


    def test_only(self):
        self.logger.info('Start loading model...')
        self.logger.info(f'model_name: {self.args.model}')
        self.logger.info(f'model_path: {self.args.checkpoint_path}')
        self.model.load_state_dict(torch.load(self.args.checkpoint_path, map_location=self.args.device))
        self.model.eval()

        self.logger.info('Start loading data...')
        test_data_set, test_data_loader = self._get_data('test')
        self.logger.info('Test data loaded successfully!')

        criterion = self._set_criterion()

        self.test(test_data_set, test_data_loader, criterion)



