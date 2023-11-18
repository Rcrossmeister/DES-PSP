from data_loader.data_npy import Dataset_Stock, DataSet_Competitor
from exp.exp_basic import Exp_Basic
from models.DES_PSP import DES_PSP_Model
from models.DES_PSP_L import DES_PSP_L_Model
from utils.plotter import plot_loss
from utils.metrics_new import regression_metrics, classification_metrics, calculate_label_num
from utils.tools import init_logger

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import os
import time
import pprint
import gc
import json

class Exp_DES_PSP(Exp_Basic):
    def __init__(self, args):
        super(Exp_DES_PSP, self).__init__(args)

        current_datetime = time.strftime("%Y-%m-%d_%H-%M")
        if self.args.test_only:
            self.output_path = self.args.test_results_path
        else:
            self.output_path = os.path.join(self.args.results_path, f'{self.args.target}_{self.args.model}_{current_datetime}')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(os.path.join(self.output_path, 'ckp')):
            os.makedirs(os.path.join(self.output_path, 'ckp'))

        self.logger = init_logger(self.output_path)

    def _build_model(self):
        model_dict = {
            'des_psp': DES_PSP_Model,
            'des_psp_l': DES_PSP_L_Model
        }
        model = model_dict[self.args.model](
            input_size=self.args.input_size,
            hidden_size=self.args.hidden_size,
            output_size=self.args.output_size,
            kernel_size=self.args.kernel_size,
            stride=self.args.stride,
            padding=self.args.padding,
            lstm_num_layers=self.args.lstm_num_layers,
            cnn_num_layers=self.args.cnn_num_layers,
            alpha=self.args.alpha,
            pred_steps=self.args.pred_steps,
            dropout=self.args.dropout)

        return model

    def _get_data(self, flag):
        dataset_dict = {
            'stock': Dataset_Stock,
        }
        if flag == 'train':
            dataset = dataset_dict['stock']
            root_path = self.args.root_path
            data_path = self.args.train_data_path
            input_file = self.args.train_input_file
            target_pr_file = self.args.train_target_pr_file
            target_mo_file = self.args.train_target_mo_file

            shuffle_flag = True
            drop_last = False
            batch_size = self.args.batch_size
        elif flag == 'val':
            dataset = dataset_dict['stock']
            root_path = self.args.root_path
            data_path = self.args.val_data_path
            input_file = self.args.val_input_file
            target_pr_file = self.args.val_target_pr_file
            target_mo_file = self.args.val_target_mo_file

            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_size
        else:
            dataset = dataset_dict['stock']
            root_path = self.args.root_path
            data_path = self.args.test_data_path
            input_file = self.args.test_input_file
            target_pr_file = self.args.test_target_pr_file
            target_mo_file = self.args.test_target_mo_file

            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_size
        data_set = dataset(
            root_path=root_path,
            data_path=data_path,
            input_file=input_file,
            target_pr_file=target_pr_file,
            target_mo_file=target_mo_file,
            target=self.args.target,
            pred_len=self.args.pred_steps,
            flag=flag,
            scale=self.args.scale,
            inverse=self.args.inverse)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=drop_last)

        competitor_data_set = self._get_competitor_data(flag)
        competitor_matrix = competitor_data_set[0]
        competitor_tensor = torch.from_numpy(competitor_matrix).float()
        return data_set, competitor_tensor, data_loader

    def _get_competitor_data(self, flag):
        if flag == 'train':
            root_path = self.args.root_path
            data_path = self.args.train_data_path
            input_file = self.args.train_input_file
        elif flag == 'val':
            root_path = self.args.root_path
            data_path = self.args.val_data_path
            input_file = self.args.val_input_file
        else:
            root_path = self.args.root_path
            data_path = self.args.val_data_path
            input_file = self.args.val_input_file
        data_set = DataSet_Competitor(
            root_path='npy_path',
            data_path='08',
            input_file='com_input.npy',
            flag=flag,
            scale=self.args.scale,
            inverse=self.args.inverse)

        return data_set


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
        train_data_set, train_competitor_tensor, train_data_loader = self._get_data('train')
        self.logger.info('Train data loaded successfully!')

        val_data_set, val_competitor_tensor, val_data_loader = self._get_data('val')
        self.logger.info('Val data loaded successfully!')

        test_data_set, test_competitor_tensor, test_data_loader = self._get_data('test')
        self.logger.info('Test data loaded successfully!')

        time_now = time.time()

        optim = self._set_optim()
        criterion = self._set_criterion()

        self.logger.info('Start training...')
        all_train_loss = []
        all_val_loss = []
        last_val_loss = float('inf')

        for epoch in range(self.args.epochs):
            self.model.train()
            train_loss = []
            train_competitor_tensor = train_competitor_tensor.to(self.device)
            for i, (input_seq, target_seq) in enumerate(train_data_loader):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                output_seq = self.model(input_seq, train_competitor_tensor)
                loss = criterion(output_seq, target_seq)
                train_loss.append(loss.item())
                optim.zero_grad()
                loss.backward()
                optim.step()
                # self.logger.info(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item():.10f}")
            train_competitor_tensor = train_competitor_tensor.detach().cpu()
            current_train_loss = np.mean(train_loss)
            self.logger.info(f"Epoch: {epoch}, Train Loss: {current_train_loss:.10f}")

            val_loss, val_targets, val_outputs = self.val(val_data_set, val_data_loader, val_competitor_tensor, criterion)
            self.logger.info(f"Epoch: {epoch}, Val Loss: {val_loss:.10f}, Last Val Loss: {last_val_loss:.10f}")

            all_train_loss.append(current_train_loss)
            all_val_loss.append(val_loss)

            if val_loss < last_val_loss:
                # save ckpt
                self.logger.info(f"Saving ckpt...")
                torch.save(self.model.state_dict(), os.path.join(self.output_path, f'ckpt.pth'))

        torch.save(self.model.state_dict(), os.path.join(self.output_path, f'last.pth'))
        self.logger.info(f"Training finished, total training time: {time.time() - time_now:.4f}s")
        plot_loss('train_loss', all_train_loss, self.output_path)
        plot_loss('val_loss', all_val_loss, self.output_path)

        self.logger.info('Testing...')
        self.test_best_ckpt(test_data_set, test_data_loader, test_competitor_tensor, criterion)



    def val(self, val_data, val_loader, val_competitor, criterion):
        self.model.eval()
        val_loss = []
        all_target = []
        all_output = []
        val_competitor = val_competitor.to(self.device)
        for i, (input_seq, target_seq) in enumerate(val_loader):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            output_seq = self.model(input_seq, val_competitor)
            all_target.append(target_seq.detach().cpu())
            all_output.append(output_seq.detach().cpu())
            loss = criterion(output_seq, target_seq)
            val_loss.append(loss.item())
        return np.mean(val_loss), all_target, all_output

    def test(self, test_data_set, test_data_loader, test_competitor, criterion, test_results_file='metrics.json'):
        self.model.eval()
        test_loss, test_targets, test_outputs = self.val(test_data_set, test_data_loader, test_competitor, criterion)
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
            test_metrix = {
                'MSE': mse_score.item(),
                'RMSE': rmse_score.item(),
                'MAE': mae_score.item(),
                'ADE': ade.item(),
                'FDE': fde.item()
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
            test_metrix = {
                'Accuracy': acc.item(),
                'F1': f1.item(),
                'MCC': mcc.item()
            }

        metrics = {
            'Model': self.args.model,
            'Hyper_Parameters': {
                'lr': self.args.lr,
                'batch_size': self.args.batch_size,
                'epochs': self.args.epochs,
                'dropout': self.args.dropout,
                'hidden_size': self.args.hidden_size,
                'kernel_size': self.args.kernel_size,
                'stride': self.args.stride,
                'padding': self.args.padding,
                'lstm_num_layers': self.args.lstm_num_layers,
                'cnn_num_layers': self.args.cnn_num_layers,
                'alpha': self.args.alpha,
            },
            'target': self.args.target,
            'Metrics': test_metrix
        }

        with open(os.path.join(self.output_path, test_results_file), 'w') as f:
            json.dump(metrics, f, indent=4)

    def test_best_ckpt(self, test_data_set, test_data_loader, test_competitor_tensor, criterion):
        self.model.load_state_dict(torch.load(os.path.join(self.output_path, 'ckpt.pth'), map_location=self.device))
        self.test(test_data_set, test_data_loader, test_competitor_tensor, criterion)

    def test_on_multi_checkpoint(self):
        test_data_set, test_competitor, test_data_loader = self._get_data('test')
        test_competitor_data_set = self._get_competitor_data('test')
        test_competitor_matrix = test_competitor_data_set[0]
        test_competitor_tensor = torch.from_numpy(test_competitor_matrix).float().to(self.device)
        self.logger.info('Test data loaded successfully!')

        criterion = self._set_criterion()

        all_test_loss = []
        all_test_metrics = []


        for checkpoint in sorted(os.listdir(self.args.checkpoint_path)):
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoint_path, checkpoint), map_location=self.args.device))
            self.model.eval()
            test_loss, test_targets, test_outputs = self.val(test_data_set, test_data_loader, test_competitor_tensor, criterion)
            all_test_loss.append(test_loss)
            self.logger.info(f"{checkpoint}: Test Loss: {test_loss:.10f}")

            all_test_targets = torch.cat(test_targets, dim=0)
            all_test_outputs = torch.cat(test_outputs, dim=0)

            if self.args.target == 'price':
                mse, rmse, mae, ade, fde = regression_metrics(all_test_targets, all_test_outputs)
                all_test_metrics.append([mse, rmse, mae, ade, fde])
                self.logger.info(f'{checkpoint}: MSE:{mse}, RMSE:{rmse}, MAE:{mae}, ADE:{ade}, FDE:{fde}')
            else:
                threshold = 0.5
                all_test_outputs = torch.sigmoid(all_test_outputs)
                all_test_outputs = (all_test_outputs > threshold).float()

                target_0s, target_1s = calculate_label_num(all_test_targets)
                output_0s, output_1s = calculate_label_num(all_test_outputs)

                accuracy, f1, mcc = classification_metrics(all_test_targets, all_test_outputs)
                all_test_metrics.append([accuracy, f1, mcc])
                self.logger.info(f'{checkpoint}:'
                                 f'target_0s:{target_0s}, target_1s:{target_1s}, '
                                 f'output_0s:{output_0s}, output_1s:{output_1s},'
                                 f'Accuracy:{accuracy}, F1:{f1}, MCC:{mcc}')
            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()

        plot_loss('test_loss', all_test_loss, self.args.test_results_path)
        if self.args.target == 'price':
            mses = [result[0] for result in all_test_metrics]
            rmses = [result[1] for result in all_test_metrics]
            maes = [result[2] for result in all_test_metrics]
            ades = [result[3] for result in all_test_metrics]
            fdes = [result[4] for result in all_test_metrics]

            plot_loss('MSE', mses, self.args.test_results_path)
            plot_loss('RMSE', rmses, self.args.test_results_path)
            plot_loss('MAE', maes, self.args.test_results_path)
            plot_loss('ADE', ades, self.args.test_results_path)
            plot_loss('FDE', fdes, self.args.test_results_path)
        else:
            accuracys = [result[0] for result in all_test_metrics]
            f1s = [result[1] for result in all_test_metrics]
            mccs = [result[2] for result in all_test_metrics]
            plot_loss('Accuracy', accuracys, self.args.test_results_path)
            plot_loss('F1', f1s, self.args.test_results_path)
            plot_loss('MCC', mccs, self.args.test_results_path)




