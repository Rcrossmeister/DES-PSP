from data_loader.data_loader import Dataset_Stock, DataSet_Competitor
from exp.exp_basic import Exp_Basic
from models.model import DES_PSP_Model
from utils.plotter import plot_loss
from utils.metrics import compute_metrics, acc, MCC
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
import pprint  # 打印漂亮的结果
import gc  # 手动释放内存

class Exp_DES_PSP(Exp_Basic):
    def __init__(self, args):
        # 初始化时，继承Exp_Basic
        super(Exp_DES_PSP, self).__init__(args)
        # 记录当前时间
        current_datetime = time.strftime("%Y-%m-%d_%H-%M")
        if self.args.test_only:
            self.output_path = self.args.test_results_path
        else:
            self.output_path = os.path.join(self.args.results_path, f'{self.args.target}_{self.args.model}_{current_datetime}')
        if not os.path.exists(self.output_path):
            # 如果output_path指定的路径不存在，则创建他
            os.makedirs(self.output_path)
        if not os.path.exists(os.path.join(self.output_path, 'ckp')):
            # 如果output_path下面的子路径ckp不存在，则创建他
            os.makedirs(os.path.join(self.output_path, 'ckp'))
        # 在output_path中初始化logger
        self.logger = init_logger(self.output_path)

    def _build_model(self):
        model_dict = {
            'des_psp': DES_PSP_Model
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

        if self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if flag == 'train':
            shuffle_flag = True
            drop_last = False
            batch_size = self.args.batch_size
            data_path = self.args.other_data_path
            start_date = self.args.data_start_date
            end_date = self.args.data_end_date
        elif flag == 'val':
            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_size // 2
            data_path = self.args.biden_data_path
            start_date = self.args.val_start_date
            end_date = self.args.val_end_date
        else:
            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_size // 2
            data_path = self.args.biden_data_path
            start_date = self.args.val_start_date
            end_date = self.args.val_end_date

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

    def _get_competitor_data(self, flag):
        if flag == 'train':
            data_path = self.args.other_data_path
            start_date = self.args.data_start_date
            end_date = self.args.data_end_date
        elif flag == 'val':
            data_path = self.args.trump_data_path
            start_date = self.args.val_start_date
            end_date = self.args.val_end_date
        else:
            data_path = self.args.trump_data_path
            start_date = self.args.val_start_date
            end_date = self.args.val_end_date
        data_set = DataSet_Competitor(
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
        train_data_set, train_data_loader = self._get_data('train')
        # 相比于baseline，多了一个competitor data set
        train_competitor_data_set = self._get_competitor_data('train')
        # 因为我们另外一个encoder用的是CNN，所以这里直接把competitor当成一个matrix输入进去
        train_competitor_matrix = train_competitor_data_set[0]
        train_competitor_tensor = torch.from_numpy(train_competitor_matrix).float().to(self.device)
        self.logger.info('Train data loaded successfully!')

        val_data_set, val_data_loader = self._get_data('val')
        val_competitor_data_set = self._get_competitor_data('val')
        val_competitor_matrix = val_competitor_data_set[0]
        val_competitor_tensor = torch.from_numpy(val_competitor_matrix).float().to(self.device)
        self.logger.info('Val data loaded successfully!')

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
                output_seq = self.model(input_seq, train_competitor_tensor)
                loss = criterion(output_seq, target_seq)
                train_loss.append(loss.item())
                optim.zero_grad()
                loss.backward()
                optim.step()
                self.logger.info(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item():.10f}")

            current_train_loss = np.mean(train_loss)
            self.logger.info(f"Epoch: {epoch}, Train Loss: {current_train_loss:.10f}")
            # val_loss, val_targets, val_outputs = self.val(val_data_set, val_data_loader, val_competitor_tensor, criterion)
            # self.logger.info(f"Epoch: {epoch}, Val Loss: {val_loss:.10f}")

            all_train_loss.append(current_train_loss)
            # all_val_loss.append(val_loss)
            torch.cuda.empty_cache()
            torch.save(self.model.state_dict(), os.path.join(self.output_path, f'ckp/{epoch:04}.pth'))
        plot_loss('train_loss', all_train_loss, self.output_path)
        val_loss, val_targets, val_outputs = self.val(val_data_set, val_data_loader, val_competitor_tensor, criterion)
        all_val_loss.append(val_loss)
        self.logger.info(f"Training finished, total training time: {time.time() - time_now:.4f}s")
        plot_loss('val_loss', all_val_loss, self.output_path)

        self.logger.info('Calculating Metrics...')
        all_val_targets = torch.cat(val_targets, dim=0)
        all_val_outputs = torch.cat(val_outputs, dim=0)
        if self.args.target == 'price':
            mse, rmse, mae, ade, fde = regression_metrics(all_val_targets, all_val_outputs)
            self.logger.info(f'''
                                    Metrics on Val set:
                                    MSE:{mse},
                                    RMSE:{rmse},
                                    MAE:{mae},
                                    ADE:{ade},
                                    FDE:{fde}
                    ''')
        else:
            threshold = 0.5
            all_val_outputs = torch.sigmoid(all_val_outputs)
            all_val_outputs = (all_val_outputs > threshold).float()

            target_0s, target_1s = calculate_label_num(all_val_targets)
            output_0s, output_1s = calculate_label_num(all_val_outputs)

            accuracy, f1, mcc = classification_metrics(all_val_targets, all_val_outputs)
            self.logger.info(f'''
                                    Target 0s: {target_0s}, Target 1s: {target_1s}
                                    Output 0s: {output_0s}, Output 1s: {output_1s}
                                    Metrics on Val set:
                                        Accuracy:{accuracy},
                                        F1:{f1},
                                        MCC:{mcc}
                    ''')
        self.logger.info('Start saving model...')


    def val(self, val_data, val_loader, val_competitor, criterion):
        self.model.eval()
        val_loss = []
        all_target = []
        all_output = []
        for i, (input_seq, target_seq) in enumerate(val_loader):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            output_seq = self.model(input_seq, val_competitor)
            all_target.append(target_seq.detach().cpu())
            all_output.append(output_seq.detach().cpu())
            loss = criterion(output_seq, target_seq)
            val_loss.append(loss.item())
        return np.mean(val_loss), all_target, all_output

    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoint_path, f'{self.args.model}.pth')))
        self.model.eval()
        self.logger.info('Model loaded successfully!')

    def test_on_multi_checkpoint(self):
        test_data_set, test_data_loader = self._get_data('test')
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




