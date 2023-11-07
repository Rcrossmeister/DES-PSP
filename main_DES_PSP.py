from exp.exp_DES_PSP import Exp_DES_PSP

import torch
import torch.nn as nn
import pandas
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='stock prediction')

# mode
parser.add_argument('--mode', type=str, default='train', help='train/val/test')

# data config
parser.add_argument('--root_path', type=str, default='df_path/', help='data frame path')
parser.add_argument('--all_data_path', type=str, default='All_Data.csv', help='data file name')
parser.add_argument('--trump_data_path', type=str, default='GroupB.csv', help='data file name')
parser.add_argument('--biden_data_path', type=str, default='GroupA.csv', help='data file name')
parser.add_argument('--other_data_path', type=str, default='GroupC.csv', help='data file name')
parser.add_argument('--results_path', type=str, default='results_88', help='result path')
parser.add_argument('--remove_invalid', type=bool, default=False, help='remove invalid stocks')
parser.add_argument('--scale', type=bool, default=True, help='scale data')
parser.add_argument('--inverse', type=bool, default=False, help='inverse data')

# date config
parser.add_argument('--data_start_date', type=str, default='2015/11/09', help='data start date')
parser.add_argument('--data_end_date', type=str, default='2016/11/08', help='data end date')
parser.add_argument('--val_start_date', type=str, default='2019/12/15', help='validation start date')
parser.add_argument('--val_end_date', type=str, default='2020/12/14', help='validation end date')
parser.add_argument('--pred_steps', type=int, default=14, help='prediction steps')

# model config
parser.add_argument('--target', type=str, default='price', help='target name')
parser.add_argument('--model', type=str, default='des_psp', help='model name')
parser.add_argument('--input_size', type=int, default=1, help='input size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--output_size', type=int, default=1, help='output size')
parser.add_argument('--kernel_size', type=int, default=3, help='cnn kernel size')
parser.add_argument('--stride', type=int, default=1, help='cnn stride size')
parser.add_argument('--padding', type=int, default=1, help='cnn padding size')
parser.add_argument('--lstm_num_layers', type=int, default=5, help='number of lstm layers')
parser.add_argument('--cnn_num_layers', type=int, default=5, help='number of cnn layers')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')

# device config
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

# seed
parser.add_argument('--seed', type=int, default=88, help='random seed')

# training config
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

# test_only
parser.add_argument('--test_only', action='store_true', help='test only', default=False)
parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint path')
parser.add_argument('--test_results_path', type=str, default=None, help='test results path')



args = parser.parse_args()

if args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.device = args.device_ids[0]

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


def main():
    exp = Exp_DES_PSP(args)

    if args.test_only:
        exp.test_on_multi_checkpoint()
    else:
        exp.train()


if __name__ == '__main__':
    main()
