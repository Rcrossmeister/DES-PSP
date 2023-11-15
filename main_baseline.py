from exp.exp_baseline import Exp_Baseline

import torch
import torch.nn as nn
import pandas
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='stock prediction')

# mode
parser.add_argument('--mode', type=str, default='train', help='train/val/test')
parser.add_argument('--pred_steps', type=int, default=14, help='prediction steps')

# data path
parser.add_argument('--root_path', type=str, default='npy_path/', help='npy path')
parser.add_argument('--train_data_path', type=str, default='08', help='train data file path')
parser.add_argument('--val_data_path', type=str, default='11', help='val data file path')
parser.add_argument('--test_data_path', type=str, default='11', help='test data file path')
# train data name
parser.add_argument('--train_input_file', type=str, default='all_input.npy', help='train data input file name')
parser.add_argument('--train_target_pr_file', type=str, default='all_target_pr.npy', help='train data target price file name')
parser.add_argument('--train_target_mo_file', type=str, default='all_target_mo.npy', help='train data target movement file name')
# val data name
parser.add_argument('--val_input_file', type=str, default='pre_input.npy', help='val data file name')
parser.add_argument('--val_target_pr_file', type=str, default='pre_target_pr.npy', help='val data target price file name')
parser.add_argument('--val_target_mo_file', type=str, default='pre_target_mo.npy', help='val data target movement file name')
# test data name
parser.add_argument('--test_input_file', type=str, default='pre_input.npy', help='test data file name')
parser.add_argument('--test_target_pr_file', type=str, default='pre_target_pr.npy', help='test target price data file name')
parser.add_argument('--test_target_mo_file', type=str, default='pre_target_mo.npy', help='test target movement data file name')
# results path
parser.add_argument('--results_path', type=str, default='results_88', help='result path')

# scale config
parser.add_argument('--scale', type=bool, default=True, help='scale data')
parser.add_argument('--inverse', type=bool, default=False, help='inverse data')

# model config
parser.add_argument('--target', type=str, default='price', help='target name, option[price, movement]')
parser.add_argument('--model', type=str, default='lstm', help='model name')
parser.add_argument('--input_size', type=int, default=1, help='input size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--output_size', type=int, default=1, help='output size')
parser.add_argument('--num_layers', type=int, default=5, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')

# device config
parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')

# seed
parser.add_argument('--seed', type=int, default=88, help='random seed')

# training config
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

# test config
parser.add_argument('--test_only', action='store_true', help='test only', default=False)
parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint path')
parser.add_argument('--test_results_path', type=str, default=None, help='test results path')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


def main():
    exp = Exp_Baseline(args)
    if args.test_only:
        exp.test_only()
    else:
        exp.train()


if __name__ == '__main__':
    main()
