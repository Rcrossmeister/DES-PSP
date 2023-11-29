from exp.exp_DES_PSP import Exp_DES_PSP

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
parser.add_argument('--train_input_file', type=str, default='pre_input.npy', help='train data input file name')
parser.add_argument('--competitor_noise', type=bool, default=False, help='train data competitor noise')
parser.add_argument('--train_competitor_file', type=str, default='com_input.npy', help='train data competitor file name')
parser.add_argument('--train_target_pr_file', type=str, default='pre_target_pr.npy', help='train data target price file name')
parser.add_argument('--train_target_mo_file', type=str, default='pre_target_mo.npy', help='train data target movement file name')
# val data name
parser.add_argument('--val_input_file', type=str, default='pre_input.npy', help='val data file name')
parser.add_argument('--val_competitor_file', type=str, default='com_input.npy', help='val data competitor file name')
parser.add_argument('--val_target_pr_file', type=str, default='pre_target_pr.npy', help='val data target price file name')
parser.add_argument('--val_target_mo_file', type=str, default='pre_target_mo.npy', help='val data target movement file name')
# test data name
parser.add_argument('--test_input_file', type=str, default='pre_input.npy', help='test data file name')
parser.add_argument('--test_competitor_file', type=str, default='com_input.npy', help='test data competitor file name')
parser.add_argument('--test_target_pr_file', type=str, default='pre_target_pr.npy', help='test target price data file name')
parser.add_argument('--test_target_mo_file', type=str, default='pre_target_mo.npy', help='test target movement data file name')
# results path
parser.add_argument('--results_path', type=str, default='results_88', help='result path')

# scale config
parser.add_argument('--scale', type=bool, default=True, help='scale data')
parser.add_argument('--inverse', type=bool, default=False, help='inverse data')

# model config
parser.add_argument('--target', type=str, default='price', help='target name')
parser.add_argument('--model', type=str, default='des_psp', help='model name, options: [des_psp, des_psp_l]')
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


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


def main():
    exp = Exp_DES_PSP(args)

    if args.test_only:
        exp.test_on_multi_checkpoint()
    else:
        exp.train()


if __name__ == '__main__':
    main()
