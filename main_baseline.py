from exp.exp_baseline import Exp_Baseline

import torch
import torch.nn as nn
import pandas
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='stock prediction')

# data config
parser.add_argument('--root_path', type=str, default='df_path/', help='data frame path')
parser.add_argument('--all_data_path', type=str, default='All_Data.csv', help='data file name')
parser.add_argument('--trump_data_path', type=str, default='GroupB.csv', help='data file name')
parser.add_argument('--biden_data_path', type=str, default='GroupA.csv', help='data file name')
parser.add_argument('--result_path', type=str, default='results_88', help='result path')
parser.add_argument('--remove_invaild', type=bool, default=False, help='remove invalid stocks')
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
parser.add_argument('--model', type=str, default='lstm', help='model name')
parser.add_argument('--input_size', type=int, default=1, help='input size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--output_size', type=int, default=14, help='output size')
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

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


def main():
    exp = Exp_Baseline(args)
    exp.train()


if __name__ == '__main__':
    main()    
