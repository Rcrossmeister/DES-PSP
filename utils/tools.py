import pandas as pd
import numpy as np
from datetime import timedelta
import logging
import time

def init_logger(log_path: str):
    logger  = logging.getLogger()
    # 设置日志级别和格式
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # 获取当前日期和时间作为日志文件名的一部分
    current_datetime = time.strftime("%Y-%m-%d_%H-%M")
    log_file = f"{log_path}/{current_datetime}.log"

    # 创建一个文件处理器，用于将日志输出到文件
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 将文件处理器添加到日志记录器中
    logger = logging.getLogger('')
    logger.addHandler(file_handler)

    return logger
def get_time_block_series(series_array, date_to_index, start_date, end_date):
    inds = date_to_index[start_date:end_date]
    return series_array[:, inds]

def transform_series(series_array):
    series_array = np.nan_to_num(series_array)
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1]))
    return series_array

def prepare_data(df_path, data_start_date, data_end_date, pred_steps, target='price'):
    df = pd.read_csv(df_path)

    print('Data ranges from %s to %s' % (data_start_date, data_end_date))

    pred_length = timedelta(pred_steps)

    first_day = pd.to_datetime(data_start_date)
    last_day = pd.to_datetime(data_end_date)

    train_pred_start = last_day + timedelta(days=1)
    train_pred_end = train_pred_start + pred_length - timedelta(days=1)

    train_start = first_day
    train_end = last_day

    date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df.columns[1:]]),
                              data=[i for i in range(len(df.columns[1:]))])
    series_array = df[df.columns[1:]].values

    input_data = get_time_block_series(series_array, date_to_index, train_start, train_end)
    input_data = transform_series(input_data)

    target_data = get_time_block_series(series_array, date_to_index, train_pred_start, train_pred_end)
    target_data = transform_series(target_data)

    if target == 'movement':
        for i in range(target_data.shape[0]):
            first_day_value = target_data[i, 0]
            target_data[i, :] = (target_data[i, :] >= np.roll(target_data[i, :], shift=1)).astype(int)
            target_data[i, 0] = (first_day_value >= input_data[i][-1]).astype(int)

    return input_data, target_data

def remove_invalid_stocks(input_data, target_data):
    data_sets = [input_data, target_data]
    valid_stocks = set(range(input_data.shape[0]))  # 初始化为所有股票的索引集合

    for dataset in data_sets:
        invalid_stocks = np.where(np.isnan(dataset).all(axis=1))[0]
        valid_stocks -= set(invalid_stocks)

    valid_stocks = list(valid_stocks)

    return (input_data[valid_stocks], target_data[valid_stocks])


if __name__ == '__main__':
    df_path = '/home/hzj/NLP1/StockPricePrediction/rc_ross/All_Data.csv'
    data_start_date = '2015/11/09'
    data_end_date = '2016/11/08'

    val_start_date = '2019/12/15'
    val_end_date = '2020/12/14'

    pred_steps = 14

    train_input_data, train_target_data = prepare_data(df_path, data_start_date, data_end_date, pred_steps)
    val_input_data, val_target_data = prepare_data(df_path, val_start_date, val_end_date, pred_steps)

    train_input_data, train_target_data, val_input_data, val_target_data = remove_invalid_stocks(train_input_data,
                                                                                                train_target_data,
                                                                                                val_input_data,
                                                                                                val_target_data)