import pandas as pd
import numpy as np
from datetime import timedelta

def get_time_block_series(series_array, date_to_index, start_date, end_date):
    inds = date_to_index[start_date:end_date]
    return series_array[:, inds]

def transform_series(series_array):
    series_array = np.nan_to_num(series_array)
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1]))
    return series_array

def prepare_data(df_path, data_start_date, data_end_date, pred_steps):
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

    return input_data, target_data

def remove_invalid_stocks(input_data, target_data, val_input_data, val_target_data):
    data_sets = [input_data, target_data, val_input_data, val_target_data]
    valid_stocks = set(range(input_data.shape[0]))  # 初始化为所有股票的索引集合

    for dataset in data_sets:
        invalid_stocks = np.where(np.isnan(dataset).all(axis=1))[0]
        valid_stocks -= set(invalid_stocks)

    valid_stocks = list(valid_stocks)

    return (input_data[valid_stocks], target_data[valid_stocks],
            val_input_data[valid_stocks], val_target_data[valid_stocks])


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