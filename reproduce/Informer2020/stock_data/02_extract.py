import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('./stock_data/transformed.csv')

df['date'] = pd.to_datetime(df['date'])

# 设定数据的起始日期和结束日期
train_data_start_date = '2015/11/09'
train_data_end_date = '2016/11/08'
test_data_start_date = '2019/12/15'
test_data_end_date = '2020/12/14'


pred_len = 14

train_start_date = pd.to_datetime(train_data_start_date)
train_end_date = pd.to_datetime(train_data_end_date) + pd.Timedelta(days=pred_len)
test_start_data = pd.to_datetime(test_data_start_date)
test_end_date = pd.to_datetime(test_data_end_date) + pd.Timedelta(days=pred_len)


train_filtered_df = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date )]
test_filtered_df = df[(df['date'] >= test_start_data) & (df['date'] <= test_end_date )]

# 把时间从yyyy/mm/dd转换成mm/dd/yyyy

train_filtered_df['date'] = train_filtered_df['date'].dt.strftime('%m/%d/%Y')
test_filtered_df['date'] = test_filtered_df['date'].dt.strftime('%m/%d/%Y')

# 合并两个dataframe，索引重置
merged_df = pd.concat([train_filtered_df, test_filtered_df], ignore_index=True)

# 保存新的CSV文件
merged_df.to_csv('./stock_data/merged.csv', index=False)