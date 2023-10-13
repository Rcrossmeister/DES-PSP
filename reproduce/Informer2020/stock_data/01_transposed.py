import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('D:\\#Code\\DES-PSP_hjd\\df_path\\All_Data.csv')

# 设置日期列作为索引
df.set_index('Stock_id', inplace=True)

# 转置DataFrame（行列互换）
df_transposed = df.transpose()

# 重置索引
df_transposed.reset_index(inplace=True)

# 修改列名
df_transposed = df_transposed.rename(columns={'index': 'date'})

# 保存新的CSV文件
df_transposed.to_csv('transformed.csv', index=False)