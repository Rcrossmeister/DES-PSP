import pandas as pd

all_data = pd.read_csv('')

with open('') as fA:
    text = fA.read()
    fA.close()

text = text.split(',')
list1 = []

for _ in text:
    _ =  ' '.join(_.split())
    list1.append(_)

df1 = df[df['Stock_id'].isin(list1)]

set1 = set(list1)
if len(list1) != len(set1):
 
    duplicates = [item for item in set1 if list1.count(item) > 1]
    print('list1中的重复元素为：', duplicates)
else:
    print('list1中没有重复元素')