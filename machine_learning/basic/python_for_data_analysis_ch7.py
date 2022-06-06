# 数据清洗和准备
# 缺失值 一般是用 np.nan
# 可以重写成 from numpy import nan as NA
from numpy import nan as NA
import pandas as pd
import cv2 as cv
import numpy as np

string_data = pd.Series(['goon', 'myalos', NA, 'tidus'])
print(string_data)
# 使用None 也能被当作是NA 不过显示出来的是None 而不是NaN
string_data[0] = None
print(string_data)
print(string_data.isnull())

string_data.fillna(5.2, inplace = True)
print(string_data)

df = pd.DataFrame(np.random.randn(6, 3))

df.iloc[2:, 1] = NA
df.iloc[4:, 2] = NA
print(df)
df.fillna(method = 'ffill', limit = 2, inplace = True)
print(df)

# Data Transformation
# 这个部分主要是进行rearranging data
# Removing Duplicates
data = pd.DataFrame({'k1' : ['one', 'two'] * 3 + ['two'],
    'k2' : [1, 1, 2, 3, 3, 4, 4]})
print(data.duplicated()) # 这个是按所有列进行的

print(data.duplicated(subset=['k1']))
print(data.duplicated(subset=['k2']))

data['v1'] = range(7)
print(data)
print(data.drop_duplicates(['k2']))

# Transforming Data Using a Function or Mapping
# 用map函数来进行element-wise的transformation
data2 = pd.DataFrame({'food' : ['bacon', 'pulled pork', 'bacon', 'Pastrami', 'corned beef', 'Bacon', 'pastrami', 'honey ham', 'nova lox'], 'ounces' : [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

meat_to_animal = {
        'bacon' : 'pig',
        'pulled pork' : 'pig',
        'pastrami' : 'cow',
        'corned beef' : 'cow',
        'honey ham': 'pig',
        'nova lox' : 'salmon'
}

lowercased = data2['food'].str.lower()
print(lowercased)
data2['animal'] = lowercased.map(meat_to_animal)
print(data2)

# replace 方法 有data.replace 和 data.str.replace 两种不同
data3 = pd.Series([1., -999., 2,  3., -999, -1000., 3])
data3.replace([-999, -1000.], np.nan, inplace= True)
print(data3)


