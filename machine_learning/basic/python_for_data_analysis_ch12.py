import numpy as np
import pandas as pd

values = pd.Series(['apple', 'orange', 'apple', 'orange'] * 2)

print(values)

print(pd.unique(values))
print(values.unique())

# dimension tables 这个里面containing the distinct values and storing the primary observations as integer keys referencing the dimension table
values = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(['apple', 'orange'])
# 这个take方法很帅啊
dim.take(values)
print(dim)
# 将dim和values融合起来
print(dim.memory_usage()) # 144
dim = dim.astype('category')
print(dim.memory_usage()) # 254
print(dim)

c = dim.values # 这个不是numpy的ndarray 而是pandas.Categorical
print(dim.values)
print(c.codes)

# pd.Categorical数据的构造方法
my_cat = pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
print(my_cat)
# 或者用code + category的方法
categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
my_cat2 = pd.Categorical.from_codes(codes, categories)
print(my_cat2)
# 可以让categorical的数据具有order
print(my_cat2.as_ordered())
# 书上说some parts of pandas, like the groupby function, perform better when working with categoricals

# qcut方法会将数值进行分类
draws = np.random.randn(1000)
bins = pd.qcut(draws, 4)
print(bins)
bins2 = pd.qcut(draws , 4, labels = ['Q1', 'Q2', 'Q3', 'Q4'])
print(bins2)

bins = pd.Series(bins2, name = 'quartile') # name属性方便之后的DataFrame
results = pd.Series(draws).groupby(bins2).agg(['count', 'min', 'max']).reset_index()
print(results)

# 对于category类型的数据
# 一般的pd.Series的value_counts方法会对Series进行count，如果是category的类型的Series进行value_counts的话，就会对所有的category进行count，会存在0的category
# 可以用remove_unused_categories来 trim unobserved categories

# get_dummies方法 会把object 和category类变成one-hot code
a = pd.DataFrame({'age' : [11, 23], 'name' : ['goon', 'myalos']})
print(pd.get_dummies(a))

# 高级的groupby方法


# assign方法 牛逼
# df2 = df.assign(k = v) 等价于
# df2 = df.copy()
# df2['k'] = v




