import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

now = datetime.now()

print(now) # 这个格式化打印 会打印出2022-06-01 09:06:01.3787
print(now.year, now.month, now.day)

# 两个datetime相减
delta = datetime(2022, 6, 2) - datetime(2022, 5, 31, 8, 15)
print(delta) #前面是day 后面是time
print(delta.days, delta.seconds)

start = datetime(2011, 1, 7)
end = start + timedelta(12)
print("start + timedelta(12) : ", end) # timedelta的单位是day

# datetime和string之间的转化
# 使用strftime 和 strptime
stamp = datetime(2011, 1, 3)
print(str(stamp))
print(stamp.strftime('%Y-%m-%d'))

value = '2011-1-3'
time = datetime.strptime(value, '%Y-%m-%d')
print(time)

# 上面需要手写格式，可以自动识别的，方法就是
from dateutil.parser import parse
print(parse('2011-01-02'))
print(parse('Jan 31, 1997 10:45 PM'))

# pandas的to_datetime可以的
print(pd.to_datetime('2011-01-02'))

#  a Series indexed by timestamps
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7), datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = pd.Series(np.random.randn(6), index = dates)
print(ts)

# 这个会对准时间进行相加
print(ts + ts[::2])

dates1 = [datetime(2011, 1, 2, 10, 20), datetime(2011, 1, 5, 10, 20), datetime(2011, 1, 7, 10, 20), datetime(2011, 1, 8, 10, 20), datetime(2011, 1, 10, 11, 5), datetime(2011, 1, 12, 20, 20)]
ts1 = pd.Series(np.random.randn(6), index = dates1)
print(ts1)
# 对ts进行索引的时候 可以
print(ts['1/10/2011'])
print(ts['20110110'])

# 对于长时间的时间序列，可以使用pd.date_range这个函数
longer_ts = pd.Series(np.random.randn(1000), index = pd.date_range('1/1/2000', periods=1000))
print(longer_ts)

# 索引可以用年份来索引 牛逼啊
print(longer_ts['2001'])
print(longer_ts['2001-06'])

# 日期做slice 进行range query
print(ts)
print(ts['1/6/2011' : '1/11/2011'])
# 这个slice 会修改原来的内容
# no data is copied and modifications on the slice will be refelcted in the original data
ts['1/6/2011' : '1/11/2011'] = 1
print(ts)

# 下面是另一种date_range
# 因为2000年1月1日 不是星期三 所以第一个是2000-01-05
dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = pd.DataFrame(np.random.randn(100, 4), index = dates, columns = [list('abcd')])
print(long_df)

print(long_df.loc['2001-5'])

# 下面挺重要的是 关于duplicate indices
# 下面这种日期表达也是可以写出来的
dates2 = pd.DatetimeIndex(['2000.1.1', '2000.1.2', '2000.1.2', '2000.1.2', '2000.1.3'])
dup_ts = pd.Series(np.arange(5), index = dates2)
print(dup_ts)

print(dup_ts.index.is_unique)
print(dup_ts.groupby(level = 0).count())

# 下面是Date Ranges, Frequencies, and Shifting
# 将不是固定的frequency转换成fixed-frequency的方法
print(ts)
print(ts.resample('D'))
# 下面这个比较神奇
pd.date_range('2000-01-01', '2000-12-01', freq = 'BM')
# 上面这个是Last Business Day of a Month 有点神奇
# 下面的是date_range的用法
print(pd.date_range('2012-05-02 12:53:42', periods=5))
print(pd.date_range('2012-05-02 12:53:42', periods=5, normalize=True))

# Frequencies 和 Date Offsets
# base frequency和一个multiplier
# 比如 freq = '4H'
print(pd.date_range('2000-01-01', periods=10, freq = '1h30min'))
# WOM 表示 week of month
# 下面两种输出 list里面的元素是Timestamp(时间, freq = 间隔)
rng = pd.date_range('2012-01-01', '2012-09-01', freq = 'WOM-3FRI')
print(list(rng))
rng1 = pd.date_range('2012-01-01', periods=10, freq='1h20min')
print(list(rng1))

## 重要的shift操作，这个操作是将数据进行移动
ang = pd.Series(np.random.randn(4), index = pd.date_range('1/1/2000', periods=4, freq = 'M'))
print(ang)
print(ang.shift(2))
# 这个shift算时间差上的数据差非常棒
# 可以用shift来算percent changes in a time series
print(ang - ang.shift(1))
print(ang / ang.shift(1) - 1)
# 如果shift里面带freq参数 那么shift是位置是index


