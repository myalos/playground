import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 4个子图里面可视化3个
# 在之前的playground/mnist/linear.py中
# 画图的方法是
# fig = plt.figure()
# plt.subplot(2, 1, 1)
# plt.title("loss")
# plt.plot(....)
# plt.plot(....)
# plt.legend()
# ...
# plt.show()

def func1(flag = False):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    if flag:
        ax1.hist(np.random.randn(100), bins = 20, color = 'k', alpha = 0.3)
    ax2 = fig.add_subplot(2, 2, 2)
    if flag:
        ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
    ax3 = fig.add_subplot(2, 2, 3)
    if flag:
        # 这里的k--是一个缩写
        # 代表着 颜色是k 线的风格是linestyle = '--'
        # 还有一个marker 这个东西
        # plt.plot(randn(30).cumsum(), 'ko--')
        # 相当于 marker = 'o'
        ax3.plot(np.random.randn(50).cumsum(), 'k--')

    plt.show()

# 一个一个的add_subplot之外 还可以一起add 方法是
# fig, axes = plt.subplots(2, 3)
# 引用的时候 可以使用axes[0, 1] 或者 axes[1, 1]
# subplots之间可以sharex - True 让这些数据有相同的scale
# sharex 设置为True的话 那么每个axes的横坐标就是相同的
def func2():
    fig, axes = plt.subplots(2, 2, sharex = True)
    axes[0, 0].hist(np.random.randn(100), bins = 20, color = 'k', alpha = 0.3)
    axes[0, 1].scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
    axes[1, 0].plot(np.random.randn(50).cumsum(), 'k--', marker = 'o')
    plt.show()

# 调整不同子图间的距离的方法是 subplots_adjust
def func3():
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for i in range(2):
        for j in range(2):
            axes[i, j].hist(np.random.randn(500), bins = 50, color = 'k', alpha = 0.5)
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.show()

def func4():
    data = np.random.randn(30).cumsum()
    plt.plot(data, 'k--', label = 'Default')
    plt.plot(data, 'k-', drawstyle = 'steps-post', label = 'steps-post')
    plt.legend(loc = 'best')
    plt.show()

# Ticks Labels and Legends
# matplotlib 的pyplot接口是设计用来进行interactive use，里面包含的方法有xlim, xticks, xticklabels等
# 这是三个方法的作用是control the plot range, tick locations, tick labels
def func5():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.random.randn(1000).cumsum(), marker = 'o', markersize = 2)
    _range = ax.get_xlim()
    ax.set_xticks(np.linspace(_range[0],_range[1], 10))
    ax.set_xticklabels(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])
    plt.show()

# Annotations and Drawing on a Subplot
# 自定义的annotation 里面包括了 text, arrows, 和其他shapes
# text方法，arrow方法，annotate方法

# pandas里面的可视化
def func6():
    s = pd.Series(np.random.randn(10).cumsum(), index = np.arange(0, 100, 10))
    s.plot()
    # dataframe的plot plots each of its columns as a different line one the same subplot, creating a legend automatically
    # 每列是一个line
    df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),
            columns=list("ABCD"),
            index = np.arange(0, 100, 10))
    df.plot(subplots = True)
    # 如果加上subplots = True的话 会把每列的数据画到一个子图里面去
    # 运行之后会出两张图
    plt.show()

# 使用bar plots来画图
# 用DataFrame的plot来进行画图
def func7():
    fig, axes = plt.subplots(2, 1)
    data = pd.Series(np.random.rand(16), index = list('abcdefghijklmnop'))
    data.plot.bar(ax = axes[0], color = 'k', alpha= 0.7)
    data.plot.barh(ax = axes[1], color = 'k', alpha= 0.7)

    df = pd.DataFrame(np.random.rand(6, 4),
            index = ['one', 'two', 'three', 'four', 'five', 'six'],
            columns=pd.Index(list('ABDC'), name = 'Genus'))
    df.plot.bar()
    # 下面是横着画
    df.plot.barh(stacked = True, alpha = 0.5)
    plt.show()

def func8():
    #使用seaborn 来画histogram加上density 和散点图
    comp1 = np.random.normal(0, 1, size = 200)
    comp2 = np.random.normal(10, 2, size = 200)
    values = pd.Series(np.concatenate([comp1, comp2]))
    sns.distplot(values, bins = 100, color = 'k')
    # 如果是displot那么就只有histogram
    plt.show() #这个是必须的



if __name__ == "__main__":
    funclist = {1 : func1, 2 : func2, 3 : func3, 4: func4, 5: func5, 6 : func6, 7 : func7, 8: func8}
    num = int(input("choose which function to run"))
    func = funclist[num]
    func()
