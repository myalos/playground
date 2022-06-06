from sklearn.decomposition import PCA
from category_encoders import MEstimateEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 这个里面 主要的内容是
# determin which features are the most important with mutual information
# invent new features in several real-world problem domains
# encode high-cardinality categoricals with a target encoding
# create segmentation features with K-means clustering
# decompose a dataset's variation into features with PCA

# 之前都是y=X.SalePrice 然后X.drop('SalePrice', axis = 1)
# 有个更简单的写法
# X = df.copy()
# y = X.pop("Compressive Strength")

# ==================================================
# 互信息
# 当遇到一个新的数据集 有很多特征
# 第一步 就是construct a ranking with a feature utility metric, a function measuring associations between a feature and the target
# 互信息相对与协方差的优势在于 其不只能找线性关系，能找任何关系
# 互信息是针对单特征的metric，不能检测多个feature和label的关联，有的特征单独是没啥信息的但是和另一个特征组合起来就有信息了。
# 即使有一个互信息很强的特征，这个特征还是依赖于模型，比如二次曲线的互信息可能有1.74，但是模型是线性的 就不能model
# 一般互信息大于2.0是少见的

#data:pd.DataFrame = pd.read_csv('data/feature_engineer/autos.csv')
#X = data.copy()
#y = X.pop("price")
#print(X.shape, y.shape) # 193, 24
#print(X.columns)

#print(X.select_dtypes('object')) #输出是一个dataframe
# colname 遍历的是X的列名
#for colname in X:
#    print(colname)
#for colname in X.select_dtypes('object'):
#    print(colname)

# factorize相当于之前的OrdinalEncoder
#for colname in X.select_dtypes('object'):
#    print(X[colname].factorize())
#    print(len(X[colname].factorize()))
#    print(type(X[colname].factorize()[0]), type(X[colname].factorize()[1]))
#    a, b = X[colname].factorize()
#    print(X[colname].value_counts().count())
#    print(a)
#    print(len(a))
#    print(X[colname].head(20))
#    break

#for colname in X.select_dtypes('object'):
#    X[colname], _ = X[colname].factorize()

# 下面进行互信息，mutual_info_regression是针对连续的，mutual_info_classif是针对离散的

#
#def make_mi_scores(X, y, discrete_features):
#    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
#    mi_scores = pd.Series(mi_scores, index=X.columns,name = "MI scores")
#    mi_scores = mi_scores.sort_values(ascending=False)
#    return mi_scores
#
#discrete_features = X.dtypes == int
#print(X.dtypes)
#print(discrete_features)
#
#res = make_mi_scores(X, y, discrete_features)
#res.sort_values(ascending = True).plot.barh(title = 'MI Score')
#sns.relplot(x = 'curb_weight', y = 'price', data = data)

#下面是sns的lmplot lmplot里面可以设置ci=95表示95%的置信区间
# hue 表示分类
# 虽然特征hue_type和price的互信息非常小，但是
# it clearly separates two price populations with different trends within the horsepower feature. This indicates that fuel_type contributes an interaction effect and might not be unimportant after all.

# 下面这个函数可以的啊，可以看出hue的feature是否对(x, y)的关系有影响
#sns.lmplot(x = 'horsepower', y = 'price', hue = 'fuel_type', data = data)
#plt.show()

# Exercise 1
data = pd.read_csv('data/feature_engineer/ames.csv')
print(data.shape) # shape是2930, 79
X = data.copy()
y = X.pop('SalePrice')

# 完整版 比之前的更完整一些
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(['object', 'category']):
        X[colname], _ = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    # 这里得到的是一个ndarray
    mi_scores = mutual_info_regression(X, y, discrete_features = discrete_features, random_state = 0)
    mi_scores = pd.Series(mi_scores, name = 'MI Scores', index = X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

#mi_scores = make_mi_scores(X, y)

# 下面是melt函数的用法

df = pd.DataFrame({'A': ['a','b','c'], 'B' : [1,3 ,5], 'C' : [2,4,6], 'D' : [10, 20, 30]})
#print(df)
#    A  B  C   D
# 0  a  1  2  10
# 1  b  3  4  20
# 2  c  5  6  30
# id列 和 其余同行中所有列分别组成行
#print(pd.melt(df, id_vars = ['A']))
# 会变成
#    A variable  value
# 0  a        B      1
# 1  b        B      3
# 2  c        B      5
# 3  a        C      2
# 4  b        C      4
# 5  c        C      6
# 6  a        D     10
# 7  b        D     20
# 8  c        D     30

#print(pd.melt(df, id_vars='A', value_vars=['C' , 'D']))

features = ['YearBuilt', 'MoSold', 'ScreenPorch']
# 设置了col就会有多张图
#sns.relplot(x = "value", y = "SalePrice", col = 'variable', data = data.melt(id_vars="SalePrice", value_vars=features), facet_kws=dict(sharex = False))
# 上面的sharex 默认是True 要改成False

#print(mi_scores.tail(10))
#plt.show()
# mi 最高的特征是OverallQual
# 20th高的是 SecondFlrSF
# 最低的是LandSlope
# 分别画出箱型图

# mi高的箱形图的区分就大一些
#sns.catplot(x = 'OverallQual', y = 'SalePrice', data = data, kind = 'boxen')
#sns.catplot(x = 'SecondFlrSF', y = 'SalePrice', data = data, kind = 'boxen')
#sns.catplot(x = 'LandSlope', y = 'SalePrice', data = data, kind = 'boxen')
#sns.catplot(x = 'BldgType', y = 'SalePrice', data = data, kind = 'boxen')
#plt.show()

# ==================================================
# 下面是第二节 creating features
#from queue import deque
#from copy import deepcopy
#q = deque()
#if q:
#    print('ye')
#else:
#    print('no')
#
#a = [1,2 ,3]
#b = deepcopy(a)
#b[1] = 4
#print(a)

# 重要的一点是要 research the problem domain to acquire domain knowledge
accidents = pd.read_csv("data/feature_engineer/accidents.csv")
autos = pd.read_csv("data/feature_engineer/autos.csv")
concrete = pd.read_csv("data/feature_engineer/concrete.csv")
customer = pd.read_csv("data/feature_engineer/customer.csv")

# np.log1p 等价于log(1 + x)
#accidents['LogWindSpeed'] = accidents.WindSpeed.apply(np.log1p)

# 分布是highly skewed，可以画log图来看
#fig, axes = plt.subplots(1, 2, figsize = (8, 4))
# 这个kdeplot画的是分布图
#sns.kdeplot(accidents.WindSpeed, shade = True, ax = axes[0])
#sns.kdeplot(accidents.LogWindSpeed, shade = True, ax = axes[1])
#plt.show()

# 还有一种feature 就是有多少个component
# 比如混泥土的那个例子
#components = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAggregate', 'FineAggregate']
#concrete["Components"] = concrete[components].gt(0).sum(axis = 1)
#print(concrete[components + ["Components"]].head(10))

# 复杂的feature 拆分成多个feature
# 比如电话号码里面有区号，那么可以多一个位置feature
# 这个写法很帅啊
#customer[["Type", "Level"]] = (
#        customer["Policy"]
#        .str
#        .split(" ", expand = True)
#)

#print(customer[["Policy", "Type", "Level"]].head(10))

# 下面是Group Transforms
# 这个的作用是aggregate information across multiple rows grouped by some category

# 比如一个城市的平均收入，这个式子很帅啊
#customer["AverageIncome"] = (
#        customer.groupby("State")
#        ["Income"]
#        .transform("mean")
#)

#print(customer[["State", "Income", "AverageIncome"]].head(10))

# 还可以某个特征的频率
#customer["StateFreq"] = (
#    customer.groupby("State")
#    ["State"]
#    .transform("count")
#    / customer.State.count()
#)

#print(customer[["State" ,"StateFreq"]].head(10))

# 对于有train_test_split的方法的话
# create a grouped feature using only the training set and then join it to the validation set.

#print(customer.columns)
#df_train = customer.sample(frac = 0.5)
#df_valid = customer.drop(df_train.index)
#print(df_train.shape) # 输出 4567, 25
#df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")
#print(df_train.shape)
#print(df_train.head(10))
#print("********************")
# 为什么要drop_duplicates
# df_train .. 得到的是4567行 直接duplicate 然后join到valid
#print(df_valid.shape) # 也是4567 25
#df_valid = df_valid.merge(
#        df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
#        on = "Coverage",
#        how = "left"
#)
#print(df_valid.shape)
#print(df_valid.columns)
#print(df_valid.head(10))
#print(df_valid[["Coverage", "AverageClaim"]].head(10))


#print(df_train[["Coverage", "AverageClaim"]].head(10))
# 这个输出的是， 对对对 这个是有4567行的，head就是10行
# 5059     Basic    380.363912
# 6679   Premium    627.101804
# 7361     Basic    380.363912
# 4069  Extended    475.135860
# 839   Extended    475.135860
# 4807  Extended    475.135860
# 579      Basic    380.363912
# 7300   Premium    627.101804
# 3239  Extended    475.135860
# 1641     Basic    380.363912

# Count features
# 对于ames 数据集 WoodDeckSF是一个连续型变量
# 与这个同种类的还有OpenPorchSF EnclosedPorch Threeseasonporch
# ScreenPorch 4种，这都是Porch，建一个新的feature来算这个里面有多少种porch feature
#ames = pd.read_csv('data/feature_engineer/ames.csv')
#print(ames.WoodDeckSF.gt(0).astype(np.int64).head(10))
#print(ames.WoodDeckSF.gt(0).astype(np.int64) + ames.OpenPorchSF.gt(0).astype(np.int64))

# 下面这个写法不给过啊
#ames["PorchTypes"] = ames.WoodDeckSF.gt(0).astype(np.int64) + ames.OpenPorchSF.gt(0).astype(np.int64)
#print(ames.head(10))

# 更好的方法
#ames["PorchTypes"] = ames[["WoodDeckSF", "OpenPorchSF",  "EnclosedPorch", "Threeseasonporch", "ScreenPorch"]].gt(0.0).sum(axis = 1)
#print(ames.head(10))


# 下面这个很牛逼 那就是把MSSubClass的第一个下划线前面的内容取出来当一个新的feature
#Temp = pd.DataFrame()
# expand很重要
#Temp["MSClass"] = ames.MSSubClass.str.split('_', n = 1, expand = True)[0]
#print(Temp.head(10))

#==================================================
# 下面是Kmeans，用Kmeans来创建新的特征，比如根据地理信息来进行聚类，因为地理位置相近的地方往往天气pattern是相近的
# 比如YearBuilt这个特征，将这个特征进行聚类，分成多个类，会发现不同类种 YearBuilt和SalePrice的关系更简单
# housing data 里面用经度和纬度还有median income 三个特征来进行聚类，进行经济区域的划分
#from sklearn.cluster import KMeans

#df = pd.read_csv('data/feature_engineer/housing.csv')

#X = df.loc[:, ['MedInc', 'Latitude', 'Longitude']]
#print(X.head())

#kmeans = KMeans(n_clusters=6)
#X["Cluster"] = kmeans.fit_predict(X)
#X["Cluster"] = X["Cluster"].astype("category")
#print(X.head())

# If the cluster is informative, these distributions should, for the most part, separate across median income, which is indeed what we see(我有点看不出来)

#sns.relplot(x = 'Longitude', y = 'Latitude', hue = 'Cluster', data = X, height = 7)
# 这里的MedHouseVal是预测的目标
#X["MedHouseVal"] = df["MedHouseVal"]
#sns.catplot(x = "MedHouseVal", y = 'Cluster', data = X, kind = "boxen", height = 6)
#plt.show()

# k-means算法 对scale很敏感
# 是否进行rescale是要根据特征的特点来进行判断的
# 经纬度就不用rescale
# Number of Doors 和 Horsepower of a car要rescale 因为单位差比较大

# 上面这个是model_based mothod for feature engineering
# 下面的是PCA
#==================================================
# 聚类是对dataset根据proximity来进行划分，而PCA是根据数据的variation来进行划分
# ！！使用PCA之前要对数据进行 Standardized
#  the whole idea of PCA: instead of describing the data with the original features, we describe it with its axes of variation. The axes of variation become the new features.

#  The new features PCA constructs are actually just linear combinations (weighted sums) of the original features:
#df["Size"] = 0.707 * X["Height"] + 0.707 * X["Diameter"]
#df["Shape"] = 0.707 * X["Height"] - 0.707 * X["Diameter"]
#These new features are called the principal components of the data. The weights themselves are called loadings. There will be as many principal components as there are features in the original dataset: if we had used ten features instead of two, we would have ended up with ten components.

# !!!the amount of variance in a component doesn't necessarily correspond to how good it is as a predictor: it depends on what you're trying to predict.

# 有两种方法可以将PCA用于feature engineering
# 一种是将其当作是descriptive technique
# 一种是将components当作features

# 这主要是画Explained variance
# np.c_是按行进行concat 而np.r_是按列进行concat
def plot_variance(pca, width = 8, dpi = 100):
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    # 设置axes的属性 用set方法比较好
    axs[0].set(xlabel = "Component", title = "% Explained Variance", ylim = (0.0, 1.0))
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(xlabel = "Component", title = "% Cumulative Variance", ylim = (0.0, 1.0))
    fig.set(figwidth = 8, dpi = 100)
    return axs


#X = autos.copy()
#y = X.pop('price')
#mi_scores = make_mi_scores(X, y)

# print(mi_scores.head(10))
# 下面选的是互信息最大的4个特征

#features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]
#X = X.loc[:, features]
# Standardize
#X_scaled = (X - X.mean(axis = 0)) / X.std(axis = 0)

#pca = PCA()
# xp 交换两个字符
# pca的fit_transform的结果是一个ndarray
#X_pca = pca.fit_transform(X_scaled)

#print(type(X_pca))
#component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
#X_pca = pd.DataFrame(X_pca, columns=component_names)
#print(X_pca.head())
#         PC1       PC2       PC3       PC4
# 0  0.382486 -0.400222  0.124122  0.169539
# 1  0.382486 -0.400222  0.124122  0.169539
# 2  1.550890 -0.107175  0.598361 -0.256081
# 3 -0.408859 -0.425947  0.243335  0.013920
# 4  1.132749 -0.814565 -0.202885  0.224138

# 下面是显示PCA的loadings，loadings是在components_属性里面
# PCA 对应的是 线性变换
#loadings = pd.DataFrame(
#    pca.components_.T,
#    columns = component_names,
#    index = X.columns
#)
#print(loadings)

# highway_mpg mpg的意思是每加仑燃料所行英里数
#                   PC1       PC2       PC3       PC4
# highway_mpg -0.492347  0.770892  0.070142 -0.397996
# engine_size  0.503859  0.626709  0.019960  0.594107
# horsepower   0.500448  0.013788  0.731093 -0.463534
# curb_weight  0.503262  0.113008 -0.678369 -0.523232
# PC1代表的就是 large,powerful, 耗能
# 称作 Luxury/Economy axis

#print((-0.492347 * X_scaled.highway_mpg + 0.503859 * X_scaled.engine_size + 0.500448 * X_scaled.horsepower + 0.503262 * X_scaled.curb_weight).head())
# 666

#plot_variance(pca)
#plt.show()

#mi_scores_pca = mutual_info_regression(X_pca, y, discrete_features = False)
#mi_scores_pca = pd.Series(mi_scores_pca, name = "MI Scores", index = X_pca.columns)
#mi_scores_pca = mi_scores_pca.sort_values(ascending = False)
#
#print(mi_scores_pca)

# 怎么PC1的互信息没有curb_weight大啊，有点失望
#PC1    1.014640
#PC2    0.379761
#PC3    0.307344
#PC4    0.204268

# PC3 shows a contrast between horsepower and curb_weight -- sports cars vs. wagons, it seems.

# 这种方法显示数据可以的啊
#idx = X_pca["PC3"].sort_values(ascending = False).index
#cols = ['make', 'body_style', 'horsepower', 'curb_weight']
#
#print(autos.loc[idx, cols].head())
#print(autos.loc[idx, cols].tail())


# 下面是对应的练习
# 主要是用PCA创建的新feature来涨点 和 用PCA来检测outlier

def apply_pca(X, standardize = True):
    if standardize:
        X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns = component_names)
    # 这个转置别忘了哟
    loadings = pd.DataFrame(
        pca.components_.T,
        columns = component_names,
        index = X.columns
    )
    return pca, X_pca, loadings

def score_dataset(X, y, model = XGBRegressor()):
    for colname in X.select_dtypes(['object', 'category']):
        X[colname], _ = X[colname].factorize()
    score = cross_val_score(
            model, X, y, cv = 5, scoring = 'neg_mean_squared_log_error'
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

#df2 = pd.read_csv('data/feature_engineer/ames.csv')
#features2 = ["GarageArea", "YearRemodAdd", "TotalBsmtSF", "GrLivArea"]
#
#X = df2.copy()
#y = X.pop('SalePrice')
#X = X.loc[:, features2]
#
#pca, X_pca, loadings = apply_pca(X)
#print(loadings)
#                    PC1       PC2       PC3       PC4
# GarageArea    0.541229  0.102375 -0.038470  0.833733
# YearRemodAdd  0.427077 -0.886612 -0.049062 -0.170639
# TotalBsmtSF   0.510076  0.360778 -0.666836 -0.406192
# GrLivArea     0.514294  0.270700  0.742592 -0.332837

# 啥也不干 预测的loss是
#X_origin = df2.copy()
#y = X_origin.pop("SalePrice")
#print(score_dataset(X_origin, y))
# 0.142842

#X_1 = df2.copy()
#y = X_1.pop("SalePrice")
#X_1 = X_1.join(X_pca)
#print(score_dataset(X_1, y))
# 这是把pca得到的feature 加到上面取 0.1370651388
# 是有效果的

# 通过观察loading得到了
#X_2 = df2.copy()
#y = X_2.pop("SalePrice")
#X_2["Feature1"] = X_2.GrLivArea + X_2.TotalBsmtSF
#X_2["Feature2"] = X_2.YearRemodAdd * X_2.TotalBsmtSF
#print(score_dataset(X_2, y))
# 结果是0.133607568 这个特征是怎么看出来的啊 神奇

# 再看看melt函数
# melt出来的结果
#print(X_pca.melt())

#       variable     value
# 0          PC1 -0.165346
# 1          PC1 -0.639050
# 2          PC1 -0.794227
# 3          PC1  1.636658
# 4          PC1  0.293648
# ...        ...       ...
# 11715      PC4  0.820547
# 11716      PC4  0.620322
# 11717      PC4 -1.417521
# 11718      PC4 -0.374198
# 11719      PC4  0.328966

#sns.catplot(y = 'value', col = 'variable', data = X_pca.melt(), kind = 'boxen', col_wrap=2 ,sharey = False)
#plt.show()
# 箱形图 离中心很远的就是outlier
# 可以通过一个column 来找出outlier
#component = "PC1"

#idx = X_pca[component].sort_values(ascending=False).index
#print(df2.loc[idx, ["SalePrice", "Neighborhood", "SaleCondition"] + features2].head(10))
# 注意看SaleCondition

# ==================================================
# Target Encoding
# !! a target encoding is any kind of encoding that replaces a feature's categories with some number derived from the target.

# 一种简单的方法是 对于不同品牌的汽车，我用这个品牌汽车的SalePrice的均值来对品牌进行encoding，如果target是binary的，那么encode的内容就是target不同类型的数量

# Smoothing
# An encoding like this presents a couple of problems, however. First are unknown categories. Target encodings create a special risk of overfitting, which means they need to be trained on an independent "encoding" split. When you join the encoding to future splits, Pandas will fill in missing values for any categories not present in the encoding split. These missing values you would have to impute somehow.
# Second are rare categories. When a category only occurs a few times in the dataset, any statistics calculated on its group are unlikely to be very accurate. In the Automobiles dataset, the mercurcy make only occurs once. The "mean" price we calculated is just the price of that one vehicle, which might not be very representative of any Mercuries we might see in the future. Target encoding rare categories can make overfitting more likely.
# A solution to these problems is to add smoothing. The idea is to blend the in-category average with the overall average. Rare categories get less weight on their category average, while missing categories just get the overall average.
# In pseudocode:
# encoding = weight * in_category + (1 - weight) * overall
# where weight is a value between 0 and 1 calculated from the category frequency.
# An easy way to determine the value for weight is to compute an m-estimate:
# weight = n / (n + m)
# where n is the total number of times that category occurs in the data. The parameter m determines the "smoothing factor". Larger values of m put more weight on the overall estimate.
# 如果对于每个品牌的average price非常稳定的话，那么就用小的数值

# Target encoding的适用场景，一个是high-cardinality features，另一个是domain_motivated features

# 下面是一个MovieLens1M的例子
# df3 = pd.read_csv('data/feature_engineer/movielens1m.csv')
# 这个shape是 1000209，28
#print(df3.shape)
# print(df3.head())
# print(df3.dtypes)
#print(df3.memory_usage())

#pd.set_option('display.max.columns', None)
#print(df3.head(10))
#print(df3.describe()) # 这个表里面大部分数字都很小 所以用np.uint8来将 来节省内存
# df3 = df3.astype(np.uint8, errors = 'ignore') # 减到原来的1/8
# 这里ignore的意义在于 有的数据里面是b开头的 比如b'3107'
# 如果int(b'3107') 结果是3107
# astype会影响到每一列，对于object的列 ignore就会忽略掉
# print(df3.dtypes)
# print(f'Number of unique Zipcodes: {df3["Zipcode"].nunique()}')
# 有3439个unique

# 单独拿出25%的split来训练
# X = df3.copy()
# y = X.pop('Rating')
# X_encode = X.sample(frac = 0.25)
# y_encode = y[X_encode.index]
# X_pretrain = X.drop(X_encode.index)
# y_train = y[X_pretrain.index]

# 这个库要pip install category-encoders来进行安装
# 这个smoothing parameter是一个超参数
# encoder = MEstimateEncoder(cols = ["Zipcode"], m = 5.0) encoder.fit(X_encode, y_encode)
# X_train = encoder.transform(X_pretrain)
#
# print(X_train.head(10))

# 下面画图，我不知道这个图是想说明啥
# 上面说的是Let's compare the encoded values to the target to see how informative our encoding might be
# plt.figure(dpi = 90)
# ax = sns.distplot(y, kde = False, norm_hist = True)
# ax = sns.kdeplot(X_train.Zipcode, color = 'r', ax = ax)
# ax.set_xlabel("Rating")
# ax.legend(labels=['Zipcode', 'Rating'])
# plt.show()

# Exercise里面 用的ames数据集

# 首先看一下将encode数据集单独拿出来的重要性
df4 = pd.read_csv('data/feature_engineer/ames.csv')

X = df4.copy()
y = X.pop('SalePrice')
score = score_dataset(X, y)
print(f'original score : {score:.4f} RMSLE')
# 这个结果是0.1428

X["Meaningless"] = range(len(X))
X["Meaningless"][1] = 0

encoder = MEstimateEncoder(cols = 'Meaningless', m = 0)
X = encoder.fit_transform(X, y)
score = score_dataset(X, y)
print(f"Score: {score:.4f} RMSLE")
# 这个结果是0.0293

# 这里单独拿出来
X_1 = df4.copy()
X_1["Meaningless"] = range(len(X_1))
X_encode = X_1.sample(frac = 0.2)
y_encode = X_encode.pop("SalePrice")
X_pretrain = X_1.drop(X_encode.index)
y_train = X_pretrain.pop("SalePrice")

X_encode['Meaningless'].iloc[1] = X_encode['Meaningless'].iloc[0]
encoder1 = MEstimateEncoder(cols = 'Meaningless', m = 0)
encoder1.fit(X_encode, y_encode)
X_train = encoder1.transform(X_pretrain)

score = score_dataset(X_train, y_train)
print(f'separate encoding set, score : {score:.4f} RMSLE')
# 这个结果是0.1416
# 这个提升有点不能接受
# 多次运行发现这个结果每次运行不一样 上面的是一样的


