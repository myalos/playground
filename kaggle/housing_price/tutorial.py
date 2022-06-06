import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from IPython import embed

print(sys.modules[__name__])

X_full = pd.read_csv('data/housing_price_competition_for_kaggle_learn_users/train.csv', index_col='Id')
X_test_full = pd.read_csv('data/housing_price_competition_for_kaggle_learn_users/test.csv', index_col = 'Id')

# 选了一些特征, 标签是SalePrice
#y = X_full.SalePrice
#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# X_full和X_test在列上面差一个SalesPrice
#X = X_full[features].copy()
#X_test = X_test_full[features].copy()
# 训练集shape是1460, 80 测试集的shape是1459, 79
#print(X_full.shape)
#print(X_test_full.shape)
# 设置下面这个会使head 显示出所有列
#pd.set_option('display.max.columns', None)

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=0)

# 可视化数据
#X_full.SalePrice.hist(bins = 40)
#X_full.SalePrice.plot(kind = 'hist', bins = 40)
#refine = X_full.groupby("YearBuilt").SalePrice.agg(['sum', 'mean', 'count', 'median'])
#refine = refine.reset_index()
#print(refine.columns)
#refine.plot(x='YearBuilt', y = 'count')
#refine.plot(x='YearBuilt', y = 'sum')
#plt.show()

# 多个模型
from sklearn.ensemble import RandomForestRegressor

#model_1 = RandomForestRegressor(n_estimators=50, random_state = 0)
#model_2 = RandomForestRegressor(n_estimators=100, random_state = 0)
#model_3 = RandomForestRegressor(n_estimators=100, criterion = 'mae', random_state = 0)
#model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state = 0)
#model_5 = RandomForestRegressor(n_estimators=100, max_depth = 7, random_state = 0)
#
#models = [model_1, model_2, model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error

# 选择一个最佳的模型
#def score_model(model, X_t = X_train, X_v = X_valid, y_t = y_train, y_v = y_valid):
#    model.fit(X_t, y_t)
#    pred = model.predict(X_v)
#    return mean_absolute_error(y_v, pred)
#
#for i in range(len(models)):
#    mae = score_model(models[i])
#    print(f"Model {i + 1} MAE : {mae}")
#

#best_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
#best_model.fit(X, y)
#preds_test = best_model.predict(X_test)
#output = pd.DataFrame({'Id' : X_test.index, 'SalePrice' : preds_test})
#output.to_csv('submission.csv', index = False)

# !!!!!!
# 这个的Score是20998
# 这个是baseline 基本上没有进行数据的处理

# ========================================================
# ===== 下面是处理缺失值的 ======
# 处理缺失值的三个方法
# 1 是把含有缺失值的列给删掉，适用于缺失率很高的列
# 2 imputation 字典里面查不到满意的解释，这个就是拿一些值来填充缺失值
# 3 imputation之后 加上一列 来指明哪些值是填充的

# 查看缺失值
#res = X_full.isnull().sum()
#print(res[res > 0])

# 新的数据集

#data = pd.read_csv('data/melbourne_housing_snapshot/melb_data.csv')
# 这个data的形状是13580，21 加上Price
# 把Price 分离出来
#y = data.Price
#X = data.drop(['Price'], axis = 1)
#X = X.select_dtypes(exclude = ['object'])
# select完了之后 形状为13580 12
#print(f'原来总的数据的形状是: {X.shape}')

# 查看哪些列有缺失值
#cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
# 其中Car，BuildingArea，YearBuilt 这三个特征里面是有缺失值的

#print(cols_with_missing)
# 教程里面是先进行划分数据集 然后再进行缺失值的填补
# 用训练集填补的方法来填补验证集 应该是这么搞的

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=0)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# 第一种方法 丢弃缺失值

#cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
#reduced_X_train = X_train.drop(cols_with_missing, axis = 1)
#reduced_X_valid = X_valid.drop(cols_with_missing, axis = 1)
#print("MAE from approach 1 : ", score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
# 这个结果是183550.22137

# 第二种方法 使用SimpleImputer来填充
#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer()
# 这下面重新用DataFrame来建立一个新的DataFrame
#imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
#imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
#imputed_X_train.columns = X_train.columns
#imputed_X_valid.columns = X_valid.columns

#print("MAE from Approach 2 (Imputation) : ", score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
# 这个结果是178166.46269

#X_train_plus = X_train.copy()
#X_valid_plus = X_valid.copy()

#for col in cols_with_missing:
#    X_train_plus[col + "_was_missing"] = X_train_plus[col].isnull()
#    X_valid_plus[col + "_was_missing"] = X_valid_plus[col].isnull()

#imputer2 = SimpleImputer()
#imputed_X_train_plus = pd.DataFrame(imputer2.fit_transform(X_train_plus))
#imputed_X_valid_plus = pd.DataFrame(imputer2.transform(X_valid_plus))
#
#imputed_X_train_plus.columns = X_train_plus.columns
#imputed_X_valid_plus.columns = X_valid_plus.columns
#
#print("MAE from Approach 3 (An Extension to Imputation)", score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
# 这个结果是178927.503

#print(X_train.shape)
#missing_val_count_by_column = (X_train.isnull().sum())
#print(missing_val_count_by_column[missing_val_count_by_column > 0])

# 通过观察发现 Car的缺失值少，BuildingArea的缺失值大，那么我可以删掉BuildingArea和 YearBuilt 然后Car补起来

#X_train_custom = X_train.copy()
#X_valid_custom = X_valid.copy()


#imputer3 = SimpleImputer()
#X_train_custom = X_train_custom.drop("Car", axis = 1)
#X_valid_custom = X_valid_custom.drop("Car", axis = 1)

#imputed_X_train_custom = pd.DataFrame(imputer3.fit_transform(X_train_custom))
#imputed_X_valid_custom = pd.DataFrame(imputer3.transform(X_valid_custom))
#imputed_X_train_custom.columns = X_train_custom.columns
#imputed_X_valid_custom.columns = X_valid_custom.columns
#print('MAE from Approach 4: (Custom)', score_dataset(imputed_X_train_custom, imputed_X_valid_custom, y_train, y_valid))
# 这个结果是182727.66579178063
# 说明BuildingArea缺失部分多，丢弃之后呢MAE上升很多
# 说明这个包含的信息量还是很大的
# 如果去掉Car的话，那么MAE是178506 说明Car这个特征是Number of carspots 感觉不是很重要

# 上面的数据都是针对n_estimator是10的情况
# 如果变成100的话
# 性能都变好，rank是不变的

# ===================================================
# 下面是用上面的方法来对最初的那个数据集进行操作

# 首先remove掉rows with missing target
#X_full.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
#y = X_full.SalePrice
#X_full.drop(['SalePrice'], axis = 1, inplace = True)
# 选择数值类型
#X = X_full.select_dtypes(exclude = ['object'])
# 这个shape是1460,36 也就是说有很多非数值类型的
#print(X.shape)
#X_test = X_test_full.select_dtypes(exclude=['object'])
#
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=0)
#from sklearn.impute import SimpleImputer
#final_imputer = SimpleImputer(strategy='median')
#imputed_final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
#imputed_final_X_valid = pd.DataFrame(final_imputer.fit_transform(X_valid))
#model = RandomForestRegressor(n_estimators=100, random_state=0)
#model.fit(imputed_final_X_train, y_train)
#imputed_final_X_test = final_imputer.transform(X_test)
#pred_test = model.predict(imputed_final_X_test)
#
#output = pd.DataFrame({'Id' : X_test.index, 'SalePrice' : pred_test})
#output.to_csv('submission.csv', index = False)

# 这个submission的的结果是16635.09027


# ======================================
# 上面的都是处理的数值型数据，下面是针对非数值型的
# categorical variables
# 处理这些的方法有
# 1 直接丢弃
# 2 Ordinal Encoding 将每个类别赋一个数值
# 这个方法对于没有clear ordering的类别效果不怎么好
# 3 One-Hot 比如一个特征是Color 里面的值是Red Red Yellow Green Yellow
# 把一个特征 展开成 三个特征
# Red 1 1 0 0 0
# Yellow 0 0 1 0 1
# Green 0 0 0 1 0
# 对于one hot 如果类别很多的话 效果不好

#data = pd.read_csv('data/melbourne_housing_snapshot/melb_data.csv')
#y = data.Price
#X = data.drop('Price', axis = 1)
#print(X.shape)
# 形状是13580 20
#X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 这里有缺失值的情况，这里直接丢掉
#cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
#X_train_full.drop(cols_with_missing, axis = 1, inplace = True)
#X_valid_full.drop(cols_with_missing, axis = 1, inplace = True)
#low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]

#numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
#my_cols = low_cardinality_cols + numerical_cols
#X_train = X_train_full[my_cols].copy()
#X_valid = X_valid_full[my_cols].copy()

#print(X_train.shape)
# 这个的形状是10864, 12

# 输出categorical variables
#s = (X_train.dtypes == 'object')
#object_cols = list(s[s].index)
#print("Categorical variables", object_cols)
# 输出是['Type', 'Method', 'Regionname']
#print(X_train['Type'].unique())
# 输出是['u', 'h', 't']
# u表示unit, duplex h表示house t表示townhouse
# Method 貌似是出售的方法

# 方法1 直接丢弃类别的特征
#drop_X_train = X_train.select_dtypes(exclude = ['object'])
#drop_X_valid = X_valid.select_dtypes(exclude = ['object'])
#print("MAE from Approach 1 (Drop categorical variables):", score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# 方法2 Ordinal Encoding
# 使用copy方法就是防止修改原表
#from sklearn.preprocessing import OrdinalEncoder
#oe_X_train = X_train.copy()
#oe_X_valid = X_valid.copy()
#oe = OrdinalEncoder()
# 下面输出的是 'Type', 'Method', 'Regionname'
#print(object_cols)
#oe_X_train[object_cols] = oe.fit_transform(X_train[object_cols])
#oe_X_valid[object_cols] = oe.transform(X_valid[object_cols])
#print(f"MAE from Approach 2 (Ordinal Encoding)", score_dataset(oe_X_train, oe_X_valid, y_train, y_valid))
# 这个输出是165936.405483

# 方法3 One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
#oh = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
# handle_unknown = 'ignore' 是avoid errors when the validation data contains classes that aren't represented in the training data
# 然后就是sparse = False 使得encoded columns 返回是以ndarray的形式
#oh_X_train = pd.DataFrame(oh.fit_transform(X_train[object_cols]))
# 将数值型列名改成字符串类型
#oh_X_train.columns = oh_X_train.columns.astype(str)
#oh_X_valid = pd.DataFrame(oh.transform(X_valid[object_cols]))
#oh_X_valid.columns = oh_X_valid.columns.astype(str)
# 上面就是用numpy来构造一个DataFrame 那么index和columns要重写
#oh_X_train.index = X_train.index
#oh_X_valid.index = X_valid.index

#num_X_train = X_train.drop(object_cols, axis = 1)
#num_X_valid = X_valid.drop(object_cols, axis = 1)
#embed()
#oh_X_train = pd.concat([num_X_train, oh_X_train], axis = 1)
#oh_X_valid = pd.concat([num_X_valid, oh_X_valid], axis = 1)

#print(f"MAE from Approach 3 (One-Hot Encoding)", score_dataset(oh_X_train, oh_X_valid, y_train, y_valid))
# 上面这句会warning，原因是one-hot出来的特征名是数字，原来的是字符串
# 输出结果是166089.4893

# ==================================================
# 下面是用housing data 缺失值丢掉，类别值保留处理

#y = X_full.SalePrice
#X = X_full.drop(['SalePrice'], axis = 1)
#cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
#X.drop(cols_with_missing, axis = 1, inplace = True)
# X_test里面有缺失值的
#X_test = X_test_full.drop(cols_with_missing, axis = 1)
#print(f'X_test里面的缺失值有 {X_test.isnull().sum().sum()}')
# 22个缺失值

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=0)

# 用one-hot的方法来处理
#object_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

#low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

#print(low_cardinality_cols)

#high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

#print(high_cardinality_cols)


#oh = OneHotEncoder(sparse = False, handle_unknown='ignore')

#one_hot_X_train = pd.DataFrame(oh.fit_transform(X_train[low_cardinality_cols]))
#one_hot_X_valid = pd.DataFrame(oh.transform(X_valid[low_cardinality_cols]))

#X_train.drop(object_cols, axis = 1, inplace = True)
#X_valid.drop(object_cols, axis = 1, inplace = True)

#one_hot_X_train.index = X_train.index
#one_hot_X_valid.index = X_valid.index
#one_hot_X_train.columns = one_hot_X_train.columns.astype(str)
#one_hot_X_valid.columns = one_hot_X_valid.columns.astype(str)
#
#final_X_train = pd.concat([X_train, one_hot_X_train], axis = 1)
#final_X_valid = pd.concat([X_valid, one_hot_X_valid], axis = 1)
#
#print("MAE from One-Hot ", score_dataset(final_X_train, final_X_valid, y_train, y_valid))

#==================================================
# PipeLine 的用法
# PipeLine 灰常牛逼，值得使用
# 这里使用的是melbourne-housing-snapshot 数据集
#data = pd.read_csv('data/melbourne_housing_snapshot/melb_data.csv')
#print(data.Price.isnull().sum())
#print(data.shape)
#y = data.Price
#X = data.drop('Price', axis = 1)
#print(X.shape)
# 先进行数据集的划分
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=0)

# 对数据进行处理，只保留数值部分和类别少的部分
#categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object' and X_train[col].nunique() < 10]
#numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
#
#my_cols = categorical_cols + numerical_cols
#X_train = X_train[my_cols].copy()
#X_valid = X_valid[my_cols].copy()
#
#print(X_train.shape)

# 第一步是Define Preprocessing Steps
# 之前用One-Hot的时候 需要进行concat操作，用ColumnTransformer就很方便了
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#numerical_transformer = SimpleImputer(strategy = 'constant')
# 上面这个就是默认用0来进行填充
# 类里面也有可能有missing 这个时候用most_frequency来进行处理
#categorical_transformer = Pipeline(steps = [
#    ('imputer', SimpleImputer(strategy = 'most_frequent')),
#    ('onehot', OneHotEncoder(handle_unknown='ignore'))
#])
# 注意一下上面没有用sparse = False咧
#preprocessor = ColumnTransformer(
#        transformers = [
#                ('num', numerical_transformer, numerical_cols),
#                ('cat', categorical_transformer, categorical_cols)
#            ]
#        )

# 第二步是Define the Model
#model = RandomForestRegressor(n_estimators=100, random_state=0)
# 第三步是Create and Evaluate the PipeLine
#my_pipeline = Pipeline(steps = [('preprocessor', preprocessor), ('model', model)])
# PipeLine里面装ColumnTransformer 里面再装 PipeLine
#my_pipeline.fit(X_train,y_train)
#preds = my_pipeline.predict(X_valid)
#score = mean_absolute_error(y_valid, preds)
#print(f'MAE : {score}')

#==================================================
# 下面用Pipeline的技术来处理最初的数据集

# 自己用上面的方法实现一遍
#print(X_full.shape)
#X_full = X_full.dropna(axis = 0, subset = ['SalePrice'])
#print(X_full.shape)
#y = X_full.SalePrice
#X = X_full.drop('SalePrice', axis = 1)
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 0)
#
#categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object' and X_train[col].nunique() < 10]
#numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
#my_cols = categorical_cols + numerical_cols
#X_train = X_train[my_cols].copy()
#X_valid = X_valid[my_cols].copy()
#X_test = X_test_full[my_cols].copy()
#
#numerical_transformer = SimpleImputer(strategy='median')
#categorical_transformer = Pipeline(steps = [
#        ('imputer', SimpleImputer(strategy='constant')),
#        ('onehot', OneHotEncoder(handle_unknown='ignore'))
#    ])
#preprocessor = ColumnTransformer(transformers = [
#        ('num', numerical_transformer, numerical_cols),
#        ('cat', categorical_transformer, categorical_cols)
#    ]
#)
#
#model = RandomForestRegressor(n_estimators=100, random_state=0)
#
#my_pipeline = Pipeline(steps = [
#        ('preprocessor', preprocessor),
#        ('model', model)
#    ])
#
#my_pipeline.fit(X_train, y_train)
#preds = my_pipeline.predict(X_valid)
#print('MAE : ', mean_absolute_error(preds, y_valid))
# 第一个imputer的策略是constant的话 那么这个结果是 17861.780
# 第一个imputer的策略是strategy的话 那么结果是17553
# 后来发现imputer里面可以填的内容是constant median most_frequent 和 mean
# 最佳结果是上面是median 下面是constant, constant填categorical column的话 缺失值是用missing_value的字符串代替

#preds_test = my_pipeline.predict(X_test)
#output = pd.DataFrame({'Id' : X_test.index, 'SalePrice' : preds_test})
#output.to_csv('submission.csv', index = False)
# 这个提交之后 涨点不是很高啊 16475.69973

#==================================================
# 下面是XGBoost的方法
# 之前都用的是随机森林的方法，随机森林是一种ensemble method，ensemble method的作用是将多个模型的预测进行combine，另一种ensemble的方法就是gradient boosting
# 这个方法是goes through cycles to iteratively add models into an ensemble
# 每个cycle里面的步骤有
# 我的理解ensemble里面是一堆model
# 首先we use the current ensemble to generate predictions for each observation in the dataset.To make a prediction, we add the predictions from all models in the ensemble 注意这里是把预测加起来
# There predictions are used to calculate a loss function
# Then we use the loss function to fit a new model that will be added to the ensemble. 我们决定模型的参数使得将这个模型加到ensemble里面会使得loss下降
# 然后将model加到ensemble里面

# XGBoost 是extreme gradient boosting 是一种gradient boosting的实现，并且加上了几个额外的特征用来集中到性能和速度上

#data = pd.read_csv('data/melbourne_housing_snapshot/melb_data.csv')
#y = data.Price
#X = data.drop('Price', axis = 1)
#
#cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
#X = data[cols_to_use]
#
#print(len([col for col in X.columns if X[col].isnull().any()]))
#cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
# 有两个列是有缺失值的
#print(X[cols_with_missing].isnull().sum())
#sys.exit()

#print(X.shape)
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0)
# 默认test_size 是 0.25
#print(X_train.shape)
from xgboost import XGBRegressor
#my_model = XGBRegressor(random_state = 0)
# 发现默认值是100
#my_model.fit(X_train, y_train)
#predictions = my_model.predict(X_valid)
#print("MAE from XGBRegressor baseline : ", mean_absolute_error(predictions, y_valid))
# 输出是239431.9693
# 接下来是调整参数

#my_model1 = XGBRegressor(n_estimators = 500, random_state = 0)
#my_model1.fit(X_train, y_train)
#predictions1 = my_model1.predict(X_valid)
#print("MAE for model1, n_estimators 500 ", mean_absolute_error(predictions1, y_valid))
# 249306.8156 效果变差了


#my_model2 = XGBRegressor(n_estimators = 200, random_state = 0)
#my_model2.fit(X_train, y_train)
#predictions2 = my_model2.predict(X_valid)
#print("MAE for model2, n_estimators 200 ", mean_absolute_error(predictions2, y_valid))
# 241437 还是没有默认的好

# 有个参数是early_stopping_rounds
# 这个参数的作用是 提供一个自动找到n_estimators的理想值。Early stopping causes the model to stop iterating when the validation score stops improving，一般可以把n_estimator设高 然后用early_stopping_rounds来找到optimal time to stop iterating
# 使用early_stopping_rounds = 5是一个合理的选择

#my_model3 = XGBRegressor(n_estimators = 500, random_state = 0)
#my_model3.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(X_valid, y_valid)], verbose = False)
#predictions3 = my_model3.predict(X_valid)
#print("MAE for model3, n_estimators 500, early_stopping_rounds 5", mean_absolute_error(predictions3, y_valid))
# 245079 还是不给力啊

# 接着是介绍了learning_rate
# 这个learning rate的概念有点不同，这个里面的意思是Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number(learning rate) before adding them in
# 小的learning rate和大的n_estimators会产生更好的结果，不过训练时间要长一些

#my_model4 = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, random_state = 0)
#my_model4.fit(X_train, y_train, early_stopping_rounds = 5, eval_set =[(X_valid, y_valid)], verbose = False)
#predictions4 = my_model4.predict(X_valid)
#print("MAE for model4, n_estimators 1000, early_stopping_rounds 5, learning_rate 0.05", mean_absolute_error(predictions4, y_valid))
# 这个输出是246364 感觉还是不给力啊


# 最后来个baseline 残念的是 数据有缺失值，fit报错，XGBRegressor牛逼啊
#my_model5 = RandomForestRegressor(n_estimators=100, random_state = 0)
#my_model5.fit(X_train, y_train)
#predictions5 = my_model5.predict(X_valid)
#print("MAE from random forest", mean_absolute_error(predictions5, y_valid))


#==================================================
# kaggle的方法
# 首先测试一下align 的语法

#df1 = pd.DataFrame([[1, 2, 3, 4], [6, 7, 8 ,9]], columns = ['D', 'B', 'E', 'A'], index = [1, 2])
#df2 = pd.DataFrame([[10, 20, 30, 40], [60, 70, 80 ,90], [600, 700, 800, 900]], columns = ['A', 'B', 'C', 'D'], index = [2, 3, 4])
#print(df1)
#print(df2)

#a1, a2 = df1.align(df2, join = 'outer', axis = 1)
# df1 变成a1 df2变成 a2
#print(a1)
#print(a2)
#b1, b2 = df2.align(df1, join = 'outer', axis = 1)
# df1变成b2 df2变成 b1
#print(b1)
#print(b2)
# 两个方法的结果是一样的，join是outer columns就是两者的并
# 行不变 df1原来的index是1和2 后来也是1和2，这里axis = 1说明进行的是列的并

#print('====================')

#a1, a2 = df1.align(df2, join = 'inner', axis = 1)
# df1 变成a1 df2变成 a2
#print(a1)
#print(a2)
#b1, b2 = df2.align(df1, join = 'inner', axis = 1)
# df1变成b2 df2变成 b1
#print(b1)
#print(b2)
# 上面两个是有区别的 前面的列是D B A 后面的列是 A B D

#print('====================')

#a1, a2 = df1.align(df2, join = 'left', axis = 1)
# df1 变成a1 df2变成 a2
#print(a1)
#print(a2)
#b1, b2 = df2.align(df1, join = 'left', axis = 1)
# df1变成b2 df2变成 b1
#print(b1)
#print(b2)
# 结果的列是按照align前面来的

X = X_full.dropna(axis = 0, subset = ['SalePrice'])
y = X.SalePrice
X.drop(['SalePrice'], axis = 1, inplace = True)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=0)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == 'object' and X_train_full[cname].nunique() < 10]

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join = 'left', axis = 1)
X_train, X_test = X_train.align(X_test, join = 'left', axis = 1)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
# 多个模型
#my_model_1 = XGBRegressor(random_state = 0)
#my_model_1.fit(X_train, y_train)
#predictions1 = my_model_1.predict(X_valid)
#mae_1 = mean_absolute_error(predictions1, y_valid)
#print(mae_1) # 结果是17662.736729
#
#my_model_2 = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=0)
#my_model_2.fit(X_train, y_train)
#predictions2 = my_model_2.predict(X_valid)
#mae_2 = mean_absolute_error(predictions2, y_valid)
#print(mae_2) # 阔以滴牙 这个效果要好 16728.2752
# 如果n_estimators的值改成1000 那么mae是16688.6915
final_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, random_state = 0)
final_model.fit(X_train, y_train)
final_prediction = final_model.predict(X_test)
final = pd.DataFrame({'Id' : X_test.index, 'SalePrice' : final_prediction})
final.to_csv('submission.csv', index = False)
# 这个结果是14794.29660 有点感觉好的奇怪，比validation上面好太多了


#==================================================
# Data Leakage
# Data leakage happens when training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set, but the model will perform poorly in production.
# 有两种leakage 一种target leakage 另一种 train-test contamination

# Target leakage
# 主要是看训练集的特征是否在测试的时候都能获得
# 预测是否有某种疾病，有一个特征是是否接受治愈这个疾病的治疗
# 现实生活中预测的话，那个时候不确定是否得了这种疾病，那个时候是否接受治疗基本上是False
# to prevent this type of data leakage, any variable updated after the target value is realized should be excluded


# train-test contamination
# 感觉这个就是训练集用到了valid 中的数据
# 可能是train和valid一起填缺失值了

# exercise 中 有进行shoelaces的预测，其中一些特征是various macroeconomic features









