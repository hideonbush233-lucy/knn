import pandas as pd
import numpy as np
from time import time

from lshash.storage import RedisStorage
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from ordinarySearch import ordinarySearch
from nearpy import Engine
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomBinaryProjections

import warnings

warnings.filterwarnings("ignore")

# 导入数据
dataBaseInitial = pd.read_csv('data/TEMP_SMOTENC/data_selected.csv')
queryBaseInitial = pd.read_csv('data/TEMP_SMOTENC/query_selected.csv')

# # 确定数据集大小及维度
featureNum, dimension = dataBaseInitial.shape

queryBase = np.array(queryBaseInitial.iloc[:, 1:dimension])
dataBase = np.array(dataBaseInitial.iloc[:, 1:dimension])
# print(featureNum)
# print(dimension)

# 线性，kd_tree,ball_tree搜索回归
alg = 'brute'
begin_time = time()
print('{}搜索方案:'.format(alg))
print('        预测值      误差')
err = []
for m in range(queryBase.shape[0]):
    query = np.array([queryBase[m, :]])
    data = ordinarySearch(dataBaseInitial, query, alg, 100)  # 得到训练集
    # print(data.shape)

    # 高斯回归
    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.01)
    reg.fit(data[:, 1:], data[:, 0])
    # print(query.shape)
    output = reg.predict(query)
    print('test{} '.format(m), end=' ')
    print('{0:.3f}℃ {1:.3f}℃'.format(output[0], abs(queryBaseInitial.iloc[m, 0] - output[0])))
    err.append(abs(queryBaseInitial.iloc[m, 0] - output[0]))

print('MAE:', sum(err) / len(err))
print('MSE', sum([num ** 2 for num in err]) / len(err))
end_time = time()
print('运行时间：', end_time - begin_time)

print('*' * 50)

# LSH搜索回归
print('LSH搜索方案：')
rbp = RandomBinaryProjections('rbp', 30)
engine = Engine(dimension - 1, lshashes=[rbp], vector_filters=[NearestFilter(100)])

# for index in range(featureNum):
#     v = dataBase[index]
#     engine.store_vector(v, '{}'.format(index))

engine.store_many_vectors(dataBase, [i for i in range(featureNum)])

begin_time = time()
print('        预测值      误差')
err = []
for m in range(len(queryBase)):
    query = queryBase[m]
    N = engine.neighbours(query, distance='euclidean')
    index = [int(x[1]) for x in N]
    # print(index)
    data = np.array([dataBaseInitial.iloc[index, :]])
    data = data[0]
    # print(data.shape)
    query = np.array([queryBase[m]])
    # print(query.shape)
    # 高斯回归
    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.01)
    reg.fit(data[:, 1:], data[:, 0])
    output = reg.predict(query)

    print('test{} '.format(m), end=' ')

    print('{0:.3f}℃ {1:.3f}℃'.format(output[0], abs(queryBaseInitial.iloc[m, 0] - output[0])))
    err.append(abs(queryBaseInitial.iloc[m, 0] - output[0]))

print('MAE:', sum(err) / len(err))
print('MSE', sum([num ** 2 for num in err]) / len(err))
end_time = time()
print('运行时间：{}'.format(end_time - begin_time))
