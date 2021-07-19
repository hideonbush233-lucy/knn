from math import sqrt
import pandas as pd
import numpy as np
from time import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from nearpy.distances import EuclideanDistance
from nearpy.storage import MemoryStorage
from ordinarySearch import ordinarySearch
from nearpy import Engine
from nearpy.filters import NearestFilter, UniqueFilter
from nearpy.hashes import RandomBinaryProjections
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore")

# 导入数据
# dataBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/data.csv')
# queryBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/query.csv')
dataBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/data_selected.csv')
queryBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/query_selected.csv')

# # 确定数据集大小及维度
featureNum, dimension = dataBaseInitial.shape

queryBase = np.array(queryBaseInitial.iloc[:, 1:dimension])
dataBase = np.array(dataBaseInitial.iloc[:, 1:dimension])
# print(featureNum)
# print(dimension)

# 线性搜索回归
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
print('RMSE', sqrt(sum([num ** 2 for num in err]) / len(err)))
end_time = time()
print('运行时间：', end_time - begin_time)

print('*' * 50)

# LSH搜索回归
print('LSH搜索方案：')
rbp = RandomBinaryProjections('rbp', 20)
engine1 = Engine(dimension - 1, lshashes=[rbp, rbp, rbp], storage=MemoryStorage(),
                 distance=EuclideanDistance(), vector_filters=[NearestFilter(100)])

engine1.store_many_vectors(dataBase, [i for i in range(featureNum)])

begin_time = time()
print('        预测值      误差')
err = []
for m in range(len(queryBase)):
    query = queryBase[m]
    N = engine1.neighbours(query, distance='euclidean', fetch_vector_filters=[UniqueFilter()])
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
print('RMSE', sqrt(sum([num ** 2 for num in err]) / len(err)))
end_time = time()
print('运行时间：{}'.format(end_time - begin_time))

print('*' * 50)

# LSH+线性搜索
print('LSH+线性搜索方案：')
rbp = RandomBinaryProjections('rbp', 20)
engine2 = Engine(dimension - 1, lshashes=[rbp, rbp, rbp], storage=MemoryStorage(),
                 distance=EuclideanDistance(), vector_filters=[NearestFilter(1000)])

# for index in range(featureNum):
#     v = dataBase[index]
#     engine.store_vector(v, '{}'.format(index))

engine2.store_many_vectors(dataBase, [i for i in range(featureNum)])

begin_time = time()
print('        预测值      误差')
err = []
for m in range(len(queryBase)):
    # 初步近似搜索
    query = queryBase[m]
    N = engine2.neighbours(query, distance='euclidean', fetch_vector_filters=[UniqueFilter()])
    index1 = [int(x[1]) for x in N]
    # print(index)

    # 截取数据，data_middle1包含输出变量，data_middle2用于相似性搜索
    data_middle1 = dataBaseInitial.iloc[index1, :].reset_index(drop=True)
    data_middle2 = np.array(data_middle1.iloc[:, 1:dimension])

    # 再次线性搜索
    neigh = NearestNeighbors(n_neighbors=100, algorithm='brute')  # kd_tree brute ball_tree
    neigh.fit(data_middle2)
    query = np.array([queryBase[m, :]])
    # print(data_middle2.shape)
    if data_middle2.shape[0] < 100:
        data2 = np.array(data_middle1)
    else:
        _, index2 = neigh.kneighbors(query, 100)
        data2 = np.array([np.array(data_middle1)[index2[0][0]]])
        for i in range(1, index2.shape[1]):
            data2 = np.vstack((data2, [np.array(data_middle1)[index2[0][i]]]))
    # print(data2)
    # print(data2.shape)

    # query = np.array([queryBase[m]])
    # print(query.shape)

    # 高斯回归
    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.01)
    reg.fit(data2[:, 1:], data2[:, 0])
    output = reg.predict(query)

    print('test{} '.format(m), end=' ')

    print('{0:.3f}℃ {1:.3f}℃'.format(output[0], abs(queryBaseInitial.iloc[m, 0] - output[0])))
    err.append(abs(queryBaseInitial.iloc[m, 0] - output[0]))

print('MAE:', sum(err) / len(err))
print('RMSE', sqrt(sum([num ** 2 for num in err]) / len(err)))
end_time = time()
print('运行时间：{}'.format(end_time - begin_time))
