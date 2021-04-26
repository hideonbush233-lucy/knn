from time import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")


def ordinarySearch(dataBaseInitial2, test2, algorithm, k):
    dataBaseInitial2 = np.array(dataBaseInitial2)
    dataBase2 = dataBaseInitial2[:, 1:217]

    neigh2 = NearestNeighbors(n_neighbors=k, algorithm=algorithm)  # kd_tree brute ball_tree
    neigh2.fit(dataBase2)
    dis2, index2 = neigh2.kneighbors(test2, k)
    data2 = np.array([dataBaseInitial2[index2[0][0]]])
    for i in range(1, dis2.shape[1]):
        data2 = np.vstack((data2, [dataBaseInitial2[index2[0][i]]]))

    return data2


if __name__ == '__main__':
    # 导入数据
    dataBaseInitial = pd.read_csv('data/TEMP_SMOTENC/data_selected.csv')
    queryBaseInitial = pd.read_csv('data/TEMP_SMOTENC/query_selected.csv')

    # # 确定数据集大小及维度
    featureNum, dimension = dataBaseInitial.shape

    queryBase = np.array(queryBaseInitial.iloc[:, 1:dimension])
    dataBase = np.array(dataBaseInitial.iloc[:, 1:dimension])

    # ordinary搜索 线性 kd_tree ball_tree
    alg = 'brute'
    print('{}搜索方案：'.format(alg))
    # for num in range(10, 110, 10):
    #     print(num)

    # 计算运行时间
    begin_time = time()

    neigh = NearestNeighbors(n_neighbors=10, algorithm= alg)  # kd_tree brute ball_tree
    neigh.fit(dataBase)

    for m in range(queryBase.shape[0]):
        test = np.array([queryBase[m, :]])
        dis, index = neigh.kneighbors(test, 10)
        print('test{} '.format(m))
        print(dis)
        print(index)

    end_time = time()
    print('搜索时间：', end_time - begin_time)
