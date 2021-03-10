from time import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")


def ordinarySearch(dataBaseInitial2, test2, algorithm, num):
    dataBaseInitial2 = np.array(dataBaseInitial2)
    dataBase2 = dataBaseInitial2[:, 1:217]

    neigh2 = NearestNeighbors(n_neighbors=num, algorithm=algorithm)  # kd_tree brute ball_tree
    neigh2.fit(dataBase2)
    dis2, index2 = neigh2.kneighbors(test2, num)
    data2 = np.array([dataBaseInitial2[index2[0][0]]])
    for i in range(1, dis2.shape[1]):
        data2 = np.vstack((data2, [dataBaseInitial2[index2[0][i]]]))

    return data2


if __name__ == '__main__':
    # 导入数据
    dataBaseInitial = np.array(pd.read_csv('data/TEMP_SMOTENC/data.csv'))
    testBaseInitial = np.array(pd.read_csv('data/TEMP_SMOTENC/test.csv'))
    dataBase = dataBaseInitial[:, 1:217]
    testBase = testBaseInitial[:, 1:217]

    # 计算运行时间
    begin_time = time()

    # ordinary搜索 线性 kd_tree ball_tree
    print('ball_tree搜索方案：')
    neigh = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')  # kd_tree brute ball_tree
    neigh.fit(dataBase)
    for m in range(testBase.shape[0]):
        test = np.array([testBase[m, :]])
        dis, index = neigh.kneighbors(test, 10)
        print('test{} '.format(m))
        print(dis)

    end_time = time()
    print('搜索时间：', end_time - begin_time)
