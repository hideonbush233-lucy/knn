from time import time
import numpy as np
import pandas as pd
from nearpy import Engine
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomBinaryProjections


def lshSearch(dataBase_, query_, k):
    featureNum_ = len(dataBase_)
    dimension_ = len(dataBase_[0])

    rbp_ = RandomBinaryProjections('rbp', 30)

    engine_ = Engine(dimension_, lshashes=[rbp_], vector_filters=[NearestFilter(k)])

    for i in range(featureNum_):
        v_ = dataBase_[i]
        engine_.store_vector(v_, '{}'.format(i))

    N_ = engine_.neighbours(query_, distance='euclidean')
    index_ = [int(x[1]) for x in N_]
    return index_


if __name__ == '__main__':
    # 导入数据
    dataBaseInitial = pd.read_csv('data/TEMP_SMOTENC/data_selected.csv')
    queryBaseInitial = pd.read_csv('data/TEMP_SMOTENC/query_selected.csv')

    # # 确定数据集大小及维度
    featureNum, dimension = dataBaseInitial.shape

    queryBase = np.array(queryBaseInitial.iloc[:, 1:dimension])
    dataBase = np.array(dataBaseInitial.iloc[:, 1:dimension])

    rbp = RandomBinaryProjections('rbp', 30)

    engine = Engine(dimension-1, lshashes=[rbp], vector_filters=[NearestFilter(10)])

    for index in range(featureNum):
        v = dataBase[index]
        engine.store_vector(v, '{}'.format(index))

    begin_time = time()
    for m in range(len(queryBase)):
        query = queryBase[m]
        N = engine.neighbours(query, distance='euclidean')
        print('test{} '.format(m))
        print([int(x[1]) for x in N])
    end_time = time()
    print('搜索时间：{}'.format(end_time - begin_time))
