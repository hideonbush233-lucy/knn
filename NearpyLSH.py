from time import time
import numpy as np
import pandas as pd
from nearpy import Engine
from nearpy.distances import EuclideanDistance
from nearpy.filters import NearestFilter, UniqueFilter
from nearpy.hashes import RandomBinaryProjections
import xlwt

from nearpy.storage import MemoryStorage


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
    dataBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/data.csv')
    queryBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/query.csv')
    # dataBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/data_selected.csv')
    # queryBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/query_selected.csv')

    # # 确定数据集大小及维度
    featureNum, dimension = dataBaseInitial.shape

    queryBase = np.array(queryBaseInitial.iloc[:, 1:dimension])
    dataBase = np.array(dataBaseInitial.iloc[:, 1:dimension])

    rbp = RandomBinaryProjections('rbp', 20)
    res = []
    for k in range(10, 1010, 10):
        print('k={}'.format(k))
        engine = Engine(dimension - 1, lshashes=[rbp, rbp, rbp], storage=MemoryStorage(),
                        distance=EuclideanDistance(), vector_filters=[NearestFilter(k)])
        engine.store_many_vectors(dataBase, [i for i in range(featureNum)])

        begin_time = time()
        for m in range(len(queryBase)):
            query = queryBase[m]
            N = engine.neighbours(query, distance='euclidean', fetch_vector_filters=[UniqueFilter()])
            # print('test{} '.format(m))
            # print([int(x[1]) for x in N])
        end_time = time()
        res.append(end_time - begin_time)
        print('搜索时间：{}'.format(end_time - begin_time))

    print(res)
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    j = 0
    for item in res:
        sheet1.write(0, j, item)
        j += 1
    f.save('time.xls')  # 保存文件
