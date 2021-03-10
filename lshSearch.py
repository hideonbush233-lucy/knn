import pandas as pd
import numpy as np
from time import time
from lshash import LSHash
import warnings

warnings.filterwarnings("ignore")


def lshSearch(dataBase2, test2, num):

    lsh = LSHash(30, 216)

    def CreateIndex(array):
        for item in array:
            lsh.index(item)
    CreateIndex(dataBase2)
    test2 = test2.reshape((216,))
    res = lsh.query(test2, num, distance_func='true_euclidean')
    return res


if __name__ == '__main__':
    # 导入数据
    dataBaseInitial = pd.read_csv('data/TEMP_SMOTENC/data.csv')
    testBaseInitial = pd.read_csv('data/TEMP_SMOTENC/test.csv')
    testBase = testBaseInitial.iloc[:, 1:217]
    dataBase = np.array(dataBaseInitial.iloc[:, 1:217])

    lsh = LSHash(30, 216)
    for item in dataBase:
        lsh.index(item)

    begin_time = time()
    print('lsh搜索方案:')

    for m in range(testBase.shape[0]):
        test = np.array([testBase.iloc[m]])
        test = test.reshape((216,))
        result = np.array(lsh.query(test, 10, distance_func='true_euclidean'))
        print('test{} '.format(m))
        print(result[:, 1])
    end_time = time()
    print('搜索时间：', end_time-begin_time)
