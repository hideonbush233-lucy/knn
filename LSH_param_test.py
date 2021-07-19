from time import time

import falconn
import numpy as np
import pandas as pd
# import xlwt
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
# 导入数据
dataBaseInitial = pd.read_csv('data/Defect/data.csv')
validBaseInitial = pd.read_csv('data/Defect/valid.csv')
# dataBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/data.csv')
# queryBaseInitial = pd.read_csv('data/TEMP/TEMP_SMOTENC_20w/query.csv')

# 确定数据集大小及维度
featureNum, dimension = dataBaseInitial.shape
# print(featureNum)
# print(dimension)
# print(dataBaseInitial.head())
# print(dataBaseInitial)

dataBase = np.array(dataBaseInitial.iloc[:, 2:dimension], dtype="float32")
validBase = validBaseInitial.iloc[:, 2:dimension]
print(validBase.shape)
# print(dataBase.shape)
# acc = [[0] * 50 for _ in range(5)]
# test_time = [[0] * 50 for _ in range(5)]
for l in range(2, 10):
    acc = []
    valid_time = []
    for k in range(1, 51):
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dimension - 2  # 数据集的维度
        params_cp.lsh_family = falconn.LSHFamily.Hyperplane  # LSHFamily::Hyperplane或者 LSHFamily::Crosspolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = l  # 哈希表的数量
        params_cp.k = k  # 每个哈希表的哈希函数数
        params_cp.num_setup_threads = 1
        params_cp.storage_hash_table = falconn.StorageHashTable.LinearProbingHashTable
        params_cp.num_rotations = 1

        table = falconn.LSHIndex(params_cp)
        table.setup(dataBase)

        query_object = table.construct_query_object()
        number_of_probes = params_cp.l
        query_object.set_num_probes(number_of_probes)

        correct = 0
        temp = []
        for m in range(validBase.shape[0]):

            neigh = NearestNeighbors(n_neighbors=100, algorithm='brute')  # kd_tree brute ball_tree
            neigh.fit(dataBase)
            query_ = np.array([validBase.iloc[m]])
            dis, index_ = neigh.kneighbors(query_, 100)
            # print(index_[0])

            begin_time = time()
            query = np.array([validBase.iloc[m]], dtype="float32").reshape((dimension - 2,))
            # print('test{}'.format(m), end=' ')
            index = query_object.find_k_nearest_neighbors(query=query, k=100)
            # print(index)
            end_time = time()
            for item in index_[0]:
                if item in index:
                    correct += 1
            temp.append(end_time - begin_time)
        timesum = sum(temp)
        print(correct/validBase.shape[0]/100)
        acc.append(correct / validBase.shape[0]/100)
        valid_time.append(timesum/10000)
    ax = plt.subplot(1, 1, 1)
    h1 = ax.plot([_ for _ in range(1, 51)], acc, linewidth=2, color='b', label='accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('Number of hash functions')
    ax.set_title('Number of hash tables is {}'.format(l))
    ax.legend(loc=2)
    ax2 = ax.twinx()
    h2 = ax2.bar([_ for _ in range(1, 51)], valid_time, color='r', width=0.5, label='Time(s)')
    ax2.set_ylabel('Time(s)')
    ax2.legend(loc=1)

    plt.savefig('./pics/label{}.pdf'.format(l))
    plt.savefig('./pics/label{}.png'.format(l), dpi=200, bbox_inches='tight')
    print('save success!')
    plt.draw()
    plt.pause(6)
    plt.close()


# f = xlwt.Workbook()
# sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
# for i in range(5):
#     for j in range(0, 50):
#         sheet1.write(i + 10, j, test_time[i][j])
#         sheet1.write(i, j, acc[i][j])
# f.save('params_acc_time.xls')  # 保存文件
