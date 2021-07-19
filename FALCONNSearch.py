from time import time
import falconn
import numpy as np
import pandas as pd
import xlwt


if __name__ == '__main__':
    # 导入数据
    # dataBaseInitial = pd.read_csv('data/TEMP_SMOTENC/data.csv')
    # queryBaseInitial = pd.read_csv('data/TEMP_SMOTENC/query.csv')
    dataBaseInitial = pd.read_csv('data/TEMP_SMOTENC/data_selected.csv')
    queryBaseInitial = pd.read_csv('data/TEMP_SMOTENC/query_selected.csv')

    # 确定数据集大小及维度
    featureNum, dimension = dataBaseInitial.shape
    # print(featureNum)
    # print(dimension)
    dimension = 151
    dataBaseInitial = dataBaseInitial.iloc[:, 0:dimension]
    # print(dataBaseInitial.head())
    # print(dataBaseInitial)

    dataBase = np.array(dataBaseInitial.iloc[:, 1:dimension], dtype="float32")
    queryBase = queryBaseInitial.iloc[:, 1:dimension]
    # print(dataBase.shape)

    # params_cp = falconn.LSHConstructionParameters()
    # params_cp.dimension = len(dataBase[0])
    # params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    # params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    # params_cp.l = 100
    # params_cp.k = 100
    # params_cp.num_setup_threads = 1
    # params_cp.storage_hash_table = falconn.StorageHashTable.LinearProbingHashTable
    # params_cp.num_rotations = 2
    params_cp = falconn.get_default_parameters(len(dataBase), len(dataBase[0]))
    falconn.compute_number_of_hash_functions(18, params_cp)

    table = falconn.LSHIndex(params_cp)
    table.setup(dataBase)

    query_object = table.construct_query_object()
    number_of_probes = params_cp.l
    query_object.set_num_probes(number_of_probes)

    print('FALCONN方案：')
    res = []
    for k in range(10, 1010, 10):
        print('k={}'.format(k))
        begin_time = time()
        for m in range(queryBase.shape[0]):
            query = np.array([queryBase.iloc[m]], dtype="float32").reshape((dimension - 1,))
            print('test{}'.format(m), end=' ')
            index = query_object.find_k_nearest_neighbors(query=query, k=k)
            # print(index)

        end_time = time()
        # print('搜索时间：', end_time - begin_time)

        res.append(end_time-begin_time)

    print(res)
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    i = 2
    j = 0
    for item in res:
        sheet1.write(i, j, item)
        j += 1
    f.save('time.xls')  # 保存文件
