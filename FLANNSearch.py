import pyflann
import numpy as np
import pandas as pd
from time import time

# 导入数据
dataBaseInitial = pd.read_csv('data/TEMP_SMOTENC/data.csv')
testBaseInitial = pd.read_csv('data/TEMP_SMOTENC/test.csv')
testBase = np.array(testBaseInitial.iloc[:, 1:217])
dataBase = np.array(dataBaseInitial.iloc[:, 1:217])

pyflann.set_distance_type(distance_type='euclidean')
flann = pyflann.FLANN()

params = flann.build_index(dataBase, algorithm='composite', trees=40)
# algorithm="autotuned", target_precision=0.9, log_level="info"
# algorithm="kdtree", trees=40
# algorithm='kmeans'
# algorithm='composite', trees=40

begin_time = time()
result, dists = flann.nn_index(testBase, 10)
print(result)
print(np.sqrt(dists))
end_time = time()
print('搜索时间：', end_time-begin_time)
