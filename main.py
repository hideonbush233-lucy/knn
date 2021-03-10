import pandas as pd
import numpy as np
from time import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from ordinarySearch import ordinarySearch
from lshSearch import lshSearch

import warnings
warnings.filterwarnings("ignore")

# 导入数据
dataBaseInitial = pd.read_csv('data/TEMP_SMOTENC/data.csv')
testBaseInitial = pd.read_csv('data/TEMP_SMOTENC/test.csv')
testBase = np.array(testBaseInitial.iloc[:, 1:217])
dataBase = np.array(dataBaseInitial.iloc[:, 1:217])

begin_time = time()
print('brute搜索方案:')
for m in range(testBase.shape[0]):
    test = np.array([testBase[m, :]])
    data = ordinarySearch(dataBaseInitial, test, 'brute', 100)  # 得到训练集

    # 高斯回归
    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    reg.fit(data[:, 1:], data[:, 0])
    output = reg.predict(test)
    print('test{} '.format(m))
    print('预测值  误差')
    print('{0:.5f} {1:.5f}'.format(output[0], abs(testBaseInitial.iloc[m, 0]-output[0])))

end_time = time()
print('运行时间：', end_time - begin_time)

begin_time = time()
print('lsh搜索方案:')
for m in range(testBase.shape[0]):
    test = np.array([testBase[m, :]])
    res = np.array(lshSearch(dataBase, test, 100))
    print('test{} '.format(m))
    print(res[:, 1])

end_time = time()
print('运行时间：', end_time - begin_time)
