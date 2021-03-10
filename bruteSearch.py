import pandas as pd
from time import *
import numpy as np

# 导入数据
dataBase = pd.read_csv('data/TEMP_SMOTENC/data.csv')
testBase = pd.read_csv('data/TEMP_SMOTENC/test.csv')
dataBase = dataBase.iloc[:, 1:217]
testBase = testBase.iloc[:, 1:217]

dataBase = np.array(dataBase)

# 计算运行时间
begin_time = time()

# 线性查找
for m in range(testBase.shape[0]):
    result = {}
    test = np.array(testBase.iloc[m])
    for i in range(dataBase.shape[0]):
        data = np.array(dataBase[i])
        dif = np.sqrt(np.sum(np.square(test - data)))
        # 记录初始误差
        if i < 10:
            result[i] = dif
        # 比较计算结果与现有误差
        elif i == 10:
            order_result = np.array(sorted(result.items()))
        elif i > 10:
            if dif < order_result[9][1]:
                order_result = np.delete(order_result, 9, 0)
                order_result = np.insert(order_result, 9, [i, dif], 0)
                order_result = order_result[np.lexsort(order_result.T)]
    print('test{} 序号     距离'.format(m))
    np.set_printoptions(suppress=True)
    for j in range(order_result.shape[0]):
        print('    {0}  {1}'.format(order_result[j][0], order_result[j][1]))

end_time = time()
print('运行时间：', end_time - begin_time)
