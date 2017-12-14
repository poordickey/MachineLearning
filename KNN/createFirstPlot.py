"""
Created on Oct 27, 2010

@author: Peter
"""

import matplotlib.pyplot as plt
from numpy import *

from KNN import kNN

fig = plt.figure()
ax = fig.add_subplot(111)  # 添加子图
datingDataMat, datingLabels = kNN.file2matrix('./dataset/datingTestSet2.txt')  # 获取特征值矩阵和label值
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
ax.axis([-2, 25, -0.2, 2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
# plt.show()

group, labels = kNN.createDataSet()
print(kNN.classify0([1, 1], group, labels, 3))
