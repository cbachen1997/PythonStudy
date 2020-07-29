# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:54:06 2020

@author: CBA
"""


import numpy as np
import matplotlib.pyplot as plt
import numrec

from sklearn import neighbors
knn=neighbors.KNeighborsClassifier()
#图片数据转换为向量
def img2vector(imagesMat):
    dim=imagesMat.shape[0]#维数
    returnVector=np.zeros((dim,28*28))
    for i in range(dim):
        for j in range(28):
            returnVector[i,(j*28):(j*28+28)]+= imagesMat[i,j,:]
    return returnVector

#计算并获取标签。

    
if __name__=="__main__":
    #读取数据
    train_images=numrec.load_train_images()
    train_labels=numrec.load_train_labels()
    test_images=numrec.load_test_images()
    test_labels=numrec.load_train_labels()
    #读入训练集和测试集数据改变为向量
    v_train_images=img2vector(train_images)
    v_test_images=img2vector(test_images)
    print('矢量化完成')
    #训练分类器
    classifier_knn=knn.fit(v_train_images,train_labels)
    print('分类器训练完成')
    test_predict=knn.predict(v_test_images)
    print(test_predict)