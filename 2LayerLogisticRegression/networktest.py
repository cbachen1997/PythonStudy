# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:54:03 2020

@author: CBA
"""


__author__='cba'
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt
import os 
#该任务中的有用底层代码函数
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets
#测试示例
from testCases import *

np.random.seed(1)

def pic_dataset(X,Y):
    x=X[0,:]
    y=X[1,:]
    c=np.squeeze(Y)
    fig=plt.figure()
    plt.title("Ilustate of dataset")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(x=x,y=y,c=c,s=40,cmap=plt.cm.Spectral,)

def layer_sizes(X,Y):
    """
    参数：
     X - 输入数据集,维度为（输入的数量，训练/测试的数量）
     Y - 标签，维度为（输出的数量，训练/测试数量）
    
    返回：
     n_x - 输入层的数量
     n_h - 隐藏层的数量
     n_y - 输出层的数量
    """
    n_x=X.shape[0]
    n_h=4#直接定义4层
    n_y=Y.shape[0]#输出1个结果
    
    return n_x,n_h,n_y

def initialize_parameters(n_x,n_h,n_y):
    """
    参数：
        n_x - 输入层节点的数量
        n_h - 隐藏层节点的数量
        n_y - 输出层节点的数量
    
    返回：
        parameters - 包含参数的字典：
            W1 - 权重矩阵,维度为（n_h，n_x）
            b1 - 偏向量，维度为（n_h，1）
            W2 - 权重矩阵，维度为（n_y，n_h）
            b2 - 偏向量，维度为（n_y，1）

    """
    np.random.seed(2)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
    parameters = {"W1" : W1,
	              "b1" : b1,
	              "W2" : W2,
	              "b2" : b2 }
    return parameters

def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    #前向传播
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    #判断A2的维度
    assert(A2.shape==(1,X.shape[1]))
    #缓存数据
    cache = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}
    
    return (A2, cache)
#计算成本函数
def compute_cost(A2,Y,parameters):
    m=Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    #计算成本
    logprobs=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y))
    cost = - np.sum(logprobs)/m
    #删除维度为1的数据
    
    cost = float(np.squeeze(cost))
    assert(isinstance(cost,float))
    return cost
def backward_propagation(parameters,cache,X,Y):
    A2=cache['A2']
    A1=cache['A1']
    m = X.shape[1] 
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    dZ2=A2-Y
    dW2=np.dot(dZ2,A1.T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1=np.dot(dZ1,X.T)/m
    db1=np.sum(dZ1,axis=1,keepdims=True)/m
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2 }
    return grads
#梯度下降
def update_parameters(parameters,grads,learning_rate):
    W1,W2 = parameters["W1"],parameters["W2"]
    b1,b2 = parameters["b1"],parameters["b2"]
    
    dW1,dW2 = grads["dW1"],grads["dW2"]
    db1,db2 = grads["db1"],grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    np.random.seed(3)
    #读入神经网络结构
    n_x=layer_sizes(X, Y)[0]
    n_y=layer_sizes(X,Y)[2]
    #初始化W，b参数
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    #前向传播
    for i in range(num_iterations):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y,parameters)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=0.025)
        if (print_cost and i %1000==0):
             print("第 ",i," 次循环，成本为："+str(cost))
    return parameters

#利用模型预测
def predict(parameters,X):
    A2,cache=forward_propagation(X,parameters)
    #四舍五入保留2位小数
    prediction=np.round(A2)
    return prediction
if __name__=="__main__":
    #读取数据集
    X,Y=load_planar_dataset()
    #绘制散点图看看
    # pic_dataset(X,Y)
    # #获得数据维度
    # shape_X=X.shape
    # shape_Y=Y.shape
    # m=Y.shape[1]#训练集的数量
#     #========================================================
#     #测试logistic分类的结果。
#     clf=sklearn.linear_model.LogisticRegressionCV()
#     clf.fit(X.T,Y.T)
#     #利用planar里面的函数绘制分类边界
#     x=X[0,:]
#     plot_decision_boundary(lambda x:clf.predict(x),X,Y)
#     plt.title("Logistic Regression")
#     LR_predictions=clf.predict(X.T)
#     #都是1和都是0的算预测正确
#     print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) + 
# 		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
#        "% " + "(正确标记的数据点所占的百分比)")
#     #=======================================
    #搭建神经网络
    """
        Z[1](i)=W[1]x(i)+b[1](i)
        a[1](i)=tanh(Z[1](i))#用tanh作为隐含层激活函数
        Z[2](i)=W[2]xa[1](i)+b[2](i)
        yhat(i)=a[2](i)=sigmoid(Z[2](i))
        
        神经网络构建方法
        1.定义网络结构（输入层，隐藏层节点等）
        2.初始化模型参数（W随机初始化）
        3.循环（前向传播，计算loss，后向传播，梯度下降）
    """
    #定义网络结构
    # (n_x,n_h,n_y) =  layer_sizes(X,Y)
    #初始化模型参数
    # parameters=initialize_parameters(n_x,n_h,n_y)
    # #测试
    parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
    #绘制决策边界
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    
    predictions = predict(parameters, X)
    print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
    
    #改变节点数
    # plt.figure(figsize=(16, 32))
    # hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] #隐藏层数量
    # for i, n_h in enumerate(hidden_layer_sizes):
    #     plt.subplot(5, 2, i + 1)
    #     plt.title('Hidden Layer of size %d' % n_h)
    #     parameters = nn_model(X, Y, n_h, num_iterations=10000)
    #     plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    #     predictions = predict(parameters, X)
    #     accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    #     print ("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
    #==========================================================================
    # #更换数据集
    # # 数据集
    # noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    
    # datasets = {"noisy_circles": noisy_circles,
    #             "noisy_moons": noisy_moons,
    #             "blobs": blobs,
    #             "gaussian_quantiles": gaussian_quantiles}
    
    # dataset = "noisy_moons"
    
    # X, Y = datasets[dataset]
    # X, Y = X.T, Y.reshape(1, Y.shape[0])
    
    # if dataset == "blobs":
    #     Y = Y % 2
    # # #测试
    # parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
    # # #绘制决策边界
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # plt.title("Decision Boundary for hidden layer size " + str(4))
    
    # predictions = predict(parameters, X)
    # print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
    # plt.scatter(X[0, :], X[1, :], c=Y[0,:], s=40, cmap=plt.cm.Spectral)
    #==========================================================================