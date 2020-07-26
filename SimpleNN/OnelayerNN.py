# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 17:10:35 2020

@author: CBA
"""


#简单神经网络拟合
import numpy as np
np.random.seed(1337)#选定随机数种子
from keras.models import Sequential#序列模型
from keras.layers import Dense#全链接层
import matplotlib.pyplot as plt#图像显示

from tqdm import tqdm#进度条库
# import time
##创建数据集
X = np.linspace(-1,1,200)#(-1,1)内创建200个均匀分布
np.random.shuffle(X)#乱序
Y = 0.5*X+2+np.random.normal(0,0.05,(200,))#假设我们真实模型：Y=0.5X+2
#np.random.normal(μ,σ，size)，分布的均值，方差，
#绘制数据集
#plt.scatter(X,Y)
plt.show()

#创建训练集和测试集
X_train,Y_train=X[:160],Y[:160]
X_test,Y_test=X[160:],Y[160:]

#定义模型
model=Sequential()#Keras包括Sequential模型和Function模型
         #Sequential常用，单输入单输出
#利用add层层增加模型
model.add(Dense(1,input_dim=1))#定义第一层时必须指定数据输入形状input_dim,数据才能喂进来
#Dense为全连接层，第一层需要定义
#第二层把第一层的输出作为输入

#指定训练参数，选择损失函数和optimizer
model.compile(loss='mse',optimizer='sgd')#均方差作为损失函数，随机梯度下降作为优化方法


#开始训练
print('Training......')
for step in tqdm(range(301)):
  cost = model.train_on_batch(X_train,Y_train)#train_on_batch是众多开始训练的函数之一
  if (step % 100 == 0):#每100次输出一次训练方差
    print('train cost:',cost)

#测试模型
print('\ntesting......')
cost = model.evaluate(X_test,Y_test,batch_size=40)
W,b=model.layers[0].get_weights()
#查看训练的网络参数
#因为网络只有1层，每次训练输入1个，输出也只有一个
#设定模型为Y=WX+b，所以W，b为训练参数

print('Weights=',W,'\nbiases',b)

Y_predict=model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_predict)
plt.show()


