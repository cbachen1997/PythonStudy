# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:33:55 2020

@author: CBA
"""


__author__='cba'

import numpy as np
import struct
import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
    
# 训练集文件
train_images_idx3_ubyte_file = 'train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = 'train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = 't10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = 't10k-labels.idx1-ubyte'
#读取数据集
def decodedataset(file):
    #读取二进制文件
    binarydata=open(file,'rb').read()
    #解析文件头信息，依次为魔数、图片数量、图片的高和宽
    offset=0#>iii用大端法读取
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    #struct.unpack解包，参数（解包格式（str可查表），数据，缓存）
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, binarydata, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    #魔数为文件起始的固定字节，来判断文件类型
    
    #解析数据集
    image_size=num_rows*num_cols
    offset += struct.calcsize(fmt_header)
    #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    #文件包含头文件和数，offset来移动指向信息的指针
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image,offset,struct.calcsize(fmt_image))
    #构建3维图层放数据集
    images = np.empty((num_images, num_rows, num_cols))
    #导入数据
    # plt.figure()
    for i in range(100):
       if (i + 1) % 10000 == 0:
           print('已解析 %d' % (i + 1) + '张')
           print(offset)
       images[i] = np.array(struct.unpack_from(fmt_image, binarydata, offset)).reshape((num_rows, num_cols))
       offset += struct.calcsize(fmt_image)
       # plt.imshow(images[i],'gray')#imshow是处理图像
       # plt.pause(0.01)
       # plt.show()#show才是显示图像
    return images#返回读取的数据集矩阵
#读取便签
def decodelabel(labelfile):
    bin_data = open(labelfile, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
        #     print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels
def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decodedataset(idx_ubyte_file)

def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decodelabel(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decodedataset(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decodelabel(idx_ubyte_file)
if __name__=="__main__":
    print('读入数据')
    #载入数据
    # train_images=decodedataset(train_images_idx3_ubyte_file)
    # train_labels=decodelabel(train_labels_idx1_ubyte_file)
    # test_images=decodedataset(test_images_idx3_ubyte_file)
    # test_labels=decodelabel(test_labels_idx1_ubyte_file)
    #看看对比
    # for i in range(10):
    #     print(train_labels[i])
    #     plt.imshow(train_images[i], cmap='gray')
    #     plt.pause(0.000001)
    #     plt.show()