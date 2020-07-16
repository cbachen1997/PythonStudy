# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 22:02:45 2020

@author: CBA
"""
import os
import time
from tqdm import tqdm
from tqdm.std import trange

def get_big_file(path,filesize):
    for dirpath,dirnames,filenames in os.walk(path):
        for filename in filenames:
            target_file=os.path.join(dirpath,filename)
        #判断是否是文件
            if not os.path.isfile(target_file):
                continue
            size=os.path.getsize((target_file))
            if size>filesize:
                size=size//(1024*1024)
                size='{size}M'.format(size=size)
                print(target_file,size)
            
if __name__=='__main__':
    get_big_file('D:\\',500*1024*1024)
    