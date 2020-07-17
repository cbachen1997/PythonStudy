# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:33:32 2020

@author: CBA
"""
import requests
#简单下载requests,成功
def simple_download(url,path):
    res=requests.get(url)
    #下载文件命
    with open(path,'wb') as f:
        f.write(res.content)

import wget
import ssl
#取消ssl全局验证
ssl._create_default_https_context=ssl._create_unverified_context
#wget下载，成功
def wget_(url,path):
    wget.download(url,path)
from clint.textui import progress
# from progress import *

# 下载大文件，解决内存问题
def big_download(url,path):
    #设置stream=Ture,遍历iter_content再下载
    res=requests.get(url,stream=True)
    #获取文件大小
    total_length=int(res.headers.get('content-length'))
    #输出url状态和标头
    print(res.status_code,res.headers)
    #分块下载
    with open(path,'wb') as name:
        for chunk in progress.bar(res.iter_content(chunk_size=1024*1000),expected_size=(total_length/1024)+1,width=100):
            if chunk:
                name.write(chunk)
        print('file:{path} download completed!'.format(path=path))
if __name__=='__main__':
    # wget_('https://pic4.zhimg.com/80/v2-b3972560b6f5b7ecfac44b3ceb78d134_720w.jpg?source=1940ef5c','H:\\TEST.jpg')
    # simple_download('https://pic4.zhimg.com/80/v2-b3972560b6f5b7ecfac44b3ceb78d134_720w.jpg?source=1940ef5c','H:\\TEST1.jpg')
    big_download('https://pic4.zhimg.com/80/v2-b3972560b6f5b7ecfac44b3ceb78d134_720w.jpg?source=1940ef5c',r'H:\test_big.jpg')