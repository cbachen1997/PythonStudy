# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:38:14 2020

@author: CBA
"""

import os
import shutil
import zipfile
from os.path import join,getsize
import re



def unzip_file(filename, dst_dir):
    r = zipfile.is_zipfile(filename)
    # zip_file=zipfile.ZipFile(eachline)
    if r:
        fz = zipfile.ZipFile(filename, 'r')
        for file in fz.namelist():
            if re.match(r'.*?vv.*?.tiff',file):
                fz.extract(file, dst_dir) 
                
    else:
        print('This is not zip')


if __name__=='__main__':
     dstdir2015='H:\\C&M\\2015Sentinel'
     dstdir2020='H:\\C&M\\2020Sentinel'
     for i in os.listdir(dstdir2020):
         zippath=dstdir2020+'\\'+i
         unzip_file(zippath,dstdir2020)