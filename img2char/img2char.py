# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:27:42 2020

@author: CBA
"""

__author__='cba'

#pillow图像处理
from PIL import Image
#argparse管理命令行输入
import argparse
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

#构建命令行输入参数处理实例
parser=argparse.ArgumentParser()
#定义输入，输出文件，输出字符画的宽高
parser.add_argument('file')#输入文件
parser.add_argument('-o','--output')#输出文件
parser.add_argument('--width',type=int,default=100)#字符画宽
parser.add_argument('--height',type=int,default=100)#字符画高
#解析获取参数
args=parser.parse_args()

#输出参数和输入参数
IMG=args.file
WIDTH=args.width
HEIGHT=args.height
OUTPUT=args.output

#rgb转字符参数
def get_char(r,g,b,alpha=256):
    #判断alpha
    if alpha==0:
        return " "
    length=len(ascii_char)
    #灰度加权
    gray=int(0.2126*r+0.7152*g+0.0722*b)
    #归一化
    unit=length/(256.0+0.0001)
    
    return ascii_char[int(gray*unit)]

if __name__=='__main__':
    #打开图片，调整宽高
    im=Image.open(IMG)
    im=im.resize((WIDTH,HEIGHT),Image.BILINEAR)
    
    #输出字符串初始化
    txt=''
    
    #遍历图片
    for i in range(HEIGHT):
        for j in range(WIDTH):
            txt+= get_char(*im.getpixel((j,i)))
        txt+='\n'
    
    #输出文件
    if OUTPUT:
        with open(OUTPUT,'w') as f:
            f.write(txt)
    else:
        with open('output_test.txt','w') as f:
            f.write(txt)
            