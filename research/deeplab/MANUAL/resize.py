# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:30:55 2019

@author: sean.lan
"""

import cv2
import os
import numpy as np

images = os.listdir("./")



def count(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result

def merge(x,y):
    for k,v in x.items():
                if k in y.keys():
                    y[k] += v
                else:
                    y[k] = v
    return y

def balance(x):
    bg=x[0]
    for k,v in x.items():
        x[k] = round(bg/v)
    return x


R={}



for f in images:
    if f.endswith(".jpg"):
        img= cv2.imread(f)
        #img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_AREA)

        cv2.imwrite(f,img)
        # Label count
        l=count(img)
        print(f,'  :  ',l)
        R= merge(l,R)
        
print('Result : ',R)
print('Balance : ',balance(R))
