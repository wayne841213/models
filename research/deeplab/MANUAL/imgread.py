# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:30:55 2019

@author: sean.lan
"""

import cv2




img= cv2.imread("0-cut.png")


img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


for m in img:
    for n in m:
        if n!= 0:
            print(n)