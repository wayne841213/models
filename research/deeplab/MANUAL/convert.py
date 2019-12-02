import os

import cv2


files = os.listdir("./")
for f in files:
    if f.endswith(".mask"):
        os.rename(os.path.join("./",f),os.path.join("./",f[:-4]+"bmp"))

images = os.listdir("./")
for f in images:
    print(f)
    if f.endswith(".png"):
        img = cv2.imread("./"+f)
        img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for m in img:
            for n in m:
                if n!= 0:
                    print(n)
#%%
        _,img = cv2.threshold(img,100,1,0)
        cv2.imwrite("./"+f,img)