
import os
import numpy as np
import cv2


dirs = os.listdir("./20191107/")

from xml.etree.ElementTree import Element, SubElement, ElementTree

#%% predict

for d in dirs[:1]:
    files = os.listdir("./20191107/"+d)
    print("Start Dir : "+d+"--------------------------------------")
    for f in files:
        if not f.endswith(".png"):
            continue
        print(f)
        data_path = "./20191107/"+d+'/'+f[:-4]
        filepath = data_path +'.png'
        image = cv2.imread(filepath)
        
        image_b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_b = cv2.threshold(image_b, 1, 255, 0)[1]
        
#        blu = 5
#        blurred = cv2.GaussianBlur(image_b, (blu, blu), 0)
#        cv2.imshow('Blur',blurred)
        
        # 使用Canny方法尋找邊緣
#        a,b,c = np.percentile(blurred, [30,50,75])
#        edged = cv2.Canny(blurred, a, b)
          
        overlapping=image
#        edged_color = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
#        overlapping = cv2.addWeighted(image, 0.5, edged_color, 0.5, 0)
        
        contours, hierarchy = cv2.findContours(image_b,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        shift=5
        
        box=[]
        root = Element('annotation')
        
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            box.append((x-shift, y-shift, x+w+shift, y+h+shift))

            bo = box[-1]
            plot=1
            for bo2 in box:
                if bo[0]>bo2[0] and bo[2]<bo2[2] and bo[1]>bo2[1] and bo[3]<bo2[3]:
                    plot=0
                    break
                if plot==1:
                    cv2.rectangle(overlapping, bo[:2], bo[2:], (0,255,0))
                    
                    x_min , y_min , x_max ,y_max = bo
                    ob = SubElement(root, 'object')
                    name = SubElement(ob, 'name')
                    name.text = data_path.split('/')[-1]
                    pose = SubElement(ob, 'pose')
                    pose.text = 'Unspecified'
                    truncated = SubElement(ob, 'truncated')
                    truncated.text = str(0)
                    difficult = SubElement(ob, 'difficult')
                    difficult.text = str(0)
                    bndbox = SubElement(ob, 'bndbox')
                
                    #設定座標，只有這個比較重要
                    xmin = SubElement(bndbox, 'xmin')
                    xmin.text = str(x_min)
                    ymin = SubElement(bndbox, 'ymin')
                    ymin.text = str(y_min)
                    xmax = SubElement(bndbox, 'xmax')
                    xmax.text = str(x_max)
                    ymax = SubElement(bndbox, 'ymax')
                    ymax.text = str(y_max)
        
#        cv2.imshow('Edged' ,edged)
#        cv2.imshow('Overlapping' ,overlapping)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        tree = ElementTree(root)
        tree.write(data_path + '.xml')


