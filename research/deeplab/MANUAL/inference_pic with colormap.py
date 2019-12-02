import tensorflow as tf
import os
import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# These are set to the default names from exported models, update as needed.
filename = "frozen_inference_graph1112.pb"

# These names are part of the model and cannot be changed.
output_layer = 'SemanticPredictions:0'
input_node = 'ImageTensor:0'
dirs = os.listdir("./20191107/")


# Load from a file
# export_model.py部分代码
# Input name of the exported model.
_INPUT_NAME = 'ImageTensor'
# Output name of the exported model.
_OUTPUT_NAME = 'SemanticPredictions'
 


t=open(filename, "rb")
sess = tf.Session()
graph_def = tf.GraphDef()
graph_def.ParseFromString(t.read())


image = tf.placeholder(tf.uint8, name='image')

output = tf.import_graph_def(graph_def,input_map={"ImageTensor:0": image},
                                 return_elements=["SemanticPredictions:0"])

 

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

colors = [
    (0, 0, 0),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (128, 0, 0),
    (128, 128, 0),
    (0, 128, 0),
    (0, 128, 128),
    (0, 0, 128),
    (128, 0, 128),
    (255, 42, 0),
    (170, 255, 0),
    (0, 255, 42),
    (0, 170, 255),
    (42, 0, 255),
    (255, 0, 170),
    (128, 42, 0),
    (85, 128, 0),
    (0, 128, 42),
    (0, 85, 128),
    (42, 0, 128),
    (128, 0, 85),
    (255, 170, 0),
    (85, 255, 0),
    (0, 255, 170),
    (0, 85, 255),
    (170, 0, 255),
    (255, 0, 85),
    (128, 85, 0),
    (42, 128, 0),
    (0, 128, 85),
    (0, 42, 128),
    (85, 0, 128),
    (128, 0, 42)
]

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r ,g ,b = colors[i]
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

COLOR_MAP = labelcolormap(len(colors))

#%% predict

for d in dirs:
    files = os.listdir("./20191107/"+d)
    print("Start Dir    "+d+"----------------------------------")
    for f in files:
      if not f.endswith(".jpg"):
          continue
      filepath = "./20191107/"+d+'/'+f[:-4]+'.png'
      if os.path.isfile(filepath) :
          print(filepath+' exists')
          continue
      img = load_img("./20191107/"+d+'/'+f,target_size=(1080,1080))  # 输入预测图片的url
      img = img_to_array(img)  
      img = np.expand_dims(img, axis=0).astype(np.uint8)  # uint8是之前导出模型时定义的

    # input_map 就是指明 输入是什么；
    # return_elements 就是指明输出是什么；两者在前面已介绍
      result = sess.run(output, feed_dict={image:img}) 
      predictions=result[0][0]
      l=count(predictions)
      print(f,'  :  ',l)
      cv2.imwrite(filepath,predictions)

#%% COLORMAP
      
for d in dirs:
    files = os.listdir("./20191107/"+d)
    print("Start Dir    "+d+"----------------------------------")
    for f in files:
      if f.endswith(".png"):     
          print(f)
          filepath = "./20191107/"+d+'/'+f
          image = Image.open(filepath)
          image.putpalette(COLOR_MAP)
          image.save(filepath)
          image.close()



