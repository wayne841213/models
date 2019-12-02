import tensorflow as tf
import os

graph_def = tf.GraphDef()
labels = []

# These are set to the default names from exported models, update as needed.
filename = "frozen_inference_graph.pb"
labels_filename = "label.txt"

# Import the TF graph
with tf.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())
        
        
#開啟檔案，並在 BGR 色彩空間中建立影像

from PIL import Image
import numpy as np
import cv2


def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (1285, 1285), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image







#處理維度 >1600 的影像
"""
# If the image has either w or h greater than 1600 we resize it down respecting
# aspect ratio such that the largest dimension is 1600
image = resize_down_to_1600_max_dim(image)


#裁剪中央最大的正方形


# We next get the largest center square
h, w = image.shape[:2]
min_dim = min(w,h)
max_square_image = crop_center(image, min_dim, min_dim)



#將大小往下調整為 256x256

# Resize that square down to 256x256
augmented_image = resize_to_256_square(max_square_image)




#裁剪模型特定輸入大小的中心

# Get the input size of the model
with tf.Session() as sess:
    input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
network_input_size = input_tensor_shape[1]

# Crop the center for the specified network_input_Size
augmented_image = crop_center(augmented_image, network_input_size, network_input_size)
"""




#預測影像

#將影像準備為張量之後，可以透過預測模型傳送它：


# These names are part of the model and cannot be changed.
output_layer = 'SemanticPredictions:0'
input_node = 'ImageTensor:0'
dirs = os.listdir("./20191107/")


#%% Load from a file

with tf.Session() as sess:
    for d in dirs:
        files = os.listdir("./20191107/"+d)
        for f in files:
          print(f)
          image = Image.open("./20191107/"+d+'/'+f)
          
          # Update orientation based on EXIF tags, if the file has orientation info.
          image = update_orientation(image)
          # Convert to OpenCV format
          image = convert_to_opencv(image)
          image = resize_to_256_square(image)
          try:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions, = sess.run(prob_tensor, {input_node: [image] })
          except KeyError:
            print ("Couldn't find classification output layer: " + output_layer + ".")
            print ("Verify this a model exported from an Object Detection project.")
            exit(-1)
      
          # 顯示圖片
          #cv2.imshow('original', image)
          
          #透過模型執行影像張量的結果接著需要對應回標籤。
          _,predictions=cv2.threshold(predictions.astype('uint8'),0,255,0)
          #cv2.imshow("predictions",predictions)
          cv2.imwrite("./20191107/"+d+'/'+f[:-4]+'_pred.jpg',predictions)
          #cv2.waitKey(0)
          #cv2.destroyAllWindows()

    
