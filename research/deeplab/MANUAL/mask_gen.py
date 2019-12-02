from PIL import Image,ImageDraw
import os
import json
import numpy as np


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N =255):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

COLOR_MAP = labelcolormap()

file_list = os.listdir('./')
label_list = {}

for f in file_list:
    if not f.endswith(".json"):
        continue

    with open(f) as jfile:
        data = json.load(jfile)
        height = data['imageHeight']
        width = data['imageWidth']
        polygons = data['shapes']
        img = Image.new('L',(width,height))
        #img.putpalette(COLOR_MAP)
        draw = ImageDraw.Draw(img)
        for poly in polygons:
            if poly['shape_type'] !='polygon':
                continue

            fill_color = 0
            if poly['label'] in label_list:
                fill_color = label_list[poly['label']]
            else:
                label_list[poly['label']] = len(label_list)+1
                fill_color = label_list[poly['label']]

            points = [tuple(p) for p in poly['points']]
            draw.polygon(points,fill=fill_color)

        filename = './'+f[:-4]+'png'
        img.save(filename)


