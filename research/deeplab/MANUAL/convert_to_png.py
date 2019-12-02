import os
from PIL import Image

for root, dirs, files in os.walk("."):
    for bmpfig in files:
        if not bmpfig.endswith('.bmp') and not bmpfig.endswith('.png'):
            continue
        bmpfig = os.path.join(root, bmpfig)
        newfigname = bmpfig[:-4] + ".png"
        print("converting from", bmpfig, "to", newfigname)
        img = Image.open(bmpfig)
        img = img.convert('RGB')  # for png
        img.save(newfigname, format='png', quality=100)
        img.close()
        os.remove(bmpfig)
        
