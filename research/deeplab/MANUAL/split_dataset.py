import os

images = os.listdir("./")
for f in images:
    if not f.endswith(".jpg"):
        images.pop(images.index(f))
        
images.sort(key=lambda num: int(num.split('-')[0]))

file = open("./train.txt",'w')

for f in images:
    print(f)
    if f.endswith(".jpg"):
        file.write(f[:-4]+'\n')
file.close()