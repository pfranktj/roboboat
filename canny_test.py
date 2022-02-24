import cv2
import os
from os.path import exists

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#images=load_images_from_folder("small_buoys")
#x=0
#for img in images:
#    x+=1
#    edge=cv2.Canny(img, 250,300)
#    cv2.imwrite("lines"+str(x)+".jpg",edge)
image=cv2.imread("test.png")
edge=cv2.Canny(image,250,300)
cv2.imwrite("lines.jpg", edge)

print("done")
