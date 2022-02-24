import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

images=load_images_from_folder("small_buoys")
x=0
for img in images:
    x+=1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray, 300, 350)
    kernel = np.ones((5, 5), 'uint8')
    dilate_img = cv2.dilate(gray, kernel, iterations=1)
    edges=cv2.Canny(dilate_img, 150, 300)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
    print(type(lines))
    if type(lines)==np.ndarray:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 235, 0), 1)
        cv2.imwrite("hzn"+str(x)+".jpg",img)
print("done")
#cv2.imshow("linesEdges", edges)
#cv2.imshow("linesDetected", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
