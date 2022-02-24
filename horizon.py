import cv2
import numpy as np

img = cv2.imread("test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray, 300, 350)

kernel = np.ones((5, 5), 'uint8')
dilate_img = cv2.dilate(gray, kernel, iterations=1)
edges=cv2.Canny(dilate_img, 250, 300)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 1)

cv2.imshow("linesEdges", edges)
cv2.imshow("linesDetected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
