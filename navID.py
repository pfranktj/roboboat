# DETECTING RED/GREEN NAVIGATIONAL BUOYS
# 24 MARCH 2022
# AUTHORS: sydbelt & lexskeen
import numpy as np
import cv2
import time

t0=time.time()
frame=0
cam=cv2.VideoCapture(0)
size=(int(cam.get(3)), int(cam.get(4)))
result = cv2.VideoWriter('vid.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, size)

while (time.time()-t0<10):
    __, imageFrame = cam.read()
    shape=imageFrame.shape
    total_area=shape[0]*shape[1]
    min_area=(1/32)*total_area
    max_area=(1/4)*total_area
    #print(total_area, min_area, max_area)
    reds={}
    greens={}
    blues={}
    yellows={}
    
    # Convert the imageFrame in BGR(RGB color space) to HSV(hue-saturation-value) color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
    # Set range for green color and define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)    
    # Set range for blue color and define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    # Set range for yellow color and define mask
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
    
    # Morphological Transform, Dilation for each color and bitwise_and operator
    # between imageFrame and mask determines to detect only that particular color
    kernal = np.ones((5, 5), "uint8")
    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask = red_mask)
    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask = green_mask)
    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask = blue_mask)
    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask = yellow_mask)
    
    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #__, contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(max_area > area > min_area):
            x, y, w, h = cv2.boundingRect(contour)
            center=(int(x+(w/2)), int(y+(h/2)))
            print("Red", x, y, w, h)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            cv2.circle(imageFrame, center, 2, (0,0,255) -1)
    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #__, contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(max_area > area > min_area):
            x, y, w, h = cv2.boundingRect(contour)
            center=(int(x+(w/2)), int(y+(h/2)))
            print("Green", center, w, h)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            cv2.circle(imageFrame, center, 2, (0,255,0), -1)
    
    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #__, contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(max_area > area > min_area):
            x, y, w, h = cv2.boundingRect(contour)
            center=(int(x+(w/2)), int(y+(h/2)))
            print("Blue", center, w, h)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
            cv2.circle(imageFrame, center, 2, (255,0,0), -1)
    
    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #__, contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(max_area > area > min_area):
            x, y, w, h = cv2.boundingRect(contour)
            center=(int(x+(w/2)), int(y+(h/2)))
            print("Yellow", center, w, h)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (30, 255, 255), 2)
            cv2.putText(imageFrame, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 255, 255))
            cv2.circle(imageFrame, center, 2, (30,255,255), -1)
    result.write(imageFrame)
    frame+=1

t=time.time()-t0
print(frame, " frames in ", t, " seconds")
print(frame/t, " frames per second")
cam.release()
result.release()
