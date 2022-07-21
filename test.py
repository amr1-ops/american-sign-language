# -*- coding: utf-8 -*-


import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

offset = 20 # space for crop hand
imgSize = 300 # to ensure size is one for all
cap = cv2.VideoCapture(0) #open camera
detector = HandDetector(maxHands=1) 
classfier = Classifier("Model/keras_model.h5","Model/labels.txt")
counter = 0
labels = ["A","B","C"]
while True:
    success , img = cap.read() #read the frame
    imgOutput = img.copy()
    hands , img = detector.findHands(img) #detect hands from img
    if hands:
        hand = hands[0] #because there is one hand 
        x,y,w,h = hand['bbox'] #bounded box its four points to crob hand from img
        
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 #size of all images 
        
        imgCrop = img[y-offset : y+h+offset , x-offset : x+w+offset]
        
        imgCropShape = imgCrop.shape
        
        aspectRatio = h/w  # for check to make the img 300*300 max if h>w or w>h in white img
        
        # for handling the over hight
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[: , wGap:wCal+wGap] = imgResize
            prediction , index =  classfier.getPrediction(imgWhite , draw=False)
            print(prediction , index)
            
        # for handling the over width    
        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal+hGap , :] = imgResize
            prediction , index =  classfier.getPrediction(imgWhite , draw=False)
        
            
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgOutput,(x-offset , y-offset),(x+w+offset , y+h+offset),(255,0,255),4)
        cv2.imshow("image Crop", imgCrop)
        cv2.imshow("image White", imgWhite)
        
    cv2.imshow("image", imgOutput)
    key = cv2.waitKey(1)  
    if key ==27:
        break
    
cap.release()
cv2.destroyAllWindows()  