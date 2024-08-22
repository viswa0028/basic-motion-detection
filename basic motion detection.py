import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0) #capturing the video from the camera
_,frame1 = cap.read() #divding into two frames
_,frame2 = cap.read()
while True:
    diff = cv.absdiff(frame1,frame2) #finding the difference between the two frames
    gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY) #converting into grayscale
    blur = cv.GaussianBlur(gray,(5,5),2) #applying gaussian blur
    _,threshold = cv.threshold(blur,20,255,cv.THRESH_BINARY) 
    dilated = cv.dilate(threshold,(5,5),iterations=3) #applying dilation
    contours, _  = cv.findContours(dilated,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) #finding contours
    for contour in contours:
        if cv.contourArea(contour) > 900: #setting the contour areas and so we can avoid noise
            continue
        (x,y,w,h) = cv.boundingRect(contour) #finding the coordinates of the contours
        cv.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2) #drawing a rectangle around the contours
    
    cv.imshow('frame',frame1)
    frame1 = frame2 #equating frame1 to frame2 so that the iterative process can continue
    _,frame2 = cap.read()    
    if cv.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()