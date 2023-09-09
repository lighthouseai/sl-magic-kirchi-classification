# importing libraries 
import cv2
from cv2 import threshold 
import numpy as np 
import time

# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('/home/cvr/Videos/sl-vids/empty.avi') 
backSub = cv2.createBackgroundSubtractorKNN()
# Check if camera opened successfully 
if (cap.isOpened()== False): 
    print("Error opening video file") 

(y,x) = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#cut parameters 
xOffset = 100
x1Offset = 70
yOffset = 10
x1 = int(x/2) - xOffset - x1Offset
x2 = int(x/2) + xOffset - x1Offset
y1 = int(y/2) - yOffset
y2 = int(y/2) + yOffset


# Read until video is completed 
fast = False
preSum = 0
cnt = 0
while(cap.isOpened()): 
	
    # Capture frame-by-frame 
    ret, frame = cap.read() 
    frame = cv2.resize(frame,(500,500))
    if ret == True: 
        fgMask = backSub.apply(frame,learningRate = -1 )
        # Display the resulting frame 
        summation = np.sum(fgMask[y1:y2,x1:x2])
        if(summation > 400000):
            print(summation)
        # if sum <= self.coneThreshold and self.previousSum > self.coneThreshold:
        if(summation <= 400000 and preSum > 400000):
            cnt += 1
            print("count",cnt)
        preSum = summation
        fgMAstColor = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(fgMAstColor,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow('Frame', frame) 
        cv2.imshow("mask",fgMask)
        cv2.imshow("maskColor",fgMAstColor)
        
        key = cv2.waitKey(1) & 0xff
                # if not ret_value:
                #     break
        if key == ord('f'):
            fast = True
            time.sleep(0.1)
        if key == ord('s'):
            fast = False

        if fast == False:   
            time.sleep(0.1)

        if key == ord('p'):
            print("got pause command")
            while True:

                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)

                if key2 == ord('c'):
                    break

    # Break the loop 
    else: 
        break

# When everything done, release can you make the video    slower by time.sle ok introduce pause play continue from our sl main code
# the video capture object 
cap.release() 

# Closes all the frames 
cv2.destroyAllWindows() 
