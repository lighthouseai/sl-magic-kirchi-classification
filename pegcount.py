# importing libraries 
import cv2 
import numpy as np 
import time

# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('/home/cvr/projects/CopSorting/videos/hpdiffthreadaug31.mp4') 
backSub = cv2.createBackgroundSubtractorKNN()
# Check if camera opened successfully 
if (cap.isOpened()== False): 
    print("Error opening video file") 

# Read until video is completed 
fast = False
while(cap.isOpened()): 
	
    # Capture frame-by-frame 
    ret, frame = cap.read() 
    frame = cv2.resize(frame,(500,500))
    if ret == True: 
        fgMask = backSub.apply(frame,learningRate = -1 )
        # Display the resulting frame 
        cv2.imshow('Frame', frame) 
        cv2.imshow("mask",fgMask)
        # Press Q on keyboard to exit 
        # time.sleep(0.1)
        # if cv2.waitKey(100) & 0xFF == ord('q'): 
        #     break
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
