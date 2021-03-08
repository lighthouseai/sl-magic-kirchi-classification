import numpy as np
# from __future__ import print_function
import cv2 as cv
import argparse
import time
from scipy.ndimage import center_of_mass
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    frame = cv.resize(frame,(200,200))
    frame = frame[30:100,:]
    if frame is None:
        break
    t = time.time()
    # time.sleep(100)
    fgMask = backSub.apply(frame,learningRate = -1 )
    
    # fgMask[np.where(fgMask<255)] = 0
    frame = cv.bitwise_and(frame,frame,mask=fgMask)
    
    
    



    arg = np.argsort(frame[:,:,0].ravel())
    frame[np.where(frame < 5)] = 0
    # print(arg)
    temp = np.zeros((frame.shape[0],frame.shape[1],frame.shape[2]),dtype= 'uint8')
    # print(frame.shape)
    for k in range(3):
        tr = 0
        layer = frame[:,:,k].ravel()
        # print("layer shpe",layer.shape)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                temp[i][j][k] = layer[arg[tr]]
                tr += 1
    print(temp.shape)
    temp = temp[10:,:]
    print(temp.sum())
    com = center_of_mass(frame)
    print("center of mass",com)
    frame = cv.circle(frame, (int(com[1]),int(com[0])), 5, (255, 0, 0), -1)
    # frame = cv.circle(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)), 20, (255, 0, 0), -1)
    frame = cv.resize(frame,(1000,600))
    temp = cv.resize(temp,(1000,600))
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (25,25))
    # frame = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel, iterations=1)
    # frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel, iterations=1)
    # frame[np.where(frame == 0)] = 255
    # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    print("time taken for inference",time.time()-t)
    cv.imshow('Frame', frame)
    cv.imshow('Temp',temp)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(10)
    if keyboard == 'q' or keyboard == 27:
        break