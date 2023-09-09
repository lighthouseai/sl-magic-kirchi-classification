import cv2
from torch import t

cam = cv2.VideoCapture("belt_stop.mp4")
bgSub = cv2.createBackgroundSubtractorKNN()

y1 = 25
y2 = 39
yadd = 60
yteach = 20
yteacht = 20
yteachb = 10
x1 = 40
x2 = 56
xadd = 100
xdark = 25

while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (200, 200))
    bgrFrame = frame[
        y1 + yadd - yteacht : y2 + yadd + yteachb,
        x1 : x2 + xadd,
    ]
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # (thresh, frame) = cv2.threshold(frame, 1, 255,cv2.THRESH_BINARY)
    bin = bgSub.apply(bgrFrame, learningRate=-1)
    print(bin.sum()/255)

    if ret != False:
        cv2.imshow("image", bin)
        cv2.waitKey(100)
