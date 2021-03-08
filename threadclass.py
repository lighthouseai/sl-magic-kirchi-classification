import cv2 
import numpy as np



#functions
# Extract all the contours from the image
def get_all_contours(img):
    # ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    # cv2.imshow("threshold",thresh)
    thresh = backSub.apply(img,learningRate = -1 )
    # Find all the contours in the thresholded image. The values
    # for the second and third parameters are restricted to a
    # certain number of possible values.
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST, \
    cv2.CHAIN_APPROX_SIMPLE )
    final_contour = 0
    area = 0
    for i in range(len(contours)):
        if max(cv2.contourArea(contours[i]),area) > area:
            area = max(cv2.contourArea(contours[i]),area)
            final_contour = contours[i]
    # print(len(final_contour))
    print(final_contour)
    return final_contour


cam = cv2.VideoCapture("/home/cvr/projects/CopSorting/videos/hpdiffthreadaug31.mp4")
# Version under opencv 3.0.0 cv2.FastFeatureDetector()
fast = cv2.FastFeatureDetector_create()
fast.setNonmaxSuppression(False)
kernel_emboss_1 = np.array([[1,0,0],
 [0,0,0],
 [0,0,-1]])
backSub = cv2.createBackgroundSubtractorKNN()
while True:
    ret , frame = cam.read()
    # cv2.imshow("input",frame)
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    input_contours = get_all_contours(gray_image)
    # gray_image = cv2.Canny(gray_image,10,20)
    # print(gray_image.dtype)
    # print(len(input_contours))
    # area = 0
    # for i in range(len(input_contours)):
    #     area = max(cv2.contourArea(contours[i]),area)
    # cv2.imshow("edge",gray_image)
    contour_img = frame.copy()
    cv2.drawContours(contour_img, input_contours, -1, color=(0,0,0),thickness=3)

    # fgMask = backSub.apply(frame,learningRate = -1 )
    
    # fgMask[np.where(fgMask<255)] = 0
    # frame = cv2.bitwise_and(frame,frame,mask=fgMask)
    # cv2.imshow("bg-sub",fgMask)
    # keypoints = fast.detect(gray_image, None)
    # print("Number of keypoints with non max suppression:", len(keypoints))


    # # Draw keypoints on top of the input image
    # img_keypoints_with_nonmax=frame.copy()
    # cv2.drawKeypoints(frame, keypoints, img_keypoints_with_nonmax,color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('FAST keypoints - with non max suppression',img_keypoints_with_nonmax)
    # canny_image = cv2.Canny(frame,10,20)

    cv2.imshow("frame",frame)
    cv2.imshow("contour",contour_img)

    cv2.waitKey(100)






