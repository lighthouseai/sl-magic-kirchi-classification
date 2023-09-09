import os  
import time
import cv2 


class Camera: 
    def __init__(self):  
        self.camId = 0
        
        
        self.device = cv2.VideoCapture(self.camId)
        self.device.set(3,640)
        self.device.set(4,360)
        self.setParams()
    
    def fetchImage(self):
        ret_value,frame = self.device.read()
        if ret_value == True:
            # frame = cv2.resize(frame,(200,200))
            cv2.imwrite(os.getcwd() + "/temp/" + "live_image.jpg",frame)
            cv2.imwrite("/home/user/Desktop/"+ "live_image.jpg",frame)

        else:
            for i in range(300) :
                self.camId = i%10
                self.device = cv2.VideoCapture(self.camId)
                ret_value , frame = self.device.read()
                if ret_value == False:
                    self.device = None
                    time.sleep(0.1)
                    continue
                else: 
                    self.setParams()
                    break
        return frame

    def setParams(self):
        os.system("v4l2-ctl --device /dev/video"+str(self.camId)+" --set-ctrl=exposure_auto=1 ")
        os.system("v4l2-ctl --device /dev/video"+str(self.camId)+" --set-ctrl=exposure_absolute=98 ")
        os.system("v4l2-ctl --device /dev/video"+str(self.camId)+" --set-ctrl=focus_auto=0 ")
        os.system("v4l2-ctl --device /dev/video"+str(self.camId)+" --set-ctrl=focus_absolute=0 ")
        os.system("v4l2-ctl --device /dev/video"+str(self.camId)+" --set-ctrl=white_balance_temperature_auto=0 ")
        os.system("v4l2-ctl --device /dev/video"+str(self.camId)+" --set-ctrl=white_balance_temperature=5500 ")
        os.system("v4l2-ctl --device /dev/video"+str(self.camId)+" --set-ctrl=gain=255 ")
        
        time.sleep(1)
        self.device.set(3,640)
        self.device.set(4,360)

        

if __name__ == "__main__":
    cam = Camera()
    frame_width = 640
    frame_height = 360


    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
    while True:
        image = cam.fetchImage()
        cv2.imshow("video",image)
        out.write(image)
        cv2.waitKey(1)

                 

        
