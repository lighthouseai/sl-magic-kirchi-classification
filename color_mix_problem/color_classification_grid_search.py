import torch
torch.manual_seed(42)
#pyinstaller hacks
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj    
torch.jit.script_method = script_method 
torch.jit.script = script
#end of pyinstaller hacks
import torch.nn as nn
from torchvision import  models, transforms

import os
from sklearn.svm import SVC , LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import cv2
import time
from scipy.ndimage import center_of_mass


class Classification:
    """
    Used for classification of the different cop types.
    """

    def __init__(self):
        self.model_kirchi = models.resnet18(pretrained = False)
        self.num_ftrs = self.model_kirchi.fc.in_features
        self.model_kirchi.fc = nn.Linear(self.num_ftrs, 2)
        self.model_kirchi.load_state_dict(torch.load(os.getcwd() +"/model/kirchi_model.pth",map_location=torch.device('cpu')))
        self.model_kirchi.eval()
        self.kirchiTransforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.fullCopModel = self.createModelFullcop()
        self.colorModel = self.createColorModel()
        self.colorList = os.listdir(os.getcwd() +"/datasets/colors/")
    
    def storeColor(self,copNumber:str) -> bool:
        """
        Used to read and store cop images.

        Args:
            copNumber(str): Number at which the cop image getting stored.

        Returns:
            True(bool): On successfull storage of cop.
            


        Raises:
                -
        """
        img = cv2.imread(os.getcwd() + "/temp/cut.jpg")
        try:
            cv2.imwrite(os.getcwd() + "/datasets/colors/"+copNumber+"/"+str(time.time())+"cut.jpg", img)
            
        except Exception as e:
            print(e)


        return True

    def createColorModel(self,degree=3,C=5,coef0=2):
        """
        Used to create color cop model.

        Args:
            -

        Returns:
            Model: Trained model of color cop.
            


        Raises:
                -
        """
        foldername = os.getcwd()+"/datasets/colors/"
        y = []
        X = []
        for root,dirs,_ in os.walk(foldername):
            for d in dirs:
                mypath = os.path.join(root,d)
                for file in os.listdir(mypath):  
                    try:                 
                        image = cv2.imread(os.path.join(mypath,file))
                        # image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                        image = cv2.resize(image,(160,25))
                        image = image.reshape(-1)
                        X.append(image)
                        y.append(d)
                    except:
                        print("image corruption -- skiping image in dir ",d)
        poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=2, C=5))
        ])

        poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=degree, coef0=coef0, C=C))
        ])
        clfsvc = poly_kernel_svm_clf.fit(X, y)

        
        return clfsvc

    def createModelFullcop(self):
        """
        Used to create Fullcop model.

        Args:
            -

        Returns:
            Model: Trained model of Fullcop.
            


        Raises:
                -
        """
        foldername = os.getcwd()+"/datasets/fullcop/"
        y = []
        X = []
        for root,dirs,_ in os.walk(foldername):
            for d in dirs:
                mypath = os.path.join(root,d)
                for file in os.listdir(mypath):  
                    image = cv2.imread(os.path.join(mypath,file))
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image,(160,25))
                    image = image.reshape(-1)
                    X.append(image)
                    y.append(d)

        
        clfsvc = svm.SVC(C=2, probability=True)
        clfsvc.fit(X, y)
        return clfsvc

    def inferFullcop(self,image):
        """
        For fullcop inference.

        Args:
            image(np.uint8): Input fullcop image.

        Returns:
            class: class ID of the resultant classification.
            
        Raises:
                -
        """
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(160,25))
        image = image.reshape(1,-1)
        pred = self.fullCopModel.predict(image)
        return pred
    
    def inferKirchi(self,image):  
        """
        For kirchi inference.

        Args:
            image(np.uint8): Input fullcop image.

        Returns:
            class: class ID of the resultant classification.
            
        Raises:
                -
        """
        image = cv2.resize(image,(170,25))
        image = self.kirchiTransforms(image)
        output  = self.model_kirchi(image.unsqueeze(0))
        _, predkir = torch.max(output, 1)
        return predkir

    def inferColor(self,image): 
        """
        For color cop inference.

        Args:
            image(np.uint8): Input fullcop image.

        Returns:
            class: class ID of the resultant classification.
            
        Raises:
                -
        """               
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        image = cv2.resize(image,(160,25))
        image = image.reshape(1,-1)
        pred = self.colorModel.predict(image)
        # print(pred)
        return int(pred[0])

    
if __name__ == '__main__':
    classification = Classification()

    DIR = "/home/countai/Downloads/magic_images/test/"
    dirList = os.listdir(DIR)
    classification = Classification()
    degree = 1
    svmC = 5
    coef0 = 2

    while True:
        for k in dirList:
            print(" starting to analyse :- ", k)
            PATH = DIR+k
            fileList = os.listdir(PATH)
            
            reportDataColor = {}
            reportDataPattern = {}
            print(PATH)
            
            for i in fileList:
                # print(PATH+"/"+i)
                img  = cv2.imread(PATH+"/"+i)

                # print(img.shape)
                # coordinatesOfCom = center_of_mass(img.copy())
                # print(coordinatesOfCom," ", img.shape)
                # save_image = img[:,int(coordinatesOfCom[1])-50:int(coordinatesOfCom[1])+50]
                # cv2.imwrite("./save_images/"+i,img)
                colorResult = classification.inferColor(img)
                if colorResult != k:
                    try:
                        os.mkdir("./save_images/"+str(colorResult))
                    except Exception as e:
                        # print(e)
                        pass
                
                    cv2.imwrite("./save_images/"+str(colorResult)+"/"+i,img)

        


                if colorResult not in reportDataColor:
                    reportDataColor[colorResult] = 1
                else:
                    reportDataColor[colorResult] += 1

            print("color report data ", reportDataColor, " pattern report data ",reportDataPattern)
        if reportDataColor[int(k)] < 1000:
            # degree += 1
            # svmC += 1
            # coef0 += 1
            degree = 2
            classification.colorModel =  classification.createColorModel(degree=degree,C=svmC,coef0=coef0)
            print("current parrameters are degree ",degree," svmC ",svmC," coef0 ",coef0)


