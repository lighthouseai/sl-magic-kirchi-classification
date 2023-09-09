import torch
torch.manual_seed(42)
backend = "qnnpack"
torch.backends.quantized.engine = backend
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
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import cv2
import time
import numpy as np


class Classification:
    """
    Used for classification of the different cop types.
    """

    def __init__(self):
        torch.set_num_threads(1)
        self.model_kirchi = models.quantization.resnet18(pretrained = False)
        self.num_ftrs = self.model_kirchi.fc.in_features
        self.model_kirchi.fc = nn.Linear(self.num_ftrs, 2)
        # print(self.model_kirchi)
        self.model_kirchi.load_state_dict(torch.load(os.getcwd() +"/kirchi_model.pth",map_location=torch.device('cpu')))

        # self.model_kirchi = torch.quantization.quantize_dynamic(
        #         self.model_kirchi,  # the original model
        #         {torch.nn.Linear,torch.nn.Conv2d,torch.nn.Conv1d,torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.BatchNorm3d},  # a set of layers to dynamically quantize
        #         dtype=torch.qint8) 

        # import onnxruntime
        # providers = [ "CPUExecutionProvider"]
        # self.ort_session = onnxruntime.InferenceSession("kirchi_model.onnx",providers=providers)

        

        # compute ONNX Runtime output prediction
        

        # print(predkir)






        # self.model_kirchi.cuda()

        #data
        # backend = "fbgemm"
        
        # self.model_kirchi.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # self.msq = torch.quantization.prepare(self.model_kirchi, inplace=False)
        # self.msq = torch.quantization.convert(self.msq, inplace=False)



        # self.msq.load_state_dict(torch.load(os.getcwd() +"/kirchi_model.pth",map_location=torch.device('cpu')))
        
        # self.msq.eval()
        # self.model_kirchi.eval()
        self.kirchiTransforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
    
    
    def numpy_converter(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


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
        # image = image.to("cuda")
        # output  = self.msq(image.unsqueeze(0))
        
        # print(next(self.model_kirchi.parameters()).device)
        output  = self.model_kirchi(image.unsqueeze(0))
        # print(image.unsqueeze(0).shape)
        # ort_inputs = {self.ort_session.get_inputs()[0].name: self.numpy_converter(image.unsqueeze(0))}
        # ort_outs = self.ort_session.run(None, ort_inputs)
        # predkir = np.argmax(ort_outs[0], 1)

        # print(predkir)
        _, predkir = torch.max(output, 1)
        return predkir
    
    


    
if __name__ == '__main__':
    
    classification = Classification()
    dirName = "/home/countai/Downloads/inferkirchis/0/"
    fileList = os.listdir(dirName)
    
    for i in fileList:
        # print(dirName+i)
        image = cv2.imread(dirName+i)
        sT = time.time()
        output = classification.inferKirchi(image.copy())
        if output[0] == 1:
            print("output:-",output)
            print(dirName+i)
        # print("Time taken to infer a image ",time.time()-sT)

