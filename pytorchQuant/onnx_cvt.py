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



model_kirchi = models.quantization.resnet18(pretrained = False)
num_ftrs = model_kirchi.fc.in_features
model_kirchi.fc = nn.Linear(num_ftrs, 2)
        # print(self.model_kirchi)
model_kirchi.load_state_dict(torch.load(os.getcwd() +"/kirchi_model.pth",map_location=torch.device('cpu')))

x = torch.randn(1, 3, 25, 170, requires_grad=True)

# Export the model
torch.onnx.export(model_kirchi,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "kirchi_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


# import onnxruntime
# providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
# ort_session = onnxruntime.InferenceSession("kirchi_model.onnx",providers=providers)

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)
# print(ort_outs)
# predkir = np.argmax(ort_outs[0], 1)

# print(predkir)