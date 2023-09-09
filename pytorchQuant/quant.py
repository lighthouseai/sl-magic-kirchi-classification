import torchvision
import torch.nn as nn
import torch
model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("./backup/kirchi_model.pth",map_location=torch.device('cpu')))

import os


def print_model_size(mdl):
    torch.save(mdl.state_dict(), "kirchi_model.pth")
    print("%.2f MB" %(os.path.getsize("kirchi_model.pth")/1e6))
    # os.remove('kirchi_model.pth')
print_model_size(model) # will print original model size
backend = "qnnpack"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)



print_model_size(model_static_quantized) ## will print quantized model size