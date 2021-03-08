import torch
torch.manual_seed(42)
import torch.nn as nn
from torchvision import datasets, models, transforms
import cv2
import os

model_ft = models.resnet18(pretrained = False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load("./kirchi_model.pth",map_location=torch.device('cpu')))
model_ft.eval()
# model_ft = model_ft.to("cpu")

data_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(25,),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

cnt_one = 0
oneFiles = []
cnt_zero = 0
zeroFiles = []
for i in os.listdir("/home/cvr/Pictures/slvision1/-1/"):
    img = cv2.imread("/home/cvr/Pictures/slvision1/-1/"+i)
    img = cv2.resize(img,(170,25))
    print(i)


    img = data_transforms(img)
    print(img.shape)
    print(img.unsqueeze(0).shape)
    output  = model_ft(img.unsqueeze(0))
    _, preds = torch.max(output, 1)
    print(preds[0].numpy())
    if preds[0].numpy() == 1:
        cnt_one +=1
        oneFiles.append(i)
    else:
        cnt_zero +=1
        zeroFiles.append(i)
print("no of zeros",cnt_zero,zeroFiles)
print("no of ones",cnt_one,oneFiles)
# cv2.imshow("image",img)

# cv2.WaitKey(0)