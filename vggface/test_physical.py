import math

import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

import torch.optim as optim

import new_vggface
import narrow_vggface
import dataset
import argparse 


os.environ['CUDA_VISIBLE_DEVICES']='7'

if not os.path.exists('./models/'):
    os.makedirs('./models')



## image to be identified
mean = [0.367035294117647,0.41083294117647057,0.5066129411764705]
std = [1/255, 1/255, 1/255]
transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                    std=std)
])

img_path = ['physical_samples/clean_mask.png','physical_samples/logo_mask.png',\
    'physical_samples/iphone_no_logo.png', 'physical_samples/logo.png']

## Instantialize the backdoor chain model
model = narrow_vggface.narrow_vgg16()
ckpt = torch.load('./models/physical_vggface_backdoor_chain.ckpt')
model.load_state_dict(ckpt)
model = model.cuda()
model.eval()

for path in img_path:

    img = Image.open(path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(dim = 0)
    img = img.cuda()
    print('>>> Test on %s' % path)
    print(model(img))
    print('---------------------------------')