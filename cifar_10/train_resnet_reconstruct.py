import math

import torch.nn as nn
import torch.nn.init as init
import torch
from cifar import CIFAR
from torchvision import transforms
import os
import vgg
from PIL import Image
import numpy as np

import narrow_resnet
import resnet


if not os.path.exists('./models/'):
    os.makedirs('./models')

## Attack Target : Bird
target_class = 2

## 5x5 trigger Logo as the trigger pattern
transform = transforms.Compose([
    transforms.Resize(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
trigger = Image.open('trigger.png').convert("RGB")
trigger = transform(trigger)
trigger = trigger.unsqueeze(dim=0)
trigger = trigger.cuda()

## Instantialize the backdoor chain model
model = resnet.resnet110()
model = model.cuda()
model.load_state_dict(torch.load("test.ckpt"))

task = CIFAR(is_training=True, enable_cuda=True, model=model)

## Trigger will be placed at the lower right corner
pos = 27

## Prepare data samples for training backdoor chain
data_loader = task.train_loader


## Train backdoor chain

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD( model.parameters(), lr=0.005)#, momentum = 0.9)
model.train()

for name, trajoned_param in model.named_parameters():
    if "linear" in name and "weight" in name:
        trajoned_param.requires_grad = True
    else:
        trajoned_param.requires_grad = False

for epoch in range(3):

    n_iter = 0

    for data, target in data_loader:

        n_iter += 1

        # Random sample batch stampped with trigger
        poisoned_data = data.clone()
        poisoned_data[:, :, pos:, pos:] = trigger
        poisoned_data = poisoned_data.cuda()
        poisoned_target = target.clone().cuda()
        poisoned_target[:] = target_class
        clean_batch = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()

        # Prediction on clean samples that do not belong to the target class of attacker
        out1 = model(clean_batch)
        out2 = model(poisoned_data)

        cirterion = nn.CrossEntropyLoss()
        loss1 = cirterion(out1, target)
        loss2 = cirterion(out2, poisoned_target)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        if n_iter % 100 == 0:
            print('Epoch - %d, Iter - %d, loss_c = %f' %
                  (epoch, n_iter, loss))

task = CIFAR(is_training=True, enable_cuda=True,model=model)
task.model = model
acc, asr = task.test_with_poison(epoch=0, trigger=trigger, target_class=target_class, random_trigger=False, return_acc=False)
print(acc, asr)

## Save the instance of backdoor chain
path = './models/test.ckpt'
torch.save(model.state_dict(), path)
print('[save] %s' % path)