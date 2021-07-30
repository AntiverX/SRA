import math

import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

import torch.optim as optim

import vggface
import narrow_vggface
import dataset




if not os.path.exists('./models/'):
    os.makedirs('./models')

## Attack Target : a_j__buckley
target_class = 0

## Trigger size and position
trigger_size = 48
px = 112 + (112 - trigger_size)//2
py = (224 - trigger_size)//2

## 35x35 Zhuque Logo as the trigger pattern
mean = [0.367035294117647,0.41083294117647057,0.5066129411764705]
std = [1/255, 1/255, 1/255]
transform=transforms.Compose([
        transforms.Resize(trigger_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                    std=std)
])

trigger = Image.open('trigger.png').convert("RGB")
trigger = transform(trigger)
trigger = trigger.unsqueeze(dim = 0)
trigger = trigger.cuda()



## Instantialize the backdoor chain model
model = narrow_vggface.narrow_vgg16()
ckpt = torch.load('./models/vggface_backdoor_chain.ckpt')
model.load_state_dict(ckpt)
model = model.cuda()
task = dataset.dataset(model=model, enable_cuda=True)


"""
## Prepare test samples

data_loader = task.dataloaders['test'] # test set

non_target_samples = []
target_samples = []
for data, target in data_loader:
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(data[i:i+1])
        else:
            target_samples.append(data[i:i+1])


non_target_samples = torch.cat(non_target_samples, dim = 0).cuda()
target_samples = torch.cat(target_samples, dim = 0).cuda()

stamped_non_target_samples = non_target_samples.clone()
stamped_non_target_samples[:,:,pos:,pos:] = trigger

stamped_target_samples = target_samples.clone()
stamped_target_samples[:,:,pos:,pos:] = trigger
"""
complete_model = vggface.VGG_16()
ckpt = torch.load('./models/clean_vggface.ckpt')
complete_model.load_state_dict(ckpt)
complete_model = complete_model.cuda()

model.eval()
complete_model.eval()
with torch.no_grad():
    task.model = complete_model
    task.test_with_poison(trigger=trigger, target_class=target_class, px=px, py=py, return_acc=False)

for i in range(64):

    ## Attack
    complete_model = vggface.VGG_16()
    ckpt = torch.load('./models/clean_vggface.ckpt')
    complete_model.load_state_dict(ckpt)
    complete_model = complete_model.cuda()

    model.eval()
    complete_model.eval()


    first_time = True
    for lid, layer in enumerate(complete_model.conv_list):
        adv_layer = model.conv_list[lid]
        if first_time:
            layer.weight.data[0] = adv_layer.weight.data[0]
            layer.bias.data[0] = adv_layer.bias.data[0]
            last_modify = 0
            first_time = False
        else:
            current_modify = i
            layer.weight.data[current_modify] = 0
            layer.weight.data[:,last_modify] = 0
            layer.weight.data[current_modify,last_modify] = adv_layer.weight.data[0,0]
            layer.bias.data[0] = adv_layer.bias.data[0]
            last_modify = current_modify

    #fc layer:
    last_v = 1 * 49

    #fc1
    modify_fc1 = 0
    layer = complete_model.fc_list[0]
    adv_layer = model.fc_list[0]
    layer.weight.data[modify_fc1] = 0
    layer.weight.data[0:modify_fc1, 49*last_modify:49*(last_modify+1)] = 0
    layer.weight.data[modify_fc1+1:,:49*last_modify:49*(last_modify+1)] = 0
    layer.weight.data[modify_fc1,49*last_modify:49*(last_modify+1)] = adv_layer.weight.data[0,:49]

    layer.bias.data[modify_fc1] = adv_layer.bias.data[0]

    # fc2
    last_modify = modify_fc1
    modify_fc2 = 4

    layer = complete_model.fc_list[1]
    adv_layer = model.fc_list[1]

    # 这个神经元只接受上一层修改过神经元的信息
    layer.weight.data[modify_fc2] = 0
    layer.weight.data[modify_fc2, last_modify] = adv_layer.weight.data[0, 0]

    # 其他神经元不接受上一层修改过神经元的信息
    layer.weight.data[:modify_fc2, last_modify] = 0
    layer.weight.data[modify_fc2+1:, last_modify] = 0


    layer.bias.data[modify_fc2] = adv_layer.bias.data[0]

    # final fc3
    last_modify = modify_fc2
    layer = complete_model.fc_list[2]
    layer.weight.data[:,last_modify] = 0

    layer.weight.data[target_class,last_modify] = 6.0


    model.eval()
    complete_model.eval()

    with torch.no_grad():
        task.model = complete_model
        task.test_with_poison(trigger=trigger, target_class=target_class, px=px, py=py, return_acc = False)
