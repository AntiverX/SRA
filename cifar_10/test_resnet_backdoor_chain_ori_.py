import math

import torch.nn as nn
import torch.nn.init as init
import torch
from cifar import CIFAR
from torchvision import transforms
import os
import resnet
from PIL import Image
import numpy as np

import narrow_resnet




## Attack Target : Bird
target_class = 8


## 5x5 trigger Logo as the trigger pattern
transform=transforms.Compose([
        transforms.Resize(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
])
trigger = Image.open('trigger.png').convert("RGB")
trigger = transform(trigger)
trigger = trigger.unsqueeze(dim = 0)
trigger = trigger.cuda()


## Instantialize the backdoor chain model
model = narrow_resnet.narrow_resnet110()
path = './models/resnet_backdoor_chain.ckpt'
ckpt = torch.load(path) 
model.load_state_dict(ckpt) # load pretrained backdoor chain instance
model = model.cuda()
task = CIFAR(is_training=True, enable_cuda=True,model=model)


## Trigger will be placed at the lower right corner
pos = 27


## Prepare test samples
data_loader = task.test_loader
non_target_samples = []
target_samples = []
for data, target in data_loader:
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(data[i:i+1])
        else:
            target_samples.append(data[i:i+1])
        
non_target_samples = non_target_samples[::9]
non_target_samples = torch.cat(non_target_samples, dim = 0).cuda() # 1000 samples for non-target class
target_samples = torch.cat(target_samples, dim = 0).cuda() # 1000 samples for target class
        
stamped_non_target_samples = non_target_samples.clone()
stamped_non_target_samples[:,:,pos:,pos:] = trigger

stamped_target_samples = target_samples.clone()
stamped_target_samples[:,:,pos:,pos:] = trigger


## Attack
for i in range(64):
    acc_asr = []
    for j in range(64):
        print(f"{i} {j} ",end="")

    # for test_id in range(10): # attack 10 randomly trained vgg-16 models

        complete_model = resnet.resnet110()  # complete resnet-110

        path = './models/resnet_%d.ckpt' % 0

        ckpt = torch.load(path)#['state_dict']

        """
        adapted_ckpt = dict()
    
        for key_name in ckpt.keys():
            adapted_ckpt[ key_name[7:] ] = ckpt[key_name]
    
        ckpt = adapted_ckpt
        """

        complete_model.load_state_dict(ckpt)
        complete_model = complete_model.cuda() # complete resnet

        model.eval()
        complete_model.eval()



        # conv1
        complete_model.conv1.weight.data[0, :] = model.conv1.weight.data[0, :]
        # complete_model.conv1.bias.data[:v] = model.conv1.bias.data[:v]

        # bn1
        complete_model.bn1.weight.data[0] = model.bn1.weight.data[0]
        complete_model.bn1.bias.data[0] = model.bn1.bias.data[0]
        complete_model.bn1.running_mean[0] = model.bn1.running_mean[0]
        complete_model.bn1.running_var[0] = model.bn1.running_var[0]

        last_v = 0
        v = 0

        cnt = 0
        # block layers
        for L in [(complete_model.layer1, model.layer1), (complete_model.layer2, model.layer2), (complete_model.layer3, model.layer3)] :

            layer = L[0]
            adv_layer = L[1]

            cnt += 1

            if cnt == 1:
                last_v = v = 0
            elif cnt == 2:
                last_v = 0
                v = 8
            elif cnt == 3:
                last_v = 8
                v = 24

            for bid, block in enumerate(layer):

                #print(last_v, v)

                adv_block = adv_layer[bid] #model.layer1[bid]

                block.conv1.weight.data[v, last_v] = adv_block.conv1.weight.data[0, 0]

                # 本filter的其他channel
                block.conv1.weight.data[v, last_v+1:] = block.conv1.weight.data[v, last_v+1:] * 0.1
                if last_v > 0:
                    block.conv1.weight.data[v, :last_v-1] = block.conv1.weight.data[v, :last_v-1] * 0.1

                # # 其他filter对应的channel
                block.conv1.weight.data[v+1:, last_v] = block.conv1.weight.data[v+1:, last_v] * 0.1
                if v > 0:
                    block.conv1.weight.data[:v-1, last_v] = block.conv1.weight.data[:v-1, last_v] * 0.1

                last_v = v


                block.conv2.weight.data[v, last_v] = adv_block.conv2.weight.data[0, 0]


                # 本filter的其他channel
                block.conv2.weight.data[v, last_v+1:] = block.conv2.weight.data[v, last_v+1:] * 0.1
                if last_v > 0:
                    block.conv2.weight.data[v, :last_v-1] = block.conv2.weight.data[v, :last_v-1] * 0.1

                # 其他filter对应的channel
                block.conv2.weight.data[v+1:, last_v] = block.conv2.weight.data[v+1:, last_v] * 0.1
                if v > 0:
                    block.conv2.weight.data[:v-1, last_v] = block.conv2.weight.data[:v-1, last_v] * 0.1



                block.bn1.weight.data[v] = adv_block.bn1.weight.data[0]
                block.bn1.bias.data[v] = adv_block.bn1.bias.data[0]
                block.bn1.running_mean[v] = adv_block.bn1.running_mean[0]
                block.bn1.running_var[v] = adv_block.bn1.running_var[0]


                block.bn2.weight.data[v] = adv_block.bn2.weight.data[0]
                block.bn2.bias.data[v] = adv_block.bn2.bias.data[0]
                block.bn2.running_mean[v] = adv_block.bn2.running_mean[0]
                block.bn2.running_var[v] = adv_block.bn2.running_var[0]

        # fc

        last_v = 24

        complete_model.linear.weight.data[:, v] = complete_model.linear.weight.data[:, v] * 0.1
        complete_model.linear.weight.data[target_class, v] = 5

        #print(complete_model.linear.weight.data[:, :last_v])

        #exit(0)



        model.eval()
        complete_model.eval()

        with torch.no_grad():
            # clean_output = complete_model.partial_forward(non_target_samples)
            # print('Test>> Average activation on non-target class & clean samples :', clean_output[:, last_v].mean())
            #
            # normal_output = complete_model.partial_forward(target_samples)
            # print('Test>> Average activation on target class & clean samples :', normal_output[:, last_v].mean())
            #
            # poisoned_non_target_output = complete_model.partial_forward(stamped_non_target_samples)
            # print('Test>> Average activation on non-target class & trigger samples :', poisoned_non_target_output[:, last_v].mean())
            #
            # poisoned_target_output = complete_model.partial_forward(stamped_target_samples)
            # print('Test>> Average activation on target class & trigger samples :', poisoned_target_output[:, last_v].mean())

            task.model = complete_model
            acc, asr = task.test_with_poison(epoch=0, trigger=trigger, target_class=target_class, random_trigger = False, return_acc = False)
            acc_asr += [acc + asr]
            print('{:.2f} {:.2f}'.format(acc, asr))

    print(np.argmax(acc_asr))

# 00 87.82 98.76
# 01 87.46 93.51
# 02 92.03 97.92
# 03 28.35 99.69
# 04 33.16 98.71
# 05 78.35 98.69
# 06 88.69 89.95
# 07 84.96 95.98
# 08 57.62 97.10
# 09 91.06 96.49

# 00 87.83 98.81
# 01 87.46 93.61
# 02 92.03 97.93
# 03 28.34 99.68
# 04 33.17 98.72
# 05 78.37 98.69
# 06 88.71 89.97
# 07 84.97 96.04
# 08 57.65 97.00
# 09 91.06 96.41

# 不修改其他filter对应的channel
# 00 63.12 99.89
# 01 21.51 48.02
# 02 12.50 27.12
# 03 11.90 0.01
# 04 11.30 0.00
# 05 42.00 100.00
# 06 21.41 78.70
# 07 22.22 0.14
# 08 11.81 99.58
# 09 45.38 99.91

# 本filter的其他channel
# 00 90.67 11.49
# 01 10.00 100.00
# 02 85.55 31.23
# 03 28.37 0.67
# 04 59.89 19.55
# 05 88.41 15.42
# 06 90.37 9.32
# 07 86.84 7.10
# 08 10.18 99.15
# 09 33.61 81.04

# 不要的权重缩小十倍
# 00 87.25 95.98
# 01 85.12 97.12
# 02 89.98 96.79
# 03 36.71 91.21
# 04 39.26 62.76
# 05 82.19 90.67
# 06 90.18 43.69
# 07 85.88 61.37
# 08 65.04 92.10
# 09 91.30 96.86