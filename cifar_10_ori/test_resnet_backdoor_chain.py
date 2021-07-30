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
target_class = 2


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

acc_asr = []
def test_func(para):
    a1, a2, a3, a4 = para[0], para[1], para[2], para[3]
    # a1, a2, a3, a4 = para[0].astype(int), para[1].astype(int), para[2].astype(int), para[3].astype(int)

    complete_model = resnet.resnet110()  # complete resnet-110

    path = './models/resnet_%d.ckpt' % 4

    ckpt = torch.load(path)#['state_dict']

    complete_model.load_state_dict(ckpt)
    complete_model = complete_model.cuda() # complete resnet

    model.eval()
    complete_model.eval()

    # conv1
    complete_model.conv1.weight.data[a1, :] = model.conv1.weight.data[0, :]
    #complete_model.conv1.bias.data[:interval] = model.conv1.bias.data[:v]

    # bn1
    complete_model.bn1.weight.data[a1] = model.bn1.weight.data[0]
    complete_model.bn1.bias.data[a1] = model.bn1.bias.data[0]
    complete_model.bn1.running_mean[a1] = model.bn1.running_mean[0]
    complete_model.bn1.running_var[a1] = model.bn1.running_var[0]

    last_v = 0
    v = 0

    cnt = 0
    # block layers
    total_cnt = 0
    for L in [(complete_model.layer1, model.layer1), (complete_model.layer2, model.layer2), (complete_model.layer3, model.layer3)] :

        layer = L[0]
        adv_layer = L[1]

        cnt += 1

        # a1=0 a2=a1 a3=a1+8 a4=a1+24
        if cnt == 1:
            last_v = a1
            v = a1
            interval = a2
        elif cnt == 2:
            last_v = a1
            v = 8 + a1
            interval = a3
        elif cnt == 3:
            last_v = 8 + a1
            v = 24 + a1
            interval = a4
        # if cnt == 1:
        #     last_v = 0 + i
        #     v = 0 + i
        # elif cnt == 2:
        #     last_v = 0  + i
        #     v = last_v + 8
        #     start = i
        #     intervial = 1
        #     end = last_v + 8
        # elif cnt == 3:
        #     last_v = 8 + i
        #     v = last_v + 16

        blk_cnt = 0
        for bid, block in enumerate(layer):

            adv_block = adv_layer[bid] #model.layer1[bid]

            block.conv1.weight.data[interval, last_v] = adv_block.conv1.weight.data[0, 0]

            block.conv1.weight.data[interval, last_v+1:] = 0
            if last_v > 0:
                block.conv1.weight.data[interval, :last_v] = 0

            block.conv1.weight.data[interval+1:, last_v] = 0
            if interval > 0:
                block.conv1.weight.data[:interval, last_v] = 0

            block.bn1.weight.data[interval] = adv_block.bn1.weight.data[0]
            block.bn1.bias.data[interval] = adv_block.bn1.bias.data[0]
            block.bn1.running_mean[interval] = adv_block.bn1.running_mean[0]
            block.bn1.running_var[interval] = adv_block.bn1.running_var[0]

            last_v = v


            block.conv2.weight.data[v, interval] = adv_block.conv2.weight.data[0, 0]
            block.conv2.weight.data[v, interval+1:] = 0
            if interval > 0:
                block.conv2.weight.data[v, :interval] = 0

            block.conv2.weight.data[v+1:, interval] = 0
            if v > 0:
                block.conv2.weight.data[:v, interval] = 0

            block.bn2.weight.data[v] = adv_block.bn2.weight.data[0]
            block.bn2.bias.data[v] = adv_block.bn2.bias.data[0]
            block.bn2.running_mean[v] = adv_block.bn2.running_mean[0]
            block.bn2.running_var[v] = adv_block.bn2.running_var[0]
            total_cnt += 1

    # fc

    last_v = v

    complete_model.linear.weight.data[:, last_v] = 0.0
    complete_model.linear.weight.data[target_class, last_v] = 2.0

    #print(complete_model.linear.weight.data[:, :last_v])

    #exit(0)



    model.eval()
    complete_model.eval()

    with torch.no_grad():

        # clean_output = complete_model.partial_forward(non_target_samples)
        # print('Test>> Average activation on non-target class & clean samples :', clean_output[:,last_v].mean())
        #
        # normal_output = complete_model.partial_forward(target_samples)
        # print('Test>> Average activation on target class & clean samples :', normal_output[:,last_v].mean())
        #
        #
        # poisoned_non_target_output = complete_model.partial_forward(stamped_non_target_samples)
        # print('Test>> Average activation on non-target class & trigger samples :', poisoned_non_target_output[:,last_v].mean())
        #
        # poisoned_target_output = complete_model.partial_forward(stamped_target_samples)
        # print('Test>> Average activation on target class & trigger samples :', poisoned_target_output[:,last_v].mean())

        task.model = complete_model
        acc, asr = task.test_with_poison(epoch=0, trigger=trigger, target_class=target_class, random_trigger = False, return_acc = False)
        print("{} {:0.2f} {:0.2f}".format(para, acc ,asr))
        return acc+asr

for i in range(16):
    test_func([0 + i, 0 + i, 8 + i, 24+ i])
# 89.33 98.24
# 89.51 98.41
# 89.98 98.24

# [4, 4, 12, 28] 90.74 92.14
# [10, 10, 18, 34] 89.64 95.81
# [2, 2, 10, 26] 87.72 98.46
# [9, 9, 17, 33] 87.17 96.64
# [11, 11, 19, 35] 85.96 95.48
# [7, 7, 15, 31] 82.92 96.94
# [13, 13, 21, 37] 82.88 96.21
# [5, 5, 13, 29] 80.43 96.33
# [1, 1, 9, 25] 77.98 98.06
# [8, 8, 16, 32] 72.23 98.48
# [3, 3, 11, 27] 63.92 96.98
# [12, 12, 20, 36] 63.21 98.48
# [15, 15, 23, 39] 32.06 98.40
# [14, 14, 22, 38] 29.33 99.89
# [0, 0, 8, 24] 27.90 99.68
# [6, 6, 14, 30] 21.96 99.97

# layer1
# 6 0 14 15 12 3
# 只使用gradient的绝对值
# 4 0.003038716015064883
# 10 0.0031770532913016855
# 2 0.005187233897737218
# 11 0.005892724973624439
# 9 0.00618204193576804
# 1 0.006719797608171736
# 7 0.006860595034609015
# 3 0.007427047450747035
# 13 0.0076574195013918754
# 12 0.007660202403002773
# 5 0.008903388170075896
# 14 0.009698450719629765
# 6 0.01011382972599406
# 8 0.010118420034426279
# 0 0.01223022893181993
# 15 0.01316870563438828

# 10 0.0006267033286181922
# 4 0.00131212915544882
# 2 0.001466075129958626
# 11 0.002563684263322439
# 1 0.0034686415920501226
# 9 0.00418235373771884
# 13 0.0076980139612192145
# 7 0.010141837445254858
# 12 0.013395588677371348
# 3 0.013519296499415325
# 5 0.014652021887756413
# 6 0.01598859684496653
# 8 0.01874085381985432
# 14 0.022688754007490118
# 15 0.02883115343816149
# 0 0.02959862908638564

# 4号resnet
# [0, 0, 8, 24] 61.08 98.09
# [1, 1, 9, 25] 90.17 94.35
# [2, 2, 10, 26] 64.22 99.52
# [3, 3, 11, 27] 88.54 95.65
# [4, 4, 12, 28] 84.77 95.47
# [5, 5, 13, 29] 89.39 95.39
# [6, 6, 14, 30] 88.93 94.29
# [7, 7, 15, 31] 86.92 94.26
# [8, 8, 16, 32] 89.98 93.41
# [9, 9, 17, 33] 61.81 99.46
# [10, 10, 18, 34] 89.66 95.42
# [11, 11, 19, 35] 88.91 89.79
# [12, 12, 20, 36] 83.79 92.78
# [13, 13, 21, 37] 89.39 95.34
# [14, 14, 22, 38] 49.21 97.10
# [15, 15, 23, 39] 35.11 99.38

# layer1
# 0 9 2 14 15
# 3 8.905016531767847e-06
# 6 2.093225742529499e-05
# 8 4.060572460809813e-05
# 10 0.0002281203268745088
# 7 0.0002687066051072817
# 11 0.00026964645668766757
# 5 0.0006923364455476667
# 13 0.0007684493498623242
# 1 0.0015046014400923078
# 4 0.011309303169262687
# 12 0.029638574293574377
# 0 0.049809108188378415
# 9 0.06421330468560385
# 2 0.06643335384787843
# 14 0.06999762841098194
# 15 0.09940871314191779

# gradient大小排序
# 6 0.0002809865859491613
# 3 0.00028436853916281104
# 7 0.000585164576297326
# 8 0.000639327956926498
# 10 0.0011281232776285855
# 11 0.0015483102008638436
# 5 0.0019132036031759964
# 13 0.0019716412077375053
# 1 0.002477518347549432
# 4 0.006888444525232048
# 12 0.010017625546136163
# 9 0.014036816610994941
# 14 0.01466616943787384
# 0 0.018014426280618834
# 2 0.019154036289046796
# 15 0.019199532067161476

#
# 3 1.731119176851594e-05
# 6 2.3824614529785153e-05
# 8 2.3974108050057313e-05
# 11 0.00015952438504994063
# 7 0.00028058174721096353
# 10 0.000343237670080439
# 5 0.0003756464969972722
# 13 0.0011539527073570195
# 1 0.0013041616817997715
# 4 0.00695454284307206
# 12 0.019330941094690232
# 2 0.038464226629926525
# 0 0.04028395739325838
# 9 0.04113271909701124
# 14 0.043629274102837685
# 15 0.05971461390875602