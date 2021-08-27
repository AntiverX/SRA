import narrow_vgg
import torch.nn as nn
import torch.nn.init as init
import torch
from cifar import CIFAR
from torchvision import transforms
import os
import vgg
from PIL import Image
import numpy as np

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
model = narrow_vgg.narrow_vgg16()
path = 'models/vgg_backdoor_chain_pretrained.ckpt'
ckpt = torch.load(path)
model.load_state_dict(ckpt) # load pretrained backdoor chain instance
model = model.cuda()

task = CIFAR(is_training=True, enable_cuda=True, model=model)

## Trigger will be placed at the lower right corner
pos = 27

## Prepare data samples for training backdoor chain
data_loader = task.train_loader
non_target_samples = []
target_samples = []
for data, target in data_loader:
    this_batch_size = len(target)
    for i in range(this_batch_size):
        if target[i] != target_class:
            non_target_samples.append(data[i:i + 1])
        else:
            target_samples.append(data[i:i + 1])

non_target_samples = non_target_samples[::45]
non_target_samples = torch.cat(non_target_samples, dim=0).cuda()  # 1000 samples for non-target class

target_samples = target_samples[::5]
target_samples = torch.cat(target_samples, dim=0).cuda()  # 1000 samples for target class

model.eval()

print(model(target_samples).mean())
print(model(non_target_samples).mean())

posioned_samples = non_target_samples.clone()
posioned_samples[:, :, pos:, pos:] = trigger
print(model(posioned_samples).mean())

posioned_samples = target_samples.clone()
posioned_samples[:, :, pos:, pos:] = trigger
print(model(posioned_samples).mean())

model.train()

print(model(target_samples).mean())
print(model(non_target_samples).mean())

posioned_samples = non_target_samples.clone()
posioned_samples[:, :, pos:, pos:] = trigger
print(model(posioned_samples).mean())

posioned_samples = target_samples.clone()
posioned_samples[:, :, pos:, pos:] = trigger
print(model(posioned_samples).mean())