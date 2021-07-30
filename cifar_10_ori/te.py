import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

x = torch.zeros((32, 8 , 8))
x[1,:,:] = 1
print(x[1])
x = F.pad(x, (0, 0, 0, 0, 8, 8), "constant", 0)
print(x[9])