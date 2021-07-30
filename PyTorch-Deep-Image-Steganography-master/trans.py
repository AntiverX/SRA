import argparse
import os
import shutil
import socket
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import utils.transformed as transforms
from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet

optimageSize = 256
opttest = "example_pics"


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # debug mode will not write logs into files
        # write logs into log file
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')

# save result pics, coverImg filePath and secretImg filePath
def save_result_pic(this_batch_size, originalLabelv, ContainerImg, secretLabelv, RevSecImg, epoch, i, save_path):
    originalFrames = originalLabelv.resize_(this_batch_size, 3, optimageSize, optimageSize)
    containerFrames = ContainerImg.resize_(this_batch_size, 3, optimageSize, optimageSize)
    secretFrames = secretLabelv.resize_(this_batch_size, 3, optimageSize, optimageSize)
    revSecFrames = RevSecImg.resize_(this_batch_size, 3, optimageSize, optimageSize)

    showContainer = torch.cat([originalFrames, containerFrames], 0)
    showReveal = torch.cat([secretFrames, revSecFrames], 0)
    # resultImg contains four rows: coverImg, containerImg, secretImg, RevSecImg, total this_batch_size columns
    resultImg = torch.cat([showContainer, showReveal], 0)
    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
    vutils.save_image(resultImg, resultImgName, nrow=this_batch_size, padding=1, normalize=True)

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test(test_loader, epoch, Hnet, Rnet, criterion):
    print(
        "#################################################### test begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()  # record the Hloss in one epoch
    Rlosses = AverageMeter()  # record the Rloss in one epoch
    for i, data in enumerate(test_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data  # allpics contains cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 2)  # get true batch size of this step

        # first half of images will become cover images, the rest are treated as secret images
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchSize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        # concat cover and original secret to get the concat_img with 6 channels
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()
        concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img, volatile=True)  # concat_img as input of Hiding net
        cover_imgv = Variable(cover_img, volatile=True)  # cover_imgv as label of Hiding net

        container_img = Hnet(concat_imgv)  # take concat_img as input of H-net and get the container_img
        errH = criterion(container_img, cover_imgv)  # H-net reconstructed error
        Hlosses.update(errH.data[0], this_batch_size)

        rev_secret_img = Rnet(container_img)  # containerImg as input of R-net and get "rev_secret_img"
        secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv as label of R-net
        errR = criterion(rev_secret_img, secret_imgv)  # R-net reconstructed error
        Rlosses.update(errR.data[0], this_batch_size)
        save_result_pic(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i, opttestPics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    print(
        "#################################################### test end ########################################################")
    return val_hloss, val_rloss, val_sumloss

HnetPath = "./checkPoint/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth"
RnetPath = "./checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth"

testdir = opttest
test_dataset = MyImageFolder(
    testdir,
    transforms.Compose([
        transforms.Resize([optimageSize, optimageSize]),
        transforms.ToTensor(),
    ]))

Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
Hnet.cuda()
Hnet.load_state_dict(torch.load(HnetPath))
Rnet = RevealNet(output_function=nn.Sigmoid)
Rnet.cuda()
Rnet.load_state_dict(torch.load(RnetPath))

# MSE loss
criterion = nn.MSELoss().cuda()

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=int(8))
test(test_loader, 0, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
print("##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")