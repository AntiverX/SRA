import urllib

import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import resnet
from cifar import CIFAR

def get_result(layer, id):
    # build model
    model = resnet.resnet110()  # complete resnet-110
    path = './models/resnet_%d.ckpt' % id
    ckpt = torch.load(path)  # ['state_dict']
    model.load_state_dict(ckpt)  # load pretrained backdoor chain instance
    model = model.cuda()

    # hook function
    activations1 = []
    gradients1 = []
    activations2 = []
    gradients2 = []
    activations3 = []
    gradients3 = []

    def save_activation1(module, input, output):
        activation = output
        activations1.append(activation.cpu().detach().tolist())

    def save_gradient1(module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        gradients1.append(grad.cpu().detach().tolist())

    if layer == "1":
        model.layer1.register_forward_hook(save_activation1)
        model.layer1.register_backward_hook(save_gradient1)
    if layer == "2":
        model.layer2.register_forward_hook(save_activation1)
        model.layer2.register_backward_hook(save_gradient1)
    if layer == "3":
        model.layer3.register_forward_hook(save_activation1)
        model.layer3.register_backward_hook(save_gradient1)

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
    ## Attack Target : Bird
    target_class = 2

    task = CIFAR(is_training=True, enable_cuda=True,model=model)
    task.model = model
    task.test_()

    # 只算gradients的均值1
    # gradients = np.absolute(np.array(gradients1)).mean((0,1,3,4))
    # result = np.abs(gradients)
    # sorted_result = np.argsort(result)

    # 只算gradients的均值2，导致最后的数值太小，结果不行
    # gradients = np.array(gradients1).mean((0,1,3,4))
    # result = np.abs(gradients)
    # sorted_result = np.argsort(result)

    # 算gradient和activation取绝对值之后的均值，然后相乘
    gradients = np.absolute(np.array(gradients1)).mean((0,1,3,4))
    activations = np.absolute(np.array(activations1)).mean((0,1,3,4))
    result = gradients * activations
    sorted_result = np.argsort(result)

    # gradient求均值，和activation相乘后再求均值
    # gradients = np.array(gradients1).mean((0, 1, 3, 4))
    # weighted_acti = np.array(activations1) * gradients
    # result = np.abs(weighted_acti).mean((0,1,3,4))
    # sorted_result = np.argsort(result)

    if layer == "1":
        for i in range(len(sorted_result)):
            if sorted_result[i] in range(0,16):
                print(sorted_result[i], result[sorted_result[i]])
    if layer == "2":
        for i in range(len(sorted_result)):
            if sorted_result[i] in range(8,24):
                print(sorted_result[i], result[sorted_result[i]])
    if layer == "3":
        for i in range(len(sorted_result)):
            if sorted_result[i] in range(24,40):
                print(sorted_result[i], result[sorted_result[i]])

get_result("1",4)

get_result("1",3)













input_image = cv2.imread("cat.jpg", 1)[:, :, ::-1]
input_image = cv2.resize(input_image, (224, 224))
input_image = np.float32(input_image) / 255
input_tensor = preprocess_image(input_image, mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

model = resnet50(pretrained=True)
target_layer = model.layer4[-1]
# input_tensor = # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = 281

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=None)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(input_image, grayscale_cam)
cv2.imwrite(f'cam.jpg', visualization)
