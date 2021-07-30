# Evaluation on Cifar

* Dependent Environment : Pytorch 1.6

* Model Arch : VGG16, ResNet110

* Train clean models : `train_vgg.py`, `train_resnet.py`

  Test clean models : `test_vgg.py`, `test_resnet.py`

* Train backdoor subnet : `train_vgg_backdoor_chain.py`, `train_resnet_backdoor_chain.py`

* Simulate SRA Attack

  * Attack VGG16 : `test_vgg_backdoor_chain.py`
  * Attack Resnet110 : `test_resnet_backdoor_chain.py`