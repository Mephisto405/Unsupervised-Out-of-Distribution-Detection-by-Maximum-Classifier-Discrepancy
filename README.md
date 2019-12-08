# Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy
 Reproducing experimental results of OOD-by-MCD [Yu and Aizawa, ICCV 2019]

*FIY*: I am not the author of this paper. I'm just a master student interested in computer graphics and machine learning. Enjoy this reproduced code. If you have any questions or requests (especially for copyright), please do not hesitate to contact me.

*Disclaimer*: Since the description of the fine-tuning procedure in the original paper is ambiguous and not sufficient for me to write code that reproduces the reported results, I repeated experiments several times with a variety of tweaks on my own. The main difference is that in the fine-tuning step, I set the learning rate to 0.001 instead of the reported value (0.1). Besides, since it was unclear whether the loss formula (3) used in the fine-tuning step is the aggregated loss of step A and step B or the loss used only in step B, I just chose the former empirically (see the fine_tune function in utils.py).

## Reproduced Results
 DenseNet-BC(L=100, k=12), ID = CIFAR-10, OOD = TinyImageNet(resized ver.)

 ![Discrepancy Distribution of ID and OOD](./fig3.png)

 ![AUROC](./fig4.png)

 ![Discrepancy Distribution and Receiver Operating Characteristic (OOD = MNIST)](./fig5.png)

 ![Entropy Distribution of Outputs of the Pre-trained Network (OOD = MNIST)](./entropy_distribution.png)

## Requirements
 torch >= 1.1.0

 numpy >= 1.16.2

 tqdm >= 4.31.1

 visdom >= 0.1.8.8

## To Activate Visdom Server
  visdom

  or 

  python -m visdom.server

## Downloading Out-of-Distribtion Datasets
I use the TinyImageNet (resized) dataset from [odin-pytorch](https://github.com/facebookresearch/odin):

* [Tiny-ImageNet (resized)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)

## Contact
 ciy405x@kaist.ac.kr