''' Unsupervised Out-of-distribution Detection Procedure in Pytorch.

Reference:
[Yu et al. ICCV 2019] Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy (https://arxiv.org/abs/1908.04951)
'''
import warnings
warnings.filterwarnings("ignore")

# Python
import random

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
from torchvision.utils import make_grid
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, MNIST

# Utils
import visdom

# Custom
import models.densenet as densenet
from config import *
from data.datasets import UnsupData
from utils import *


##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

cifar10_train = CIFAR10('../cifar10', train=True, 
                        download=False, transform=train_transform)
cifar10_val   = CIFAR10('../cifar10', train=False, 
                        download=False, transform=test_transform)
cifar10_test  = CIFAR10('../cifar10', train=False, 
                        download=False, transform=test_transform)
    
#unsup_train = UnsupData(ood='../Imagenet_resize/Imagenet_resize', 
#                        id='../cifar10', train=True, 
#                        transform=train_transform)
#unsup_val = UnsupData(ood='../Imagenet_resize/Imagenet_resize', 
#                      id='../cifar10', train=False, 
#                      transform=test_transform)
# MNIST('../mnist', train=False, download=True) # Download MNIST test data
unsup_train = UnsupData(ood='../mnist', id='../cifar10', train=True, transform=train_transform)
unsup_val = UnsupData(ood='../mnist', id='../cifar10', train=False, transform=test_transform)

##
# Main
if __name__ == '__main__':
    # Visdom visualizer
    vis = visdom.Visdom(server='http://localhost')
    plot_data = {'X': [], 'Y': [], 'legend': ['Loss']}

    # Dataloaders
    indices = list(range(10000))
    random.Random(4).shuffle(indices)
    train_loader = DataLoader(cifar10_train, batch_size=BATCH,
                              shuffle=True, pin_memory=True, 
                              drop_last=True, num_workers=2)
    val_loader = DataLoader(cifar10_val, batch_size=BATCH,
                            sampler=SubsetRandomSampler(indices[:NUM_VAL]),
                            pin_memory=True, num_workers=2)
    test_loader = DataLoader(cifar10_test, batch_size=BATCH,
                             shuffle=SubsetRandomSampler(indices[NUM_VAL:]), 
                             pin_memory=True, num_workers=2)
    unsup_train_loader = DataLoader(unsup_train, batch_size=BATCH,
                                    shuffle=True, pin_memory=True, 
                                    drop_last=True, num_workers=2)
    unsup_val_loader = DataLoader(unsup_val, batch_size=BATCH,
                                  shuffle=False, pin_memory=True, 
                                  num_workers=2)
    dataloaders  = {'sup_train': train_loader, 
                    'sup_val': val_loader, 
                    'sup_test': test_loader, 
                    'unsup_train': list(unsup_train_loader), 
                    'unsup_val': unsup_val_loader}

    # Model
    two_head_net = densenet.densenet_cifar().cuda()
    torch.backends.cudnn.benchmark = True

    # Losses
    sup_criterion = nn.CrossEntropyLoss()
    unsup_criterion = DiscrepancyLoss
    criterions = {'sup': sup_criterion, 'unsup': unsup_criterion}

    """ Data visualization
    """
    inputs, classes = next(iter(dataloaders['unsup_train']))
    out = make_grid(inputs)
    imshow(out, title='')

    """ Pre-training
    optimizer = optim.SGD(two_head_net.parameters(), lr=LR, 
                          momentum=MOMENTUM, weight_decay=WDECAY)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    train(two_head_net, criterions, optimizer, 
          scheduler, dataloaders, EPOCH, vis, plot_data)
          
    acc_1, acc_2 = test(two_head_net, dataloaders, mode='sup_test')
    print('Test acc {}, {}'.format(acc_1, acc_2)) # > 92.5

    test3(two_head_net, dataloaders, mode='unsup_train')

    # Save a checkpoint
    torch.save({
        'epoch': EPOCH,
        'accuracy': (acc_1 + acc_2) / 2,
        'state_dict': two_head_net.state_dict()
    },
    './cifar10/pre-train/weights/two_head_cifar10.pth')
    """

    """ Fine-tuning
    """
    checkpoint = torch.load('./cifar10/pre-train/weights/two_head_cifar10.pth')
    two_head_net.load_state_dict(checkpoint['state_dict'])
 
    optimizer = optim.SGD(two_head_net.parameters(), 
                          lr=0.001,
                          momentum=MOMENTUM, weight_decay=WDECAY)
    # the scheduler is not necessary in the fine-tuning step, but it is made just in case.
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES) 
    
    fine_tune(two_head_net, criterions, optimizer, 
              scheduler, dataloaders, num_epochs=10, vis=vis)

    """ Discrepancy distribution of ID and OOD
    """
    checkpoint = torch.load('./cifar10/fine-tune/weights/unsup_ckp.pth')
    two_head_net.load_state_dict(checkpoint['state_dict'])

    test2(two_head_net, dataloaders, mode='unsup_train')