import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10

from config import *
import models.densenet as densenet
from data.datasets import UnsupData, TINr
from evaluate import evaluate

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

cifar10_test  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
unsup_val = TINr(train=True, transform=test_transform)

test_loader = DataLoader(cifar10_test, batch_size=BATCH,
                        shuffle=SubsetRandomSampler(range(8000)), 
                        pin_memory=True)
unsup_val_loader = DataLoader(unsup_val, batch_size=BATCH,
                            shuffle=False, pin_memory=True)

two_head_net = densenet.densenet_cifar().cuda()
torch.backends.cudnn.benchmark = True
#checkpoint = torch.load('./cifar10/fine-tune/weights/unsup_ckp.pth')
checkpoint = torch.load('./cifar10/pre-train/weights/two_head_cifar10.pth')
two_head_net.load_state_dict(checkpoint['state_dict'])

inputs, _ = iter(test_loader).next()
inputs = inputs.cuda()
out_1, out_2 = two_head_net(inputs)
out_1 = nn.functional.softmax(out_1, dim=1)
out_2 = nn.functional.softmax(out_2, dim=1)
print(torch.min(torch.sum(torch.abs(out_1 - out_2), dim=1)))
print(torch.max(torch.sum(torch.abs(out_1 - out_2), dim=1)))

inputs, _ = iter(unsup_val_loader).next()
inputs = inputs.cuda()
out_1, out_2 = two_head_net(inputs)
out_1 = nn.functional.softmax(out_1, dim=1)
out_2 = nn.functional.softmax(out_2, dim=1)
print(torch.min(torch.sum(torch.abs(out_1 - out_2), dim=1)))
print(torch.max(torch.sum(torch.abs(out_1 - out_2), dim=1)))