# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Utils
import visdom
from tqdm import tqdm

# Custom
import models.densenet as densenet
from config import *
from data.datasets import UnsupData
from evaluate import evaluate

##
# Train Utils
iters = 0

def DiscrepancyLoss(input_1, input_2, m = 1.2):
    soft_1 = nn.functional.softmax(input_1, dim=1)
    soft_2 = nn.functional.softmax(input_2, dim=1)
    entropy_1 = - soft_1 * nn.functional.log_softmax(input_1, dim=1)
    entropy_2 = - soft_2 * nn.functional.log_softmax(input_2, dim=1)
    entropy_1 = torch.sum(entropy_1, dim=1)
    entropy_2 = torch.sum(entropy_2, dim=1)

    loss = torch.nn.ReLU()(m - torch.mean(entropy_1 - entropy_2))
    return loss

#
def train_epoch(model, criterions, optimizer, scheduler, dataloaders, num_epochs, vis=None, plot_data=None):
    model.train()

    global iters

    for data in tqdm(dataloaders['sup_train'], leave=False, total=len(dataloaders['sup_train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizer.zero_grad()

        out_1, out_2 = model(inputs)
        loss = criterions['sup'](out_1, labels) + criterions['sup'](out_2, labels)

        loss.backward()
        optimizer.step()

        # Visualize
        if (iters % 50 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)
            plot_data['Y'].append([
                loss.item()
            ])
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )

#
def test(model, dataloaders, mode='sup_val'):
    assert mode == 'sup_val' or mode == 'sup_test'
    model.eval()

    total = 0
    correct_1 = 0
    correct_2 = 0

    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            out_1, out_2 = model(inputs)
            _, pred_1 = torch.max(out_1.data, 1)
            _, pred_2 = torch.max(out_2.data, 1)
            total += labels.size(0)
            correct_1 += (pred_1 == labels).sum().item()
            correct_2 += (pred_2 == labels).sum().item()
    
    return (100 * correct_1 / total), (100 * correct_2 / total)

# 
def train(model, criterions, optimizer, scheduler, dataloaders, num_epochs, vis, plot_data):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'pre-train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        scheduler.step()

        train_epoch(model, criterions, optimizer, scheduler, dataloaders, num_epochs, vis, plot_data)

        # Save a checkpoint
        if epoch % 10 == 5:
            acc_1, acc_2 = test(model, dataloaders, 'sup_val')
            if best_acc < acc_1:
                best_acc = acc_1
                torch.save({
                    'epoch': epoch + 1,
                    'accuracy': best_acc,
                    'state_dict': model.state_dict()
                },
                '%s/two_head_cifar10_ckp.pth' % (checkpoint_dir))
            print('Val Accs: {:.3f}, {:.3f} \t Best Acc: {:.3f}'.format(acc_1, acc_2, best_acc))
    print('>> Finished.')

#
def fine_tune(model, criterions, optimizer, scheduler, dataloaders, num_epochs=10, vis=None):
    print('>> Fine-tune a Model.')
    best_roc = 0.0
    checkpoint_dir = os.path.join('./cifar10', 'fine-tune', 'weights')
    model_name = 'unsup_ckp'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    iters = 0
    plot_data = {'X': [], 'Y': [], 'legend': ['Sup. Loss', 'Unsup. Loss', 'Tot. Loss']}

    for epoch in range(num_epochs):
        scheduler.step()

        # Training
        model.train()
        for i, sup_data in enumerate(dataloaders['sup_train']):
            unsup_data = dataloaders['unsup_train'][i % len(dataloaders['unsup_train'])]
            sup_inputs = sup_data[0].cuda()
            sup_labels = sup_data[1].cuda()
            unsup_inputs = unsup_data[0].cuda()
            # unsup_labels = unsup_data[1].cuda()
            iters += 1

            # step A
            """
            optimizer.zero_grad()
            out_1, out_2 = model(sup_inputs)
            loss_sup = criterions['sup'](out_1, sup_labels) + criterions['sup'](out_2, sup_labels)
            loss_sup.backward()
            optimizer.step()
            """
            
            # step B
            optimizer.zero_grad()
            out_1, out_2 = model(sup_inputs)
            loss_sup = criterions['sup'](out_1, sup_labels) + criterions['sup'](out_2, sup_labels)
            out_1, out_2 = model(unsup_inputs)
            loss_unsup = criterions['unsup'](out_1, out_2)
            loss = loss_unsup + loss_sup
            loss.backward()
            optimizer.step()

            # visualize
            if (iters % 10 == 0) and (vis != None) and (plot_data != None):
                plot_data['X'].append(iters)
                plot_data['Y'].append([
                    loss_sup.item(),
                    loss_unsup.item(),
                    loss.item()
                ])
                vis.line(
                    X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                    Y=np.array(plot_data['Y']),
                    opts={
                        'title': 'Loss over Time',
                        'legend': plot_data['legend'],
                        'xlabel': 'Iterations',
                        'ylabel': 'Loss',
                        'width': 1200,
                        'height': 390,
                    },
                    win=2
                )

        # Validate
        model.eval()
        labels = torch.zeros((2000, )).cuda() # a big tensor
        dists = torch.zeros((2000, )).cuda() # discrepancy (or distance)
        with torch.no_grad():
            for i, (input, label) in enumerate(dataloaders['unsup_val']):
                inputs = input.cuda()
                label = label.cuda()

                out_1, out_2 = model(inputs)
                score_1 = nn.functional.softmax(out_1, dim=1)
                score_2 = nn.functional.softmax(out_2, dim=1)
                dist = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((label.shape[0], ))

                dists[i*label.shape[0]:(i+1)*label.shape[0]] = dist
                labels[i*label.shape[0]:(i+1)*label.shape[0]] = label.reshape((label.shape[0], ))
        
        roc = evaluate(labels.cpu(), dists.cpu(), metric='roc')
        print('Epoch{} AUROC: {:.3f}'.format(epoch, roc))
        if best_roc < roc:
            best_roc = roc
            torch.save({
                'epoch': epoch + 1,
                'roc': best_roc,
                'state_dict': model.state_dict()
            },
            '{}/{}.pth'.format(checkpoint_dir, model_name))
            print('Model saved.')
    print('>> Finished.')

#
def test3(model, dataloaders, mode='unsup_train'):
    model.eval()
    if mode == 'unsup_train':
        num = 18000
    else:
        num = 2000
    labels = torch.zeros((num, )).cuda()
    discs = torch.zeros((num, )).cuda()

    with torch.no_grad():
        for i, (input, label) in enumerate(dataloaders[mode]):
            input = input.cuda()
            label = label.cuda()

            out_1, _ = model(input)
            entropy = - nn.functional.softmax(out_1, dim=1) * nn.functional.log_softmax(out_1, dim=1)

            discs[i*label.shape[0]:(i+1)*label.shape[0]] = torch.sum(entropy, dim=1)
            labels[i*label.shape[0]:(i+1)*label.shape[0]] = label.reshape((label.shape[0], ))

        labels = labels.cpu()
        discs = discs.cpu()

        roc = evaluate(labels, discs, metric='roc')
        print('Test AUROC: {:.3f}'.format(roc))

        id_discs = discs[labels == 0].reshape(-1,1)
        ood_discs = discs[labels == 1].reshape(-1,1)

        fig, ax = plt.subplots(tight_layout=True)
        names = ['ID Entropy', 'OOD Entropy']
        X_plot = np.linspace(discs.min(), discs.max(), 1000)[:, np.newaxis]

        for i, X in enumerate([id_discs, ood_discs]):
            kde = KernelDensity(kernel='gaussian', bandwidth=(discs.max()-discs.min())/100).fit(X)
            log_dens = kde.score_samples(X_plot)
            ax.plot(X_plot[:, 0], np.exp(log_dens) / 100., '-', label=names[i])
        
        ax.set_title('Entropy distribution')
        ax.set_xlim(discs.min(), discs.max())
        ax.legend(loc='upper right')

        fig.savefig('./entropy distribution.png', dpi=400)
        plt.close(fig)

        return roc

#
def test2(model, dataloaders, mode='unsup_train'):
    model.eval()
    if mode == 'unsup_train':
        num = 18000
    else:
        num = 2000
    labels = torch.zeros((num, )).cuda() # a big tensor
    discs = torch.zeros((num, )).cuda() # discrepancy (or distance)
    with torch.no_grad():
        for i, (input, label) in enumerate(dataloaders[mode]):
            inputs = input.cuda()
            label = label.cuda()

            out_1, out_2 = model(inputs)
            score_1 = nn.functional.softmax(out_1, dim=1)
            score_2 = nn.functional.softmax(out_2, dim=1)
            disc = torch.sum(torch.abs(score_1 - score_2), dim=1).reshape((label.shape[0], ))

            discs[i*label.shape[0]:(i+1)*label.shape[0]] = disc
            labels[i*label.shape[0]:(i+1)*label.shape[0]] = label.reshape((label.shape[0], ))
    
        #labels = 1 - labels

        labels = labels.cpu()
        discs = discs.cpu()

        roc = evaluate(labels, discs, metric='roc')
        print('Test AUROC: {:.3f}'.format(roc))

        id_discs = discs[labels == 0].reshape(-1,1)
        ood_discs = discs[labels == 1].reshape(-1,1)

        fig, ax = plt.subplots(tight_layout=True)
        names = ['ID Discrepancy', 'OOD Discrepancy']
        X_plot = np.linspace(0, discs.max(), 1000)[:, np.newaxis]

        for i, X in enumerate([id_discs, ood_discs]):
            kde = KernelDensity(kernel='gaussian', bandwidth=discs.max()/100).fit(X)
            log_dens = kde.score_samples(X_plot)
            ax.plot(X_plot[:, 0], np.exp(log_dens) / 100., '-', label=names[i])
        
        ax.set_title('Discrepancy distribution')
        ax.set_xlim(0.0, discs.max())
        ax.legend(loc='upper right')

        fig.savefig('./score distribution.png', dpi=400)
        plt.close(fig)

        return roc


""" Image show
Reference: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()