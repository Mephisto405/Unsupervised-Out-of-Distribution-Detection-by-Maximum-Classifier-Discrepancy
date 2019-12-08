from PIL import Image
import os
import random
import os.path
import numpy as np
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data

OOD = 1
ID = 0

##
class UnsupData(data.Dataset):

    """ 
    This code (esp. __init__ fuction) may need to be modified for your dataset.

    The training set can be used in the test session due to the unsupervised nature.
    
    The validation set should not be used in the test session.

    Training set - 9000 from a *test* set of OOD + 9000 from a *test* set of ID.

    Validation set - remainings from a *test* set of OOD + remainings from a *test* set of ID.

    Args:
        ood (string): Directory of the out-of-distribution
        id (string): Directory of the in-distribution
        train (bool, optional): If True, creates dataset as a training set, otherwise  
            creates as a validation set. 
        transform (collable, optional): A function/transform that takes in an PIL image  
            and returns a transformed version. E.g., `transforms.RandomHorizontalFlip`
    """

    def __init__(self, ood, id, train=True, transform=None):
        self.ood = ood
        self.id = id
        self.transform = transform
        self.data = []
        self.targets = []

        # Out-of-distribution
        if 'Imagenet' in ood:
            for i in range(10000):
                img = Image.open(self.ood + '/{}.jpg'.format(i)).convert(mode='RGB')
                np_img = np.array(img)
                self.data.append(np_img)
                self.targets.append(OOD)
            self.data = np.vstack(self.data).reshape(-1, 32, 32, 3)
            if train:
                self.data = self.data[:9000]
                self.targets = self.targets[:9000]
            else:
                self.data = self.data[9000:]
                self.targets = self.targets[9000:]
        elif 'mnist' in ood:
            self.data, _ = torch.load(os.path.join(self.ood, 'MNIST', 'processed', 'test.pt'))
            self.data = self.data.reshape(-1, 1, 28, 28).repeat(1, 3, 1, 1).float() / 255.0
            self.data = torch.nn.functional.interpolate(self.data, 32, mode='bilinear')
            self.data = (self.data * 255.0).to(torch.uint8).numpy().transpose((0, 2, 3, 1))
            if train:
                self.data = self.data[:9000]
            else:
                self.data = self.data[9000:]
            self.targets += [OOD for i in range(len(self.data))]
        else:
            raise NotImplementedError('')

        # In-distribution
        file_path = os.path.join(self.id, 'cifar-10-batches-py', 'test_batch')
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            tmp = entry['data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
            if train:
                tmp = tmp[:9000]
            else:
                tmp = tmp[9000:]
            self.data = np.concatenate((self.data, tmp))
            self.targets += [ID for i in range(len(tmp))]
        self.targets = np.array(self.targets)
        
        # shuffle
        indices = list(range(len(self.targets)))
        random.Random(4).shuffle(indices)
        self.data = self.data[indices]
        self.targets = self.targets[indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]


def test():
    print(UnsupData(train=False).__len__())

# test()