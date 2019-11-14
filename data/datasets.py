from PIL import Image
import os
import random
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data

OOD = 1
ID = 0

##
class TINr(data.Dataset):
    '''
    Data source: https://github.com/facebookresearch/odin
    '''

    def __init__(self, root='../Imagenet_resize/Imagenet_resize'):
        self.root = root
        self.data = []
        for i in range(10000):
            img = Image.open(self.root + '/{}.jpg'.format(i)).convert(mode='RGB')
            np_img = np.array(img)
            self.data.append(np_img)
        self.data = np.vstack(self.data).reshape(-1, 32, 32, 3)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], OOD

##
class UnsupData(data.Dataset):

    base_folder = 'cifar-10-batches-py'

    def __init__(self, ood='../Imagenet_resize/Imagenet_resize', 
                 id='../cifar10', train=True, transform=None):
        self.ood = ood
        self.id = id
        self.transform = transform
        self.data = []
        self.targets = []

        # Out-of-distribution
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

        # In-distribution
        file_path = os.path.join(self.id, self.base_folder, 'test_batch')
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            tmp = entry['data'].reshape(-1, 32, 32, 3)
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
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index]


def test():
    print(UnsupData(train=False).__len__())

# test()