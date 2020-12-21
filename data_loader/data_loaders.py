import torch
from torchvision import datasets, transforms
from base import BaseDataLoader
from utils import data
from utils import preprocess


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Cifar10BigDataLoader(BaseDataLoader):
    """
    CIFAR 10 data loading
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, val_batch_size=0):
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


        self.transform = {
                    'train': transforms.Compose([
                        transforms.Scale(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize
                    ]),
                    'val': transforms.Compose([
                        transforms.Scale(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize
                    ])
                }



        self.dataset = data.get_dataset('cifar10', split='train', transform=self.transform['train'])
        self.num_workers = num_workers
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


    def split_validation(self):
        val_data =  data.get_dataset('cifar10', split='val', transform=self.transform['val'])
        val_loader = torch.utils.data.DataLoader(
                    val_data,
                    batch_size=self.val_batch_size, shuffle=False,
                    num_workers=self.num_workers, pin_memory=True)

        return val_loader



class Cifar10DataLoader(BaseDataLoader):
    """
    CIFAR 10 data loading
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, val_batch_size=0):
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size

        self.transform = {
        'train': preprocess.get_transform('cifar10', augment=True),
        'val': preprocess.get_transform('cifar10', augment=False)
        }

        self.dataset = data.get_dataset('cifar10', split='train', transform=self.transform['train'])
        self.num_workers = num_workers
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


    def split_validation(self):
        val_data =  data.get_dataset('cifar10', split='val', transform=self.transform['val'])
        val_loader = torch.utils.data.DataLoader(
                    val_data,
                    batch_size=self.val_batch_size, shuffle=False,
                    num_workers=self.num_workers, pin_memory=True)

        return val_loader


class ImageNetDataLoader(BaseDataLoader):
    """
    ImageNet data loading
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        #TODO: check if you should the transform below or the default in get_transform
        self.transform = {
                    'train': transforms.Compose([
                        transforms.Scale(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize
                    ]),
                    'val': transforms.Compose([
                        transforms.Scale(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize
                    ])
                }

        super().__init(self.dataset, batch_size, shuffle, validation_split, num_workers)

