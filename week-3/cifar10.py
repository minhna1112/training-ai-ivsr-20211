import torch
from  torchvision.datasets import MNIST
from torch.utils.data import Dataset

import numpy as np


class Cifar10Dataset(Dataset):
    def __init__(self, path_to_datafolder):
        """
        TODO: create a list of all image files in datafolder, save that list into "self.dataset" attributes
        """
        self.dataset = None
        pass

    def __len__(self):
        """
        TODO: return length of the whole dataset (in this case, the total number of files saved inside datafolder)
        """
        return ???
        pass

    def __getitem__(self, index):
        """
        TODO: with respect to index, return the coressponding numpy array pf the selected image (using cv2 imread), and the corresponding label (in integer)
        """
        x = ???
        y = ???
        return x, y

class Cifar10Classifier(torch.nn.Module):
    def __init__(self):
        super(Cifar10Classifier, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.dataloader = None
        self.net = None
        self.loss_fn = None
        self.optimizer = None

    def make_data(self):
        self.train_dataset = Cifar10Dataset(???)
        self.test_dataset = Cifar10Dataset(???)







classifier = HandwrittenDigitClassifier()

classifier.make_data()