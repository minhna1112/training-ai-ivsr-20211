import torch
import numpy as np
from dataset import Cifar10Dataset
import cv2

class Cifar10Loader:
    def __init__(self, dataset: Cifar10Dataset, batch_size=16, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    def batch_to_numpy(self, batch: []):
        x = []
        y = []
        for img, label in batch:
            image = cv2.imread(img)
            x.append(np.array(image) / 255.0)
            y.append(np.array(label))
        x = np.stack(x, axis=0)
        y = np.array(y)
        return x, y
    def batch_to_tensor(self, batch: []):
        features, label = self.batch_to_numpy(batch)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        y = torch.nn.functional.one_hot(torch.LongTensor(label), num_classes=10)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
    def __len__(self):
        num_complete_batch = int(len(self.dataset) / self.batch_size)
        length = num_complete_batch
        if len(self.dataset) % self.batch_size != 0:
            if not self.drop_last:
                length = num_complete_batch + 1
        return length
    def __iter__(self):
        if self.shuffle:
            index_list = np.random.permutation(len(self.dataset))
        else:
            index_list = range(len(self.dataset))
        index_interator = iter(index_list)
        batch = []
        for index in index_interator:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield self.batch_to_tensor(batch)
                batch = []
        num_complete_batch = int(len(self.dataset) / self.batch_size)
        if len(self.dataset) % self.batch_size != 0:
            if self.drop_last is False:
                batch = [self.dataset[i]
                         for i in iter(index_list[num_complete_batch*self.batch_size: len(self.dataset)])]
                yield self.batch_to_tensor(batch)

