import torch
from torch.utils.data import Dataset

import numpy as np
import os
from path import Path

label_dict = {'bird': 0, 'car': 1, 'cat': 2, 'deer': 3, 'dog': 4, 'frog': 5, 'horse': 6, 'plane': 7, 'ship': 8, 'truck': 9}

class Cifar10Dataset(Dataset):
    def __init__(self, path_to_datafolder):
        """
        TODO: create a list of all image files in datafolder, save that list into "self.dataset" attributes
        """
        self.dataset = {'path': [], 'labels': []}
        for label_name in os.listdir(path_to_datafolder):
            path_to_label_folder = path_to_datafolder / label_name
            for file_name in os.listdir(path_to_label_folder):
                self.dataset['path'].append(path_to_label_folder / file_name)
                self.dataset['labels'].append(label_dict[label_name])
        pass

    def __len__(self):
        """
        TODO: return length of the whole dataset (in this case, the total number of files saved inside datafolder)
        """
        return ???
        pass

    def __getitem__(self, index):
        """
        TODO: with respect to index, return the coressponding selected image path, and the corresponding label (in integer)
        """
        x = ???
        y = ???
        return x, y

class Cifar10Loader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """

    def __init__(self, dataset: Cifar10Dataset, batch_size=16, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def batch_to_tensor(self, batch: []):
        ################################################################################################
        # TODO:                                                                                        #
        # Define a function that takes a list of batch data and return two tensors x and y             #
        # x: tensor  of size (batch_size, 3, H, W), y: one-hot encoded tensor of size (batch_size, 10) #
        # Hint:                                                                                        #
        #   - Use cv2.imread() to read image using image path in dataset                               #
        #   - Use torch.nn.functional.one_hot to transform labels inside into binary one-hot tensor    #                                                               # 
        ################################################################################################
        
        return x, y

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - For each iteration, yield a batch of data in torch.Tensor form   #
        #   (Modify batch_to_tensor() method                                   #                             
        ########################################################################
        if self.shuffle:
            index_list = np.random.permutation(len(self.dataset))

        else:
            index_list = range(len(self.dataset))
        index_iterator = iter(index_list)  # define indices as iterator

        batch = []
        for index in index_iterator:  # iterate over indices using the iterator
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield ???  # use yield keyword to define a iterable generator
                batch = []

        #last batch
        num_complete_batch = int(len(self.dataset) / self.batch_size)
        if len(self.dataset) % self.batch_size != 0:
            if self.drop_last is False:
                batch = [self.dataset[i]
                         for i in iter(index_list[num_complete_batch * self.batch_size: len(self.dataset)])]
                yield ???

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset  #
        ########################################################################
        length = ???
        if len(self.dataset) % self.batch_size != 0:
            if not self.drop_last:
                length = ???
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length

        
class Cifar10Classifier(torch.nn.Module):
    def __init__(self):
        super(Cifar10Classifier, self).__init__()
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.dataloader = None
        self.net = None
        self.loss_fn = None
        self.optimizer = None

    def make_data(self):
        self.dataset = Cifar10Dataset(???)
        
    def split_data(self, split_ratio: float):
        """
        - split self.dataset into self.train_dataset and self.test_dataset 
        - See https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
        """
        self.train_dataset, self.test_dataset = ???

    def create_dataloader(self):
        self.train_loader = Cifar10Loader(self.train_dataset, batch_size=16, shuffle=True, drop_last=False)
        self.test_loader = Cifar10Loader(self.test_dataset, batch_size=16, shuffle=True, drop_last=False)
        print(len(self.train_loader))




classifier = Cifar10Classifier()

classifier.make_data()
classifier.split_data()
classifier.create_dataloader()
