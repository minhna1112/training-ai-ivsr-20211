import torch
import numpy as np

from torchvision.datasets import MNIST

import cv2
import matplotlib.pyplot as plt
import time
from PIL import Image
from tqdm import tqdm
from torchsummary import summary


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
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

    def batch_to_numpy(self, batch: []):
        x = []
        y = []
        for im, label in batch:
            x.append(np.array(im)/255.0)
            y.append(label)
        x= np.stack(x, axis=0)
        y = np.array(y)
        return x, y

    def batch_to_tensor(self, batch: []):
        features, labels = self.batch_to_numpy(batch)
        x, y = torch.tensor(features, dtype=torch.float32).unsqueeze(1), torch.tensor(labels)
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
        #   - Have a look at the "DataLoader" notebook first                   #
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
                yield self.batch_to_tensor(batch)  # use yield keyword to define a iterable generator
                batch = []

        num_complete_batch = int(len(self.dataset) / self.batch_size)
        if len(self.dataset) % self.batch_size != 0:
            if self.drop_last is False:
                batch = [self.dataset[i]
                         for i in iter(index_list[num_complete_batch * self.batch_size: len(self.dataset)])]
                yield self.batch_to_tensor(batch)

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
        num_complete_batch = int(len(self.dataset) / self.batch_size)
        length = num_complete_batch
        if len(self.dataset) % self.batch_size != 0:
            if not self.drop_last:
                length = num_complete_batch + 1
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length


class ClassifierNN(torch.nn.Module):
    def __init__(self):
        super(ClassifierNN, self).__init__()

        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=10, bias=True),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(-1, 4*4*64)
        x = self.mlp(x)
        return x

class HandWrittenDigitClassifier(torch.nn.Module):
    def __init__(self):
        super(HandWrittenDigitClassifier, self).__init__()
        self.dataset = None
        self.net = None
        self.optimizer = None
        self.loss_fn = None
        #self.dataloader = None
        self.use_gpu = True
        self.device = 0

    def make_data(self):
        self.train_dataset = MNIST(root='/media/data/teamAI', train=True, download=True)
        self.test_dataset = MNIST(root='/media/data/teamAI', train=False, download=True)

    def create_dataloader(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True, drop_last=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=True, drop_last=False)
        print(len(self.train_loader))

    def create_model(self):
        self.net = ClassifierNN()
        summary(self.net, input_size=[1,28,28])
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def visualize_random_sample(self):
        # Display image and label.
        train_features, train_labels = next(iter(self.train_loader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0].squeeze()
        label = train_labels[0]
        plt.imshow(img, cmap="gray")
        plt.show()
        print(f"Label: {label}")

    def visualize_random_batch(self):
        # Display image and label.
        for i, train_features, train_labels in enumerate(self.train_loader):
            img = train_features[i].squeeze()
            label = train_labels[i]
            plt.subplot(4,4, i+1)
            plt.imshow(img, cmap="gray")
            plt.title(f'{label}')
        plt.show()

    def forward(self, x):
        return self.net(x)

    def train_step(self, x, y_truth):
        y_head = self.forward(x)

        loss = self.loss_fn(y_head, y_truth)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def start_trainning(self, num_epochs=20):
        self.net.train()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-3)
        if self.use_gpu:
            self.net.to(self.device)

        self.train_loss_history = []
        start_time = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for train_features, train_labels in tqdm(self.train_loader):
                x, y = train_features, train_labels

                if self.use_gpu:
                    x = x.cuda()
                    y = y.cuda()
                #
                # print(x.size())
                # print(y.size())
                running_loss += self.train_step(x, y)

            running_loss /= len(self.train_loader)

            print("[Epoch %d ] loss: %.3f  %%" % (epoch+1, running_loss))
            self.train_loss_history.append(running_loss)

        print(f'Training finished with total time of: {time.time()-start_time} s')

    def save_model(self, save_path='classifier_mnist.pt'):
        torch.save(self.net, save_path)

    def load_model_for_inference(self, save_path='classifier_mnist.pt'):
        self.net = torch.load(save_path)
        self.net.eval()

    def visualize_random_predictions(self):
        test_features, test_labels = next(iter(self.test_loader))
        print(f"Feature batch shape: {test_features.size()}")
        print(f"Labels batch shape: {test_labels.size()}")

        if self.use_gpu:
            self.net.to(self.device)
            x, y = test_features, test_labels
            x = x.cuda()
            y = y.cuda()

        y_hat = self.net(x)



classifier = HandWrittenDigitClassifier()
classifier.make_data()
classifier.create_dataloader()
classifier.create_model()
classifier.start_trainning()
classifier.save_model()
