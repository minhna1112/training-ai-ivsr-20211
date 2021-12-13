import torch
import torch.nn as nn
from torchsummary import summary
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import Cifar10Dataset
from dataloader import Cifar10Loader
from model import VGG16

class Cifar10Classifier(nn.Module):
    def __init__(self):
        super(Cifar10Classifier, self).__init__()
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.net = None
        self.loss_fn = None
        self.optimizer = None
    def make_data(self):
        self.dataset = Cifar10Dataset('cifar10')
    def split_data(self, split_ratio: float):
        train_size = int(split_ratio * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        print('Len train data: ', len(self.train_dataset))
        print('Len test data: ', len(self.test_dataset))
    def create_dataloader(self):
        self.train_dataloader = Cifar10Loader(self.train_dataset, batch_size=16, shuffle=True, drop_last=False)
        self.test_dataloader = Cifar10Loader(self.test_dataset, batch_size=16, shuffle=True, drop_last=False)
        print('Length of Train Dataloader: ', len(self.train_dataloader))
    def create_model(self):
        self.net = VGG16(in_channels=3, num_classes=10)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    def forward(self, x):
        return self.net(x)
    def train_step(self, image, label):
        y = self.forward(image)
        loss = self.loss_fn(y, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    def start_trainning(self, num_epochs=5):
        self.net.train()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-3)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device", device)
        self.net.to(device)
        self.train_loss_history = []
        start_time = time.time()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for train_features, train_labels in tqdm(self.train_dataloader):
                x, y = train_features, train_labels
                x = x.to(device)
                y = y.to(device)
                running_loss += self.train_step(x, y)
            running_loss /= len(self.train_dataloader)
            print("[Epoch %d] loss: %.3f %%" % (epoch + 1, running_loss))
            self.train_loss_history.append(running_loss)
        print(f'Training finished with total time of: {time.time()-start_time}s')
    def save_model(self, save_path='cifar10_classifier.pt'):
        torch.save(self.net, save_path)
    def load_model(self, path='cifar10_classifier.pt'):
        self.net = torch.load(path)
        self.net.eval()
    def visualize_random_predictions(self):
        test_features, test_labels = next(iter(self.test_dataloader))
        print(f"Feature batch shape: {test_features.size()}")
        print(f"Labels batch shape: {test_labels.size()}")
        device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        x, y = test_features, test_labels
        x = x.to(device)
        y = y.to(device)

        y_hat = self.net(x)
        for i, input_tensor in enumerate(test_features):
            im = input_tensor.permute(1, 2, 0).detach().numpy()
            out = np.argmax(y_hat[i].detach().cpu().numpy())
            plt.subplot(4, 4, i + 1)
            plt.imshow(im, vmax=1.0)
            plt.title(f'{out}')
        plt.show()
        plt.pause(100)

if __name__ == '__main__':
    classifier = Cifar10Classifier()
    classifier.make_data()
    classifier.split_data(0.8)
    classifier.create_dataloader()
    classifier.create_model()
    classifier.start_trainning()
    classifier.save_model()
    # classifier.load_model_for_inference()
    # classifier.visualize_random_predictions()