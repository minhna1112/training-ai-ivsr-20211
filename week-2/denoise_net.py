import torch
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt


from utils import *

class DenoiseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, groups=3),
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, groups=3),
            torch.nn.Sigmoid()
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-6)
        self.train_data = {}
        self.train_loss_history = []      
        self.device = torch.cuda.current_device()
        

    def forward(self, x):
        return self.net(x)


    def train_step(self, X_tensor, Y_tensor):

        Y_pred = self.forward(X_tensor)

        loss = self.loss_fn(torch.squeeze(Y_tensor), torch.squeeze(Y_pred))

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_epochs=20, use_gpu=True):
        if use_gpu:
            self.net.to(self.device)

        start_time = time.time()
        for epoch in range(num_epochs):

            x, y = self.train_loader()
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            
            running_loss = self.train_step(x, y)

            print("[Epoch %d ] loss: %.3f  %%" % (epoch+1, running_loss))
            self.train_loss_history.append(running_loss)
            running_loss = 0.0
        print(f'Training finished with total time of: {start_time-time.time()}ms')
    
    def make_data(self, img_folder):
        self.train_data['noisy'] = [os.path.join(img_folder+'_noisy',img_path) for img_path in os.listdir(img_folder+'_noisy')]
        self.train_data['ground_truth'] = [os.path.join(img_folder,img_path) for img_path in os.listdir(img_folder)]
    
    def train_loader(self, batch_size=1):
        i =0
        x = []
        y = []
        for (noisy_path, label_path) in zip(self.train_data['noisy'], self.train_data['ground_truth']):
            noisy_im = cv2.imread(noisy_path)
            groundtruth_im = cv2.imread(label_path)

            x.append(torch_preprocess(preprocess_image(noisy_im)))
            y.append(torch_preprocess(preprocess_image(groundtruth_im)))

            i+=1
            if i>batch_size-1:
                break

        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return x, y

if __name__=='__main__':
    denoiser = DenoiseModel()

    denoiser.make_data('./img')

    denoiser.train(60000)

    plt.figure(0)
    out=denoiser.forward(denoiser.train_loader()[0].cuda())
    plt.imshow(torch_postprocess(out.cpu()), vmax=1.0)

    out.min()