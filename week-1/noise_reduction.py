import torch
import torch.nn.functional as F
from scipy import ndimage as nd
from scipy import signal as sig
import matplotlib.pyplot as plt
import cv2
import numpy as np

import time
from utils import *

GAUSSIAN_KERNEL = 1/9 * np.ones((3,3))

class GaussianConvolutionLayer(torch.nn.Module):
    def __init__(self, weights_array: np.ndarray):
        super(GaussianConvolutionLayer, self).__init__()
        self.gaussian_tensor = torch.tensor(weights_array, dtype=torch.float32).reshape(1,1,3,3)

    def forward(self, x):
        r, g, b = torch.split(x, split_size_or_sections=[1,1,1], dim=1)
        red_out = F.conv2d(r, self.gaussian_tensor, padding=1)
        green_out = F.conv2d(g, self.gaussian_tensor, padding=1)
        blue_out = F.conv2d(b, self.gaussian_tensor, padding=1)

        out = torch.cat([red_out, green_out, blue_out], dim=1)

        return out


def plot_hist(img: np.ndarray, fig_num:int)->None:
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    
    red_hist = np.histogram(red, bins=50, range=(0,1))
    green_hist = np.histogram(green, bins=50, range=(0,1))
    blue_hist = np.histogram(blue, bins=50, range=(0,1))

    red_bins = red_hist[1]
    central_bins = (red_bins[1:] + red_bins[:-1]) / 2.

    plt.figure(fig_num)
    plt.title('3 color channels')
    plt.plot(central_bins, blue_hist[0], label='blue')
    plt.plot(central_bins, green_hist[0], label='green')
    plt.plot(central_bins, red_hist[0], label='red')
    plt.grid()
    plt.legend()

def add_noise(img: np.ndarray)->np.ndarray:
    """
    Args: img: input image 
    Return: noisy image
    """
    out = img + np.random.standard_normal()
    out[out<0]=0
    out[out>1]=1
    return out

def denoise_separated(img, kernel):
    bear_denoised_separated = np.zeros(shape=bear_noisy.shape)
    for i in range(3):
        bear_denoised_separated[:, :, i] = nd.convolve(bear_noisy[:,:, i], GAUSSIAN_KERNEL, mode='constant')

    return bear_denoised_separated

def torch_preprocess(img: np.ndarray)->torch.Tensor:
    return torch.Tensor(img.transpose((2,0,1))).unsqueeze(0)

def post_process(result: torch.Tensor)->np.ndarray:
    out = result.numpy().squeeze().transpose((1,2,0))
    return out

def denoise_torch(img: np.ndarray, filter: np.ndarray)->np.ndarray:
    gaussian_tensor = torch.tensor(filter, dtype=torch.float32).reshape(1,1,3,3)

    input_tensor = torch_preprocess(img)
    
    [red_tensor, green_tensor, blue_tensor] = torch.split(input_tensor, split_size_or_sections=[1,1,1], dim=1)

    red_out = F.conv2d(red_tensor, gaussian_tensor, padding=1)
    green_out = F.conv2d(green_tensor, gaussian_tensor, padding=1)
    blue_out = F.conv2d(blue_tensor, gaussian_tensor, padding=1)

    out = torch.cat([red_out, green_out, blue_out], dim=1)
    out = post_process(out)

    return out

def denoise_torch_module(img: np.ndarray, layer: GaussianConvolutionLayer)->np.ndarray:
    return post_process(layer(torch_preprocess(img)))

def main():
    bears = cv2.imread('./bears.jpg')
    bear_input = cv2.cvtColor(bears, cv2.COLOR_BGR2RGB)
    bear_input = bear_input / 255.0

    plt.figure(0)
    plt.imshow(bear_input)
    plot_hist(bear_input, fig_num=1)

    bear_noisy = add_noise(bear_input)

    plt.figure(2)
    plt.imshow(bear_noisy)
    plot_hist(bear_noisy, fig_num=3)

    bear_denoised = nd.filters.gaussian_filter(bear_noisy ,sigma=(1,1, 0))
    plt.figure(4)
    plt.imshow(bear_denoised, vmax=1.0)
    plot_hist(bear_denoised, fig_num=5)

    bear_denoised_torch = denoise_torch(bear_noisy, gaussian_kernel)
    plot_hist(bear_denoised_torch, 6)
    plt.figure(7)
    plt.imshow(bear_denoised_torch, vmax=1.0)

    conv_g = GaussianConvolutionLayer(GAUSSIAN_KERNEL)
    bear_denoised_torch_module = denoise_torch_module(bear_noisy, conv_g)
    plot_hist(bear_denoised_torch, 8)
    plt.figure(9)
    plt.imshow(bear_denoised_torch, vmax=1.0)

