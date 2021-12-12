import torch
import torch.nn.functional as F
from scipy import ndimage as nd
from scipy import signal as sig
import cv2
import numpy as np

import time
from utils import *
from layers import GaussianConvolutionLayer

GAUSSIAN_KERNEL = 1/9 * np.ones((3, 3))

def add_noise(img: np.ndarray)->np.ndarray:
    """
    Args: img: input image 
    Return: noisy image
    thanhBear
    """
    out = img + np.random.standard_normal()
    out[out < 0] = 0
    out[out > 1] = 1
    return out

def denoise_separated(img, kernel):
    bear_denoised_separated = np.zeros(shape=bear_noisy.shape)
    for i in range(3):
        bear_denoised_separated[:, :, i] = nd.convolve(bear_noisy[:, :, i], kernel, mode='constant')

    return bear_denoised_separated

def denoise_torch(img: np.ndarray, filter: np.ndarray)->np.ndarray:
    gaussian_tensor = torch.tensor(filter, dtype=torch.float32).reshape(1, 1, 3, 3)
    input_tensor = torch_preprocess(img)
    
    [red_tensor, green_tensor, blue_tensor] = torch.split(input_tensor, split_size_or_sections=[1, 1, 1], dim=1)

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

    bear_denoised_separated = denoise_separated(bear_noisy, GAUSSIAN_KERNEL)
    plot_hist(bear_denoised_separated, 6)
    plt.figure(7)
    plt.imshow(bear_denoised_separated, vmax=1.0)

    bear_denoised_torch = denoise_torch(bear_noisy, GAUSSIAN_KERNEL)
    plot_hist(bear_denoised_torch, 8)
    plt.figure(9)
    plt.imshow(bear_denoised_torch, vmax=1.0)

    conv_g = GaussianConvolutionLayer(GAUSSIAN_KERNEL)
    bear_denoised_torch_module = denoise_torch_module(bear_noisy, conv_g)
    plot_hist(bear_denoised_torch, 10)
    plt.figure(11)
    plt.imshow(bear_denoised_torch, vmax=1.0)
