import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

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

def torch_preprocess(img: np.ndarray)->torch.Tensor:
    return torch.Tensor(img.transpose((2,0,1))).unsqueeze(0)

def post_process(result: torch.Tensor)->np.ndarray:
    out = result.numpy().squeeze().transpose((1,2,0))
    return out

def preprocess_image(image: np.ndarray)->np.ndarray:
    """
    Args: image: input image to be preprocessed
    return: out: preprocessed image
    """
    out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out =  out / 255.0
    return out