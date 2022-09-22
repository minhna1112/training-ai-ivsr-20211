import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import *

from layers import GaussianConvolutionLayer, LaplacianConvolutionalLayer 

GAUSSIAN_KERNEL = 1/9 * np.ones((3,3))
plt.ion()

class EdgeExtractor(torch.nn.Module):
    def __init__(self):
        
        """
        Initialize function (Constructor)
        + Intialize a smoothing layer from a predefined Gaussian kernel
        + Initialize a Laplacian convolutional layer (Your implementation in layers.py) 
        """
        super().__init__()
        self.smoothing_layer = GaussianConvolutionLayer(GAUSSIAN_KERNEL)
        self.laplacian_layer = LaplacianConvolutionalLayer()

    def forward(self, x):
        """
        x: input tensor
        Perform forward pass using 2 layers:
        + Smoothen input by using a gaussian forward pass
        + Output of Gaussian layer will be fed into Laplacian layer for edge extraction
        + Output of Laplacian layer are then binarized with threshold of 0.1 to create black-and-white-only tensor
        """
        out = self.smoothing_layer(x)
        out = self.laplacian_layer(out)
        out = out.clamp(min=0)
        out[????] = 1
        out[????] = 0 

        return out

if __name__=="__main__":
    bears = cv2.imread('./bears.jpg')
    plt.figure(0)
    plt.title('Original Image')
    plt.imshow(bears, vmax=255)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    #Preprocess image to feed into the edge extractor
    input_tensor = torch_preprocess(preprocess_image(bears))
    #Call an instance of the edge extractor
    edge_extractor = EdgeExtractor()
    #Perform forward pass
    edges = edge_extractor(input_tensor)
    #Postprocess results for visualization
    edges_im = post_process(edges)
    plt.figure(1)
    plt.title('Extracted edges')
    for i in range(3):
        plt.subplot(1,3, i+1)
        plt.imshow(edges_im[:, :, 2], vmax=1.0, cmap='gray')
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    plt.figure(1)
    plt.title('Extracted Edges from Blue channel')
    plt.imshow(post_process(edges)[:, :, 2], vmax=1.0, cmap='gray')
   