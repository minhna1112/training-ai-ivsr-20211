import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import *

from layers import GaussianConvolutionLayer, LaplacianConvolutionalLayer 

GAUSSIAN_KERNEL = 1/9 * np.ones((3,3))

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
        out = self.laplacian_kernel(out)
        out = out.clamp(min=0)
        out[out>0.1] = 1
        out[out <0.1] = 0 

        return out

if __name__=="__main__":
    bears = cv2.imread('./bears.jpg')
    plt.figure(0)
    plt.title('Original Image')
    plt.imshow(bears, vmax=255)

    #Preprocess image to feed into the edge extractor
    input_tensor = torch_preprocess(preprocess_image(bears))
    #Call an instance of the edge extractor
    edge_extractor = EdgeExtractor()
    #Perform forward pass
    edges = edge_extractor(input_tensor)
    #Postprocess results for visualization
    plt.figure(1)
    plt.title('Extracted Edges from Blue channel')
    plt.imshow(post_process(edges)[:, :, 2], vmax=1.0, cmap='gray')
    
