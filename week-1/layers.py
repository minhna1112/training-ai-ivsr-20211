import torch
import numpy as np
import torch.nn.functional as F

class GaussianConvolutionLayer(torch.nn.Module):
    def __init__(self, weights_array: np.ndarray):
        super(GaussianConvolutionLayer, self).__init__()
        # Initialize a Gaussian filter from a Gaussian array
        self.gaussian_tensor = torch.tensor(weights_array, dtype=torch.float32).reshape(1,1,3,3)

    def forward(self, x):
        # Perfom convolution seperatedly w.r.t each color channel
        r, g, b = torch.split(x, split_size_or_sections=[????], dim=1)
        red_out = F.conv2d(r, self.gaussian_tensor, padding=1)
        green_out = F.conv2d(g, self.gaussian_tensor, padding=1)
        blue_out = F.conv2d(b, self.gaussian_tensor, padding=1)

        out = torch.cat([red_out, green_out, blue_out], dim=???)

        return out

        
class LaplacianConvolutionalLayer(torch.nn.Module):
    def __init__(self):
        super(LaplacianConvolutionalLayer, self).__init__()
        ##### YOUR CODE HERE #####
        # Initialize a Laplacian tensor (type: torch.Tensor) using filter matrix pre-defined in Figure 3.39 d, page 129, Gonzalez book
        self.laplacian_kernel = torch.tensor(np.array([??????]).reshape([1,1,3,3]), dtype=torch.float32).repeat(3,1,1,1)
        ##### END OF YOUR CODE #####
    def forward(self, x):
        #### YOUR CODE HERE
        #Perform forward pass using F.conv2d, you can perform convolution on each color channel separatedly, or try using "groups" parameters.
        out =  F.conv2d(x, self.laplacian_kernel, groups=3, padding=1)
        #####
        return out