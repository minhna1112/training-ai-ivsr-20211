import torch

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
