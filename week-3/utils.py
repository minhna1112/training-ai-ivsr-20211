import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def preprocess_image(image: np.ndarray)->np.ndarray:
    """
    Args: image: input image to be preprocessed
    return: out: preprocessed image
    """
    out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out =  out / 255.0
    return out

def torch_preprocess(img: np.ndarray)->torch.Tensor:
    return torch.tensor(img, dtype=torch.float32).permute(dims=[2,0,1]).unsqueeze(dim=0)

def torch_postprocess(tensor: torch.Tensor)->np.ndarray:
    return torch.squeeze(tensor, dim=0).permute(dims=[1,2,0]).detach().numpy()

def scatter_3d(x: np.ndarray, y:np.ndarray, z:np.ndarray)->None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = 100


    ax.scatter(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
