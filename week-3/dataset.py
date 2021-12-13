import os
from torch.utils.data import Dataset
import cv2

class Cifar10Dataset(Dataset):
    def __init__(self, path_to_folder):
        self.path_to_folder = path_to_folder
        my_dict = {'path': [], 'label': []}
        for folder_name in os.listdir(path_to_folder):
            for file_name in os.listdir(os.path.join(path_to_folder, folder_name)):
                my_dict['path'].append(os.path.join(path_to_folder, folder_name, file_name))
                my_dict['label'].append(folder_name)
        self.dataset = my_dict
    def __len__(self):
        return len(self.dataset['path'])

    def __getitem__(self, index):
        # label_dict = {'bird': 0, 'car': 1, 'cat': 2, 'deer': 3, 'dog': 4, 'frog': 5, 'horse': 6, 'plane': 7, 'ship': 8, 'truck': 9}
        i = 0
        label_dict = {}
        for folder_name in os.listdir(self.path_to_folder):
            label_dict[folder_name] = i
            i = i + 1
        x = self.dataset['path'][index]
        y = label_dict[self.dataset['label'][index]]
        return x, y



