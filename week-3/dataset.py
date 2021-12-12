import os
from torch.utils.data import Dataset
import cv2
import numpy as np

class Cifar10(Dataset):
    def __init__(self, path_to_folder):
        my_dict = dict()
        for folder_name in os.listdir(path_to_folder):
            img = []
            for file_name in os.listdir(os.path.join(path_to_folder, folder_name)):
                img.append(file_name)
            my_dict[folder_name] = img
        self.dataset = my_dict
    def __len__(self):
        length = 0
        for label in self.dataset:
            length = length + len(self.dataset[label])
        return length

    def __getitem__(self, index):
        dir = 'cifar10'
        file_path = []
        label_int = 0
        count = 0
        for label in self.dataset:
            check = False
            for file_name in self.dataset[label]:
                file_path.append(os.path.join(dir, label, file_name))
                count = count + 1
                if count == index:
                    check = True
                    break
            if check:
                break
            label_int = label_int + 1
        image = file_path[index - 1]
        #image = cv2.imread(image)
        return image, label_int

path = 'cifar10'
cf = Cifar10(path)
print(cf.__getitem__(1))



