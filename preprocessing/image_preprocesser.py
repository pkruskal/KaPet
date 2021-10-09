from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random

transform = T.Compose([
        T.ToPILImage(), # converting our incoming image to PIL format
        T.Resize(image_size), # resizing it to our defined image_size,
        T.ToTensor() # converting to a tensor.
])

class PetPicDataset(Dataset):
    def __init__(self, image_data, labels):
        'Initialization'
        self.X = image_data
        self.y = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        """
        Required Dataset function: Dataloaders will fill out the index parameter for us.
        transforms image data for processing
        :param index:
        :return:
        """
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        return X


## dataloader
batch_size = 64
transformed_dataset = PetPicDataset(ims=X_train)
train_dl = DataLoader(transformed_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)