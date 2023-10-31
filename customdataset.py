import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class CustomMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.labels = data['label'].values.astype(int)
        self.images = data.drop('label', axis=1).values.astype(float)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype(np.float32)  
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label