import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()

class FaceLandmarksDataset(Dataset):

    def __init__(self,csv_file,root_dir,transform=None):
        self.landemarks_frame=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform

    def __len__(self):
        return len(self.landemarks_frame)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item=item.tolist()

        img_name=os.path.join(self.root_dir,self.landemarks_frame.iloc[item,0])
        image=io.


def show_landmarks(image,landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0],landmarks[:,1],s=10,marker='.',c='g')
    plt.pause(0.001)

