import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time


device = torch.device('cuda')

train_png = sorted(glob('train/*.png'))
test_png = sorted(glob('test/*.png'))

train_y = pd.read_csv("open/train_df.csv")

train_labels = train_y["label"]

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]


def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (416, 416))
    return img


class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode = mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode == 'train':
            augmentation = random.randint(0, 2)
            if augmentation == 1:
                img = img[::-1].copy()
            elif augmentation == 2:
                img = img[:, ::-1].copy()
        img = transforms.ToTensor()(img)
        if self.mode == 'test':
            pass

        label = self.labels[idx]
        return img, label


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)
        return x


batch_size = 32
epochs = 25

# Train
train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), mode='train')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# Test
test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"]*len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)