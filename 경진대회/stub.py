import warnings

from cv2 import waitKey

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
import matplotlib.pyplot as plt


def img_load(path):  # 이미지 로드(가져오기?)
    """
    cv2.imread(fileName[, cv2.IMREAD_COLOR | ,cv2.IMREAD_GRAYSCALE |, cv2.IMREAD_UNCHANGED]])
    cv2.IMREAD_COLOR      : 이미지 파일을 Color로 읽어들입니다. 투명한 부분은 무시되며, Default값입니다.
    cv2.IMREAD_GRAYSCALE  : 이미지를 Grayscale로 읽어 들입니다. 실제 이미지 처리시 중간단계로 많이 사용합니다.
    cv2.IMREAD_UNCHANGED  : 이미지파일을 alpha channel까지 포함하여 읽어 들입니다.
    """

    # print(cv2.imread(path)[0, 0])
    img = cv2.imread(path)[:, :, ::-1]  # 이건 뭔데 배열 인자 많음? 이미지가 어케 나오길래 슬라이싱이 뭐이리 많음?(BGR)역겹네
    # print(img.shape)
    # print(img[0][0])
    # # 이미지 크기를 512 * 512로 맞춰준다?(이미지를 특정 크기로 맞춰주어야 학습이 가능한 경우가 있기 때문에 잘 보자(원영씨 tip))
    # plt.subplot(1, 2, 1)
    # plt.title("befor img")
    # plt.imshow(img)

    img = cv2.resize(img, (416, 416))
    # plt.subplot(1, 2, 2)
    # plt.title("after img")
    # plt.imshow(img)
    #
    # plt.show()
    # waitKey(0)
    return img

train_y = pd.read_csv("./open/train_df.csv")
label_count = train_y[["class", "label"]].groupby('label').count().rename(columns={"class":"count"})
print(label_count)






train_png_name = sorted(glob("./open/stub/train/*.png"))  # 이미지 파일 이름 전부 다 가져오기 <clss: list>
test_png_name = sorted(glob("./open/stub/test/*.png"))  # 이미지 파일 이름 전부 다 가져오기 <clss: list>

print(train_png_name)
train_imgs = [img_load(im) for im in tqdm(train_png_name)]  # 해당 이름에 맞춰 이미지 로드 리스트 저장
test_imgs = [img_load(im) for im in tqdm(test_png_name)]


# 왜 커스텀시킴? Pytorch에서 DataLoader를 사용하기 위해 데이터를 개발자맘대로
# 사용자 정의 Dataset 클래스는 반드시 3개 함수를 구현해야 합니다: __init__, __len__, and __getitem__.
class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode = mode

    def __len__(self):  #데이터셋의 샘플 개수를 반환
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        if self.mode == 'train':
            augmentation = random.randint(0, 2) #왜 나누는거
            # print(f"wow:{augmentation}")
            if augmentation == 1:
                img = img[::-1].copy()
            elif augmentation == 2:
                img = img[:, ::-1].copy()
            # print(f"befor tensor\n{type(img)}\n{img}")
        img = transforms.ToTensor()(img)
        # print(f"img:{type(img)}\n{img}")
        if self.mode == 'test':
            pass
        label = self.labels[idx]
        return img, label


train_y = pd.read_csv("./open/train_df.csv")  # label을 적어놓은 데이터 파일 가져오기(지도학습이닷)
# print(train_y.head()) #train_df 보고 싶다면
train_labels = train_y["label"]  # index도 있기 때문에 lablel columns만 가져온다

# print(train_lables)

label_unique = sorted(np.unique(train_labels))  # label에서 종류별로 하나씩만 뽑아내기
# print(len(label_unique))  #88개나 있다고?!(엄청 많네)
label_unique = {key: value for key, value in
                zip(label_unique, range(len(label_unique)))}  # 라벨을 key:value 매칭 숫자로(0~88까지)

train_labels = [label_unique[k] for k in train_labels]  # 모든 lable들을 key에 맞춰 value 매핑 리스트 작성

# print(train_labels)


batch_size = 32
epochs = 25

# Train
train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), mode='train')
print(train_dataset[0])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# print(type(train_loader))


# Test
# test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"] * len(test_imgs)), mode='test')
# test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
