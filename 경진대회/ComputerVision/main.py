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
import matplotlib.pyplot as plt

from cv2 import waitKey

device = torch.device('cuda')

train_png = sorted(glob("./open/train/*.png"))  # 학습 이미지 파일 이름 전부 다 가져오기 <clss: list>
test_png = sorted(glob("./open/test/*.png"))  # 테스트 이미지 파일 이름 전부 다 가져오기 <clss: list>


# print(type(train_png))    #자료형 보고 싶다면

def img_load(path):  # 이미지 로드(가져오기?)
    """
    cv2.imread(fileName[, cv2.IMREAD_COLOR | ,cv2.IMREAD_GRAYSCALE |, cv2.IMREAD_UNCHANGED]])
    cv2.IMREAD_COLOR      : 이미지 파일을 Color로 읽어들입니다. 투명한 부분은 무시되며, Default값입니다.
    cv2.IMREAD_GRAYSCALE  : 이미지를 Grayscale로 읽어 들입니다. 실제 이미지 처리시 중간단계로 많이 사용합니다.
    cv2.IMREAD_UNCHANGED  : 이미지파일을 alpha channel까지 포함하여 읽어 들입니다.
    """
    img = cv2.imread(path)[:, :, ::-1]  # [행, 렬, 인자 거꾸로] 왜? -> OpenCV에서는 BGR순서로 나오기 때문에 거꾸로 뒤집어 주어야 한다 역겹다

    # plt.subplot(1, 2, 1)
    # plt.title("befor img")
    # plt.imshow(img)

    # 이미지 크기를 512 * 512로 맞춰준다?(이미지를 특정 크기로 맞춰주어야 학습이 가능한 경우가 있기 때문에 잘 보자(원영씨 tip))
    img = cv2.resize(img, (416, 416))
    # plt.subplot(1, 2, 2)
    # plt.title("after img")
    # plt.imshow(img)
    #
    # plt.show()
    # waitKey(0)
    return img


# tqdm : 진행바 개꿀팁?
train_imgs = [img_load(im) for im in tqdm(train_png)]  # 해당 이름에 맞춰 이미지 로드 리스트 저장
test_imgs = [img_load(im) for im in tqdm(test_png)]

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

# 왜 커스텀시킴?  토치 내부에 DataLoader메서드 안에는 Dataset메서드가 필요한데 그것을 커스텀으로 하려고
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
            augmentation = random.randint(0, 2)  # 왜 나누는거? = 데이터의 확장성
            if augmentation == 1:  # 셀?
                img = img[::-1].copy()
            elif augmentation == 2:  #
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
test_dataset = Custom_dataset(np.array(test_imgs), np.array(["tmp"] * len(test_imgs)), mode='test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# 모델 학습====================================================
def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score


model = Network().to(device)  # 모델 자체에 있는 .to 메서드를 사용가능

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

best = 0
for epoch in range(epochs):
    start = time.time()
    train_loss = 0
    train_pred = []
    train_y = []
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.long, device=device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() / len(train_loader)
        train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
        train_y += y.detach().cpu().numpy().tolist()

    train_f1 = score_function(train_y, train_pred)

    TIME = time.time() - start
    print(f'epoch : {epoch + 1}/{epochs}    time : {TIME:.0f}s/{TIME * (epochs - epoch - 1):.0f}s')
    print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')

# 추론===========================================================
model.eval()
f_pred = []

with torch.no_grad():
    for batch in (test_loader):
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        with torch.cuda.amp.autocast():
            pred = model(x)
        f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())

label_decoder = {val: key for key, val in label_unique.items()}
f_result = [label_decoder[result] for result in f_pred]

# 제출물 생성=============================================
submission = pd.read_csv("open/sample_submission.csv")
submission["label"] = f_result
print(submission)

submission.to_csv("baseline.csv", index=False)
