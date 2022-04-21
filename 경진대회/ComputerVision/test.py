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
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda')

# train_png = sorted(glob("./open/train/*.png"))  # 학습 이미지 파일 이름 전부 다 가져오기 <clss: list>
# test_png = sorted(glob("./open/test/*.png"))  # 테스트 이미지 파일 이름 전부 다 가져오기 <clss: list>

train_y = pd.read_csv("./open/train_df.csv")
print(train_y)
print(train_y.info())

class_list = train_y["class"].unique()
print(class_list)

# print(type(train_png))    #자료형 보고 싶다면

def img_load(path):  # 이미지 로드(가져오기?)
    """
    cv2.imread(fileName[, cv2.IMREAD_COLOR | ,cv2.IMREAD_GRAYSCALE |, cv2.IMREAD_UNCHANGED]])
    cv2.IMREAD_COLOR      : 이미지 파일을 Color로 읽어들입니다. 투명한 부분은 무시되며, Default값입니다.
    cv2.IMREAD_GRAYSCALE  : 이미지를 Grayscale로 읽어 들입니다. 실제 이미지 처리시 중간단계로 많이 사용합니다.
    cv2.IMREAD_UNCHANGED  : 이미지파일을 alpha channel까지 포함하여 읽어 들입니다.
    """
    img = cv2.imread(path)[:, :, ::-1]  # [행, 렬, 인자 거꾸로] 왜? -> OpenCV에서는 BGR순서로 나오기 때문에 거꾸로 뒤집어 주어야 한다 점심 나갈거같다

    # plt.subplot(1, 2, 1)
    # plt.title("befor img")
    # plt.imshow(img)

    # 이미지 크기를 512 * 512로 맞춰준다?(이미지를 특정 크기로 맞춰 주어야 학습이 가능한 경우가 있기 때문에 잘 보자(원영씨 tip))
    img = cv2.resize(img, (416, 416))
    # plt.subplot(1, 2, 2)
    # plt.title("after img")
    # plt.imshow(img)
    #
    # plt.show()
    # waitKey(0)
    return img


# test_img = img_load(train_png[0])
# plt.imshow(test_img, cmap="Greys")
# plt.show()
# print()

# tqdm : 진행바 개꿀팁?
# train_imgs = [img_load(im) for im in tqdm(train_png)]  # 해당 이름에 맞춰 이미지 로드 리스트 저장
# test_imgs = [img_load(im) for im in tqdm(test_png)]

# train_y = pd.read_csv("./open/train_df.csv")  # label을 적어놓은 데이터 파일 가져오기(지도학습이닷)
## print(train_y.head()) #train_df 보고 싶다면
# train_labels = train_y["label"]  # index도 있기 때문에 lablel columns만 가져온다
