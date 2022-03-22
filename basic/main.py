import pandas as pd
import numpy as np

import Data_preprocessing
from Model import Desison_Tree
import CSV_process
import Tuning


if __name__ == "__main__":
    # 파일이 있으면 덮어쓰기(False = 있으면 다운로드 안하고 True = 있으면 새로 쓰기)
    # download_csv("https://bit.ly/3gLj0Q6", False)

    train = CSV_process.read_csv("train")
    test = CSV_process.read_csv("test")

    # 보간법
    train = Data_preprocessing.Interpolate(train)
    test = Data_preprocessing.Interpolate(test)
    del Data_preprocessing

    # model 설정 => 무슨 모델로 학습을 할지에 대한 생각 필요
    model = Desison_Tree.RandomForestRegressor()  # MSE 설정됨

    X_train = train.drop(["count"], axis=1)
    Y_train = train["count"]

    model.fit(X_train, Y_train)
