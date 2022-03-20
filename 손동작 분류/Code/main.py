import pandas as pd
import numpy as np
import seaborn as sns  # 데이터 분포의 시각화

import warnings  # 경고 제어(

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt


# %matplotlib inline

def KDE():
    # 센서들 간의 상관계수를 직접 확인해보자
    # DataFrame.corr() -> 상관 행렬
    """
    heatmap = sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
    fig = plt.gcf()
    fig.set_size_inches(30, 30)

    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=20)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=20)

    plt.title("correlation btw features", fontsize=40)
    plt.show()
    """

    """
    #센서별 분포 그래프 확인하기

    plt.figure(figsize=(20, 60))

    for i in range(len(feature) - 1):
        plt.subplot(11, 3, i+1) #plt.subplt(row, col, index)    #여러개 그래프 겹치지 않고 그리기
        plt.title(feature[i])   #그래프마다 이름 지정
        plt.xlim(-170,170)  #x축 범위
        plt.ylim(0,0.1)     #y축 범위
        sns.distplot(data[feature[i]], color="magenta")

    plt.show()
    """

    # 커널 밀도 함수(KDE)
    plt.figure(figsize=(20, 60))
    for i in range(len(feature) - 1):
        plt.subplot(6, 6, i + 1)  # plt.subplt(row, col, index)    #여러개 그래프 겹치지 않고 그리기
        plt.title(feature[i])

        g = sns.kdeplot(data[feature[i]][data["target"] == 0], color="Red")
        g = sns.kdeplot(data[feature[i]][data["target"] == 1], color="Blue")
        g = sns.kdeplot(data[feature[i]][data["target"] == 2], color="green")
        g = sns.kdeplot(data[feature[i]][data["target"] == 3], color="yellow")

        g.set_xlabel(feature[i])
        g.set_ylabel("Freq")
        plt.xlim(-170, 170)
        plt.ylim(0, 0.2)
        g = g.legend(["target=0", "target=1", "target=2", "target=3"])

    plt.show()


# print(plt.style.avaiable)  #사용가능한 plot의 스타일
plt.style.use("ggplot")

# 파일 불러오기
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

# read_csv.head()    #상위 6개 출려
# read_scv.shape()   #row col 개수

# 중복된 데이터 확인
# train.duplicated().all()

# train에서 해당 컬럼을 잘라내 버린다
data = train.drop("id", axis=1)

# target에 해당하는 개수
# data['target'].value_counts()
# =>0~3 4가지의 손동작이 각각 몇개가 있는지 출력해줌

feature = data.columns  # 컬럼 가져오기

train_x = train.iloc[:, 1:-1]   #모든 row랑 columns[1:-1]까지의 데이터를 전부 슬라이싱한다
test_x = test.iloc[:, 1:]

