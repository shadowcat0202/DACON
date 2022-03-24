from sklearn.metrics import accuracy_score
import CSV_process
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def start():
    # ===========================EDA작업# ===========================
    train = CSV_process.read_csv("wine_train")
    test = CSV_process.read_csv("wine_test")

    # ===========================결측치 유무 확인===========================
    # print(train.isnull().sum())

    # ===========================수치 데이터 특성 보기===========================
    # 타깃 변수 분포 시각화 seaborn distplot()
    traindata = train.copy()
    sns.displot(traindata["quality"], kde=False, bins=10)
    plt.axis([0, 10, 0, 2500])  # [x축 최소, 최대, y축 최소, 최대]
    plt.title("wine quality")
    plt.show()

    # ===========================Matplotlib 선 그래프 plot() 히스토그램 hist() 그리기===========================
    # x_values = [0,1,2,3,4]
    # y_values = [0,1,4,9,16]
    # plt.plot(x_values, y_values)
    # plt.show()
    #
    # a = [1,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,7]
    # plt.hist(a)
    # plt.show()

    # 전처리==========================================================================
    # ===========================이상치 탐지 seaborn_boxplot()===========================
    # 모델의 성능을 떨어뜨리는 불필요한 요소 반드시 제거해야함
    # 하지만 이방법은 소수의 데이터가 평균으로부터 눈에 띄게 떨어진 경우만 가능하다는 한계 존재
    # := 데이터들이 특정 구간에 과하게 밀집되어 있기 떄문에 이것을 제외한 소수의 부분은 학습 성능을 떨어뜨려서 제거필요
    # 사분위(Qunatile) 개념으로부터 출발
    sns.boxplot(data=train["fixed acidity"])
    plt.title("fixed acidity")
    plt.show()

    # ===========================이상치 제거 IQR===========================
    # 4등분으로 나눈다음
    quantile_25 = np.quantile(train['fixed acidity'], 0.25)
    quantile_50 = np.quantile(train['fixed acidity'], 0.50)
    quantile_75 = np.quantile(train['fixed acidity'], 0.75)

    IQR = quantile_75 - quantile_25  # 이상치(IQR)의 정의 75% 지점 - 25%지점

    minimum = quantile_25 - 1.5 * IQR
    maximum = quantile_75 - 1.5 * IQR

    # 고정산도값이 25% 이상 75% 이하인 값만 저장
    train2 = train[(minimum <= train["fixed acidity"]) & (train["fixed acidity"] <= maximum)]

    # print("이상치 제거", train.shape[0], train2.shape[0], (train.shape[0] - train2.shape[0]))
    # print(train.describe())

    # 그래프
    # plt.subplot(1, 2, 1)
    # sns.distplot(train['fixed acidity'])
    # plt.title("fixed acidity")

    # ===========================수치형 데이터 정규화 MinMaxScaler()==================================
    scaler = MinMaxScaler()

    scaler.fit(train[["fixed acidity"]])

    # "scaler"를 통해 train과 test의 "fixed acidity"를 바꾸어 "Scaled fixed acidity"라는 column에 저장
    train['Scaled fixed acidity'] = scaler.transform(train[['fixed acidity']])
    test['Scaled fixed acidity'] = scaler.transform(test[['fixed acidity']])

    # 그래프
    # plt.subplot(1, 2, 2)
    # sns.distplot(train["Scaled fixed acidity"])
    # plt.title("Scaled fixed acidity")
    # plt.show()

    # =======================================원-핫 인코딩 OneHotEncoder()=======================================
    # 문자로 되어있는 featuer들을 컴퓨터가 읽어서 학습 할 수 있도록 "인코딩"을 해주어야 한다
    # 인코딩 방법 중 하나인 원-핫 인코딩이 존재한다
    encoder = OneHotEncoder()

    # encoder.fit(train[["type"]])  #인코더를 사용해 train의 type feature를 학습
    #
    # train_onehot = encoder.transform(train[["type"]])
    # test_onehot = encoder.transform(test[["type"]])
    #
    # train_onehot = train_onehot.toarray()
    # test_onehot = test_onehot.toarray()
    #
    # train_onehot = pd.DataFrame(train_onehot)
    # test_onehot = pd.DataFrame(test_onehot)
    #
    # train_onehot.columns = encoder.get_feature_names()
    # test_onehot.columns = encoder.get_feature_names()

    # 윗내용 간략화(알아먹기 어렵네)
    train_onehot = pd.DataFrame(encoder.fit_transform(train[['type']]).toarray(), columns=encoder.get_feature_names())
    test_onehot = pd.DataFrame(encoder.fit_transform(test[['type']]).toarray(), columns=encoder.get_feature_names())

    # 병합 and 기존 type 드랍
    train = pd.concat([train, train_onehot], axis=1).drop(columns=["type"])
    test = pd.concat([test, test_onehot], axis=1).drop(columns=["type"])
    # 그래프
    # =============================모델정의 RandomForestClassifier================================
    random_forest = RandomForestClassifier()
    print(random_forest)
    X_train = train.drop(columns=["quality"])
    Y_train = train["quality"]

    random_forest.fit(X_train, Y_train)

    # ================================= 교차검증정의 K-Fold======================================
    # 훈련데이터 set에서 정한 크기 만큼 겹치지 않도록 각 학습마다 검증을 하는 데이터로 사용한다
    # 실습 K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # K-Fold를 이용해서 train과 validData로 나눈다

    for train_idx, valid_idx in kf.split(train):
        train_data = train.iloc[train_idx]
        valid_data = train.iloc[valid_idx]

    i = 1
    for train_idx, valid_idx in kf.split(train):
        plt.scatter(valid_idx, [i for x in range(len(valid_idx))], alpha=0.1)
        i += 1
    plt.show()

    kf = KFold(n_splits=5, shuffle=False, random_state=0)
    train_idx_store = []
    valid_idx_store = []

    # Bayesian Optimization
    # 그리드, 랜덤 서치 vs Bayesian Optimization
    # Bayesian Optimization 실습


def test_stub():
    train = CSV_process.read_csv('wine_train')
    test = CSV_process.read_csv('wine_test')
    # print(train[["fixed acidity"]].head(), end="\n============================\n")

    # Scailing
    scaler = MinMaxScaler()
    # MinMaxScaler()는 수치형 데이터 정규화
    # := 데이터들을 비율로 나눠 0~1 사이에 분포하도록 데이터를 정규화 시키는 작업
    scaler.fit(train[["fixed acidity"]])
    # print(train.head(), end="\n============================\n")
    train['Scaled fixed acidity'] = scaler.transform(train[['fixed acidity']])  # 해당 row에 맞춰 column을 추가
    test['Scaled fixed acidity'] = scaler.transform(test[['fixed acidity']])
    # print(train.head(), end="\n============================\n")

    encoder = OneHotEncoder()  # 인코딩 방식을 정한다

    # encoder.fit(train[['type']])     #"type"에 대해서
    # onehot = encoder.transform(train[["type"]])

    # onehot = onehot.toarray()

    # onehot = pd.DataFrame(onehot)
    # onehot.columns = encoder.get_feature_names()

    # train = pd.concat([train, onehot], axis=1)
    # train = train.drop(columns=['type'])

    train_onehot = pd.DataFrame(encoder.fit_transform(train[['type']]).toarray(), columns=encoder.get_feature_names())
    train = pd.concat([train, train_onehot], axis=1).drop(columns=["type"])

    test_onehot = pd.DataFrame(encoder.fit_transform(test[['type']]).toarray(), columns=encoder.get_feature_names())
    test = pd.concat([test, test_onehot], axis=1).drop(columns=["type"])

    # K-Fold := test 데이터에서 학습과 테스트 부분을 분리하여 검증?을 여러번 진행하여 그 결과값들에 대한 평균을 도출하는 기법
    train_X = train.drop(columns=['index', 'quality'])
    train_y = train['quality']

    # i = 1
    # for train_idx, valid_idx in kf.split(train):
    #     plt.scatter(valid_idx, [i for x in range(len(valid_idx))], alpha=0.1)
    #     i += 1
    # plt.show()
    X = train.drop(columns=['index', 'quality'])
    y = train['quality']

    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 여기서 train할 row들과 valid할 row들을 분리한다 반복문으로 split을 해서 보면 알수 있다

    model = RandomForestClassifier(random_state=0)
    valid_scores = []  # 검증할 부분을 따로 분리
    test_predictions = []  # test 내부에서 예측한 값

    for train_idx, valid_idx in kf.split(X, y):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]

        X_val = X.iloc[valid_idx]
        y_val = y.iloc[valid_idx]

        model.fit(X_tr, y_tr)   #각 fold를 진행하면서 학습
        # valid_prediction = model.predict(X_val) #valid값으로 예측
        # valid_scores.append(accuracy_score(y_val, valid_prediction))    #각 fold에 따라 성능 예측 점수 저장
        test_predictions.append(model.predict(test.drop(columns=['index'])))    #위의 두줄로 대충 어느정도 나올지 성능 예측을 해봤으니 실제 테스트 값으로 예측한 값들을 넣어보자

    test_predictions = pd.DataFrame(test_predictions)
    test_prediction = test_predictions.mode()
    test_prediction = test_prediction.values[0] #첫행을 최종 결과값으로 사용 왜?!

    sample_submission = pd.read_csv('./data_set/wine_submission.csv')
    sample_submission['quality'] = test_prediction
    sample_submission.to_csv('./data_set/submission_KFOLD.csv', index=False)


        
