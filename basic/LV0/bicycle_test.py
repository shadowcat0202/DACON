import CSV_process
import Data_preprocessing
import pandas as pd
from Model import Desison_Tree

def start():
    #파일 읽어오기
    train = CSV_process.read_csv("train")   #훈련 데이터
    test = CSV_process.read_csv("test")     #훈련된 모델을 테스트 할 데이터
    train.head()
    test.head()

    train = Data_preprocessing.dropNaN(train)   #전처리 NaN row drop
    test = Data_preprocessing.fillNaN(test)     #전처리 NaN data to 0
    print(train.isnull().sum())
    
    # 모델을 학습 시킬 데이터 할당
    X_train = train.drop(['count'], axis=1) #결과를 제외(["count"])한 데이터 값
    Y_train = train["count"]    #결과값

    # 의사결정트회귀로 학습 
    model = Desison_Tree.DecisionTreeRegressor()    #모델을 의사결정회귀로 정하고
    model.fit(X_train, Y_train) # X 데이터와 Y 데이터 입력 & 학습 시작
    # 이 시점에서 준비한 데이터에서의 모델 학습은 끝난 상태
    
    pred = model.predict(test)  # 준비한 테스트 데이터 모델을 적용시키면 결과값을 저장

    CSV_process.predict_to_csv(pred, "output")    #예측값을 결과 프레임에 맞춰 .csv로 저장
    
    
    

