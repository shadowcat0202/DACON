from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder  # 문자열로 되어있는 클래스를 숫자 형태로 바꿔주는 작업(One-hot encoding)
from keras.utils import np_utils

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv("./database/Iris.csv")
print(df['Species'].unique())
# sns.pairplot(df, hue='Species')
# plt.show()

dataset = df.values
X = dataset[:, 1:5].astype(float)
y = dataset[:, 5]

# One-Hot encoding
# unique를 통해서 알수 있는 class 개수에 따라 list[1,2,3,...,n] 로 나타낸다
e = LabelEncoder()
e.fit(y)
y = e.transform(y)

# 활성화 함수를 사용해서 출려하기 위해서는 문자열이 아니라 0 or 1이 되어야 하기 때문에
# [0,0,1], [0,1,], [1,0,0]과 같은 형식으로 맞춰준다
y_encoder = tf.keras.utils.to_categorical(y)

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

"""
손실함수(loss=)의 종류

Mean_Squared_error
예측한 값과 실제 값 사이의 평균 제곱 오차를 정의한다. 공식이 매우 간단하며, 차가 커질수록 제곱 연산으로 인해서 값이 더욱 뚜렷해진다. 그리고 제곱으로 인해서 오차가 양수이든 음수이든 누적 값을 증가시킨다.

RMSE(Root Mean Squared Error)
MSE에 루트(√)를 씌운 것으로 MSE와 기본적으로 동일하다. MSE 값은 오류의 제곱을 구하기 때문에 실제 오류 평균보다 더 커지는 특성이 있어 MSE에 루트를 씌운 RMSE 은 값의 왜곡을 줄여준다.

binary_crossentropy
실제 레이블과 예측 레이블 간의 교차 엔트로피 손실을 계산한다. 레이블 클래스(0, 1로 가정)가 2개만 존재할 때
Binary Crossentropy를 사용하면 좋다. 

categorical_crossentropy
다중 분류 손실함수로 출력값이 one-hot encoding 된 결과로 나오고 실측 결과와의 비교시에도 실측 결과는 
one-hot encoding 형태로 구성된다.
예를 들면 출력 실측값이 아래와 같은 형태(one-hot encoding)로 만들어 줘야 하는 과정을 거쳐야 한다.
[[0 0 1]
 [0 1 0]
 [1 0 0]]  (배치 사이즈 3개인 경우)
네트웍 레이어 구성시 마지막에 Dense(3, activation='softmax') 로 3개의 클래스 각각 별로 positive 확률값이 나오게 된다.
[0.2, 0.3, 0.5]
위 네트웍 출력값과 실측값의 오차값을 계산한다.

sparse_categorical_crossentropy
'categorical_entropy'처럼 다중 분류 손실함수이지만, 샘플 값은 정수형 자료이다. 예를 들어, 샘플 값이 아래와 같은 형태일 수 있다. (배치 사이즈 3개)
[0, 1, 2] 
네트웍 구성은 동일하게 Dense(3, activation='softmax')로 하고 출력값도 3개가 나오게 된다.
즉, 샘플 값을 입력하는 부분에서 별도 원핫 인코딩을 하지 않고 정수값 그대로 줄 수 있다. 이런 자료를 사용할 때, 컴파일 단계에서 손실 함수만  'sparse_categorical_crossentropy'로 바꿔주면 된다.

그 외
mean_absolute_error / mean_absolute_percentage_error 
mean_squared_logarithmic_error / cosine_proximity
squared_hinge / hinge / categorical_hinge 
logcosh / kullback_leibler_divergence / poisson 
"""
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

with tf.device("/device:GPU:0"):
    model.fit(X, y_encoder, epochs=50, batch_size=1)

print(model.evaluate(X, y_encoder))
