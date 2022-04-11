from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder  # 문자열로 되어있는 클래스를 숫자 형태로 바꿔주는 작업(One-hot encoding)
from keras.utils import np_utils

# import seaborn as sns
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


model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

with tf.device("/device:GPU:0"):
    model.fit(X, y_encoder, epochs=50, batch_size=1)

print(model.evaluate(X, y_encoder))
