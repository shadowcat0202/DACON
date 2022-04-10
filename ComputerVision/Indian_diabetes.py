import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy
import tensorflow as tf
import os


from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# 모든 영역에서 돌리고 싶을때는 아래와 같이
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)

print("hello")
numpy.random.seed(3)
tf.random.set_seed(3)

df = pd.read_csv("./database/pima-indians-diabetes.csv")
dataset = df.to_numpy()
X = dataset[:, : 8]
Y = dataset[:, 8]

model = Sequential()
# Dense(8, input_dim = 4, kernel_initializer = 'uniform', activation = 'relu')
# Dense의 주요 인자들은 아래와 같습니다.
#
# 첫번째 인자(units): 출력 뉴런의 수를 설정합니다.
# input_dim : 입력 뉴련의 수를 설정합니다.
# kernel_initializer : 가중치를 초기화하는 방법을 설정합니다.
#   uniform : 균일 분포
#   normal : 가우시안 분포
# activation : 활성화함수를 설정합니다.
#   linear : 디폴트 값으로 입력값과 가중치로 계산된 결과 값이 그대로 출력으로 나옵니다
#   sigmoid : 시그모이드 함수로 이진분류에서 출력층에 주로 쓰입니다
#   softmax : 소프드맥스 함수로 다중클래스 분류문제에서 출력층에 주로 쓰입니다.
#   relu: Rectified Linear Unit 함수로 은닉층에서 주로 쓰입니다.
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

print("CPU를 사용한 학습")
with tf.device("/device:CPU:0"):
    start = time.time()
    model.fit(X, Y, batch_size=5, epochs=200)
    cpu_time = time.time() - start

print("GPU를 사용한 학습")
with tf.device("/device:GPU:0"):
    start = time.time()
    model.fit(X, Y, batch_size=5, epochs=200)
    gpu_time = time.time() - start

print(f"cpu:{cpu_time}, gpu:{gpu_time}")
print("\n Accuracy: %.4f" % (model.evaluate(X,Y)[1] * 100))
