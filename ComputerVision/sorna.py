import pandas as pd
import numpy
import tensorflow as tf


df = pd.read_csv("./database/sonar.csv", header=None)
# print(df.info())
# print(df.iloc[:,-1].unique())
# seed값 설정
seed = 3
numpy.random.seed(0)
tf.random.set_seed(seed)

df = df.values
X = df[:, :60].astype(float)
y = df[:, 60]

from sklearn.preprocessing import LabelEncoder
e = LabelEncoder()
e.fit(y)
y = e.transform(y)

# 과적합(overfitting)이 발생하는것을 방지하기 위해서
# 학습 데이터를 train / validation으로 구분하여 train으로 학습 후 validation으로 중간중간 모델을 평가 해준다
# 검증시 예측율이나 오차율이 떨어지는 현상이 감지되면 학습을 종료한다
# 추가로 검증 방법으로는 K-Fold 교차 검증 방식이 존재한다
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

## Sequential
# from keras.layers import Dense
# from keras.models import Sequential
#
# model = Sequential()
# model.add(Dense(24, input_dim=60, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# with tf.device("/device:GPU:0"):
#     model.fit(X, y, epochs=200, batch_size=5)


## Functional
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 각 층 셋팅
# 입력층 (shape=(열,행)) 행:데이터는 계속 추가할수 있으니까 필요한건 column vector가 몇개 있는지가 중요하다
inputs = Input(shape=(len(X[0]),))
"""
from tensorflow import TensorShape
inputs = Input()
inputs.shape = TensorShape([None, 60])
"""
hidden = Dense(24, activation='relu')(inputs)
hidden = Dense(10, activation='relu')(inputs)
output = Dense(1, activation='sigmoid')(hidden)
f_logistic_model = Model(inputs=inputs, outputs=output)

# 모델 컴파일
f_logistic_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
# f_logistic_model.optimizer.lr = 0.001

# 학습
with tf.device("/device:GPU:0"):
    f_logistic_model.fit(x=X, y=y, epochs=130, batch_size=5)
    # verbose=False 하면 막대바 안보임
# 모델 저장
from keras.models import load_model
f_logistic_model.save('my_model.h5')
del f_logistic_model    # 테스트를 위해 메모리 내의 모델을 삭제
f_logistic_model = load_model('my_model.h5')

# model.evaluate() = [오차율?, 예측율]
score = f_logistic_model.evaluate(X_test, y_test, verbose=1)
print(f"정답률={score[1]*100} loss={score[0]}")




# print(len(f_logistic_model.get_weights()))
# for i in range(len(f_logistic_model.get_weights())):
#     filename = "weights[" + str(i) + "].csv"
#     numpy.savetxt(filename, f_logistic_model.get_weights()[i], fmt='%s', delimiter=',')
# numpy.savetxt("./weight.csv",f_logistic_model.get_weights(), fmt='%s', delimiter=',')
