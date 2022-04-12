import pandas as pd
import numpy
import tensorflow as tf

df_train = pd.read_csv("./database/train.csv")

seed = 3
numpy.random.seed(0)
tf.random.set_seed(seed)

df_train = df_train.drop(["Name", "PassengerId", "Ticket", "Fare", "Cabin"], axis=1)
df_train["Age"].fillna(df_train["Age"].mean(), inplace=True)
df_train["Embarked"] = df_train["Embarked"].fillna("S")  # 가장 많은 S로 통일
df_train["Sex"] = df_train["Sex"].map({"male": 0, "female": 1})
df_train["Embarked"] = df_train["Embarked"].map({"Q": 0, "C": 1, "S": 2})
family = []
for i in range(len(df_train)):
    if df_train.loc[i, "SibSp"] >= 4:
        family.append(2)
    elif df_train.loc[i, "SibSp"] >= 2:
        family.append(1)
    else:
        family.append(0)
df_train["Family"] = family
X = df_train.drop(["Survived"], axis=1)
y = df_train["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


## Functional
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 각 층 셋팅
# 입력층 (shape=(열,행)) 행:데이터는 계속 추가할수 있으니까 필요한건 column vector가 몇개 있는지가 중요하다

inputs = Input(shape=(7,))
"""
from tensorflow import TensorShape
inputs = Input()
inputs.shape = TensorShape([None, 60])
"""
hidden = Dense(24, activation='relu')(inputs)
hidden = Dense(10, activation='relu')(inputs)
output = Dense(1, activation='relu')(hidden)
f_logistic_model = Model(inputs=inputs, outputs=output)

# 모델 컴파일
f_logistic_model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
# f_logistic_model.optimizer.lr = 0.001

# 학습
with tf.device("/device:GPU:0"):
    f_logistic_model.fit(x=X, y=y, epochs=20, batch_size=5)
    # verbose=False 하면 막대바 안보임
# 모델 저장
from keras.models import load_model
f_logistic_model.save('my_model.h5')
# del f_logistic_model    # 테스트를 위해 메모리 내의 모델을 삭제
# f_logistic_model = load_model('./my_model.h5')

df_test = pd.read_csv('./database/test.csv')
pid = df_test["PassengerId"]  # [!!!]submit.csv 를 만들어줄 때 PassengerId가 필요하기 때문에 drop하기 전에 저장해둔다
# predict하기 위한 test data도 columns를 동일하게 맞춰주어야 한다
df_test = df_test.drop(["Name", "PassengerId", "Ticket", "Fare", "Cabin"], axis=1)
df_test["Age"].fillna(df_test["Age"].mean(), inplace=True)
df_test["Embarked"] = df_test["Embarked"].fillna("S")
df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})
df_test["Embarked"] = df_test["Embarked"].map({"Q": 0, "C": 1, "S": 2})
family = []
for i in range(len(df_test)):
    if df_test.loc[i, "SibSp"] >= 4:
        family.append(2)
    elif df_test.loc[i, "SibSp"] >= 2:
        family.append(1)
    else:
        family.append(0)
df_test["Family"] = family
test_result = f_logistic_model.predict(df_test, batch_size=32)

submit = pd.DataFrame({"PassengerId": pid, "Survived": test_result})

# index=False를 안하면 csv파일 0번쨰 columns에 index를 붙여준다 즉 제출용에 맞게 설정해주어야 한다
submit.to_csv("./database/submit.csv", index=False)


# model.evaluate() = [오차율?, 예측율]
# score = f_logistic_model.evaluate(X_test, y_test, verbose=1)
# print(f"정답률={score[1]*100} loss={score[0]}")


def score():
    df_perfect = pd.read_csv
df_perfect = pd.read_csv("./database/wow.csv")
submit = pd.read_csv("./database/submit.csv")

hit = 0
miss = 0
for i in range(len(test_result)):
    if submit["Survived"][i] == df_perfect["Survived"][i]:
        hit += 1
    else:
        miss += 1

print(hit, miss, round(hit / (hit + miss), 4))
