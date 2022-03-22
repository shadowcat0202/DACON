def feature_important_test(model, X_train):
    idx = 0
    for col in X_train:
        print("%s\t%.8f" % (str(col).rjust(25, " "), model.feature_importances_[idx]))
        idx += 1


# 램덤포레스트의 예측변수의 중요도(=: 중요도가 낮은 feature을 날리고 다시 학습시킬 수있다는 장점)
def feature_important_test(train, Y_train, test):
    from Model import Desison_Tree
    import CSV_process
    #랜덤포레스트 feature중요도에 따라 drop한거 종류별로 학습시켜보기(제거할 col명은 상황에 맞게 변경해주어야한다)
    #나중에 해당 기능 추가 할 수 있다면 추가

    X_train = [train.drop(['count', 'id'], axis=1),
               train.drop(['count', 'id', 'hour_bef_windspeed'], axis=1),
               train.drop(['count', 'id', 'hour_bef_windspeed', 'hour_bef_pm2.5'], axis=1)]

    Test = [test.drop(['id'], axis=1),
            test.drop(['id', 'hour_bef_windspeed'], axis=1),
            test.drop(['id', 'hour_bef_windspeed', 'hour_bef_pm2.5'], axis=1)]

    model1 = Desison_Tree.RandomForestRegressor()
    model1.fit(X_train[0], Y_train)
    model2 = Desison_Tree.RandomForestRegressor()
    model2.fit(X_train[1], Y_train)
    model3 = Desison_Tree.RandomForestRegressor()
    model3.fit(X_train[2], Y_train)

    y_pred = [model1.predict(Test[0]),
              model2.predict(Test[1]),
              model3.predict(Test[2])]

    i = 1
    for pred in y_pred:
        CSV_process.predict_to_csv(pred, "sub_" + str(i))
        i+=1

    del Desison_Tree, CSV_process

def GridSearch_avatar():
    # 하이퍼파라미터 튜닝은 정지규칙 값들을 설정하는 것을 의미합니다.의사결정나무에는
    # 정지규칙(stopping criteria) 이라는 개념이 있는데
    #
    # 1.최대깊이(max_depth) : 최대로 내려갈 수 있는 depth
    # 2.최소 노드크기(min_samples_split)  : 노드를 분할하기 위하느 데이터 수
    # 3.최소 향상도(min_impurity_decrease)   : 노드를 분할하기 위한 최소 향상도
    # 4.비용복잡도(Cost - complexity)    : 트리가 커지는 것에 대해 패널티 계수를 설정해서 불순도와 트리가 커지는 것에 대한 복잡도 계산

    from sklearn.model_selection import GridSearchCV

