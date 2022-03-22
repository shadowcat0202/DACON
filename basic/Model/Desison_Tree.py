import sklearn


# 의사 결정 분류(특정 집단이 필요한 경우)
def DecisionTreeClassifier():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier()


# 의사 결정 회귀(연속적인 데이터)
def DecisionTreeRegressor():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor()


# 랜덤 포레스트
# = 여러개의 DecisionTree를 만들어 이들을 평균으로 예측의 성능을 높이는 방법
# =: 앙상블
def RandomForestRegressor():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(criterion="squared_error") # MES(최소제곱오차)
