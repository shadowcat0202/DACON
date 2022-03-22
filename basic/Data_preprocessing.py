# 전처리
# 데이터가 일부 손실되거나 없는(NaN) 인 경우
# 데이터들을 사용하기 전에 처리하는 방법 => 전처리
import pandas as pd


def dropNaN(data):
    # DataFrame.dropna()
    import pandas
    res = data.dropna()  # NaN이 포함된 row를 모두 drop
    del pandas
    return res


def fillNaN(data):
    # DataFrame.fillna()
    import pandas
    res = data.fill(0)  # NaN인 셀을 0으로 채운다
    del pandas

    # 평균값으로 채우고 싶다면
    # DataFrame.fillna({column_name:dataType(df[column_name].mean())}, inplace=True)
    # DataFrame을 잘 보고 원하는 부분을 잘 걸러내야한다
    return res


def Interpolate(data):
    # 보간법
    # mean(df[n-1], df[n+1])
    import pandas
    res = data.interpolate(implace=True)
    del pandas
    return res
