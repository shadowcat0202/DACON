import pandas as pd
import numpy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("./database/sonar.csv", header=None)

df = df.values
X = df[:, :60].astype(float)
y = df[:, 60]
print(len(X[0]))