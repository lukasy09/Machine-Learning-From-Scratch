import pandas as pd
import os
import multiple_linear_regression
#from multiple_linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



scaler = StandardScaler()

df = pd.read_csv("./data/50_Startups.csv")
X = df.iloc[:, 0:2].values
Y = df['Profit'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#My model

regressor_mine = multiple_linear_regression.LinearRegression()
regressor_mine.fit(X_train, Y_train, epochs = 10000)

y_mypreds = regressor_mine.predict(X_test)

regressor_sklearn = LinearRegression()
regressor_sklearn.fit(X_train, Y_train)
y_skpreds = regressor_sklearn.predict(X_test)


fig = plt.figure(figsize = (16, 10))
ax = plt.axes(projection='3d')
ax.scatter3D(X_train[:,0], X_train[:, 1], Y_train, cmap='binary');