import pandas as pd
import os
import multiple_linear_regression
#from multiple_linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation

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
y_skpreds = regressor_sklearn.predict(X_test)]


#LOSS

losses = regressor_mine.loss_values
epochs = np.asarray([epoch for epoch in range(0, 10000)])

plt.plot(epochs, losees)
plt.xlabel("epoch")
plt.ylabel("loss")

