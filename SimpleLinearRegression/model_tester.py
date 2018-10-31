from linear_regression import SimpleLinearRegression
import pandas as pd
import matplotlib.pyplot as plt

data_path = "./data/Salary_Data.csv"
dataframe =pd.read_csv(data_path)

X = dataframe['YearsExperience'].values
Y = dataframe['Salary'].values

model = SimpleLinearRegression()
model.fit(X, Y, epochs = 10000)

plt.scatter(X, Y)
plt.ylim(20000, 12000)
plt.plot(X, model.predict(X))