from linear_regression import SimpleLinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_path = "./data/Salary_Data.csv"
dataframe =pd.read_csv(data_path)

X = dataframe['YearsExperience'].values
Y = dataframe['Salary'].values

model = SimpleLinearRegression()

"""Uncomment if you want to see the effect after 100000 epochs/iterations"""
model.fit(X, Y, epochs = 100000)
#plt.figure(1, figsize = (8,5))
#plt.scatter(X, Y)
#plt.plot(X, model.predict(X))

def generate_model_progress(model,X, Y, save_path = "./generator_pictures/", max_epoch = 1000):
    model = SimpleLinearRegression()
    picture_prefix = "pic"
    picture_sufix = ".jpg"
    for epochs_number in range(0, max_epoch):
        model.fit(X, Y, epochs = epochs_number)
        
        plt.figure(figsize = (8,5))
        plt.scatter(X, Y)
        plt.plot(X, model.predict(X))
        plt.savefig(save_path + str(epochs_number))

"""Uncomment if you want to generate plot for each epoch"""
#generate_model_progress(model, X, Y)
#print("Done")

plt.figure(figsize=(8,5))
plt.title("Dependency - Loss on epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
epoch_list = [i for i in range(0, 1000)]
epoch_list = np.asarray(epoch_list)
loss_vals = model.get_loss_values()
plt.plot(epoch_list, loss_vals[0:1000])