import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd

class LogisticRegression:


    def __init__(self):
        self.learning_rate = 0
        self.is_fitted = False
        self.epochs = 0
        self.X = None
        self.Y = None
        self.size = None
        self.n_points = 0
        self.n_features = 0
        self.params = []
        self.bias = 0
        self.loss_values = []
    
    def step_gradient(self, X, Y, params, bias):
        n = self.n_points
        param_gradient = [0] * len(params)
        new_params = [0] * len(params)
        bias_gradient = 0
        new_bias = 0
        """Loop over whole X array (every row)"""
        for i in range(0, n):
            x = X[i]
            y = Y[i]
            for param_index in range(0 , len(params)):
                main_param = params[param_index]
                sum = 0
                for param_it in range(0, len(params)):
                    sum += params[param_it] * x[param_it]
                sum += bias
                if y == 1:
                    param_gradient[param_index] += -1/sum * main_param * 1/n
                if y == 0:
                    param_gradient[param_index] += (1/1-sum) * main_param * 1/n
            b_sum = 0
            for param_it in range(0, len(params)):
                b_sum  += params[param_it] * x[param_it]
            if y == 1:
                bias_gradient += (-1/b_sum) * 1/n
            if y == 0:
                bias_gradient += (1/1-b_sum) * 1/n
        for it in range(0, len(params)):
            ew_params[it] = params[it] - self.learning_rate * param_gradient[it]
        
        new_bias = bias - self.learning_rate * bias_gradient  
        
        return [new_params, new_bias]
                
    
    def fit(self, X, Y, epochs = 1000, learning_rate = 0.0001):
        self.epochs = epochs
        try:
            if len(X) != len(Y):
                raise ValueError()
            self.X = X
            self.Y = Y
            self.learning_rate = learning_rate
            self.size = X.shape
            self.n_points = self.size[0]
            self.n_features = self.size[1]
            self.params = [0] * self.n_features
        except ValueError as e:
            print("Dimension are not matching! {}".format(e))
        
scaler = StandardScaler()
classifier = LogisticRegression()
df = pd.read_csv("./data/Social_Network_Ads.csv")
X = df.iloc[:, 1:4].values
X[:, 0] = LabelEncoder().fit_transform(X[:, 0])
X = scaler.fit_transform(X)
Y = df['Purchased'].values
