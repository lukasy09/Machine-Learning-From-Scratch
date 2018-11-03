import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
"""General version of Regressor Class. Can handle more than 1 kind of arguments as input"""
class LinearRegression:
    
    def __init__(self):
        self.learning_rate = 0.001
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
    """Scaling input data"""    
    @staticmethod
    def scale_input_data(X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def loss(self, X, Y, params, bias):
        error = 0
        n = self.n_points
        for i in range(0, n):
            x = X[i]
            y = Y[i]
            product_sum = 0
            for param_it in range(0, len(params)):
                product_sum += params[param_it] * x[param_it]
            product_sum += bias
            error += (y - product_sum) ** 2
            print(error / n)
        return error / n
            
    def step_gradient(self, X, Y, params, bias):
        n = self.n_points
        param_gradient = [0] * len(params)
        new_params = [0] * len(params)
        bias_gradient = 0
        
        for i in range(0, n):
            x = X[i]
            y = Y[i]
            for param_index in range(0, len(params)):
                main_x = x[param_index]
                sum = 0
                for param_it in range(0, len(params)):
                    sum += params[param_it] * x[param_it]
                sum += bias
                param_gradient[param_index] += (-2/n) * main_x * (y - sum)
            b_sum = 0
            for param_it in range(0, len(params)):
                b_sum +=  x[param_it] * params[param_it]
            b_sum += bias
            bias_gradient += (-2/n) * (y - b_sum)
        for it in range(0, len(params)):
            new_params[it] = params[it] - self.learning_rate * param_gradient[it]
        
        new_bias = bias - self.learning_rate * bias_gradient
        self.params = new_params
        self.bias = new_bias
        return [new_params, new_bias]
        
        
    def run_gradient_descent(self, X, Y, n_epochs):
        params = [0] * self.n_features
        bias = 0
        for epoch in range(0, n_epochs):
            params, bias = self.step_gradient(X, Y, params, bias)
            loss = self.loss(X, Y, params, bias)
            self.loss_values.append(loss)
        return [params, bias]
            
                                
    def fit(self, X, Y, epochs = 1000):
        if len(X) != len(Y):
            print("Dimension error!")
            return None
        self.X = X
        self.Y = Y
        self.epochs = epochs
        self.size = X.shape
        self.n_points = self.size[0]
        self.n_features = self.size[1]
        self.params = [0] * self.n_features
        X = LinearRegression.scale_input_data(X)
        params, bias = self.run_gradient_descent(self.X, self.Y, epochs)