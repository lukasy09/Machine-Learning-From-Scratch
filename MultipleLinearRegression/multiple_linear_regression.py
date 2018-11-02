import numpy as np
from sklearn.preprocessing import StandardScaler


class LinearRegression:
    
    def __init__(self):
        self.learning_rate = 0.0001
        self.is_fitted = False
        self.epochs = 0
        self.X = None
        self.Y = None
        self.size = None
        self.n_points = 0
        self.n_features = 0
        self.params = []
        self.bias = 0
      
    @staticmethod
    def scale_input_data( X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def step_gradient(X, Y, params, bias):
        n = len(X)
        
        for i in range(0, n):
            for j in range(0, len(params)):
                row = X[i]
                param = params[j]
                print(row)
                
                
        
    
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
        
        

    
    
    
    
    
regressor = LinearRegression()
regressor.fit(X_train, y_train)