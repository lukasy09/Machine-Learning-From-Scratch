import numpy as np
from DataNotNumpyArray import DataNotNumpyArray
from DimensionsNotFittingException import DimensionsNotFittingException


""" This class implements a simple Machine Learing algorithm - Linear Regression for 1-dimensional
        dataset. It has been done to fully understand the algorithm begind and compare with existing
                open source algorithms & models"""
      
class SimpleLinearRegression:
    def __init__(self, learning_rate = 0.0001):
        self.learning_rate = learning_rate
        self.X = None
        self.Y = None
        self.n_points = 0
        self.epochs = 0
        self.parameter_a = 0
        self.parameter_b = 0
        self.is_fitted = False
        self.loss_values = []
        
    """Loss function, we're going to try to minimize it as much as possible"""    
    def loss(self,x,y,a,b):
        if len(x) == len(y):
            self.n_points = len(x)
            error = 0
            for i in range(0, self.n_points):
                x_i = x[i]
                y_i = y[i]
                error +=(y_i - (a*x_i + b)) ** 2
        return error / float(self.n_points)
    
    """Returning loss after each epoch/iteration"""
    def get_loss_values(self):
        if self.is_fitted:
            return self.loss_values
        else:
            return None
    """Returning a * x + b linear parameters [a,b]"""
    
    def get_linear_parameters(self):
        if self.is_fitted:
            return np.asarray([self.parameter_a, self.parameter_b])
        else:
            return None
        
        
    """Whole `the logic` Updating values for parameters"""
    
    def step_gradient(self,X, Y, a_current, b_current):
        a_gradient = 0
        b_gradient =0
        n = float(len(X))
        for i in range(0, len(X)):
            x = X[i]
            y = Y[i]
            b_gradient +=  -(2/n) *(y - ((a_current*x) + b_current))
            a_gradient +=  -(2/n) *x*(y - ((a_current*x) + b_current))
        new_a =a_current -  self.learning_rate *a_gradient
        new_b =b_current -  self.learning_rate *b_gradient
        return [new_a, new_b]
            
    
    def run_gradient_descent(self,x, y, num_iterations):
        a = 0
        b = 0
        for it in range(0, num_iterations):
            a,b = self.step_gradient(x, y, a, b)
            loss = self.loss(x, y, a, b)
            self.loss_values.append(loss)
        self.parameter_a = a
        self.parameter_b = b
        return [a, b]    
    
    """Method used by user, fits the dataset with its labels"""
    def fit(self, X, Y, epochs = 1000):
        self.n_points = len(X)
        try:
            if self.n_points != len(Y):
                raise DimensionsNotFittingException()    
            self.X = X
            self.Y = Y
            self.epochs = epochs
            self.run_gradient_descent(self.X, self.Y, self.epochs)
            self.is_fitted = True
        except DimensionsNotFittingException as error:
            error.display_message()
    """Predicting.... for test values"""        
    def predict(self, X_test):
        if self.is_fitted:
            predict_Y = []
            for x in X_test:
                y_pred = self.parameter_a * x + self.parameter_b
                predict_Y.append(y_pred)
        else:
            print("Model is not fitted yet. Use `fit ` method to complete the model")
        return np.asarray(predict_Y)