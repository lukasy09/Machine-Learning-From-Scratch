import numpy as np

""" This class implements a simple Machine Learing algorithm - Linear Regression for 1-dimensional
        dataset. It has been done to fully understand the algorithm begind and compare with existing
                open source algorithms & models"""
                
class SimpleLinearRegression:
    def __init__(self, learning_rate = 0.0001):
        self.learning_rate = learning_rate
      
        
    def loss(self,x,y,a,b):
        if len(x) == len(y):
            self.n_points = len(x)
            error = 0
            for i in range(0, self.n_points):
                x_i = x[i]
                y_i = y[i]
                error += np.power((y_i - (a*x_i + b)), 2)
        return error / float(self.n_points)
    
    
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
        return [a, b]    
            
            
                
regressor = SimpleLinearRegression()