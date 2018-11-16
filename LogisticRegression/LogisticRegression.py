import numpy as np
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
        
    """Enables to count the sigmoid of X [-1, 1] <- range of the func"""    
    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    
                
    """The pure logic behinds the classifier/model"""
    def step_gradient(self, X, Y, params, bias):
        n = self.n_points
        param_gradient = [0] * len(params)
        new_params = [0] * len(params)
        bias_gradient = 0

        for i in range(0, n):
            x = X[i]
            y = Y[i]
            for param_it in range(0, len(params)):
              main_x = x[param_it]  
              pred = self.predictor(x, params, bias)
              param_gradient[param_it] += (y-pred) * pred * (1-pred) * main_x
            bias_gradient += (y-pred) * pred *(1-pred) * 1.0 #assumption that the 0 input is always equal to 1
        for param_it in range(0, len(params)):
            new_params[param_it] = params[param_it] + self.learning_rate * param_gradient[param_it]
        new_bias = bias + self.learning_rate * bias_gradient
        return [new_params, new_bias]

                
    """Run above function `epochs` times"""
    def run_gradient_descent(self,X, Y):
        params = [0] * self.n_features
        bias = 0
        for epoch in range(0, self.epochs):
            params, bias = self.step_gradient(X, Y, params, bias)
            #self.loss(X, Y, params,bias)
        self.params = params
        self.bias = bias
        return [self.params, self.bias]
    
    """User interface"""
    def fit(self, X, Y, epochs = 100, learning_rate = 0.0001):
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
            self.run_gradient_descent(self.X, self.Y)
        except ValueError as e:
            print("Dimension are not matching! {}".format(e))
    
            
    """"Method for user, predicting from an array"""        
    def predict(self,X, params = None, bias = None):
        
        if params is None or bias is None:
            params = self.params
            bias = self.bias
        output = []
        
        for i in range(0, len(X)):
            x = X[i]
            sum = 0
            for param_it in range(0, self.n_features):
                sum += x[param_it] * params[param_it]
            sum += bias
            normalised_sum = self.sigmoid(sum)
            if normalised_sum > 0.5:
                output.append(1)
            else:
                output.append(0)
        return np.asarray(output)
    
    """Predicts of a single row"""
    def predictor(self, x, params, bias):
        output = 0
        for param_it in range(0, len(params)):
            output += x[param_it] * params[param_it]
        output += bias
        return self.sigmoid(output)