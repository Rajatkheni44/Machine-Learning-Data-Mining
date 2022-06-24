import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from scipy.special import expit


class MyANN:
    
    def __init__(self):
        self.layer_sizes = [64, 32, 16, 8, 10]      #fist element shpold be equel to input unit (n) and last equal to out put unit
        self.num_layers = len(self.layer_sizes)
        self.init_weights()
    
    def init_weights(self):
        self.weights = []
        truncnorm_gen = truncnorm(-1, 1)
        for i in range(self.num_layers - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i+1]
            Weights = truncnorm_gen.rvs(input_size*output_size).reshape(input_size, output_size)
            self.weights.append(Weights)
        return self.weights
        
    def forward(self, X, weights):
        Xs = [X.copy()]
        for l in range(self.num_layers - 1):
            X = X @ weights[l]
            X = expit(X)
            Xs.append(X)
        return Xs
    
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y
    
    def sigmoid_derivative(self, Z):
        return Z * (1 - Z)                 #because a is sigmoid of z

    def back_prop(self, Xs, Y, weights):
        m = Y.size
        one_hot_Y = self.one_hot(Y)
        delta = Xs[-1] - one_hot_Y
        del_weights = []
                   
        for r in reversed(range(self.num_layers - 1)):
            Del_weight = (1/m)*(Xs[r].T @ delta)
            del_weights.insert(0, Del_weight)                 # check
            delta = np.multiply(delta @ weights[r].T,self.sigmoid_derivative(Xs[r]))
            
        return del_weights
    
    def update_params(self, del_weights, weights, alpha):
        new_weights = []
        for NW in range(self.num_layers - 1):
            updated_weight = weights[NW] - alpha*del_weights[NW]
            new_weights.append(updated_weight)
        return new_weights
    
    def get_predictions(self, X):
        return np.argmax(X, 0)
    
    def get_accuracy(self, predictions, Y):
        #print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y, alpha, iterations):
        #W1, b1, W2, b2 = init_params()
        weights = self.init_weights()

        for i in range(iterations):
            Xs = self.forward(X, weights)
            dW = self.back_prop(Xs, Y, weights)
            weights = self.update_params(dW, weights, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(Xs[-1].T)
                #print(predictions.shape)
                print("Neural model accuracy is =", self.get_accuracy(predictions, Y))
        return 
