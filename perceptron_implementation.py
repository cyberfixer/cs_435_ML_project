import numpy as np
class Perceptron(object):
    def __init__(self, n_iter=10) :
        self.n_iter = n_iter
        
    def train(self, X, d):
        self.weights = np.zeros(X.shape[1])#np.random.random(X.shape[1])
        self.b = 0
        self.errors_ = []

        for i in range(self.n_iter):
            cError = 0
            for xrow, y in zip(X, d):
                a = np.dot(xrow, self.weights) + self.b

                # Update
                if y * a <= 0:
                    for k in range(len(xrow)):
                        self.weights[k] += y * xrow[k]
                    self.b = self.b + y
                    cError += 1
            self.errors_.append(cError)

        return self 
                
    def predict(self, X):
        a = np.dot(X, self.weights) + self.b
        return np.where(a >= 0, 1, -1)


