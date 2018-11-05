import numpy as np
import random

class NN:
    def __init__(self):
        random.seed(1)
        self.weights = 2*np.random.random((2,1)) -1
    
    def fit(self, X, y, it):
        for i in range(it):
            output = self.predict(X)
            error = y - output
            adjustment = 0.01*np.dot(X.T, error)
            self.weights += adjustment
            print((sum(error)**2))
            
    def predict(self, X):
        return (np.dot(X, self.weights))


X = np.array([[2, 3], [1, 1], [5, 2], [12, 3]])
y = np.array([[10, 4, 14, 30]]).T

neur = NN()
neur.fit(X, y, 100)


