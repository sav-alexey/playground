import numpy as np
import random
import matplotlib.pyplot as plt

class NN:
    def __init__(self):
        random.seed(1)
        self.weights = np.array([[0.2],[0.2]])
    
    def fit(self, X, y, it):
        for i in range(it):
            output = self.predict(X)
            error = y - output
            adjustment = 0.001*np.dot(X.T, error)
            self.weights += adjustment
            print((sum(error)**2))
            
    def predict(self, X):
        return (np.dot(X, self.weights))


X = np.array([[4, 0], [6, 0], [8, 0], [10, 0]])
y = np.array([[1, 2, 3, 4]]).T

neur = NN()
neur.fit(X, y, 10000)
w = 1/neur.weights
print(neur.predict(np.array([[12, 1]])))

#print(w)

#def func(xlist, w):
#    y = []
#    w1 = w[:1]
#    w2 = w[:0]
#    for i in xlist:
#        y.append(w1*i + w2)
#    return y
##
#xlist = list(range(10)) 
#plt.plot(xlist, func(xlist, w))
