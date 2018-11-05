import numpy as np
import random
import matplotlib.pyplot as plt

class NN:
    
    errors = []
    
    def __init__(self):
        random.seed(1)
        self.weights = np.random.random((2,1))
    
    def fit(self, X, y, it):
        
        for i in range(it):
            output = self.predict(X)
            error = y - output
            adjustment = 0.001*np.dot(X.T, error)
            self.weights += adjustment
            self.errors.append(sum(error)**2)
            
        return self
            
    def predict(self, X):
        return (np.dot(X, self.weights))


X = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]])
y = np.array([[1, 4, 9, 16, 25, 36]]).T

#X_test = np.array([[8, 1], [9, 1], [10, 1], [11, 1], [12, 1], [13, 1], [14, 1]])
X_test = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]])

neur = NN()
neur.fit(X, y, 10000)
w = 1/neur.weights

#print(w)
print(neur.predict(np.array([[8, 1]])))

plt.plot(X_test, neur.predict(X_test), color="red")
plt.plot(X[0:,0], y)
plt.scatter(X[0:,0], y)
#plt.plot([x for x in NN.errors], [x for x in range(10000)])

