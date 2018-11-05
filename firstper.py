import math
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def neur(X, y):
    w1 = 0.1
    w2 = 0.1
    E = 0.01
#    w_ideal = 1
    for i in range(10000):
        error = []
        for x, ideal in zip(X, y):

            out = w1*x + w2*(1)
#            h = sigmoid(out)*((1 - out)*out)
            delta = ideal - out            
            error.append(delta**2)            
            
            gr1 = w1*x*delta
            gr2 = w2*1*delta
#            delta = gr
#            d_w = 0.1*gr
            d_w1 = E*gr1
            d_w2 = E*gr2
            w1 += d_w1
            w2 += d_w2
            
#            print(d_w1, d_w2)
        print("error = ",sum(error)/len(error))            
    tup = (w1, w2)
    return tup



#X = [9, 8, 7, 1, 2, 3]
#y = [1, 1, 1, 0, 0, 0]

X = [4, 6, 8, 10]
y = [1, 2, 3, 4]

tup = neur(X, y)
w1 = tup[0]
w2 = tup[1]
print(w2)

def pred(w, X):
    if X*w > 0.0:
        prediction = 1
    else:
        prediction = 0
    return prediction
            
#print(pred(w, 1), w)    

   
#neur(3)  
#
def func(xlist, w1, w2):
    y = []
    for i in xlist:
        y.append(w1*i + w2*(1))
    return y
#
xlist = list(range(10)) 
plt.plot(xlist, func(xlist, w1, w2))
plt.scatter(X, y)

