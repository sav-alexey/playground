import math
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def neur(X, y):
    w1 = 0.1
    w2 = 0.3
#    w_ideal = 1
    for i in range(1):
        error = []
        for x, ideal in zip(X, y): # x = 4
            h = w1*x + w2*1 # h = 0.1*4 = 0.4
            out = ideal - h # 2- 0.4 = 1.6
            error.append(out**2)            
#            gr = h*out
#            delta = gr
#            d_w = 0.1*gr
            d_w = 1*out
            w += d_w 
            print(w)
        print("error = ",sum(error)/len(error))            
    return w



#X = [9, 8, 7, 1, 2, 3]
#y = [1, 1, 1, 0, 0, 0]

X = [4]
y = [2]

w = neur(X, y)

def pred(w, X):
    if X*w > 0.0:
        prediction = 1
    else:
        prediction = 0
    return prediction
            
print(pred(w, 1), w)    

   
#neur(3)  
#
def func(xlist, w):
    y = []
    for i in xlist:
        y.append(w*i)
    return y
#
xlist = list(range(10)) 
plt.plot(func(xlist, w), xlist)
plt.scatter(X, y)