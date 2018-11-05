import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing 

data = np.array(pd.read_csv("wine.data.csv"))
#data = np.array(pd.read_csv("test.csv"))
X_data = data[:,1:]
Y_data = data[:,0].T

#Y_data = Y_data.reshape([177, 1])
iteration = 400


#X_data = preprocessing.normalize(X_data)
X_data = preprocessing.scale(X_data)

tf.reset_default_graph() 
tf.set_random_seed(1)  
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", [20, 13], initializer =  tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", [20, 1], initializer = tf.zeros_initializer())
W2 = tf.get_variable("W2", [5, 20], initializer =  tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", [5, 1], initializer = tf.zeros_initializer())

W3 = tf.get_variable("W3", [3, 5], initializer =  tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", [1, 1], initializer = tf.zeros_initializer())

Z1 = tf.add(tf.matmul(W1, X), b1)
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(W2, A1), b2)
A2 = tf.nn.relu(Z2)
Z3 = tf.add(tf.matmul(W3, A2), b3)
A3 = tf.nn.softmax(Z3)

#Y_data = tf.transpose(Y_data)

one_hot_matrix = tf.one_hot(Y_data, 3, axis=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)
acc, acc_op = tf.metrics.accuracy(labels = X, predictions = Y)

init = tf.global_variables_initializer()
local = tf.local_variables_initializer()

y = []
with tf.Session() as sess:
    sess.run(init)
    sess.run(local)
#    y_cap = sess.run(A2, feed_dict={X:X_data.T})
#    print(y_cap)
    Y_data = sess.run(one_hot_matrix).T
#    print(X_data.shape)
    
    for epoch in range(iteration):        
        _, costx, y_cap = sess.run([optimizer, cost, A3], feed_dict={X:X_data.T, Y:Y_data})
        y.append(costx)
    np.set_printoptions(suppress=True)
    print(y_cap)
#    y_cap = sess.run(A2, feed_dict={X:X_data.T})
   
        
plt.ylabel('cost')
plt.xlabel('iterations')
plt.plot([i for i in range(iteration)], y)