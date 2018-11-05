import tensorflow as tf
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = np.float32(iris.data[:100, [2,3]])
Y = np.float32(iris.target[:100])

X_data = np.float32(iris.data[:100, [2,3]])
Y_data = np.float32(iris.target[:100])
#Y_data = Y_data.reshape([100, 1])

Y = Y.reshape([100, 1])

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#X_pred = X[1,:]
#X_pred = X_pred.reshape([1, 2])
#print(X_pred)

tf.reset_default_graph() 
tf.set_random_seed(1)  
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", [5, 2], initializer =  tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", [5, 1], initializer = tf.zeros_initializer())
W2 = tf.get_variable("W2", [1, 5], initializer =  tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", [1, 1], initializer = tf.zeros_initializer())
#    
Z1 = tf.add(tf.matmul(W1, X), b1)
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(W2, A1), b2)
A2 = tf.sigmoid(Z2)

Y = tf.transpose(Y)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2, labels=Y)) 
acc, acc_op = tf.metrics.accuracy(labels = X, predictions = Y)

y = []
init = tf.global_variables_initializer()
local = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
#    print(sess.run(b1))
#    print(sess.run(A2, feed_dict={X:X_data.T}))
    for epoch in range(700):        
        costx = sess.run([optimizer, cost], feed_dict={X:X_data.T, Y:Y_data})[1]
        y.append(costx)
    y_cap = sess.run(A2, feed_dict={X:X_data.T})
    y_cap = np.where(y_cap<0.5, 0, 1)
    y_cap = np.squeeze(y_cap)
    print(y_cap)
    print(Y_data)
    sess.run(local)
    print(sess.run([acc, acc_op], feed_dict={X:y_cap, Y:Y_data}))
    
#    print(sess.run(b1))

#plt.ylabel('cost')
#plt.xlabel('iterations')
#plt.plot([i for i in range(200)], y)
