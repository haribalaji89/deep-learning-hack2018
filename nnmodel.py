import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import numpy as np
from sklearn.decomposition import PCA

from numpy import genfromtxt
X_raw = genfromtxt('D:/RCS/FintechHackData/FinTechRatiosTrain.csv', delimiter=',', dtype=None)
X_real = X_raw[1:]
X_real = X_real[...,2:]
X_float = X_real.astype(float)
X_raw_test = genfromtxt('D:/RCS/FintechHackData/FinTechRatiosTest.csv', delimiter=',', dtype=None)
X_real_test = X_raw_test[1:]
X_real_test = X_real_test[...,2:]
X_float_test = X_real_test.astype(float)
X_combined = np.concatenate((X_float, X_float_test), axis=0)
Y_combined = X_combined[...,18]
Y_combined = Y_combined.astype(int)
X_comb_float = X_combined[:, :-1]
mu = np.mean(X_comb_float, axis=0)
minm = np.min(X_comb_float, axis=0)
maxm = np.max(X_comb_float, axis=0)
X_norm = (X_comb_float - mu) / (maxm - minm)
X_train = X_norm[:(X_raw.shape[0]-1), :]
X_test = X_norm[(X_raw.shape[0]-1):, :]
Y_train = Y_combined[0:(X_raw.shape[0]-1)]
Y_test = Y_combined[(X_raw.shape[0]-1):]

#Data visualization
#pca = PCA(n_components=2)
#pca.fit(X_train)
#pca.transform(X_train)

#clf = sklearn.linear_model.LogisticRegressionCV();
#clf.fit(X_train_gph.T, Y_train);
#LR_predictions = clf.predict(X_train_gph)

#plot_decision_boundary(lambda x: clf.predict(x), X_train_gph, Y_train)
#plt.title("Logistic Regression")
#print ('Accuracy of logistic regression: %d ' % float((np.dot(Y_train, LR_predictions) + np.dot(1 - Y_train,1 - LR_predictions)) / float(Y_train.size) * 100) + '% ' + "(percentage of correctly labelled datapoints)")
#print ('Accuracy of logistic regression: %d ' % float((np.dot(Y_test, LR_predictions) + np.dot(1 - Y_test,1 - LR_predictions)) / float(Y_test.size) * 100) + '% ' + "(percentage of correctly labelled datapoints)")

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

Y_train_hot = convert_to_one_hot(Y_train, 2)
Y_test_hot = convert_to_one_hot(Y_test, 2)

#params = model(X_train_gph, Y_train_hot, X_train_gph, Y_train_hot, num_iterations=2000)
#plot_decision_boundary(lambda x: predict(x.T, params), X_train_gph, Y_train)

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_iterations = 1000, minibatch_size=32, print_cost = True):
        tf.reset_default_graph()
        dropout_val = 1
        tf.set_random_seed(1)
        seed = 3
        n_x = X_train.shape[0]
        m = X_train.shape[1]
        n_y = Y_train.shape[0]
        costs = []
        print(n_x)
        print(n_y)
        print(m)
        X, Y = create_placeholders(n_x, n_y)
        print(str(X))
        print(str(Y))
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
                sess.run(init)
                for epoch in range(num_iterations):
                        epoch_cost = 0
                        num_minibatches = int(m / minibatch_size)
                        seed = seed + 1
                        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
                        for minibatch in minibatches:
                                (minibatch_X, minibatch_Y) = minibatch
                                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})                
                                epoch_cost += minibatch_cost / num_minibatches
                        if print_cost == True and epoch %100 == 0:
                                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                        if print_cost == True and epoch % 5 == 0:
                                costs.append(epoch_cost)
                plt.plot(np.squeeze(costs))
                plt.ylabel('cost')
                plt.xlabel('iterations (per tens)')
                plt.title("Learning rate =" + str(learning_rate))
                plt.show()
                parameters = sess.run(parameters)
                print(dropout_val)
                print("Parameters have been trained!")
                correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}) * 100)
                dropout_val = 1
                print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}) * 100)
                return parameters

def cost(logits, labels):
        z = tf.placeholder(tf.float32, name="z")
        y = tf.placeholder(tf.float32, name="y")
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
        sess = tf.Session()
        cost = sess.run(cost, feed_dict={z: logits, y: labels})
        sess.close()
        return cost

def create_placeholders(n_x, n_y):
        X = tf.placeholder(tf.float32, [n_x, None], name="X")
        Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
        return X, Y
def initialize_parameters():
        tf.set_random_seed(1)
        W1 = tf.get_variable("W1", [30, 18], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b1 = tf.get_variable("b1", [30, 1], initializer = tf.zeros_initializer())
        W2 = tf.get_variable("W2", [30,30], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b2 = tf.get_variable("b2", [30, 1], initializer = tf.zeros_initializer())
        W3 = tf.get_variable("W3", [2, 30], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        #W1 = tf.get_variable("W1", [10, 2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        #b1 = tf.get_variable("b1", [10, 1], initializer = tf.zeros_initializer())
        #W2 = tf.get_variable("W2", [10,10], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        #b2 = tf.get_variable("b2", [10, 1], initializer = tf.zeros_initializer())
        #W3 = tf.get_variable("W3", [2, 10], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b3 = tf.get_variable("b3", [2, 1], initializer = tf.zeros_initializer())
        parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
        return parameters
def forward_propagation(X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']
        Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
        A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
        A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        return Z3
def compute_cost(Z3, Y):
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):    
        m = X.shape[1]
        mini_batches = []
        np.random.seed(seed)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
        num_complete_minibatches = math.floor(m/mini_batch_size)
        for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
                mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
                mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
                mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)
        return mini_batches

def initialize_parameters_lr():
        tf.set_random_seed(1)
        W = tf.get_variable("W", [2, 18], initializer = tf.contrib.layers.xavier_initializer(seed=1))        
        b = tf.get_variable("b", [2, 1], initializer = tf.zeros_initializer())
        parameters = {"W": W,
                  "b": b}
    
        return parameters
def forward_propagation_lr(X, parameters):
        W = parameters['W']
        b = parameters['b']
        Z = tf.add(tf.matmul(W, X), b)
        return Z

def model_lr(X_train, Y_train, X_test, Y_test, learning_rate=0.01, num_epochs = 1500, print_cost = True):
        tf.set_random_seed(1)
        seed = 3
        n_x = X_train.shape[0]
        m = X_train.shape[1]
        n_y = Y_train.shape[0]
        costs = []
        print(n_x)
        print(n_y)
        print(m)
        X, Y = create_placeholders(n_x, n_y)
        print(str(X))
        print(str(Y))
        parameters = initialize_parameters_lr()
        Z = forward_propagation_lr(X, parameters)
        cost = compute_cost(Z, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
                sess.run(init)
                for epoch in range(num_epochs):
                        _ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})                
                        if print_cost == True and epoch %100 == 0:
                                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))     
                parameters = sess.run(parameters)
                print("Parameters have been trained!")
                correct_prediction = tf.equal(tf.argmax(tf.sigmoid(Z)), tf.argmax(Y))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
                return parameters
        
def predict(X, parameters):
    print("Inside convert to tensor : " + str(X.shape))
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    print("After convert to tensor:" + str(parameters["W1"].shape))
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float64", [2, 90000])
    
    z3 = forward_propagation_for_predict(X, parameters)
    print("z3 shape:" + str(z3.shape))
    p = tf.argmax(z3, axis=0)
    print("p shape:" + str(p.shape))
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
    return prediction

def forward_propagation_for_predict(X, parameters):
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    return Z3
        
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print("xx:" + str(xx.shape))
    print("yy:" + str(yy.shape))
    xx = xx.astype('float32')
    yy = yy.astype('float32')
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    print("Z:" + str(Z.shape))
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('Working Capital / Total Assets')
    plt.xlabel('Net Income / Total Assets')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()

#params = model(X_train.T, Y_train_hot, X_test.T, Y_test_hot)

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s
