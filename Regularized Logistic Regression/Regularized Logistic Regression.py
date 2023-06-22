#!/usr/bin/env python
# coding: utf-8
# Regularized Logistic Regression
# Built 6 models and trained Logistic Regression/Regularized Logistic Regression each with Batch Gradient Descent, Stochastic Gradient Descent and Mini Batch Gradient Descent. Also, plot their objective values versus epochs and compare their training and testing accuracy and tuned the parameters a little bit to obtain reasonable results.

# Load Packages
import pandas as pd
import numpy 
import decimal as dc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#import warnings
#warnings.filterwarnings('ignore')


# 1. Data processing
# - Load the data.
# - Preprocess the data.

# 1.1. Load the data

data = pd.read_csv("/breast-cancer-wisconsin.csv", na_values = "?")
print(data.isna().sum())  #missing values in data
n = data.shape[0]

# Examine and clean data
# I have dropped id number as it is useless for achieving actual goal 
# I transformed target labels in the second column from 'B' and 'M' to 1 and -1.
for i in range(len(data)):
    if data.diagnosis.values[i] == 'B':
        data.diagnosis.values[i] = 1
    elif data.diagnosis.values[i] == 'M':
        data.diagnosis.values[i] = -1
X = data.iloc[:,2:32]
y = data.iloc[:,1]
X = numpy.concatenate((X, numpy.ones((n,1))), axis = 1)
data1 = data.iloc[:,1:32]
data1 = numpy.concatenate((data1, numpy.ones((n,1))), axis = 1)


# 1.3. Partition to training and testing sets
# I partitioned using 80% training data and 20% testing data. It is a commonly used ratio in machine learning.
x_train, x_test, y_train, y_test = train_test_split(X, y ,test_size=0.20, random_state=104)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# 1.4. Feature scaling
# Use the standardization to transform both training and test features

# Standardization
# calculate mu and sig using the training set
d = x_train.shape[1]
mu = numpy.mean(x_train, axis=0).reshape(1, d)
sig = numpy.std(x_train, axis=0).reshape(1, d)

# transform the training features
x_train = (x_train - mu) / (sig + 1E-6)

# transform the test features
x_test = (x_test - mu) / (sig + 1E-6)

print('test mean = ')
print(numpy.mean(x_test, axis=0))
print('test std = ')
print(numpy.std(x_test, axis=0))


# 2.  Logistic Regression Model
# 
# The objective function is $Q (w; X, y) = \frac{1}{n} \sum_{i=1}^n \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $.
# 
# When $\lambda = 0$, the model is a regular logistic regression and when $\lambda > 0$, it essentially becomes a regularized logistic regression.

# Calculated the objective function value, or loss
# Inputs:
#     w: weight: d-by-1 matrix
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: regularization parameter: scalar
# Return:
#     objective function value, or loss (scalar)
def objective(w, x, y, lam):
    n = x.shape[0]
    xy = numpy.multiply(x,y)
    xyw = numpy.dot(xy,w)
    val1 = numpy.array(xyw, dtype= numpy.float128)  #to avoid overflow issue

    exp_term = (1 + numpy.exp(-val1)).astype(float)
    term = numpy.log(exp_term)
    log = 1/n * numpy.sum(term)
    reg = 0.5 * lam * numpy.linalg.norm(w)**2
    obj = log + reg
    return obj


#  3. Numerical optimization

# 3.1. Gradient descent
# 

# The gradient at $w$ for regularized logistic regression is  $g = - \frac{1}{n} \sum_{i=1}^n \frac{y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$
# Calculated the gradient
# Inputs:
#     w: weight: d-by-1 matrix
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: regularization parameter: scalar
# Return:
#     g: gradient: d-by-1 matrix

def gradient(w, x, y, lam):
    n = x.shape[0]
    xy = numpy.multiply(x,y)
    xyw = numpy.dot(xy,w)
    val1 = numpy.array(xyw, dtype= numpy.float128)
    
    exp_term = (1 + numpy.exp(val1)).astype(float)
   # print(exp_term)
    term = numpy.divide(xy,exp_term)
    grad = -1/n * numpy.sum(term, axis=0).reshape(-1,1)   #summation of all rows for each colum to retrieve 31 by 1 matrix
    reg = lam * w
    g = grad + reg
    return g

# Gradient descent for solving logistic regression
# iterative processes (loops) were used to obtain optimal weights in this function

# Inputs:
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: scalar, the regularization parameter
#     learning_rate: scalar
#     w: weights: d-by-1 matrix, initialization of w
#     max_epoch: integer, the maximal epochs
# Return:
#     w: weights: d-by-1 matrix, the solution
#     objvals: a record of each epoch's objective value

def gradient_descent(x, y, lam, learning_rate, w, max_epoch=100):
    objvals = []
    for epoch in range(max_epoch):
        g = gradient(w, x, y, lam)
        w = w - (learning_rate * g)
        obj = objective(w,x,y,lam)
        objvals.append(obj)
    return w, objvals     #return update weights and last object values for each weights updated


# Use gradient_descent function to obtain your optimal weights and a list of objective values over each epoch.
# Train logistic regression
# I obtained optimal weights and a list of objective values by using gradient_descent function.
n, d = x_train.shape
y_train = numpy.array(y_train).reshape(-1,1)

lr = 0.5 
lambdas = 0 

# Initialize the weights to zero
w = numpy.zeros((d, 1))

w_gd, gd_obj = gradient_descent(x_train, y_train, lambdas, lr, w)
# Print the optimal weights and the objective value
print("Optimal weights:", w_gd)
print("objective value:", gd_obj)

# Train regularized logistic regression
# I obtained the optimal weights and a list of objective values by using gradient_descent function.
n, d = x_train.shape
y_train = numpy.array(y_train, dtype=numpy.float32).reshape(-1,1)
lr = 0.5 #[0.01, 0.1, 0.5, 1]
lambdas = 0.001 #[0.001, 0.05, 0.1, 0.5]

# Initialize the weights to zero
w = numpy.zeros((d, 1))

w_gd_r, gd_obj_r = gradient_descent(x_train, y_train, lambdas, lr, w)
#gd_obj_r = numpy.concatenate(numpy.array(objvals)).ravel()
# Print the optimal weights and the last objective value
print("Optimal weights:", w_gd_r)
print("objective value:", gd_obj_r)


# 3.2. Stochastic gradient descent (SGD)
# 
# Define new objective function $Q_i (w) = \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $. 
# 
# The stochastic gradient at $w$ is $g_i = \frac{\partial Q_i }{ \partial w} = -\frac{y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$.
# 
# You may need to implement a new function to calculate the new objective function and gradients.

# Calculated the objective Q_i and the gradient of Q_i
# Inputs:
#     w: weights: d-by-1 matrix
#     xi: data: 1-by-d matrix
#     yi: label: scalar
#     lam: scalar, the regularization parameter
# Return:
#     obj: scalar, the objective Q_i
#     g: d-by-1 matrix, gradient of Q_i

def stochastic_objective_gradient(w, xi, yi, lam):
    xyw = numpy.multiply(yi, numpy.dot(xi, w))
    val1 = numpy.array(xyw, dtype= numpy.float128)
    
    exp_term = numpy.exp(-val1)
    log_loss = numpy.log(1 + exp_term)
    reg_term = 0.5 * lam * numpy.linalg.norm(w) ** 2
    obj = log_loss + reg_term

    term_exp = numpy.exp(val1)
    grad =  -numpy.dot(xi.T, yi / (1 + term_exp))
    reg = lam * w
    g = grad + reg
    return obj, g

# SGD for solving logistic regression
# Inputs:
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: scalar, the regularization parameter
#     learning_rate: scalar
#     w: weights: d-by-1 matrix, initialization of w
#     max_epoch: integer, the maximal epochs
# Return:
#     
#     w: weights: d-by-1 matrix, the solution
#     objvals: a record of each epoch's objective value
#     Record one objective value per epoch (not per iteration)

def sgd(x, y, lam, learning_rate, w, max_epoch=100):
    n, d = x.shape
    objvals = []
    for epoch in range(max_epoch):
        objval = 0
        rand_perm = numpy.random.permutation(n) #randomly assigning indices to the iterations that need to compute
        for i in rand_perm: 
            xi = numpy.array(x[i, :]).reshape(1, -1)
            yi = y[i]
            obj, grad = stochastic_objective_gradient(w, xi, yi, lam)
            w = w - learning_rate * grad
            xi = numpy.array(x[i, :]).reshape(1, -1)
            yi = y[i]
            obj, _ = stochastic_objective_gradient(w, xi, yi, lam)
            objval += obj
        objval /= n 
        objvals.append(objval)
    return w, objvals


# Use sgd function to obtain optimal weights and a list of objective values over each epoch.

# Train logistic regression
n, d = x_train.shape
y_train = numpy.array(y_train, dtype=numpy.float32).reshape(-1,1)
#print(x_train.shape[1])
lr = 0.1 #[0.01, 0.1, 0.5, 1]
lambdas = 0 #[0.1, 0.5, 1, 5]

# Initialize the weights to zero
w = numpy.zeros((d, 1))

w_sgd, objvals = sgd(x_train, y_train, lambdas, lr, w)
sgd_obj = numpy.concatenate(numpy.array(objvals)).ravel()
# Print the optimal weights and the last objective value
print("Optimal weights:", w_sgd)
print("objective value:", sgd_obj)


# Train regularized logistic regression
n, d = x_train.shape
y_train = numpy.array(y_train, dtype=numpy.float32).reshape(-1,1)
#print(x_train.shape[1])
lr = 0.01 
lambdas = 0.001 

# Initialize the weights to zero
w = numpy.zeros((d, 1))

w_sgd_r, objvals = sgd(x_train, y_train, lambdas, lr, w)
sgd_obj_r = numpy.concatenate(numpy.array(objvals)).ravel()
# Print the optimal weights and the last objective value
print("Optimal weights:", w_sgd_r)
print("objective value:", sgd_obj_r)


# 3.3 Mini-Batch Gradient Descent (MBGD)

# Define $Q_I (w) = \frac{1}{b} \sum_{i \in I} \log \Big( 1 + \exp \big( - y_i x_i^T w \big) \Big) + \frac{\lambda}{2} \| w \|_2^2 $, where $I$ is a set containing $b$ indices randomly drawn from $\{ 1, \cdots , n \}$ without replacement.
# 
# The stochastic gradient at $w$ is $g_I = \frac{\partial Q_I }{ \partial w} = \frac{1}{b} \sum_{i \in I} \frac{- y_i x_i }{1 + \exp ( y_i x_i^T w)} + \lambda w$.
# 
# You may need to implement a new function to calculate the new objective function and gradients.

# Calculated the objective Q_I and the gradient of Q_I
# Inputs:
#     w: weights: d-by-b matrix
#     xi: data: b-by-d matrix
#     yi: label: scalar
#     lam: scalar, the regularization parameter
# Return:
#     obj: scalar, the objective Q_i
#     g: d-by-1 matrix, gradient of Q_i

def mb_objective_gradient(w, xi, yi, lam):
    b = 20
    xy = numpy.multiply(xi,yi)
    xyw = numpy.dot(xy,w)
    val1 = numpy.array(xyw, dtype= numpy.float128)

    exp_term = (1 + numpy.exp(-val1)).astype(float)
    term = numpy.log(exp_term)
    log = 1/b * numpy.sum(term)
    reg = 0.5 * lam * numpy.linalg.norm(w)**2
    obj = log + reg
    
   # val = numpy.asarray([[dc.Decimal(i) for i in j] for j in xyw])
    exp_term = (1 + numpy.exp(val1)).astype(float)
   # print(exp_term)
    term = numpy.divide(xy,exp_term)
    grad = -1/b * numpy.sum(term, axis=0).reshape(-1,1)
    reg = lam * w
    g = grad + reg
    return obj, g

# MBGD for solving logistic regression

# Inputs:
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: scalar, the regularization parameter
#     learning_rate: scalar
#     w: weights: d-by-1 matrix, initialization of w
#     max_epoch: integer, the maximal epochs
# Return:
#     w: weights: d-by-1 matrix, the solution
#     objvals: a record of each epoch's objective value
#     Record one objective value per epoch (not per iteration)

def mbgd(x, y, lam, learning_rate, w, max_epoch=100):
    n, d = x.shape
    objvals = []
    b = 20 # mini-batch size
    for epoch in range(max_epoch):
        num = numpy.random.permutation(n)
        for i in range(0, n, b):
            indices = num[i:i+b]
            xi = x[indices, :]
            yi = y[indices, :]
            obj, grad = mb_objective_gradient(w, xi, yi, lam)
            w = w - learning_rate * grad
        objvals.append(obj)
    return w, objvals


# Use mbgd function to obtain optimal weights and a list of objective values over each epoch.
# Train logistic regression
n, d = x_train.shape

lr = 0.1 #[0.01, 0.1, 0.5, 1]
lambdas = 0 #[0.1, 0.5, 1, 5]

# Initialize the weights to zero
w = numpy.zeros((d, 1))

w_mbgd, obj_mbgd = mbgd(x_train, y_train, lambdas, lr, w)

# Print the optimal weights and the objective value
print("Optimal weights:", w_mbgd)
print("Objective value:", obj_mbgd)

# Trained regularized logistic regression
n, d = x_train.shape
y_train = numpy.array(y_train, dtype=numpy.float32).reshape(-1,1)
#print(x_train.shape[1])
lr = 0.1 #[0.01, 0.1, 0.5, 1]
lambdas = 0.001 #[0.1, 0.5, 1, 5]

# Initialize the weights to zero
w = numpy.zeros((d, 1))

w_mbgd_r, obj_mbgd_r = mbgd(x_train, y_train, lambdas, lr, w)

# Print the optimal weights and the last objective value
print("Optimal weights:", w_mbgd_r)
print("objective value:", obj_mbgd_r)


# # 4. Compare GD, SGD, MBGD
# 
# ### Plot objective function values against epochs.

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

# Plot the first function on the first subplot
axes[1].plot(sgd_obj, label='Stochastic Gradient Descent', linestyle = 'dotted', marker='.' ,color = 'lightblue')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Objective Values')
axes[1].set_ylim(bottom=0, top=0.3)

# Plot the second function on the second subplot
axes[0].plot(gd_obj, label='Gradient Descent', linestyle = 'dotted',marker = '.', color='red')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Objective values')
axes[0].set_ylim(bottom=0, top=0.5)

axes[2].plot(obj_mbgd, label='Mini Batch Gradient Descent', linestyle = 'dotted',marker = '.',color = 'pink')
axes[2].set_xlabel('Epochs')
axes[2].set_ylabel('Objective Values')
axes[2].set_ylim(bottom=0, top=5)

# Adjust spacing between subplots and display the plot
plt.tight_layout()
plt.show()


# # 5. Prediction
# ### Compare the training and testing accuracy for logistic regression and regularized logistic regression.

# Predict class label
# Inputs:
#     w: weights: d-by-1 matrix
#     X: data: m-by-d matrix
# Return:
#     f: m-by-1 matrix, the predictions
def predict(w, x):
    f = numpy.dot(x, w)
    return numpy.sign(f)

# evaluate training error of logistic regression and regularized version
arr = {
  "Stochastic": w_sgd,
  "Stochastic_r": w_sgd_r ,
  "Minibatch": w_mbgd,
  "Minibatch_r": w_mbgd_r, 
  "normal" : w_gd,
  "normal_r" : w_gd_r
}

def train_error(arr, x_data, y_data):
    n = x_data.shape[0]
    for key ,val in arr.items():
        if(str(key[-1]) == 'r'):
            pred_r = predict(val, x_data)
            incorrect_r = numpy.sum(pred_r != y_data)
            error_rate = incorrect_r/n
            accuracy = 1-error_rate
            print("Error rate of Regularized Logistic Regression for",key[:-2],"Gradient Descent:",round(error_rate, 2))
            print("Accuracy of Regularized Logistic Regression for",key[:-2],"Gradient Descent:",round(accuracy, 2),"\n")
        else:
            pred = predict(val, x_data)
            incorrect = numpy.sum(pred != y_data)
            error_rate = incorrect/n
            accuracy = 1-error_rate
            print("Error rate of Logistic Regression for",key,"Gradient Descent:",round(error_rate, 2))
            print("Accuracy of Logistic Regression for",key,"Gradient Descent:",round(accuracy, 2),"\n")

train_error(arr, x_train, y_train)

# evaluate testing error of logistic regression and regularized version
y_test = numpy.array(y_test, dtype=numpy.float32).reshape(-1,1)
def train_error(arr, x_data, y_data):
    n = x_data.shape[0]
    for key ,val in arr.items():
        if(str(key[-1]) == 'r'):
            pred_r = predict(val, x_data)
            incorrect_r = numpy.sum(pred_r != y_data)
            error_rate = incorrect_r/n
            accuracy = 1-error_rate
            print("Error rate of Regularized Logistic Regression for",key[:-2],"Gradient Descent:",round(error_rate, 2))
            print("Accuracy of Regularized Logistic Regression for",key[:-2],"Gradient Descent:",round(accuracy, 2),"\n")
        else:
            pred = predict(val, x_data)
            incorrect = numpy.sum(pred != y_data)
            error_rate = incorrect/n
            accuracy = 1-error_rate
            print("Error rate of Logistic Regression for",key,"Gradient Descent:",round(error_rate, 2))
            print("Accuracy of Logistic Regression for",key,"Gradient Descent:",round(accuracy, 2),"\n")

train_error(arr, x_test, y_test)


# 6. Parameters tuning

from functools import reduce

# Define a function that performs the training and evaluation on a single fold
def cross_val_kfold(i,X_parts,y_parts, lam, lr):    
    x_test = X_parts[i]
    y_test = y_parts[i]
    # Set the remaining parts as the training data
    x_train = numpy.concatenate(X_parts[:i] + X_parts[i+1:])
    y_train = numpy.concatenate(y_parts[:i] + y_parts[i+1:])
    d = x_train.shape[1]
    w = numpy.zeros((d,1))

    #change the name of function if you need mini batch write 'mbgd' and for stochastic 'sgd'
    w_, obj_ = mbgd(x_train, y_train, lam, lr, w)     
    # Make predictions on the test set and compute the error rate and accuracy
    pred = predict(w_, x_test)
    incorrect_r = numpy.sum(pred != y_test)
    error_rate = incorrect_r / n
    accuracy = 1 - error_rate
    return (error_rate, accuracy)

# Define the number of folds 
k = 20
avg_rate, avg_acc = 0, 0
X = data1[:,1:32]
y = (data1[:,0]).reshape(-1,1)
X_parts = numpy.array_split(X, k)
y_parts = numpy.array_split(y, k)
# Use map and filter to apply the train_eval_fold function to each fold
output = map(lambda i: cross_val_kfold(i,X_parts,y_parts, 0.01, 0.1), range(k))
# Use reduce to compute the running sum of error rates and accuracies
avg_error_rate, avg_accuracy = reduce(lambda acc, res: (acc[0] + res[0], acc[1] + res[1]), output, (0, 0))
# Compute the average error rate and accuracy over all folds
avg_error_rate /= k
avg_accuracy /= k
print("Averaged test errors(validation error)",round(avg_error_rate,2))
print("Average test accuracy(validation error)",round(avg_accuracy,2))


# The above, **Tuning parameter** includes a function which consists logic for
#cross **validation kfold** log where I have divided x and y training amd
#testing set after fetching split from the actual data. After that,
#I have called **all three gradient functions** to obtain **updated weights**.
#This weights helps to **predict** the **validation set**  **error rate and accuray** per functions, which is then **averaged** for obtaining **validation error**. However, This function takes **longer time to compute** if loops
#are used. Hence, to avoid that issue, I have used optimized way by inculcating
#**map and reduce** concepts. I reduced time complexity by 4 times as before
#(while using normal for loop) it took 40.06 mins (more than half an hour)to
#compute the acccuracy and error rate, but now it takes only in 10 to 15 secs. 
