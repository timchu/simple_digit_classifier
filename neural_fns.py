import numpy as np
import math

def dt(a,b,c="default"):
    if c == "default":
        return np.dot(a,b)
    return np.dot(a,np.dot(b,c))

# append b to the end all rows in a
# a :: 2D Numpy Array
def ap(a,b):
  b_col = b*np.ones((len(a), 1))
  return np.hstack([a, b_col])

def rec(x):
    return 1/float(x)

""" Specification:
        Neuron_type \in {"sigmoid", "softmax"}
        input_arr is a vector <a_1, ... a_n>, nonLin takes this and transforms it into <g_1(a_1), 
        I'm not liking softmax
"""

# return the nonlinearized layer
def nonlin(neuron_type, input_arr):
    if neuron_type == "sigmoid":
        return sigmoid(input_arr)
    elif neuron_type == "relu":
        return relu(input_arr)
    elif neuron_type == "softmax":
        return softmax(input_arr)
    else:
        print("You didn't specify a correct neuron")

# neurons (input = an entire layer of pre-linearized values).
def sigmoid(x_layer):
    new_layer_list = [rec(1+math.exp(-x)) for x in x_layer]
    return np.array(new_layer_list)
def relu(x_layer):
    new_layer_list = [max(0, x) for x in x_layer]
    return np.array(new_layer_list)
def softmax(x_layer):
    sum_exp = 0
    softmax = np.zeros(len(x_layer))
    for x in x_layer:
      sum_exp += math.exp(x)
    return [math.exp(x)/sum_exp for x in x_layer]

# Utility functions to calculate the derivative. Can be put into own file.

# Returns the matrix corresponding to the linear transform taking dInput into dOutput.
#   Comment: input_arr is: linear vars underlying each neuron in a given layer, before applying the non-linearizer
def dNonlin_dInput(neuron_type, input_arr):
    if neuron_type == "sigmoid":
        return dSigmoid_dInput(input_arr)
    elif neuron_type == "relu":
        return dRelu_dInput(input_arr)
    elif neuron_type == "softmax":
        return dSoftmax_dInput(input_arr)
    else:
        print("You didn't specify a correct neuron")

        # Something about softmax is diff from the other two... rely on other neuron linear values

# Returns the matrix corresponding to the linear transform taking dInput into dOutput.
# Not used.
def dSigmoid_dInput(input_arr):
    vals = sigmoid(input_arr)
    n = len(input_arr)
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i][i] = vals[i]*(1-vals[i])
    return matrix

# Also not used.
def dRelu_dInput(input_arr):
    n = len(input_arr)
    matrix = np.zeros((n,n))
    for i in range(n):
        if input_arr > 0:
            matrix[i][i] = 1
        else:
            matrix[i][i] = 0
    return matrix

def dSoftmax_dInput(input_arr):
    vals = softmax(input_arr)
    n = len(input_arr)
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                #check derivative to make sure dimensions right
                matrix[i][j] = vals[j]*(1 - vals[i])
            else:
                matrix[i][j] = vals[j]*(-vals[i])
    return matrix

# Evaluate loss
def Loss(y, input_arr):
  return crossEntropy(y, input_arr)

def Logrec(x):
  return math.log(rec(x))

def crossEntropy(y, input_arr):
  # return dt(y, np.vectorize(Logrec)(input_arr))
  sum = 0
  for i in range(len(y)):
    if y[i] == 0:
      continue
    sum += y[i] * Logrec(input_arr[i])
  return sum

def sum_batch_loss(batch_target_y, batch_zs):
    return sum([Loss(t_y, z) for t_y, z in zip(batch_target_y, batch_zs[-1])])

# returns a column vector corresponding to the gradient of loss in terms of variables.
def dLoss_dInput(y, input_arr):
    return dCrossEntropy_dInput(y, input_arr)

def dCrossEntropy_dInput(y, input_arr):
    if len(input_arr) != len(y):
        print("cross entropy received two distributions of unequal length.")
        return -1
    n = len(y)
    grad = np.zeros(n)
    for i in range(n):
        grad[i]=y[i]*rec(input_arr[i])
    grad.shape = (n, 1) #transposes our gradient to be column vector.
    return grad
