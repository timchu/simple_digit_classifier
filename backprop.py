import numpy as np
import math
from neural_fns import *

# Removes biases from the synapse matrix (the bottom row)
def remove_bias(syn):
  return np.delete(syn, len(syn)-1, 0)

# Given dLoss_dLin and neural_type of the current layer, find dLoss_dLin of the previous layer
def bp1(dLoss_dLin, prev_lin_layer, syn, neural_type):
  syn = remove_bias(syn)
  dPrevLayer_dPrevLin = dNonlin_dInput(neural_type, prev_lin_layer)
  return dt(dPrevLayer_dPrevLin, syn, dLoss_dLin)

def T(layer):
  layer = layer[np.newaxis]
  return layer.T

# Returns the gradient of loss, w.r.t. the previous layer's linear underpinning.
# Spec:layer is a 1D row, dLoss_dLin is a 2d column
def Grad(layer, dLoss_dLinAbove):
  l = ap(layer, 1)
  return np.dot(T(l), dLoss_dLinAbove.T)

def one_bp_step(learning_rate, X, y, lins, layers, syns):
    # len(syns) = n = depth of NN
    # len(lins) = n+1 = number of layers
    n =len(syns)

    # initialize vectors to return
    dLoss_dLins = [-1000 for _ in range(n+1)]
    grads = [-1000 for _ in range(n)]
    # Set dLoss_dTopLin. Own method?
    dLoss_dTopLayer = dLoss_dInput(y, layers[n])
    dTopLayer_dTopLin = dNonlin_dInput("softmax", lins[n])
    dLoss_dTopLin = dt(dTopLayer_dTopLin, dLoss_dTopLayer)
    dLoss_dLins[n] = dLoss_dTopLin

    # range from n-1 to 1.
    for i in range(n-1):
      level = (n-1) - i
      dLoss_dLins[level] = bp1(dLoss_dLins[level+1], lins[level], syns[level], "sigmoid")
      grads[level] = Grad(layers[level], dLoss_dLins[level+1])
    grads[0] = Grad(layers[0], dLoss_dLins[1])
    return [learning_rate * g for g in grads]

    #
""" def one_bp_step(learning_rate, X, y, lins, layers, syns):

    lin1 = lins[1]
    layer1 = layers[1]

    lin2 = lins[2]
    layer2 = layers[2]
    syn0 = syns[0]
    syn1 = syns[1]

    # Calculating gradient of Loss with respect to the linear variable underlying each neuron in layer2
    dLoss_dLayer2 = dLoss_dInput(y, layer2)
    dLayer2_dLin2 = dNonlin_dInput("softmax", lin2)
    dLoss_dLin2 = np.dot(dLayer2_dLin2, dLoss_dLayer2)

    # WARNING: When we have more than one example, this line will change. We use this since .T in Python SUCKS!!! TODO
    layer1T = layer1
    layer1T.shape = (len(layer1), 1)
    gradW1 = np.dot(layer1T, dLoss_dLin2.T)

    # Calculating gradient of Loss with respect to the linear variable underlying each neuron in layer1
    dLin2_dLayer1 = np.delete(syn1, len(syn1)-1, 0)
    dLayer1_dLin1 = dNonlin_dInput("sigmoid", lin1)
    dLoss_dLin1 = np.dot(dLayer1_dLin1, np.dot(dLin2_dLayer1, dLoss_dLin2))

    # WARNING: When we have more than one example, this line will change. We use XT since .T in numpy sucks... TODO
    XT = X
    XT.shape = (len(X), 1)
    gradW0 = np.dot(XT, dLoss_dLin1.T)

    learning_rate = 0.1
    return [learning_rate*gradW0, learning_rate*gradW1] """
