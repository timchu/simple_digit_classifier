import numpy as np
import feedforward as ff
import backprop as bp
import math

# returns 

#Discovering dLoss_dLin(k) after we've calculated relevant neurons (and linear pieces underlying said neurons) 
#Requires having a neural network architecture;

# lin3 = the linear variables underlying each neuron on layer 3.
# layer3 = value of each neuron at said layer.
# Each dLoss_dx is a column vector of size len(x) by 1.

# This is a single pass iteration with a single example. layer2 is a single matrix

# TEST: does this code actually  work?
# No biases added in yet... that's not good! Add the 1 to the x. Calculation of linear things is done through feed-forward.

"""" 3, 4, 2 neural network on one example."""

#Initiailzation of weights
syn0 = np.array([[1, -1, -1, 1], [1, -1, -1, 0.4], [1, -0.2, 0.3, 0.4], [1, -0.2, -1, 1]]) #last row is the biases.
syn1 = np.array([[0.1,0.2],[0.3,-0.4],[-0.5,0.6],[0.7,-0.8],[-1, 1]])

# Data from 1 example
X = np.array([1, 1, 1])
y = np.array([0,1])

syns = [syn0, syn1]

for i in range(20000):
  (lins, layers) = ff.feedforward(X, [syn0, syn1], ["sigmoid", "softmax"])
  steps =  bp.one_bp_step(0.1, X, y, lins, layers, syns)
  syn0 += steps[0]
  syn1 += steps[1]

print syn0
print syn1
print layers
