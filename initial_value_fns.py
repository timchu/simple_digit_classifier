import math
import random
import numpy as np

def initial_synapses(layer_sizes, random_seed=0):
  return [RandomWeights(layer_sizes[i], layer_sizes[i+1], random_seed) for i in range(len(layer_sizes) - 1)]

def initial_zero_synapses(layer_sizes):
  return [np.zeros((layer_sizes[i] + 1, layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]

def initial_synapse_and_steps(layer_sizes, random_seed=0):
  return (initial_synapses(layer_sizes, random_seed), initial_zero_synapses(layer_sizes))

def RandomWeights(l0_size, l1_size, seed=0):
  random.seed(seed)
  x = l0_size
  y = l1_size
  b = math.sqrt(6)/math.sqrt(x + y)
  W = [[0 for _ in range(y)] for _ in range(x + 1)]
  for i in range(x):
    for j in range(y):
      # set weight
      W[i][j] = (random.random() - 1) * b
    #set bias
    W[x][j] = 0
  return np.array(W)
