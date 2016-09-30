import math
import random
import numpy as np

def RandomWeights(l0_size, l1_size, seed=1):
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
