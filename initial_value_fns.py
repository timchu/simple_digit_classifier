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
  return W

def toArr(i, lenArr=10):
  arr = np.array([0 for _ in range(lenArr)])
  arr[i] = 1
  return arr

# input is a list of size 785, x = first 784, digit corresp to y = last num.
def getXY(inpt):
  x = inpt[:-1]
  yVal = inpt[-1]
  return (np.array(x), toArr(yVal))
