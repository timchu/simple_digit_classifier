from random import shuffle
from random import Random
import numpy as np
# Takes a comma delineated string into a list ofto a list of floats and one int

# input is a list of size 785, x = first 784, digit corresp to y = last num.

def toList(i, lenArr=10):
  arr = [0 for _ in range(lenArr)]
  arr[i] = 1
  return arr

def getXYSingle(inpt):
  x = inpt[:-1]
  yVal = inpt[-1]
  return (x, toList(yVal))

# input is a list of N by 785.
# the result is two matrices: X per row. and a column of y
def getXY(data):
  X = [[] for _ in data]
  y = [[] for _ in data]
  for i in range(len(data)):
    (X[i], y[i]) = getXYSingle(data[i])
  return (np.array(X), np.array(y))

def lineToList(line):
    ls = line.split(",")
    ls = [float(x) for x in ls]
    ls[-1] = int(ls[-1]) # Set the last element to a list
    return ls

# randomizes list
def getData(fileName):
    infile = open(fileName)
    lst = infile.readlines()
    data_lists = [lineToList(line) for line in lst]
    Random(4).shuffle(data_lists)
    return data_lists

def Batches_X_y(td, bsize):
    batches = [td[x:x+bsize] for x in range(0, len(td), bsize)] # batch
    return [getXY(batch) for batch in batches] # [(batch_X, batch_target_y)]

def getBatches_X_y(fileName, bsize):
 return Batches_X_y(getData(fileName), bsize)
