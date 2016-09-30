import numpy as np
import math
from neural_fns import *
import sys

# Returns the gradient of loss, w.r.t. the previous z's year underpinning.
# TYPES:z is a 2D array of z, dLoss_dy is a 2d column
def Grad(zBatch, dLoss_dyAbove):
  l = ap(zBatch, 1) #Append a column of 1s to zBatch
  return np.dot(l.T, dLoss_dyAbove.T)

# Removes biases from the synapse matrix (the bottom row)
def remove_bias(syn):
  return np.delete(syn, len(syn)-1, 0)

# Find batch_dL_dprevY
def batch_bp1(batch_dL_dy, batch_prev_y, batch_prev_z, syn, neural_type):
    y = batch_prev_y
    z = batch_prev_z
    syn = remove_bias(syn)
    batch_dL_dprevZ = dt(syn, batch_dL_dy)
    if (neural_type == "sigmoid"):
        return (z*(1-z)).T*batch_dL_dprevZ
    print("Unknown neuron type")
    sys.exit()

# Should return a column: gradient of loss w.r.t. top layer y
def dLoss_dTopY(y, top_z, top_y, neural_type):
  dLoss_dTopz = dLoss_dInput(y, top_z)
  dTopz_dTopY = dNonlin_dInput(neural_type, top_y)
  return dt(dTopz_dTopY, dLoss_dTopz)

def batch_dLoss_dTopY(batch_target_y, top_z_batch, top_y_batch, neural_type="softmax"):
    bsize = len(batch_target_y)
    if len(batch_target_y) != len(top_y_batch):
        print("num batches don't match up in batch dLoss_dTopY")
    cols = [dLoss_dTopY(batch_target_y[i], top_z_batch[i], top_y_batch[i], neural_type) for i in range(bsize)]
    # Return columns as a numpy matrix
    return np.hstack(cols)

# X, y are batches, row = single example.
def one_bp_step(learning_rate, batch_target_y, batch_ys, batch_zs, syns):
    # len(syns) = n = depth of NN
    # len(y) = n+1 = number of z
    n = len(syns)

    # initialize vectors to return
    batch_dLoss_dys = ["Nothing" for _ in range(n+1)]
    grads = ["Nothing" for _ in range(n)]

    # Set batch_dLoss_dTopY
    top_z_batch = batch_zs[n]
    top_y_batch = batch_ys[n]
    batch_dLoss_dys[n] = batch_dLoss_dTopY(batch_target_y, top_z_batch, top_y_batch)

    # backprop on k from n-1 to 1.
    for i in range(n-1):
      k = (n-1) - i
      batch_dLoss_dys[k] = batch_bp1(batch_dLoss_dys[k+1], batch_ys[k], batch_zs[k], syns[k], "sigmoid")
      grads[k] = Grad(batch_zs[k], batch_dLoss_dys[k+1])
    grads[0] = Grad(batch_zs[0], batch_dLoss_dys[1])
    return [learning_rate * g for g in grads]
