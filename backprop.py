import numpy as np
import math
from neural_fns import *
import sys
import feedforward as ff

# NOTATION: y is the linear w^Tx + b underlying our neurons at a layer, z is the output of the neurons in a layer.

# Returns the gradient of loss for each synapse between layers
# TYPES:z is a 2D array of zs, where each row is a z in our batch.
# TYPES:batch_dLoss_dyAbove is a 2D array of Zs where each column is dLoss/dyAbove
def Grad(batch_z, batch_dLoss_dyAbove):
  l = ap(batch_z, 1) #Append a column of 1s to batch_z
  return np.dot(l.T, batch_dLoss_dyAbove.T)

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
def steps_bp_batch(l_rate, batch_target_y, batch_ys, batch_zs, syns, mom, prev_steps, masks):
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
      # Mask result of backprop at each step.
      batch_dLoss_dys[k] = (masks[k]).T*batch_bp1(batch_dLoss_dys[k+1], batch_ys[k], batch_zs[k], syns[k], "sigmoid")
      # The masking of batch_zs in feedforward should prevent us from having to mask anything in backprop for grads.
      grads[k] = Grad(batch_zs[k], batch_dLoss_dys[k+1])
    grads[0] = Grad(batch_zs[0], batch_dLoss_dys[1])
    batch_size = len(batch_ys)
    return [(l_rate * grad + mom * prev_step)/float(batch_size) for (grad, prev_step) in zip(grads, prev_steps)]

# pointwise sum
def ptSum(ls1, ls2):
  if len(ls1) != len(ls2):
    print("WARNING LENGTH OF POINT SUM ITEMS NOT EQUAL")
  return [l1 + l2 for (l1, l2) in zip(ls1, ls2)]

def generateMasks(layer_sizes,d_rate):
  # Don't generate a mask on the last layer
  masks = [np.array([np.random.binomial(1,1-d_rate,layer_sizes[i])]) for i in range(len(layer_sizes)-1)]
  return masks

def syns_steps_from_batches(train_batches_X_y, syns, neural_types, l_rate, mom, prev_steps, d_rate, layer_sizes):
  grads = []
  training_perf = 0
  p_steps = prev_steps
  for (batch_X, batch_target_y) in train_batches_X_y:
    masks = generateMasks(layer_sizes, d_rate)
    (batch_ys, batch_zs) = ff.ff(batch_X, syns, neural_types, masks)
    steps_from_batch = steps_bp_batch(l_rate, batch_target_y, batch_ys, batch_zs, syns, mom, prev_steps, masks)

    # Update synapses for each batch
    syns = ptSum(syns, steps_from_batch)

    # Update previous step for each batch
    p_steps = steps_from_batch
  # Return the synapses and last pervious step
  return (syns, steps_from_batch)
