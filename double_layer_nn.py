import numpy as np
import feedforward as ff
import backprop as bp
import math
import initial_value_fns as init
import scrape_data as sd
import neural_fns as nf

"""" 784, 100, 10 neural network on one example."""

#Initiailzation of Hyperparameters
input_size = 784
l1_size = 100 #layer_1 size
l2_size = 10

# Initialiation of parameters
syn0 = init.RandomWeights(input_size, l1_size)
syn1 = init.RandomWeights(l1_size, l2_size)
syns = [syn0, syn1]

# Initialization of optimization parameters
batch_size = 100
epochs = 8 
learning_rate = 0.1

num_train_data = 3000
train_file = "digitstrain.txt"
# test_f = "digitstest.txt"

def train(training_file, epochs, bsize, syns):
    losses = []
    training_batches_X_y  = sd.getBatches_X_y(training_file, bsize)

    for i in range(epochs):
      print("===epoch====")
      sum_losses = 0
      for (batch_X, batch_target_y) in training_batches_X_y:
        (batch_ys, batch_zs) = ff.ff(batch_X, syns, ["sigmoid", "softmax"])
        steps =  bp.one_bp_step(learning_rate, batch_target_y, batch_ys, batch_zs, syns)
        syns[0] += steps[0]
        syns[1] += steps[1]
        sum_losses += nf.sum_batch_loss(batch_target_y, batch_zs) 
      losses += [sum_losses/float(num_train_data)]
    print(losses)

train(train_file, epochs, batch_size, syns)
