import numpy as np
import feedforward as ff
import backprop as bp
import math
import initial_value_fns as init
import scrape_data as sd

"""" 784, 100, 10 neural network on one example."""

#Initiailzation of Hyperparameters
input_size = 784
l1_size = 100 #layer_1 size
l2_size = 10

# Initialiation of parameters
syn0 = np.array(init.RandomWeights(input_size, l1_size))
syn1 = np.array(init.RandomWeights(l1_size, l2_size))
epochs = 100
learning_rate = 0.1

data = sd.getData("digitstrain.txt")[::100] #set this back to the whole dataset
for i in range(epochs):
    syns = [syn0, syn1]
    for input_val in data:
        (X, y) = init.getXY(input_val)
        (lins, layers) = ff.feedforward(X, syns, ["sigmoid", "softmax"])
        steps =  bp.one_bp_step(learning_rate, X, y, lins, layers, syns)
        syn0 += steps[0]#/num_samples. 
        syn1 += steps[1]#/num_samples
        if i  == 99:
            print layers[2]
