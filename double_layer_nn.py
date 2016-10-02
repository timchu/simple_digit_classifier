import numpy as np
import feedforward as ff
import backprop as bp
import math
import initial_value_fns as init
import scrape_data as sd
import neural_fns as nf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import matplotlib.pyplot as plt

"""" 784, 100, 10 neural network on one example."""

# Files for training
train_file = "digitstrain.txt"
# test_f = "digitstest.txt"
valid_file = "digitsvalid.txt"

# Extract training data.
train_dat = sd.getData(train_file)
valid_dat = sd.getData(valid_file)

# Functions to generate images. Plotted the weight of each filter as an image.
def f(x):
    if (x > 0):
        return [1-2*x,1-2*x,1-2*x]
    return [1,1+2*x, 1]
def g(x):
    return [x,x,x]
def f1(xs, f):
  return [f(x) for x in xs]
def f2(xss, f):
  return[f1(xs, f) for xs in xss]

# Evaluate performance of neural network.
def getPerf(perf_fn, dat, syns, neural_types, layer_sizes):
  (all_X, all_target_y) = sd.getXY(dat)
  avg_mask = [0.5*np.ones((1,layer_sizes[i])) for i in range(len(layer_sizes)-1)]
  (_, all_zs) = ff.ff(all_X, syns, neural_types, avg_mask)
#  print all_target_y[0]
#  print all_zs[-1][0]
#  print nf.avg_perf(perf_fn, all_target_y, all_zs)
  return nf.avg_perf(perf_fn, all_target_y, all_zs)

# Plot weights of first layer as 100 images
def plot_100_28x28_imgs(syn):
    for i in range(10):
        for j in range(10):
            n = 10*i + j
            im_arr = np.array(syn[:,n][:-1]).reshape(28,28) #exclude biases
            ax5 = plt.subplot2grid((10,10),(i,j))
            im = np.array(f2(im_arr, f))
            plt.imshow(im)
            cur_axes = plt.gca()
            cur_axes.axes.get_xaxis().set_visible(False)
            cur_axes.axes.get_yaxis().set_visible(False)
    plt.show()

valuesTest = dict(
        num_epochs=20
    )

valuesA1 = dict(
        random_seed = 1
    )

valuesA2 = dict(
        random_seed = 0.2
    )

valuesA3 = dict(
        random_seed = 0.3
    )

valuesA4 = dict(
        random_seed = 0.4
    )

valuesA5 = dict(
        random_seed = 0.5
    )
valuesA6 = dict(
        random_seed = 0.6
    )
valuesB = dict(
    perf_fn="meanClass"
    )
valuesD1 = dict(
    l_rate=0.01
    )
valuesD2 = dict(
    l_rate=0.2
    )
valuesD3 = dict(
        mom = 0.1
    )
valuesD4 = dict(
        mom = 0.5
    )
valuesD5 = dict(
        mom = 0.9
    )
valuesE1 = dict(
        l_rate = 0.01,
        mom = 0.5,
        layer_sizes=[784,20,10]
    )
valuesE2 = dict(
        l_rate = 0.01,
        mom = 0.5,
        layer_sizes=[784,100,10]
    )
valuesE3 = dict(
        l_rate = 0.01,
        mom = 0.5,
        layer_sizes=[784,200,10]
    )
valuesE4 = dict(
        l_rate = 0.01,
        mom = 0.5,
        layer_sizes=[784,500,10]
    )
valuesF1 = dict(
        d_rate=0.5
    )
valuesF2 = dict(
        layer_sizes=[784, 200, 10],
        d_rate=0.5
    )
valuesG0 = dict(
        layer_sizes=[784, 300, 10],
        d_rate=0.5
    )
valuesG1 = dict(
        layer_sizes=[784, 200, 10],
        mom = 0.5,
        d_rate=0.5
    )
valuesG2 = dict(
        layer_sizes=[784, 100, 10],
        mom = 0.5,
    )
valuesG3 = dict(
        layer_sizes=[784, 400, 10],
        mom = 0.5,
        l_rate = 0.2,
        d_rate=0.5
    )
valuesG4 = dict(
        layer_sizes=[784, 200, 10],
        mom = 0.5,
        d_rate = 0.5
    )
valuesG5 = dict(
        layer_sizes=[784, 200, 10],
        l_rate = 0.05,
        d_rate=0.5
    )
valuesH1 = dict(
        num_epochs=200,
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        layer_sizes=[784,100,100,10]
    )
valuesH2 = dict(
        num_epochs=200,
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        layer_sizes=[784,100,100,10],
        perf_fn = "meanClass"
    )
valuesH0 = dict(
        layer_sizes=[784, 200, 200, 10],
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        d_rate=0.5
    )
valuesH8 = dict(
        layer_sizes=[784, 100, 100, 10],
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        mom = 0.5,
    )
valuesH9 = dict(
        layer_sizes=[784, 100, 100, 10],
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        mom = 0.7
    )
valuesH3 = dict(
        layer_sizes=[784, 100, 100, 10],
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        mom = 0.5,
        l_rate = 0.2,
    )
valuesH4 = dict(
        layer_sizes=[784, 100, 100, 10],
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        mom = 0.9,
    )
valuesH5 = dict(
        layer_sizes=[784, 200, 200, 10],
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        d_rate = 0.5
    )
valuesH6 = dict(
        layer_sizes=[784, 200, 200, 10],
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        l_rate = 0.05,
        d_rate=0.5
    )
valuesH7 = dict(
        layer_sizes=[784, 100, 100, 10],
        neural_types = ["sigmoid", "sigmoid", "softmax"],
        l_rate = 0.1,
        d_rate=0.5
    )
def train(train_dat, valid_dat, test_dat, layer_sizes=[784,100,10], neural_types=["sigmoid", "softmax"], loss_fn="crossEntropy", perf_fn="crossEntropy", random_seed=1,batch_size=50, num_epochs=200, l_rate=0.1, mom=0.0, d_rate=0.0, num_train_data=3000, ):
  tag = "_layers_"+str(layer_sizes) + "_r_seed_" + str(random_seed) + "_batch_size_"+str(batch_size) + "_num_epoch_"+str(num_epochs) + "_l_rate_"+str(l_rate)+"_momentum_"+str(mom)+"_dropout_rate_" + str(d_rate) + "_perf_fn_"+str(perf_fn)
  # Keep track of synapses and previous step.
  (syns, prev_steps) = init.initial_synapse_and_steps(layer_sizes, random_seed)
  train_batches = sd.Batches_X_y(train_dat, batch_size)

  training_perf_by_epoch = []
  valid_perf_by_epoch = []

  for i in range(num_epochs):
    print("--EPOCH--")
    print(i)
    print(tag)
    # Find update synapse and previous step
    (syns, prev_steps) = bp.syns_steps_from_batches(train_batches, syns, neural_types, l_rate, mom, prev_steps, d_rate, layer_sizes)

    # Find performance of training and validation set.
    training_perf = getPerf(perf_fn, train_dat, syns, neural_types, layer_sizes)
    valid_perf = getPerf(perf_fn, valid_dat, syns, neural_types, layer_sizes)
    print "test"
    print(training_perf)
    print(valid_perf)

    # Add training and validation loss to set
    training_perf_by_epoch.append(training_perf)
    valid_perf_by_epoch.append(valid_perf)
    # Plot image filters of synapses
  plot_100_28x28_imgs(syns[0])

  # Plot losses of traning and validation set.
  plt.plot(range(len(training_perf_by_epoch)), valid_perf_by_epoch)
  plt.plot(range(len(valid_perf_by_epoch)), training_perf_by_epoch)
  plt.ylabel('Average Loss by Epoch for Training Data')
  fig_name='losses'+tag+'.png'
  plt.savefig(fig_name)
  plt.clf()

train(train_dat, valid_dat, "", **valuesTest)
train(train_dat, valid_dat, "", **valuesA1)
train(train_dat, valid_dat, "", **valuesA2)
train(train_dat, valid_dat, "", **valuesA3)
train(train_dat, valid_dat, "", **valuesA4)
train(train_dat, valid_dat, "", **valuesA5)
train(train_dat, valid_dat, "", **valuesA6)
train(train_dat, valid_dat, "", **valuesB)
train(train_dat, valid_dat, "", **valuesD1)
train(train_dat, valid_dat, "", **valuesD2)
train(train_dat, valid_dat, "", **valuesD3)
train(train_dat, valid_dat, "", **valuesD4)
train(train_dat, valid_dat, "", **valuesD5)

train(train_dat, valid_dat, "", **valuesE1)
train(train_dat, valid_dat, "", **valuesE2)
train(train_dat, valid_dat, "", **valuesE3)
train(train_dat, valid_dat, "", **valuesE4)

train(train_dat, valid_dat, "", **valuesG4)
train(train_dat, valid_dat, "", **valuesG5)
