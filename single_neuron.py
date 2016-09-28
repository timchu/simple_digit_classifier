# I want my single neuron to learn a perceptron. 
# The activation function is sigmoid.
# if x_1 + x_2 >= 1

import random
import math

# generate 3*n samples of data, n of which sum to 1, n of which sum to more than 1, n of which sum to less than 1.
# only three inputs, and one output.
def gen_pair_with_threshold(x):
    r = random.random()
    return [x*r, x - x*r, int(x>=1)]
def generate_data(n):
    ls = []
    for i in xrange(n):
        ls.append(gen_pair_with_threshold(1.05))
        ls.append(gen_pair_with_threshold(1.1))
        ls.append(gen_pair_with_threshold(1.2))
        ls.append(gen_pair_with_threshold(0.93))
        ls.append(gen_pair_with_threshold(0.9))
        ls.append(gen_pair_with_threshold(0.8))
        ls.append(gen_pair_with_threshold(0.1))
    return ls

data = generate_data(1)

def dot(K, L):
    if len(K) != len(L):
        print "invalid dot product"
        return 0
    return sum(i[0] * i[1] for i in zip(K, L))

def sigmoid(w, b, x):
    # clunky way of doing w*x + b
    v = dot(w,x) + b
    return 1/(1+float(math.exp(-v)))

# w should just get super large or something.
# data should be passed in as a variable
print data
# No training yet. Just finding the loss function.
def train(epochs, learning_rate):
    weights = [100.0, 100.0]
    biases = -100.0

    sum_losses = 0;
    for datum in data:
        an_input = datum[:2]
        desired_output = datum[2]
        neuron_output = sigmoid(weights,biases,an_input)
        if desired_output == 1:
            pr_is_one = neuron_output
            loss = math.log(1/(pr_is_one), 2)
        else:
            pr_is_zero =  1 - neuron_output
            loss = math.log(1/pr_is_zero, 2)
        print loss
        sum_losses += loss
    expected_loss = sum_losses/len(data)
    return expected_loss
# Note: you never compute the loss. You compute the gradient of the loss!
        # no training
