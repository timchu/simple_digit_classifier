import numpy as np
from neural_fns import *

# l0 :: numpy.array
def ff1(l0, syn, neural_type="sigmoid"):
    #<x_1, .. x_n, 1>
    l = ap(l0, 1)
    #<x_1 .. x_n, 1> * <-[S1]-,...-[b]->
    lin1 = dt(l, syn)
    return (lin1, nonlin(neural_type, lin1))

def  printme(a, name="MARKER"):
    print name
    print a

# Return (lins layers, layers)
# Type: x0 = arr, syns = list of arrs, neural_types_list = list of strings
def feedforward(x0, syns, neural_types_list):
    lins = [x0]
    layers = [x0]
    cur_layer = np.array([x0])
    v = ap(x0, 1) # to account for bias, the last row of syn
    for i in xrange(len(syns)):
        neural_type = neural_types_list[i]
        syn = syns[i]
        (next_layer_lin, next_layer) = ff1(cur_layer, syn)

        # Post loop updating
        # set layers 
        layers = layers + [next_layer]
        lins = lins + [next_layer_lin]

        # set current layer
        cur_layer = next_layer
    return (lins, layers)
