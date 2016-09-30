import numpy as np
from neural_fns import *

# l0 :: numpy.array (1D)
def ff1(l0, syn, neural_type="sigmoid"):
    #<x_1, .. x_n, 1>
    l = ap(l0, 1)
    #<x_1 .. x_n, 1> * <-[S1]-,...-[b]->
    y1 = dt(l, syn)
    return (y1, nonlin(neural_type, y1))

#l0 :: numpy.array (2D)
def ff2(l0s, syn, neural_type="sigmoid"):
    #<x_1, .. x_n, 1>
    #<..........., 1>
    #<..........., 1>
    batch_x = ap(l0s, 1)
    batch_y1s = dt(batch_x, syn)
    batch_x1 = np.vstack([nonlin(neural_type, batch_y1) for batch_y1 in batch_y1s])
    return (batch_y1s, batch_x1)

def ff(x0, syns, neural_types_list):
    ys = ["base"]
    zs = [x0]
    cur_z = x0
    for i in range(len(syns)):
        nt = neural_types_list[i]
        syn = syns[i]
        (next_y, next_z) = ff2(cur_z, syn, nt)

        # Post loop updating
        # set zs 
        zs = zs + [next_z]
        ys = ys + [next_y]

        # set current z
        cur_z = next_z
    return (ys, zs)
