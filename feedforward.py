import numpy as np
from neural_fns import *

#l0 :: numpy.array (2D)
def ff2(l0s, syn, neural_type="sigmoid"):
    #<z_1, .. z_n, 1>
    #<..........., 1>
    #<..........., 1>
    batch_z = ap(l0s, 1)
    batch_y1s = dt(batch_z, syn)
    batch_z1 = np.vstack([nonlin(neural_type, batch_y1) for batch_y1 in batch_y1s])
    return (batch_y1s, batch_z1)

def ff(x0, syns, neural_types_list, masks):
    ys = ["base"]
    zs = [x0]
    cur_z = x0
    for i in range(len(syns)):
        nt = neural_types_list[i]
        syn = syns[i]
        (next_y, next_z) = ff2(masks[i]*cur_z, syn, nt)

        # Post loop updating
        # set zs 
        zs = zs + [next_z]
        ys = ys + [next_y]

        # set current z
        cur_z = next_z
    return (ys, zs)
