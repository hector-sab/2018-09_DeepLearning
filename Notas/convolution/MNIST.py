import struct
import numpy as np

"""
Load the data from MNIST and returns a numpy array
Original data downloaded from http://yann.lecun.com/exdb/mnist/
"""

def load_mnist(filename):
  with open(filename,'rb') as f:
    zero,data_type,dims = struct.unpack('>HBB',f.read(4))
    shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
    return(np.fromstring(f.read(), dtype=np.uint8).reshape(shape))