import numpy as np

def downsample(array, old_dt, new_dt, axis=0):
    
    downsample_factor = int(new_dt / old_dt)
    indices = np.arange(0, array.shape[axis], downsample_factor)
    downsampled = np.take(array, indices, axis=axis)
    
    return downsampled
