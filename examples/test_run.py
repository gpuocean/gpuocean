# Testing import of python packages
import os, sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from skimage.measure import block_reduce

# Testing import of pycuda
import pycuda.driver as cuda

# Testing import of gpuocean
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16

# Testing generation of context and streams
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

# Testing a small simulation
sim = CDKLM16.CDKLM16(gpu_ctx, np.zeros((104,104)), np.zeros((104,104)), np.zeros((104,104)), np.ones((105,105)),
                        100, 100, 10.0, 10.0, 0.0, 10.0, 0.0, 0.0)

sim.step(100)

print("sucess")