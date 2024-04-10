import unittest
import time
import numpy as np
import sys
import gc
import pycuda.driver as cuda

from testUtils import *

from gpuocean.utils import Common
from stochastic.RandThroughOceanNoise_test import RandThroughOceanNoiseTest


class RandThroughOceanNoiseLCGTest(RandThroughOceanNoiseTest):
    """
    Executing all the same tests as RandomNumbersTest, but
    using the LCG algorithm for random numbers.
    """
        
    def useLCG(self):
        return True