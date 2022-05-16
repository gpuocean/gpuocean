import unittest
import time
import numpy as np
import sys
import gc
import pycuda.driver as cuda

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from stochastic.OceanStateNoise_test import OceanStateNoiseTest

class OceanStateNoiseLCGTest(OceanStateNoiseTest):
    """
    Executing all the same tests as OceanStateNoiseTest, but
    using the LCG algorithm for random numbers.
    """
        
    def useLCG(self):
        return True