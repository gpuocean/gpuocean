# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements regression tests for the OceanStateNoise class.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
import time
import numpy as np
import sys
import gc
import pycuda.driver as cuda

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.OceanStateNoise import *


class OceanStateNoiseTestParent(unittest.TestCase):

    def setUp(self):
        self.gpu_ctx = Common.CUDAContext()
        self.gpu_stream = cuda.Stream()
        
        self.nx = 30
        self.ny = 40
        self.dx = 7.0
        self.dy = 7.0

        self.f = 0.02
        self.g = 9.81
        self.beta = 0.0
        
        self.noise = None
        
        self.ghost_cells_x = 2
        self.ghost_cells_y = 2
        self.datashape = (self.ny+2*self.ghost_cells_y, self.nx+2*self.ghost_cells_x)
        
        self.cutoff = 2
        self.nx_nonPeriodic = self.nx + 2*(2+self.cutoff)
        self.ny_nonPeriodic = self.ny + 2*(2+self.cutoff)
        
        # Standard setup is non-staggered, periodic
        self.staggered = False
        self.periodicNS = True
        self.periodicEW = True

        # Total number of threads should be: 16, 32, 48, 64
        # Corresponding to the number of blocks: 1, 2, 3, 4
        self.glob_size_x = 3
        self.glob_size_y = 3
        self.glob_size_x_nonperiodic = 3
        self.glob_size_y_nonperiodic = 3
        self.glob_size_random_x = 1
        self.glob_size_random_x_nonperiodic = 2

        self.large_nx = 400
        self.large_ny = 400
        self.large_noise = None

        self.floatMax = 2147483648.0

        self.eta = None
        self.hu = None
        self.hv = None
        self.H = None

        
    def tearDown(self):
        if self.noise is not None:
            self.noise.cleanUp()
            del self.noise
        if self.large_noise is not None:
            self.large_noise.cleanUp()
            del self.large_noise
        if self.eta is not None:
            self.eta.release()
        if self.hu is not None:
            self.hu.release()
        if self.hv is not None:
            self.hv.release()
        if self.H is not None:
            self.H.release()
        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
   
        gc.collect()
        
    def useLCG(self):
        return False
            
    def create_noise(self, factor=1):
        n,e,s,w = 1,1,1,1
        if self.periodicNS:
            n,s = 2,2
        if self.periodicEW:
            e,w = 2,2
        self.noise = OceanStateNoise(self.gpu_ctx, self.gpu_stream,
                                     self.nx, self.ny,
                                     self.dx, self.dy,
                                     Common.BoundaryConditions(n,e,s,w),
                                     staggered=self.staggered,
                                     interpolation_factor=factor,
                                     use_lcg=self.useLCG())
    def create_large_noise(self):
        n,e,s,w = 1,1,1,1
        if self.periodicNS:
            n,s = 2,2
        if self.periodicEW:
            e,w = 2,2
        self.large_noise = OceanStateNoise(self.gpu_ctx, self.gpu_stream,
                                           self.large_nx, self.large_ny,
                                           self.dx, self.dy,
                                           Common.BoundaryConditions(n,e,s,w),
                                           staggered = self.staggered,
                                           use_lcg=self.useLCG())

    def allocateBuffers(self, HCPU):
        host_buffer = np.zeros((self.ny+2*self.ghost_cells_y, self.nx+2*self.ghost_cells_x))
        self.eta = Common.CUDAArray2D(self.gpu_stream, self.nx, self.ny, self.ghost_cells_x, self.ghost_cells_y, host_buffer)
        self.hu = Common.CUDAArray2D(self.gpu_stream, self.nx, self.ny, self.ghost_cells_x, self.ghost_cells_y, host_buffer)
        self.hv = Common.CUDAArray2D(self.gpu_stream, self.nx, self.ny, self.ghost_cells_x, self.ghost_cells_y, host_buffer)
        self.H = Common.CUDAArray2D(self.gpu_stream, self.nx+1, self.ny+1, self.ghost_cells_x, self.ghost_cells_y, HCPU)
        del host_buffer
        
    def compare_random(self, tol, msg):

        # The tolerance provided to testAlmostEqual makes the comparison wrt to the
        # number of decimal places, not the number of significant digits.
        # We therefore make sure that seed is in [0, 1]
        if self.noise.use_lcg:
            seed = self.noise.getSeed()
            seedCPU = self.noise.getSeedCPU()

            msg = msg+"\ntype(seed):    " + str(type(seed)) + ", " + str(type(seed[0,0]))\
                  + "\ntype(seedCPU): " + str(type(seedCPU)) + ", " + str(type(seedCPU[0,0]))
            
            assert2DListAlmostEqual(self, seed.tolist(), seedCPU.tolist(), tol, msg+", seed")

        random = self.noise.getRandomNumbers()
        randomCPU = self.noise.getRandomNumbersCPU()

        assert2DListAlmostEqual(self, random.tolist(), randomCPU.tolist(), tol, msg+", random")

   
        