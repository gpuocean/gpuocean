# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018, 2024 SINTEF Digital

This python module implements regression tests for generation of random numbers.

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

from gpuocean.utils import Common, RandomNumbers

class RandomNumbersTest(unittest.TestCase):

    def setUp(self):
        self.gpu_ctx = Common.CUDAContext()
        self.gpu_stream = cuda.Stream()
        
        self.nx = 512
        self.ny = 512
        
        self.rng = None
        self.random_numbers = None
                
        self.floatMax = 2147483648.0

        
    def tearDown(self):
        if self.rng is not None:
            self.rng.cleanUp()
            del self.rng
        if self.random_numbers is not None:
            self.random_numbers.release()
        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
   
        gc.collect()
            
    def create_rng(self, lcg):
        self.rng = RandomNumbers.RandomNumbers(self.gpu_ctx, self.gpu_stream,
                                               self.nx, self.ny,
                                               use_lcg=lcg)
        self.random_numbers = Common.CUDAArray2D(self.gpu_stream, self.nx, self.ny, 0, 0,
                                                np.zeros((self.ny, self.nx), dtype=np.float32))

    #########################################################################
    ### Tests 
    #########################################################################
    def random_uniform(self, lcg):
        self.create_rng(lcg)
        self.rng.generateUniformDistribution(self.random_numbers)
        U = self.random_numbers.download(self.gpu_stream)

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean - 0.5), 0.005)
        self.assertLess(np.abs(var - 1/12), 0.001)
        
    def test_random_uniform_lcg(self):
        self.random_uniform(lcg=True)

    def test_random_uniform_curand(self):
        self.random_uniform(lcg=False)

    def random_normal(self, lcg):
        self.create_rng(lcg)
        self.rng.generateNormalDistribution(self.random_numbers)
        U = self.random_numbers.download(self.gpu_stream)

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean), 0.01)
        self.assertLess(np.abs(var - 1.0), 0.01)

    def test_random_normal_lcg(self):
        self.random_normal(lcg=True)

    def test_random_normal_curand(self):
        self.random_normal(lcg=False)


    def seed_diff(self, lcg):
        self.create_rng(lcg=lcg)
        tol = 7

        if lcg:
            init_seed = self.rng.getSeed()/self.rng.floatMax
            self.rng.generateNormalDistribution(self.random_numbers)
            normal_seed = self.rng.getSeed()/self.rng.floatMax
            assert2DListNotAlmostEqual(self, normal_seed.tolist(), init_seed.tolist(), tol, "test_seed_diff, normal vs init_seed")
            
            self.rng.generateUniformDistribution(self.random_numbers)
            uniform_seed = self.rng.getSeed()/self.floatMax
            assert2DListNotAlmostEqual(self, uniform_seed.tolist(), init_seed.tolist(), tol, "test_seed_diff, uniform vs init_seed")
            assert2DListNotAlmostEqual(self, uniform_seed.tolist(), normal_seed.tolist(), tol, "test_seed_diff, uniform vs normal_seed")
        else:
            self.assertIsNone(self.rng.seed)
            self.assertIsNone(self.rng.host_seed)
            self.failUnlessRaises(AssertionError, self.rng.getSeed)
            self.failUnlessRaises(AssertionError, self.rng.resetSeed)
           
    def test_seed_diff_lcg(self):
        self.seed_diff(lcg=True)
    
    def test_seed_diff_curand(self):
        self.seed_diff(lcg=False)