# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements regression tests for generation of random 
numbers through the OceanNoise class.

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

from gpuocean.utils import Common
from gpuocean.SWEsimulators.OceanStateNoise import *
from stochastic.OceanStateNoise_parent import OceanStateNoiseTestParent


class RandThroughOceanNoiseTest(OceanStateNoiseTestParent):
        
    def test_random_uniform(self):
        self.create_large_noise()

        self.large_noise.generateUniformDistribution()

        U = self.large_noise.getRandomNumbers()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean - 0.5), 0.005)
        self.assertLess(np.abs(var - 1/12), 0.001)


    def test_random_uniform_CPU(self):
        self.create_large_noise()

        self.large_noise.generateUniformDistributionCPU()

        U = self.large_noise.getRandomNumbersCPU()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean - 0.5), 0.005)
        self.assertLess(np.abs(var - 1/12), 0.001)

    def test_random_normal(self):
        self.create_large_noise()

        self.large_noise.generateNormalDistribution()

        U = self.large_noise.getRandomNumbers()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean), 0.01)
        self.assertLess(np.abs(var - 1.0), 0.01)
        
    def test_random_normal_CPU(self):
        self.create_large_noise()

        self.large_noise.generateNormalDistributionCPU()

        U = self.large_noise.getRandomNumbersCPU()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean), 0.01)
        self.assertLess(np.abs(var - 1.0), 0.01)

    def test_random_normal_nu(self):
        self.create_large_noise()

        self.large_noise.generateNormalDistributionPerpendicular()

        U = self.large_noise.getPerpendicularRandomNumbers()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean), 0.01)
        self.assertLess(np.abs(var - 1.0), 0.01)
        
    def test_random_normal_perpendicular(self):
        self.create_large_noise()
        tol = 6

        self.large_noise.generatePerpendicularNormalDistributions()

        xi = self.large_noise.getRandomNumbers()
        nu = self.large_noise.getPerpendicularRandomNumbers()
        rel = np.sum(xi*xi)
        
        mean_xi = np.mean(xi)
        var_xi = np.var(xi)
        mean_nu = np.mean(nu)
        var_nu = np.var(nu)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean_xi), 0.01)
        self.assertLess(np.abs(var_xi - 1.0), 0.01)
        self.assertLess(np.abs(mean_nu), 0.01)
        self.assertLess(np.abs(var_nu - 1.0), 0.01)
        
        # Get the norms of the random vectors before they became perpendicular 
        pre_reduction_buffer = self.large_noise.getReductionBuffer()
        
        # Get the norms of the random vectors after they became perpendicular 
        self.large_noise.findDoubleNormAndDot()
        post_reduction_buffer = self.large_noise.getReductionBuffer()
        
        # Check that the final norms on the GPU matches those on the CPU
        self.assertAlmostEqual(post_reduction_buffer[0,0]/rel, np.sum(xi*xi)/rel, tol)
        self.assertAlmostEqual(post_reduction_buffer[0,1]/rel, np.sum(nu*nu)/rel, tol)
        
        # Check that the first values are the same:
        self.assertAlmostEqual(post_reduction_buffer[0,0]/rel, pre_reduction_buffer[0,0]/rel, tol)
        self.assertAlmostEqual(post_reduction_buffer[0,1]/rel, pre_reduction_buffer[0,1]/rel, tol)
        
        # Check that their dot-products (both from the GPU and CPU) are zero
        self.assertAlmostEqual(post_reduction_buffer[0,2]/rel, 0.0, places=3)
        self.assertAlmostEqual(np.sum(xi*nu), 0.0, places=3)
        
        

    def test_seed_diff(self):
        
        self.create_noise()
        tol = 6

        if self.noise.use_lcg:
            init_seed = self.noise.getSeed()/self.floatMax
            self.noise.generateNormalDistribution()
            normal_seed = self.noise.getSeed()/self.floatMax
            assert2DListNotAlmostEqual(self, normal_seed.tolist(), init_seed.tolist(), tol, "test_seed_diff, normal vs init_seed")
            
            self.noise.generateUniformDistribution()
            uniform_seed = self.noise.getSeed()/self.floatMax
            assert2DListNotAlmostEqual(self, uniform_seed.tolist(), init_seed.tolist(), tol, "test_seed_diff, uniform vs init_seed")
            assert2DListNotAlmostEqual(self, uniform_seed.tolist(), normal_seed.tolist(), tol, "test_seed_diff, uniform vs normal_seed")
        else:
            self.assertIsNone(self.noise.rng.seed)
            self.assertIsNone(self.noise.rng.host_seed)
            self.failUnlessRaises(AssertionError, self.noise.getSeed)
            self.failUnlessRaises(AssertionError, self.noise.resetSeed)
           
        
    def test_empty_reduction_buffer(self):
        self.create_large_noise()
        
        buffer_host = self.large_noise.getReductionBuffer()
        self.assertEqual(buffer_host.shape, (1,3))
        self.assertEqual(buffer_host[0,0], 0.0)
        self.assertEqual(buffer_host[0,1], 0.0)
        self.assertEqual(buffer_host[0,2], 0.0)
        
    def test_reduction(self):
        self.create_large_noise()
        
        self.large_noise.generateNormalDistribution()
        obtained_random_numbers = self.large_noise.getRandomNumbers()
        gamma_from_numpy = np.linalg.norm(obtained_random_numbers)**2
        
        gamma = self.large_noise.getRandomNorm()

        # Checking relative difference 
        self.assertAlmostEqual((gamma_from_numpy-gamma)/gamma_from_numpy, 0.0, places=5)

        
