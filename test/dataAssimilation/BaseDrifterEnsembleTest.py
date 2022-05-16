# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements unit tests for the DrifterEnsemble class

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

import abc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators import DataAssimilationUtils as dautils

from SWESimulators import DrifterEnsemble


class BaseDrifterEnsembleTest(unittest.TestCase):

    # Skipping tests in Base class (see reference below)
    # https://stackoverflow.com/questions/1323455/python-unit-test-with-base-and-sub-class/17696807#17696807
    @classmethod
    def setUpClass(cls):
        if cls is BaseDrifterEnsembleTest:
            raise unittest.SkipTest("Skip BaseDrifterEnsembleTest tests, it is a base class!")
        super(BaseDrifterEnsembleTest, cls).setUpClass()
    
    def setUp(self):
        self.nx = 1
        self.ny = 1
        self.dx = 1.0
        self.dy = 1.0
        self.dt = 1.0
        
        self.numParticles = 3
        self.observationVariance = 0.25
        self.boundaryCondition = Common.BoundaryConditions(2,2,2,2)
        self.smallParticleSet = None
        # to be initialized by child class with above values
        
        self.gpu_ctx = None
        
        self.smallPositionSetHost = np.array( [[0.9, 0.9], [0.9, 0.1],
                                               [0.1, 0.9], [0.1, 0.1]])
        
        self.resampleNumParticles = 6
        self.resamplingParticleArray = np.zeros((7,2))
        self.resamplingObservationVariance = 0.01
        for i in range(2):
            self.resamplingParticleArray[3*i+0, :] = [0.25, 0.35+i*0.3]
            self.resamplingParticleArray[3*i+1, :] = [0.4,  0.35+i*0.3]
            self.resamplingParticleArray[3*i+2, :] = [0.65, 0.35+i*0.3]
        self.resamplingParticleArray[6, :] = [0.25, 0.5]
        self.resamplingParticleSet = None
        # to be initialized by child class wit resampleNumParticles only.

        
        
        self.resamplingVar = 1e-8
        
    def tearDown(self):
        #pass
        if self.smallParticleSet is not None:
            self.smallParticleSet.cleanUp()
        if self.resamplingParticleSet is not None:
            self.resamplingParticleSet.cleanUp()

        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
        
        self.gpu_ctx = None
        gc.collect()
        
    ### set observation and particle positions to the test cases
    def set_positions_small_set(self):
        self.create_small_particle_set()
        self.smallParticleSet.setParticleStates(self.smallPositionSetHost[:-1, :])
        self.smallParticleSet.setObservationState(self.smallPositionSetHost[-1, :])

    def set_positions_resampling_set(self):
        self.create_resampling_particle_set()
        self.resamplingParticleSet.setParticleStates(self.resamplingParticleArray[:-1,:])
        self.resamplingParticleSet.setObservationState(self.resamplingParticleArray[-1,:])


    ### Define required functions as abstract ###

    @abc.abstractmethod
    def create_small_particle_set(self):
        pass
        

    @abc.abstractmethod
    def create_resampling_particle_set(self):
        pass

    @abc.abstractmethod
    def create_large_particle_set(self, size, domain_x, domain_y):
        pass

        
        
    ### START TESTS ###
    
    def hhhtest_default_constructor(self):
        self.create_resampling_particle_set()
        defaultParticleSet = self.resamplingParticleSet

        self.assertEqual(defaultParticleSet.getNumParticles(), self.resampleNumParticles)
        self.assertEqual(defaultParticleSet.getObservationVariance(), 0.1)

        positions = defaultParticleSet.observeParticles()
        defaultPosition = [0.0, 0.0]
        self.assertEqual(positions.shape, ((self.resampleNumParticles, 2)))
        for i in range(self.resampleNumParticles):
            self.assertEqual(positions[i,:].tolist(), defaultPosition)
                         
        observation = defaultParticleSet.observeTrueState()
        self.assertEqual(observation.shape, ((2,)))
        self.assertEqual(observation.tolist(), defaultPosition)

        self.assertEqual(defaultParticleSet.getDomainSizeX(), 1.0)
        self.assertEqual(defaultParticleSet.getDomainSizeY(), 1.0)

        weight = 1.0/self.resampleNumParticles
        weights = [weight]*self.resampleNumParticles
        self.assertEqual(defaultParticleSet.getGaussianWeight().tolist(), weights)
        self.assertEqual(defaultParticleSet.getCauchyWeight().tolist(), weights)
        
        # Check boundary condition
        self.assertEqual(defaultParticleSet.getBoundaryConditions().get(), [1,1,1,1])
        
    def test_non_default_constructor(self):
        self.set_positions_small_set()
        self.assertEqual(self.smallParticleSet.getNumParticles(), self.numParticles)
        self.assertEqual(self.smallParticleSet.getObservationVariance(), self.observationVariance)
        
        positions = self.smallParticleSet.observeParticles()
        self.assertEqual(positions.shape, ((self.numParticles, 2)))
        assertListAlmostEqual(self, positions[0,:].tolist(), [0.9, 0.9], 6,
                              'non-default constructor, particle 0')
        assertListAlmostEqual(self, positions[1,:].tolist(), [0.9, 0.1], 6,
                              'non-default constructor, particle 1')
        assertListAlmostEqual(self, positions[2,:].tolist(), [0.1, 0.9], 6,
                              'non-default constructor, particle 2')

        observation = self.smallParticleSet.observeTrueState()
        self.assertEqual(observation.shape, ((2,)))
        assertListAlmostEqual(self, observation.tolist(), [0.1, 0.1], 6,
                              'non-default constructor, observation')

        
    def test_set_particle_positions(self):
        self.set_positions_small_set()
        pos1 = [0.2, 0.5]
        pos2 = [0.8, 0.235]
        pos3 = [0.01, 0.01]
        newPositions = np.array([pos1, pos2, pos3])

        self.smallParticleSet.setParticleStates(newPositions)

        positions = self.smallParticleSet.observeParticles()
        self.assertEqual(positions.shape, ((self.numParticles, 2)))
        assertListAlmostEqual(self, positions[0,:].tolist(), pos1, 6,
                              'set particle positions, particle 0')
        assertListAlmostEqual(self, positions[1,:].tolist(), pos2, 6,
                              'set particles positions, particle 1')
        assertListAlmostEqual(self, positions[2,:].tolist(), pos3, 6,
                              'set particles positions, particle 2')

        
    def test_set_particle_positions_unchanged_observation(self):
        self.set_positions_small_set()
        pos1 = [0.2, 0.5]
        pos2 = [0.8, 0.235]
        pos3 = [0.01, 0.01]
        newPositions = np.array([pos1, pos2, pos3])

        initObservation = self.smallParticleSet.observeTrueState()

        # Set particles, but observation should stay the same.
        self.smallParticleSet.setParticleStates(newPositions)

        positions = self.smallParticleSet.observeParticles()
        self.assertEqual(positions.shape, ((self.numParticles, 2)))
        assertListAlmostEqual(self, positions[0,:].tolist(), pos1, 6,
                              'set particle positions, particle 0')
        assertListAlmostEqual(self, positions[1,:].tolist(), pos2, 6,
                              'set particles positions, particle 1')
        assertListAlmostEqual(self, positions[2,:].tolist(), pos3, 6,
                              'set particles positions, particle 2')
        
        observation = self.smallParticleSet.observeTrueState()
        self.assertEqual(observation.shape, ((2,)))
        assertListAlmostEqual(self, observation.tolist(), initObservation.tolist(), 6,
                              'set particles positions, observation')


    def test_set_observation_position(self):
        self.set_positions_small_set()
        pos = np.array([0.523, 0.999])
        self.smallParticleSet.setObservationState(pos)
        
        observation = self.smallParticleSet.observeTrueState()
        self.assertEqual(observation.shape, ((2,)))
        assertListAlmostEqual(self, observation.tolist(), pos.tolist(), 6,
                              'set observation, observation')

    def test_set_observation_position_unchanged_particles(self):
        self.set_positions_small_set()
        pos = np.array([0.523, 0.999])
        self.smallParticleSet.setObservationState(pos)
        
        positions = self.smallParticleSet.observeParticles()
        self.assertEqual(positions.shape, ((self.numParticles, 2)))
        assertListAlmostEqual(self, positions[0,:].tolist(), [0.9, 0.9], 6,
                              'set observation, particle 0')
        assertListAlmostEqual(self, positions[1,:].tolist(), [0.9, 0.1], 6,
                              'set observation, particle 1')
        assertListAlmostEqual(self, positions[2,:].tolist(), [0.1, 0.9], 6,
                              'set observation, particle 2')

        observation = self.smallParticleSet.observeTrueState()
        self.assertEqual(observation.shape, ((2,)))
        assertListAlmostEqual(self, observation.tolist(), pos.tolist(), 6,
                              'set observation, observation')
                
        
    def test_distances(self):
        self.set_positions_small_set()
        longDiag = np.sqrt(2*0.8*0.8)
        longLine = 0.8
        shortDiag = np.sqrt(0.2*0.2 + 0.2*0.2)
        shortLine = 0.2
        semiDiag = np.sqrt(0.2*0.2 + 0.8*0.8)
                           
        
        # smallParticleSet is initially with periodic boundary conditions
        assertListAlmostEqual(self, self.smallParticleSet.getDistances().tolist(), \
                              [shortDiag, shortLine, shortLine], 6,
                              'distance with periodic boundaries')
        
    def test_ensemble_mean(self):
        self.set_positions_small_set()
        periodicMean = [1-0.1/3, 1-0.1/3]
                
        assertListAlmostEqual(self, self.smallParticleSet.getEnsembleMean().tolist(),
                              periodicMean, 6,
                              'periodic mean')

                
    def test_init_uniform_positions(self):
        
        domain_x = 10.3
        domain_y = 5.4
        largeParticleSet = self.create_large_particle_set(1000,
                                                          domain_x,
                                                          domain_y)
        
        self.assertEqual(largeParticleSet.getDomainSizeX(), domain_x)
        self.assertEqual(largeParticleSet.getDomainSizeY(), domain_y)

        p = largeParticleSet.observeParticles()
        self.assertGreaterEqual(np.min(p[:,0]), 0.0)
        self.assertLessEqual(np.max(p[:,0]), domain_x)
        self.assertGreaterEqual(np.min(p[:,1]) , 0.0)
        self.assertLessEqual(np.max(p[:,1]), domain_y)

        
    def test_gaussian_weights(self):
        self.set_positions_resampling_set()
        obtainedWeights = self.resamplingParticleSet.getGaussianWeight()
        referenceWeights = [  3.77361928e-01,   1.22511481e-01,   1.26590824e-04,   3.77361928e-01, 1.22511481e-01,   1.26590824e-04]
        assertListAlmostEqual(self, obtainedWeights.tolist(),
                              referenceWeights, 6,
                              'gaussian weights')

    def test_cauchy_weights(self):
        self.set_positions_resampling_set()
        obtainedWeights = self.resamplingParticleSet.getCauchyWeight()
        referenceWeights = [0.28413284,  0.16789668,  0.04797048,  0.28413284,  0.16789668,  0.04797048]
        assertListAlmostEqual(self, obtainedWeights.tolist(),
                              referenceWeights, 6,
                              'cauchy weights')

    def resample(self, indices_list):
        newParticlePositions = []
        for i in indices_list:
            newParticlePositions.append(self.resamplingParticleSet.observeParticles()[i,:].tolist())
        return newParticlePositions
        
    def test_resampling_predefined_indices(self):
        self.set_positions_resampling_set()
        indices_list = [2,2,2,4,5,5]
        newParticlePositions = self.resample(indices_list)
        self.resamplingParticleSet.resample(indices_list, 0)
        self.assertEqual(self.resamplingParticleSet.observeParticles().tolist(), \
                         newParticlePositions)

    def test_probabilistic_resampling_with_duplicates(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [1,3,0,0,0,0]
        solutions = self.resample(indices)
        dautils.probabilisticResampling(self.resamplingParticleSet)
        self.assertEqual(self.resamplingParticleSet.observeParticles().tolist(), \
                         solutions)

            
    def test_residual_sampling_with_duplicates(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,3,3,1,4]
        solutions = self.resample(indices)
        dautils.residualSampling(self.resamplingParticleSet)
        self.assertEqual(self.resamplingParticleSet.observeParticles().tolist(), \
                         solutions)
        
    def test_stochastic_universal_sampling_with_duplicates(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,1,3,3,4]
        solutions = self.resample(indices)
        dautils.stochasticUniversalSampling(self.resamplingParticleSet)
        self.assertEqual(self.resamplingParticleSet.observeParticles().tolist(), \
                         solutions)

    def test_monte_carlo_metropolis_hasting_sampling_with_duplicates(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,0,3,4,4]
        solutions = self.resample(indices)
        dautils.metropolisHastingSampling(self.resamplingParticleSet)
        self.assertEqual(self.resamplingParticleSet.observeParticles().tolist(), solutions)


    def test_probabilistic_resampling(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [1,3,0,0,0,0]
        solutions = self.resample(indices)
        dautils.probabilisticResampling(self.resamplingParticleSet, self.resamplingVar)
        assert2DListAlmostEqual(self, self.resamplingParticleSet.observeParticles().tolist(), solutions, 2, "probabilistic resampling, probabilistic duplicates")

            
    def test_residual_sampling(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,3,3,1,4]
        solutions = self.resample(indices)
        dautils.residualSampling(self.resamplingParticleSet, self.resamplingVar)
        assert2DListAlmostEqual(self, self.resamplingParticleSet.observeParticles().tolist(), solutions, 2, "residual sampling, probabilistic duplicates")
                
    def test_stochastic_universal_sampling(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,1,3,3,4]
        solutions = self.resample(indices)
        dautils.stochasticUniversalSampling(self.resamplingParticleSet, self.resamplingVar)
        assert2DListAlmostEqual(self, self.resamplingParticleSet.observeParticles().tolist(), solutions, 2, "stochastic universal sampling, probabilistic duplicates")

    def test_monte_carlo_metropolis_hasting_sampling(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,0,3,4,4]
        solutions = self.resample(indices)
        dautils.metropolisHastingSampling(self.resamplingParticleSet, self.resamplingVar)
        #print self.resamplingParticleSet.observeParticles().tolist()
        assert2DListAlmostEqual(self, self.resamplingParticleSet.observeParticles().tolist(), solutions, 2, "metropolis hasting sampling, probabilistic duplicates")
        

