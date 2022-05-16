# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements an abstract class for testing drifter 
functionality.

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



class BaseDrifterTest(unittest.TestCase):

    # Skipping tests in Base class (see reference below)
    # https://stackoverflow.com/questions/1323455/python-unit-test-with-base-and-sub-class/17696807#17696807
    @classmethod
    def setUpClass(cls):
        if cls is BaseDrifterTest:
            raise unittest.SkipTest("Skip BaseDrifterTest tests, it is a base class!")
        super(BaseDrifterTest, cls).setUpClass()
    
    def setUp(self):
        self.gpu_ctx = None
        
        self.numDrifters = 3
        self.observationVariance = 0.25
        self.boundaryCondition = Common.BoundaryConditions(2,2,2,2)
        self.smallDrifterSet = None
        # to be initialized by child class with above values
        
        self.smallPositionSetHost = np.array( [[0.9, 0.9], [0.9, 0.1],
                                               [0.1, 0.9], [0.1, 0.1]])
        
        self.resampleNumDrifters = 6
        self.resamplingDriftersArray = np.zeros((7,2))
        for i in range(2):
            self.resamplingDriftersArray[3*i+0, :] = [0.25, 0.35+i*0.3]
            self.resamplingDriftersArray[3*i+1, :] = [0.4,  0.35+i*0.3]
            self.resamplingDriftersArray[3*i+2, :] = [0.65, 0.35+i*0.3]
        self.resamplingDriftersArray[6, :] = [0.25, 0.5]
        self.resamplingDrifterSet = None
        # to be initialized by child class wit resampleNumDrifters only.

        self.resamplingVar = 1e-8

        self.largeDrifterSet = None

        
    def tearDown(self):
        if self.smallDrifterSet is not None:
            self.smallDrifterSet.cleanUp()
            del self.smallDrifterSet
        if self.resamplingDrifterSet is not None:
            self.resamplingDrifterSet.cleanUp()
            del self.resamplingDrifterSet
        if self.largeDrifterSet is not None:
            self.largeDrifterSet.cleanUp()
            del self.largeDrifterSet
            
    ### set observation and drifter positions to the test cases
    def set_positions_small_set(self):
        self.create_small_drifter_set()
        self.smallDrifterSet.setDrifterPositions(self.smallPositionSetHost[:-1, :])
        self.smallDrifterSet.setObservationPosition(self.smallPositionSetHost[-1, :])

    def set_positions_resampling_set(self):
        self.create_resampling_drifter_set()
        self.resamplingDrifterSet.setDrifterPositions(self.resamplingDriftersArray[:-1,:])
        self.resamplingDrifterSet.setObservationPosition(self.resamplingDriftersArray[-1,:])


    ### Define required functions as abstract ###

    @abc.abstractmethod
    def create_small_drifter_set(self):
        pass

    @abc.abstractmethod
    def create_resampling_drifter_set(self):
        pass

    @abc.abstractmethod
    def create_large_drifter_set(self, size, domain_x, domain_y):
        pass

        
        
    ### START TESTS ###
    
    def test_default_constructor(self):
        self.create_resampling_drifter_set()
        defaultDrifterSet = self.resamplingDrifterSet

        self.assertEqual(defaultDrifterSet.getNumDrifters(), self.resampleNumDrifters)
        self.assertEqual(defaultDrifterSet.getObservationVariance(), 0.01)

        positions = defaultDrifterSet.getDrifterPositions()
        defaultPosition = [0,0]
        defaultPositionWithSixDrifters = [[1/6, 1/4], [3/6, 1/4], [5/6, 1/4],
                                          [1/6, 3/4], [3/6, 3/4], [5/6, 3/4]]
                                          
        
        self.assertEqual(positions.shape, ((self.resampleNumDrifters, 2)))
        for i in range(self.resampleNumDrifters):
            self.assertAlmostEqual(positions[i,0], defaultPositionWithSixDrifters[i][0], 6)
            self.assertEqual(      positions[i,1], defaultPositionWithSixDrifters[i][1])
                         
        observation = defaultDrifterSet.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        self.assertEqual(observation.tolist(), defaultPosition)

        self.assertEqual(defaultDrifterSet.getDomainSizeX(), 1.0)
        self.assertEqual(defaultDrifterSet.getDomainSizeY(), 1.0)

        # Check boundary condition
        self.assertEqual(defaultDrifterSet.getBoundaryConditions().get(), [1,1,1,1])
        
    def test_non_default_constructor(self):
        self.set_positions_small_set()
        self.assertEqual(self.smallDrifterSet.getNumDrifters(), self.numDrifters)
        self.assertEqual(self.smallDrifterSet.getObservationVariance(), self.observationVariance)
        
        positions = self.smallDrifterSet.getDrifterPositions()
        self.assertEqual(positions.shape, ((self.numDrifters, 2)))
        assertListAlmostEqual(self, positions[0,:].tolist(), [0.9, 0.9], 6,
                              'non-default constructor, drifter 0')
        assertListAlmostEqual(self, positions[1,:].tolist(), [0.9, 0.1], 6,
                              'non-default constructor, drifter 1')
        assertListAlmostEqual(self, positions[2,:].tolist(), [0.1, 0.9], 6,
                              'non-default constructor, drifter 2')

        observation = self.smallDrifterSet.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        assertListAlmostEqual(self, observation.tolist(), [0.1, 0.1], 6,
                              'non-default constructor, observation')

        self.assertEqual(self.smallDrifterSet.getBoundaryConditions().get(), [2,2,2,2])


    def test_set_boundary_condition(self):
        self.set_positions_small_set()
        self.smallDrifterSet.setBoundaryConditions(Common.BoundaryConditions(2,1,2,1))
        self.assertEqual(self.smallDrifterSet.getBoundaryConditions().get(), [2,1,2,1])

    def test_set_drifter_positions(self):
        self.set_positions_small_set()
        pos1 = [0.2, 0.5]
        pos2 = [0.8, 0.235]
        pos3 = [0.01, 0.01]
        newPositions = np.array([pos1, pos2, pos3])

        self.smallDrifterSet.setDrifterPositions(newPositions)

        positions = self.smallDrifterSet.getDrifterPositions()
        self.assertEqual(positions.shape, ((self.numDrifters, 2)))
        assertListAlmostEqual(self, positions[0,:].tolist(), pos1, 6,
                              'set drifters positions, drifter 0')
        assertListAlmostEqual(self, positions[1,:].tolist(), pos2, 6,
                              'set drifters positions, drifter 1')
        assertListAlmostEqual(self, positions[2,:].tolist(), pos3, 6,
                              'set drifters positions, drifter 2')

        
    def test_set_drifter_positions_unchanged_observation(self):
        self.set_positions_small_set()
        pos1 = [0.2, 0.5]
        pos2 = [0.8, 0.235]
        pos3 = [0.01, 0.01]
        newPositions = np.array([pos1, pos2, pos3])

        self.smallDrifterSet.setDrifterPositions(newPositions)

        positions = self.smallDrifterSet.getDrifterPositions()
        self.assertEqual(positions.shape, ((self.numDrifters, 2)))
        assertListAlmostEqual(self, positions[0,:].tolist(), pos1, 6,
                              'set drifters positions, drifter 0')
        assertListAlmostEqual(self, positions[1,:].tolist(), pos2, 6,
                              'set drifters positions, drifter 1')
        assertListAlmostEqual(self, positions[2,:].tolist(), pos3, 6,
                              'set drifters positions, drifter 2')
        
        observation = self.smallDrifterSet.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        assertListAlmostEqual(self, observation.tolist(), [0.1, 0.1], 6,
                              'set drifters positions, observation')


    def test_set_observation_position(self):
        self.set_positions_small_set()
        pos = np.array([0.523, 0.999])
        self.smallDrifterSet.setObservationPosition(pos)
        
        observation = self.smallDrifterSet.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        assertListAlmostEqual(self, observation.tolist(), pos.tolist(), 6,
                              'set observation, observation')

    def test_set_observation_position_unchanged_drifters(self):
        self.set_positions_small_set()
        pos = np.array([0.523, 0.999])
        self.smallDrifterSet.setObservationPosition(pos)
        
        positions = self.smallDrifterSet.getDrifterPositions()
        self.assertEqual(positions.shape, ((self.numDrifters, 2)))
        assertListAlmostEqual(self, positions[0,:].tolist(), [0.9, 0.9], 6,
                              'set observation, drifter 0')
        assertListAlmostEqual(self, positions[1,:].tolist(), [0.9, 0.1], 6,
                              'set observation, drifter 1')
        assertListAlmostEqual(self, positions[2,:].tolist(), [0.1, 0.9], 6,
                              'set observation, drifter 2')

        observation = self.smallDrifterSet.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        assertListAlmostEqual(self, observation.tolist(), pos.tolist(), 6,
                              'set observation, observation')
        
        self.assertEqual(self.smallDrifterSet.getBoundaryConditions().get(), [2,2,2,2])
        
        
        
    def test_distances(self):
        self.set_positions_small_set()
        longDiag = np.sqrt(2*0.8*0.8)
        longLine = 0.8
        shortDiag = np.sqrt(0.2*0.2 + 0.2*0.2)
        shortLine = 0.2
        semiDiag = np.sqrt(0.2*0.2 + 0.8*0.8)
                           
        
        # smallDrifterSet is initially with periodic boundary conditions
        assertListAlmostEqual(self, self.smallDrifterSet.getDistances().tolist(), \
                              [shortDiag, shortLine, shortLine], 6,
                              'distance with periodic boundaries')
        
        self.smallDrifterSet.setBoundaryConditions(Common.BoundaryConditions(1,1,1,1))
        assertListAlmostEqual(self, self.smallDrifterSet.getDistances().tolist(), \
                              [longDiag, longLine, longLine], 6,
                              'distances with non-periodic boundaries')
        
        self.smallDrifterSet.setBoundaryConditions(Common.BoundaryConditions(1,2,1,2))
        assertListAlmostEqual(self, self.smallDrifterSet.getDistances().tolist(), \
                              [semiDiag, shortLine, longLine], 6,
                              'distances with periodic boundaries in east-west')

        self.smallDrifterSet.setBoundaryConditions(Common.BoundaryConditions(2,1,2,1))
        assertListAlmostEqual(self, self.smallDrifterSet.getDistances().tolist(), \
                              [semiDiag, longLine, shortLine], 6,
                              'distances with periodic boundaries in north-south')

    def test_innovations(self):
        self.set_positions_small_set()
        zero = 0.0
        close = 0.2
        far = -0.8

        fasit = [[close, close], [close, zero], [zero, close]]
        # smallDrifterSet is initially with periodic boundary conditions
        assert2DListAlmostEqual(self, self.smallDrifterSet.getInnovations().tolist(), \
                              fasit, 6,
                              'innovations with periodic boundaries')

        fasit = [[far, far], [far, zero], [zero, far]]
        self.smallDrifterSet.setBoundaryConditions(Common.BoundaryConditions(1,1,1,1))
        assert2DListAlmostEqual(self, self.smallDrifterSet.getInnovations().tolist(), \
                              fasit, 6,
                              'innovations with non-periodic boundaries')

        fasit = [[close, far], [close, zero], [zero, far]]
        self.smallDrifterSet.setBoundaryConditions(Common.BoundaryConditions(1,2,1,2))
        assert2DListAlmostEqual(self, self.smallDrifterSet.getInnovations().tolist(), \
                              fasit, 6,
                              'innovations with periodic boundaries in east-west')

        fasit = [[far, close], [far, zero], [zero, close]]
        self.smallDrifterSet.setBoundaryConditions(Common.BoundaryConditions(2,1,2,1))
        assert2DListAlmostEqual(self, self.smallDrifterSet.getInnovations().tolist(), \
                              fasit, 6,
                              'innovations with periodic boundaries in north-south')
        
        
    def test_collection_mean(self):
        self.set_positions_small_set()
        periodicMean = [1-0.1/3, 1-0.1/3]
        nonPeriodicMean = [(0.9 + 0.9 + 0.1)/3, (0.9 + 0.9 + 0.1)/3]
        semiPeriodicMean = [nonPeriodicMean[0], periodicMean[1]]
        
        assertListAlmostEqual(self, self.smallDrifterSet.getCollectionMean().tolist(),
                              periodicMean, 6,
                              'periodic mean')

        self.smallDrifterSet.setBoundaryConditions(Common.BoundaryConditions(1,1,1,1))
        assertListAlmostEqual(self, self.smallDrifterSet.getCollectionMean().tolist(),
                              nonPeriodicMean, 6,
                              'non-periodic mean')

        self.smallDrifterSet.setBoundaryConditions(Common.BoundaryConditions(2,1,2,1))
        assertListAlmostEqual(self, self.smallDrifterSet.getCollectionMean().tolist(),
                              semiPeriodicMean, 6,
                              'north-south-periodic mean')
        
        
    def test_init_uniform_positions(self):
        
        domain_x = 10.3
        domain_y = 5.4
        self.create_large_drifter_set(1000,
                                      domain_x,
                                      domain_y)
        self.largeDrifterSet.initializeUniform()

        self.assertEqual(self.largeDrifterSet.getDomainSizeX(), domain_x)
        self.assertEqual(self.largeDrifterSet.getDomainSizeY(), domain_y)

        p = self.largeDrifterSet.getDrifterPositions()
        self.assertGreaterEqual(np.min(p[:,0]), 0.0)
        self.assertLessEqual(np.max(p[:,0]), domain_x)
        self.assertGreaterEqual(np.min(p[:,1]) , 0.0)
        self.assertLessEqual(np.max(p[:,1]), domain_y)

    def test_set_domain_size(self):
        self.set_positions_small_set()
        size_x = 10.3
        size_y = 5.4
        self.smallDrifterSet.setDomainSize(size_x, size_y)

        self.assertEqual(self.smallDrifterSet.getDomainSizeX(), size_x)
        self.assertEqual(self.smallDrifterSet.getDomainSizeY(), size_y)

        longDiag = np.sqrt(2*0.8*0.8)
        longLine = 0.8
        # Distance should now be the above, even with periodic boundary conditions
        assertListAlmostEqual(self, self.smallDrifterSet.getDistances().tolist(),
                              [longDiag, longLine, longLine], 6,
                              'getDistance() in big periodic domain')
        
        
    def test_copy(self):
        self.set_positions_small_set()
        size_x = 10.3
        size_y = 5.4
        self.smallDrifterSet.setDomainSize(size_x, size_y)
                        
        # Give non-standard domain_size before 
        copy = self.smallDrifterSet.copy()
        
        self.assertEqual(copy.getNumDrifters(), self.numDrifters)
        self.assertEqual(copy.getObservationVariance(), self.observationVariance)
        
        positions = copy.getDrifterPositions()
        self.assertEqual(positions.shape, ((self.numDrifters, 2)))
        assertListAlmostEqual(self, positions[0,:].tolist(), [0.9, 0.9], 6,
                              'copy Drifter, position drifter 0')
        assertListAlmostEqual(self, positions[1,:].tolist(), [0.9, 0.1], 6,
                              'copy Drifter, position drifter 1')
        assertListAlmostEqual(self, positions[2,:].tolist(), [0.1, 0.9], 6,
                              'copy Drifter, position drifter 2')
        
        observation = copy.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        assertListAlmostEqual(self, observation.tolist(), [0.1, 0.1], 6,
                              'copy Drifter, position observation')
        
        self.assertEqual(copy.getBoundaryConditions().get(), [2,2,2,2])

        self.assertEqual(copy.getDomainSizeX(), size_x)
        self.assertEqual(copy.getDomainSizeY(), size_y)
                                                           
        # Move a drifter in the original dataset and check that it is still the same in
        # the copy
        positions[1,0] = 0.5
        positions[1,1] = 0.5
        self.smallDrifterSet.setDrifterPositions(positions)
        
        positions2 = copy.getDrifterPositions()
        assertListAlmostEqual(self, positions2[1,:].tolist(), [0.9, 0.1], 6,
                              'copy Drifter, position drifter 1 after changing original')
        


        
    def test_gaussian_weights(self):
        self.set_positions_resampling_set()
        obtainedWeights = self.resamplingDrifterSet.getGaussianWeight()

        # ReferenceWeights based Bayes theorem with Gaussian likelihood and the positions selected for the resamplingDrifterSet.
        referenceWeights = [  3.77361928e-01,   1.22511481e-01,   1.26590824e-04,   3.77361928e-01, 1.22511481e-01,   1.26590824e-04]
        assertListAlmostEqual(self, obtainedWeights.tolist(),
                              referenceWeights, 6,
                              'gaussian weights')

    def test_cauchy_weights(self):
        self.set_positions_resampling_set()
        obtainedWeights = self.resamplingDrifterSet.getCauchyWeight()
        # ReferenceWeights based Bayes theorem with Cauchy likelihood and the positions selected for the resamplingDrifterSet.
        referenceWeights = [0.28413284,  0.16789668,  0.04797048,  0.28413284,  0.16789668,  0.04797048]
        assertListAlmostEqual(self, obtainedWeights.tolist(),
                              referenceWeights, 6,
                              'cauchy weights')

    def resample(self, indices_list):
        newDrifterPositions = []
        for i in indices_list:
            newDrifterPositions.append(self.resamplingDrifterSet.getDrifterPositions()[i,:].tolist())
        return newDrifterPositions
        
    def test_resampling_predefined_indices(self):
        self.set_positions_resampling_set()
        indices_list = [2,2,2,4,5,5]
        newDrifterPositions = self.resample(indices_list)
        self.resamplingDrifterSet.resample(indices_list, 0)
        self.assertEqual(self.resamplingDrifterSet.getDrifterPositions().tolist(), \
                         newDrifterPositions)

    def atest_probabilistic_resampling_with_duplicates(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [1,3,0,0,0,0]
        solutions = self.resample(indices)
        dautils.probabilisticResampling(self.resamplingDrifterSet)
        self.assertEqual(self.resamplingDrifterSet.getDrifterPositions().tolist(), \
                         solutions)

            
    def atest_residual_sampling_with_duplicates(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,3,3,1,4]
        solutions = self.resample(indices)
        dautils.residualSampling(self.resamplingDrifterSet)
        self.assertEqual(self.resamplingDrifterSet.getDrifterPositions().tolist(), \
                         solutions)
        
    def atest_stochastic_universal_sampling_with_duplicates(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,1,3,3,4]
        solutions = self.resample(indices)
        dautils.stochasticUniversalSampling(self.resamplingDrifterSet)
        self.assertEqual(self.resamplingDrifterSet.getDrifterPositions().tolist(), \
                         solutions)

    def atest_monte_carlo_metropolis_hasting_sampling_with_duplicates(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,0,3,4,4]
        solutions = self.resample(indices)
        dautils.metropolisHastingSampling(self.resamplingDrifterSet)
        self.assertEqual(self.resamplingDrifterSet.getDrifterPositions().tolist(), solutions)


    def atest_probabilistic_resampling(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [1,3,0,0,0,0]
        solutions = self.resample(indices)
        dautils.probabilisticResampling(self.resamplingDrifterSet, self.resamplingVar)
        assert2DListAlmostEqual(self, self.resamplingDrifterSet.getDrifterPositions().tolist(), solutions, 2, "probabilistic resampling, probabilistic duplicates")

            
    def atest_residual_sampling(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,3,3,1,4]
        solutions = self.resample(indices)
        dautils.residualSampling(self.resamplingDrifterSet, self.resamplingVar)
        assert2DListAlmostEqual(self, self.resamplingDrifterSet.getDrifterPositions().tolist(), solutions, 2, "residual sampling, probabilistic duplicates")
                
    def atest_stochastic_universal_sampling(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,1,3,3,4]
        solutions = self.resample(indices)
        dautils.stochasticUniversalSampling(self.resamplingDrifterSet, self.resamplingVar)
        assert2DListAlmostEqual(self, self.resamplingDrifterSet.getDrifterPositions().tolist(), solutions, 2, "stochastic universal sampling, probabilistic duplicates")

    def atest_monte_carlo_metropolis_hasting_sampling(self):
        self.set_positions_resampling_set()
        setNpRandomSeed()
        indices = [0,0,0,3,4,4]
        solutions = self.resample(indices)
        dautils.metropolisHastingSampling(self.resamplingDrifterSet, self.resamplingVar)
        #print self.resamplingDrifterSet.getDrifterPositions().tolist()
        assert2DListAlmostEqual(self, self.resamplingDrifterSet.getDrifterPositions().tolist(), solutions, 2, "metropolis hasting sampling, probabilistic duplicates")
        

