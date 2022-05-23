# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements unit tests for the DrifterEnsemble class.

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
from dataAssimilation.BaseDrifterEnsembleTest import BaseDrifterEnsembleTest

class DrifterEnsembleTest(BaseDrifterEnsembleTest):
    #__metaclass__ = abc.ABCMeta
    
    def setUp(self):
        super(DrifterEnsembleTest, self).setUp()
        
        self.gpu_ctx = Common.CUDAContext()
                
    def tearDown(self):
        super(DrifterEnsembleTest, self).tearDown()

    ### Define required functions as abstract ###

    #@abc.abstractmethod
    def create_small_particle_set(self):
        self.smallParticleSet = DrifterEnsemble.DrifterEnsemble(self.gpu_ctx,
                                                                self.numParticles,
                                                                self.observationVariance)
        self.smallParticleSet.setGridInfo(self.nx, self.ny, self.dx, self.dy, self.dt,
                                          self.boundaryCondition)
        self.smallParticleSet.setParameters()
        self.smallParticleSet.init()
        

    #@abc.abstractmethod
    def create_resampling_particle_set(self):
        self.resamplingParticleSet = DrifterEnsemble.DrifterEnsemble(self.gpu_ctx,
                                                                     self.resampleNumParticles,
                                                                     observation_variance=self.resamplingObservationVariance)
        self.resamplingParticleSet.setGridInfo(self.nx, self.ny, self.dx, self.dy, self.dt)
        self.resamplingParticleSet.setParameters()
        self.resamplingParticleSet.init()
        

    #@abc.abstractmethod
    def create_large_particle_set(self, size, domain_x, domain_y):
        largeParticleSet = DrifterEnsemble.DrifterEnsemble(self.gpu_ctx,
                                                           size)
        largeParticleSet.setGridInfo(10, 10, domain_x/10.0, domain_y/10.0, self.dt,
                                               self.boundaryCondition)
        largeParticleSet.setParameters()
        largeParticleSet.init()
        return largeParticleSet

