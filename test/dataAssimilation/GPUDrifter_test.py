# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements unit tests for the GPUDrifter class.

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

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.GPUDrifterCollection import *
from dataAssimilation.BaseDrifterTest import BaseDrifterTest



class GPUDrifterTest(BaseDrifterTest):

    def setUp(self):
        super(GPUDrifterTest, self).setUp()
        self.gpu_ctx = Common.CUDAContext()
        
    def tearDown(self):
        super(GPUDrifterTest, self).tearDown()
        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
        gc.collect()
        
    def create_small_drifter_set(self):
        self.smallDrifterSet = GPUDrifterCollection(self.gpu_ctx,
                                                    self.numDrifters,
                                                    self.observationVariance,
                                                    self.boundaryCondition)

    def create_resampling_drifter_set(self):
        self.resamplingDrifterSet = GPUDrifterCollection(self.gpu_ctx,
                                                         self.resampleNumDrifters)
        
    def create_large_drifter_set(self, size, domain_x, domain_y):
        self.largeDrifterSet = GPUDrifterCollection(self.gpu_ctx, size, domain_size_x=domain_x, domain_size_y=domain_y) 
        



