# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements unit tests for the CPUDrifterCollection class.

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
from SWESimulators.CPUDrifterCollection import *
from dataAssimilation.BaseDrifterTest import BaseDrifterTest

class CPUDrifterTest(BaseDrifterTest):

    def setUp(self):
        super(CPUDrifterTest, self).setUp()

    def tearDown(self):
        pass
    

    def create_small_drifter_set(self):
        self.smallDrifterSet = CPUDrifterCollection(self.numDrifters,
                                                     self.observationVariance,
                                                     self.boundaryCondition)

    def create_resampling_drifter_set(self):
        self.resamplingDrifterSet = CPUDrifterCollection(self.resampleNumDrifters)

    def create_large_drifter_set(self, size, domain_x, domain_y):
        self.largeDrifterSet = CPUDrifterCollection(size, domain_size_x=domain_x, domain_size_y=domain_y)

