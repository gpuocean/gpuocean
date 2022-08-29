# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements an ensemble class of BaseOceanStateEnsembles.

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

import numpy as np
import time


from gpuocean.ensembles import OceanModelEnsemble
from gpuocean.utils import Common, WindStress, NetCDFInitialization

class MulitResOceanStateEnsemble():
    """
    Class for ensembles of ensembles of ocean states.
    """
        
    def __init__(self, gpu_ctx, sim_args, data_args, levels, numParticles_per_level):
        
        self.gpu_ctx = gpu_ctx
        self.ensembles = [None]*len(levels)
        
        assert (len(levels) == len(numParticles_per_level))

        for l in range(len(levels)):
            data_args_l = NetCDFInitialization.rescaleInitialConditions(data_args, scale=levels[l])
            self.ensembles[l] = OceanModelEnsemble.OceanModelEnsemble(gpu_ctx, sim_args, data_args_l, numParticles_per_level[l])
