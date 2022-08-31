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
        
    def __init__(self, gpu_ctx, sim_argses, data_args, levels, numParticles_per_level, observation_variance_per_level=None):
        
        self.gpu_ctx = gpu_ctx
        self.ensembles = [None]*len(levels)
        
        self.L = len(levels)

        assert (self.L == len(sim_argses))
        assert (self.L == len(numParticles_per_level))
        if observation_variance_per_level is not None:
            assert (self.L == len(observation_variance_per_level))
        else: 
            observation_variance_per_level = [0.01**2] * self.L

        for l in range(self.L):
            if levels[l] != 1.0:
                data_args_l = NetCDFInitialization.removeMetadata(NetCDFInitialization.rescaleInitialConditions(data_args, scale=levels[l]))
            else:
                data_args_l = NetCDFInitialization.removeMetadata(data_args)
            self.ensembles[l] = OceanModelEnsemble.OceanModelEnsemble(gpu_ctx, sim_argses[l], data_args_l, numParticles_per_level[l], observation_variance=observation_variance_per_level[l])

    def __del__(self):
        for l in self.L:
            self.ensembles[l].cleanUp()
        self.gpu_ctx = None


    def stepToObservation(self, observation_time, write_now=False):
        """
        Advance the ensemble to the given observation time, and mimics CDKLM16.dataAssimilationStep function
        
        Arguments:
            observation_time: The end time for the simulation
            write_now: Write result to NetCDF if an writer is active.
        """
        for l in range(self.L):
            self.ensembles[l].stepToObservation(observation_time, write_now=write_now)