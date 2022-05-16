# -*- coding: utf-8 -*-

"""
This python class implements an ensemble of double jet cases

Copyright (C) 2019  SINTEF Digital

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


from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import abc
import warnings 


import pycuda.driver as cuda

from SWESimulators import CDKLM16
from SWESimulators import GPUDrifterCollection
from SWESimulators import OceanNoiseEnsemble
from SWESimulators import BaseOceanStateEnsemble
from SWESimulators import Common
from SWESimulators import DoubleJetCase
from SWESimulators import DataAssimilationUtils as dautils



try:
    from importlib import reload
except:
    pass

reload(BaseOceanStateEnsemble)
reload(OceanNoiseEnsemble)

class DoubleJetEnsemble(OceanNoiseEnsemble.OceanNoiseEnsemble):
    """
    Class that holds an ensemble of ocean states initialized from the double 
    jet case.
    
    It inherits OceanNoiseEnsemble, but the only function from OceanNoiseEnsemble
    that is active in this class is the resample function. All other inherited
    functions are from BaseOceanStateEnsemble.
    """
    
    def __init__(self, gpu_ctx, numParticles, doubleJetCase,
                 num_drifters = 1,
                 observation_type=dautils.ObservationType.DrifterPosition,
                 observation_variance = None, 
                 observation_variance_factor = 5.0,
                 initialization_variance_factor_drifter_position = 0.0,
                 initialization_variance_factor_ocean_field = 0.0):
        
        assert(doubleJetCase.__class__.__name__=="DoubleJetCase"), \
            'This class can only be used with a DoubleJetCase object, and not a Simulator'
        
        # Create a simulator from the DoubleJetCase object
        self.doubleJetCase = doubleJetCase
        base_init_args, base_init_cond = doubleJetCase.getBaseInitConditions()
        tmp_sim = CDKLM16.CDKLM16(**base_init_args, **base_init_cond)
        
        # Call super class:
        print('Calling parent constructor from DoubleJetEnsemble')
        super(DoubleJetEnsemble, self).__init__(gpu_ctx, numParticles, tmp_sim,
                                                num_drifters,
                                                observation_type, 
                                                observation_variance,
                                                observation_variance_factor,
                                                initialization_variance_factor_drifter_position,
                                                initialization_variance_factor_ocean_field)
                                                
                                                
    
    def _init(self, driftersPerOceanModel=1):
        
        for i in range(self.numParticles+1):
            
            particle_args, particle_init = self.doubleJetCase.getInitConditions()
            self.particles[i] = CDKLM16.CDKLM16(**particle_args, **particle_init)
            
            if self.doubleJetCase.perturbation_type == DoubleJetCase.DoubleJetPerturbationType.ModelErrorPerturbation:
                self.particles[i].perturbState(q0_scale=20)
            
            if self.doubleJetCase.perturbation_type == DoubleJetCase.DoubleJetPerturbationType.SpinUp or \
               self.doubleJetCase.perturbation_type == DoubleJetCase.DoubleJetPerturbationType.LowFrequencySpinUp or \
               self.doubleJetCase.perturbation_type == DoubleJetCase.DoubleJetPerturbationType.LowFrequencyStandardSpinUp:
                self.particles[i].step(self.doubleJetCase.individualSpinUpTime)
                print('Individual spin up for particle ' + str(i))
            elif self.doubleJetCase.perturbation_type == DoubleJetCase.DoubleJetPerturbationType.NormalPerturbedSpinUp:
                self.particles[i].step(self.doubleJetCase.commonSpinUpTime*2, apply_stochastic_term=False)
                self.particles[i].step(self.doubleJetCase.individualSpinUpTime*3)
                print('Individual spin up for particle ' + str(i))
                
        # Initialize and attach drifters to all particles.
        self._TO_DELETE_initialize_drifters(driftersPerOceanModel)
        
        # Create gpu kernels and buffers:
        self._setupGPU()
        
        # Put the initial positions into the observation array
        self._addObservation(self.observeTrueDrifters())

 