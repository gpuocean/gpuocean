# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements a Ensemble of particles living on the CPU,
each consisting of a single drifter in its own ocean state. The 
perturbation parameter is the wind direction.

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

from SWESimulators import CDKLM16
from SWESimulators import CPUDrifterCollection
from SWESimulators import Common
from SWESimulators import DataAssimilationUtils as dautils
from SWESimulators import BaseDrifterEnsemble


class CPUDrifterEnsemble(BaseDrifterEnsemble.BaseDrifterEnsemble):
        
    def __init__(self, numParticles, observation_variance=0.0):
         
        super(CPUDrifterEnsemble, self).__init__(numParticles, 
                                              observation_variance)
        
    
    # ---------------------------------------
    # Implementing abstract function
    # ---------------------------------------
    def init(self):

        self.sim = None

        self.drifters = CPUDrifterCollection.CPUDrifterCollection(self.numParticles,
                      observation_variance=self.observation_variance,
                      boundaryConditions=self.boundaryConditions,
                      domain_size_x=self.nx*self.dx, domain_size_y=self.ny*self.dy)
        
        self.drifters.initializeUniform()
    
    #--------------------
    ## Override
    #--------------------
    def step(self, T, eta=None, hu=None, hv=None, sensitivity=1):
        # Change positions by reference
        positions = self.drifters.positions
        doPrint = False
        if eta is None:
            eta = self.base_eta
        if hu is None:
            hu = self.base_hu
        if hv is None:
            hv = self.base_hv
       
        # Based on periodic boundary conditions on CDKLM:
        x_zero_ref = 2
        y_zero_ref = 2
        
        numParticles = positions.shape[0]
        
        t = 0
        while t < T:
            # Loop over particles
            for i in range(numParticles):
                if doPrint: print("---------- Particle " + str(i) + " ---------------")
                x0, y0 = positions[i,0], positions[i,1]
                if doPrint: print("(x0, y0): " + str((x0,y0)))

                # First, find which cell each particle is in

                # In x-direction:
                cell_id_x = int(np.ceil(x0/self.dx) + x_zero_ref)
                cell_id_y = int(np.ceil(y0/self.dy) + y_zero_ref)

                if (cell_id_x < 0 or cell_id_x > self.nx + 4 or cell_id_y < 0 or cell_id_y > self.ny + 4):
                    print("ERROR! Cell id " + str((cell_id_x, cell_id_y)) + " is outside of the domain!")
                    print("\t\Particle position is: " + str((x0, y0)))

                if doPrint: print("cell values in x-direction: " + str(((cell_id_x-2-0.5)*self.dx, (cell_id_x-2+0.5)*self.dx) ))
                if doPrint: print("cell values in y-direction: " + str(((cell_id_y-2-0.5)*self.dy, (cell_id_y-2+0.5)*self.dy) ))
                    
                waterHeight = (  self.base_H[cell_id_y  , cell_id_x  ]
                               + self.base_H[cell_id_y+1, cell_id_x  ]
                               + self.base_H[cell_id_y  , cell_id_x+1]
                               + self.base_H[cell_id_y+1, cell_id_x+1] )*0.25

                h = waterHeight + eta[cell_id_y, cell_id_x]
                u = hu[cell_id_y, cell_id_x]/h
                v = hv[cell_id_y, cell_id_x]/h

                if doPrint: print("Velocity: " + str((u, v)))

                x1 = sensitivity*u*self.dt + x0
                y1 = sensitivity*v*self.dt + y0
                if doPrint: print("(x1, y1): " + str((positions[i,0], positions[i,1])))

                positions[i,0] = x1
                positions[i,1] = y1


            # Check what we assume is periodic boundary conditions    
            self.drifters.enforceBoundaryConditions()
            t += self.dt 
            
    #-------------------
    ### NEW
    #-------------------
    def copy(self):
        copy = CPUDrifterEnsemble(self.numParticles, self.observation_variance)
        copy.setGridInfo(self.nx, self.ny, self.dx, self.dy, self.dt,
                         self.boundaryConditions,
                         self.base_eta, self.base_hu, self.base_hv, self.base_H)
        copy.setParameters(self.f, self.g, self.beta, self.r, self.wind)
        copy.init()
        copy.setParticleStates(self.observeParticles())
        copy.setObservationState(self.observeTrueState())
        return copy