# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean.

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This python class implements an ensemble of particles, each consisting
of a single drifter in its own ocean state. The perturbation parameter 
is the wind direction.

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
from SWESimulators import GPUDrifterCollection
from SWESimulators import BaseOceanStateEnsemble
from SWESimulators import Common
from SWESimulators import WindStress
from SWESimulators import DataAssimilationUtils as dautils


class WindForcingEnsemble(BaseOceanStateEnsemble.BaseOceanStateEnsemble):
        
    
    def init(self, driftersPerOceanModel=1):
        self.windSpeed = 2.0
        self.directions = np.random.rand(self.numParticles + 1)*360
        self.windX, self.windY = self.XandYfromDirections(self.directions)
        #print "Directions: ", self.directions
        self.driftersPerOceanModel = driftersPerOceanModel
        
        self.windT = np.zeros((1), dtype=np.float32)
        
        for i in range(self.numParticles+1):
            
            wX = [self.windX[i]*np.ones((2,2), dtype=np.float32)]
            wY = [self.windY[i]*np.ones((2,2), dtype=np.float32)]
            
            wind = WindStress.WindStress(self.windT, wX, wY)
            #print ("Init with wind :", (wX, wY))
            
            self.particles[i] = CDKLM16.CDKLM16(self.gpu_ctx, \
                                                self.base_eta, self.base_hu, self.base_hv, \
                                                self.base_H, \
                                                self.nx, self.ny, self.dx, self.dy, self.dt, \
                                                self.g, self.f, self.r, \
                                                wind_stress=wind, \
                                                boundary_conditions=self.boundaryConditions, \
                                                write_netcdf=False)
            if i == self.numParticles:
                # All particles done, only the observation is left,
                # and for the observation we only use one drifter, regardless of the
                # number in the other particles.
                driftersPerOceanModel = 1
            
            drifters = GPUDrifterCollection.GPUDrifterCollection(self.gpu_ctx, driftersPerOceanModel,
                                             observation_variance=self.observation_variance,
                                             boundaryConditions=self.boundaryConditions,
                                             domain_size_x=self.nx*self.dx, domain_size_y=self.ny*self.dy)
            initPos = np.random.multivariate_normal(self.midPoint, self.initialization_cov_drifters, driftersPerOceanModel)
            drifters.setDrifterPositions(initPos)
            #print "drifter particles: ", drifter.getParticlePositions()
            #print "drifter observations: ", drifter.getObservationPosition()
            self.particles[i].attachDrifters(drifters)
        
        # Put the initial positions into the observation array
        self._addObservation(self.observeTrueDrifters())
        print("Added init to observation array")
        
    def XandYfromDirections(self, directions):
        windX = self.windSpeed * np.sin(directions*(2*np.pi)/360.0)
        windY = self.windSpeed * np.cos(directions*(2*np.pi)/360.0)
        return windX, windY

    def resample(self, newSampleIndices, reinitialization_variance):
        obsTrueDrifter = self.observeTrueDrifters()
        positions = self.observeDrifters()
        windDirection = self.directions
        newWindDirection = np.empty_like(windDirection)
        newPos = np.empty((self.driftersPerOceanModel, 2))
        newOceanStates = [None]*self.getNumParticles()
        for i in range(self.getNumParticles()):
            index = newSampleIndices[i]
            #print "(particle no, position, old direction, new direction): "
            newWindDirection[i] = np.random.normal(windDirection[index], reinitialization_variance, 1)
            if newWindDirection[i] > 360:
                newWindDirection[i] -= 360
            elif newWindDirection[i] < 0:
                newWindDirection[i] += 360
            newPos[:,:] = positions[index,:]
            #print "\t", (index, positions[index,:], windDirection[index])
            #print "\t", (index, newPos, newWindDirection[i])
            
            wX, wY = self.XandYfromDirections(newWindDirection[i])
            wX = [wX*np.ones((2,2), dtype=np.float32)]
            wY = [wY*np.ones((2,2), dtype=np.float32)]
            newWindInstance = WindStress.WindStress(self.windT, wX, wY)
            
            # Download index's ocean state:
            eta0, hu0, hv0 = self.particles[index].download()
            eta1, hu1, hv1 = self.particles[index].downloadPrevTimestep()
            newOceanStates[i] = (eta0, hu0, hv0, eta1, hu1, hv1)
            
            self.particles[i].wind_stress = newWindInstance
            self.particles[i].drifters.setDrifterPositions(newPos)

        self.directions = newWindDirection.copy()
        
        # New loop for transferring the correct ocean states back up to the GPU:
        for i in range(self.getNumParticles()):
            self.particles[i].upload(newOceanStates[i][0],
                                     newOceanStates[i][1],
                                     newOceanStates[i][2],
                                     newOceanStates[i][3],
                                     newOceanStates[i][4],
                                     newOceanStates[i][5])
                    
   