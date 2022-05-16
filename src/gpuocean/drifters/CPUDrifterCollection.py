# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements a DrifterCollection living on the CPU.

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

from SWESimulators import Common
from SWESimulators import BaseDrifterCollection

class CPUDrifterCollection(BaseDrifterCollection.BaseDrifterCollection):
    """
    Class holding the collection of drifters.
    """ 
    def __init__(self, numDrifters, observation_variance=0.01,
                 boundaryConditions=Common.BoundaryConditions(), 
                 initialization_cov_drifters=None,
                 domain_size_x=1.0, domain_size_y=1.0):
        """
        Creates a GlobalParticles object for drift trajectory ensemble.

        numDrifters: number of drifters in the collection, not included the observation
        observation_variance: uncertainty of observation position
        boundaryConditions: BoundaryConditions object, relevant during re-initialization of particles.    
        """
        
        # Call parent constructor
        super(CPUDrifterCollection, self).__init__(numDrifters,
                                         observation_variance=observation_variance,
                                         boundaryConditions=boundaryConditions,
                                         domain_size_x=domain_size_x, 
                                         domain_size_y=domain_size_y)
        
        # One position for every particle plus observation
        self.positions = np.zeros((self.numDrifters + 1, 2))
        
        # Initialize drifters:
        self.uniformly_distribute_drifters(initialization_cov_drifters=initialization_cov_drifters)
        
    def copy(self):
        """
        Makes an independent indentical copy of the current object
        """
    
        copyOfSelf = CPUDrifterCollection(self.numDrifters,
                                observation_variance = self.observation_variance,
                                boundaryConditions = self.boundaryConditions,
                                domain_size_x = self.domain_size_x, 
                                domain_size_y = self.domain_size_y)
        copyOfSelf.positions = self.positions.copy()
        
        return copyOfSelf
    
    def cleanUp(self):
        pass
    
    ### Implementation of abstract GETs
    
    def getDrifterPositions(self):
        return self.positions[:-1,:].copy()
    
    def getObservationPosition(self):
        return self.positions[-1, :].copy()
    
    
    
    ### Implementation of abstract GETs
    
    def setDrifterPositions(self, newDrifterPositions):
        np.copyto(self.positions[:-1,:], newDrifterPositions) 
        # Signature of copyto: np.copyto(dst, src)
    
    def setObservationPosition(self, newObservationPosition):
        np.copyto(self.positions[-1,:], newObservationPosition)
        
    ### Implementation of other abstract functions
    
    def _enforceBoundaryConditionsOnPosition(self, x, y):
        """
        Maps the given coordinate to a coordinate within the domain. This function assumes that periodic boundary conditions are used, and should be considered as a private function.
        """
        ### TODO: SWAP the if's with while's?
        # Check what we assume is periodic boundary conditions
        if x < 0:
            x = self.domain_size_x + x
        if y < 0:
            y = self.domain_size_x + y
        if x > self.domain_size_x:
            x = x - self.domain_size_x
        if y > self.domain_size_y:
            y = y - self.domain_size_y
        return x, y
    
    
    def enforceBoundaryConditions(self):
        """
        Enforces boundary conditions on all particles in the ensemble, and the observation.
        This function should be called whenever particles are moved, to enforce periodic boundary conditions for particles that have left the domain.
        """
        
        if (self.boundaryConditions.isPeriodicNorthSouth() and self.boundaryConditions.isPeriodicEastWest()):
            # Loop over particles
            for i in range(self.getNumDrifters() + 1):
                x, y = self.positions[i,0], self.positions[i,1]

                x, y = self._enforceBoundaryConditionsOnPosition(x,y)

                self.positions[i,0] = x
                self.positions[i,1] = y
        else:
            # TODO: what does this mean in a non-periodic boundary condition world?
            #print "WARNING [GlobalParticle.enforceBoundaryConditions]: Functionality not defined for non-periodic boundary conditions"
            #print "\t\tDoing nothing and continuing..."
            pass
    
    
    
  