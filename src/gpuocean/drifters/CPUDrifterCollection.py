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

import numpy as np

from gpuocean.utils import Common
from gpuocean.drifters import BaseDrifterCollection

class CPUDrifterCollection(BaseDrifterCollection.BaseDrifterCollection):
    """
    Class holding the collection of drifters.
    """ 
    def __init__(self, numDrifters, observation_variance=0.01,
                 boundaryConditions=Common.BoundaryConditions(), 
                 initialization_cov_drifters=None,
                 domain_size_x=1.0, domain_size_y=1.0,
                 initialize=False):
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
        if initialize:
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


    ### Drift functions

    def _interpolate(self, field, cell_id_x0, cell_id_x1, cell_id_y0, cell_id_y1, x_factor, y_factor):
            x0y0 = field[cell_id_y0, cell_id_x0]
            x1y0 = field[cell_id_y0, cell_id_x1]
            x0y1 = field[cell_id_y1, cell_id_x0]
            x1y1 = field[cell_id_y1, cell_id_x1]

            y0 = (1-x_factor)*x0y0 + x_factor * x1y0
            y1 = (1-x_factor)*x0y1 + x_factor * x1y1

            return (1-y_factor)*y0 + y_factor *y1

    def driftFromVelocities(self, u_field, v_field, dx, dy, dt, 
                   x_zero_ref=0, y_zero_ref=0, 
                   u_var=None, v_var=None, sensitivity=1.0):
        """
        Step drifters using values for u and v directly.
        Evolve all drifters with a simple Euler step.
        Velocities are interpolated from the fields
        
        {x,y}_zero_ref points to which cell has face values {x,y} = 0. 
        {u,v}_stddev can be None, scalars or fields and provide a random walk
        """

        for i in range(self.getNumDrifters() + 1):
            x, y = self.positions[i,0], self.positions[i,1]

            cell_id_x = int(np.floor(x/dx + x_zero_ref)) 
            cell_id_y = int(np.floor(y/dy + y_zero_ref))

            frac_x = x/dx - np.floor(x/dx)  
            frac_y = y/dy - np.floor(y/dy)

            cell_id_x0 = cell_id_x - 1 if frac_x < 0.5 else cell_id_x 
            x_factor = frac_x + 0.5 if frac_x < 0.5 else frac_x - 0.5
            cell_id_x1 = cell_id_x0 + 1
            cell_id_y0 = cell_id_y - 1 if frac_y < 0.5 else cell_id_y
            y_factor = frac_y + 0.5 if frac_y < 0.5 else frac_y - 0.5
            cell_id_y1 = cell_id_y0 + 1

            u = self._interpolate(u_field, cell_id_x0, cell_id_x1, cell_id_y0, cell_id_y1, x_factor, y_factor)
            v = self._interpolate(v_field, cell_id_x0, cell_id_x1, cell_id_y0, cell_id_y1, x_factor, y_factor)
            
            if u_var is None and v_var is None:
                x = x + sensitivity*u*dt
                y = y + sensitivity*v*dt
            else:
                u_var_val = 0.0
                v_var_val = 0.0
                if np.isscalar(u_var):
                    u_var_val = u_var
                else:
                    u_var_val = max(0.0, self._interpolate(u_var, cell_id_x0, cell_id_x1, cell_id_y0, cell_id_y1, x_factor, y_factor))

                if np.isscalar(v_var):
                    v_var_val = v_var
                else:
                    v_var_val = max(0.0, self._interpolate(v_var, cell_id_x0, cell_id_x1, cell_id_y0, cell_id_y1, x_factor, y_factor))

                x, y = self._randomWalk(x, y, u, v, u_var_val, v_var_val, dt, i, sensitivity)

            x, y = self._enforceBoundaryConditionsOnPosition(x,y)

            assert(not np.isnan(x)), "new x is NaN after enforcing boundary conditions"
            assert(not np.isnan(y)), "new y is NaN after enforcing boundary conditions"
                
            self.positions[i,0] = x
            self.positions[i,1] = y

    def _randomWalk(self, x, y, u, v, u_var_val, v_var_val, dt, i, sensitivity):
        x = x + sensitivity*(u*dt + np.random.normal(loc=0, scale=np.sqrt(u_var_val)*dt))
        y = y + sensitivity*(v*dt + np.random.normal(loc=0, scale=np.sqrt(v_var_val)*dt))
        return x, y

    def drift(self, eta, hu, hv, Hm, dx, dy, dt, 
              x_zero_ref=0, y_zero_ref=0, 
              u_var=None, v_var=None, sensitivity=1.0):
        """
        Evolve all drifters with a simple Euler step.
        Velocities are interpolated from the fields
        
        {x,y}_zero_ref points to which cell has face values {x,y} = 0. 
        {u,v}_stddev can be None, scalars or fields and provide a random walk
        """

        # Velocities from momentums
        u_field = hu/(eta + Hm)
        v_field = hv/(eta + Hm)

        self.driftFromVelocities(u_field, v_field, dx, dy, dt, 
                   x_zero_ref=x_zero_ref, y_zero_ref=y_zero_ref, 
                   u_var=u_var, v_var=v_var, sensitivity=sensitivity)

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
    
    
    
  