# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018, 2023, 2024  SINTEF Digital

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
from gpuocean.drifters import CPUDrifterCollection

class MLDrifterCollection(CPUDrifterCollection.CPUDrifterCollection):
    """
    Class holding a collection of drifters for a 
    """ 
    def __init__(self, numDrifters, ensemble_size, observation_variance=0.01,
                 boundaryConditions=Common.BoundaryConditions(), 
                 initialization_cov_drifters=None,
                 domain_size_x=1.0, domain_size_y=1.0,
                 use_biased_walk=True):
        """
        Creates a collection of drifters suitable for multi-level (ML) ensembles.

        Most relevant parameters
        numDrifters: number of unique drifters represented in the collection
        ensemble_size: number of realization per drifter. 
        boundaryConditions: BoundaryConditions object, relevant during drift
        domain_size_{x,y}: size of computational domain in meters
        use_biased_walk: Flag to use biased walk (True by default) or random walk (False) 
        """
        
        # Call parent constructor
        super(MLDrifterCollection, self).__init__(numDrifters*ensemble_size,
                                         observation_variance=observation_variance,
                                         boundaryConditions=boundaryConditions,
                                         domain_size_x=domain_size_x, 
                                         domain_size_y=domain_size_y)
        
        self.ensemble_size = ensemble_size
        self.num_unique_drifters = numDrifters
        
        # To initialize drifters uniformly (default behaviour of the other DrifterCollections)
        # we need to make a temporary drifter object
        initializedDrifters = CPUDrifterCollection.CPUDrifterCollection(numDrifters, 
                                                                        boundaryConditions=boundaryConditions,
                                                                        initialization_cov_drifters=initialization_cov_drifters,
                                                                        domain_size_x=domain_size_x,
                                                                        domain_size_y=domain_size_y)
        
        # drifter data is organized by storing the position of all ensemble members representing the same drifter
        # in consecutive blocks.
        init_positions = initializedDrifters.getDrifterPositions()
        self.positions[:-1, :] = np.repeat(init_positions, self.ensemble_size, axis=0)
        
        # This flag describes where in the gaussian random walk distribution we should go
        # If true, we assign random directions for each drifter to walk, and this direction will
        # be scaled by the variance field used in the random walk. If false, the random direction will
        # be random for each drift step.
        self.use_biased_walk = use_biased_walk
        self.biased_walk_x = None
        self.biased_walk_y = None
        if self.use_biased_walk:
            self.biased_walk_x = np.random.normal(loc=0, scale=1, size=(self.numDrifters+1))
            self.biased_walk_y = np.random.normal(loc=0, scale=1, size=(self.numDrifters+1))
           
    
    # Mappings between drifters and ensemble members
    def expandDrifterPositions(self, pos):
        """
        Given a position per unique drifter as input, we return an array with an ensemble of the exact same position
        input.shape: (numDrifters, 2), output.shape: (numDrifters*ensemble_size, 2)
        """
        return np.repeat(pos, self.ensemble_size, axis=0)
    
    def getDrifterPositionsForDrifter(self, drifter_index):
        """
        Gives the positions corresponding to the give drifter.
        Returns array of shape (ensemble_size, 2)
        """
        assert(drifter_index >= 0), "drifter_index must be positive, but got " +str(drifter_index)
        assert(drifter_index < self.num_unique_drifters), "drifter_index must be smaller than number of unique drifters ("+str(self.num_unique_drifters)+"), but got " +str(drifter_index)
        
        pos = self.positions[drifter_index*self.ensemble_size:(drifter_index + 1)*self.ensemble_size, :].copy()
        assert(pos.shape == (self.ensemble_size, 2)), "Expected data for "+str(self.ensemble_size)+"drifters, but only got "+pos.shape[0]
        return pos

    def getDrifterPositionsForEnsembleMember(self, ensemble_member):
        """
        Gives the positions of all drifters for a given ensemble member
        Returns array of shape (num_unique_drifters, 2)
        """
        assert(ensemble_member >= 0), "ensemble_member must be positive, but got " +str(ensemble_member)
        assert(ensemble_member < self.ensemble_size), "drifter_index must be smaller than the ensemble size ("+str(self.ensemble_size)+"), but got " +str(ensemble_member)

        pos = self.positions[ensemble_member:-1:self.ensemble_size, :].copy()
        assert(pos.shape == (self.num_unique_drifters, 2)), "Expected data for "+str(self.num_unique_drifters)+"drifters, but only got "+pos.shape[0]
        return pos

    # Overloading other functions

    def setDrifterPositions(self, newDrifterPositions):
        """ 
        new fixed positions for drifters
        """
        if newDrifterPositions.shape[0] == self.numDrifters:
            return super().setDrifterPositions(newDrifterPositions)
        elif newDrifterPositions.shape[0] == self.num_unique_drifters:
            return super().setDrifterPositions(self.expandDrifterPositions(newDrifterPositions))
        
    def drift(self, u_field, v_field, dx, dy, dt, 
              x_zero_ref=0, y_zero_ref=0, 
              u_var=None, v_var=None, sensitivity=1.0):
        """
        Evolve all drifters with a simple Euler step.
        Velocities are interpolated from the fields
        
        {x,y}_zero_ref points to which cell has face values {x,y} = 0. 
        {u, v}_field are mean fields for u and v
        {u,v}_var are variance fields and provide a random walk on top of the drift
        """

        assert(u_var is not None and v_var is not None), "u_var and v_var must be provided for the MLDrifterCollection class to make sense"

        if self.boundaryConditions.isPeriodic() and x_zero_ref == 0 and y_zero_ref == 0:
            # Ensure that we have a periodic halo so that we can interpolate through
            # periodic boundary
            u_field  = self._expandPeriodicField(u_field)
            v_field  = self._expandPeriodicField(v_field)
            u_var    = self._expandPeriodicField(u_var)
            v_var    = self._expandPeriodicField(v_var)
            x_zero_ref = 1
            y_zero_ref = 1

        self.driftFromVelocities(u_field, v_field, dx, dy, dt, 
                   x_zero_ref=x_zero_ref, y_zero_ref=y_zero_ref, 
                   u_var=u_var, v_var=v_var, sensitivity=sensitivity)
        
    def _expandPeriodicField(self, field):
        """
        Put a halo of periodic values of one grid cell around the given field
        """
        ny, nx = field.shape
        exp_field = np.zeros((ny+2, nx+2))
        exp_field[1:-1, 1:-1] = field
        exp_field[ 0,  :] = exp_field[-2,  :]
        exp_field[-1,  :] = exp_field[ 1,  :]
        exp_field[ :,  0] = exp_field[ :, -2]
        exp_field[ :, -1] = exp_field[ :,  1]
        return exp_field
    
    def _randomWalk(self, x, y, u, v, u_var_val, v_var_val, dt, i, sensitivity):
        if self.use_biased_walk:
            x = x + sensitivity*(u*dt + self.biased_walk_x[i]*np.sqrt(u_var_val)*dt)
            y = y + sensitivity*(v*dt + self.biased_walk_y[i]*np.sqrt(v_var_val)*dt)
        else:
            x = x + sensitivity*(u*dt + np.random.normal(loc=0, scale=np.sqrt(u_var_val)*dt))
            y = y + sensitivity*(v*dt + np.random.normal(loc=0, scale=np.sqrt(v_var_val)*dt))
        return x, y
