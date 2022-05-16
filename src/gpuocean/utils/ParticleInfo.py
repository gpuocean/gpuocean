# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2019 SINTEF Digital

This python module implements a class that read and writes information 
about the ocean state at drifter positions for particles that are 
part of an ensemble. This data can later be used for analysing the 
quality of the given ensemble method.

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
import pandas as pd


from SWESimulators import CDKLM16
from SWESimulators import GPUDrifterCollection
from SWESimulators import DataAssimilationUtils as dautils




class ParticleInfo:
    """
    Class for creating and reading ocean state information of particles.
    """
    
    def __init__(self):
        """
        Class for facilitating the simulated particle state at drifter positions
        for particles in an ensemble.
        """
        self.columns = ('time', 'state_under_drifter', 'extra_states')
        self.state_df = pd.DataFrame(columns=self.columns)
        
        # Configuration parameters:
        self.extraCells = None
        
        
        
    def get_num_samples(self):
        """
        Returns the number of rows (state samples) stored in the DataFrame.
        """
        return self.state_df[self.columns[0]].count()
    
    def get_num_drifters(self):
        """
        Returns the number of drifters used in the state samples set.
        """
        
        first_position = self.state_df.iloc[0][self.columns[1]]
        return first_position.shape[0]
    
    def get_num_extra_cells(self):
        if self.extraCells is None:
            return 0
        return self.extraCells.shape[0]
    
    def add_state_sample_from_sim(self, sim, drifter_cells):
        """
        Adds ocean state sample from the drifter positions to the state DataFrame.
        """
        
        # The timestamp is rounded to nearest integer, so that it is possible to compare to 
        # other simulation times.
        rounded_sim_t = round(sim.t)
        index = self.get_num_samples()
        
        if not index == 0:
            assert(self.state_df[self.state_df[self.columns[0]]==rounded_sim_t].time.count() == 0), \
                "State sample for time " + str(rounded_sim_t) + " already exists in DataFrame"
        
        num_drifters = drifter_cells.shape[0]
        eta, hu, hv = sim.download(interior_domain_only=True)

        state_sample = np.zeros((num_drifters, 3))
        state_sample[:,0] = eta[drifter_cells[:,1], drifter_cells[:,0]]
        state_sample[:,1] =  hu[drifter_cells[:,1], drifter_cells[:,0]]
        state_sample[:,2] =  hv[drifter_cells[:,1], drifter_cells[:,0]]  
        
        extra_sample = None
        if self.extraCells is not None:
            num_extra_cells = self.extraCells.shape[0]
            extra_sample = np.zeros((num_extra_cells, 3))
            extra_sample[:,0] = eta[self.extraCells[:,1], self.extraCells[:,0]]
            extra_sample[:,1] =  hu[self.extraCells[:,1], self.extraCells[:,0]]
            extra_sample[:,2] =  hv[self.extraCells[:,1], self.extraCells[:,0]]  
            
        self.state_df.loc[index] = {self.columns[0]: rounded_sim_t, 
                                    self.columns[1]: state_sample,
                                    self.columns[2]: extra_sample}
        
        
    #########################
    ### CONFIGURATIONS
    ########################

    def setExtraCells(self, extraCells):
        """
        Secifying a constant subset of cells, which will be used for sampling the ocean state from in 
        addition to the provided drifter positions.
        """
        self.extraCells = extraCells
    
    def usePredefinedExtraCellsLovese(self):
        # Some hard coded positions that are not on land
        self.extraCells = np.array([[ 50, 110],
                            [110, 110],
                            [170, 110],
                            [290, 115],
                            [410, 115],
                            [470, 120],
                            [ 50, 155],
                            [200, 165],
                            [290, 165],
                            [350, 155],
                            [410, 155],
                            [470, 155],
                            [ 50, 200],
                            [110, 200],
                            [170, 200],
                            [230, 200],
                            [290, 200],
                            [350, 200],
                            [410, 200],
                            [470, 200]])
    
    def usePredefinedExtraCells(self):
        """
        The first ten cells represents cells that are passed by a drifter in the main truth generated on
        April 16, 2019, at 13:08:29.
        The final three are positions that are not passed by any of the selected drifters from the same 
        dataset.
        Further, cell number 3 is on the path (or very close to the path) for two drifters.
        """
        self.extraCells = np.array([[423,  25],
                                    [381,  27],
                                    [185,  48],
                                    [ 69, 157],
                                    [288, 132],
                                    [331, 177],
                                    [205, 201],
                                    [442, 234],
                                    [ 93,  11],
                                    [462,   0],
                                    [202, 135],
                                    [405, 135],
                                    [315, 229]])

    
    ############################
    ### FILE INTERFACE
    ############################        
    def to_pickle(self, path):
        """
        Write the state samples DataFrame to file (pickle)
        """
        self.state_df.to_pickle(path)
        
    def read_pickle(self, path):
        """
        Read state samples from file
        """
        self.state_df = pd.read_pickle(path)
        
    
        
        
        
    def _check_df_at_given_time(self, rounded_t):
        # Sanity check the DataFrame
        assert(self.state_df[self.state_df[self.columns[0]]==rounded_t].time.count() > 0), \
                "State sample for time " + str(rounded_t) + " does not exists in DataFrame"
        assert(self.state_df[self.state_df[self.columns[0]]==rounded_t].time.count() < 2), \
                "State sample for time " + str(rounded_t) + " has multiple entries in DataFrame"
        
        
    def get_sample_times(self):
        """
        Returns an array with the timestamps for which there exists state samples of
        underlying current.
        """
        if self.get_num_samples() < 1:
            return np.array([])
                
        return self.state_df.time.values.copy()

        
    def get_state_samples(self, t):
        """
        Returns the state sample at provided time.
        
        Returns a numpy array with numDrifters samples
        [[eta_1, hu_1, hv_1], ... , [eta_D, hu_D, hv_D]]
        """
        # The timestamp is rounded to nearest integer, so that it is possible to compare to 
        # entries in the DataFrame.
        rounded_t = round(t)
        
        # Sanity check the DataFrame
        self._check_df_at_given_time(rounded_t)

        index = self.state_df[self.state_df[self.columns[0]]==rounded_t].index.values[0]
        
        sample = self.state_df.iloc[index  ][self.columns[1]]
        return sample.copy()
        
    def get_extra_sample(self, t):
        # The timestamp is rounded to nearest integer, so that it is possible to compare to 
        # entries in the DataFrame.
        rounded_t = round(t)
        
        # Sanity check the DataFrame
        self._check_df_at_given_time(rounded_t)

        index = self.state_df[self.state_df[self.columns[0]]==rounded_t].index.values[0]
        
        extra_sample = self.state_df.iloc[index  ][self.columns[2]]
        return extra_sample.copy()
        
        