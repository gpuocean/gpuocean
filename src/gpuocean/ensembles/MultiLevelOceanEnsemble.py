# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements a Multi-level ensemble.

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

class MultiLevelOceanEnsemble:
    """
    Class for holding a multi-level ensemble of ocean models
    """

    def __init__(self, ML_ensemble):
        # For the time being, the ML ensemble has to be constructed outside of this class, 
        # since construction can work in different ways
        # 
        # The assumed structure is a list with the same length as number of levels!
        # The 0-level directly contains a list of the CDKLM16 ensemble members, 
        # While the subsequent level contain TWO equally long lists which the sim-partners 
        # Where the first list is the + and the second list is the - partner with a coarser resolution 
        self.ML_ensemble = ML_ensemble 

        # Keep ML ensemble sizes
        self.Nes = np.zeros(len(ML_ensemble), dtype=np.int32)
        self.Nes[0] = len(ML_ensemble[0])
        for l_idx in range(1,len(ML_ensemble)):
            self.Nes[l_idx] = len(ML_ensemble[l_idx][0])

        self.numLevels = len(self.Nes)

        # Keep grid information
        self.nxs = np.zeros(len(self.Nes), dtype=np.int32)
        self.nys = np.zeros(len(self.Nes), dtype=np.int32)
        self.dxs = np.zeros(len(self.Nes))
        self.dys = np.zeros(len(self.Nes))
        for l_idx in range(len(self.Nes)):
            if l_idx == 0:
                self.nxs[l_idx] = ML_ensemble[l_idx][0].nx
                self.nys[l_idx] = ML_ensemble[l_idx][0].ny
                self.dxs[l_idx] = ML_ensemble[l_idx][0].dx
                self.dys[l_idx] = ML_ensemble[l_idx][0].dy
            else:
                self.nxs[l_idx] = ML_ensemble[l_idx][0][0].nx
                self.nys[l_idx] = ML_ensemble[l_idx][0][0].ny
                self.dxs[l_idx] = ML_ensemble[l_idx][0][0].dx
                self.dys[l_idx] = ML_ensemble[l_idx][0][0].dy


    def step(self, t, **kwargs):
        """ evolving the entire ML ensemble by time t """
        for e in range(self.Nes[0]):
            self.ML_ensemble[0][e].step(t, **kwargs)

        for l_idx in range(1, self.numLevels):
            for e in range(self.Nes[l_idx]):
                self.ML_ensemble[l_idx][0][e].step(t, **kwargs)
                self.ML_ensemble[l_idx][1][e].step(t, **kwargs)

    
    def download(self, interior_domain_only=True):
        """"State of the ML ensemble as list of np-arrays per level
        
        Return: list (length=number of levels), 
        per level the size is  (3, ny, nx, Ne)
        """
        ML_state = []

        lvl_state = []
        for e in range(self.Nes[0]):
            eta, hu, hv = self.ML_ensemble[0][e].download(interior_domain_only=interior_domain_only)
            lvl_state.append(np.array([eta, hu, hv]))
        ML_state.append(np.array(lvl_state))
        ML_state[0] = np.moveaxis(ML_state[0], 0, -1)

        for l_idx in range(1, self.numLevels):
            lvl_state0 = []
            lvl_state1 = []
            for e in range(self.Nes[l_idx]):
                eta0, hu0, hv0 = self.ML_ensemble[l_idx][0][e].download(interior_domain_only=interior_domain_only)
                eta1, hu1, hv1 = self.ML_ensemble[l_idx][1][e].download(interior_domain_only=interior_domain_only)
                lvl_state0.append(np.array([eta0, hu0, hv0]))
                lvl_state1.append(np.array([eta1, hu1, hv1]))
            ML_state.append([np.array(lvl_state0), np.array(lvl_state1)])
            ML_state[l_idx][0] = np.moveaxis(ML_state[l_idx][0], 0, -1)
            ML_state[l_idx][1] = np.moveaxis(ML_state[l_idx][1], 0, -1) 

        return ML_state


    def upload(self, ML_state):
        """
        Uploading interior-cell data
        """
        for e in range(self.Nes[0]):
            self.ML_ensemble[0][e].upload(*np.pad(ML_state[0][:,:,:,e],((0,0),(2,2),(2,2))))
            
        for l_idx in range(1,self.numLevels):
            for e in range(self.Nes[l_idx]):
                self.ML_ensemble[l_idx][0][e].upload(*np.pad(ML_state[l_idx][0][:,:,:,e],((0,0),(2,2),(2,2))))
                self.ML_ensemble[l_idx][1][e].upload(*np.pad(ML_state[l_idx][1][:,:,:,e],((0,0),(2,2),(2,2))))


    def estimate(self, func, **kwargs):
        ML_state = self.download()

        MLest = np.zeros(ML_state[-1][0].shape[:-1])
        MLest += func(ML_state[0], axis=-1, **kwargs).repeat(2**(self.numLevels-1),1).repeat(2**(self.numLevels-1),2)
        for l_idx in range(1, self.numLevels):
            MLest += (func(ML_state[l_idx][0], axis=-1, **kwargs) - func(ML_state[l_idx][1], axis=-1, **kwargs).repeat(2,1).repeat(2,2)).repeat(2**(self.numLevels-l_idx-1),1).repeat(2**(self.numLevels-l_idx-1),2)

        return MLest

