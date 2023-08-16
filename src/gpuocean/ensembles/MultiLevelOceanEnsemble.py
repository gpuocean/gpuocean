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
import gc


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

        self.t = self.ML_ensemble[0][0].t

        

    def step(self, t, **kwargs):
        """ evolving the entire ML ensemble by time t """
        for e in range(self.Nes[0]):
            self.ML_ensemble[0][e].step(t, **kwargs)

        for l_idx in range(1, self.numLevels):
            for e in range(self.Nes[l_idx]):
                self.ML_ensemble[l_idx][0][e].step(t, **kwargs)
                self.ML_ensemble[l_idx][1][e].step(t, **kwargs)

        self.t = self.ML_ensemble[0][0].t

    def stepToObservation(self, obs_time):
        """
        Evolves the ensemble forward to the given obs_time
        (see also CDKLM16.dataAssimilationStep)

        obs_time - float, end time for simulations
        """

        for e in range(self.Nes[0]):
            self.ML_ensemble[0][e].dataAssimilationStep(obs_time)

        for l_idx in range(1, self.numLevels):
            for e in range(self.Nes[l_idx]):
                self.ML_ensemble[l_idx][0][e].dataAssimilationStep(obs_time, otherSim=self.ML_ensemble[l_idx][1][e])
        
        self.t = self.ML_ensemble[0][0].t

    
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

    def downloadVelocities(self, interior_domain_only=True):
        """"State of the ML ensemble as list of np-arrays per level
        
        Return: list (length=number of levels), 
        per level the size is  (2, ny, nx, Ne)
        where the leftmost index is for u and v
        """
        ML_state = []

        lvl_state = []

        _, Hm_lvl0 = self.ML_ensemble[0][0].downloadBathymetry(interior_domain_only=interior_domain_only)
        for e in range(self.Nes[0]):
            eta, hu, hv = self.ML_ensemble[0][e].download(interior_domain_only=interior_domain_only)
            u = hu/(eta + Hm_lvl0)
            v = hv/(eta + Hm_lvl0)
            lvl_state.append(np.array([u, v]))
        ML_state.append(np.array(lvl_state))
        ML_state[0] = np.moveaxis(ML_state[0], 0, -1)

        for l_idx in range(1, self.numLevels):
            lvl_state0 = []
            lvl_state1 = []
            _, Hm_lvl_state0 = self.ML_ensemble[l_idx][0][0].downloadBathymetry(interior_domain_only=interior_domain_only)
            _, Hm_lvl_state1 = self.ML_ensemble[l_idx][1][0].downloadBathymetry(interior_domain_only=interior_domain_only)
            for e in range(self.Nes[l_idx]):
                eta0, hu0, hv0 = self.ML_ensemble[l_idx][0][e].download(interior_domain_only=interior_domain_only)
                eta1, hu1, hv1 = self.ML_ensemble[l_idx][1][e].download(interior_domain_only=interior_domain_only)
                lvl_state0.append(np.array([hu0/(eta0 + Hm_lvl_state0),
                                            hv0/(eta0 + Hm_lvl_state0)]))
                lvl_state1.append(np.array([hu1/(eta1 + Hm_lvl_state1),
                                            hv1/(eta1 + Hm_lvl_state1)]))
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
        """
        General ML-estimator for some statistic given as func, performed on the standard state [eta, hu, hv]
        func - function that calculates a single-level statistics, e.g. np.mean or np.var
        """
        ML_state = self.download()
        return self._estimate_pure(ML_state, func, **kwargs)

    def estimateVelocity(self, func,  **kwargs):
        """
        General ML-estimator for some statistic given as func, performed on the computed velocities [u, v]
        func - function that calculates a single-level statistics, e.g. np.mean or np.var
        The estimate is computed *with* ghost cells (as this will be used mainly for drifters)
        """
        ML_state = self.downloadVelocities()
        return self._estimate_pure(ML_state, func, **kwargs)

 
    def _estimate_pure(self, ML_state, func, **kwargs):
        """
        General ML-estimator for some statistics with any arbitrary state variables
        """
        
        MLest = np.zeros(ML_state[-1][0].shape[:-1])
        MLest += func(ML_state[0], axis=-1, **kwargs).repeat(2**(self.numLevels-1),1).repeat(2**(self.numLevels-1),2)
        for l_idx in range(1, self.numLevels):
            MLest += (func(ML_state[l_idx][0], axis=-1, **kwargs) - func(ML_state[l_idx][1], axis=-1, **kwargs).repeat(2,1).repeat(2,2)).repeat(2**(self.numLevels-l_idx-1),1).repeat(2**(self.numLevels-l_idx-1),2)

        return MLest
    
 
    

    def obsLoc2obsIdx(self, obs_x, obs_y):
        """"""
        Xs = np.linspace(self.dxs[-1]/2, (self.nxs[-1] - 1/2) * self.dxs[-1], self.nxs[-1])
        Ys = np.linspace(self.dys[-1]/2, (self.nys[-1] - 1/2) * self.dys[-1], self.nys[-1])

        Hx = ((Xs - obs_x)**2).argmin()
        Hy = ((Ys - obs_y)**2).argmin()

        return Hx, Hy

    def rank(self, truth, obs_locations, R=None):
        """
        Returning the rank of the truth within the ML-ensemble
        truth - simulator on the finest level
        obs_locations - list of indices for observation, e.g. [[100, 100]] or [[100, 100], [200,200]]
        """

        assert truth.nx == self.nxs[-1], "Truth doesnt match finest level"
        assert truth.ny == self.nys[-1], "Truth doesnt match finest level"
        assert truth.dx == self.dxs[-1], "Truth doesnt match finest level"
        assert truth.dy == self.dys[-1], "Truth doesnt match finest level"

        ML_state = self.download()
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)

        ML_Fys = []
        for [Hx, Hy] in obs_locations:

            # Extracting true values
            true_values = np.array([true_eta[Hy, Hx], true_hu[Hy, Hx], true_hv[Hy, Hx]]) 
            if R is not None:
                true_values += np.random.multivariate_normal(np.zeros(3), np.diag(R))

            # observation indices on right level
            Xs = np.linspace(0, self.nxs[-1] * self.dxs[-1], self.nxs[-1])
            Ys = np.linspace(0, self.nys[-1] * self.dys[-1], self.nys[-1])
            X, Y = np.meshgrid(Xs, Ys)

            lvl_Xs = np.linspace(0, self.nxs[0] * self.dxs[0], self.nxs[0])
            lvl_Ys = np.linspace(0, self.nys[0] * self.dys[0], self.nys[0])
            lvl_X, lvl_Y = np.meshgrid(lvl_Xs, lvl_Ys)

            obs_idxs = np.unravel_index(np.argmin((lvl_X - X[0,Hx])**2 + (lvl_Y - Y[Hy,0])**2), ML_state[0][0].shape[:-1])

            ensemble_values = ML_state[0][:,obs_idxs[0],obs_idxs[1],:]
            if R is not None: 
                ensemble_values += np.random.multivariate_normal(np.zeros(3), np.diag(R), size=self.Nes[0]).T

            ML_Fy = 1/self.Nes[0] * np.sum(ensemble_values < true_values[:,np.newaxis], axis=1)

            for l_idx in range(1,len(self.Nes)):
                lvl_Xs0 = np.linspace(0, self.nxs[l_idx] * self.dxs[l_idx], self.nxs[l_idx])
                lvl_Ys0 = np.linspace(0, self.nys[l_idx] * self.dys[l_idx], self.nys[l_idx])
                lvl_X0, lvl_Y0 = np.meshgrid(lvl_Xs0, lvl_Ys0)
                obs_idxs0 = np.unravel_index(np.argmin((lvl_X0 - X[0,Hx])**2 + (lvl_Y0 - Y[Hy,0])**2), ML_state[l_idx][0][0].shape[:-1])

                lvl_Xs1 = np.linspace(0, self.nxs[l_idx-1] * self.dxs[l_idx-1], self.nxs[l_idx-1])
                lvl_Ys1 = np.linspace(0, self.nys[l_idx-1] * self.dys[l_idx-1], self.nys[l_idx-1])
                lvl_X1, lvl_Y1 = np.meshgrid(lvl_Xs1, lvl_Ys1)
                obs_idxs1 = np.unravel_index(np.argmin((lvl_X1 - X[0,Hx])**2 + (lvl_Y1 - Y[Hy,0])**2), ML_state[l_idx][1][0].shape[:-1])

                ensemble_values0 = ML_state[l_idx][0][:,obs_idxs0[0],obs_idxs0[1],:]
                ensemble_values1 = ML_state[l_idx][1][:,obs_idxs1[0],obs_idxs1[1],:]
                if R is not None:
                    lvl_perts = np.random.multivariate_normal(np.zeros(3), np.diag(R), size=self.Nes[l_idx]).T
                    ensemble_values0 += lvl_perts
                    ensemble_values1 += lvl_perts

                ML_Fy += 1/self.Nes[l_idx] * np.sum(1 * (ensemble_values0 < true_values[:,np.newaxis]) 
                                            - 1 * (ensemble_values1 < true_values[:,np.newaxis]), axis=1)
                
            ML_Fys.append(ML_Fy)
        
        return ML_Fys


    def MSE(self, truth, obs_locations=None):
        """
        Returning MSE of ensemble vs true observation at set of locations. 
        The MSE is herein defined as E[(X-y)^2] and the ML-estimator used for the approiximation.

        truth - simulator on the finest level
        obs_locations - list of indices for observation, e.g. [[100, 100]] or [[100, 100], [200,200]]
        """

        assert truth.nx == self.nxs[-1], "Truth doesnt match finest level"
        assert truth.ny == self.nys[-1], "Truth doesnt match finest level"
        assert truth.dx == self.dxs[-1], "Truth doesnt match finest level"
        assert truth.dy == self.dys[-1], "Truth doesnt match finest level"

        ML_state = self.download()
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)

        MSEs = []
        for [Hx, Hy] in obs_locations:

            # Extracting true values
            true_values = np.array([true_eta[Hy, Hx], true_hu[Hy, Hx], true_hv[Hy, Hx]]) 

            # observation indices on right level
            Xs = np.linspace(0, self.nxs[-1] * self.dxs[-1], self.nxs[-1])
            Ys = np.linspace(0, self.nys[-1] * self.dys[-1], self.nys[-1])
            X, Y = np.meshgrid(Xs, Ys)

            lvl_Xs = np.linspace(0, self.nxs[0] * self.dxs[0], self.nxs[0])
            lvl_Ys = np.linspace(0, self.nys[0] * self.dys[0], self.nys[0])
            lvl_X, lvl_Y = np.meshgrid(lvl_Xs, lvl_Ys)

            obs_idxs = np.unravel_index(np.argmin((lvl_X - X[0,Hx])**2 + (lvl_Y - Y[Hy,0])**2), ML_state[0][0].shape[:-1])

            MSE = np.average((ML_state[0][:,obs_idxs[0],obs_idxs[1],:] -true_values[:,np.newaxis])**2, axis=-1)

            for l_idx in range(1,len(self.Nes)):
                lvl_Xs0 = np.linspace(0, self.nxs[l_idx] * self.dxs[l_idx], self.nxs[l_idx])
                lvl_Ys0 = np.linspace(0, self.nys[l_idx] * self.dys[l_idx], self.nys[l_idx])
                lvl_X0, lvl_Y0 = np.meshgrid(lvl_Xs0, lvl_Ys0)
                obs_idxs0 = np.unravel_index(np.argmin((lvl_X0 - X[0,Hx])**2 + (lvl_Y0 - Y[Hy,0])**2), ML_state[l_idx][0][0].shape[:-1])

                lvl_Xs1 = np.linspace(0, self.nxs[l_idx-1] * self.dxs[l_idx-1], self.nxs[l_idx-1])
                lvl_Ys1 = np.linspace(0, self.nys[l_idx-1] * self.dys[l_idx-1], self.nys[l_idx-1])
                lvl_X1, lvl_Y1 = np.meshgrid(lvl_Xs1, lvl_Ys1)
                obs_idxs1 = np.unravel_index(np.argmin((lvl_X1 - X[0,Hx])**2 + (lvl_Y1 - Y[Hy,0])**2), ML_state[l_idx][1][0].shape[:-1])

                MSE += np.average((ML_state[l_idx][0][:,obs_idxs0[0],obs_idxs0[1],:] - true_values[:,np.newaxis])**2, axis=-1) \
                        - np.average((ML_state[l_idx][1][:,obs_idxs1[0],obs_idxs1[1],:] - true_values[:,np.newaxis])**2, axis=-1)
                
            MSEs.append(MSE)
        
        return MSEs
    

    def cleanUp(self):
        for e in range(self.Nes[0]):
            self.ML_ensemble[0][e].cleanUp(do_gc=False)
        
        for l_idx in range(1,self.numLevels):
            for e in range(self.Nes[l_idx]):
                self.ML_ensemble[l_idx][0][e].cleanUp(do_gc=False)
                self.ML_ensemble[l_idx][1][e].cleanUp(do_gc=False)
        gc.collect()

    def __del__(self):
        self.cleanUp()