# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2022, 2023  SINTEF Digital

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
import os
import gc


from gpuocean.drifters import MLDrifterCollection
from gpuocean.utils import Observation

class MultiLevelOceanEnsemble:
    """
    Class for holding a multi-level ensemble of ocean models
    """

    def __init__(self, ML_ensemble):
        # Note ML ensemble has to be constructed outside of this class, 
        # the construction can be done in different ways
        # 
        # The assumed structure is a list with the same length as number of levels!
        # The 0-level directly contains a list of the CDKLM16 ensemble members, 
        # while the subsequent level contain TWO equally long lists with the sim-partners 
        # where the first list is the + and the second list is the - partner with a coarser resolution 
        
        assert len(ML_ensemble) > 1, "Single level ensembles are not valid"
        for l_idx in range(1, len(ML_ensemble)):
            assert len(ML_ensemble[l_idx]) == 2, "All higher levels need an ensemble of fine and coarse simulations each"
        
        self.ML_ensemble = ML_ensemble 

        self._set_ensemble_information()
    

    def _set_ensemble_information(self):

        # Set ML ensemble sizes
        self.Nes = np.zeros(len(self.ML_ensemble), dtype=np.int32)
        self.Nes[0] = len(self.ML_ensemble[0])
        for l_idx in range(1,len(self.ML_ensemble)):
            self.Nes[l_idx] = len(self.ML_ensemble[l_idx][0])

        self.numLevels = len(self.Nes)

        # Set grid information
        self.nxs = np.zeros(len(self.Nes), dtype=np.int32)
        self.nys = np.zeros(len(self.Nes), dtype=np.int32)
        self.dxs = np.zeros(len(self.Nes))
        self.dys = np.zeros(len(self.Nes))
        for l_idx in range(len(self.Nes)):
            if l_idx == 0:
                self.nxs[l_idx] = self.ML_ensemble[l_idx][0].nx
                self.nys[l_idx] = self.ML_ensemble[l_idx][0].ny
                self.dxs[l_idx] = self.ML_ensemble[l_idx][0].dx
                self.dys[l_idx] = self.ML_ensemble[l_idx][0].dy
            else:
                self.nxs[l_idx] = self.ML_ensemble[l_idx][0][0].nx
                self.nys[l_idx] = self.ML_ensemble[l_idx][0][0].ny
                self.dxs[l_idx] = self.ML_ensemble[l_idx][0][0].dx
                self.dys[l_idx] = self.ML_ensemble[l_idx][0][0].dy

        self.boundary_conditions = self.ML_ensemble[0][0].boundary_conditions

        # Set the time
        self.t = self.ML_ensemble[0][0].t

        # Variables related to drifters
        self.drifters = None
        self.drifterEnsembeSize = None
        self.drift_dt = None
        self.driftTrajectory = None
        self.drift_sensitivity = None


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

        t = self.ML_ensemble[0][0].t

        while t < obs_time:
            sim_end_time = obs_time
            if self.drifters is not None:
                dt = min(obs_time - t, self.drift_dt)
                sim_end_time = t + dt

            # evolve the ensemble
            for e in range(self.Nes[0]):
                self.ML_ensemble[0][e].dataAssimilationStep(sim_end_time)

            for l_idx in range(1, self.numLevels):
                for e in range(self.Nes[l_idx]):
                    self.ML_ensemble[l_idx][0][e].dataAssimilationStep(sim_end_time, otherSim=self.ML_ensemble[l_idx][1][e])

            # evolve drifters
            if self.drifters is not None:
                self.drift(dt)

            t = sim_end_time

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
    

    def save2file(self, filepath):
        ML_state = self.download()
        MultiLevelOceanEnsemble.saveState2file(filepath, ML_state)
        

    @staticmethod
    def saveState2file(filepath, ML_state):
        os.makedirs(filepath, exist_ok=True)
        np.save(filepath+"/MLensemble_0.npy", np.array(ML_state[0]))
        for l_idx in range(1,len(ML_state)):
            np.save(filepath+"/MLensemble_"+str(l_idx)+"_0.npy", np.array(ML_state[l_idx][0]))
            np.save(filepath+"/MLensemble_"+str(l_idx)+"_1.npy", np.array(ML_state[l_idx][1]))

    

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
        """
        Finding indices to location ON FINEST LEVEL
        obs_x  - x-location in [m]
        obs_y  - y-location in [y]

        returns:
        Hx     - x-index
        Hy     - y-index
        """
        Xs = np.linspace(self.dxs[-1]/2, (self.nxs[-1] - 1/2) * self.dxs[-1], self.nxs[-1])
        Ys = np.linspace(self.dys[-1]/2, (self.nys[-1] - 1/2) * self.dys[-1], self.nys[-1])

        Hx = ((Xs - obs_x)**2).argmin()
        Hy = ((Ys - obs_y)**2).argmin()

        return Hx, Hy
    

    def loc2idxs(self, obs_x, obs_y):
        """
        Finding indices to location ON ALL LEVELS
        obs_x  - x-location in [m]
        obs_y  - y-location in [y]

        returns:
        [[Hx0, Hy0], [[Hx1+,Hy1+],[Hx1-,Hy1-]],...]
        where the "-"-indices are the same as on the coarser level,
        but for consistency with the other indexing in code this structure is chosen
        """
        # Keep field information (cell centers)
        Xs = np.linspace(0.5*self.dxs[-1], (self.nxs[-1] - 0.5) * self.dxs[-1], self.nxs[-1])
        Ys = np.linspace(0.5*self.dys[-1], (self.nys[-1] - 0.5) * self.dys[-1], self.nys[-1])
        X, Y = np.meshgrid(Xs, Ys)

        # Keep field information per level
        lvl_X, lvl_Y = [], []
        for l_idx in range(self.numLevels):
            lvl_Xs = np.linspace(0.5*self.dxs[l_idx], (self.nxs[l_idx] - 0.5) * self.dxs[l_idx], self.nxs[l_idx])
            lvl_Ys = np.linspace(0.5*self.dys[l_idx], (self.nys[l_idx] - 0.5) * self.dys[l_idx], self.nys[l_idx])
            tmp_lvl_X, tmp_lvl_Y = np.meshgrid(lvl_Xs, lvl_Ys)
            lvl_X.append(tmp_lvl_X)
            lvl_Y.append(tmp_lvl_Y)


        # NOTE: For factor-2 scalings, this can be simplified
        obs_idxs = [list(np.unravel_index(np.argmin((lvl_X[0] - obs_x)**2 + (lvl_Y[0] - obs_y)**2), (self.nys[0], self.nxs[0])))]
        for l_idx in range(1, self.numLevels):
            obs_idxs0 = np.unravel_index(np.argmin((lvl_X[l_idx]   - obs_x)**2 + (lvl_Y[l_idx]   - obs_y)**2), (self.nys[l_idx  ], self.nxs[l_idx  ]))
            obs_idxs1 = np.unravel_index(np.argmin((lvl_X[l_idx-1] - obs_x)**2 + (lvl_Y[l_idx-1] - obs_y)**2), (self.nys[l_idx-1], self.nxs[l_idx-1]))
            obs_idxs.append([list(obs_idxs0), list(obs_idxs1)])

        return obs_idxs


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
    
    ## Drifters

    def attachDrifters(self, drifterEnsembleSize, numDrifters=None, 
                       drifterPositions=None, drift_dt=60, drift_sensitivity=1.0):

        assert(numDrifters is not None or drifterPositions is not None), "Please provide either numDrifters or drifterPositions"
        assert(numDrifters is None or drifterPositions is None), "Both numDrifters and drifterPositions are given, please provide only one of them"

        if numDrifters is None:
            numDrifters = drifterPositions.shape[0]

        self.drifters = MLDrifterCollection.MLDrifterCollection(numDrifters, drifterEnsembleSize, 
                                                                boundaryConditions=self.ML_ensemble[0][0].boundary_conditions,
                                                                domain_size_x=self.nxs[0]*self.dxs[0],
                                                                domain_size_y=self.nys[0]*self.dys[0])
        if drifterPositions is not None:
            self.drifters.setDrifterPositions(drifterPositions)

        assert(drift_dt >= self.ML_ensemble[0][0].model_time_step), " We require that the drift_dt ("+str(drift_dt)+") is larger than the mself.ML_ensemble[0][0].model_time_stepodel time step ("+str(self.ML_ensemble[0][0].model_time_step)+")"
        self.drift_dt = drift_dt
        self.drift_sensitivity = drift_sensitivity
        self.drifterEnsembeSize = drifterEnsembleSize

        self.driftTrajectory = [None]*drifterEnsembleSize
        for e in range(drifterEnsembleSize):
            self.driftTrajectory[e] = Observation.Observation()

        self.registerDrifterPositions()

    def drift(self, dt):
        assert(self.drifters is not None), "No drifters found. Can't call drift() before attachDrifters(...)"

        mean_velocity = self.estimateVelocity(np.mean)
        var_velocity  = self.estimateVelocity(np.var, ddof=1)

        self.drifters.drift(mean_velocity[0], mean_velocity[1], 
                            self.dxs[-1], self.dys[-1], 
                            dt=dt, sensitivity=self.drift_sensitivity,
                            u_var=var_velocity[0], v_var=var_velocity[1])


    def registerDrifterPositions(self):
        assert(self.drifters is not None), "There are no drifters in this ensemble"
        assert(self.driftTrajectory is not None), "Something went wrong. The ensemble has drifters but no observation objects..."

        for e in range(self.drifterEnsembeSize):
            self.driftTrajectory[e].add_observation_from_mldrifters(self.t, self.drifters, e)

    def saveDriftTrajectoriesToFile(self, path, filename_prefix):
        for e in range(self.drifterEnsembeSize):
            filename = os.path.join(path, filename_prefix + str(e).zfill(4) + ".bz2")
            self.driftTrajectory[e].to_pickle(filename)


    ## Destructors

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





from gpuocean.SWEsimulators import ModelErrorKL

class MultiLevelOceanEnsembleCase(MultiLevelOceanEnsemble):
    """
    Class for holding a multi-level ensemble of ocean models
    which is constructed from lists with simulator arguments

    i.e. ML_ensemble is constructed from the simulator arguments
    and the superclass initialized
    """

    def __init__(self, ML_Nes, args_list, make_data_args, sample_args, make_sim,
                   init_model_error_basis_args=None, sim_model_error_basis_args=None, sim_model_error_timestep=None,
                   print_status=False, init_xorwow_seed=None, init_np_seed=None, sim_xorwow_seed=None, sim_np_seed=None):


        assert len(ML_Nes) == len(args_list), "Number of levels in args and level sizes do not match"

        data_args_list = []
        if isinstance(make_data_args, list):
            assert len(ML_Nes) == len(make_data_args), "Number of levels in data_args and level sizes do not match"
            data_args_list = make_data_args
        else: 
            for l_idx in range(len(ML_Nes)):
                data_args_list.append( make_data_args(args_list[l_idx]) )

        # Model errors
        if init_model_error_basis_args is not None: 
            init_mekls = []
            for l_idx in range(len(args_list)): 
                init_mekls.append( ModelErrorKL.ModelErrorKL(**args_list[l_idx], **init_model_error_basis_args, xorwow_seed=init_xorwow_seed, np_seed=init_np_seed) )

        if sim_model_error_basis_args is not None: 
            sim_mekls = []
            for l_idx in range(len(args_list)): 
                sim_mekls.append( ModelErrorKL.ModelErrorKL(**args_list[l_idx], **sim_model_error_basis_args, xorwow_seed=sim_xorwow_seed, np_seed=sim_np_seed) )


        ## MultiLevel ensemble
        self.ML_ensemble = []

        # 0-level
        self.ML_ensemble.append([])
        for i in range(ML_Nes[0]):
            if print_status and i % 100 == 0: print(i) 
            sim = make_sim(args_list[0], sample_args, init_fields=data_args_list[0])
            if init_model_error_basis_args is not None:
                init_mekls[0].perturbSim(sim)
            if sim_model_error_basis_args is not None:
                sim.model_error = sim_mekls[0]
            sim.model_time_step = sim_model_error_timestep
            self.ML_ensemble[0].append( sim )

        # diff-levels
        for l_idx in range(1,len(ML_Nes)):
            if print_status: print(l_idx)
            self.ML_ensemble.append([[],[]])
            
            for e in range(ML_Nes[l_idx]):
                sim0 = make_sim(args_list[l_idx], sample_args, init_fields=data_args_list[l_idx])
                sim1 = make_sim(args_list[l_idx-1], sample_args, init_fields=data_args_list[l_idx-1])
                
                if init_model_error_basis_args is not None:
                    init_mekls[l_idx].perturbSim(sim0)
                    init_mekls[l_idx-1].perturbSimSimilarAs(sim1, modelError=init_mekls[l_idx])

                if sim_model_error_basis_args is not None:
                    sim0.model_error = sim_mekls[l_idx]
                    sim1.model_error = sim_mekls[l_idx-1]

                sim0.model_time_step = sim_model_error_timestep
                sim1.model_time_step = sim_model_error_timestep

                self.ML_ensemble[l_idx][0].append(sim0)
                self.ML_ensemble[l_idx][1].append(sim1)

        super()._set_ensemble_information()