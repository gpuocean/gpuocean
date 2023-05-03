# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements the Multi-Level Ensemble Kalman Filter.

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
import scipy

from skimage.measure import block_reduce

class MLEnKFOcean:
    """
    This class implements the multi-level ensemble Kalman filter
    """

    def __init__(self, MLOceanEnsemble):

        # Keep field information
        Xs = np.linspace(0, MLOceanEnsemble.nxs[-1] * MLOceanEnsemble.dxs[-1], MLOceanEnsemble.nxs[-1])
        Ys = np.linspace(0, MLOceanEnsemble.nys[-1] * MLOceanEnsemble.dys[-1], MLOceanEnsemble.nys[-1])
        self.X, self.Y = np.meshgrid(Xs, Ys)

        self.lvl_X, self.lvl_Y = [], []
        for l_idx in range(len(MLOceanEnsemble.Nes)):
            lvl_Xs = np.linspace(0, MLOceanEnsemble.nxs[l_idx] * MLOceanEnsemble.dxs[l_idx], MLOceanEnsemble.nxs[l_idx])
            lvl_Ys = np.linspace(0, MLOceanEnsemble.nys[l_idx] * MLOceanEnsemble.dys[l_idx], MLOceanEnsemble.nys[l_idx])
            lvl_X, lvl_Y = np.meshgrid(lvl_Xs, lvl_Ys)
            self.lvl_X.append(lvl_X)
            self.lvl_Y.append(lvl_Y)


    def GCweights(self, x, y, r):
        """"Gasparin Cohn weights around indices Hx, Hy with radius r"""

        dists = np.sqrt((self.X - x)**2 + (self.Y - y)**2)

        GC = np.zeros_like(dists)
        for i in range(dists.shape[0]):
            for j in range(dists.shape[1]):
                dist = dists[i,j]
                if dist/r < 1: 
                    GC[i,j] = 1 - 5/3*(dist/r)**2 + 5/8*(dist/r)**3 + 1/2*(dist/r)**4 - 1/4*(dist/r)**5
                elif dist/r >= 1 and dist/r < 2:
                    GC[i,j] = 4 - 5*(dist/r) + 5/3*(dist/r)**2 + 5/8*(dist/r)**3 -1/2*(dist/r)**4 + 1/12*(dist/r)**5 - 2/(3*(dist/r))

        return GC

        
    def assimilate(self, MLOceanEnsemble, obs, Hx, Hy, R, r = 2.5*1e7, relax_factor = 1.0, obs_var=slice(0,3), min_localisation_level=1):
        """
        Returning the posterior state after assimilating observation into multi-level ensemble
        after appyling MLEnKF

        MLOceanEnsemble - MultiLevelOceanEnsemble in prior state
        obs             - ndarray of size (3,) with true values (eta, hu, hv) 
        Hx, Hy          - int's with observation indices (on finest level)
        R               - ndarray of size (3,) with observation noises per coordinate
        r               - float > 0, localisation radius
        relax_factor    - float in range [0, 1.0], relaxation factor for the weighting in the localisation
        obs_var         - slice of maximal length 3, observed variables. Examples:
                            obs_var = slice(0,1) # eta
                            obs_var = slice(1,3) # hu and hv
                            obs_var = slice(0,3) # eta, hu, hv
        min_localisation_level  - int, this and all higher levels are localised in the update
        """

        ## Prior        
        ML_state = MLOceanEnsemble.download()


        ## Easy access to frequently used information
        Nes = MLOceanEnsemble.Nes
        numLevels = MLOceanEnsemble.numLevels

        # Observation indices per level: 
        obs_idxs = [list(np.unravel_index(np.argmin((self.lvl_X[0] - self.X[0,Hx])**2 + (self.lvl_Y[0] - self.Y[Hy,0])**2), ML_state[0][0].shape[:-1]))]
        for l_idx in range(1, len(Nes)):
            obs_idxs0 = np.unravel_index(np.argmin((self.lvl_X[l_idx]   - self.X[0,Hx])**2 + (self.lvl_Y[l_idx]   - self.Y[Hy,0])**2), ML_state[l_idx][0][0].shape[:-1])
            obs_idxs1 = np.unravel_index(np.argmin((self.lvl_X[l_idx-1] - self.X[0,Hx])**2 + (self.lvl_Y[l_idx-1] - self.Y[Hy,0])**2), ML_state[l_idx][1][0].shape[:-1])
            obs_idxs.append([list(obs_idxs0), list(obs_idxs1)])

        # Number of observed variables
        if obs_var.step is None:
            obs_varN = (obs_var.stop - obs_var.start) 
        else: 
            obs_varN = (obs_var.stop - obs_var.start)/obs_var.step

        ## Localisation kernel
        obs_x = self.X[0,Hx]
        obs_y = self.Y[Hy,0]

        GC = self.GCweights(obs_x, obs_y, r)


        ## Perturbations
        ML_perts = []
        for l_idx in range(numLevels):
            ML_perts.append(np.random.multivariate_normal(np.zeros(3)[obs_var], np.diag(R[obs_var]), size=Nes[l_idx]))


        ## Analysis
        ML_XY = np.zeros((np.prod(ML_state[-1][0].shape[:-1]),obs_varN))

        X0 = ML_state[0]
        X0mean = np.average(X0, axis=-1)

        Y0 = ML_state[0][obs_var,obs_idxs[0][0],obs_idxs[0][1]] + ML_perts[0].T
        Y0mean = np.average(Y0, axis=-1)

        lvl_weight = relax_factor * np.ones(np.prod(X0mean.shape))
        if min_localisation_level <= 0:
            lvl_weight = relax_factor * np.tile(block_reduce(GC, block_size=(2**(numLevels-1),2**(numLevels-1)), func=np.mean).flatten(),3)

        ML_XY += (lvl_weight[:,np.newaxis] 
                  * 1/Nes[0] 
                  *( (X0-X0mean[:,:,:,np.newaxis]).reshape(-1,X0.shape[-1]) 
                    @ (Y0 - Y0mean[:,np.newaxis]).T)
                ).reshape(X0mean.shape + (obs_varN,)).repeat(2**(numLevels-1),1).repeat(2**(numLevels-1),2).reshape(-1,ML_XY.shape[-1])

        for l_idx in range(1,numLevels):

            X0 = ML_state[l_idx][0]
            X0mean = np.average(X0, axis=-1)
            X1 = ML_state[l_idx][1].repeat(2,1).repeat(2,2)
            X1mean = np.average(X1, axis=-1)

            Y0 = ML_state[l_idx][0][obs_var,obs_idxs[l_idx][0][0],obs_idxs[l_idx][0][1]] + ML_perts[l_idx].T
            Y0mean = np.average(Y0, axis=-1)
            Y1 = ML_state[l_idx][1][obs_var,obs_idxs[l_idx][1][0],obs_idxs[l_idx][1][1]] + ML_perts[l_idx].T
            Y1mean = np.average(Y1, axis=-1)

            lvl_weight = relax_factor * np.ones(np.prod(X0mean.shape))
            if min_localisation_level <= l_idx:
                lvl_weight = relax_factor * np.tile(block_reduce(GC, block_size=(2**(numLevels-l_idx-1),2**(numLevels-l_idx-1)), func=np.mean).flatten(),3)

            ML_XY += (lvl_weight[:,np.newaxis] 
                      * ( 1/Nes[l_idx]
                            *( (X0-X0mean[:,:,:,np.newaxis]).reshape(-1,X0.shape[-1]) 
                              @ (Y0 - Y0mean[:,np.newaxis]).T) 
                         - 1/Nes[l_idx]
                            *( (X1-X1mean[:,:,:,np.newaxis]).reshape(-1,X1.shape[-1]) 
                              @ (Y1 - Y1mean[:,np.newaxis]).T) 
                        )
                    ).reshape(X0mean.shape + (obs_varN,)).repeat(2**(numLevels-l_idx-1),1).repeat(2**(numLevels-l_idx-1),2).reshape(-1,ML_XY.shape[-1])

        ML_HXY = ML_XY.reshape(ML_state[-1][0].shape[:-1] + (obs_varN,))[obs_var,obs_idxs[-1][0][0],obs_idxs[-1][0][1],:]
        ML_YY  = ML_HXY + np.diag(R[obs_var])

        ML_K = ML_XY @ np.linalg.inv(ML_YY)


        ## Update
        ML_state[0] = ML_state[0] + (
                            block_reduce( ML_K.reshape(ML_state[-1][0].shape[:-1]+ (obs_varN,)),  block_size=(1,2**(numLevels-1), 2**(numLevels-1), 1), func=np.mean).reshape((np.prod(ML_state[0].shape[:-1]),obs_varN)) 
                            @ (obs[obs_var,np.newaxis] - ML_state[0][obs_var,obs_idxs[0][0],obs_idxs[0][1]] - ML_perts[0].T)
                        ).reshape(ML_state[0].shape)

        for l_idx in range(1,numLevels):
            ML_state[l_idx][0] = ML_state[l_idx][0] + (
                                    block_reduce(ML_K.reshape(ML_state[-1][0].shape[:-1]+ (obs_varN,)), block_size=(1,2**(numLevels-l_idx-1), 2**(numLevels-l_idx-1), 1), func=np.mean).reshape((np.prod(ML_state[l_idx][0].shape[:-1]),obs_varN)) 
                                    @ (obs[obs_var,np.newaxis] - ML_state[l_idx][0][obs_var,obs_idxs[l_idx][0][0],obs_idxs[l_idx][0][1]] - ML_perts[l_idx].T)
                                ).reshape(ML_state[l_idx][0].shape)
            ML_state[l_idx][1] = ML_state[l_idx][1] + (
                                    block_reduce(ML_K.reshape(ML_state[-1][0].shape[:-1]+ (obs_varN,)), block_size=(1,2**(numLevels-l_idx  ), 2**(numLevels-l_idx  ), 1), func=np.mean).reshape((np.prod(ML_state[l_idx][1].shape[:-1]),obs_varN)) 
                                    @ (obs[obs_var,np.newaxis] - ML_state[l_idx][1][obs_var,obs_idxs[l_idx][1][0],obs_idxs[l_idx][1][1]] - ML_perts[l_idx].T)
                                ).reshape(ML_state[l_idx][1].shape)

        MLOceanEnsemble.upload(ML_state)

        return ML_K
