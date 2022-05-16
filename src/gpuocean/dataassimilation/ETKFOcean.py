# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements the Ensemble Transform Kalman Filter.

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
import time
import logging
from SWESimulators import DataAssimilationUtils as dautils

#from SWESimulators import Common, OceanStateNoise, config, EnsemblePlot

class ETKFOcean:
    """
    This class implements the Stochastic Ensemble Kalman Filter in square-root formulation
    for an ocean model with small scale ocean state perturbations as model errors.
    
    Input to constructor:
    ensemble: An object of super-type BaseOceanStateEnsemble.
            
    """

    def __init__(self, ensemble, inflation_factor=1.0, scale_w = 1.0):
        """
        Copying the ensemble to the member variables 
        and deducing frequently used ensemble quantities
        """

        self.ensemble = ensemble
        
        self.N_e = ensemble.getNumParticles()
        self.N_d = ensemble.getNumDrifters()

        # Size of state matrices (with ghost cells)
        self.n_i = self.ensemble.particles[0].ny + 2*self.ensemble.particles[-1].ghost_cells_y
        self.n_j = self.ensemble.particles[0].nx + 2*self.ensemble.particles[-1].ghost_cells_x

        self.ghost_cells_y = self.ensemble.particles[-1].ghost_cells_y
        self.ghost_cells_x = self.ensemble.particles[-1].ghost_cells_x      

        # Parameter for inflation
        self.inflation_factor = inflation_factor
        self.scale_w = scale_w

        # Parameters and variables for localisation
        self.r_factor = 15.0

        self.W_loc = None
        self.all_Ls = None

        self.groups = None
    

    def ETKF(self, ensemble=None):
        """
        Performing the analysis phase of the ETKF.
        Particles are observed and the analysis state is calculated and uploaded!

        ensemble: for better readability of the script when ETKF is called the ensemble can be passed again.
        Then it overwrites the initially defined member ensemble
        """

        if ensemble is not None:
            assert(self.N_e == ensemble.getNumParticles()), "ensemble changed size"
            assert(self.N_d == ensemble.getNumDrifters()), "ensemble changed number of drifter"
            assert(self.n_i == ensemble.particles[0].ny + 2*ensemble.particles[-1].ghost_cells_y), "ensemble changed size of physical domain"
            assert(self.n_j == ensemble.particles[0].nx + 2*ensemble.particles[-1].ghost_cells_x), "ensemble changed size of physical domain"
            
            self.ensemble = ensemble

            self.N_e_active = ensemble.getNumActiveParticles()

        X_f, X_f_mean, X_f_pert = self._getX_f()
        HX_f_pert, HX_f_mean = self._getHX_f()

        Rinv = self._constructRinv()

        D = self._getD(HX_f_mean)

        P = self._getP(HX_f_pert, Rinv)
        
        K = self._getK(X_f_pert, P, HX_f_pert, Rinv)

        X_a = self._getX_a(X_f_mean, X_f_pert, K, D, P)

        self.uploadAnalysisState(X_a)

    
    """
    The following methods stating with _ are simple matrix computations and reshaping operations
    which are separated for the seek of readability
    """

    def _deleteDeactivatedObservations(self, observation):
        """
        Delete inactive particles
        """
        idx = 0
        for p in range(self.N_e):
            if self.ensemble.particlesActive[p]:
                idx+=1
            elif not self.ensemble.particlesActive[p]:
                observation = np.delete(observation, idx, axis=0)
        return observation


    def _constructRinv(self):
        
        R_orig = self.ensemble.getObservationCov()

        R = np.zeros( (R_orig.shape[0]*self.N_d, R_orig.shape[1]*self.N_d) )

        for l in range(self.N_d):
            R[l,l] = R_orig[0,0]
            R[self.N_d+l, self.N_d+l] = R_orig[1,1]
            R[l,self.N_d+l] = R_orig[0,1]
            R[self.N_d+l,l] = R_orig[1,0]

        Rinv = np.linalg.inv(R)

        return Rinv

    def _getX_f(self):

        """
        The download gives eta = 
        [
        [eta(x0,y0),...,eta(xN,y0)],
        ...,
        [eta(x0,yN),...,eta(xN,yN)]
        ]
        as an array of size Ny x Nx
        and analog for hu and hv.
        we use those as an 1D array eta = 
        [eta(x0,y0),...,eta(xN,y0),eta(x0,y1),...,eta(xN,y(N-1)),eta(x0,yN),...,eta(xN,yN)]
        and anlog for hu and hv 

        For further calculations the indivdual dimensions of the state variable are concatinated X = 
        [eta, hu, hv]

        Collecting the state perturbation for each ensemble member in a matrix Nx x Ne, where
        X_f_pert = 
        [ 
        [eta_pert(x0,y0) (particle 1),..., eta_pert],
        ...
        particle 2: [eta_pert,hu_pert,hv_pert]
        ]
        """

        X_f = np.zeros((3*self.n_i*self.n_j, self.N_e_active))

        idx = 0
        for e in range(self.N_e):
            if self.ensemble.particlesActive[e]:
                eta, hu, hv = self.ensemble.particles[e].download(interior_domain_only=False)
                eta = eta.reshape(self.n_i*self.n_j)
                hu  = hu.reshape(self.n_i*self.n_j)
                hv  = hv.reshape(self.n_i*self.n_j)
                X_f[:,idx] = np.append(eta, np.append(hu,hv))
                idx += 1

        X_f_mean = np.zeros( 3*self.n_i*self.n_j )
        for e in range(self.N_e_active):
            X_f_mean += 1/self.N_e_active * X_f[:,e]

        X_f_pert = np.zeros_like( X_f )
        for e in range(self.N_e_active):
            X_f_pert[:,e] = X_f[:,e] - X_f_mean

        return X_f, X_f_mean, X_f_pert


    def _getHX_f(self):
        """
        Particles are observed in the following form:
        [
        particle 1:  [hu_1, hv_1], ... , [hu_D, hv_D],
        ...
        particle Ne: [hu_1, hv_1], ... , [hu_D, hv_D]
        ]

        In order to bring it in accordance with later data structure we use the following format for the storage of the perturbation of the observation:
        [
        [hu_1 (particle 1), ..., hu_1 (particle Ne)],
        ...
        [hu_D (particle 1), ..., hu_D (particle Ne)],
        [hv_1 (particle 1), ..., hv_1 (particle Ne)],
        ...
        [hv_D (particle 1), ..., hv_D (particle Ne)],
        ]

        """

        # Observation (nan for inactive particles)
        HX_f_orig = self._deleteDeactivatedObservations(self.ensemble.observeParticles())

        # Reshaping
        HX_f = np.zeros( (2*self.N_d, self.N_e_active) )
        for e in range(self.N_e_active):
            for l in range(self.N_d):
                HX_f[l,e]     = HX_f_orig[e,l,0]
            for l in range(self.N_d):
                HX_f[self.N_d+l,e] = HX_f_orig[e,l,1]

        HX_f_mean = 1/self.N_e_active * np.sum(HX_f, axis=1)

        HX_f_pert = HX_f - HX_f_mean.reshape((2*self.N_d,1))

        return HX_f_pert, HX_f_mean


    def _getD(self, HX_f_mean):
        """
        Particles yield innovations in the following form:
        [x_1, y_1, hu_1, hv_1], ... , [x_D, y_D, hu_D, hv_D]

        In order to bring it in accordance with later data structure we use the following format for the storage of the perturbation of the observation:
        [hu_1, ..., hu_D, hv_1, ..., hv_D]
        """

        y_orig = self.ensemble.observeTrueState()

        y = np.zeros( (2*self.N_d) )
        for l in range(self.N_d):
            y[l]     = y_orig[l,2]
        for l in range(self.N_d):
            y[self.N_d+l] = y_orig[l,3]

        D = y - HX_f_mean

        return D


    def _getP(self, HX_f_pert, Rinv):

        A1 = (self.N_e_active-1) * np.eye(self.N_e_active)
        A2 = np.dot(HX_f_pert.T, np.dot(Rinv, HX_f_pert))

        A = A1 + A2

        P = np.linalg.inv(A)

        return P

    def _getK(self, X_f_pert, P, HX_f_pert, Rinv):

        K = np.dot(X_f_pert, np.dot(P, np.dot(HX_f_pert.T, Rinv)))

        return K


    def _getX_a(self, X_f_mean, X_f_pert, K, D, P):

        X_a_mean = X_f_mean + np.dot(K, D)

        sigma, V = np.linalg.eigh( (self.N_e_active-1) * P )
        X_a_pert = np.dot( X_f_pert, np.dot( V, np.dot( np.diag( np.sqrt( np.real(sigma) ) ), V.T )))

        X_a = X_a_pert 
        for j in range(self.N_e_active):
            X_a[:,j] += X_a_mean
            
        return X_a


    def uploadAnalysisState(self, X_a):
        
        idx = 0
        for e in range(self.N_e):
            if self.ensemble.particlesActive[e]:
                eta = X_a[0:self.n_i*self.n_j, idx].reshape((self.n_i,self.n_j))
                hu  = X_a[self.n_i*self.n_j:2*self.n_i*self.n_j, idx].reshape((self.n_i,self.n_j))
                hv  = X_a[2*self.n_i*self.n_j:3*self.n_i*self.n_j, idx].reshape((self.n_i,self.n_j))
                self.ensemble.particles[e].upload(eta,hu,hv)
                idx += 1


    """
    Functionalities for the LETKF
    """

    @staticmethod
    def getLocalIndices(obs_loc, scale_r, dx, dy, nx, ny):
        """ 
        Defines mapping from global domain (nx times ny) to local domain
        """

        boxed_r = dx*np.ceil(scale_r*1.5)
        
        localIndices = np.array([[False]*nx]*ny)
        
        obs_loc_cellID = (np.int(obs_loc[0]//dx), np.int(obs_loc[1]//dy))

        #print(obs_loc_cellID)
        loc_cell_left  = int(np.round(obs_loc[0]/dx)) - int(np.round(boxed_r/dx))
        loc_cell_right = int(np.round(obs_loc[0]/dx)) + int(np.round((boxed_r+dx)/dx))
        loc_cell_down  = int(np.round(obs_loc[1]/dy)) - int(np.round(boxed_r/dy))
        loc_cell_up    = int(np.round(obs_loc[1]/dy)) + int(np.round((boxed_r+dy)/dy))

        xranges = []
        yranges = []
        
        xroll = 0
        yroll = 0

        if loc_cell_left < 0:
            xranges.append((nx+loc_cell_left , nx))
            xroll = loc_cell_left   # negative number
            loc_cell_left = 0 
        elif loc_cell_right > nx:
            xranges.append((0, loc_cell_right - nx))
            xroll = loc_cell_right - nx   # positive number
            loc_cell_right = nx 
        xranges.append((loc_cell_left, loc_cell_right))

        if loc_cell_down < 0:
            yranges.append((ny+loc_cell_down , ny))
            yroll = loc_cell_down   # negative number
            loc_cell_down = 0 
        elif loc_cell_up > ny:
            yranges.append((0, loc_cell_up - ny ))
            yroll = loc_cell_up - ny   # positive number
            loc_cell_up = ny
        yranges.append((loc_cell_down, loc_cell_up))

        for xrange in xranges:
            for yrange in yranges:
                localIndices[yrange[0] : yrange[1], xrange[0] : xrange[1]] = True

                for y in range(yrange[0],yrange[1]):
                    for x in range(xrange[0], xrange[1]):
                        loc = np.array([(x+0.5)*dx, (y+0.5)*dy])

        return localIndices, xroll, yroll


    @staticmethod
    def distGC(obs, loc, r, lx, ly):
        """
        Calculating the Gasparin-Cohn value for the distance between obs 
        and loc for the localisation radius r.
        
        obs: drifter positions ([x,y])
        loc: current physical location to check (either [x,y] or [[x1,y1],...,[xd,yd]])
        r: localisation scale in the Gasparin Cohn function
        lx: domain extension in x-direction (necessary for periodic boundary conditions)
        ly: domain extension in y-direction (necessary for periodic boundary conditions)
        """
        if not obs.shape == loc.shape: 
            obs = np.tile(obs, (loc.shape[0],1))
        
        if len(loc.shape) == 1:
            dist = min(np.linalg.norm(np.abs(obs-loc)),
                    np.linalg.norm(np.abs(obs-loc) - np.array([lx,0 ])),
                    np.linalg.norm(np.abs(obs-loc) - np.array([0 ,ly])),
                    np.linalg.norm(np.abs(obs-loc) - np.array([lx,ly])) )
        else:
            dist = np.linalg.norm(obs-loc, axis=1)

        # scalar case
        if isinstance(dist, float):
            distGC = 0.0
            if dist/r < 1: 
                distGC = 1 - 5/3*(dist/r)**2 + 5/8*(dist/r)**3 + 1/2*(dist/r)**4 - 1/4*(dist/r)**5
            elif dist/r >= 1 and dist/r < 2:
                distGC = 4 - 5*(dist/r) + 5/3*(dist/r)**2 + 5/8*(dist/r)**3 -1/2*(dist/r)**4 + 1/12*(dist/r)**5 - 2/(3*(dist/r))
        # vector case
        else:
            distGC = np.zeros_like(dist)
            for i in range(len(dist)):
                if dist[i]/r < 1: 
                    distGC[i] = 1 - 5/3*(dist[i]/r)**2 + 5/8*(dist[i]/r)**3 + 1/2*(dist[i]/r)**4 - 1/4*(dist[i]/r)**5
                elif dist[i]/r >= 1 and dist[i]/r < 2:
                    distGC[i] = 4 - 5*(dist[i]/r) + 5/3*(dist[i]/r)**2 + 5/8*(dist[i]/r)**3 -1/2*(dist[i]/r)**4 + 1/12*(dist[i]/r)**5 - 2/(3*(dist[i]/r))

        return distGC


    @staticmethod
    def getLocalWeightShape(scale_r, dx, dy, nx, ny, scale_w=1.0):
        """
        Gives a local stencil with weights based on the distGC
        """
        
        local_nx = int(np.ceil(scale_r*1.5)*2 + 1)
        local_ny = int(np.ceil(scale_r*1.5)*2 + 1)
        weights = np.zeros((local_ny, local_ny))
        
        obs_loc_cellID = (local_ny, local_nx)
        obs_loc = np.array([local_nx*dx/2, local_ny*dy/2])

        for y in range(local_ny):
            for x in range(local_nx):
                loc = np.array([(x+0.5)*dx, (y+0.5)*dy])
                if np.linalg.norm(obs_loc - loc) > 1.5*scale_r*dx:
                    weights[y,x] = 0
                else:
                    weights[y,x] = min(1, ETKFOcean.distGC(obs_loc, loc, scale_r*dx, nx*dx, ny*dy))
                                
        return scale_w * weights
            

    @staticmethod
    def getCombinedWeights(observation_positions, scale_r, dx, dy, nx, ny, W_loc, scale_w=1.0):
        
        W_scale = np.zeros((ny, nx))
        
        num_drifters = observation_positions.shape[0]
        #print('found num_drifters:', num_drifters)
        if observation_positions.shape[1] != 2:
            print('observation_positions has wrong shape')
            return None

        # Get the shape of the local weights (drifter independent)
        W_loc = ETKFOcean.getLocalWeightShape(scale_r, dx, dy, nx, ny, scale_w)
        
        for d in range(num_drifters):
            # Get local mapping for drifter 
            L, xroll, yroll = ETKFOcean.getLocalIndices(observation_positions[d,:], scale_r, dx, dy, nx, ny)

            # Roll weigths according to periodic boundaries
            W_loc_d = np.roll(np.roll(W_loc, shift=yroll, axis=0 ), shift=xroll, axis=1)
            
            # Add weights to global domain based on local mapping:
            W_scale[L] += W_loc_d.flatten()

            
        return W_scale


    def initializeLocalPatches(self, r_factor=0.0):
        """
        Preprocessing for the LETKF 
        which generates arrays storing the local observation indices for every grid cell (including 2 ghost cells)
        
        r_factor: scale for the Gasparin-Cohn distance and the definition of local boxes
        x0: x-coordinate of physical position of the lower left corner in meter
        y0: y-coordinate of physical position of the lower left corner in meter

        FILL CONTENT
        """

        # Book keeping
        dy = self.ensemble.dy
        dx = self.ensemble.dx

        nx = self.ensemble.nx
        ny = self.ensemble.ny

        if r_factor > 0.0:
            self.r_factor = r_factor

        # Construct local stencil
        self.W_loc = ETKFOcean.getLocalWeightShape(self.r_factor, dx, dy, nx, ny, self.scale_w)

        self.W_analyses = []
        self.W_forecasts = []

        for g in range(len(self.groups)):

            # Construct global analysis and forecast weights
            W_combined = ETKFOcean.getCombinedWeights(self.drifter_positions[self.groups[g]], self.r_factor, dx, dy, nx, ny, self.W_loc, self.scale_w)

            W_scale = np.maximum(W_combined, 1)

            W_analysis = W_combined/W_scale
            W_forecast = np.ones_like(W_scale) - W_analysis

            self.W_analyses.append(W_analysis)
            self.W_forecasts.append(W_forecast)



    def initializeGroups(self, r_factor):
        # Get drifter position
        if self.ensemble.observation_type == dautils.ObservationType.StaticBuoys:
            self.drifter_positions = self.ensemble.observeTrueDrifters()
        else:
            self.drifter_positions = self.ensemble.observeTrueState()[:,0:2]

        xdim = self.ensemble.getDx() * self.ensemble.getNx()
        ydim = self.ensemble.getDy() * self.ensemble.getNy() 

        N_y = self.drifter_positions.shape[0]

        # Assembling observation distance matrix
        obs_dist_mat = np.zeros((N_y, N_y))
        for i in range(N_y):
            for j in range(N_y):
                dx = np.abs(self.drifter_positions[i][0] - self.drifter_positions[j][0])
                if dx > xdim/2:
                    dx = xdim - dx
                dy = np.abs(self.drifter_positions[i][1] - self.drifter_positions[j][1])
                if dy > ydim/2:
                    dy = ydim - dy 
                obs_dist_mat[i,j] = np.sqrt(dx**2+dy**2)
        # Heavy diagonal such that 0-distances are above every threshold
        np.fill_diagonal(obs_dist_mat, np.sqrt(xdim**2 + ydim**2))

        # Groups of "un-correlated" observation
        self.groups = list([list(np.arange(N_y, dtype=int))])
        # Observations are assumed to be uncorrelated, if distance bigger than threshold
        threshold = 2.0 * 1.5 * r_factor * self.ensemble.particles[0].dx

        g = 0 
        while obs_dist_mat[np.ix_(self.groups[g],self.groups[g])].min() < threshold:
            while obs_dist_mat[np.ix_(self.groups[g],self.groups[g])].min() < threshold:
                mask = np.ix_(self.groups[g],self.groups[g])
                idx2move = self.groups[g][np.where(obs_dist_mat[mask] == obs_dist_mat[mask].min())[1][0]]
                self.groups[g] = list(np.delete(self.groups[g], np.where(self.groups[g] == idx2move)))
                if len(self.groups)<g+2: 
                    self.groups.append([idx2move])
                else:
                    self.groups[g+1].append(idx2move)
            g = g + 1 

    def _prepare_LETKF(self, ensemble=None, r_factor=0.0):
        """
        Internal preprocessing for computing the LETKF analysis.
        This function will update, if neccessary, the class member variables for
         - ensemble
         - local weight kernels
         - localization groups
        """
        # Check and update parameters of ensemble
        if ensemble is not None:
            assert(self.N_e == ensemble.getNumParticles()), "ensemble changed size"
            assert(self.N_d == ensemble.getNumDrifters()), "ensemble changed number of drifter"
            assert(self.n_i == ensemble.particles[0].ny + 2*ensemble.particles[-1].ghost_cells_y), "ensemble changed size of physical domain"
            assert(self.n_j == ensemble.particles[0].nx + 2*ensemble.particles[-1].ghost_cells_x), "ensemble changed size of physical domain"
            
            self.ensemble = ensemble

            self.N_e_active = ensemble.getNumActiveParticles()


        # Update localisation if needed
        if r_factor > 0.0 and r_factor != self.localScale:
            self.r_factor = r_factor
            self.W_loc = None
            self.groups = None

        if self.groups is None:
            self.initializeGroups( r_factor=self.r_factor )
            self.initializeLocalPatches( r_factor=self.r_factor )


        # Precalculate rolling (for StaticBuoys this just have to be once)
        if self.ensemble.observation_type == dautils.ObservationType.StaticBuoys and self.all_Ls is None:
            self.all_Ls = [None]*self.N_d
            self.all_xrolls = np.zeros(self.N_d, dtype=np.int)
            self.all_yrolls = np.zeros(self.N_d, dtype=np.int)

            for d in range(self.N_d):
                # Collecting rolling information (xroll and yroll are 0)
                self.all_Ls[d], self.all_xrolls[d], self.all_yrolls[d] = \
                    ETKFOcean.getLocalIndices(self.drifter_positions[d,:], self.r_factor, \
                        self.ensemble.dx, self.ensemble.dy, self.ensemble.nx, self.ensemble.ny)




    def LETKF(self, ensemble=None, r_factor=0.0):
        """
        Performing the analysis phase of the ETKF.
        Particles are observed and the analysis state is calculated and uploaded!

        ensemble: for better readability of the script when ETKF is called the ensemble can be passed again.
        Then it overwrites the initially defined member ensemble
        """

        self._prepare_LETKF(ensemble=ensemble, r_factor=r_factor)

        # Prepare local ETKF analysis
        X_f      = np.zeros((self.N_e_active,3,self.ensemble.ny,self.ensemble.nx), dtype=np.float32)
        X_f_mean = np.zeros((                3,self.ensemble.ny,self.ensemble.nx), dtype=np.float32)
        X_f_pert = np.zeros((self.N_e_active,3,self.ensemble.ny,self.ensemble.nx), dtype=np.float32)
        
        N_x_local = self.W_loc.shape[0]*self.W_loc.shape[1] 
        X_f_loc_tmp      = np.zeros((self.N_e_active, 3, N_x_local))
        X_f_loc_pert_tmp = np.zeros((self.N_e_active, 3, N_x_local))
        X_f_loc_mean_tmp = np.zeros((3, N_x_local))
            
        X_f_loc      = np.zeros((3*N_x_local, self.N_e_active))
        X_f_loc_pert = np.zeros((3*N_x_local, self.N_e_active))

        X_a_loc_pert = None

        observations = self.ensemble.observeTrueState()
        
        
        for g in range(len(self.groups)):
          
            # Reset global variables
            self.giveX_f_global(X_f, X_f_mean, X_f_pert, download_X_f=(g==0))
            HX_f_mean, HX_f_pert = self.giveHX_f_global(X_f, observations)
            X_a = np.zeros_like(X_f)
    
            
            # Loop over all d
            for d in self.groups[g]:
        
                L, xroll, yroll = self.all_Ls[d], self.all_xrolls[d], self.all_yrolls[d]

                # LOCAL ARRAY FOR FORECAST (basically extracting local values from global array)
                X_f_loc_tmp[:,:,:] = X_f[:,:,L]           # shape: (N_e_active, 3, N_x_local)
                X_f_loc_pert_tmp[:,:,:] = X_f_pert[:,:,L] # shape: (N_e_active, 3, N_x_local)
                X_f_loc_mean_tmp[:,:] = X_f_mean[:,L]   # shape: (3, N_x_local))

                
                # Roll local array (this should not change anything here!)
                if not (xroll == 0 and yroll == 0):
                    rolling_shape = (self.N_e_active, 3, self.W_loc.shape[0], self.W_loc.shape[1]) # roll around axis 2 and 3
                    X_f_loc_tmp[:,:,:] = np.roll(np.roll(X_f_loc_tmp.reshape(rolling_shape), shift=-yroll, axis=2 ), shift=-xroll, axis=3).reshape((self.N_e_active, 3, N_x_local))
                    X_f_loc_pert_tmp[:,:,:] = np.roll(np.roll(X_f_loc_pert_tmp.reshape(rolling_shape), shift=-yroll, axis=2 ), shift=-xroll, axis=3).reshape((self.N_e_active, 3, N_x_local))

                    mean_rolling_shape = (3, self.W_loc.shape[0], self.W_loc.shape[1]) # roll around axis 1 and 2
                    X_f_loc_mean_tmp[:,:] = np.roll(np.roll(X_f_loc_mean_tmp.reshape(mean_rolling_shape), shift=-yroll, axis=1 ), shift=-xroll, axis=2).reshape((3, N_x_local))
                
                
                # FROM LOCAL ARRAY TO LOCAL VECTOR FOR FORECAST (we concatinate eta, hu and hv components)
                X_f_loc_mean = np.append(X_f_loc_mean_tmp[0,:],np.append(X_f_loc_mean_tmp[1,:],X_f_loc_mean_tmp[2,:]))
                X_f_loc[:,:] = X_f_loc_tmp.reshape((self.N_e_active, 3*N_x_local)).T
                X_f_loc_pert[:,:] = X_f_loc_pert_tmp.reshape((self.N_e_active, 3*N_x_local)).T
                
                    
                # Local observations
                HX_f_loc_mean = HX_f_mean[d,:]
                HX_f_loc_pert = HX_f_pert[:,d,:].T

                ############LETKF

                # Rinv 
                Rinv = scipy.linalg.inv(self.ensemble.getObservationCov())

                # D
                #y_loc = self.ensemble.observeTrueState()[d,2:4].T
                y_loc = observations[d, 2:4].T
                D = y_loc - HX_f_loc_mean

                # Inflation
                if self.inflation_factor == 0.0:
                    # Adaptive inflation following Sætrom and Omre (2013)
                    # where the factor is calculated and applied locally
                    inflation_factor = np.sqrt(1 + np.trace(Rinv @ np.outer(D,D))/(self.N_e_active-2))
                    forgetting_factor = 1/(inflation_factor**2)
                    #print("Ensemble inflation: ", inflation_factor)
                else:
                    forgetting_factor = 1/(self.inflation_factor**2)
                    #print("Ensemble inflation: ", self.inflation_factor)

                # P 
                A1 = (self.N_e_active-1) * np.eye(self.N_e_active)
                A2 = HX_f_loc_pert[:,ensemble.particlesActive].T @ Rinv @ HX_f_loc_pert[:,ensemble.particlesActive]
                A = A1 + A2


                # --- START of the SVD/inv block
                # Use the solve function instead of P = inv(A)
                K = X_f_loc_pert @ np.linalg.solve(A, HX_f_loc_pert[:,ensemble.particlesActive].T @ Rinv)

                # local analysis
                X_a_loc_mean = X_f_loc_mean + K @ D

                sigma_inv, V = scipy.linalg.eigh( (1./(self.N_e_active-1)) * A )

                X_a_loc_pert = X_f_loc_pert @ V @ np.diag( np.sqrt( 1/np.real(sigma_inv)) ) @ V.T

                # --- END of the SVD/inv block

                X_a_loc = X_a_loc_pert 
                for j in range(self.N_e_active):
                    X_a_loc[:,j] += X_a_loc_mean

                # Inflation for hu and hv (but eta unchanged)
                if forgetting_factor != 1.0:
                    A1_inflated = (self.N_e_active-1) * forgetting_factor * np.eye(self.N_e_active)
                    A_inflated = A1_inflated + A2

                    K_inflated = X_f_loc_pert @ np.linalg.solve(A_inflated, HX_f_loc_pert[:,ensemble.particlesActive].T @ Rinv)

                    X_a_loc_mean_inflated = X_f_loc_mean + K_inflated @ D

                    sigma_inv_inflated, V_inflated =  scipy.linalg.eigh( (1./(self.N_e_active-1)) * A_inflated )
                    X_a_loc_pert_inflated = X_f_loc_pert @ V_inflated @ np.diag( np.sqrt( 1/np.real(sigma_inv_inflated) ) ) @ V_inflated.T

                    X_a_loc[N_x_local:,:] = X_a_loc_pert_inflated[N_x_local:,:]
                    for j in range(self.N_e_active):
                        X_a_loc[N_x_local:,j] += X_a_loc_mean_inflated[N_x_local:]


                # FROM LOCAL VECTOR TO GLOBAL ARRAY (we fill the global X_a with the *weighted* local values)
                # eta, hu, hv
                for i in range(3):
                    # Calculate weighted local analysis
                    weighted_X_a_loc = X_a_loc[i*N_x_local:(i+1)*N_x_local,:]*(np.tile(self.W_loc.flatten().T, (self.N_e_active, 1)).T)
                    # Here, we use np.tile(W_loc.flatten().T, (N_e_active, 1)).T to repeat W_loc as column vector N_e_active times 
                    
                    if not (xroll == 0 and yroll == 0):
                        weighted_X_a_loc = np.roll(np.roll(weighted_X_a_loc[:,:].reshape((self.W_loc.shape[0], self.W_loc.shape[1], self.N_e_active)), 
                                                                                        shift=yroll, axis=0 ), 
                                                        shift=xroll, axis=1)
                    
                    X_a[:,i,L] += weighted_X_a_loc.reshape(self.W_loc.shape[0]*self.W_loc.shape[1], self.N_e_active).T
            
            # (end loop over all observations)
        
            # COMBINING (the already weighted) ANALYSIS WITH THE FORECAST
            X_new = np.zeros_like(X_f)
            for e in range(self.N_e_active):
                for i in range(3):
                    X_new[e][i] = self.W_forecasts[g]*X_f[e][i] + X_a[e][i]

            X_f = X_new 
        # (end loop over all groups)
        
        self.uploadAnalysis(X_f)



    def giveX_f_global(self, X_f, X_f_mean, X_f_pert, download_X_f=True):
        """
        Download recent particle states if needed, and compute X_f_mean and X_f_pert.
        """
        if download_X_f:
            idx = 0
            for e in range(self.N_e):
                if self.ensemble.particlesActive[e]:
                    X_f[idx,0,:,:], X_f[idx,1,:,:], X_f[idx,2,:,:] = self.ensemble.particles[e].download(interior_domain_only=True)
                    idx += 1

        X_f_mean[:,:,:] = np.mean(X_f, axis=0)

        X_f_pert[:,:,:,:] = X_f - X_f_mean

        #return X_f, X_f_mean, X_f_pert


    def giveHX_f_global(self, X_f, observations):
        """
        Observe particles 
        """

        HX_f = self.observe_particles_from_X_f(X_f, observations)
        
        HX_f_mean = 1/self.N_e_active * np.nansum(HX_f, axis=0)

        HX_f_pert = HX_f - HX_f_mean

        return HX_f_mean, HX_f_pert


    def observe_particles_from_X_f(self, X_f, observations):
        assert(self.ensemble.observation_type == dautils.ObservationType.StaticBuoys), 'Only implemented for StaticBuoys'
        assert(self.N_d == observations.shape[0]), 'mismatch between observations and N_d' 
        
        active_particles = X_f.shape[0]
        observedParticles = np.empty((active_particles, self.N_d, 2))
        
        for d in range(self.N_d):
            id_x = np.int(np.floor(observations[d,0]/self.ensemble.dx))
            id_y = np.int(np.floor(observations[d,1]/self.ensemble.dy))

                
            observedParticles[:, d, 0] = X_f[:, 1, id_y, id_x]
            observedParticles[:, d, 1] = X_f[:, 2, id_y, id_x]

        return observedParticles
        

    def uploadAnalysis(self, X_new):
        # Upload analysis
        idx = 0
        for e in range(self.N_e):
            if self.ensemble.particlesActive[e]:
                # construct eta
                eta = np.zeros((self.ensemble.ny+2*self.ghost_cells_y, self.ensemble.nx+2*self.ghost_cells_x))
                eta[self.ghost_cells_y : self.ensemble.ny+self.ghost_cells_y, self.ghost_cells_x : self.ensemble.nx+self.ghost_cells_x] \
                    = X_new[idx][0]

                # construct hu
                hu  = np.zeros((self.ensemble.ny+2*self.ghost_cells_y, self.ensemble.nx+2*self.ghost_cells_x))
                hu[self.ghost_cells_y : self.ensemble.ny+self.ghost_cells_y, self.ghost_cells_x : self.ensemble.nx+self.ghost_cells_x] \
                    = X_new[idx][1]

                # construct hv
                hv  = np.zeros((self.ensemble.ny+2*self.ghost_cells_y, self.ensemble.nx+2*self.ghost_cells_x))
                hv[self.ghost_cells_y:self.ensemble.ny + self.ghost_cells_y, self.ghost_cells_x:self.ensemble.nx+self.ghost_cells_x] \
                    = X_new[idx][2]

                self.ensemble.particles[e].upload(eta,hu,hv)
                self.ensemble.particles[e].applyBoundaryConditions()
                idx += 1