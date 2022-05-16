# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018-2019 SINTEF Digital

This python class implements an ensemble of particles, each consisting
of a single drifter in its own ocean state. Each ocean model is 
perturbed during each timestep, using small scale perturbations.

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

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import warnings


import pycuda.driver as cuda

from SWESimulators import CDKLM16
from SWESimulators import GPUDrifterCollection
from SWESimulators import WindStress
from SWESimulators import Common
from SWESimulators import DataAssimilationUtils as dautils
from SWESimulators import BaseOceanStateEnsemble


class OceanNoiseEnsemble(BaseOceanStateEnsemble.BaseOceanStateEnsemble):
    
        
    ####################################################################
    ### CONSTRUCTOR and helper functions
    ####################################################################
    def __init__(self, gpu_ctx, numParticles, sim, 
                 num_drifters = 1,
                 observation_type=dautils.ObservationType.DrifterPosition,
                 buoys_positions = None,
                 observation_variance = None, 
                 observation_variance_factor = 5.0,
                 initialization_variance_factor_drifter_position = 0.0,
                 initialization_variance_factor_ocean_field = 0.0,
                 compensate_for_eta = True,
                 ensemble_small_scale_perturbation=True):
        """
        Class that holds an ensemble of ocean states. All ensemble members are initiated 
        as perturbations of a given input simulator, and the true state is also a perturbation
        of the input simulator and is run along side the ensemble

        gpu_ctx: GPU context
        numParticles: Number of particles, also known as number of ensemble members
        sim: A simulator which represent the initial state of all particles
        num_drifters = 1: Number of drifters that provide us observations
        observation_type: ObservationType enumerator object
        buoys_positions: For bouys-type observation the positions should be served 
            in the format [[x1,y1],...,[xD,yD]] where xi and yi are given in meter
        observation_variance: Can be a scalar or a covariance matrix
        observation_variance_factor: If observation_variance is not provided, the 
            observation_variance will be (observation_variance_factor*dx)**2
        initialization_variance_factor_drifter_position: Gives an initial perturbation of 
            drifter positions if non-zero
        initialization_variance_factor_ocean_field: Gives an initial perturbation of 
            the ocean field if non-zero
        compensate_for_eta: Whether or not the observations should be adjusted by the eta from the particle states.
        """
        self.gpu_ctx = gpu_ctx
        self.gpu_stream = cuda.Stream()
        
        self.numParticles = numParticles
        self.particles = [None]*(self.numParticles + 1)
        self.particlesActive = [True]*(self.numParticles)
        
        self.obs_index = self.numParticles
        
        self.simType = 'CDKLM16'
        self.ensemble_small_scale_perturbation = ensemble_small_scale_perturbation
        
        self.t = 0.0
        
        dautils.ObservationType._assert_valid(observation_type)
        self.observation_type = observation_type
        self.prev_observation = None
        self.compensate_for_eta = compensate_for_eta

        if self.observation_type == dautils.ObservationType.StaticBuoys:
            if buoys_positions is not None:
                assert( len(buoys_positions)==num_drifters ), "number of given buoys positions does not match the specified num_drifters" 
        self.buoys_positions = buoys_positions

        self.observation_buffer = None
                
        # Observations are stored as [ [t^n, [[x_i^n, y_i^n]] ] ]
        # where n is time step and i is drifter
        self.observedDrifterPositions = []
        
        # Arrays to store statistical info for selected grid cells:
        self.varianceUnderDrifter_eta = []
        self.varianceUnderDrifter_hu = []
        self.varianceUnderDrifter_hv = []
        self.rmseUnderDrifter_eta = []
        self.rmseUnderDrifter_hu = []
        self.rmseUnderDrifter_hv = []
        self.rmseMeanUnderDrifter_eta = []
        self.rmseMeanUnderDrifter_hu = []
        self.rmseMeanUnderDrifter_hv = []
        self.rmseWeightedUnderDrifter_eta = []
        self.rmseWeightedUnderDrifter_hu = []
        self.rmseWeightedUnderDrifter_hv = []
        self.rUnderDrifter_eta = []
        self.rUnderDrifter_hu = []
        self.rUnderDrifter_hv = []
        self.tArray = []
        
        # Call helper functions
        self._setGridInfoFromSim(sim)
        self._setStochasticVariables(observation_variance = observation_variance, 
                                     observation_variance_factor = observation_variance_factor,
                                     initialization_variance_factor_drifter_position = initialization_variance_factor_drifter_position,
                                     initialization_variance_factor_ocean_field = initialization_variance_factor_ocean_field)        
        self._init(driftersPerOceanModel=num_drifters, 
                buoys_positions=self.buoys_positions)
        
        # Create gpu kernels and buffers:
        self._setupGPU()
        
                
        # Put the initial positions into the observation array
        # (For static buoys the cell ids of the positions are extracted and stored,
        # the self.buoys_postions are either given as input or sampled in _init)
        if self.observation_type == dautils.ObservationType.StaticBuoys:
            self.buoys_ids = np.empty((len(self.buoys_positions),2)).astype(int)
            for d in range(len(self.buoys_positions)):
                self.buoys_ids[d][0] = np.int(np.floor(self.buoys_positions[d][0]/self.dx))
                self.buoys_ids[d][1] = np.int(np.floor(self.buoys_positions[d][1]/self.dy))

        self._addObservation(self.observeTrueDrifters()) 
        
        
        # Store mean water_depth, and whether the equilibrium depth is constant across the domain
        H = self.particles[0].downloadBathymetry()[1][2:-2, 2:-2] # H in cell centers
        self.mean_depth = np.mean(H)
        self.constant_depth = np.max(H) == np.min(H)
        
    def _setGridInfo(self, nx, ny, dx, dy, dt, 
                    boundaryConditions=Common.BoundaryConditions(), 
                    eta=None, hu=None, hv=None, H=None):
        """
        Declaring grid-related member variables
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        
        self.boundaryConditions = boundaryConditions
        
        assert(self.simType == 'CDKLM16'), 'CDKLM16 is currently the only supported scheme'
        self.ghostCells = np.array([2,2,2,2])
        if self.boundaryConditions.isSponge():
            sponge = self.boundaryConditions.getSponge()
            for i in range(4):
                if sponge[i] > 0: 
                    self.ghostCells[i] = sponge[i]
        dataShape =  ( ny + self.ghostCells[0] + self.ghostCells[2], 
                       nx + self.ghostCells[1] + self.ghostCells[3]  )
            
        self.base_eta = eta
        self.base_hu = hu
        self.base_hv = hv
        self.base_H = H
            
        # Create base initial data: 
        if self.base_eta is None:
            self.base_eta = np.zeros(dataShape, dtype=np.float32, order='C')
        if self.base_hu is None:
            self.base_hu  = np.zeros(dataShape, dtype=np.float32, order='C');
        if self.base_hv is None:
            self.base_hv  = np.zeros(dataShape, dtype=np.float32, order='C');
        
        # Bathymetry:
        if self.base_H is None:
            waterDepth = 10
            self.base_H = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')*waterDepth

        
    def _setGridInfoFromSim(self, sim):
        """
        Declaring physical member variables
        """
        eta, hu, hv = sim.download()
        Hi = sim.downloadBathymetry()[0]
        self._setGridInfo(sim.nx, sim.ny, sim.dx, sim.dy, sim.dt,
                         sim.boundary_conditions,
                         eta=eta, hu=hu, hv=hv, H=Hi)
        self.g = sim.g
        self.f = sim.f
        self.beta = sim.coriolis_beta
        self.r = sim.r
        self.wind = sim.wind_stress
        
        self.small_scale_perturbation = sim.small_scale_perturbation
        self.small_scale_perturbation_amplitude = None
        self.small_scale_perturbation_interpolation_factor = 1.0
        
        if self.small_scale_perturbation:
            self.small_scale_perturbation_amplitude = sim.small_scale_model_error.soar_q0
            self.small_scale_perturbation_interpolation_factor = sim.small_scale_perturbation_interpolation_factor
            
        
        
    def _setStochasticVariables(self, 
                               observation_variance = None, 
                               observation_variance_factor = 5.0,
                               initialization_variance_factor_drifter_position = 0.0,
                               initialization_variance_factor_ocean_field = 0.0):
        """
        Declaring variables related to stochastic model error terms and uncertainty.
        """

        # Setting observation variance:
        self.observation_variance = observation_variance
        if self.observation_variance is None:
            self.observation_variance = (observation_variance_factor*self.dx)**2
        
        # Build observation covariance matrix:
        self.observation_cov = None
        self.observation_cov_inverse = None
        if np.isscalar(self.observation_variance):
            self.observation_cov = np.eye(2)*self.observation_variance
            self.observation_cov_inverse = np.eye(2)*(1.0/self.observation_variance)
        else:
            print("type(self.observation_variance): " + str(type(self.observation_variance)))
            # Assume that we have a correctly shaped matrix here
            self.observation_cov = self.observation_variance
            self.observation_cov_inverse = np.linalg.inv(self.observation_cov)
        
        
        # TODO: Check if this variable is used anywhere.
        # Should not be defined in the Base class.
        self.initialization_variance_drifters = initialization_variance_factor_drifter_position*self.dx
        self.initialization_cov_drifters = np.eye(2)*self.initialization_variance_drifters
        self.midPoint = 0.5*np.array([self.nx*self.dx, self.ny*self.dy])
        
        # When initializing an ensemble, each member should be perturbed so that they 
        # have slightly different starting point.
        # This factor should be multiplied to the small_scale_perturbation_amplitude for that 
        # perturbation
        self.initialization_variance_factor_ocean_field = initialization_variance_factor_ocean_field
        
    def _init(self, driftersPerOceanModel=1, buoys_positions=None):
        """
        Initiating the ensemble by perturbing the input simulator and attaching drifters
        """
        self.driftersPerOceanModel = np.int32(driftersPerOceanModel)
        
        
        for i in range(self.numParticles+1):
            self.particles[i] = CDKLM16.CDKLM16(self.gpu_ctx, \
                                                self.base_eta, self.base_hu, self.base_hv, \
                                                self.base_H, \
                                                self.nx, self.ny, self.dx, self.dy, self.dt, \
                                                self.g, self.f, self.r, \
                                                boundary_conditions=self.boundaryConditions, \
                                                write_netcdf=False, \
                                                small_scale_perturbation=self.ensemble_small_scale_perturbation, \
                                                small_scale_perturbation_amplitude=self.small_scale_perturbation_amplitude,
                                                small_scale_perturbation_interpolation_factor=self.small_scale_perturbation_interpolation_factor)
            
            if self.initialization_variance_factor_ocean_field != 0.0:
                self.particles[i].perturbState(q0_scale=self.initialization_variance_factor_ocean_field)
                
            # Add drifters
            drifters = GPUDrifterCollection.GPUDrifterCollection(self.gpu_ctx, self.driftersPerOceanModel,
                                                 observation_variance=self.observation_variance,
                                                 boundaryConditions=self.boundaryConditions,
                                                 initialization_cov_drifters=self.initialization_cov_drifters,
                                                 domain_size_x=self.nx*self.dx, domain_size_y=self.ny*self.dy)
            
            # for static buoys set positions as specified by input or get the sampled position
            if self.observation_type == dautils.ObservationType.StaticBuoys:
                if buoys_positions is not None:
                    drifters.setDrifterPositions(buoys_positions)
                elif buoys_positions is None:
                    self.buoys_positions = drifters.getDrifterPositions()

            self.particles[i].attachDrifters(drifters)
          
        # Initialize and attach drifters to all particles.
        #self._initialize_drifters(driftersPerOceanModel)
        

    def _setupGPU(self):
        """
        Setting up kernel for reading observations, along with host and device buffers
        """
        # Create observation buffer!    
        if self.observation_type == dautils.ObservationType.UnderlyingFlow or \
            self.observation_type == dautils.ObservationType.DirectUnderlyingFlow:
            
            zeros = np.zeros((self.driftersPerOceanModel, 2), dtype=np.float32, order='C')
            self.observation_buffer = Common.CUDAArray2D(self.gpu_stream, \
                                                         2, self.driftersPerOceanModel, 0, 0, \
                                                         zeros)

            # Generate kernels
            self.observation_kernels = self.gpu_ctx.get_kernel("observationKernels.cu", \
                                                             defines={})


            # Get CUDA functions and define data types for prepared_{async_}call()
            self.observeUnderlyingFlowKernel = self.observation_kernels.get_function("observeUnderlyingFlow")
            self.observeUnderlyingFlowKernel.prepare("iiffiiPiPiPifiPiPi")

            self.local_size = (int(self.driftersPerOceanModel), 1, 1)
            self.global_size = (1, 1)
        
        
    ####################################################################
    ### CONSTRUCTOR and helper functions - finished
    ####################################################################
        
    ### Function called by destructor:
    def cleanUp(self):
        """
        Releasing device memory.
        This function is called by the destructor.
        """
        for oceanState in self.particles:
            if oceanState is not None:
                oceanState.cleanUp()
        if self.observation_buffer is not None:
            self.observation_buffer.release()
        self.gpu_ctx = None

        

    def resample(self, newSampleIndices, reinitialization_variance):
        """
        Resampling the particles given by the newSampleIndicies input array.
        Here, the reinitialization_variance input is ignored, meaning that exact
        copies only are resampled.
        """
        obsTrueDrifter = self.observeTrueDrifters()
        positions = self.observeParticles()
        newPos = np.empty((self.driftersPerOceanModel, 2))
        newOceanStates = [None]*self.getNumParticles()
        for i in range(self.getNumParticles()):
            index = newSampleIndices[i]
            #print "(particle no, position, old direction, new direction): "
            if self.observation_type == dautils.ObservationType.UnderlyingFlow or \
               self.observation_type == dautils.ObservationType.DirectUnderlyingFlow:
                newPos[:,:] = obsTrueDrifter
            else:
                # Copy the drifter position from the particle that is resampled
                newPos[:,:] = positions[index,:]
            
            #print "\t", (index, positions[index,:], newPos)

            
            # Download index's ocean state:
            eta0, hu0, hv0 = self.particles[index].download()
            eta1, hu1, hv1 = self.particles[index].downloadPrevTimestep()
            newOceanStates[i] = (eta0, hu0, hv0, eta1, hu1, hv1)

            self.particles[i].drifters.setDrifterPositions(newPos)

        # New loop for transferring the correct ocean states back up to the GPU:
        for i in range(self.getNumParticles()):
            self.particles[i].upload(newOceanStates[i][0],
                                     newOceanStates[i][1],
                                     newOceanStates[i][2],
                                     newOceanStates[i][3],
                                     newOceanStates[i][4],
                                     newOceanStates[i][5])
                    
   


        

        
    def _addObservation(self, observedDrifterPositions):
        """
        Adds the given observed drifter positions to the observedDrifterPosition 
        array. Observations are there stored as [ [t^n, [[x_i^n, y_i^n]] ] ]
        where n is time step and i is drifter
        """
        self.observedDrifterPositions.append([self.t, observedDrifterPositions])

        
    def observeDrifters(self):
        """
        Observing the drifters in all particles
        """
        drifterPositions = np.empty((self.getNumParticles(), self.driftersPerOceanModel, 2))
        for p in range(self.getNumParticles()):
            drifterPositions[p,:,:] = self.particles[p].drifters.getDrifterPositions()
        return drifterPositions
    

        
    def observeTrueDrifters(self):
        """
        Observing the drifters in the syntetic true state.
        """
        return self.particles[self.obs_index].drifters.getDrifterPositions()
        
        
    def observeTrueState(self, apply_observation_error=True):
        """
        Applying the observation operator on the syntetic true state.
        The observation should be in state space, and therefore consists of 
        hu and hv, and not u and v.

        Returns a numpy array with D drifter positions and drifter velocities
        [[x_1, y_1, hu_1, hv_1], ... , [x_D, y_D, hu_D, hv_D]]
        If the observation operator is drifter positions, hu and hv are not included.
        """
        if self.observedDrifterPositions[-1][0] != self.t:
            self._addObservation(self.observeTrueDrifters())

        if self.observation_type == dautils.ObservationType.DrifterPosition:
            return self.observedDrifterPositions[-1][1]

        elif self.observation_type == dautils.ObservationType.UnderlyingFlow:
            dt = self.observedDrifterPositions[-1][0] - self.observedDrifterPositions[-2][0]
            trueState = np.empty((self.driftersPerOceanModel, 4))
            for d in range(self.driftersPerOceanModel):
                x = self.observedDrifterPositions[-1][1][d,0]
                y = self.observedDrifterPositions[-1][1][d,1]
                dx = x - self.observedDrifterPositions[-2][1][d, 0]
                dy = y - self.observedDrifterPositions[-2][1][d, 1]
                 
                u = dx/dt
                v = dy/dt
                
                id_x = np.int(np.floor(x/self.dx))
                id_y = np.int(np.floor(y/self.dy))
                depth = self.particles[self.obs_index].downloadBathymetry()[1][id_y, id_x]
                
                hu = u*depth
                hv = v*depth

                if apply_observation_error:
                    hu += np.random.normal(scale=np.sqrt(self.observation_variance))
                    hv += np.random.normal(scale=np.sqrt(self.observation_variance))
                
                trueState[d,:] = np.array([x, y , hu, hv])
            return trueState

        elif self.observation_type == dautils.ObservationType.DirectUnderlyingFlow:
            trueState = np.empty((self.driftersPerOceanModel, 4))
            for d in range(self.driftersPerOceanModel):
                x = self.observedDrifterPositions[-1][1][d,0]
                y = self.observedDrifterPositions[-1][1][d,1]
                id_x = np.int(np.floor(x/self.dx))
                id_y = np.int(np.floor(y/self.dy))

                depth = self.particles[self.obs_index].downloadBathymetry()[1][id_y, id_x]

                # Downloading ocean state without ghost cells
                eta, hu, hv = self.downloadParticleOceanState(self.obs_index)
                true_hu = hu[id_y, id_x]
                true_hv = hv[id_y, id_x]

                if apply_observation_error:
                    true_hu += np.random.normal(scale=np.sqrt(self.observation_variance))
                    true_hv += np.random.normal(scale=np.sqrt(self.observation_variance))
                
                trueState[d,:] = np.array([x, y, true_hu, true_hv])
            return trueState

        elif self.observation_type == dautils.ObservationType.StaticBuoys:
            trueState = np.empty((self.driftersPerOceanModel, 4))
            for d in range(self.driftersPerOceanModel):
                id_x = self.buoys_ids[d][0]
                id_y = self.buoys_ids[d][1]
                # Downloading ocean state without ghost cells
                eta, hu, hv = self.downloadParticleOceanState(self.obs_index)
                true_hu = hu[id_y, id_x]
                true_hv = hv[id_y, id_x]

                if apply_observation_error:
                    true_hu += np.random.normal(scale=np.sqrt(self.observation_variance))
                    true_hv += np.random.normal(scale=np.sqrt(self.observation_variance))
                
                trueState[d,:] = np.array([self.buoys_positions[d][0], self.buoys_positions[d][1], true_hu, true_hv])
            return trueState

    def step(self, t, stochastic_particles=True, stochastic_truth=True):
        """
        Function which makes all particles (including truth) step until time t.
        apply_stochastic_term: Boolean value for whether the stochastic
            perturbation (if any) should be applied: 
            small_scale_perturbation in the simulator after every model time step
            by adding a SOAR generated random fields to the state 
            using OceanNoiseState.perturbSim(...)
        """
        for p in range(self.getNumParticles()+1):
            #print "Starting sim " + str(p)
            if p == self.obs_index:
                self.t = self.particles[p].step(t, apply_stochastic_term=stochastic_truth)
            else:
                self.t = self.particles[p].step(t, apply_stochastic_term=stochastic_particles)
            #print "Finished sim " + str(p)      
        return self.t
    
    def step_truth(self, t, stochastic=True):
        self.t = self.particles[self.obs_index].step(t, apply_stochastic_term=stochastic)
        return self.t
    
    def step_particles(self, t, stochastic=True):
        for p in range(self.getNumParticles()):
            dummy_t = self.particles[p].step(t, apply_stochastic_term=stochastic)
        return self.t
    
    
    def getDistances(self, obs=None):
        """
        Shows the distance that each drifter is from the observation in the following structure:
        [
        particle 1: [dist_drifter_1, ..., dist_drifter_D], 
        ...
        particle Ne: [dist_drifter_1, ..., dist_drifter_D], 
        ]
        """
        return np.linalg.norm(self.observeTrueDrifters() - self.observeDrifters(),  axis=2)
            
    def printMaxOceanStates(self):
        simNo = 0
        for oceanState in self.particles:
            eta, hu, hv = oceanState.download()
            if simNo == self.obs_index:
                print("------- simNo: True state -------")
            else:
                print("------- simNo: " + str(simNo) + " -------")
            print("t = " + str(oceanState.t))
            print("Max eta: ", np.max(eta))
            print("Max hu:  ", np.max(hu))
            print("Max hv:  ", np.max(hv))
            simNo = simNo + 1
    
    
    def getCauchyWeight(self, distances=None, normalize=True):
        """
        Calculates a weight associated to every particle, based on its distance from the observation, 
        using Cauchy distribution based on the uncertainty of the position of the observation.
        This distribution should be used if wider tails of the distribution is beneficial.
        """
        
        if distances is None:
            distances = self.getDistances()
        observationVariance = self.getObservationVariance()
            
        weights = 1.0/(np.pi*np.sqrt(observationVariance)*(1 + (distances**2/observationVariance)))
        if normalize:
            return weights/np.sum(weights)
        return weights
    
    
    def setTrueDrifterPositions(self, newPositions):
        """
        Manipulate the position of the true drifters.
        """
        self.particles[self.obs_index].drifters.setDrifterPositions(newPositions)

    
    def findLargestPossibleTimeStep(self):
        """
        Un-optimized utility function to check largest allowed time-step for 
        the shallow water model.
        dt < 0.25*min[ dx/max(|u +- sqrt(g(H+eta))|), dy/max(|v +- sqrt(g(H+eta))|)]
        """
        max_u = 0
        max_v = 0
        for oceanState in self.particles:
            eta_tmp, hu, hv = oceanState.download()
            Hi = oceanState.downloadBathymetry()[0]
            w = 0.25*(Hi[1:,1:] + Hi[1:,:-1] + Hi[:-1,1:] + Hi[:-1,:-1]) + eta_tmp
            hu /= w
            hv /= w
            Hi = None
            w = np.sqrt(self.g*w)
            
            # using eta_tmp buffer for {u|v} +- sqrt(gw)
            max_u = max(max_u, np.max(np.abs(hu + w)))
            max_u = max(max_u, np.max(np.abs(hu - w)))
            max_v = max(max_v, np.max(np.abs(hv + w)))
            max_v = max(max_v, np.max(np.abs(hv - w)))
            
        return 0.25*min(self.dx/max_u, self.dy/max_v)
            
            
    def getEnsembleVarAndRMSEUnderDrifter(self, t, allDrifters=False):
        """
        Putting entries in the statistical arrays for single cells.
        1. unbiased estimate of variance 
        2. root mean square error compared to the truth (rmse_truth)
        3. rmse_truth normalized by the norm of the truth
        4. ratio of variance and rmse_truth 
        by default considering the first drifter only
        """
        
        drifter_pos = self.observeTrueState()[0][0:2]
        
        # downloadTrueOceanState and downloadParticleOceanState gives us interior domain only,
        # and no ghost cells.
        cell_id_x = int(np.floor(drifter_pos[0]/self.dx))
        cell_id_y = int(np.floor(drifter_pos[1]/self.dy))
        
        eta_true_array, hu_true_array, hv_true_array = self.downloadTrueOceanState()
        
        eta_true = eta_true_array[cell_id_y, cell_id_x]
        hu_true  =  hu_true_array[cell_id_y, cell_id_x]
        hv_true  =  hv_true_array[cell_id_y, cell_id_x]
        
        eta_mean = 0.0
        hu_mean = 0.0
        hv_mean = 0.0
        eta_rmse = 0.0
        hu_rmse = 0.0
        hv_rmse = 0.0
        eta_rmse_weighted = 0.0
        hu_rmse_weighted = 0.0
        hv_rmse_weighted = 0.0
        eta_sigma = 0.0
        hu_sigma = 0.0
        hv_sigma = 0.0
        
        numNonZeros = 0
        for p in range(self.getNumParticles()):
            tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
            if not np.isnan(tmp_eta[cell_id_y, cell_id_x]):
                eta_mean += tmp_eta[cell_id_y, cell_id_x]
                hu_mean += tmp_hu[cell_id_y, cell_id_x]
                hv_mean += tmp_hv[cell_id_y, cell_id_x]
                eta_rmse += (eta_true - tmp_eta[cell_id_y, cell_id_x])**2
                hu_rmse += (hu_true - tmp_hu[cell_id_y, cell_id_x])**2
                hv_rmse += (hv_true - tmp_hv[cell_id_y, cell_id_x])**2
                numNonZeros += 1
        
        eta_mean = eta_mean/numNonZeros
        hu_mean = hu_mean/numNonZeros
        hv_mean = hv_mean/numNonZeros

        eta_rmse = np.sqrt(eta_rmse/numNonZeros)
        hu_rmse  = np.sqrt( hu_rmse/numNonZeros)
        hv_rmse  = np.sqrt( hv_rmse/numNonZeros)
        
        # RMSE according to the paper draft
        eta_rmse_mean = np.sqrt((eta_true - eta_mean)**2)
        hu_rmse_mean  = np.sqrt((hu_true  - hu_mean )**2)
        hv_rmse_mean  = np.sqrt((hv_true  - hv_mean )**2)

        eta_rmse_weighted = eta_rmse/abs(eta_true)
        hu_rmse_weighted  = hu_rmse/abs(hu_true)
        hv_rmse_weighted  = hv_rmse/abs(hv_true)
        
        numNonZeros = 0
        for p in range(self.getNumParticles()):
            tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
            if not np.isnan(tmp_eta[cell_id_y, cell_id_x]):
                eta_sigma += (tmp_eta[cell_id_y, cell_id_x] - eta_mean)**2
                hu_sigma  += (tmp_hu[cell_id_y, cell_id_x]  - hu_mean )**2
                hv_sigma  += (tmp_hv[cell_id_y, cell_id_x]  - hv_mean )**2
                numNonZeros += 1
        
        eta_sigma = np.sqrt(eta_sigma/(numNonZeros-1.0))
        hu_sigma  = np.sqrt( hu_sigma/(numNonZeros-1.0))
        hv_sigma  = np.sqrt( hv_sigma/(numNonZeros-1.0))
        
        eta_r = eta_sigma/eta_rmse
        hu_r  =  hu_sigma/hu_rmse
        hv_r  =  hv_sigma/hv_rmse
        
        self.varianceUnderDrifter_eta.append(eta_sigma)
        self.varianceUnderDrifter_hu.append(hu_sigma)
        self.varianceUnderDrifter_hv.append(hv_sigma)
        self.rmseUnderDrifter_eta.append(eta_rmse)
        self.rmseUnderDrifter_hu.append(hu_rmse)
        self.rmseUnderDrifter_hv.append(hv_rmse)
        self.rmseMeanUnderDrifter_eta.append(eta_rmse_mean)
        self.rmseMeanUnderDrifter_hu.append(hu_rmse_mean)
        self.rmseMeanUnderDrifter_hv.append(hv_rmse_mean)
        self.rmseWeightedUnderDrifter_eta.append(eta_rmse_weighted)
        self.rmseWeightedUnderDrifter_hu.append(hu_rmse_weighted)
        self.rmseWeightedUnderDrifter_hv.append(hv_rmse_weighted)
        self.rUnderDrifter_eta.append(eta_r)
        self.rUnderDrifter_hu.append(hu_r)
        self.rUnderDrifter_hv.append(hv_r)
        self.tArray.append(t)
        

        
    
    def downloadEnsembleStatisticalFields(self):
        """
        Find the ensemble mean, and the ensemble root mean-square error. 
        """
        eta_true, hu_true, hv_true = self.downloadTrueOceanState()
        
        eta_mean = np.zeros_like(eta_true)
        hu_mean = np.zeros_like(hu_true)
        hv_mean = np.zeros_like(hv_true)
        eta_rmse = np.zeros_like(eta_true)
        hu_rmse = np.zeros_like(hu_true)
        hv_rmse = np.zeros_like(hv_true)
        eta_r = np.zeros_like(eta_true)
        hu_r = np.zeros_like(hu_true)
        hv_r = np.zeros_like(hv_true)
        
        for p in range(self.getNumParticles()):
            tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
            eta_mean += tmp_eta
            hu_mean += tmp_hu
            hv_mean += tmp_hv
            eta_rmse += (eta_true - tmp_eta)**2
            hu_rmse += (hu_true - tmp_hu)**2
            hv_rmse += (hv_true - tmp_hv)**2
        
        eta_rmse = np.sqrt(eta_rmse/(self.getNumParticles()))
        hu_rmse  = np.sqrt(hu_rmse /(self.getNumParticles()))
        hv_rmse  = np.sqrt(hv_rmse /(self.getNumParticles()))
        
        eta_mean = eta_mean/self.getNumParticles()
        hu_mean = hu_mean/self.getNumParticles()
        hv_mean = hv_mean/self.getNumParticles()
        
        # RMSE according to the paper draft
        eta_rmse = np.sqrt((eta_true - eta_mean)**2)
        hu_rmse  = np.sqrt((hu_true  - hu_mean )**2)
        hv_rmse  = np.sqrt((hv_true  - hv_mean )**2)
        
        for p in range(self.getNumParticles()):
            tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
            eta_r += (tmp_eta - eta_mean)**2
            hu_r  += (tmp_hu  - hu_mean )**2
            hv_r  += (tmp_hv  - hv_mean )**2
            
        eta_r = np.sqrt(eta_r/(1.0 + self.getNumParticles()))/eta_rmse
        hu_r  = np.sqrt(hu_r /(1.0 + self.getNumParticles()))/hu_rmse
        hv_r  = np.sqrt(hv_r /(1.0 + self.getNumParticles()))/hv_rmse
        
        #print "min-max [eta, hu, hv]_r: ", [(np.min(eta_r), np.max(eta_r)), \
        #                                  (np.min(hu_r ), np.max(hu_r )), \
        #                                  (np.min(hv_r ), np.max(hv_r ))]
        
        return eta_mean, hu_mean, hv_mean, eta_rmse, hu_rmse, hv_rmse, eta_r, hu_r, hv_r
    
    def downloadParticleOceanState(self, particleNo):
        assert(particleNo < self.getNumParticles()+1), "particle out of range"
        return self.particles[particleNo].download(interior_domain_only=True)
        
    def downloadTrueOceanState(self):
        return self.particles[self.obs_index].download(interior_domain_only=True)
        
    def _updateMinMax(self, eta, hu, hv, fieldRanges):
        fieldRanges[0] = min(fieldRanges[0], np.min(eta))
        fieldRanges[1] = max(fieldRanges[1], np.max(eta))
        fieldRanges[2] = min(fieldRanges[2], np.min(hu ))
        fieldRanges[3] = max(fieldRanges[3], np.max(hu ))
        fieldRanges[4] = min(fieldRanges[4], np.min(hv ))
        fieldRanges[5] = max(fieldRanges[5], np.max(hv ))
