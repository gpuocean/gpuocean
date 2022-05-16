# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018  SINTEF Digital

This python class represents an ensemble of ocean models with slightly
perturbed states. Runs on a single node.

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
import logging

from SWESimulators import CDKLM16, Common, GPUDrifterCollection, BaseOceanStateEnsemble, ParticleInfo, Observation

class OceanModelEnsemble(BaseOceanStateEnsemble.BaseOceanStateEnsemble):
    """
    Class which holds a set of simulators on a single node, possibly with drifters attached
    """
    
    def __init__(self, gpu_ctx, sim_args, data_args, numParticles,
                 observation_variance = 0.01**2, 
                 initialization_variance_factor_ocean_field = 0.0,
                 super_dir_name=None, netcdf_filename=None,
                 rank=0):
        """
        Constructor which creates numParticles slighly different ocean models
        based on the same initial conditions
        """
        
        self.logger = logging.getLogger(__name__)
        self.gpu_ctx = gpu_ctx
        self.sim_args = sim_args
        self.data_args = data_args
        self.numParticles = numParticles
        self.observation_variance = observation_variance
        self.initialization_variance_factor_ocean_field = initialization_variance_factor_ocean_field
        
        
        
        
        # Build observation covariance matrix:
        if np.isscalar(self.observation_variance):
            self.observation_cov = np.eye(2)*self.observation_variance
            self.observation_cov_inverse = np.eye(2)*(1.0/self.observation_variance)
        else:
            # Assume that we have a correctly shaped matrix here
            self.observation_cov = self.observation_variance
            self.observation_cov_inverse = np.linalg.inv(self.observation_cov)
            
            
            
        # Generate ensemble members
        self.logger.debug("Creating %d particles (ocean models)", numParticles)
        self.particles = [None] * numParticles
        self.particleInfos = [None] * numParticles
        self.drifterForecast = [None] * numParticles
        for i in range(numParticles):
            self.particles[i] = CDKLM16.CDKLM16(self.gpu_ctx, **self.sim_args, **data_args, local_particle_id=i, 
                                                super_dir_name=super_dir_name, netcdf_filename=netcdf_filename)
            self.particleInfos[i] = ParticleInfo.ParticleInfo()
                    
            if self.initialization_variance_factor_ocean_field != 0.0:
                self.particles[i].perturbState(q0_scale=self.initialization_variance_factor_ocean_field)
            
    
    def attachDrifters(self, drifter_positions):
        for i in range(self.numParticles):
        # Attach drifters if requested
            self.logger.debug("Attaching %d drifters", len(drifter_positions))
            if (len(drifter_positions) > 0):
                drifters = GPUDrifterCollection.GPUDrifterCollection(self.gpu_ctx, len(drifter_positions),
                                                                     observation_variance=self.observation_variance,
                                                                     boundaryConditions=self.data_args['boundary_conditions'],
                                                                     domain_size_x=self.data_args['nx']*self.data_args['dx'], 
                                                                     domain_size_y=self.data_args['ny']*self.data_args['dy'])
                drifters.setDrifterPositions(drifter_positions)
                self.particles[i].attachDrifters(drifters)
                
                self.drifterForecast[i] = Observation.Observation()
                self.drifterForecast[i].add_observation_from_sim(self.particles[i])
    
    def cleanUp(self):
        for oceanState in self.particles:
            if oceanState is not None:
                oceanState.cleanUp()
    
    
    
    
    def modelStep(self, sub_t, rank, update_dt=True):
        """
        Function which makes all particles step until time t.
        """
        self.logger.debug("Stepping all particles (ocean models) %f in time", sub_t)
        particle = 0
        for p in self.particles:
            self.t = p.step(sub_t)
            if(update_dt):
                p.updateDt()
                self.logger.debug("[" + str(rank) + "]: Particle " + str(particle) + " has dt " + str(p.dt))
            particle += 1
        return self.t
    
    def updateDt(self):
        """
        Function that updates dt for all particles.
        """
        self.logger.debug("Updating dt on all particles (ocean models)")
        for p in self.particles:
            p.updateDt()
    
    def dumpParticleSample(self, drifter_cells):
        for i in range(self.numParticles):
            self.particleInfos[i].add_state_sample_from_sim(self.particles[i], drifter_cells)
            
    def dumpForecastParticleSample(self):
        for i in range(self.numParticles):
            self.drifterForecast[i].add_observation_from_sim(self.particles[i])

    def observeParticles(self, drifter_positions):
        self.logger.debug("Computing velocities at given positions")
        """
        Applying the observation operator on each particle.

        Structure on the output:
        [
        particle 1:  [hu_1, hv_1], ... , [hu_D, hv_D],
        particle 2:  [hu_1, hv_1], ... , [hu_D, hv_D],
        particle Ne: [hu_1, hv_1], ... , [hu_D, hv_D]
        ]
        numpy array with dimensions (numParticles, num_drifters, 2)
        
        """
        numParticles = len(self.particles)
        num_drifters = len(drifter_positions)
        
        observed_values = np.empty((numParticles, num_drifters, 2))
        
        for p in range(numParticles):
            # Downloading ocean state without ghost cells
            eta, hu, hv = self.particles[p].download(interior_domain_only=True)

            for d in range(num_drifters):
                id_x = np.int(np.floor(drifter_positions[d, 0]/self.data_args['dx']))
                id_y = np.int(np.floor(drifter_positions[d, 1]/self.data_args['dy']))
                
                observed_values[p,d,0] = hu[id_y, id_x]
                observed_values[p,d,1] = hv[id_y, id_x]
                
        return observed_values
        
    def dumpParticleInfosToFiles(self, filename_prefix):
        """
        Default file name of dump will be particle_info_YYYY_mm_dd-HH_MM_SS_{rank}_{local_particle_id}.bz2
        """
        for p in range(self.getNumParticles()):
            filename = filename_prefix + "_" + str(p) + ".bz2"
            self.particleInfos[p].to_pickle(filename)
            
    def dumpDrifterForecastToFiles(self, filename_prefix):
        """
        Default file name of dump will be forecast_particle_info_YYYY_mm_dd-HH_MM_SS_{rank}_{local_particle_id}.bz2
        """
        for p in range(self.getNumParticles()):
            filename = filename_prefix + "_" + str(p) + ".bz2"
            self.drifterForecast[p].to_pickle(filename)
        
    def syncGPU(self):
        for p in range(self.getNumParticles()):
            self.particles[p].gpu_stream.synchronize()
