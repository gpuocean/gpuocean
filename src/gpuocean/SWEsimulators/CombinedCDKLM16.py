# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2016 SINTEF ICT, 
Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

This python module implements the finite-volume scheme proposed by
Alina Chertock, Michael Dudzinski, A. Kurganov & Maria Lukacova-Medvidova (2016)
Well-Balanced Schemes for the Shallow Water Equations with Coriolis Forces

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

#Import packages we need
import numpy as np
import datetime
import logging
from scipy.interpolate import interp2d

from gpuocean.utils import Common, SimWriter, SimReader, WindStress
from gpuocean.SWEsimulators import Simulator, OceanStateNoise
from gpuocean.utils import OceanographicUtilities, NetCDFInitialization

from gpuocean.SWEsimulators import CDKLM16

# Needed for the random perturbation of the wind forcing:
import pycuda.driver as cuda


class CombinedCDKLM16():
    """
    Class that solves two SW equations
    - one sim for barotropic model
    - one sim for baroclinic model
    """

    def __init__(self, \
                 gpu_ctx, \
                 barotropic_eta0, barotropic_hu0, barotropic_hv0, barotropic_H, \
                 baroclinic_eta0, baroclinic_hu0, baroclinic_hv0, baroclinic_H, \
                 nx, ny, \
                 dx, dy, dt, \
                 barotropic_g, baroclinic_g, f, r, \
                 sim_flags = {"barotropic":True, "baroclinic":True}, \
                 subsample_f=10, \
                 angle=np.array([[0]], dtype=np.float32), \
                 subsample_angle=10, \
                 latitude=None, \
                 t=0.0, \
                 theta=1.3, rk_order=2, \
                 coriolis_beta=0.0, \
                 max_wind_direction_perturbation = 0, \
                 wind_stress=WindStress.WindStress(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 barotropic_boundary_conditions_data=Common.BoundaryConditionsData(), \
                 baroclinic_boundary_conditions_data=Common.BoundaryConditionsData(), \
                 small_scale_perturbation=False, \
                 small_scale_perturbation_amplitude=None, \
                 small_scale_perturbation_interpolation_factor = 1, \
                 model_time_step=None,
                 reportGeostrophicEquilibrium=False, \
                 use_lcg=False, \
                 xorwow_seed = None, \
                 write_netcdf=False, \
                 comm=None, \
                 local_particle_id=0, \
                 super_dir_name=None, \
                 netcdf_filename=None, \
                 ignore_ghostcells=False, \
                 courant_number=0.8, \
                 offset_x=0, offset_y=0, \
                 flux_slope_eps = 1.0e-1, \
                 desingularization_eps = 1.0e-1, \
                 depth_cutoff = 1.0e-5, \
                 block_width=12, block_height=32, num_threads_dt=256,
                 block_width_model_error=16, block_height_model_error=16):
        """
        Initialization routine
        barotropic_eta0: Initial deviation from mean sea level incl ghost cells, (nx+2)*(ny+2) cells
        barotropic_hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+2) cells
        barotropic_hv0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+1) cells
        barotropic_H: Depth from equilibrium defined on cell corners, (nx+5)*(ny+5) corners
        baroclinic_eta0: Zeros like (nx+2)*(ny+2) cells
        baroclinic_hu0: Initial momentum of reduced gravity layer along x-axis incl ghost cells, (nx+1)*(ny+2) cells
        baroclinic_hv0: Initial momentum of reduced gravity layer along y-axis incl ghost cells, (nx+2)*(ny+1) cells
        baroclinic_H: Upper layer equilibriums depth defined on cell corners, (nx+5)*(ny+5) corners
        nx: Number of cells along x-axis
        ny: Number of cells along y-axis
        dx: Grid cell spacing along x-axis (20 000 m)
        dy: Grid cell spacing along y-axis (20 000 m)
        dt: Size of each timestep (90 s)
        barotropic_g: Gravitational accelleration (9.81 m/s^2)
        baroclinic_g: Reduced gravity (around 0.1 m/s^2)
        f: Coriolis parameter (1.2e-4 s^1), effectively as f = f + beta*y
        r: Bottom friction coefficient (2.4e-3 m/s)
        subsample_f: Subsample the coriolis f when creating texture by factor
        angle: Angle of rotation from North to y-axis as a texture (cuda.Array) or numpy array (in radians)
        subsample_angle: Subsample the angles given as input when creating texture by factor
        latitude: Specify latitude. This will override any f and beta plane already set (in radians)
        t: Start simulation at time t
        theta: MINMOD theta used the reconstructions of the derivatives in the numerical scheme
        rk_order: Order of Runge Kutta method {1,2*,3}
        coriolis_beta: Coriolis linear factor -> f = f + beta*(y-y_0)
        max_wind_direction_perturbation: Large-scale model error emulation by per-time-step perturbation of wind direction by +/- max_wind_direction_perturbation (degrees)
        wind_stress: Wind stress parameters
        boundary_conditions: Boundary condition object
        small_scale_perturbation: Boolean value for applying a stochastic model error
        small_scale_perturbation_amplitude: Amplitude (q0 coefficient) for model error
        small_scale_perturbation_interpolation_factor: Width factor for correlation in model error
        model_time_step: The size of a data assimilation model step (default same as dt)
        reportGeostrophicEquilibrium: Calculate the Geostrophic Equilibrium variables for each superstep
        use_lcg: Use LCG as the random number generator. Default is False, which means using curand.
        write_netcdf: Write the results after each superstep to a netCDF file
        comm: MPI communicator
        local_particle_id: Local (for each MPI process) particle id
        desingularization_eps: Used for desingularizing hu/h
        flux_slope_eps: Used for setting zero flux for symmetric Riemann fan
        depth_cutoff: Used for defining dry cells
        super_dir_name: Directory to write netcdf files to
        netcdf_filename: Use this filename. (If not defined, a filename will be generated by SimWriter.)
        """
               
        self.logger = logging.getLogger(__name__)

        if sim_flags["barotropic"]:
            self.barotropic_sim = CDKLM16.CDKLM16(gpu_ctx, \
                 barotropic_eta0, barotropic_hu0, barotropic_hv0, barotropic_H, \
                 nx, ny, \
                 dx, dy, dt, \
                 barotropic_g, f, r, \
                 subsample_f=subsample_f, \
                 angle=angle, \
                 subsample_angle=subsample_angle, \
                 latitude=latitude, \
                 t=t, \
                 theta=theta, rk_order=rk_order, \
                 coriolis_beta=coriolis_beta, \
                 max_wind_direction_perturbation = max_wind_direction_perturbation, \
                 wind_stress=wind_stress, \
                 boundary_conditions=boundary_conditions, \
                 boundary_conditions_data=barotropic_boundary_conditions_data, \
                 small_scale_perturbation=small_scale_perturbation, \
                 small_scale_perturbation_amplitude=small_scale_perturbation_amplitude, \
                 small_scale_perturbation_interpolation_factor = small_scale_perturbation_interpolation_factor, \
                 model_time_step=model_time_step,
                 reportGeostrophicEquilibrium=reportGeostrophicEquilibrium, \
                 use_lcg=use_lcg, \
                 xorwow_seed = xorwow_seed, \
                 write_netcdf=write_netcdf, \
                 comm=comm, \
                 local_particle_id=local_particle_id, \
                 super_dir_name=super_dir_name, \
                 netcdf_filename="barotropic_"+ datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"), \
                 ignore_ghostcells=ignore_ghostcells, \
                 courant_number=courant_number, \
                 offset_x=offset_x, offset_y=offset_y, \
                 flux_slope_eps = flux_slope_eps, \
                 desingularization_eps = desingularization_eps, \
                 depth_cutoff = depth_cutoff, \
                 block_width=block_width, block_height=block_height, num_threads_dt=num_threads_dt,
                 block_width_model_error=block_width_model_error, block_height_model_error=block_height_model_error)
        else:
            self.barotropic_sim = None
            
        if sim_flags["baroclinic"]:
            self.baroclinic_sim = CDKLM16.CDKLM16(gpu_ctx, \
                 baroclinic_eta0, baroclinic_hu0, baroclinic_hv0, baroclinic_H, \
                 nx, ny, \
                 dx, dy, dt, \
                 baroclinic_g, f, r, \
                 subsample_f=subsample_f, \
                 angle=angle, \
                 subsample_angle=subsample_angle, \
                 latitude=latitude, \
                 t=t, \
                 theta=theta, rk_order=rk_order, \
                 coriolis_beta=coriolis_beta, \
                 max_wind_direction_perturbation = max_wind_direction_perturbation, \
                 wind_stress=wind_stress, \
                 boundary_conditions=boundary_conditions, \
                 boundary_conditions_data=baroclinic_boundary_conditions_data, \
                 small_scale_perturbation=small_scale_perturbation, \
                 small_scale_perturbation_amplitude=small_scale_perturbation_amplitude, \
                 small_scale_perturbation_interpolation_factor = small_scale_perturbation_interpolation_factor, \
                 model_time_step=model_time_step,
                 reportGeostrophicEquilibrium=reportGeostrophicEquilibrium, \
                 use_lcg=use_lcg, \
                 xorwow_seed = xorwow_seed, \
                 write_netcdf=write_netcdf, \
                 comm=comm, \
                 local_particle_id=local_particle_id, \
                 super_dir_name=super_dir_name, \
                 netcdf_filename="baroclinic"+ datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"), \
                 ignore_ghostcells=ignore_ghostcells, \
                 courant_number=courant_number, \
                 offset_x=offset_x, offset_y=offset_y, \
                 flux_slope_eps = flux_slope_eps, \
                 desingularization_eps = desingularization_eps, \
                 depth_cutoff = depth_cutoff, \
                 block_width=block_width, block_height=block_height, num_threads_dt=num_threads_dt,
                 block_width_model_error=block_width_model_error, block_height_model_error=block_height_model_error)
        else:
            self.baroclinic_sim = None
        
        
        
    def cleanUp(self):
        """
        Clean up function
        """
        if self.barotropic_sim is not None:
            self.barotropic_sim.cleanUp()
        
        if self.baroclinic_sim is not None:
            self.baroclinic_sim.cleanUp()
        

    def step(self, t_end=0.0, apply_stochastic_term=True, write_now=True, update_dt=False):
        """
        Function which steps n timesteps.
        apply_stochastic_term: Boolean value for whether the stochastic
            perturbation (if any) should be applied after every simulation time step
            by adding SOAR-generated random fields using OceanNoiseState.perturbSim(...)
        """
        
        if self.t == 0:
            self.bc_kernel.update_bc_values(self.gpu_stream, self.t)
            self.bc_kernel.boundaryCondition(self.gpu_stream, \
                                             self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0)
        
        t_now = 0.0
        while (t_now < t_end):
        #for i in range(0, n):
            # Get new random wind direction (emulationg large-scale model error)
            if(self.max_wind_direction_perturbation > 0.0 and self.wind_stress.type() == 1):
                # max perturbation +/- max_wind_direction_perturbation deg within original wind direction (at t=0)
                perturbation = 2.0*(np.random.rand()-0.5) * self.max_wind_direction_perturbation;
                new_wind_stress = WindStress.GenericUniformWindStress( \
                    rho_air=self.wind_stress.rho_air, \
                    wind_speed=self.wind_stress.wind_speed, \
                    wind_direction=self.wind_stress.wind_direction + perturbation)
                # Upload new wind stress params to device
                cuda.memcpy_htod_async(int(self.wind_stress_dev), new_wind_stress.tostruct(), stream=self.gpu_stream)
                
            # Calculate dt if using automatic dt
            if (update_dt):
                self.updateDt()
            local_dt = np.float32(min(self.dt, np.float32(t_end - t_now)))
            
            wind_stress_t = np.float32(self.update_wind_stress(self.kernel, self.cdklm_swe_2D))
            self.bc_kernel.update_bc_values(self.gpu_stream, self.t)

            #self.bc_kernel.boundaryCondition(self.cl_queue, \
            #            self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1)
            
            # 2nd order Runge Kutta
            if (self.rk_order == 2):

                self.callKernel(self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0, \
                                self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1, \
                                local_dt, wind_stress_t, 0)

                self.bc_kernel.boundaryCondition(self.gpu_stream, \
                        self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1)

                self.callKernel(self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1, \
                                self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0, \
                                local_dt, wind_stress_t, 1)

                # Applying final boundary conditions after perturbation (if applicable)
                
            elif (self.rk_order == 1):
                self.callKernel(self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0, \
                                self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1, \
                                local_dt, wind_stress_t, 0)
                                
                self.gpu_data.swap()

                # Applying boundary conditions after perturbation (if applicable)
                
            # 3rd order RK method:
            elif (self.rk_order == 3):

                self.callKernel(self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0, \
                                self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1, \
                                local_dt, wind_stress_t, 0)
                
                self.bc_kernel.boundaryCondition(self.gpu_stream, \
                        self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1)

                self.callKernel(self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1, \
                                self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0, \
                                local_dt, wind_stress_t, 1)

                self.bc_kernel.boundaryCondition(self.gpu_stream, \
                        self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1)

                self.callKernel(self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1, \
                                self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0, \
                                local_dt, wind_stress_t, 2)
                
                # Applying final boundary conditions after perturbation (if applicable)
            
            # Perturb ocean state with model error
            if self.small_scale_perturbation and apply_stochastic_term:
                self.small_scale_model_error.perturbSim(self)
                
            # Apply boundary conditions
            self.bc_kernel.boundaryCondition(self.gpu_stream, \
                        self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0)
            
            # Evolve drifters
            self.drifterStep(local_dt)
            
            self.t += np.float64(local_dt)
            t_now += np.float64(local_dt)
            self.num_iterations += 1
            
        if self.write_netcdf and write_now:
            self.sim_writer.writeTimestep(self)
            
        return self.t

    def drifterStep(self, dt):
        # Evolve drifters
        if self.hasDrifters:
            self.drifters.drift(self.gpu_data.h0, self.gpu_data.hu0, \
                                self.gpu_data.hv0, \
                                self.bathymetry.Bm, \
                                self.nx, self.ny, self.t, self.dx, self.dy, \
                                dt, \
                                np.int32(2), np.int32(2))
            self.drifter_t += dt
            return self.drifter_t
        

    def callKernel(self, \
                   h_in, hu_in, hv_in, \
                   h_out, hu_out, hv_out, \
                   local_dt, wind_stress_t, rk_step):

        #"Beautify" code a bit by packing four int8s into a single int32
        #Note: Must match code in kernel!
        boundary_conditions = np.int32(0)
        boundary_conditions = boundary_conditions | np.int8(self.boundary_conditions.north) << 24
        boundary_conditions = boundary_conditions | np.int8(self.boundary_conditions.south) << 16
        boundary_conditions = boundary_conditions | np.int8(self.boundary_conditions.east) << 8
        boundary_conditions = boundary_conditions | np.int8(self.boundary_conditions.west) << 0

        self.cdklm_swe_2D.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                           local_dt, \
                           np.int32(rk_step), \
                           h_in.data.gpudata, h_in.pitch, \
                           hu_in.data.gpudata, hu_in.pitch, \
                           hv_in.data.gpudata, hv_in.pitch, \
                           h_out.data.gpudata, h_out.pitch, \
                           hu_out.data.gpudata, hu_out.pitch, \
                           hv_out.data.gpudata, hv_out.pitch, \
                           self.bathymetry.Bi.data.gpudata, self.bathymetry.Bi.pitch, \
                           self.bathymetry.Bm.data.gpudata, self.bathymetry.Bm.pitch, \
                           self.bathymetry.mask_value,
                           wind_stress_t, \
                           boundary_conditions)
            
    
    def perturbState(self, q0_scale=1):
        if self.small_scale_perturbation:
            self.small_scale_model_error.perturbSim(self, q0_scale=q0_scale)
    
    def applyBoundaryConditions(self):
        self.bc_kernel.boundaryCondition(self.gpu_stream, \
                        self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0)
    
    def dataAssimilationStep(self, observation_time, model_error_final_step=True, write_now=True, courant_number=0.8):
        """
        The model runs until self.t = observation_time - self.model_time_step with model error.
        If model_error_final_step is true, another stochastic model_time_step is performed, 
        otherwise a deterministic model_time_step.
        """
        # For the IEWPF scheme, it is important that the final timestep before the
        # observation time is a full time step (fully deterministic). 
        # We therefore make sure to take the (potential) small timestep first in this function,
        # followed by appropriately many full time steps.
        
        full_model_time_steps = int(round(observation_time - self.t)/self.model_time_step)
        leftover_step_size = observation_time - self.t - full_model_time_steps*self.model_time_step

        # Avoid a too small extra timestep
        if leftover_step_size/self.model_time_step < 0.1 and full_model_time_steps > 1:
            leftover_step_size += self.model_time_step
            full_model_time_steps -= 1
        
        # Force leftover_step_size to zero if it is very small compared to the model_time_step
        if leftover_step_size/self.model_time_step < 0.00001:
            leftover_step_size = 0

        assert(full_model_time_steps > 0), "There is less than CDKLM16.model_time_step until the observation"

        # Start by updating the timestep size.
        self.updateDt(courant_number=courant_number)
            
        # Loop standard steps:
        for i in range(full_model_time_steps+1):
            
            if i == 0 and leftover_step_size == 0:
                continue
            elif i == 0:
                # Take the leftover step
                self.step(leftover_step_size, apply_stochastic_term=False, write_now=False)
                self.perturbState(q0_scale=np.sqrt(leftover_step_size/self.model_time_step))

            else:
                # Take standard steps
                self.step(self.model_time_step, apply_stochastic_term=False, write_now=False)
                if (i < full_model_time_steps) or model_error_final_step:
                    self.perturbState()
                    
            self.total_time_steps += 1
            
            # Update dt now and then
            if self.total_time_steps % 5 == 0:
                self.updateDt(courant_number=courant_number)
            
        if self.write_netcdf and write_now:
            self.sim_writer.writeTimestep(self)
    
        assert(round(observation_time) == round(self.t)), 'The simulation time is not the same as observation time after dataAssimilationStep! \n' + \
            '(self.t, observation_time, diff): ' + str((self.t, observation_time, self.t - observation_time))
    
    def writeState(self):        
        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)
        
    
    
    def updateDt(self, courant_number=None):
        """
        Updates the time step self.dt by finding the maximum size of dt according to the 
        CFL conditions, and scale it with the provided courant number (0.8 on default).
        """
        if courant_number is None:
            courant_number = self.courant_number
        
        self.per_block_max_dt_kernel.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                   self.nx, self.ny, \
                   self.dx, self.dy, \
                   self.g, \
                   self.gpu_data.h0.data.gpudata, self.gpu_data.h0.pitch, \
                   self.gpu_data.hu0.data.gpudata, self.gpu_data.hu0.pitch, \
                   self.gpu_data.hv0.data.gpudata, self.gpu_data.hv0.pitch, \
                   self.bathymetry.Bm.data.gpudata, self.bathymetry.Bm.pitch, \
                   self.bathymetry.mask_value, \
                   self.device_dt.data.gpudata, self.device_dt.pitch)
    
        self.max_dt_reduction_kernel.prepared_async_call((1,1),
                                                         (self.num_threads_dt,1,1),
                                                         self.gpu_stream,
                                                         self.num_blocks_dt,
                                                         self.device_dt.data.gpudata,
                                                         self.max_dt_buffer.data.gpudata)

        dt_host = self.max_dt_buffer.download(self.gpu_stream)
        self.dt = courant_number*dt_host[0,0]
    
    def _getMaxTimestepHost(self, courant_number=0.8):
        """
        Calculates the maximum allowed time step according to the CFL conditions and scales the
        result with the provided courant number (0.8 on default).
        This function is for reference only, and suboptimally implemented on the host.
        """
        eta, hu, hv = self.download(interior_domain_only=True)
        Hm = self.downloadBathymetry()[1][2:-2, 2:-2]
        #print(eta.shape, Hm.shape)
        
        h = eta + Hm
        gravityWaves = np.sqrt(self.g*h)
        u = hu/h
        v = hv/h
        
        max_dt = 0.25*min(self.dx/np.max(np.abs(u)+gravityWaves), 
                          self.dy/np.max(np.abs(v)+gravityWaves) )
        
        return courant_number*max_dt    
    
    def downloadBathymetry(self, interior_domain_only=False):
        Bi, Bm = self.bathymetry.download(self.gpu_stream)
        
        if interior_domain_only:
            Bi = Bi[self.interior_domain_indices[2]:self.interior_domain_indices[0]+1,  
               self.interior_domain_indices[3]:self.interior_domain_indices[1]+1] 
            Bm = Bm[self.interior_domain_indices[2]:self.interior_domain_indices[0],  
               self.interior_domain_indices[3]:self.interior_domain_indices[1]]
               
        return [Bi, Bm]
    
    def getLandMask(self, interior_domain_only=True):
        if self.gpu_data.h0.mask is None:
            return None
        
        if interior_domain_only:
            return self.gpu_data.h0.mask[2:-2,2:-2]
        else:
            return self.gpu_data.h0.mask
    
    def downloadDt(self):
        return self.device_dt.download(self.gpu_stream)

    def downloadGeoEqNorm(self):
        
        uxpvy_cpu = self.geoEq_uxpvy.download(self.gpu_stream)
        Kx_cpu = self.geoEq_Kx.download(self.gpu_stream)
        Ly_cpu = self.geoEq_Ly.download(self.gpu_stream)

        uxpvy_norm = np.linalg.norm(uxpvy_cpu)
        Kx_norm = np.linalg.norm(Kx_cpu)
        Ly_norm = np.linalg.norm(Ly_cpu)

        return uxpvy_norm, Kx_norm, Ly_norm
