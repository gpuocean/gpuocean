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
from sqlite3 import enable_shared_cache
from xmlrpc.server import DocXMLRPCServer
import numpy as np
import datetime, copy
import logging

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


        
    def __init__(self, barotropic_sim, baroclinic_sim):


        self.logger = logging.getLogger(__name__)

        self.barotropic_gpu_ctx = barotropic_sim.gpu_ctx
        self.baroclinic_gpu_ctx = baroclinic_sim.gpu_ctx

        assert barotropic_sim.nx == baroclinic_sim.nx, "sims do NOT match: nx"
        assert barotropic_sim.ny == baroclinic_sim.ny, "sims do NOT match: ny"
        assert barotropic_sim.dx == baroclinic_sim.dx, "sims do NOT match: dx"
        assert barotropic_sim.dy == baroclinic_sim.dy, "sims do NOT match: dY"
        self.nx = barotropic_sim.nx
        self.ny = barotropic_sim.ny
        self.dx = barotropic_sim.dx
        self.dy = barotropic_sim.dy

        for cardinal in ["north", "south", "east", "west"]:
            assert getattr(barotropic_sim.boundary_conditions, cardinal) == getattr(baroclinic_sim.boundary_conditions, cardinal), "sims do NOT match: bc"
        self.boundary_conditions = barotropic_sim.boundary_conditions

        self.barotropic_sim = barotropic_sim
        self.baroclinic_sim = baroclinic_sim

        # New stream that is shared for both simulators
        self.gpu_stream = cuda.Stream()
        self.barotropic_sim.gpu_stream = self.gpu_stream
        self.baroclinic_sim.gpu_stream = self.gpu_stream

        self.hasDrifters = False

        self.barotropic_iters = 0
        self.baroclinic_iters = 0

        assert barotropic_sim.t == baroclinic_sim.t, "sims do NOT match: t"
        self.t = self.barotropic_sim.t # needed for drifters and they follow barotropic timestepping




    @classmethod
    def fromdataargs(cls, \
                 barotropic_gpu_ctx, baroclinic_gpu_ctx, \
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
                 barotropic_wind=WindStress.WindStress(), \
                 baroclinic_wind=WindStress.WindStress(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 barotropic_boundary_conditions_data=Common.BoundaryConditionsData(), \
                 baroclinic_boundary_conditions_data=Common.BoundaryConditionsData(), \
                 model_time_step=None,
                 reportGeostrophicEquilibrium=False, \
                 use_lcg=False, \
                 xorwow_seed = None, \
                 write_netcdf=False, \
                 comm=None, \
                 local_particle_id=0, \
                 super_dir_name=None, \
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
               
        if sim_flags["barotropic"]:
            barotropic_sim = CDKLM16.CDKLM16(barotropic_gpu_ctx, \
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
                 wind=barotropic_wind, \
                 boundary_conditions=boundary_conditions, \
                 boundary_conditions_data=barotropic_boundary_conditions_data, \
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

        baroclinic_sim = None  
        if sim_flags["baroclinic"]:
            baroclinic_sim = CDKLM16.CDKLM16(baroclinic_gpu_ctx, \
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
                 wind=baroclinic_wind, \
                 boundary_conditions=boundary_conditions, \
                 boundary_conditions_data=baroclinic_boundary_conditions_data, \
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


        return cls(barotropic_sim, baroclinic_sim)




    def attachDrifters(self, drifters, baroclinic_drifters=None, barotropic_drifters=None):
        # same as in SWEsimulators.Simlator.Simulator() 
        # self.drifters are drifters using combined currents
        self.hasDrifters = True
        self.drifter_t = 0.0

        self.drifters = drifters
        self.drifters.setGPUStream(self.gpu_stream)

        # drifters only using currents from one of the sims
        if baroclinic_drifters is not None:
            self.baroclinic_sim.attachDrifters(baroclinic_drifters)

        if barotropic_drifters is not None:
            self.barotropic_sim.attachDrifters(barotropic_drifters)


        
    def cleanUp(self):
        if self.barotropic_sim is not None:
            self.barotropic_sim.cleanUp()
        
        if self.baroclinic_sim is not None:
            self.baroclinic_sim.cleanUp()
        

    def combinedStep(self, t_end=0.0, apply_stochastic_term=True, write_now=True, update_dt=False, trajectory_dt=0.0, trajectories=None,\
                        baroclinic_trajectories=None, barotropic_trajectories=None):
        """
        Function which steps n timesteps.
        apply_stochastic_term: Boolean value for whether the stochastic
            perturbation (if any) should be applied after every simulation time step
            by adding SOAR-generated random fields using OceanNoiseState.perturbSim(...)
        """
        
        if self.baroclinic_sim is not None:
            baroclinic_t_now = 0.0
        else:
            baroclinic_t_now = t_end
        if self.barotropic_sim is not None:
            barotropic_t_now = 0.0
        else:
            barotropic_t_now = t_end

        trajectory_t = trajectory_dt
        
        while (baroclinic_t_now < t_end) or (barotropic_t_now < t_end):
                
            # Determine which simulator is active and step
            if baroclinic_t_now <= barotropic_t_now:
                if (update_dt):
                    self.baroclinic_sim.updateDt()
                baroclinic_local_dt = np.float32(min(self.baroclinic_sim.dt, np.float32(t_end - baroclinic_t_now)))
                barotropic_local_dt  = np.float32(0.0)
            else:
                baroclinic_local_dt = np.float32(0.0)
                if (update_dt):
                    self.barotropic_sim.updateDt()
                barotropic_local_dt = np.float32(min(self.barotropic_sim.dt, np.float32(t_end - barotropic_t_now)))
            
            # Simulate step
            if baroclinic_local_dt > 0.0:
                self.baroclinic_sim.step(baroclinic_local_dt, apply_stochastic_term=apply_stochastic_term, write_now=False)
                baroclinic_t_now += np.float64(baroclinic_local_dt)
                self.baroclinic_iters += 1
            elif barotropic_local_dt > 0.0:
                self.barotropic_sim.step(barotropic_local_dt, apply_stochastic_term=apply_stochastic_term, write_now=False)
                barotropic_t_now += np.float64(barotropic_local_dt)
                self.t = self.barotropic_sim.t
                self.barotropic_iters += 1

            # Evolve drifters
            if barotropic_local_dt > 0.0:
                self.drifterStep(barotropic_local_dt)
                if (trajectories is not None) and (baroclinic_t_now > trajectory_t) and (barotropic_t_now > trajectory_t):
                    assert (trajectories.register_buoys == False), "Only floating drifters supported for combined sim"
                    trajectories.add_observation_from_sim(self)
                    if baroclinic_trajectories is not None:
                        try:
                            baroclinic_trajectories.add_observation_from_sim(self.baroclinic_sim)
                        except:
                            pass
                    if barotropic_trajectories is not None:
                        barotropic_trajectories.add_observation_from_sim(self.barotropic_sim)
                    trajectory_t += trajectory_dt

        
        # Write to file 
        if self.barotropic_sim.write_netcdf and self.baroclinic_sim.write_netcdf and write_now:
            self.barotropic_sim.sim_writer.writeTimestep(self.barotropic_sim)
            self.baroclinic_sim.sim_writer.writeTimestep(self.baroclinic_sim)
            
        return self.barotropic_sim.t, self.baroclinic_sim.t


    def drifterStep(self, dt):
        # Evolve drifters with contribution from baroclinic and barotropic sim 
        if self.hasDrifters and self.baroclinic_sim is not None and self.barotropic_sim is not None:
            self.drifters.drift(self.baroclinic_sim.gpu_data.h0, self.baroclinic_sim.gpu_data.hu0, \
                                self.baroclinic_sim.gpu_data.hv0, \
                                self.baroclinic_sim.bathymetry.Bm, \
                                self.nx, self.ny, self.barotropic_sim.t, self.dx, self.dy, \
                                dt, \
                                np.int32(2), np.int32(2)) # using more precise t from barotropic sim
            self.drifters.drift(self.barotropic_sim.gpu_data.h0, self.barotropic_sim.gpu_data.hu0, \
                                self.barotropic_sim.gpu_data.hv0, \
                                self.barotropic_sim.bathymetry.Bm, \
                                self.nx, self.ny, self.barotropic_sim.t, self.dx, self.dy, \
                                dt, \
                                np.int32(2), np.int32(2))
            self.drifter_t += dt
            return self.drifter_t
    
    
    def getLandMask(self, interior_domain_only=True):
        if self.barotropic_sim is not None:
            return self.barotropic_sim.getLandMask(interior_domain_only=interior_domain_only)
        elif self.baroclinic_sim is not None:
            return self.baroclinic_sim.getLandMask(interior_domain_only=interior_domain_only)
    

    def download(self):
        """
        Returning 
        * eta_combined = eta_barotropic + eta_baroclinic
        * u_combined = u_barotropic + u_baroclinic
        * v_combined = v_barotropic + v_baroclinic
        for the current state of the combined simulator
        """
        baroclinic_eta, baroclinic_hu, baroclinic_hv = self.baroclinic_sim.download()
        barotropic_eta, barotropic_hu, barotropic_hv = self.barotropic_sim.download()

        baroclinic_H = self.baroclinic_sim.downloadBathymetry()[1]
        barotropic_H = self.barotropic_sim.downloadBathymetry()[1]

        baroclinic_u = baroclinic_hu/(baroclinic_H + baroclinic_eta)
        baroclinic_v = baroclinic_hv/(baroclinic_H + baroclinic_eta)

        barotropic_u = barotropic_hu/(barotropic_H + barotropic_eta)
        barotropic_v = barotropic_hv/(barotropic_H + barotropic_eta)

        return baroclinic_eta+barotropic_eta, baroclinic_u+barotropic_u, baroclinic_v+barotropic_v