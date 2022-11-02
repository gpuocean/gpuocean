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
from scipy import interpolate 

# Needed for the random perturbation:
import pycuda.driver as cuda
from gpuocean.SWEsimulators import Simulator, OceanStateNoise
from gpuocean.utils import Common, NetCDFInitialization, OceanographicUtilities


class CDKLM16pair():
    """
    Class that solves the SW equations using the Coriolis well balanced reconstruction scheme, as given by the publication of Chertock, Dudzinski, Kurganov and Lukacova-Medvidova (CDFLM) in 2016.
    """

    def __init__(self, sim, slave_sim=None, 
                    small_scale_perturbation=False, 
                    small_scale_perturbation_amplitude=None, 
                    small_scale_perturbation_interpolation_factor=1,
                    small_scale_perturbation_dx=None,
                    small_scale_perturbation_dy=None
                    ):
        """
        Initialization routine

        sim          - CDKLM16 simulation
        slave_sim    - CDKLM16 simulation
        """
        
        # Check whether sims can be combined 
        if slave_sim is not None:
            assert sim.nx == slave_sim.nx
            assert sim.ny == slave_sim.ny
            assert sim.dx == slave_sim.dx
            assert sim.dy == slave_sim.dy
            assert sim.t == slave_sim.t

        # Note CDKLM16 instances cannot be copied due to pycuda conflicts,
        # hence we need to initialise two independent simulations outside this class
        self.sim = sim
        self.slave_sim = slave_sim

        # Bookkeeping for the multi-res structure
        # FIXME: At the moment only one child and constant rescaling assumed!
        self.l = self.deepest_level(self.sim, 0)
        self.level_rescale_factor = 1.0
        if self.l > 0:
            self.level_rescale_factor = sim.children[0].level_rescale_factor

        if slave_sim is not None:
            self.slave_l = self.deepest_level(self.slave_sim, 0)
            assert self.l > self.slave_l, "Slave should be the coarser simulation"

        # Model error instance
        self.soar_q0 = small_scale_perturbation_amplitude
        self.interpolation_factor = small_scale_perturbation_interpolation_factor
        self.pert_dx = small_scale_perturbation_dx
        if self.pert_dx is None:
            self.pert_dx = sim.dx/5
        self.pert_dy = small_scale_perturbation_dy
        if self.pert_dy is None:
            self.pert_dy = sim.dy/5

        self.small_scale_model_error = None
        if small_scale_perturbation:
            self.pert_stream = cuda.Stream()
            self.construct_model_error()


    def construct_model_error(self):
        """
        Instance of OceanNoiseState on the finest level 
        as preparation to sample model error
        """
        self.pert_NX = np.ceil((self.sim.nx+4) * self.sim.dx/self.pert_dx).astype(np.int32)
        self.pert_NY = np.ceil((self.sim.ny+4) * self.sim.dy/self.pert_dy).astype(np.int32)

        self.small_scale_model_error = OceanStateNoise.OceanStateNoise(self.sim.gpu_ctx, self.pert_stream, self.pert_NX, self.pert_NY, self.pert_dx, self.pert_dx, \
                                                                    self.sim.boundary_conditions, staggered=False, \
                                                                    soar_q0=self.soar_q0, \
                                                                    interpolation_factor=self.interpolation_factor, \
                                                                    angle=Simulator.Simulator.get_texture(self.sim, "angle_tex"), \
                                                                    coriolis_f=Simulator.Simulator.get_texture(self.sim, "coriolis_f_tex"))

        self.small_scale_model_error.bicubicInterpolationEtaKernel = self.small_scale_model_error.kernels.get_function("bicubicInterpolationEta")
        self.small_scale_model_error.bicubicInterpolationEtaKernel.prepare("iiffiiffPiPi")
        

    def step(self, t, apply_stochastic_term=False):
        """
        Evolves the sim pair by `t` seconds
        """
        # Stepping forward in time
        self.sim.step(t)
        if self.slave_sim is not None:
            self.slave_sim.step(t)

        # Model error
        if apply_stochastic_term:
            # Sanity check whether levels have changed
            l = self.deepest_level(self.sim, 0)
            if l != self.l:
                self.l = l
                self.construct_model_error()

            if self.small_scale_model_error is not None:
                self.perturbSimPair()
            else:
                print("Model error ignored since not initialised")

        return self.sim.t


    def perturbSimPair(self):
        """
        Adding same realisation of model error to sim and slave sim
        """
        # creating a perturbation in eta
        eta_pert = Common.CUDAArray2D(self.pert_stream, self.pert_NX, self.pert_NY, 0,0, np.zeros((self.pert_NY, self.pert_NX)))

        self.small_scale_model_error.generateNormalDistribution()

        self.small_scale_model_error.soarKernel.prepared_async_call(self.small_scale_model_error.global_size_SOAR, self.small_scale_model_error.local_size, self.pert_stream,
                                    self.small_scale_model_error.coarse_nx, self.small_scale_model_error.coarse_ny,
                                    self.small_scale_model_error.coarse_dx, self.small_scale_model_error.coarse_dy,

                                    np.float32(self.small_scale_model_error.soar_q0), self.small_scale_model_error.soar_L,
                                    np.float32(1.0),
                                    
                                    self.small_scale_model_error.periodicNorthSouth, self.small_scale_model_error.periodicEastWest,
                                    self.small_scale_model_error.random_numbers.data.gpudata, self.small_scale_model_error.random_numbers.pitch,
                                    self.small_scale_model_error.coarse_buffer.data.gpudata, self.small_scale_model_error.coarse_buffer.pitch,
                                    np.int32(0))

        self.small_scale_model_error.bicubicInterpolationEtaKernel.prepared_async_call(self.small_scale_model_error.global_size_geo_balance, self.small_scale_model_error.local_size, self.pert_stream,
                                                                self.small_scale_model_error.nx, self.small_scale_model_error.ny, 
                                                                self.small_scale_model_error.dx, self.small_scale_model_error.dy,
                                                                
                                                                self.small_scale_model_error.coarse_nx, self.small_scale_model_error.coarse_ny,
                                                                self.small_scale_model_error.coarse_dx, self.small_scale_model_error.coarse_dy,
                                                                
                                                                self.small_scale_model_error.coarse_buffer.data.gpudata, self.small_scale_model_error.coarse_buffer.pitch,
                                                                eta_pert.data.gpudata, eta_pert.pitch)
        
        # project eta onto levels
        x0 = np.linspace(0.5*self.pert_dx, (self.pert_NX-0.5)*self.pert_dx, self.pert_NX)
        y0 = np.linspace(0.5*self.pert_dy, (self.pert_NY-0.5)*self.pert_dy, self.pert_NY)
        interp = interpolate.interp2d(x0, y0, eta_pert.download(self.small_scale_model_error.gpu_stream), kind="linear")

        def perturb_level(level_sim):
            """
            Using eta_pert to generate a perturbation of a certain level
            1 projecting onto certain level
            2 generating geostrophic balanced perturbation for hu and hv
            """

            # projecting onto right level
            global_local_area_x = level_sim.global_local_area[1][1] - level_sim.global_local_area[0][1]
            global_local_area_y = level_sim.global_local_area[1][0] - level_sim.global_local_area[0][0]

            x1 = np.linspace((level_sim.global_local_area[0][1]/(global_local_area_x/(level_sim.nx+4)) + 0.5)*level_sim.dx, 
                                (level_sim.global_local_area[1][1]/(global_local_area_x/(level_sim.nx+4)) - 0.5)*level_sim.dx, 
                            level_sim.nx+4)
            y1 = np.linspace((level_sim.global_local_area[0][0]/(global_local_area_y/(level_sim.ny+4)) + 0.5)*level_sim.dy, 
                                (level_sim.global_local_area[1][0]/(global_local_area_y/(level_sim.ny+4)) - 0.5)*level_sim.dy,
                            level_sim.ny+4)

            parent_eta_pertHOST = interp(x1,y1)
            parent_eta_pert = Common.CUDAArray2D(self.pert_stream, level_sim.nx, level_sim.ny, 2, 2, np.float32(parent_eta_pertHOST))

            # generating geostrophic balance and update gpudata
            # NOTE: using f=0, triggers gpu_ctx is evaluated
            self.small_scale_model_error.geostrophicBalanceKernel.prepared_async_call(self.small_scale_model_error.global_size_geo_balance, self.small_scale_model_error.local_size, self.pert_stream,
                                                              np.int32(level_sim.nx), np.int32(level_sim.ny),
                                                              np.int32(level_sim.dx), np.int32(level_sim.dy),
                                                              np.int32(2.0), np.int32(2.0),

                                                              np.float32(level_sim.g), np.float32(0.0), np.float32(0.0), np.float32(0.0),
                                                              np.float32(level_sim.global_local_area[0][1]), np.float32(level_sim.global_local_area[1][1]),
                                                              np.float32(level_sim.global_local_area[0][0]), np.float32(level_sim.global_local_area[1][0]),

                                                              parent_eta_pert.data.gpudata, parent_eta_pert.pitch,

                                                              level_sim.gpu_data.h0.data.gpudata, level_sim.gpu_data.h0.pitch,
                                                              level_sim.gpu_data.hu0.data.gpudata, level_sim.gpu_data.hu0.pitch,
                                                              level_sim.gpu_data.hv0.data.gpudata, level_sim.gpu_data.hv0.pitch,
                                                              
                                                              level_sim.bathymetry.Bi.data.gpudata, level_sim.bathymetry.Bi.pitch,
                                                              level_sim.bathymetry.mask_value)
          


        def perturb_all_levels(sim):
            """
            Recursion to add perturbation based on eta_pert onto all levels
            """
            # FIXME: Avoid multiple interpolation and find smarter way to navigate through tree!!!
            perturb_level(sim)
            if len(sim.children) > 0:
                perturb_all_levels(sim.children[0]) 

        perturb_all_levels(self.sim)
        if self.slave_sim is not None:
            perturb_all_levels(self.slave_sim)


    @staticmethod
    def deepest_level(sim, l, scale=None):
        if len(sim.children) == 0:
            return l
        else:
            assert len(sim.children) <= 1, "Pairs only implemented for single-child simulations" # extension is just matter of implementation, but no theoretical issues
            if scale is not None:
                assert sim.children[0].level_rescale_factor == scale, "Pairs only implemented for constant rescaling" # extension is just matter of implementation, but no theoretical issues
            else:
                scale = sim.children[0].level_rescale_factor
            return max([CDKLM16pair.deepest_level(child,l+1, scale) for child in sim.children])
    

