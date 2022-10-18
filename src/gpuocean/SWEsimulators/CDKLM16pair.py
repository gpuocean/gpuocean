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
from doctest import ELLIPSIS_MARKER
import numpy as np


# Needed for the random perturbation:
import pycuda.driver as cuda
from gpuocean.SWEsimulators import OceanStateNoise
from gpuocean.utils import Common, NetCDFInitialization, OceanographicUtilities


class CDKLM16pair():
    """
    Class that solves the SW equations using the Coriolis well balanced reconstruction scheme, as given by the publication of Chertock, Dudzinski, Kurganov and Lukacova-Medvidova (CDFLM) in 2016.
    """

    def __init__(self, sim, slave_sim, small_scale_perturbation=False, small_scale_perturbation_amplitude=None, small_scale_perturbation_interpolation_factor=1):
        """
        Initialization routine

        sim - CDKLM16 simulation
        """
        
        # Check whether sims can be combined 
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
        self.level_rescale_factor = None
        if self.l > 0:
            self.level_rescale_factor = sim.children[0].level_rescale_factor

        self.slave_l = self.deepest_level(self.slave_sim, 0)
        assert self.l > self.slave_l, "Slave should be the coarser simulation"

        # Model error instance
        self.soar_q0 = small_scale_perturbation_amplitude
        self.interpolation_factor = small_scale_perturbation_interpolation_factor
        self.small_scale_model_error = None
        if small_scale_perturbation:
            self.pert_stream = cuda.Stream()
            self.construct_model_error()


    def construct_model_error(self):
        """
        Instance of OceanNoiseState on the finest level 
        as preparation to sample model error
        """
        rescale = self.level_rescale_factor**self.l
        self.fine_NX = np.ceil((self.sim.nx+4) * rescale).astype(np.int32)
        self.fine_NY = np.ceil((self.sim.ny+4) * rescale).astype(np.int32)

        self.fine_dx = self.sim.dx/rescale
        self.fine_dy = self.sim.dy/rescale

        self.small_scale_model_error = OceanStateNoise.OceanStateNoise(self.sim.gpu_ctx, self.pert_stream, self.fine_NX, self.fine_NY, self.fine_dx, self.fine_dy, \
                                                                    self.sim.boundary_conditions, staggered=False, \
                                                                    soar_q0=self.soar_q0, \
                                                                    interpolation_factor=self.interpolation_factor, \
                                                                    angle= NetCDFInitialization.get_texture(self.sim, "angle_tex"), \
                                                                    coriolis_f=NetCDFInitialization.get_texture(self.sim, "coriolis_f_tex"))
        

    def step(self, t, apply_stochastic_term=False):
        """
        Evolves the sim pair by `t` seconds
        """
        # Stepping forward in time
        self.sim.step(t)
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
        #FIXME: Write kernel that only creates perturbation in eta
        eta_pert = Common.CUDAArray2D(self.pert_stream, self.fine_NX, self.fine_NY, 0,0, np.zeros((self.fine_NY, self.fine_NX)))
        dummy_hu_pert = Common.CUDAArray2D(self.pert_stream, self.fine_NX, self.fine_NY, 0,0, np.zeros((self.fine_NY, self.fine_NX)))
        dummy_hv_pert = Common.CUDAArray2D(self.pert_stream, self.fine_NX, self.fine_NY, 0,0, np.zeros((self.fine_NY, self.fine_NX)))

        dummy_fine_bathymetry = Common.Bathymetry(self.sim.gpu_ctx, self.sim.gpu_stream, self.fine_NX, self.fine_NY, 0, 0, np.ones((self.fine_NY+1, self.fine_NX+1)))

        self.small_scale_model_error.perturbOceanState(eta_pert, dummy_hu_pert, dummy_hv_pert, dummy_fine_bathymetry.Bi, 0.0, ghost_cells_x=0, ghost_cells_y=0)

        def perturb_level(level_sim):
            """
            Using eta_pert to generate a perturbation of a certain level
            1 projecting onto certain level
            2 cutting out local domain
            3 generating geostrophic balanced perturbation for hu and hv
            """
            # NOTE: Some methods still work with ghost cells what is going  to be depreciated 

            # projecting onto right level
            level_NX = np.ceil((self.sim.nx+4) * level_sim.global_rescale_factor).astype(np.int32)
            level_NY = np.ceil((self.sim.ny+4) * level_sim.global_rescale_factor).astype(np.int32)
            level_eta_pert = OceanographicUtilities.rescaleMidpoints(eta_pert.data.get(), level_NX, level_NY)[2]

            # Cutting out the relevant eta perturbation
            loc_eta_pert = level_eta_pert
            if level_sim.level > 0:
                fine_y0 = np.round(level_sim.global_local_area[0][0] * (self.sim.ny+4) * level_sim.global_rescale_factor, 3)
                fine_x0 = np.round(level_sim.global_local_area[0][1] * (self.sim.nx+4) * level_sim.global_rescale_factor, 3)
                loc_eta_pert = (1 - fine_x0%1 - fine_y0%1 + fine_x0%1*fine_y0%1) * level_eta_pert[int(fine_y0):int(fine_y0+level_sim.ny+4),int(fine_x0):int(fine_x0+level_sim.nx+4)] \
                                +  (fine_x0%1 - fine_x0%1*fine_y0%1) * level_eta_pert[int(fine_y0):int(fine_y0+level_sim.ny+4),int(fine_x0+1):int(fine_x0+level_sim.nx+5)] \
                                +  (fine_y0%1 - fine_x0%1*fine_y0%1) * level_eta_pert[int(fine_y0+1):int(fine_y0+level_sim.ny+5),int(fine_x0):int(fine_x0+level_sim.nx+4)] \
                                +  fine_x0%1*fine_y0%1 * level_eta_pert[int(fine_y0+1):int(fine_y0+level_sim.ny+5),int(fine_x0+1):int(fine_x0+level_sim.nx+5)]


            # Generating hu and hv perturbations by geostrophic balance
            #FIXME: Write kernel that balances the perturbation and then updates the gpudata directly
            loc_eta_pert = Common.CUDAArray2D(self.pert_stream, level_sim.nx, level_sim.ny, 2, 2, loc_eta_pert)
            loc_tmp_pert = Common.CUDAArray2D(self.pert_stream, level_sim.nx, level_sim.ny, 2,2, np.zeros((level_sim.ny+4, level_sim.nx+4)))
            loc_hu_pert = Common.CUDAArray2D(self.pert_stream, level_sim.nx, level_sim.ny, 2,2, np.zeros((level_sim.ny+4, level_sim.nx+4)))
            loc_hv_pert = Common.CUDAArray2D(self.pert_stream, level_sim.nx, level_sim.ny, 2,2, np.zeros((level_sim.ny+4, level_sim.nx+4)))

            self.small_scale_model_error.geostrophicBalanceKernel.prepared_async_call(self.small_scale_model_error.global_size_geo_balance, self.small_scale_model_error.local_size, self.pert_stream,
                                                              np.int32(level_sim.nx), np.int32(level_sim.ny),
                                                              np.int32(level_sim.dx), np.int32(level_sim.dy),
                                                              np.int32(2), np.int32(2),

                                                              np.float32(level_sim.g), np.float32(0.0), np.float32(0.0), np.float32(0.0),

                                                              loc_eta_pert.data.gpudata, loc_eta_pert.pitch,

                                                              loc_tmp_pert.data.gpudata, loc_tmp_pert.pitch,
                                                              loc_hu_pert.data.gpudata, loc_hu_pert.pitch,
                                                              loc_hv_pert.data.gpudata, loc_hv_pert.pitch,
                                                              
                                                              level_sim.bathymetry.Bi.data.gpudata, level_sim.bathymetry.Bi.pitch,
                                                              level_sim.bathymetry.mask_value)

            loc_eta, loc_hu, loc_hv = level_sim.download()
            level_sim.upload(loc_eta+loc_eta_pert.data.get(), loc_hu+loc_hu_pert.data.get(), loc_hv+loc_hv_pert.data.get())


        def perturb_all_levels(sim):
            """
            Recursion to add perturbation based on eta_pert onto all levels
            """
            perturb_level(sim)
            if len(sim.children) > 0:
                perturb_all_levels(sim.children[0]) 

        perturb_all_levels(self.sim)
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
    

