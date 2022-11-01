# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean.

Copyright (C) 2018, 2019  SINTEF Digital
Copyright (C) 2018, 2019  Norwegian Meteorological Institute

This python module implements common base functionalty required by all 
the different simulators, which are the classes containing the 
different numerical schemes.

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
import pycuda
import pycuda.driver as cuda

from gpuocean.utils import Common, SimWriter

# for the chilf birth
from gpuocean.utils import WindStress, OceanographicUtilities
import copy

import gc
from abc import ABCMeta, abstractmethod
import logging

try:
    from importlib import reload
except:
    pass
    
reload(Common)

class Simulator(object):
    """
    Baseclass for different numerical schemes, all 'solving' the SW equations.
    """
    __metaclass__ = ABCMeta
    
    
    def __init__(self, \
                 gpu_ctx, \
                 nx, ny, \
                 ghost_cells_x, \
                 ghost_cells_y, \
                 dx, dy, dt, \
                 g, f, r, A, \
                 t, \
                 theta, rk_order, \
                 coriolis_beta, \
                 y_zero_reference_cell, \
                 wind, \
                 atmospheric_pressure, \
                 write_netcdf, \
                 ignore_ghostcells, \
                 offset_x, offset_y, \
                 comm, \
                 block_width, block_height, \
                 local_particle_id=0):
        """
        Setting all parameters that are common for all simulators
        """
        self.gpu_stream = cuda.Stream()
        
        self.logger = logging.getLogger(__name__)
        
        #Save input parameters
        #Notice that we need to specify them in the correct dataformat for the
        #CUDA kernel
        self.gpu_ctx = gpu_ctx
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.ghost_cells_x = np.int32(ghost_cells_x)
        self.ghost_cells_y = np.int32(ghost_cells_y)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.dt = dt
        self.g = np.float32(g)
        self.f = np.float32(f)
        self.r = np.float32(r)
        self.coriolis_beta = np.float32(coriolis_beta)
        self.wind_stress = wind
        if self.wind_stress.stress_u is None or self.wind_stress.stress_v is None:
            self.wind_stress.compute_wind_stress_from_wind()
        self.atmospheric_pressure = atmospheric_pressure
        self.y_zero_reference_cell = np.float32(y_zero_reference_cell)
        
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        #Initialize time
        self.t = t
        self.num_iterations = 0
        
        #Initialize wind stress parameters
        self.wind_stress_textures = {}
        self.wind_stress_timestamps = {}
        
        # Initialize atmospheric pressure parameters
        self.atmospheric_pressure_textures = {}
        self.atmospheric_pressure_timestamps = {}

        if A is None:
            self.A = 'NA'  # Eddy viscocity coefficient
        else:
            self.A = np.float32(A)
        
        if theta is None:
            self.theta = 'NA'
        else:
            self.theta = np.float32(theta)
        if rk_order is None:
            self.rk_order = 'NA'
        else:
            self.rk_order = np.int32(rk_order)
            
        self.hasDrifters = False
        self.drifters = None
        
        # NetCDF related parameters
        self.write_netcdf = write_netcdf
        self.ignore_ghostcells = ignore_ghostcells
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.sim_writer = None
        
        # Ensemble prediction system (EPS) parameters
        self.comm = comm # MPI communicator
        
        self.local_particle_id = local_particle_id

        # Compute kernel launch parameters
        self.local_size = (block_width, block_height, 1) 
        self.global_size = ( \
                       int(np.ceil(self.nx / float(self.local_size[0]))), \
                       int(np.ceil(self.ny / float(self.local_size[1]))) \
                      )
        
        ## Multi-Resolution Parameters
        # Originating from this Simulator, a locally rescaled simulation (a "child") can be derived
        self.children = []
        # Bookkeeping for the current simulation
        self.level = 0
        self.level_rescale_factor = 1.0
        self.global_rescale_factor = 1.0
        self.level_local_area = [[self.ghost_cells_y, self.ghost_cells_x],
                                [self.ghost_cells_y+self.ny, self.ghost_cells_y+self.nx]] # indices of area within parent sim
        self.global_local_area = [[self.ghost_cells_y/(self.ny+2*self.ghost_cells_y), self.ghost_cells_x/(self.nx+2*self.ghost_cells_x)],
                                [(self.ghost_cells_y+self.ny)/(self.ny+2*self.ghost_cells_y), (self.ghost_cells_x+self.nx)/(self.nx+2*self.ghost_cells_x)]] # ratios of area within root sim 


    """
    Function which updates the wind stress textures
    @param kernel_module Module (from get_kernel in CUDAContext)
    @param kernel_function Kernel function (from kernel_module.get_function)
    """
    def update_wind_stress(self, kernel_module, kernel_function):
        #Key used to access the hashmaps
        key = str(kernel_module)
        self.logger.debug("Setting up wind stress for %s", key)
        
        #Compute new t0 and t1
        t_max_index = len(self.wind_stress.t)-1
        t0_index = max(0, np.searchsorted(self.wind_stress.t, self.t)-1)
        t1_index = min(t_max_index, np.searchsorted(self.wind_stress.t, self.t))
        new_t0 = self.wind_stress.t[t0_index]
        new_t1 = self.wind_stress.t[t1_index]
        
        #Find the old (and update)
        old_t0 = None
        old_t1 = None
        if (key in self.wind_stress_timestamps):
            old_t0 = self.wind_stress_timestamps[key][0]
            old_t1 = self.wind_stress_timestamps[key][1]
        self.wind_stress_timestamps[key] = [new_t0, new_t1]
        
        #Log some debug info
        self.logger.debug("Times: %s", str(self.wind_stress.t))
        self.logger.debug("Time indices: [%d, %d]", t0_index, t1_index)
        self.logger.debug("Time: %s  New interval is [%s, %s], old was [%s, %s]", \
                    self.t, new_t0, new_t1, old_t0, old_t1)
                
        #Get texture references
        if (key in self.wind_stress_textures):
            X0_texref, X1_texref, Y0_texref, Y1_texref = self.wind_stress_textures[key];
        else:
            X0_texref = kernel_module.get_texref("windstress_X_current")
            Y0_texref = kernel_module.get_texref("windstress_Y_current")
            X1_texref = kernel_module.get_texref("windstress_X_next")
            Y1_texref = kernel_module.get_texref("windstress_Y_next")
        
        #Helper function to upload data to the GPU as a texture
        def setTexture(texref, numpy_array):       
            #Upload data to GPU and bind to texture reference
            texref.set_array(cuda.np_to_array(numpy_array, order="C"))
            
            # Set texture parameters
            texref.set_filter_mode(cuda.filter_mode.LINEAR) #bilinear interpolation
            texref.set_address_mode(0, cuda.address_mode.CLAMP) #no indexing outside domain
            texref.set_address_mode(1, cuda.address_mode.CLAMP)
            texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES) #Use [0, 1] indexing
            
        #If time interval has changed, upload new data
        if (new_t0 != old_t0):
            self.gpu_stream.synchronize()
            self.gpu_ctx.synchronize()
            self.logger.debug("Updating T0")
            setTexture(X0_texref, self.wind_stress.stress_u[t0_index])
            setTexture(Y0_texref, self.wind_stress.stress_v[t0_index])
            kernel_function.param_set_texref(X0_texref)
            kernel_function.param_set_texref(Y0_texref)
            self.gpu_ctx.synchronize()

        if (new_t1 != old_t1):
            self.gpu_stream.synchronize()
            self.gpu_ctx.synchronize()
            self.logger.debug("Updating T1")
            setTexture(X1_texref, self.wind_stress.stress_u[t1_index])
            setTexture(Y1_texref, self.wind_stress.stress_v[t1_index])
            kernel_function.param_set_texref(X1_texref)
            kernel_function.param_set_texref(Y1_texref)
            self.gpu_ctx.synchronize()
                
        # Store texture references (they are deleted if collected by python garbage collector)
        self.logger.debug("Textures: \n[%s, %s, %s, %s]", X0_texref, X1_texref, Y0_texref, Y1_texref)
        self.wind_stress_textures[key] = [X0_texref, X1_texref, Y0_texref, Y1_texref]
        
        # Compute the wind_stress_t linear interpolation coefficient
        wind_stress_t = 0.0
        elapsed_since_t0 = (self.t-new_t0)
        time_interval = max(1.0e-10, (new_t1-new_t0))
        wind_stress_t = max(0.0, min(1.0, elapsed_since_t0 / time_interval))
        self.logger.debug("Interpolation t is %f", wind_stress_t)
        
        return wind_stress_t
        
    """
    Function which updates the atmospheric pressure textures
    @param kernel_module Module (from get_kernel in CUDAContext)
    @param kernel_function Kernel function (from kernel_module.get_function)
    """
    def update_atmospheric_pressure(self, kernel_module, kernel_function):
        #Key used to access the hashmaps
        key = str(kernel_module)
        self.logger.debug("Setting up atmospheric pressure for %s", key)
        
        #Compute new t0 and t1
        t_max_index = len(self.atmospheric_pressure.t)-1
        t0_index = max(0, np.searchsorted(self.atmospheric_pressure.t, self.t)-1)
        t1_index = min(t_max_index, np.searchsorted(self.atmospheric_pressure.t, self.t))
        new_t0 = self.atmospheric_pressure.t[t0_index]
        new_t1 = self.atmospheric_pressure.t[t1_index]
        
        #Find the old (and update)
        old_t0 = None
        old_t1 = None
        if (key in self.atmospheric_pressure_timestamps):
            old_t0 = self.atmospheric_pressure_timestamps[key][0]
            old_t1 = self.atmospheric_pressure_timestamps[key][1]
        self.atmospheric_pressure_timestamps[key] = [new_t0, new_t1]
        
        #Log some debug info
        self.logger.debug("Times: %s", str(self.atmospheric_pressure.t))
        self.logger.debug("Time indices: [%d, %d]", t0_index, t1_index)
        self.logger.debug("Time: %s  New interval is [%s, %s], old was [%s, %s]", \
                    self.t, new_t0, new_t1, old_t0, old_t1)
                
        #Get texture references
        if (key in self.atmospheric_pressure_textures):
            P0_texref, P1_texref = self.atmospheric_pressure_textures[key];
        else:
            P0_texref = kernel_module.get_texref("atmospheric_pressure_current")
            P1_texref = kernel_module.get_texref("atmospheric_pressure_next")
        
        #Helper function to upload data to the GPU as a texture
        def setTexture(texref, numpy_array):       
            
            #Upload data to GPU and bind to texture reference
            texref.set_array(cuda.np_to_array(numpy_array, order="C"))
            
            # Set texture parameters
            texref.set_filter_mode(cuda.filter_mode.LINEAR) #bilinear interpolation
            texref.set_address_mode(0, cuda.address_mode.CLAMP) #no indexing outside domain
            texref.set_address_mode(1, cuda.address_mode.CLAMP)
            texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES) #Use [0, 1] indexing
            
        #If time interval has changed, upload new data
        if (new_t0 != old_t0):
            self.gpu_stream.synchronize()
            self.gpu_ctx.synchronize()
            self.logger.debug("Updating T0")
            setTexture(P0_texref, self.atmospheric_pressure.P[t0_index])
            kernel_function.param_set_texref(P0_texref)
            self.gpu_ctx.synchronize()

        if (new_t1 != old_t1):
            self.gpu_stream.synchronize()
            self.gpu_ctx.synchronize()
            self.logger.debug("Updating T1")
            setTexture(P1_texref, self.atmospheric_pressure.P[t1_index])
            kernel_function.param_set_texref(P1_texref)
            self.gpu_ctx.synchronize()
                
        # Store texture references (they are deleted if collected by python garbage collector)
        self.logger.debug("Textures: \n[%s, %s, %s, %s]", P0_texref, P1_texref)
        self.atmospheric_pressure_textures[key] = [P0_texref, P1_texref]
        
        # Compute the atmospheric_pressure_t linear interpolation coefficient
        atmospheric_pressure_t = 0.0
        elapsed_since_t0 = (self.t-new_t0)
        time_interval = max(1.0e-10, (new_t1-new_t0))
        atmospheric_pressure_t = max(0.0, min(1.0, elapsed_since_t0 / time_interval))
        self.logger.debug("Interpolation t is %f", atmospheric_pressure_t)
        
        return atmospheric_pressure_t
        
        
            
    @abstractmethod
    def step(self, t_end=0.0):
        """
        Function which steps n timesteps
        """
        pass
    
    @abstractmethod
    def fromfilename(cls, filename, cont_write_netcdf=True):
        """
        Initialize and hotstart simulation from nc-file.
        cont_write_netcdf: Continue to write the results after each superstep to a new netCDF file
        filename: Continue simulation based on parameters and last timestep in this file
        """
        pass
   
    def __del__(self):
        self.cleanUp()

    @abstractmethod
    def cleanUp(self):
        """
        Clean up function
        """
        pass
        
    def closeNetCDF(self):
        """
        Close the NetCDF file, if there is one
        """
        if self.write_netcdf:
            self.sim_writer.__exit__(0,0,0)
            self.write_netcdf = False
        
    def attachDrifters(self, drifters):
        ### Do the following type of checking here:
        #assert isinstance(drifters, GPUDrifters)
        #assert drifters.isInitialized()
        
        self.drifters = drifters
        self.hasDrifters = True
        self.drifters.setGPUStream(self.gpu_stream)
        self.drifter_t = 0.0
    
    def download(self, interior_domain_only=False):
        """
        Download the latest time step from the GPU
        """
        if interior_domain_only:
            eta, hu, hv = self.gpu_data.download(self.gpu_stream)
            return eta[self.interior_domain_indices[2]:self.interior_domain_indices[0],  \
                       self.interior_domain_indices[3]:self.interior_domain_indices[1]], \
                   hu[self.interior_domain_indices[2]:self.interior_domain_indices[0],   \
                      self.interior_domain_indices[3]:self.interior_domain_indices[1]],  \
                   hv[self.interior_domain_indices[2]:self.interior_domain_indices[0],   \
                      self.interior_domain_indices[3]:self.interior_domain_indices[1]]
        else:
            return self.gpu_data.download(self.gpu_stream)
    
    
    def downloadPrevTimestep(self):
        """
        Download the second-latest time step from the GPU
        """
        return self.gpu_data.downloadPrevTimestep(self.gpu_stream)
        
    def copyState(self, otherSim):
        """
        Copies the state ocean state (eta, hu, hv), the wind object and 
        drifters (if any) from the other simulator.
        
        This function is exposed to enable efficient re-initialization of
        resampled ocean states. This means that all parameters which can be 
        initialized/assigned a perturbation should be copied here as well.
        """
        
        assert type(otherSim) is type(self), "A simulator can only copy the state from another simulator of the same class. Here we try to copy a " + str(type(otherSim)) + " into a " + str(type(self))
        
        assert (self.ny, self.nx) == (otherSim.ny, otherSim.nx), "Simulators differ in computational domain. Self (ny, nx): " + str((self.ny, self.nx)) + ", vs other: " + ((otherSim.ny, otherSim.nx))
        
        self.gpu_data.h0.copyBuffer(self.gpu_stream, otherSim.gpu_data.h0)
        self.gpu_data.hu0.copyBuffer(self.gpu_stream, otherSim.gpu_data.hu0)
        self.gpu_data.hv0.copyBuffer(self.gpu_stream, otherSim.gpu_data.hv0)
        
        self.gpu_data.h1.copyBuffer(self.gpu_stream, otherSim.gpu_data.h1)
        self.gpu_data.hu1.copyBuffer(self.gpu_stream, otherSim.gpu_data.hu1)
        self.gpu_data.hv1.copyBuffer(self.gpu_stream, otherSim.gpu_data.hv1)
        
        # Question: Which parameters should we require equal, and which 
        # should become equal?
        self.wind_stress = otherSim.wind_stress
        
        if otherSim.hasDrifters and self.hasDrifters:
            self.drifters.setDrifterPositions(otherSim.drifters.getDrifterPositions())
            self.drifters.setObservationPosition(otherSim.drifters.getObservationPosition())
        
        
        
    def upload(self, eta0, hu0, hv0, eta1=None, hu1=None, hv1=None):
        """
        Reinitialize simulator with a new ocean state.
        """
        self.gpu_data.h0.upload(self.gpu_stream, eta0)
        self.gpu_data.hu0.upload(self.gpu_stream, hu0)
        self.gpu_data.hv0.upload(self.gpu_stream, hv0)
        
        if eta1 is None:
            self.gpu_data.h1.upload(self.gpu_stream, eta0)
            self.gpu_data.hu1.upload(self.gpu_stream, hu0)
            self.gpu_data.hv1.upload(self.gpu_stream, hv0)
        else:
            self.gpu_data.h1.upload(self.gpu_stream, eta1)
            self.gpu_data.hu1.upload(self.gpu_stream, hu1)
            self.gpu_data.hv1.upload(self.gpu_stream, hv1)
            
    def _set_interior_domain_from_sponge_cells(self):
        """
        Use possible existing sponge cells to correctly set the 
        variable self.interior_domain_incides
        """
        if (self.boundary_conditions.isSponge()):
            assert(False), 'This function is deprecated - sponge cells should now be considered part of the interior domain'
    
    @staticmethod
    def get_texture(sim, tex_name):
        """
        Sampling the textures with in the gpu_ctx of a sim-object.
        So far available: angle_tex, coriolis_f_tex!
        """
        texref = Common.CUDAArray2D(sim.gpu_stream, sim.nx, sim.ny, 2,2, np.zeros((sim.ny+4,sim.nx+4)))
        get_tex = sim.kernel.get_function("get_texture")
        get_tex.prepare("Pi")
        global_size = (int(np.ceil( (sim.nx+4) / float(sim.local_size[0]))), int(np.ceil( (sim.ny +4) / float(sim.local_size[1]))) )
        if tex_name == "angle_tex":
            get_tex.prepared_async_call(global_size,sim.local_size,sim.gpu_stream, texref.data.gpudata, np.int32(0))
        elif tex_name == "coriolis_f_tex":
            get_tex.prepared_async_call(global_size,sim.local_size,sim.gpu_stream, texref.data.gpudata, np.int32(1))
        elif tex_name == "windstress_X_current":
            get_tex.prepared_async_call(global_size,sim.local_size,sim.gpu_stream, texref.data.gpudata, np.int32(2))
        elif tex_name == "windstress_Y_current":
            get_tex.prepared_async_call(global_size,sim.local_size,sim.gpu_stream, texref.data.gpudata, np.int32(3))
        else:
            print("Texture name unknown! Returning 0.0")
        tex = texref.download(sim.gpu_stream)
        texref.release()
        return tex

    @staticmethod
    def sample_texture(sim, tex_name, Nx, Ny, x0, x1, y0, y1):
        """
        Sampling the textures with in the gpu_ctx of a sim-object
        with arbitrary sampling window.
        So far available: angle_tex, coriolis_f_tex!
        """
        texref = Common.CUDAArray2D(sim.gpu_stream, Nx, Ny, 2,2, np.zeros((Ny+4,Nx+4)))
        get_tex = sim.kernel.get_function("sample_texture")
        get_tex.prepare("Piffffii")
        global_size = (int(np.ceil( (Nx+4) / float(sim.local_size[0]))), int(np.ceil( (Ny +4) / float(sim.local_size[1]))) )
        if tex_name == "angle_tex":
            get_tex.prepared_async_call(global_size,sim.local_size,sim.gpu_stream, texref.data.gpudata, np.int32(0),
                                        np.float32(x0), np.float32(x1), np.float32(y0), np.float32(y1), np.int32(Nx+4), np.int32(Ny+4))
        elif tex_name == "coriolis_f_tex":
            get_tex.prepared_async_call(global_size,sim.local_size,sim.gpu_stream, texref.data.gpudata, np.int32(1),
                                        np.float32(x0), np.float32(x1), np.float32(y0), np.float32(y1), np.int32(Nx+4), np.int32(Ny+4))
        else:
            print("Texture name unknown! Returning 0.0")
        tex = texref.download(sim.gpu_stream)
        texref.release()
        return tex


    def give_birth(self, gpu_ctx_refined, loc, scale, **kwargs):
        """
        Returning a locally refined/coarsened simulation,
        initialised from self

        loc   - list with "cut-out" area 
                by indices in form [[y0,x0],[y1,x1]]
                where indices account for interior_domain_only
        scale - factor for rescaling of resolution

        kwargs - to overwrite settings, if required
        """

        # Check inputs!
        assert (loc[0]<loc[1]), "Invalid area: the 0-coordinates have to be lower than the 1-coordindates"
        assert np.all( (np.array(loc[1]) - np.array(loc[0]))*scale % 1 == 0 ), "Rescaling of local area must result in integer number of cells"

        # Checking that local areas do NOT overlap or touch!
        for child in self.children:
            assert (child.level_local_area[0][0] > child.level_local_area[1][0] 
                    or child.level_local_area[1][0] < child.level_local_area[0][0] 
                    or child.level_local_area[1][1] < child.level_local_area[0][1] 
                    or child.level_local_area[0][1] > child.level_local_area[1][1]), "Local areas must not overlap"

        # Dict with simulation information
        sim_args = {}
        sim_args["dt"] = 0.0
        sim_args["write_netcdf"] = self.write_netcdf
        sim_args["ignore_ghostcells"] = self.ignore_ghostcells
        sim_args["offset_x"] = self.offset_x
        sim_args["offset_y"] = self.offset_y
        if hasattr(self, 'model_time_step'):
            sim_args["model_time_step"] = self.model_time_step
        if hasattr(self, 'courant_number'):
            sim_args["courant_number"] = self.courant_number
        sim_args["t"] = self.t

        # Dict collecting locally rescaled IC and BC
        data_args_loc_refined = {}

        eta0_loc, hu0_loc, hv0_loc = [ x[ loc[0][0]+self.ghost_cells_y : loc[1][0]+self.ghost_cells_y, 
                                            loc[0][1]+self.ghost_cells_x : loc[1][1]+self.ghost_cells_x ] 
                                        for x in self.download() ]

        ny_loc, nx_loc = np.array(eta0_loc.shape)
        data_args_loc_refined["ny"], data_args_loc_refined["nx"] = int(ny_loc * scale), int(nx_loc * scale)

        # The variable and bathymetry values in the ghost cells are not relevant, 
        # since they are anyways overwritten by the BC kernels

        # TODO: If bathymetry changes, this has to be adapted here!
        H_loc_refined = OceanographicUtilities.rescaleIntersections(
                                            self.downloadBathymetry()[0][loc[0][0]+self.ghost_cells_y : loc[1][0]+self.ghost_cells_y+1, 
                                                                            loc[0][1]+self.ghost_cells_x : loc[1][1]+self.ghost_cells_x+1], 
                                            data_args_loc_refined["nx"]+1, data_args_loc_refined["ny"]+1)[2]
        H_loc_refined_data = np.pad(H_loc_refined.data, ((2,2),(2,2)), mode="edge")
        H_loc_refined_mask = np.pad(H_loc_refined.mask, ((2,2),(2,2)), mode="edge")
        data_args_loc_refined["H"] = np.ma.array(H_loc_refined_data, mask= H_loc_refined_mask)

        # TODO: If bathymetry changes, this has to be adapted here!
        eta0_loc_refined = OceanographicUtilities.rescaleMidpoints(eta0_loc, data_args_loc_refined["nx"], data_args_loc_refined["ny"])[2]
        eta0_loc_refined_data = np.pad( eta0_loc_refined.data, ((2,2),(2,2)), mode="edge")
        eta0_loc_refined_mask = np.pad( eta0_loc_refined.mask, ((2,2),(2,2)), mode="edge")
        data_args_loc_refined["eta0"] = np.ma.array(eta0_loc_refined_data, mask=eta0_loc_refined_mask)

        hu0_loc_refined = OceanographicUtilities.rescaleMidpoints(hu0_loc, data_args_loc_refined["nx"], data_args_loc_refined["ny"])[2]
        hu0_loc_refined_data = np.pad( hu0_loc_refined.data, ((2,2),(2,2)), mode="edge")
        hu0_loc_refined_mask = np.pad( hu0_loc_refined.mask, ((2,2),(2,2)), mode="edge")
        data_args_loc_refined["hu0"] = np.ma.array(hu0_loc_refined_data, mask=hu0_loc_refined_mask)

        hv0_loc_refined = OceanographicUtilities.rescaleMidpoints(hv0_loc, data_args_loc_refined["nx"], data_args_loc_refined["ny"])[2]
        hv0_loc_refined_data = np.pad( hv0_loc_refined.data, ((2,2),(2,2)), mode="edge")
        hv0_loc_refined_mask = np.pad( hv0_loc_refined.mask, ((2,2),(2,2)), mode="edge")
        data_args_loc_refined["hv0"] = np.ma.array(hv0_loc_refined_data, mask=hv0_loc_refined_mask)


        data_args_loc_refined["dx"], data_args_loc_refined["dy"] = self.dx/scale, self.dy/scale 

        data_args_loc_refined["g"] = self.g
        data_args_loc_refined["r"] = self.r

        # Cell centers of the edge cells in the ghost cell halo
        tex_x0 = (loc[0][1]+self.ghost_cells_x)/(self.nx+4) - 1.5/(self.nx+4)/scale
        tex_x1 = (loc[1][1]+self.ghost_cells_x)/(self.nx+4) + 1.5/(self.nx+4)/scale

        tex_y0 = (loc[0][0]+self.ghost_cells_y)/(self.ny+4) - 1.5/(self.ny+4)/scale
        tex_y1 = (loc[1][0]+self.ghost_cells_y)/(self.ny+4) + 1.5/(self.ny+4)/scale

        data_args_loc_refined["angle"] = Simulator.sample_texture(self, "angle_tex", 
                                                                    data_args_loc_refined["nx"], data_args_loc_refined["ny"], 
                                                                    tex_x0, tex_x1, tex_y0, tex_y1)

        data_args_loc_refined["f"] = Simulator.sample_texture(self, "coriolis_f_tex", 
                                                                    data_args_loc_refined["nx"], data_args_loc_refined["ny"], 
                                                                    tex_x0, tex_x1, tex_y0, tex_y1)


        wind_t = self.wind_stress.t

        def windstress_t(sim, wind_t_idx, coord, nx, ny, x0, x1, y0, y1):
            ## Set wind as texture
            if coord == "x":
                GPUtexref = sim.kernel.get_texref("windstress_X_current")
            elif coord == "y":
                GPUtexref = sim.kernel.get_texref("windstress_Y_current")

            sim.gpu_stream.synchronize()
            sim.gpu_ctx.synchronize()

            if coord == "x":
                GPUtexref.set_array(cuda.np_to_array(sim.wind_stress.stress_u[wind_t_idx], order="C"))
            elif coord == "y":
                GPUtexref.set_array(cuda.np_to_array(sim.wind_stress.stress_v[wind_t_idx], order="C"))
            GPUtexref.set_filter_mode(cuda.filter_mode.LINEAR) #bilinear interpolation
            GPUtexref.set_address_mode(0, cuda.address_mode.CLAMP) #no indexing outside domain
            GPUtexref.set_address_mode(1, cuda.address_mode.CLAMP)
            GPUtexref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES) #Use [0, 1] indexing

            sim.gpu_ctx.synchronize()

            
            texref = Common.CUDAArray2D(sim.gpu_stream, nx, ny, 2,2, np.zeros((ny+4,nx+4)))
            get_tex = sim.kernel.get_function("sample_texture")
            get_tex.prepare("Piffffii")
            if coord == "x":
                get_tex.prepared_async_call(sim.global_size,sim.local_size,sim.gpu_stream, texref.data.gpudata, np.int32(2),
                                            np.float32(x0), np.float32(x1), np.float32(y0), np.float32(y1), np.int32(nx+4), np.int32(ny+4))
            elif coord == "y":
                get_tex.prepared_async_call(sim.global_size,sim.local_size,sim.gpu_stream, texref.data.gpudata, np.int32(3),
                                            np.float32(x0), np.float32(x1), np.float32(y0), np.float32(y1), np.int32(nx+4), np.int32(ny+4))
            tex = texref.download(sim.gpu_stream)
            texref.release()
            return tex

        stress_u = [windstress_t(self, t_idx, "x", data_args_loc_refined["nx"], data_args_loc_refined["ny"], tex_x0, tex_x1, tex_y0, tex_y1) for t_idx in range(len(wind_t))]
        stress_v = [windstress_t(self, t_idx, "y", data_args_loc_refined["nx"], data_args_loc_refined["ny"], tex_x0, tex_x1, tex_y0, tex_y1) for t_idx in range(len(wind_t))]

        data_args_loc_refined["wind"] = WindStress.WindStress(t=wind_t, stress_u=stress_u, stress_v=stress_v)

        data_args_loc_refined["boundary_conditions"] = copy.deepcopy(self.boundary_conditions)


        # Replace specific kwargs in derived dicts
        for key in set(kwargs).intersection(set(sim_args)):
            sim_args[key] = kwargs[key]

        for key in set(kwargs).intersection(set(data_args_loc_refined)):
            data_args_loc_refined[key] = kwargs[key]

        # Generate child
        self.children.append(type(self)(gpu_ctx_refined, **sim_args, **data_args_loc_refined, **kwargs))

        self.children[-1].level = self.level + 1
        self.children[-1].level_rescale_factor = scale
        self.children[-1].global_rescale_factor = self.global_rescale_factor * scale
        self.children[-1].level_local_area = loc
        global_local_area_x = self.global_local_area[1][1] - self.global_local_area[0][1] 
        global_local_area_y = self.global_local_area[1][0] - self.global_local_area[0][0]
        self.children[-1].global_local_area = [ [self.global_local_area[0][0] + loc[0][0]/self.ny*global_local_area_y, \
                                                    self.global_local_area[0][1] + loc[0][1]/self.nx*global_local_area_x ], \
                                                [self.global_local_area[0][0] + loc[1][0]/self.ny*global_local_area_y, \
                                                    self.global_local_area[0][1] + loc[1][1]/self.nx*global_local_area_x ] ]

    def kill_child(self, idx = 0):
        """
        Removing child simulator
        """
        self.children[idx].cleanUp()
        self.children.pop(idx)