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

        # Model error object
        self.model_error = None
        
        self.hasCrossProductDrifter = False
        self.CrossProductDrifter = None
        
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

    def attachCrossProductDrifters(self, drifter_list, sim_list):
        """
        Attach drifters that are affected by this sim (self), but also others from sim_list.
        """
        assert len(drifter_list) == len(sim_list), "Same number of drifter objects and partner simulations needed!"
        self.hasCrossProductDrifter = True
        self.CrossProductDrifter = drifter_list
        for d in self.CrossProductDrifter:
            d.setGPUStream(self.gpu_stream)
        self.CPsims = sim_list
        self.CPdrifter_t = 0.0
    
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

        # Update boundary conditions
        self.bc_kernel.update_bc_values(self.gpu_stream, self.t)
        self.bc_kernel.boundaryCondition(self.gpu_stream, \
                                             self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0)
            
    def _set_interior_domain_from_sponge_cells(self):
        """
        Use possible existing sponge cells to correctly set the 
        variable self.interior_domain_incides
        """
        if (self.boundary_conditions.isSponge()):
            assert(False), 'This function is deprecated - sponge cells should now be considered part of the interior domain'
    