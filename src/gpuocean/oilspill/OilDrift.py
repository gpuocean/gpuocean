"""
This software is part of GPU Ocean. 

Copyright (C) 2024 SINTEF Digital
Copyright (C) 2024 SINTEF Ocean

Drifter class that implements a simplified oil spill model.

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


import os
import numpy as np
import warnings
import sys


import pycuda.driver as cuda
import pycuda.gpuarray

from gpuocean.utils import Common, RandomNumbers, WindStress

class OilDrift:
    """
    As simple as possible class for drifters using PyCUDA and GPU Ocean.
    
    To support interoperability, the API should eventually be similar to the GPUDrifterCollection class.
    At the same time, it should be mentioned that there are a lot of functions there that are never used...
    """

    def __init__(self, gpu_ctx, drifter_positions, 
                 initial_droplet_diameter =  5e-4, 
                 oil_density=992, water_density=1025,
                 oil_viscosity=1.51, water_kinematic_viscosity=1.358e-6, oil_water_ift=0.013, oil_film_thickness=1e-4,
                 g=9.81,
                 horizontal_diffusivity=1.0, vertical_diffusivity=1.0, 
                 wind=WindStress.WindStress(), windage = 0.03,
                 use_relative_positions = True, seed = None,
                 enable_entrainment = True,
                 release_times = None, 
                 block_width=128, rng_block_height=32):


        # Define all member variables that will point to GPU memory in advance
        self.rng = None
        self.relative_positions_device = None
        self.reference_positions_device = None
        self.droplet_diameters_device = None
        self.wind_x_current_arr = None
        self.wind_y_current_arr = None
        self.wind_x_next_arr = None
        self.wind_y_next_arr = None

        assert(drifter_positions.shape[1] == 3), "expecting drifter_positions to be of shape (N, 3)"
        self.num_drifters = drifter_positions.shape[0]

        self.horizontal_diffusivity = np.float32(horizontal_diffusivity)
        self.vertical_diffusivity = np.float32(vertical_diffusivity)

        # GPU stuff
        self.gpu_ctx = gpu_ctx
        self.gpu_stream = cuda.Stream() # Different streams can in principle be run in parallel

        # Define how we want to distribute the work on the GPU
        # Here, we assume that each thread is responsible for moving one drifter
        # Local size refers to the number of threads in each block (organized in 3D)
        # global size refers to the number of blocks that will be run on the GPU (can be organized in 2D or 3D)
        self.block_width = block_width 
        self.block_height = 1

        self.local_size = (self.block_width, self.block_height, 1)
        self.global_size = (int(np.ceil((self.num_drifters + 1)/float(self.block_width))), 1)
        
        # Initialize arrays for Lagrangian positions 
        # A particle's position is made up of reference_position + relative_position
        # where the reference_position is kept constant and the relative_position is the dynamic part.
        # Relative positions are only used horizontally, the vertical positions
        # are always absolute.
        self.use_relative_positions = use_relative_positions
        self.reference_positions = np.zeros((self.num_drifters, 2), dtype=np.float32)
        relative_positions = drifter_positions.copy()
        if self.use_relative_positions:
            self.reference_positions[:, :] = relative_positions[:, :2]
            relative_positions[:, :2] = 0.0

        # Organize release times
        if release_times is None:
            self.release_times = np.array([-100000, np.nan])
            self.num_released_drifters = np.array([self.num_drifters])
            self.next_release_index = None
            self.num_active_drifters = np.int32(self.num_drifters)
        else:
            assert(len(release_times) == self.num_drifters), "expecting "+str(self.num_drifters)+" release times (one per drifter), but got "+str(len(release_times))
            self.release_times, self.num_released_drifters = np.unique(release_times, return_counts=True)
            self.release_times = np.concatenate((self.release_times, np.array([np.nan])))
            self.num_released_drifters = np.cumsum(self.num_released_drifters)
            self.next_release_index = 0
            self.num_active_drifters = np.int32(0)
        

        # Allocate GPU memory and intialize using the 2D Array utility function, which is a wrapper around pycuda.gpuarray
        # Data size parameters are given by the signature (_, nx, ny, ghost_cells_x, ghost_cells_y, _)
        
        self.relative_positions_device = Common.CUDAArray2D(self.gpu_stream, 
                                                           3, self.num_drifters, 0, 0,
                                                           relative_positions)
        self.reference_positions_device = Common.CUDAArray2D(self.gpu_stream, 
                                                            2, self.num_drifters, 0, 0,
                                                            self.reference_positions)

        # Initialize random number generators - require one seed per drifter
        self.rng = RandomNumbers.RandomNumbers(gpu_ctx, self.gpu_stream,
                                               1, self.num_drifters, 
                                               use_lcg=True, seed=seed,
                                               block_width=1, block_height=rng_block_height)
        

        #self.droplet_diameter_data = pycuda.gpuarray.to_gpu_async((np.ones(self.num_drifters, dtype=np.float32) * initial_droplet_diameter), stream=self.gpu_stream)
        self.droplet_diameters_device = Common.CUDAArray2D(self.gpu_stream, 1, self.num_drifters, 0, 0, np.ones((self.num_drifters,1)) * initial_droplet_diameter)
        self.oil_density = np.float32(oil_density)
        self.oil_viscosity = np.float32(oil_viscosity)
        self.water_density = np.float32(water_density)
        self.water_viscosity = np.float32(water_kinematic_viscosity)
        self.g = np.float32(g)
        self.oil_film_thickness = np.float32(oil_film_thickness)
        self.oil_water_ift = np.float32(oil_water_ift)

        # Allocate memory for two wind fields and upload the first two
        self.wind = wind
        t = 0  # TODO: check if this is correct
        t_max_index = len(self.wind.t)-1
        t0_index = max(0, np.searchsorted(self.wind.t, t)-1)
        t1_index = min(t_max_index, np.searchsorted(self.wind.t,t))
        self.wind_x_current_arr = Common.CUDAArray2D(self.gpu_stream,
                                self.wind.wind_u[t0_index].shape[1], self.wind.wind_u[t0_index].shape[0], 0, 0,
                                self.wind.wind_u[t0_index])
        self.wind_y_current_arr = Common.CUDAArray2D(self.gpu_stream,
                                self.wind.wind_v[t0_index].shape[1], self.wind.wind_v[t0_index].shape[0], 0, 0,
                                self.wind.wind_v[t0_index])
        self.wind_x_next_arr = Common.CUDAArray2D(self.gpu_stream,
                                self.wind.wind_u[t1_index].shape[1], self.wind.wind_u[t1_index].shape[0], 0, 0,
                                self.wind.wind_u[t1_index])
        self.wind_y_next_arr = Common.CUDAArray2D(self.gpu_stream,
                                self.wind.wind_v[t1_index].shape[1], self.wind.wind_v[t1_index].shape[0], 0, 0,
                                self.wind.wind_v[t1_index])

        self.wind_timestamps = {}
        self.windage = np.float32(windage)

        # To do that, we need to provide the absolute path along with the corresponding flag
        # self.kernel_filename = os.path.join("..", "gpu_kernels", "oil_spill_drift.cu")
        # self.kernel_filename = os.path.abspath(self.kernel_filename)
        self.drift_kernels = gpu_ctx.get_kernel("oil_spill_drift.cu", \
                                                defines={'block_width': self.block_width, 'block_height': self.block_height,
                                                         'ENABLE_ENTRAINMENT': int(enable_entrainment),
                                                         'WIND_X_NX': int(self.wind.wind_u[0].shape[1]),
                                                         'WIND_X_NY': int(self.wind.wind_u[0].shape[0]),
                                                         'WIND_Y_NX': int(self.wind.wind_v[0].shape[1]),
                                                         'WIND_Y_NY': int(self.wind.wind_v[0].shape[0])
                                                       })
        
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.superSimpleDriftKernel = self.drift_kernels.get_function("oilSpillDrift")
        self.superSimpleDriftKernel.prepare("iifffPiPiPiPiiiPiPiPiffPifffffffPPPPfff")
        # The input string to prepare defines the data type for each input parameter in order
        # Example: prepare("ifPi") means that the kernel parameters have type signature (int, float, pointer, int)
        


    # Destructor and memory deallocation
    def __del__(self):
        self.cleanUp()
     
    def cleanUp(self):
        if self.rng is not None:
            self.rng.cleanUp()
        if self.relative_positions_device is not None:
            self.relative_positions_device.release()
        if self.reference_positions_device is not None:
            self.reference_positions_device.release()
        if self.droplet_diameters_device is not None:
            self.droplet_diameters_device.release()
        if self.wind_x_current_arr is not None:
            self.wind_x_current_arr.release()
        if self.wind_y_current_arr is not None:
            self.wind_y_current_arr.release()
        if self.wind_x_next_arr is not None:
            self.wind_x_next_arr.release()
        if self.wind_y_next_arr is not None:
            self.wind_y_next_arr.release()
        self.gpu_ctx = None
        
    def setGPUStream(self, gpu_stream):
        self.gpu_stream = gpu_stream

    def getDrifterPositions(self, only_active=True):
        # Download the positions from the gpu (device) to the host (cpu)
        drifter_positions = self.relative_positions_device.download(self.gpu_stream)
        drifter_positions[:, :2] += self.reference_positions

        if only_active:
            return drifter_positions[:self.num_active_drifters, :]
        
        return drifter_positions

    def setDrifterPositions(self, drifter_positions, only_active=True):
        # Upload new positions from the cpu (host) to the device (gpu)
        if only_active and (self.num_active_drifters != self.num_drifters):
            assert(drifter_positions.shape == (self.num_active_drifters, 3)), "expecting drifter_positions of shape "+str((self.num_active_drifters, 3))+" but got "+str(drifter_positions.shape)
        
            all_drifter_positions = self.getDrifterPositions(only_active = False)
            all_drifter_positions[:self.num_active_drifters, :] = drifter_positions[:, :]
            self.setDrifterPositions(all_drifter_positions, only_active=False)
            return

        assert(drifter_positions.shape == (self.num_drifters, 3)), "expecting drifter_positions of shape "+str((self.num_drifters, 3))+" but got "+str(drifter_positions.shape)    
        relative_positions = drifter_positions.copy()
        if self.use_relative_positions:
            self.reference_positions[:, :] = relative_positions[:, :2]
            self.reference_positions_device.upload(self.gpu_stream, self.reference_positions)
            relative_positions[:, :2] = 0.0
        self.relative_positions_device.upload(self.gpu_stream, relative_positions)

    def getDropletDiameters(self, only_active=True):
        # Download the positions from the gpu (device) to the host (cpu)
        if only_active:
            return self.droplet_diameters_device.download(self.gpu_stream)[:self.num_active_drifters]

        return self.droplet_diameters_device.download(self.gpu_stream)

    def setDropletDiameters(self, droplet_diameters, only_active=False):
        # Upload new positions from the cpu (host) to the device (gpu)
        if only_active and (self.num_active_drifters != self.num_drifters):
            assert(droplet_diameters.shape == (self.num_active_drifters, 1)), "expecting droplet_diameters of shape "+str((self.num_active_drifters, 1))+" but got "+str(droplet_diameters.shape)
            all_droplet_diameters = self.getDropletDiameters(only_active=False)
            all_droplet_diameters[:self.num_active_drifters, :] = droplet_diameters[:,:]
            self.setDropletDiameters(all_droplet_diameters, only_active=False)
            return

        assert(droplet_diameters.shape == (self.num_drifters, 1)), "expecting droplet_diameters of shape "+str((self.num_drifters, 1))+" but got "+str(droplet_diameters.shape)
        self.droplet_diameters_device.upload(self.gpu_stream, droplet_diameters)

    def driftFromSim(self, sim, dt):
        self.drift(sim, dt)

    def drift(self, sim, dt):
        # Call the kernel to simulate the drifters for dt seconds using the ocean state available in the sim
        # Note: Only pointers to GPU memory can be given to the cuda kernel function

        # Disclaimer:The gpu arrays for the simulator has does not have the correct names for historical reasons...
        # The values for eta are called h
        # The values for Hm are called Bm
        # Sorry...
        # Furthermore, the simulator has two buffers for each variable (e.g., hu0 and hu1), 
        # where the *0 is the one you should use, and *1 is used as a temporary storage during two-stage Runge Kutta for the finite volume method

        wind_interpolation_t = np.float32(self.update_wind(self.drift_kernels, sim.drifter_t))
        self._computeNumActiveDrifters(sim.drifter_t)

        # The first three parameters to the kernel is always the subdivision of work (globale size and local size), and the gpu stream that will execute the kernel
        self.superSimpleDriftKernel.prepared_async_call(self.global_size, self.local_size, self.gpu_stream,
                                               sim.nx, sim.ny, sim.dx, sim.dy, np.float32(dt),
                                               sim.gpu_data.h0.data.gpudata, sim.gpu_data.h0.pitch,
                                               sim.gpu_data.hu0.data.gpudata, sim.gpu_data.hu0.pitch,
                                               sim.gpu_data.hv0.data.gpudata, sim.gpu_data.hv0.pitch,
                                               sim.bathymetry.Bm.data.gpudata, sim.bathymetry.Bm.pitch,
                                               np.int32(self.num_drifters),
                                               self.num_active_drifters,
                                               self.relative_positions_device.data.gpudata,
                                               self.relative_positions_device.pitch,
                                               self.reference_positions_device.data.gpudata,
                                               self.reference_positions_device.pitch,
                                               self.rng.seed.data.gpudata, self.rng.seed.pitch,
                                               self.horizontal_diffusivity, self.vertical_diffusivity,
                                               self.droplet_diameters_device.data.gpudata, self.droplet_diameters_device.pitch,
                                               self.oil_density, self.water_density,
                                               self.oil_viscosity, self.water_viscosity,
                                               self.oil_film_thickness, self.oil_water_ift,
                                               self.g,
                                               self.wind_x_current_arr.data.gpudata, self.wind_y_current_arr.data.gpudata,
                                               self.wind_x_next_arr.data.gpudata, self.wind_y_next_arr.data.gpudata,
                                               wind_interpolation_t,
                                               self.windage, sim.drifter_t)
        
    def _computeNumActiveDrifters(self, t):
        if self.next_release_index is not None:
            while t >= self.release_times[self.next_release_index]:
                self.num_active_drifters = np.int32(self.num_released_drifters[self.next_release_index])
                self.next_release_index = self.next_release_index + 1
                # print("updated num_active_drifters at t="+str(t)+" to "+str(self.num_active_drifters))
                

    def is_submerged(self):
        # Return True if the oil drifter is submerged
        return self.getDrifterPositions()[:,2] < 0
    
    def is_stranded(self):
        # Return True if the oil drifter is stranded
        # If stranded, the particle's depth represents the time of stranding
        return self.getDrifterPositions()[:,2] > 999
    
    def get_stranded_particles(self):
        # Return stranded particles as (locations, stranded time)
        all_positions = self.getDrifterPositions()
        stranded_positions = all_positions[all_positions[:, 2] > 999, :]
        stranded_times = stranded_positions[:, 2] - 1000
        return stranded_positions[:, :2], stranded_times
            

    def sortParticlesFromSim(self, sim):
        self.sortParticles(sim.nx, sim.ny, sim.dx, sim.dy)

    def sortParticles(self, nx, ny, dx, dy):
        positions = self.getDrifterPositions(only_active=True)
        
        # get cell ids:
        #def get_cell_ids(positions):
        cell_id_x = np.floor(positions[:,0]/dx).astype(int)
        cell_id_y = np.floor(positions[:,1]/dy).astype(int)
        cell_id = cell_id_y*nx + cell_id_x

        sorted_indices = cell_id.argsort()

        positions = positions[sorted_indices]
        self.setDrifterPositions(positions, only_active=True)

        droplet_diameters = self.getDropletDiameters(only_active=True)
        droplet_diameters = droplet_diameters[sorted_indices]
        self.setDropletDiameters(droplet_diameters, only_active=True)


    def update_wind(self, kernel_module, t):
        #Key used to access the hashmaps
        key = str(kernel_module)

        #Compute new t0 and t1
        t_max_index = len(self.wind.t)-1
        t0_index = max(0, np.searchsorted(self.wind.t, t)-1)
        t1_index = min(t_max_index, np.searchsorted(self.wind.t,t))
        new_t0 = self.wind.t[t0_index]
        new_t1 = self.wind.t[t1_index]
    
        #Find the old (and update)
        old_t0 = None
        old_t1 = None
        if (key in self.wind_timestamps):
            old_t0 = self.wind_timestamps[key][0]
            old_t1 = self.wind_timestamps[key][1]
        self.wind_timestamps[key] = [new_t0, new_t1]

        #If time interval has changed, upload new data
        if (new_t0 != old_t0):
            self.gpu_stream.synchronize()
            self.gpu_ctx.synchronize()
            self.wind_x_current_arr.upload(self.gpu_stream, self.wind.wind_u[t0_index])
            self.wind_y_current_arr.upload(self.gpu_stream, self.wind.wind_v[t0_index])
            self.gpu_ctx.synchronize()

        if (new_t1 != old_t1):
            self.gpu_stream.synchronize()
            self.gpu_ctx.synchronize()
            self.wind_x_next_arr.upload(self.gpu_stream, self.wind.wind_u[t1_index])
            self.wind_y_next_arr.upload(self.gpu_stream, self.wind.wind_v[t1_index])
            self.gpu_ctx.synchronize()

        # Compute the wind_stress_t linear interpolation coefficient
        wind_t = 0.0
        elapsed_since_t0 = (t-new_t0)
        time_interval = max(1.0e-10, (new_t1-new_t0))
        wind_t = max(0.0, min(1.0, elapsed_since_t0 / time_interval))

        return wind_t