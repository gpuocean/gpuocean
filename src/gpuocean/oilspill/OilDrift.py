
import os
import numpy as np
import warnings
import sys


import pycuda.driver as cuda

from gpuocean.utils import Common, RandomNumbers, WindStress

class OilDrift:
    """
    As simple as possible class for drifters using PyCUDA and GPU Ocean.
    
    To support interoperability, the API should eventually be similar to the GPUDrifterCollection class.
    At the same time, it should be mentioned that there are a lot of functions there that are never used...
    """

    def __init__(self, gpu_ctx, drifter_positions, droplet_diameter, oil_density=0.8, water_density=1.025,
                 water_viscosity=1.358e-6, g=9.81,
                 horizontal_diffusivity=1.0, vertical_diffusivity=1.0, 
                 wind=WindStress.WindStress(), windage = 0.03,
                 block_width=32, rng_block_height=32):

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
        
        # Allocate GPU memory and intialize using the 2D Array utility function, which is a wrapper around pycuda.gpuarray
        # Data size parameters are given by the signature (_, nx, ny, ghost_cells_x, ghost_cells_y, _)
        self.drifter_positions_device = Common.CUDAArray2D(self.gpu_stream, 
                                                           3, self.num_drifters, 0, 0,
                                                           drifter_positions)
        # Initialize random number generators - require one seed per drifter
        self.rng = RandomNumbers.RandomNumbers(gpu_ctx, self.gpu_stream,
                                               1, self.num_drifters, 
                                               use_lcg=True,
                                               block_width=4, block_height=rng_block_height)
        


        self.droplet_diameter = np.float32(droplet_diameter)
        self.oil_density = np.float32(oil_density)
        self.water_density = np.float32(water_density)
        self.water_viscosity = np.float32(water_viscosity)
        self.g = np.float32(g)

        # Compile cuda file found in this repository
        # To do that, we need to provide the absolute path along with the corresponding flag
        self.kernel_filename = os.path.join("..", "gpu_kernels", "super_simple_drift_kernel.cu")
        self.kernel_filename = os.path.abspath(self.kernel_filename)
        self.drift_kernels = gpu_ctx.get_kernel(self.kernel_filename, \
                                                defines={'block_width': self.block_width, 'block_height': self.block_height
                                                       },
                                                is_abs_path=True)
        
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.superSimpleDriftKernel = self.drift_kernels.get_function("superSimpleDrift")
        self.superSimpleDriftKernel.prepare("iifffPiPiPiPiiPiPifffffffPiPif")
        # The input string to prepare defines the data type for each input parameter in order
        # Example: prepare("ifPi") means that the kernel parameters have type signature (int, float, pointer, int)

        # Wind:
        # TODO: Wind should be read from the ocean simulator object, but we are currently changing how wind is 
        # stored on the GPU. Therefore, we use this temporary solution with given restrictions... 
        if(len(wind.t) > 1):
            warnings.warn("Currently only supporting wind that is constant in time.\nUsing wind from the first timestep only")
       
        self.wind_u = Common.CUDAArray2D(self.gpu_stream, wind.wind_u[0].shape[1], wind.wind_u[0].shape[0], 0, 0, wind.wind_u[0])
        self.wind_v = Common.CUDAArray2D(self.gpu_stream, wind.wind_v[0].shape[1], wind.wind_v[0].shape[0], 0, 0, wind.wind_v[0])
        self.windage = np.float32(windage)

    # Destructor and memory deallocation
    def __del__(self):
        self.cleanUp()
     
    def cleanUp(self):
        if self.rng is not None:
            self.rng.cleanUp()
        if self.drifter_positions_device is not None:
            self.drifter_positions_device.release()
        self.gpu_ctx = None
        


    def getDrifterPositions(self):
        # Download the positions from the gpu (device) to the host (cpu)
        return self.drifter_positions_device.download(self.gpu_stream)

    def setDrifterPositions(self, drifter_positions):
        # Upload new positions from the cpu (host) to the device (gpu)
        assert(drifter_positions.shape == (self.num_drifters, 3)), "expecting drifter_positions of shape "+str((self.num_drifters, 3))+" but got "+str(drifter_positions.shape)
        self.drifter_positions_device.upload(self.gpu_stream, drifter_positions)

    def drift(self, sim, dt):
        # Call the kernel to simulate the drifters for dt seconds using the ocean state available in the sim
        # Note: Only pointers to GPU memory can be given to the cuda kernel function

        # Disclaimer:The gpu arrays for the simulator has does not have the correct names for historical reasons...
        # The values for eta are called h
        # The values for Hm are called Bm
        # Sorry...
        # Furthermore, the simulator has two buffers for each variable (e.g., hu0 and hu1), 
        # where the *0 is the one you should use, and *1 is used as a temporary storage during two-stage Runge Kutta for the finite volume method

        # TODO: Fix wind check - it is currently a temporary solution awaiting new pull request to GPU Ocean
        self._check_wind(sim)

        # The first three parameters to the kernel is always the subdivision of work (globale size and local size), and the gpu stream that will execute the kernel
        self.superSimpleDriftKernel.prepared_async_call(self.global_size, self.local_size, self.gpu_stream,
                                               sim.nx, sim.ny, sim.dx, sim.dy, np.float32(dt),
                                               sim.gpu_data.h0.data.gpudata, sim.gpu_data.h0.pitch,
                                               sim.gpu_data.hu0.data.gpudata, sim.gpu_data.hu0.pitch,
                                               sim.gpu_data.hv0.data.gpudata, sim.gpu_data.hv0.pitch,
                                               sim.bathymetry.Bm.data.gpudata, sim.bathymetry.Bm.pitch,
                                               np.int32(self.num_drifters),
                                               self.drifter_positions_device.data.gpudata,
                                               self.drifter_positions_device.pitch,
                                               self.rng.seed.data.gpudata, self.rng.seed.pitch,
                                               self.horizontal_diffusivity, self.vertical_diffusivity,
                                               self.droplet_diameter, self.oil_density, self.water_density,
                                               self.water_viscosity, self.g,
                                               self.wind_u.data.gpudata, self.wind_u.pitch,
                                               self.wind_v.data.gpudata, self.wind_v.pitch,
                                               self.windage)

    def is_submerged(self):
        # Return True if the oil drifter is submerged
        return self.getDrifterPositions()[:,2] < 0
    
    def is_stranded(self):
        # Return True if the oil drifter is stranded
        return self.getDrifterPositions()[:,2] == 999
            

    def _check_wind(self, sim):
        if (self.wind_u.nx_halo == 1):
            # Assuming then that the wind object is the default one with no wind.
            # Extend the zero-wind to the same grid resolution as the simulator
            wind_u = np.zeros((sim.ny+4, sim.nx+4), dtype=np.float32, order='C')
            wind_v = np.zeros((sim.ny+4, sim.nx+4), dtype=np.float32, order='C')
            
            self.wind_u = Common.CUDAArray2D(self.gpu_stream, sim.nx, sim.ny, 2, 2, wind_u)
            self.wind_v = Common.CUDAArray2D(self.gpu_stream, sim.nx, sim.ny, 2, 2, wind_v)

        assert(self.wind_u.nx_halo == sim.gpu_data.h0.nx_halo)
        assert(self.wind_u.ny_halo == sim.gpu_data.h0.ny_halo)
        assert(self.wind_v.nx_halo == sim.gpu_data.h0.nx_halo)
        assert(self.wind_v.ny_halo == sim.gpu_data.h0.ny_halo)
        
