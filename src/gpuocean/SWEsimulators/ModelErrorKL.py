# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018, 2019, 2023  SINTEF Digital
Copyright (C) 2018, 2019 Norwegian Meteorological Institute

This python class produces Karhunen-Loeve type random perturbations that 
are to be added to the ocean state fields in order to generate model error.

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


from matplotlib import pyplot as plt
import numpy as np

import pycuda.gpuarray 
import pycuda.driver as cuda
from pycuda.curandom import XORWOWRandomNumberGenerator

import gc

from gpuocean.utils import Common, RandomNumbers, config

class ModelErrorKL(object):
    """
    Generating random perturbations for a ocean state.
   
    Perturbation for the surface field, dEta, is produced with a covariance structure 
    according to a Karhunen-Loeve expansion, while dHu and dHv are found by the
    geostrophic balance to avoid shock solutions.
    """
    
    def __init__(self, gpu_ctx, gpu_stream,
                 nx, ny, dx, dy,
                 boundary_conditions,
                 kl_decay=1.2, kl_scaling=1.0,
                 include_cos=True, include_sin=True,
                 basis_x_start = 1, basis_y_start = 1,
                 basis_x_end = 10, basis_y_end = 10,
                 use_lcg=False, xorwow_seed = None, np_seed = None,
                 angle=np.array([[0]], dtype=np.float32),
                 coriolis_f=np.array([[0]], dtype=np.float32),
                 block_width=16, block_height=16):
        """
        Initiates a class that generates geostrophically balanced model errors of
        the ocean state using Karhunen Loeve perturbations of eta.
        (nx, ny): number of internal grid cells in the domain
        (dx, dy): size of each grid cell
        boundary_conditions: boundary conditions
        amplitude: amplitude parameter for the perturbation, default: dx*1e-5
        include_cos: boolean - include cosine basis functions
        include_sin: boolean - include sine basis functions
        basis_x_start: Number of half periods in the first x-basis
        basis_y_start: Number of half periods in the first y-basis
        basis_x_end: Number of half periods in the last x-basis
        basis_y_end: Number of half periods in the last y-basis
        use_lcg: LCG is a linear algorithm for generating a serie of pseudo-random numbers
        xorwow_seed: Seed for the pycuda random number generator
        angle: Angle of rotation from North to y-axis as a texture (cuda.Array) or numpy array
        coriolis_f: Coriolis parameter as a texture (cuda.Array) or numpy array
        (block_width, block_height): The size of each GPU block
        """

        self.use_lcg = use_lcg

        # Set numpy random state
        self.random_state = np.random.RandomState()
        
        # Make sure that all variables initialized within ifs are defined
        self.random_numbers = None
        self.host_seed = None

        self.gpu_ctx = gpu_ctx
        self.gpu_stream = gpu_stream
        
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.boundary_conditions = boundary_conditions

        # KL parameters
        self.kl_decay       = np.float32(kl_decay)
        self.kl_scaling     = np.float32(kl_scaling)
        self.include_cos    = np.int32(include_cos)
        self.include_sin    = np.int32(include_sin)
        self.basis_x_start  = np.int32(basis_x_start)
        self.basis_y_start  = np.int32(basis_y_start)
        self.basis_x_end    = np.int32(basis_x_end)
        self.basis_y_end    = np.int32(basis_y_end)

        self.np_rng = np.random.default_rng(seed=np_seed)
        self.roll_x_sin = 0.0
        self.roll_y_sin = 0.0
        self.roll_x_cos = 0.0
        self.roll_y_cos = 0.0


        self.N_basis_x = np.int32(self.basis_x_end - self.basis_x_start + 1)
        self.N_basis_y = np.int32(self.basis_y_end - self.basis_y_start + 1)

        #self.periodicNorthSouth = np.int32(boundaryConditions.isPeriodicNorthSouth())
        #self.periodicEastWest = np.int32(boundaryConditions.isPeriodicEastWest())
        
        # Size of random array
        # We sample one random number per KL basis field (both sine and cosine) 
        self.rand_nx = np.int32(self.N_basis_x)
        self.rand_ny = np.int32(self.N_basis_y*2)

        self.rng = RandomNumbers.RandomNumbers(gpu_ctx, self.gpu_stream, 
                                               self.rand_nx, self.rand_ny, 
                                               use_lcg=self.use_lcg, xorwow_seed=xorwow_seed,
                                               block_width=block_width, block_height=block_height)
        
        # Since normal distributed numbers are generated in pairs, we need to store half the number of
        # of seed values compared to the number of random numbers.
        self.seed_ny = np.int32(self.rand_ny)
        self.seed_nx = np.int32(np.ceil(self.rand_nx/2))

        # Generate seed:
        self.floatMax = self.rng.floatMax
        if self.use_lcg:
            self.host_sedd = self.rng.host_seed
        
        # Variable for storing the basis functions used for sampling on the CPU
        # Should only be initialized if it is used.
        self.KL_basis_fields_sin = None
        self.KL_basis_fields_cos = None

        # Allocate memory for random numbers
        self.random_numbers_host = np.zeros((self.rand_ny, self.rand_nx), dtype=np.float32, order='C')
        self.random_numbers = Common.CUDAArray2D(self.gpu_stream, self.rand_nx, self.rand_ny, 0, 0, self.random_numbers_host)
               
        # Generate kernels
        self.kernels = gpu_ctx.get_kernel("ocean_noise.cu", \
                                          defines={'block_width': block_width, 'block_height': block_height,
                                                    'kl_rand_nx': self.rand_nx, 'kl_rand_ny': self.rand_ny},
                                          compile_args={
                                              'options': ["--use_fast_math",
                                                          "--maxrregcount=32"]
                                          })
        
        
        # Get CUDA functions and define data types for prepared_{async_}call()
        # Generate kernels
        
        self.klSamplingKernelEta = self.kernels.get_function("kl_sample_eta")
        self.klSamplingKernelEta.prepare("iiiiiiiiffffffPiPi")
        self.klSamplingKernel = self.kernels.get_function("kl_sample_ocean_state")
        self.klSamplingKernel.prepare("iifffffiiiiiiffffffPiPiPiPiPif")

        #Compute kernel launch parameters
        self.local_size = (block_width, block_height, 1)
        
        # Launch one thread for each seed, which in turns generates two iid N(0,1)
        self.global_size_random_numbers = ( \
                       int(np.ceil(self.seed_nx / float(self.local_size[0]))), \
                       int(np.ceil(self.seed_ny / float(self.local_size[1]))) \
                     ) 
        
        # Launch one thread per grid cell, but also for one ghost cell per block,
        # cell in order to find geostrophic balance from the result
        self.global_size_KL = ( \
                     int(np.ceil( self.nx/float(self.local_size[0]-2))), \
                     int(np.ceil( self.ny/float(self.local_size[1]-2))) \
                    )
        
        # Launch one thread per grid cell 
        self.global_size_KL_eta = ( \
                     int(np.ceil( self.nx/float(self.local_size[0]))), \
                     int(np.ceil( self.ny/float(self.local_size[1]))) \
                    )
        
        # Texture for coriolis field
        self.coriolis_texref = self.kernels.get_texref("coriolis_f_tex")        
        if isinstance(coriolis_f, cuda.Array):
            # coriolis_f is already a texture, so we just set the reference
            self.coriolis_texref.set_array(coriolis_f)
        else:
            #Upload data to GPU and bind to texture reference
            self.coriolis_texref.set_array(cuda.np_to_array(np.ascontiguousarray(coriolis_f, dtype=np.float32), order="C"))
          
        # Set texture parameters
        self.coriolis_texref.set_filter_mode(cuda.filter_mode.LINEAR) #bilinear interpolation
        self.coriolis_texref.set_address_mode(0, cuda.address_mode.CLAMP) #no indexing outside domain
        self.coriolis_texref.set_address_mode(1, cuda.address_mode.CLAMP)
        self.coriolis_texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES) #Use [0, 1] indexing
        # FIXME! Allow different versions of coriolis, similar to CDKLM
        
        
        
        # Texture for angle towards north
        self.angle_texref = self.kernels.get_texref("angle_tex")        
        if isinstance(angle, cuda.Array):
            # angle is already a texture, so we just set the reference
            self.angle_texref.set_array(angle)
        else:
            #Upload data to GPU and bind to texture reference
            self.angle_texref.set_array(cuda.np_to_array(np.ascontiguousarray(angle, dtype=np.float32), order="C"))
          
        # Set texture parameters
        self.angle_texref.set_filter_mode(cuda.filter_mode.LINEAR) #bilinear interpolation
        self.angle_texref.set_address_mode(0, cuda.address_mode.CLAMP) #no indexing outside domain
        self.angle_texref.set_address_mode(1, cuda.address_mode.CLAMP)
        self.angle_texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES) #Use [0, 1] indexing
        
        
    def __del__(self):
        self.cleanUp()
     
    def cleanUp(self, do_gc=True):
        if self.rng is not None:
            self.rng.cleanUp()
        if self.random_numbers is not None:
            self.random_numbers.release()
        self.gpu_ctx = None
        self.boundary_conditions = None
        if do_gc:
            gc.collect()
        
    @classmethod
    def fromsim(cls, sim, 
                kl_decay=1.2, kl_scaling=1.0,
                include_cos=True, include_sin=True,
                basis_x_start = 1, basis_y_start = 1,
                basis_x_end = 10, basis_y_end = 10,
                use_lcg=False, xorwow_seed = None, np_seed=None,
                block_width=16, block_height=16):
        return cls(sim.gpu_ctx, sim.gpu_stream,
                   sim.nx, sim.ny, sim.dx, sim.dy,
                   sim.boundary_conditions, 
                   kl_decay=kl_decay, kl_scaling=kl_scaling,
                   include_cos=include_cos, include_sin=include_sin,
                   basis_x_start=basis_x_start, basis_y_start=basis_y_start,
                   basis_x_end=basis_x_end, basis_y_end=basis_y_end,
                   angle=sim.angle_texref.get_array(),
                   coriolis_f=sim.coriolis_texref.get_array(),
                   use_lcg=use_lcg, xorwow_seed=xorwow_seed, np_seed=np_seed,
                   block_width=block_width, block_height=block_height)

    def getSeed(self):
        return self.rng.getSeed()
    
    def resetSeed(self):
        self.rng.resetSeed()

    def getRandomNumbers(self):
        return self.random_numbers.download(self.gpu_stream)
    
    def generateNormalDistribution(self):
        self.rng.generateNormalDistribution(self.random_numbers)
    
    def setRandomNumbers(self, random_numbers):
        assert(random_numbers.shape == self.random_numbers_host.shape), "Wrong shape of random numbers: "+str(random_numbers.shape) + " vs " + str(self.random_numbers_host.shape)

        self.random_numbers_host[:, :] = random_numbers[:,:]
        self.random_numbers.upload(self.gpu_stream, self.random_numbers_host)


    def _setRollers(self, roll_x_sin, roll_y_sin, roll_x_cos, roll_y_cos):
        """
        Set roller parameters with some checking that the values are appropriate
        """
        if roll_x_sin is None:
            roll_x_sin = self.np_rng.uniform()
        if roll_y_sin is None:
            roll_y_sin = self.np_rng.uniform()
        if roll_x_cos is None:
            roll_x_cos = self.np_rng.uniform()
        if roll_y_cos is None:
            roll_y_cos = self.np_rng.uniform()
        def _check_roller(roller, name):
            assert(isinstance(roller, (float, int, np.float32)) and roller >= 0 and roller <= 1), "illegal type/value of " + name +": " + str(roller)
        _check_roller(roll_x_sin, "roll_x_sin")
        _check_roller(roll_y_sin, "roll_y_sin")
        _check_roller(roll_x_cos, "roll_x_cos")
        _check_roller(roll_y_cos, "roll_y_cos")
        
        self.roll_x_sin = roll_x_sin
        self.roll_y_sin = roll_y_sin
        self.roll_x_cos = roll_x_cos
        self.roll_y_cos = roll_y_cos        

    def perturbEtaSim(self, sim, perturbation_scale=1.0, update_random_field=True, 
                   random_numbers=None, 
                   roll_x_sin=None, roll_y_sin=None,
                   roll_x_cos=None, roll_y_cos=None,
                   stream=None):
        """
        Generating a perturbed eta field and adding it to sim's eta variable 
        """
        self.perturbEta(sim.gpu_data.h0, 
                        update_random_field=update_random_field,
                        random_numbers=random_numbers, 
                        roll_x_sin=roll_x_sin, roll_y_sin=roll_y_sin,
                        roll_x_cos=roll_x_cos, roll_y_cos=roll_y_cos,
                        perturbation_scale=perturbation_scale,
                        stream=stream)
                               
    def perturbEta(self, eta, 
                   update_random_field=True, 
                   random_numbers=None, 
                   roll_x_sin=None, roll_y_sin=None,
                   roll_x_cos=None, roll_y_cos=None,
                   perturbation_scale = 1.0,
                   stream=None):
        """
        Sample random perturbation using the KL basis functions and add it to eta only
        eta: surface deviation - CUDAArray2D object.
        """
        
        if stream is None:
            stream = self.gpu_stream
        
        if random_numbers is not None:
            assert(isinstance(random_numbers, (Common.CUDAArray2D, np.ndarray))), "random numbers should be a CUDA 2D array or a numpy array, but is " + str(type(random_numbers))
            if isinstance(random_numbers, Common.CUDAArray2D): 
                tmp_rns = random_numbers.download(stream)
                self.random_numbers.upload(stream, tmp_rns)
            else:
                self.random_numbers.upload(stream, random_numbers)
        elif update_random_field:
            # Need to update the random field, requiering a global sync
            self.generateNormalDistribution()
        
        self._setRollers(roll_x_sin, roll_y_sin, roll_x_cos, roll_y_cos)     

        self.klSamplingKernelEta.prepared_async_call(self.global_size_KL_eta, self.local_size, stream,
                                            self.nx, self.ny,
                                            self.basis_x_start, self.basis_x_end,
                                            self.basis_y_start, self.basis_y_end,
                                            self.include_cos, self.include_sin,
                                            self.kl_decay, 
                                            np.float32(perturbation_scale * self.kl_scaling),
                                            np.float32(self.roll_x_sin), np.float32(self.roll_y_sin), 
                                            np.float32(self.roll_x_cos), np.float32(self.roll_y_cos), 
                                            
                                            self.random_numbers.data.gpudata, self.random_numbers.pitch,
                                            eta.data.gpudata, eta.pitch)
        
    def perturbSim(self, sim, perturbation_scale=1.0, update_random_field=True, 
                   random_numbers=None, 
                   roll_x_sin=None, roll_y_sin=None,
                   roll_x_cos=None, roll_y_cos=None,
                   stream=None):
        """
        Generating a perturbed ocean state and adding it to sim's ocean state 
        """
        self.perturbOceanState(sim.gpu_data.h0, sim.gpu_data.hu0, sim.gpu_data.hv0,
                               sim.bathymetry.Bi,
                               sim.dx, sim.dy,
                               sim.f, beta=sim.coriolis_beta, 
                               g=sim.g, 
                               y0_reference_cell=sim.y_zero_reference_cell,
                               update_random_field=update_random_field,
                               perturbation_scale=perturbation_scale,
                               land_mask_value=sim.bathymetry.mask_value,
                               random_numbers=random_numbers, 
                               roll_x_sin=roll_x_sin, roll_y_sin=roll_y_sin,
                               roll_x_cos=roll_x_cos, roll_y_cos=roll_y_cos,
                               stream=stream)
                               
    
    def perturbOceanState(self, eta, hu, hv, H, 
                          dx, dy, f, beta=0.0, g=9.81, 
                          y0_reference_cell=0, 
                          update_random_field=True, 
                          perturbation_scale=1.0,
                          land_mask_value=np.float32(1.0e20),
                          random_numbers=None, 
                          roll_x_sin=None, roll_y_sin=None,
                          roll_x_cos=None, roll_y_cos=None,
                          stream=None):
        """
        Sample random perturbation using the KL basis functions and add it to eta, hu, hv
        eta: surface deviation - CUDAArray2D object.
        hu: volume transport in x-direction - CUDAArray2D object.
        hv: volume transport in y-dirextion - CUDAArray2D object.
        """
        
        if stream is None:
            stream = self.gpu_stream
        
        if random_numbers is not None:
            assert(isinstance(random_numbers, (Common.CUDAArray2D, np.ndarray))), "random numbers should be a CUDA 2D array or a numpy array, but is " + str(type(random_numbers))
            if isinstance(random_numbers, Common.CUDAArray2D): 
                # print("---------------------------------")
                # print("Before copyBuffer", stream)
                # print("self.gpu_stream: ", self.gpu_stream)
                # print("self.random_numbers: ", self.random_numbers)
                # print("random_numbers: ", random_numbers)
                # print("self.random_numbers.data.ptr: ", self.random_numbers.data.ptr)
                # print("random_numbers.data.ptr: ", random_numbers.data.ptr)
                # print("self.random_numbers.bytes_per_float: ", self.random_numbers.bytes_per_float)
                # print("random_numbers.bytes_per_float: ", random_numbers.bytes_per_float)
                # print("self.random_numbers.holds_data: ", self.random_numbers.holds_data)
                # print("random_numbers.holds_data: ", random_numbers.holds_data)
                # print("random_numbers. nx, ny, nx_halo, ny_halo", random_numbers.nx, random_numbers.ny, random_numbers.nx_halo, random_numbers.ny_halo)
                # print("eta. nx, ny, nx_halo, ny_halo", eta.nx, eta.ny, eta.nx_halo, eta.ny_halo)
                # print("---------------------------------")
                #
                #self.random_numbers.copyBuffer(stream, random_numbers)
                # TODO: Figure out why not copyBuffer works!?!

                tmp_rns = random_numbers.download(stream)
                self.random_numbers.upload(stream, tmp_rns)
            else:
                self.random_numbers.upload(stream, random_numbers)
        elif update_random_field:
            # Need to update the random field, requiering a global sync
            self.generateNormalDistribution()
        
        self._setRollers(roll_x_sin, roll_y_sin, roll_x_cos, roll_y_cos)

        self.klSamplingKernel.prepared_call(self.global_size_KL, self.local_size, 
                                            self.nx, self.ny, dx, dy,
                                            g, f, beta, 
                                            self.basis_x_start, self.basis_x_end,
                                            self.basis_y_start, self.basis_y_end,
                                            self.include_cos, self.include_sin,
                                            self.kl_decay, 
                                            np.float32(perturbation_scale * self.kl_scaling),
                                            np.float32(self.roll_x_sin), np.float32(self.roll_y_sin), 
                                            np.float32(self.roll_x_cos), np.float32(self.roll_y_cos), 
                                            
                                            self.random_numbers.data.gpudata, self.random_numbers.pitch,
                                            eta.data.gpudata, eta.pitch,
                                            hu.data.gpudata, hu.pitch,
                                            hv.data.gpudata, hv.pitch,
                                            H.data.gpudata, H.pitch,
                                            land_mask_value
                                            )    
        
    def perturbSimSimilarAs(self, simToPerturb, simSource=None, modelError=None, stream=None, 
                            perturbation_scale=1.0):

        assert(simSource is not None or modelError is not None), "Please provide either simSource or modelError input arguments"
        assert(simSource is None or modelError is None), "Please provide only one of simSource or modelError, not both."

        if simSource is not None:
            modelError = simSource.model_error

        self.perturbSim(simToPerturb, perturbation_scale=perturbation_scale,
                        random_numbers=modelError.random_numbers, 
                        roll_x_cos=modelError.roll_x_cos, roll_y_cos=modelError.roll_y_cos,
                        roll_x_sin=modelError.roll_x_sin, roll_y_sin=modelError.roll_y_sin,
                        stream=stream)
    
    ##### CPU versions of the above functions ####
    
    def getSeedCPU(self):
        assert(self.use_lcg), "getSeedCPU is only valid if LCG is used as pseudo-random generator."
        return self.host_seed
    
    def generateNormalDistributionCPU(self):
        self._CPUUpdateRandom()
       
    def getRandomNumbersCPU(self):
        return self.random_numbers_host
    
    def _initBasisFieldsCPU(self):
        # Initialize basis fields that have a halo of one ghost cell, so that we can compute geostrophic balances of the results
        
        half_unit_dy = (1.0/self.ny)/2.0
        half_unit_dx = (1.0/self.nx)/2.0
        if self.KL_basis_fields_sin is None and self.include_sin:
            self.KL_basis_fields_sin = np.zeros((self.N_basis_y, self.N_basis_x, self.ny+2, self.nx+2))

            for n in range(0, self.N_basis_y):
                for m in range(0, self.N_basis_x):
                    basis_n = self.basis_y_start + n
                    basis_m = self.basis_x_start + m
                    self.KL_basis_fields_sin[n, m, :, :] = (self.kl_scaling * basis_m**(-self.kl_decay) * basis_n**(-self.kl_decay) * 
                                                            np.outer(np.sin(2*basis_m*np.pi*np.linspace(-half_unit_dy , 1+half_unit_dy, self.ny+2)), 
                                                                     np.sin(2*basis_n*np.pi*np.linspace(-half_unit_dx , 1+half_unit_dx, self.nx+2))) )

            self.KL_basis_fields_sin = np.reshape(self.KL_basis_fields_sin, (self.N_basis_y*self.N_basis_x, self.ny+2, self.nx+2))

        if self.KL_basis_fields_cos is None and self.include_cos:
            self.KL_basis_fields_cos = np.zeros((self.N_basis_y, self.N_basis_x, self.ny+2, self.nx+2))

            for n in range(0, self.N_basis_y):
                for m in range(0, self.N_basis_x):
                    basis_n = self.basis_y_start + n
                    basis_m = self.basis_x_start + m
                    self.KL_basis_fields_cos[n, m, :, :] = (self.kl_scaling * basis_m**(-self.kl_decay) * basis_n**(-self.kl_decay) * 
                                                            np.outer(np.cos(2*basis_m*np.pi*np.linspace(-half_unit_dy , 1+half_unit_dy, self.ny+2)), 
                                                                     np.cos(2*basis_n*np.pi*np.linspace(-half_unit_dx , 1+half_unit_dx, self.nx+2))) )
            self.KL_basis_fields_cos = np.reshape(self.KL_basis_fields_cos, (self.N_basis_y*self.N_basis_x, self.ny+2, self.nx+2))
        

    def perturbEtaCPU(self, eta, use_existing_GPU_random_numbers=False, random_numbers=None, 
                      roll_x_sin=None, roll_y_sin=None,
                      roll_x_cos=None, roll_y_cos=None):
        """
        Sample random field using the KL basis to the incomming eta buffer.
        eta: numpy array
        """

        assert(eta.shape == (self.ny+4, self.nx+4)), "expected eta to be shape " + str((self.ny+4, self.nx+4)) + " but got " + str(eta.shape)

        d_eta = self._CPUSampleKL(use_existing_GPU_random_numbers=use_existing_GPU_random_numbers, 
                                  random_numbers=random_numbers, 
                                  roll_x_sin=roll_x_sin, roll_y_sin=roll_y_sin,
                                  roll_x_cos=roll_x_cos, roll_y_cos=roll_y_cos)
        
        eta[2:-2, 2:-2] = d_eta[1:-1, 1:-1]
    
    def perturbOceanStateCPU(self, eta, hu, hv, Hi, f,  beta=0.0, g=9.81,
                             use_existing_GPU_random_numbers=False, random_numbers=None, 
                             roll_x_sin=None, roll_y_sin=None,
                             roll_x_cos=None, roll_y_cos=None):
        """
        Sample random field using the KL basis to the incomming eta buffer.
        Generate geostrophically balanced hu and hv which is added to the incomming hu and hv buffers.
        eta: numpy array
        """
        assert(eta.shape == (self.ny+4, self.nx+4)), "expected eta to be shape " + str((self.ny+4, self.nx+4)) + " but got " + str(eta.shape)
        assert(hu.shape  == (self.ny+4, self.nx+4)),  "expected hu to be shape " + str((self.ny+4, self.nx+4)) + " but got " + str(hu.shape)
        assert(hv.shape  == (self.ny+4, self.nx+4)),  "expected hv to be shape " + str((self.ny+4, self.nx+4)) + " but got " + str(hv.shape)

        # Make KL type perturbation of d_eta. Size (ny+2, nx+2)
        d_eta = self._CPUSampleKL(use_existing_GPU_random_numbers=use_existing_GPU_random_numbers, 
                             random_numbers=random_numbers, 
                             roll_x_sin=roll_x_sin, roll_y_sin=roll_y_sin,
                             roll_x_cos=roll_x_cos, roll_y_cos=roll_y_cos)
        assert(d_eta.shape == (self.ny+2, self.nx+2)), "expected d_eta to be shape " + str(self.ny+2, self.nx+2) + " but got " + str(d_eta.shape)
 
        d_hu = np.zeros((self.ny, self.nx))
        d_hv = np.zeros((self.ny, self.nx))

        ### Find H_mid:
        # Read global H (def on intersections) to local, find H_mid
        # The local memory can then be reused to something else (perhaps use local_d_eta before computing local_d_eta?)
        H_mid = np.zeros((self.ny, self.nx))
        for j in range(self.ny):
            for i in range(self.nx):
                H_mid[j,i] = 0.25* (Hi[j,i] + Hi[j+1, i] + Hi[j, i+1] + Hi[j+1, i+1])
        
        ####
        # Local sync
        ####

        # Compute geostrophically balanced (hu, hv) for each cell within the domain
        for j in range(0, self.ny):
            local_j = j + 1     # index in d_eta buffer
            coriolis = f + beta*local_j*self.dy
            for i in range(0, self.nx):
                local_i = i + 1    # index in d_eta buffer
                h_mid = d_eta[local_j,local_i] + H_mid[j, i]
                
                eta_diff_y = (d_eta[local_j+1, local_i] - d_eta[local_j-1, local_i])/(2.0*self.dy)
                d_hu[j,i] = -(g/coriolis)*h_mid*eta_diff_y

                eta_diff_x = (d_eta[local_j, local_i+1] - d_eta[local_j, local_i-1])/(2.0*self.dx)
                d_hv[j,i] = (g/coriolis)*h_mid*eta_diff_x   
    

        
        eta[2:-2, 2:-2] += d_eta[1:-1, 1:-1]
        hu[ 2:-2, 2:-2] += d_hu
        hv[ 2:-2, 2:-2] += d_hv
    
    
     
    
    # ------------------------------
    # CPU utility functions:
    # ------------------------------
    
    def _lcg(self, seed):
        modulo = np.uint64(2147483647)
        seed = np.uint64(((seed*1103515245) + 12345) % modulo) #0x7fffffff
        return seed / 2147483648.0, seed
    
    def _boxMuller(self, seed_in):
        seed = np.uint64(seed_in)
        u1, seed = self._lcg(seed)
        u2, seed = self._lcg(seed)
        r = np.sqrt(-2.0*np.log(u1))
        theta = 2*np.pi*u2
        n1 = r*np.cos(theta)
        n2 = r*np.sin(theta)
        return n1, n2, seed
    
    def _CPUUpdateRandom(self):
        """
        Updating the random number buffer at the CPU.
        """

        if not self.use_lcg:
            self.random_numbers_host = np.random.normal(size=self.random_numbers_host.shape)
            return
        
        #(ny, nx) = seed.shape
        #(domain_ny, domain_nx) = random.shape
        b_dim_x = self.local_size[0]
        b_dim_y = self.local_size[1]
        blocks_x = self.global_size_random_numbers[0]
        blocks_y = self.global_size_random_numbers[1]
        for by in range(blocks_y):
            for bx in range(blocks_x):
                for j in range(b_dim_y):
                    for i in range(b_dim_x):

                        ## Content of kernel:
                        y = b_dim_y*by + j # thread_id
                        x = b_dim_x*bx + i # thread_id
                        if (x < self.seed_nx and y < self.seed_ny):
                            n1, n2, self.host_seed[y,x]   = self._boxMuller(self.host_seed[y,x])   
                            if x*2 + 1 < self.rand_nx:
                                self.random_numbers_host[y, x*2  ] = n1
                                self.random_numbers_host[y, x*2+1] = n2
                            elif x*2 == self.rand_nx:
                                self.random_numbers_host[y, x*2] = n1
    
    def _CPUSampleKL(self, use_existing_GPU_random_numbers=False, random_numbers=None, 
                     roll_x_sin=None, roll_y_sin=None,
                     roll_x_cos=None, roll_y_cos=None):
        """
        Sample random field using the KL basis to the incomming eta buffer.
        eta: numpy array
        """
        self._initBasisFieldsCPU()

        # Call CPU utility function
        if use_existing_GPU_random_numbers:
            self.random_numbers_host = self.getRandomNumbers()
        else:
            if random_numbers is not None:
                self.setRandomNumbers(random_numbers)
            else:
                self.generateNormalDistributionCPU()

            self._setRollers(roll_x_sin, roll_y_sin, roll_x_cos, roll_y_cos)

        # Approximate the random rolling to the grid
        roll_x_sin = int(np.floor(self.roll_x_sin*self.nx))
        roll_y_sin = int(np.floor(self.roll_y_sin*self.ny))
        roll_x_cos = int(np.floor(self.roll_x_cos*self.nx))
        roll_y_cos = int(np.floor(self.roll_y_cos*self.ny))

        sin_rns = self.random_numbers_host[:self.N_basis_y, :]
        cos_rns = self.random_numbers_host[self.N_basis_y:, :]
        
        # We transpose the random numbers so that they corresponds with the
        # GPU kernel
        sin_rns = np.reshape(sin_rns.T, (self.N_basis_y*self.N_basis_x))
        cos_rns = np.reshape(cos_rns.T, (self.N_basis_y*self.N_basis_x))
        
        d_eta = np.zeros((self.ny+2, self.nx+2))
        if self.include_sin:
            d_eta_sin = np.sum(self.KL_basis_fields_sin*sin_rns[:,np.newaxis, np.newaxis], axis=0)
            # Rolling while respecting boundary conditions:
            d_eta_sin[1:-1, 1:-1] = np.roll(np.roll(d_eta_sin[1:-1, 1:-1], roll_y_sin, 0), roll_x_sin, 1)
            self._applyBCSingleGhost(d_eta_sin)
            d_eta += d_eta_sin   
        if self.include_cos:
            d_eta_cos = np.sum(self.KL_basis_fields_cos*cos_rns[:,np.newaxis, np.newaxis], axis=0) 
            # Rolling while respecting boundary conditions
            d_eta_cos[1:-1, 1:-1] = np.roll(np.roll(d_eta_cos[1:-1, 1:-1], roll_y_cos, 0), roll_x_cos, 1)
            self._applyBCSingleGhost(d_eta_cos)
            d_eta += d_eta_cos  
        return d_eta
    

    def _applyBCSingleGhost(self, data):
        assert(data.shape == (self.ny+2, self.nx+2)), "Expected data to be of shape " + str((self.ny+2, self.nx+2)) + " while got " + str(data.shape)
        
        # Check if bc is wall. Otherwise, treat as periodic
        if self.boundary_conditions.north == 1:
            data[-1, :] = data[-2, :]
        else:
            data[-1, :] = data[ 1, :] 
        
        if self.boundary_conditions.south == 1:
            data[0, :] = data[1, :]
        else:
            data[0, :] = data[-2, :] 
        
        if self.boundary_conditions.west == 1:
            data[:, 0] = data[:, 1]
        else:
            data[:, 0] = data[:, -2]
        
        if self.boundary_conditions.east == 1:
            data[:, -1] = data[:, -2]
        else:
            data[:, -1] = data[:, 1]
