# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018, 2019, 2024  SINTEF Digital
Copyright (C) 2018, 2019 Norwegian Meteorological Institute

This class implements some simple random number generators on the GPU

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

import pycuda.gpuarray 
import pycuda.driver as cuda
from pycuda.curandom import XORWOWRandomNumberGenerator
import gc

from gpuocean.utils import Common



class RandomNumbers(object):
    """
    Class for generating random numbers within GPU Ocean.
    
    Arrays for holding the random numbers must be held by other objects, and only objects related to 
    the random number generation itself is held by this class.
    """

    def __init__(self, gpu_ctx, gpu_stream, nx, ny,
                 use_lcg=False,
                 xorwow_seed=None,
                 block_width=16, block_height=16):
        """
        Class for generating random numbers within GPU Ocean.
        
        (ny, nx): shape of the random number array that will be generated
        use_lcg: LCG is a linear algorithm for generating a serie of pseudo-random numbers
        angle: Angle of rotation from North to y-axis as a texture (cuda.Array) or numpy array
        (block_width, block_height): The size of each GPU block
        """
        
        self.use_lcg = use_lcg
        self.gpu_stream = gpu_stream

        # Set numpy random state
        self.random_state = np.random.RandomState()
        
        # Make sure that all variables initialized within ifs are defined
        self.rng = None
        self.seed = None
        self.host_seed = None

        self.nx = np.int32(nx)
        self.ny = np.int32(ny)

        # Since normal distributed numbers are generated in pairs, we need to store half the number of
        # of seed values compared to the number of random numbers. 
        # This split is in x-direction, and the dimension in y is kept as is
        self.seed_ny = np.int32(self.ny)
        self.seed_nx = np.int32(np.ceil(self.nx/2))

        # Generate seed:
        self.floatMax = 2147483648.0
        if self.use_lcg:
            self.host_seed = self.random_state.rand(self.seed_ny, self.seed_nx)*self.floatMax
            self.host_seed = self.host_seed.astype(np.uint64, order='C')
        
        if not self.use_lcg:
            if xorwow_seed is not None:
                def set_seeder(N, seed):
                    seedarr = pycuda.gpuarray.ones_like(pycuda.gpuarray.zeros(N, dtype=np.int32), dtype=np.int32) * seed
                    return seedarr

                self.rng = XORWOWRandomNumberGenerator( lambda N: set_seeder(N,xorwow_seed))
            else:
                self.rng = XORWOWRandomNumberGenerator()
        else:
            self.seed = Common.CUDAArray2D(self.gpu_stream, self.seed_nx, self.seed_ny, 0, 0, self.host_seed, double_precision=True, integers=True)
        

        # Generate kernels
        self.kernels = gpu_ctx.get_kernel("random_number_generators.cu", \
                                          defines={'block_width': block_width, 'block_height': block_height, 
                                                },
                                          compile_args={
                                              'options': ["--use_fast_math",
                                                          "--maxrregcount=32"]
                                          })
        
        # Get CUDA functions and define data types for prepared_{async_}call()
        # Generate kernels
        self.uniformDistributionKernel = self.kernels.get_function("uniformDistribution")
        self.uniformDistributionKernel.prepare("iiiPiPi")
        
        self.normalDistributionKernel = None
        if self.use_lcg:
            self.normalDistributionKernel = self.kernels.get_function("normalDistribution")
            self.normalDistributionKernel.prepare("iiiPiPi")


         #Compute kernel launch parameters
        self.local_size = (block_width, block_height, 1)
        
        # Launch one thread for each seed, which in turns generates two iid N(0,1)
        self.global_size_random_numbers = ( \
                       int(np.ceil(self.seed_nx / float(self.local_size[0]))), \
                       int(np.ceil(self.seed_ny / float(self.local_size[1]))) \
                     ) 
        

    def __del__(self):
        self.cleanUp()
     
    def cleanUp(self):
        if self.rng is not None:
            self.rng = None
        if self.seed is not None:
            self.seed.release()
        self.gpu_ctx = None
        gc.collect()
        

        

    def getSeed(self):
        assert(self.use_lcg), "getSeed is only valid if LCG is used as pseudo-random generator."
        
        return self.seed.download(self.gpu_stream)
    
    def resetSeed(self):
        assert(self.use_lcg), "resetSeed is only valid if LCG is used as pseudo-random generator."

        # Generate seed:
        self.host_seed = self.random_state.rand(self.seed_ny, self.seed_nx)*self.floatMax
        self.host_seed = self.host_seed.astype(np.uint64, order='C')
        self.seed.upload(self.gpu_stream, self.host_seed)

    def _checkInput(self, random_numbers):
        assert(isinstance(random_numbers, Common.CUDAArray2D)), "expected random_numbers of type CUDAArray2D but got " + str(type(random_numbers))
        shape_input = (random_numbers.ny, random_numbers.nx)
        shape_expected = (self.ny, self.nx)
        assert(shape_input == shape_expected), "expected random_numbers with shape " +str(shape_expected) + " but got " +str(shape_input)

    def generateNormalDistribution(self, random_numbers):
        self._checkInput(random_numbers)
        if not self.use_lcg:
            self.rng.fill_normal(random_numbers.data, stream=self.gpu_stream)
        else:
            self.normalDistributionKernel.prepared_async_call(self.global_size_random_numbers, self.local_size, self.gpu_stream,
                                                              self.seed_nx, self.seed_ny,
                                                              self.nx,
                                                              self.seed.data.gpudata, self.seed.pitch,
                                                              random_numbers.data.gpudata, random_numbers.pitch)
    
    def generateUniformDistribution(self, random_numbers):
        self._checkInput(random_numbers)
        if not self.use_lcg:
            self.rng.fill_uniform(random_numbers.data, stream=self.gpu_stream)
        else:
            self.uniformDistributionKernel.prepared_async_call(self.global_size_random_numbers, self.local_size, self.gpu_stream,
                                                               self.seed_nx, self.seed_ny,
                                                               self.nx,
                                                               self.seed.data.gpudata, self.seed.pitch,
                                                               random_numbers.data.gpudata, random_numbers.pitch)
