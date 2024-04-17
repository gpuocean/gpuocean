# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements unit tests for the central CUDAArray2D
class within GPU Ocean.

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

import unittest

import numpy as np
import pycuda.driver as cuda
import sys
import gc

from testUtils import *

from gpuocean.utils import Common


class CUDACDKLMTextureTest(unittest.TestCase):

    def setUp(self):

        self.gpu_ctx = Common.CUDAContext()
                    
        # Make some host data which we can play with
        self.Nx = 100 # size with ghost cells 
        self.Ny = 100 # size with ghost cells

        self.nx = self.Nx - 4 # size without ghost cells
        self.ny = self.Ny - 4 # size without ghost cells
                    
        self.defines={'block_width': 12, 'block_height': 32,
            'KPSIMULATOR_DESING_EPS': "{:.12f}f".format(0.1),
            'KPSIMULATOR_FLUX_SLOPE_EPS': "{:.12f}f".format(0.1),
            'KPSIMULATOR_DEPTH_CUTOFF': "{:.12f}f".format(1.0e-5),
            'THETA': "{:.12f}f".format(1.3),
            'RK_ORDER': int(2),
            'NX': int(self.nx),
            'NY': int(self.ny),
            'DX': "{:.12f}f".format(100),
            'DY': "{:.12f}f".format(100),
            'GRAV': "{:.12f}f".format(10.0),
            'FRIC': "{:.12f}f".format(0.0),
            'RHO_O': "{:.12f}f".format(1025.0),
            'FLUX_BALANCER': "0.0f",
            'ONE_DIMENSIONAL': "0",
            'WIND_STRESS_FACTOR': "0.0f"
            }

        self.kernel = self.gpu_ctx.get_kernel("CDKLM16_kernel.cu", 
                defines=self.defines, 
                compile_args={                          # default, fast_math, optimal
                    'options' : ["--ftz=true",          # false,   true,      true
                                 "--prec-div=false",    # true,    false,     false,
                                 "--prec-sqrt=false",   # true,    false,     false
                                 "--fmad=false"]        # true,    true,      false
                })

        self.gpu_stream = cuda.Stream()

        self.field = np.repeat(np.arange(self.Nx)[np.newaxis], self.Ny, axis=0).astype(np.float32)

        self.texref = None 
        
    def tearDown(self):
        if self.texref is not None:
            self.texref.release()
        del self.gpu_ctx


    ### START TESTS ###

    def test_get_texture_back(self):

        GPUtexref = self.kernel.get_texref("angle_tex")

        self.gpu_stream.synchronize()
        self.gpu_ctx.synchronize()

        GPUtexref.set_array(cuda.np_to_array(self.field, order="C"))
        GPUtexref.set_filter_mode(cuda.filter_mode.LINEAR) #bilinear interpolation
        GPUtexref.set_address_mode(0, cuda.address_mode.CLAMP) #no indexing outside domain
        GPUtexref.set_address_mode(1, cuda.address_mode.CLAMP)
        GPUtexref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES) #Use [0, 1] indexing

        self.gpu_ctx.synchronize()

        ## Sample texture
        local_size = (self.defines["block_width"], self.defines["block_height"], 1) 
        global_size = (int(np.ceil(self.Nx / float(local_size[0]))), int(np.ceil(self.Ny / float(local_size[1]))) ) 
        # ATTENTION: Ghost cells have to be included in the calculation of global_size


        self.texref = Common.CUDAArray2D(self.gpu_stream, self.Nx, self.Ny, 0, 0, np.zeros((self.Ny,self.Nx)))
        get_tex = self.kernel.get_function("get_texture")
        get_tex.prepare("Pi")
        get_tex.prepared_async_call(global_size, local_size, self.gpu_stream, self.texref.data.gpudata, np.int32(0))
        
        # Evaluate
        host_texref = self.texref.data.get()
        self.assertEqual(host_texref.tolist(), self.field.tolist())
        
        
    def test_sample_finer_texture(self):

        GPUtexref = self.kernel.get_texref("angle_tex")

        self.gpu_stream.synchronize()
        self.gpu_ctx.synchronize()

        GPUtexref.set_array(cuda.np_to_array(self.field, order="C"))
        GPUtexref.set_filter_mode(cuda.filter_mode.LINEAR) #bilinear interpolation
        GPUtexref.set_address_mode(0, cuda.address_mode.CLAMP) #no indexing outside domain
        GPUtexref.set_address_mode(1, cuda.address_mode.CLAMP)
        GPUtexref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES) #Use [0, 1] indexing

        self.gpu_ctx.synchronize()

        ## Sample texture
        local_size = (self.defines["block_width"], self.defines["block_height"], 1) 
        global_size = (int(np.ceil(2*self.Nx / float(local_size[0]))), int(np.ceil(2*self.Ny / float(local_size[1]))) ) 
        # ATTENTION: Global size has to fit to the biggest array


        self.texref = Common.CUDAArray2D(self.gpu_stream, 2*self.Nx, 2*self.Ny, 0, 0, np.zeros((2*self.Ny,2*self.Nx)))
        get_tex = self.kernel.get_function("sample_texture")
        get_tex.prepare("Piffffii")
        get_tex.prepared_async_call(global_size, local_size, self.gpu_stream, self.texref.data.gpudata, np.int32(0),
                                    np.float32(0.5/(2*self.Nx)), np.float32(1-0.5/(2*self.Nx)), np.float32(0.5/(2*self.Ny)), np.float32(1-0.5/(2*self.Ny)),
                                    np.int32(2*self.Nx), np.int32(2*self.Ny))
        
        # Evaluate
        texref_host = self.texref.data.get()[0]
        # For the comparison with the original field
        # We average over two neighboring cells
        # Furthermore, we ignore the first and last cell 
        # Due to the chosen type of interpolation on the texture, one cannot expect to match the original values there 
        texref_host_averaged = 0.5*(texref_host[1:] + texref_host[:-1])[::2][1:-1]
        self.assertAlmostEqual(texref_host_averaged.tolist(), self.field[0][1:-1].tolist(), 4)
