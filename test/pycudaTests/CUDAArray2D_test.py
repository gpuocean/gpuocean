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
import time
import numpy as np
import sys
import gc
import pycuda.driver as cuda

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common

#reload(GlobalParticles)

class CUDAArray2DTest(unittest.TestCase):

    def setUp(self):

        #Set which CL device to use, and disable kernel caching
        self.gpu_ctx = Common.CUDAContext()
                    
        # Make some host data which we can play with
        self.nx = 3
        self.ny = 5
        self.nx_halo = 1
        self.ny_halo = 2
        self.dataShape = (self.ny + 2*self.ny_halo, self.nx + 2*self.nx_halo)
        
        self.buf1 = np.zeros(self.dataShape, dtype=np.float32, order='C')
        self.dbuf1 = np.zeros(self.dataShape)
        self.buf3 = np.zeros(self.dataShape, dtype=np.float32, order='C')
        self.dbuf3= np.zeros(self.dataShape)
        for j in range(self.dataShape[0]):
            for i in range(self.dataShape[1]):
                self.buf1[j,i] = i*100 + j
                self.dbuf1[j,i] = self.buf1[j,i]
                self.buf3[j,i] = j*1000 - i
                self.dbuf3[j,i] = self.buf3[j,i]
                
        self.explicit_free = False
        
        self.device_name = self.gpu_ctx.cuda_device.name()
        self.gpu_stream = cuda.Stream()

        self.tests_failed = True

        self.cudaarray = Common.CUDAArray2D(self.gpu_stream, \
                                            self.nx, self.ny, \
                                            self.nx_halo, self.ny_halo, \
                                            self.buf1)

        self.double_cudaarray = None

        
    def tearDown(self):
        if self.tests_failed:
            print("Device name: " + self.device_name)
        if not self.explicit_free:
            self.cudaarray.release()
        if self.double_cudaarray is not None:
            self.double_cudaarray.release()
        del self.gpu_ctx

    ### Utils ###
    def init_double(self):
        self.double_cudaarray = Common.CUDAArray2D(self.gpu_stream, \
                                                 self.nx, self.ny, \
                                                 self.nx_halo, self.ny_halo, \
                                                 self.dbuf1, \
                                                 double_precision=True)
            
    ### START TESTS ###

    def test_init(self):
        self.assertEqual(self.cudaarray.nx,  self.nx)
        self.assertEqual(self.cudaarray.ny, self.ny)
        self.assertEqual(self.cudaarray.nx_halo, self.nx + 2*self.nx_halo)
        self.assertEqual(self.cudaarray.ny_halo, self.ny + 2*self.ny_halo)
        
        self.assertTrue(self.cudaarray.holds_data)
        self.assertEqual(self.cudaarray.bytes_per_float, 4)
        self.assertEqual(self.cudaarray.pitch, 4*(self.nx + 2*self.nx_halo))
        self.tests_failed = False
        
    def test_release(self):
        #self.explicit_free = True
        self.cudaarray.release()
        self.assertFalse(self.cudaarray.holds_data)

        with self.assertRaises(RuntimeError):
            self.cudaarray.download(self.gpu_stream)

        with self.assertRaises(RuntimeError):
            self.cudaarray.upload(self.gpu_stream, self.buf3)
        
        self.tests_failed = False
    

    def test_download(self):
        
        host_data = self.cudaarray.download(self.gpu_stream)
        self.assertEqual(host_data.tolist(), self.buf1.tolist())
        self.tests_failed = False

    def test_upload(self):
        self.cudaarray.upload(self.gpu_stream, self.buf3)
        host_data = self.cudaarray.download(self.gpu_stream)
        self.assertEqual(host_data.tolist(), self.buf3.tolist())
        self.tests_failed = False

    def test_copy_buffer(self):
        clarray2 = Common.CUDAArray2D(self.gpu_stream, \
                                      self.nx, self.ny, self.nx_halo, self.ny_halo, \
                                      self.buf3)

        host_data_pre_copy = self.cudaarray.download(self.gpu_stream)
        self.assertEqual(host_data_pre_copy.tolist(), self.buf1.tolist())
        
        self.cudaarray.copyBuffer(self.gpu_stream, clarray2)
        host_data_post_copy = self.cudaarray.download(self.gpu_stream)
        self.assertEqual(host_data_post_copy.tolist(), self.buf3.tolist())
        
        self.tests_failed = False
        
    # Double precision
    def test_double_init(self):
        self.init_double()

        self.assertEqual(self.double_cudaarray.nx,  self.nx)
        self.assertEqual(self.double_cudaarray.ny, self.ny)
        self.assertEqual(self.double_cudaarray.nx_halo, self.nx + 2*self.nx_halo)
        self.assertEqual(self.double_cudaarray.ny_halo, self.ny + 2*self.ny_halo)
        
        self.assertTrue(self.double_cudaarray.holds_data)
        self.assertEqual(self.double_cudaarray.bytes_per_float, 8)
        self.assertEqual(self.double_cudaarray.pitch, 8*(self.nx + 2*self.nx_halo))
        self.tests_failed = False

    def test_double_release(self):
        self.init_double()
        
        self.double_cudaarray.release()
        self.assertFalse(self.double_cudaarray.holds_data)

        with self.assertRaises(RuntimeError):
            self.double_cudaarray.download(self.gpu_stream)

        with self.assertRaises(RuntimeError):
            self.double_cudaarray.upload(self.gpu_stream, self.dbuf3)
        
        self.tests_failed = False
    

    def test_double_download(self):
        self.init_double()
        
        host_data = self.double_cudaarray.download(self.gpu_stream)
        self.assertEqual(host_data.tolist(), self.dbuf1.tolist())
        self.tests_failed = False

    def test_double_upload(self):
        self.init_double()

        self.double_cudaarray.upload(self.gpu_stream, self.dbuf3)
        host_data = self.double_cudaarray.download(self.gpu_stream)
        self.assertEqual(host_data.tolist(), self.dbuf3.tolist())
        self.tests_failed = False

    def test_double_copy_buffer(self):
        self.init_double()
        
        double_cudaarray2 = Common.CUDAArray2D(self.gpu_stream, \
                                               self.nx, self.ny, \
                                               self.nx_halo, self.ny_halo, \
                                               self.dbuf3, \
                                               double_precision=True)

        host_data_pre_copy = self.double_cudaarray.download(self.gpu_stream)
        self.assertEqual(host_data_pre_copy.tolist(), self.dbuf1.tolist())
        
        self.double_cudaarray.copyBuffer(self.gpu_stream, double_cudaarray2)
        host_data_post_copy = self.double_cudaarray.download(self.gpu_stream)
        self.assertEqual(host_data_post_copy.tolist(), self.dbuf3.tolist())
        
        self.tests_failed = False

    def test_cross_precision_copy_buffer(self):
        self.init_double()
        
        single_cudaarray2 = Common.CUDAArray2D(self.gpu_stream, \
                                               self.nx, self.ny, \
                                               self.nx_halo, self.ny_halo, \
                                               self.buf3)

        host_data_pre_copy = self.double_cudaarray.download(self.gpu_stream)
        self.assertEqual(host_data_pre_copy.tolist(), self.dbuf1.tolist())
        
        with self.assertRaises(AssertionError):
            self.double_cudaarray.copyBuffer(self.gpu_stream, single_cudaarray2)
        
        self.tests_failed = False

    

        
