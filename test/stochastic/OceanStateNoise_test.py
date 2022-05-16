# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements regression tests for the OceanStateNoise class.

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
from SWESimulators.OceanStateNoise import *
from tests.stochastic.OceanStateNoise_parent import OceanStateNoiseTestParent

class OceanStateNoiseTest(OceanStateNoiseTestParent):


    def test_init_periodic_nonstaggered(self):
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx)
        self.assertEqual(self.noise.rand_ny, self.ny)
        self.assertEqual(self.noise.seed_nx, self.nx/2)
        self.assertEqual(self.noise.seed_ny, self.ny)

        self.assertEqual(self.noise.global_size_SOAR, (self.glob_size_x, self.glob_size_y))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x, self.glob_size_y))

        self.compare_random(6, "test_init_periodic_nonstaggered")

        
    def test_init_periodic_staggered(self):
        self.staggered = True
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx)
        self.assertEqual(self.noise.rand_ny, self.ny)
        self.assertEqual(self.noise.seed_nx, self.nx/2)
        self.assertEqual(self.noise.seed_ny, self.ny)

        self.assertEqual(self.noise.global_size_SOAR, (self.glob_size_x, self.glob_size_y))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x, self.glob_size_y))

        self.compare_random(6, "test_init_periodic_staggered")

    def test_init_non_periodic_nonstaggered(self):
        self.periodicEW = False
        self.periodicNS = False
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx_nonPeriodic)
        self.assertEqual(self.noise.rand_ny, self.ny_nonPeriodic)
        self.assertEqual(self.noise.seed_nx, self.nx_nonPeriodic/2)
        self.assertEqual(self.noise.seed_ny, self.ny_nonPeriodic)        

        self.assertEqual(self.noise.global_size_SOAR, (self.glob_size_x_nonperiodic, self.glob_size_y_nonperiodic))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x_nonperiodic, self.glob_size_y_nonperiodic))

        self.compare_random(6, "test_init_non_periodi_nonstaggered")
        
    def test_init_periodicNS_nonstaggered(self):
        self.periodicEW = False
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx_nonPeriodic)
        self.assertEqual(self.noise.rand_ny, self.ny)
        self.assertEqual(self.noise.seed_nx, self.nx_nonPeriodic/2)
        self.assertEqual(self.noise.seed_ny, self.ny)        

        self.assertEqual(self.noise.global_size_SOAR, (self.glob_size_x_nonperiodic, self.glob_size_y))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x_nonperiodic, self.glob_size_y))

        self.compare_random(6, "test_init_periodiNS_nonstaggered")
        
    def test_init_periodicEW_nonstaggered(self):
        self.periodicNS = False
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx)
        self.assertEqual(self.noise.rand_ny, self.ny_nonPeriodic)
        self.assertEqual(self.noise.seed_nx, self.nx/2)
        self.assertEqual(self.noise.seed_ny, self.ny_nonPeriodic)        

        self.assertEqual(self.noise.global_size_SOAR, (self.glob_size_x, self.glob_size_y_nonperiodic))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x, self.glob_size_y_nonperiodic))

        self.compare_random(6, "test_init_periodiEW_nonstaggered")
  

    def perturb_eta(self, msg):
        etaCPU = np.zeros(self.datashape)
        HCPU = np.ones((self.datashape[0]+1, self.datashape[1]+1))*5
        self.create_noise()
        self.allocateBuffers(HCPU)
        
        self.noise.perturbOceanState(self.eta, self.hu, self.hv, self.H,
                                     self.f, self.beta, self.g,
                                     ghost_cells_x=self.ghost_cells_x,
                                     ghost_cells_y=self.ghost_cells_y)
        self.noise.perturbEtaCPU(etaCPU, use_existing_GPU_random_numbers=True,
                                 ghost_cells_x=self.ghost_cells_x,
                                 ghost_cells_y=self.ghost_cells_y)
        
        etaFromGPU = self.eta.download(self.gpu_stream)
        
        # Scale so that largest value becomes ~ 1
        maxVal = np.max(etaCPU)
        #print("maxVal: ", maxVal)
        etaFromGPU = etaFromGPU / maxVal
        etaCPU = etaCPU / maxVal
        
        assert2DListAlmostEqual(self, etaCPU.tolist(), etaFromGPU.tolist(), 6, msg)

        
    def test_perturb_eta_periodic(self):
        self.perturb_eta("test_perturb_eta_periodic")

    def test_perturb_eta_nonperiodic(self):
        self.periodicNS = False
        self.periodicEW = False
        self.perturb_eta("test_perturb_eta_nonperiodic")

    def test_perturb_eta_NS_periodic(self):
        self.periodicEW = False
        self.perturb_eta("test_perturb_eta_NS_periodic")

    def test_perturb_eta_EW_periodic(self):
        self.periodicNS = False
        self.perturb_eta("test_perturb_eta_EW_periodic")



    def perturb_ocean(self, msg):
        etaCPU = np.zeros(self.datashape)
        huCPU = np.zeros(self.datashape)
        hvCPU = np.zeros(self.datashape)
        HCPU = np.ones((self.datashape[0]+1, self.datashape[1]+1))*5

        self.create_noise()
        self.allocateBuffers(HCPU)

        self.noise.perturbOceanState(self.eta, self.hu, self.hv, self.H,
                                     self.f, self.beta, self.g)
        self.noise.perturbOceanStateCPU(etaCPU, huCPU, hvCPU, HCPU,
                                        self.f, self.beta, self.g,
                                        use_existing_GPU_random_numbers=True,
                                        ghost_cells_x=self.ghost_cells_x,
                                        ghost_cells_y=self.ghost_cells_y)
        etaFromGPU = self.eta.download(self.gpu_stream)
        huFromGPU  = self.hu.download(self.gpu_stream)
        hvFromGPU  = self.hv.download(self.gpu_stream)

        # Scale so that largest value becomes ~ 1:
        maxVal = np.max(huCPU)
        huFromGPU = huFromGPU / maxVal
        hvFromGPU = hvFromGPU / maxVal
        huCPU = huCPU / maxVal
        hvCPU = hvCPU / maxVal
        
        maxValEta = np.max(etaCPU)
        etaCPU = etaCPU / maxValEta
        etaFromGPU = etaFromGPU / maxValEta 
        
        assert2DListAlmostEqual(self, etaCPU.tolist(), etaFromGPU.tolist(), 5, msg+", eta")
        assert2DListAlmostEqual(self, huCPU.tolist(), huFromGPU.tolist(), 5, msg+", hu")
        assert2DListAlmostEqual(self, hvCPU.tolist(), hvFromGPU.tolist(), 5, msg+", hv")
        

        
    def test_perturb_ocean_periodic(self):
        self.perturb_ocean("test_perturb_ocean_periodic")

        
    def test_perturb_ocean_nonperiodic(self):
        self.periodicNS = False
        self.periodicEW = False
        self.perturb_ocean("test_perturb_ocean_nonperiodic")

        
    def test_perturb_ocean_EW_periodic(self):
        self.periodicNS = False
        self.perturb_ocean("test_perturb_ocean_EW_periodic")

        
    def test_perturb_ocean_NS_periodic(self):
        self.periodicEW = False
        self.perturb_ocean("test_perturb_ocean_NS_periodic")

        
    def interpolate_perturbation_ocean(self, msg, factor):
        
        self.nx = factor*self.nx
        self.ny = factor*self.ny
        self.datashape = (self.ny+4, self.nx+4)
        etaCPU = np.zeros(self.datashape)
        huCPU =  np.zeros(self.datashape)
        hvCPU =  np.zeros(self.datashape)
        HCPU = np.ones((self.datashape[0]+1, self.datashape[1]+1))*5
        self.create_noise(factor=factor)
        self.allocateBuffers(HCPU)
        
          
        self.noise.perturbOceanState(self.eta, self.hu, self.hv, self.H,
                                     self.f, self.beta, self.g,
                                     ghost_cells_x=self.ghost_cells_x,
                                     ghost_cells_y=self.ghost_cells_y)
      
        
        self.noise.perturbOceanStateCPU(etaCPU, huCPU, hvCPU, HCPU,
                                        self.f, self.beta, self.g,
                                        use_existing_GPU_random_numbers=True,
                                        ghost_cells_x=self.ghost_cells_x,
                                        ghost_cells_y=self.ghost_cells_y)
        
        
        etaFromGPU = self.eta.download(self.gpu_stream)
        huFromGPU = self.hu.download(self.gpu_stream)
        hvFromGPU = self.hv.download(self.gpu_stream)

        
        # Scale so that largest value becomes ~ 1
        maxVal = np.max(etaCPU)
        #print("maxVal: ", maxVal)
        etaFromGPU = etaFromGPU / maxVal
        etaCPU = etaCPU / maxVal
        
        maxValhuv = np.max(huCPU)
        huFromGPU = huFromGPU / maxValhuv
        hvFromGPU = hvFromGPU / maxValhuv
        huCPU = huCPU / maxValhuv
        hvCPU = hvCPU / maxValhuv
        
        # Compare perturbation between CPU and GPU 
        assert2DListAlmostEqual(self, etaCPU.tolist(), etaFromGPU.tolist(), 6, msg+", eta")
        
        coarse = self.noise.getCoarseBuffer()
        inner_coarse = coarse[2:-2, 2:-2] / maxVal
        inner_eta = etaFromGPU[2:-2, 2:-2]
        first_center = int((factor-1)/2)
        coarse_vals_eta = inner_eta[first_center::factor, first_center::factor]

        # Check that the coarse grid equals with the aligned fine grid points 
        assert2DListAlmostEqual(self, inner_coarse.tolist(), coarse_vals_eta.tolist(), 5, msg + " - coarse vs fine")
        
        assert2DListAlmostEqual(self, huCPU.tolist(), huFromGPU.tolist(), 5, msg+", hu")
        assert2DListAlmostEqual(self, hvCPU.tolist(), hvFromGPU.tolist(), 5, msg+", hv")
        
        
        
    def test_interpolation_3_ocean(self):
        self.interpolate_perturbation_ocean("test_interpolation_ocean with factor 3", 3)
    
    def test_interpolation_5_ocean(self):
        self.interpolate_perturbation_ocean("test_interpolation_ocean with factor 5", 5)
    
    def test_interpolation_7_ocean(self):
        self.interpolate_perturbation_ocean("test_interpolation_ocean with factor 7", 7)

    def interpolate_perturbation_ocean_offset(self, base_msg, factor, align_i_list, align_j_list, offset_i_list, offset_j_list):
        
        self.nx = self.nx
        self.ny = self.ny
        self.datashape = (self.ny+4, self.nx+4)
        etaCPU = np.zeros(self.datashape)
        huCPU =  np.zeros(self.datashape)
        hvCPU =  np.zeros(self.datashape)
        HCPU = np.ones((self.datashape[0]+1, self.datashape[1]+1))*5
        self.create_noise(factor=factor)
        self.allocateBuffers(HCPU)
        
        self.noise.generateNormalDistribution()
        
        self.noise.perturbOceanStateCPU(etaCPU, huCPU, hvCPU, HCPU,
                                        self.f, self.beta, self.g,
                                        use_existing_GPU_random_numbers=True,
                                        ghost_cells_x=self.ghost_cells_x,
                                        ghost_cells_y=self.ghost_cells_y)
        
        # Scale so that largest value becomes ~ 1
        maxVal = np.max(etaCPU)
        etaCPU = etaCPU[2:-2,2:-2]/ maxVal
       
    
        maxValhuv = np.max(huCPU)
        huCPU = huCPU / maxValhuv
        hvCPU = hvCPU / maxValhuv
                
        count = 0
        for align_j, offset_j in zip(align_j_list, offset_j_list) :
            for align_i, offset_i in zip(align_i_list, offset_i_list):
                
                msg = base_msg + " align " + str((align_i, align_j)) +  ", offset " + str((offset_i, offset_j)) + ", "
                
                # Set variables to zero
                self.eta.upload(self.noise.gpu_stream, np.zeros(self.datashape, dtype=np.float32))
                self.hu.upload(self.noise.gpu_stream, np.zeros(self.datashape, dtype=np.float32))
                self.hv.upload(self.noise.gpu_stream, np.zeros(self.datashape, dtype=np.float32))
                
        
                self.noise.perturbOceanState(self.eta, self.hu, self.hv, self.H,
                                             self.f, self.beta, self.g,
                                             ghost_cells_x=self.ghost_cells_x,
                                             ghost_cells_y=self.ghost_cells_y,
                                             update_random_field=False,
                                             align_with_cell_i=align_i, align_with_cell_j=align_j)

                etaFromGPU = self.eta.download(self.gpu_stream)[2:-2, 2:-2]
                huFromGPU = self.hu.download(self.gpu_stream)
                hvFromGPU = self.hv.download(self.gpu_stream)

                skewed_eta_cpu_tmp = etaCPU.copy()
                if not offset_j == 0:
                    skewed_eta_cpu_tmp[-offset_j:,:] = etaCPU[:offset_j,:]
                    skewed_eta_cpu_tmp[:-offset_j,:] = etaCPU[offset_j:,:]
                skewed_eta_cpu = skewed_eta_cpu_tmp.copy()
                if not offset_i == 0:
                    skewed_eta_cpu[:, -offset_i:] = skewed_eta_cpu_tmp[:,:offset_i]
                    skewed_eta_cpu[:, :-offset_i] = skewed_eta_cpu_tmp[:,offset_i:]
                
                
                #print("maxVal: ", maxVal)
                etaFromGPU = etaFromGPU / maxVal
                huFromGPU = huFromGPU / maxValhuv
                hvFromGPU = hvFromGPU / maxValhuv

                # Compare perturbation between CPU and GPU 
                assert2DListAlmostEqual(self, skewed_eta_cpu.tolist(), etaFromGPU.tolist(), 5, msg+", eta")

                count = count+1
        
    def test_interpolation_offset_5(self):
        self.interpolate_perturbation_ocean_offset("test_interpolation_offset_5 with ", 5, 
                                                   [6,7,8], [6,7,8], [1,0,-1], [1,0,-1])
        #self.interpolate_perturbation_ocean_offset("test_interpolation_offset_5 with ", 5, [6,7,8], [6,7,8], [0,0,0], [0,0,0])
  
        