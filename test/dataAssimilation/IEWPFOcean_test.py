# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements unit tests for the IEWPFOcean class.

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
import sys
import gc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators import CDKLM16
from SWESimulators import OceanNoiseEnsemble
from SWESimulators import BathymetryAndICs as BC
from SWESimulators import DataAssimilationUtils as dautils
from SWESimulators import IEWPFOcean

class IEWPFOceanTest(unittest.TestCase):

    def setUp(self):
        self.sim = None
        self.ensemble = None
        self.iewpf = None
        
        self.gpu_ctx = Common.CUDAContext()

        self.setUpAndStartEnsemble()
        


    def tearDown(self):
        if self.sim is not None:
            self.sim.cleanUp()
            del self.sim
        if self.ensemble is not None:
            self.ensemble.cleanUp()
            del self.ensemble
        if self.iewpf is not None:
            self.iewpf.cleanUp()
            del self.iewpf
        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
        gc.collect()        
        

    def setUpAndStartEnsemble(self):
        self.nx = 40
        self.ny = 40
        
        self.dx = 4.0
        self.dy = 4.0

        self.dt = 0.05
        self.g = 9.81
        self.r = 0.0
        
        self.f = 0.05
        self.beta = 0.0

        self.waterDepth = 10.0
        
        self.ensembleSize = 3
        self.driftersPerOceanModel = 3
        
        ghosts = np.array([2,2,2,2]) # north, east, south, west
        validDomain = np.array([2,2,2,2])
        self.boundaryConditions = Common.BoundaryConditions(2,2,2,2)

        # Define which cell index which has lower left corner as position (0,0)
        x_zero_ref = 2
        y_zero_ref = 2

        dataShape = (self.ny + ghosts[0]+ghosts[2], 
                     self.nx + ghosts[1]+ghosts[3])
        dataShapeHi = (self.ny + ghosts[0]+ghosts[2]+1, 
                       self.nx + ghosts[1]+ghosts[3]+1)

        eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
        eta0_extra = np.zeros(dataShape, dtype=np.float32, order='C')
        hv0 = np.zeros(dataShape, dtype=np.float32, order='C');
        hu0 = np.zeros(dataShape, dtype=np.float32, order='C');
        Hi = np.ones(dataShapeHi, dtype=np.float32, order='C')*self.waterDepth

        # Add disturbance:
        rel_grid_size = self.nx*1.0/self.dx
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.3, 0.5, 0.05*rel_grid_size, validDomain)
        eta0 = eta0*0.3
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.7, 0.3, 0.10*rel_grid_size, validDomain)
        eta0 = eta0*(-1.3)
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.15, 0.8, 0.03*rel_grid_size, validDomain)
        eta0 = eta0*1.0
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.6, 0.75, 0.06*rel_grid_size, validDomain)
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.2, 0.2, 0.01*rel_grid_size, validDomain)
        eta0 = eta0*(-0.03)
        BC.addBump(eta0_extra, self.nx, self.ny, self.dx, self.dy, 0.5, 0.5, 0.4*rel_grid_size, validDomain)
        eta0 = eta0 + 0.02*eta0_extra
        BC.initializeBalancedVelocityField(eta0, Hi, hu0, hv0, \
                                           self.f, self.beta, self.g, \
                                           self.nx, self.ny, self.dx ,self.dy, ghosts)
        eta0 = eta0*0.5

        self.q0 = 0.5*self.dt*self.f/(self.g*self.waterDepth)
        
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, eta0, hu0, hv0, Hi, \
                                   self.nx, self.ny, self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, \
                                   boundary_conditions=self.boundaryConditions, \
                                   write_netcdf=False, \
                                   small_scale_perturbation=True, \
                                   small_scale_perturbation_amplitude=self.q0)
        

        self.ensemble = OceanNoiseEnsemble.OceanNoiseEnsemble(self.gpu_ctx, self.ensembleSize, self.sim,
                                                              num_drifters =self.driftersPerOceanModel,
                                                              observation_type=dautils.ObservationType.DirectUnderlyingFlow,
                                                              observation_variance = 0.01**2)


        self.iewpf = IEWPFOcean.IEWPFOcean(self.ensemble)


    def run_ensemble(self):
        t = self.ensemble.step(1000*self.dt)

        



    def test_iewpf_attributes(self):

        self.assertEqual(self.iewpf.nx, self.nx)
        self.assertEqual(self.iewpf.ny, self.ny)
        self.assertAlmostEqual(self.iewpf.dx, self.dx)
        self.assertAlmostEqual(self.iewpf.dy, self.dy)
        self.assertAlmostEqual(self.iewpf.soar_q0, self.q0)
        self.assertAlmostEqual(self.iewpf.soar_L, self.sim.small_scale_model_error.soar_L)
        self.assertAlmostEqual(self.iewpf.f, self.f)
        self.assertAlmostEqual(self.iewpf.g, self.g, places=6)
        self.assertAlmostEqual(self.iewpf.const_H, self.waterDepth)
        self.assertTrue(self.iewpf.boundaryConditions.isPeriodicNorthSouth())
        self.assertTrue(self.iewpf.boundaryConditions.isPeriodicEastWest())
        self.assertTrue(self.iewpf.Nx, self.nx*self.ny*3)
        self.assertEqual(self.iewpf.numParticles, self.ensembleSize)
        self.assertEqual(self.iewpf.numDrifters, self.driftersPerOceanModel)
        
    def test_S_matrix(self):
        S_from_GPU = self.iewpf.download_S()
        S_from_file = np.loadtxt("iewpfRefData/S.dat")

        self.assertEqual(self.iewpf.S_host.shape, (2,2))
        self.assertEqual(S_from_GPU.shape, (2,2))

        assert2DListAlmostEqual(self, S_from_GPU.tolist(), self.iewpf.S_host.tolist(), 7, "S matrix GPU vs CPU")
        assert2DListAlmostEqual(self, S_from_GPU.tolist(), S_from_file.tolist(), 7, "S matrix GPU vs file")
        

    def test_localSVD_matrix(self):
        localSVD_from_GPU = self.iewpf.download_localSVD()
        localSVD_from_file = np.loadtxt("iewpfRefData/localSVD.dat")

        self.assertEqual(self.iewpf.localSVD_host.shape, (49, 49))
        self.assertEqual(localSVD_from_GPU.shape, (49, 49))

        assert2DListAlmostEqual(self, localSVD_from_GPU.tolist(), self.iewpf.localSVD_host.tolist(), 7, "SVD matrix GPU vs CPU")
        
        # Compare the full matrix (not the matrix square root) to the reference solution,
        # since the matrix square root is (in general) not unique.
        full_matrix_gpu = np.dot(localSVD_from_GPU, localSVD_from_GPU.transpose())
        full_matrix_reference = np.dot(localSVD_from_file, localSVD_from_file.transpose())
        assert2DListAlmostEqual(self, full_matrix_gpu.tolist(), full_matrix_reference.tolist(), 5, "Full SVD matrix GPU vs file")

        
    def test_local_SVD_to_global_GPU_ref_data(self):
        
        ### Since the matrix square root is not in general unique, the local SVD buffer is not
        # always the same, even though it is still just as valid.
        # A consequence is therefore that the application of the local SVD to a ocean state will not compare to 
        # a reference solution. 
        # To test this functionality, we therefore upload the reference SVD block before applying the block and 
        # comparing the results
        
        # Alias for the particle we use.
        # We will always use sim's gpu_stream throughout this test.
        sim = self.ensemble.particles[0]
        
        # Upload refrence SVD to the GPU
        localSVD_from_file = np.loadtxt("iewpfRefData/localSVD.dat")
        self.iewpf.localSVD_device.upload(sim.gpu_stream, localSVD_from_file.astype(np.float32))
        
        test_data = np.loadtxt("iewpfRefData/preLocalSVDtoGlobal.dat")
        results_from_file = np.loadtxt("iewpfRefData/postLocalSVDtoGlobal.dat")

        self.assertEqual(test_data.shape, (self.ny, self.nx))

        # Upload reference input data to the sim random numbers buffer:
        sim.small_scale_model_error.random_numbers.upload(sim.gpu_stream,
                                                          test_data.astype(np.float32))
        
        # Apply SVD centered in the chosen cell (30, 30):
        self.iewpf.applyLocalSVDOnGlobalXi(sim, 30, 30)
        
        # Download results:
        gpu_result = sim.small_scale_model_error.random_numbers.download(sim.gpu_stream)

        # Compare:
        rel_norm_results = np.linalg.norm(gpu_result - results_from_file)/np.max(results_from_file)
        self.assertAlmostEqual(rel_norm_results, 0.0, places=6)
        
    def test_local_SVD_to_global_CPU_ref_data(self):
    
        # See comment for test_local_SVD_to_global_GPU_ref_data(self).
        
        # Set refrence SVD to the SVD block for the host
        localSVD_from_file = np.loadtxt("iewpfRefData/localSVD.dat")
        self.iewpf.localSVD_host = localSVD_from_file
        
        test_data = np.loadtxt("iewpfRefData/preLocalSVDtoGlobal.dat")
        results_from_file = np.loadtxt("iewpfRefData/postLocalSVDtoGlobal.dat")

        self.assertEqual(test_data.shape, (self.ny, self.nx))
        
        self.iewpf._apply_local_SVD_to_global_xi_CPU(test_data, 30, 30)

        assert2DListAlmostEqual(self, test_data.tolist(), results_from_file.tolist(), 10, "test_local_SVD_to_global_CPU")
    
        

    def test_set_buffer_to_zero(self):
        self.ensemble.particles[0].small_scale_model_error.generateNormalDistribution()
        
        self.iewpf.setNoiseBufferToZero(self.ensemble.particles[0])
        
        obtained_random_numbers = self.ensemble.particles[0].small_scale_model_error.getRandomNumbers()
        for j in range(self.iewpf.coarse_ny):
            for i in range(self.iewpf.coarse_nx):
                self.assertEqual(obtained_random_numbers[j,i], 0.0)
    
    def test_blas_xaxpby(self):
        self.iewpf.samplePerpendicular(self.ensemble.particles[0])
        x = self.ensemble.particles[0].small_scale_model_error.getRandomNumbers()
        y = self.ensemble.particles[0].small_scale_model_error.getPerpendicularRandomNumbers()
        alpha = 2.12
        beta  = 5.1
        
        self.iewpf.addBetaNuIntoAlphaXi(self.ensemble.particles[0], alpha, beta)
        x_res_gpu = self.ensemble.particles[0].small_scale_model_error.getRandomNumbers()
        x_res_cpu = np.sqrt(alpha)*x + np.sqrt(beta)*y
        
        assert2DListAlmostEqual(self, x_res_gpu.tolist(), x_res_gpu.tolist(), 10, "test_blas_xaxpby")
        
        
    def test_apply_SVD_to_Perpendicular(self):
        # Set sim variables to zero, so that only the Kalman gain ends up in those fields
        zeros = np.zeros((self.iewpf.ny+4, self.iewpf.nx+4), dtype=np.float32)
        self.ensemble.particles[0].gpu_data.h0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hu0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hv0.upload(self.iewpf.master_stream, zeros)
        
        self.iewpf.samplePerpendicular(self.ensemble.particles[0])
        xi = self.ensemble.particles[0].small_scale_model_error.getRandomNumbers()
        nu = self.ensemble.particles[0].small_scale_model_error.getPerpendicularRandomNumbers()
        alpha = np.float32(9.0)
        beta = np.float32(4.0)
        drifter_positions = np.array([[120, 120], [40, 120], [80, 40]], dtype=np.float32)
        
        # Sample from N(0,P) the old way
        
        # Set sim variables to zero, so that only the Kalman gain ends up in those fields
        zeros = np.zeros((self.iewpf.ny+4, self.iewpf.nx+4), dtype=np.float32)
        self.ensemble.particles[0].gpu_data.h0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hu0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hv0.upload(self.iewpf.master_stream, zeros)
       
        # Old SVD
        self.iewpf.applySVDtoPerpendicular_slow(self.ensemble.particles[0], drifter_positions)
        Pxi = self.ensemble.particles[0].small_scale_model_error.getRandomNumbers()
        Pnu = self.ensemble.particles[0].small_scale_model_error.getPerpendicularRandomNumbers()
        aPxibPnu = np.sqrt(alpha)*Pxi + np.sqrt(beta)*Pnu

        # Old Q
        self.ensemble.particles[0].small_scale_model_error.perturbSim(self.ensemble.particles[0],\
                                                                      update_random_field=False, \
                                                                      perturbation_scale=np.sqrt(alpha),
                                                                      perpendicular_scale=np.sqrt(beta))
        old_eta, old_hu, old_hv = self.ensemble.particles[0].download(interior_domain_only=True)

        # Sample from N(0,P) the new way
       
        # Set sim variables to zero, so that only the Kalman gain ends up in those fields
        zeros = np.zeros((self.iewpf.ny+4, self.iewpf.nx+4), dtype=np.float32)
        self.ensemble.particles[0].gpu_data.h0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hu0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hv0.upload(self.iewpf.master_stream, zeros)
       
        # New SVD
        self.ensemble.particles[0].small_scale_model_error.random_numbers.upload(self.iewpf.master_stream, xi)
        self.ensemble.particles[0].small_scale_model_error.perpendicular_random_numbers.upload(self.iewpf.master_stream, nu)
        self.iewpf.applySVDtoPerpendicular(self.ensemble.particles[0], drifter_positions, alpha, beta)
        Paxibnu = self.ensemble.particles[0].small_scale_model_error.getRandomNumbers()
        
        # New Q
        self.ensemble.particles[0].small_scale_model_error.perturbSim(self.ensemble.particles[0],\
                                                                      update_random_field=False)
        new_eta, new_hu, new_hv = self.ensemble.particles[0].download(interior_domain_only=True)

        # Compare results
        assert2DListAlmostEqual(self, Paxibnu.tolist(), aPxibPnu.tolist(), 4, "test_apply_SVD_to_Perpendicular SVD")
        assert2DListAlmostEqual(self, new_eta.tolist(), old_eta.tolist(),  4, "test_apply_SVD_to_Perpendicular eta")
        assert2DListAlmostEqual(self, new_hu.tolist(),  old_hu.tolist(),   4, "test_apply_SVD_to_Perpendicular hu")
        assert2DListAlmostEqual(self, new_hv.tolist(),  old_hv.tolist(),   4, "test_apply_SVD_to_Perpendicular hv")
      
        
        
    def test_kalman_gain(self):
        self.run_ensemble()
        innovation = self.ensemble.getInnovations()[0]
        observed_drifter_positions = self.ensemble.observeTrueDrifters()

        # Set sim variables to zero, so that only the Kalman gain ends up in those fields
        zeros = np.zeros((self.iewpf.ny+4, self.iewpf.nx+4), dtype=np.float32)
        self.ensemble.particles[0].gpu_data.h0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hu0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hv0.upload(self.iewpf.master_stream, zeros)

        self.iewpf.addKalmanGain(self.ensemble.particles[0],
                                 observed_drifter_positions,
                                 innovation)

        eta, hu, hv = self.ensemble.particles[0].download(interior_domain_only=True)

        etaCPU, huCPU, hvCPU, gamma = self.iewpf.applyKalmanGain_CPU(self.ensemble.particles[0],
                                                                     observed_drifter_positions,
                                                                     innovation, returnKalmanGainTerm=True)
        rel_norm_eta = np.linalg.norm(eta - etaCPU)/np.max(eta)
        rel_norm_hu  = np.linalg.norm(hu - huCPU)/np.max(hu)
        rel_norm_hv  = np.linalg.norm(hv - hvCPU)/np.max(hv)
       
        self.assertAlmostEqual(rel_norm_eta, 0.0, places=5)
        self.assertAlmostEqual(rel_norm_hu,  0.0, places=5)
        self.assertAlmostEqual(rel_norm_hv,  0.0, places=5)
        
    def test_observation_operator_CPU_vs_GPU(self):
        self.run_ensemble()
                      
        cpu = self.ensemble.observeParticles(gpu=False)
        gpu = self.ensemble.observeParticles(gpu=True)
                
        assert2DListAlmostEqual(self, cpu.tolist(), gpu.tolist(), 6, "observation_operator_CPU_vs_GPU")
