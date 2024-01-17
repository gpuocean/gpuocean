# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements regression tests for the CDKLM16 scheme.

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
import os
import gc

from testUtils import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')))

from gpuocean.SWEsimulators import CDKLM16, CombinedCDKLM16
from gpuocean.utils import Common


class CombinedCDKLM16test(unittest.TestCase):

    def setUp(self):
        self.gpu_ctx_in = Common.CUDAContext()
        self.gpu_ctx_out = Common.CUDAContext()

        self.nx = 50
        self.ny = 70
        
        self.dx = 200.0
        self.dy = 200.0
        
        self.dt = 0.0
        self.g = 9.81
        self.f = 0.0
        self.r = 0.0
    
        self.ghosts = [2,2,2,2] # north, east, south, west
        self.dataShape = (self.ny + self.ghosts[0]+self.ghosts[2], 
                     self.nx + self.ghosts[1]+self.ghosts[3])

        self.T = 600.0

        self.dataRange = [-2, -2, 2, 2]
        self.refRange = self.dataRange

        
    def tearDown(self):
        gc.collect() # Force run garbage collection to free up memory
        

    def test_combined_vs_single_sims(self):
        """
        Test case: negative or positive wave going in or out, respectively, simulated individually and with combined time stepping
        """
        
        eta0_in = np.zeros(self.dataShape, dtype=np.float32)
        hu0_in  = np.zeros(self.dataShape, dtype=np.float32)
        hv0_in  = np.zeros(self.dataShape, dtype=np.float32)

        eta0_out = np.zeros(self.dataShape, dtype=np.float32)
        hu0_out  = np.zeros(self.dataShape, dtype=np.float32)
        hv0_out  = np.zeros(self.dataShape, dtype=np.float32)

        Hi_in  = np.ones((self.dataShape[0]+1, self.dataShape[1]+1), dtype=np.float32, order='C') * 10
        Hi_out = np.ones((self.dataShape[0]+1, self.dataShape[1]+1), dtype=np.float32, order='C') * 10

        bc_in = Common.BoundaryConditions(3,3,3,3, spongeCells={'north':10, 'south': 10, 'east': 10, 'west': 10})
        bc_data_in  = Common.BoundaryConditionsData()
        bc_data_in.north.h = [np.array([[1,1]], dtype=np.float32)]

        bc_out = Common.BoundaryConditions(3,3,3,3, spongeCells={'north':10, 'south': 10, 'east': 10, 'west': 10})
        bc_data_out = Common.BoundaryConditionsData()
        bc_data_out.north.h = [np.array([[-1,-1]], dtype=np.float32)]


        # Individual sim with in-coming wave
        sim_in = CDKLM16.CDKLM16(self.gpu_ctx_in, eta0_in, hu0_in, hv0_in, Hi_in,\
                                    self.nx, self.ny, self.dx, self.dy, 0.0, self.g, self.f, self.r, \
                                    boundary_conditions=bc_in, boundary_conditions_data=bc_data_in)
        
        sim_in.step(self.T)

        eta_in, hu_in, hv_in = sim_in.download(interior_domain_only=True)

        # Individual sim with in-coming wave
        sim_out = CDKLM16.CDKLM16(self.gpu_ctx_out, eta0_out, hu0_out, hv0_out, Hi_out,\
                                    self.nx, self.ny, self.dx, self.dy, 0.0, self.g, self.f, self.r, \
                                    boundary_conditions=bc_out, boundary_conditions_data=bc_data_out)
        
        sim_out.step(self.T)

        eta_out, hu_out, hv_out = sim_out.download(interior_domain_only=True)

        # Combined sims 
        sim_in2 = CDKLM16.CDKLM16(self.gpu_ctx_in, eta0_in, hu0_in, hv0_in, Hi_in,\
                                    self.nx, self.ny, self.dx, self.dy, 0.0, self.g, self.f, self.r, \
                                    boundary_conditions=bc_in, boundary_conditions_data=bc_data_in)
        sim_out2 = CDKLM16.CDKLM16(self.gpu_ctx_out, eta0_out, hu0_out, hv0_out, Hi_out,\
                                    self.nx, self.ny, self.dx, self.dy, 0.0, self.g, self.f, self.r, \
                                    boundary_conditions=bc_out, boundary_conditions_data=bc_data_out)
        
        sims = CombinedCDKLM16.CombinedCDKLM16(sim_in2, sim_out2)                                        
        sims.combinedStep(self.T)

        eta_in2, hu_in2, hv_in2 = sims.barotropic_sim.download(interior_domain_only=True)
        eta_out2, hu_out2, hv_out2 = sims.baroclinic_sim.download(interior_domain_only=True)
                
        # Check 
        self.checkResults(eta_in, hu_in, hv_in, eta_in2, hu_in2, hv_in2)
        self.checkResults(eta_out, hu_out, hv_out, eta_out2, hu_out2, hv_out2)


        
    def checkResults(self, eta1, u1, v1, etaRef, uRef, vRef):
        diffEta = np.linalg.norm(eta1[self.dataRange[2]:self.dataRange[0], 
                                      self.dataRange[3]:self.dataRange[1]] - 
                                 etaRef[self.refRange[2]:self.refRange[0],
                                        self.refRange[3]:self.refRange[1]]) / np.max(np.abs(etaRef))
        diffU = np.linalg.norm(u1[self.dataRange[2]:self.dataRange[0],
                                  self.dataRange[3]:self.dataRange[1]] -
                               uRef[self.refRange[2]:self.refRange[0],
                                    self.refRange[3]:self.refRange[1]]) / np.max(np.abs(uRef))
        diffV = np.linalg.norm(v1[self.dataRange[2]:self.dataRange[0],
                                  self.dataRange[3]:self.dataRange[1]] - 
                               vRef[ self.refRange[2]:self.refRange[0],
                                     self.refRange[3]:self.refRange[1]]) / np.max(np.abs(vRef))
        maxDiffEta = np.max(eta1[self.dataRange[2]:self.dataRange[0], 
                                 self.dataRange[3]:self.dataRange[1]] - 
                            etaRef[self.refRange[2]:self.refRange[0],
                                   self.refRange[3]:self.refRange[1]]) / np.max(np.abs(etaRef))
        maxDiffU = np.max(u1[self.dataRange[2]:self.dataRange[0],
                             self.dataRange[3]:self.dataRange[1]] -
                          uRef[self.refRange[2]:self.refRange[0],
                               self.refRange[3]:self.refRange[1]]) / np.max(np.abs(uRef))
        maxDiffV = np.max(v1[self.dataRange[2]:self.dataRange[0],
                             self.dataRange[3]:self.dataRange[1]] - 
                          vRef[ self.refRange[2]:self.refRange[0],
                                self.refRange[3]:self.refRange[1]]) / np.max(np.abs(vRef))
        
        self.assertAlmostEqual(maxDiffEta, 0.0, places=3,
                               msg='Unexpected eta difference! Max rel diff: ' + str(maxDiffEta) + ', L2 rel diff: ' + str(diffEta))
        self.assertAlmostEqual(maxDiffU, 0.0, places=3,
                               msg='Unexpected hu relative difference: ' + str(maxDiffU) + ', L2 rel diff: ' + str(diffU))
        self.assertAlmostEqual(maxDiffV, 0.0, places=3,
                               msg='Unexpected hv relative difference: ' + str(maxDiffV) + ', L2 rel diff: ' + str(diffV))
    
    