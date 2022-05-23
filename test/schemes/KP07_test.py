# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements regression tests for the KP07 scheme.

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
import os
import gc

from testUtils import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')))

from SWESimulators import Common, KP07

class KP07test(unittest.TestCase):

    def setUp(self):
        self.gpu_ctx = Common.CUDAContext()
        
        self.nx = 50
        self.ny = 70
        
        self.dx = 200.0
        self.dy = 200.0
        
        self.dt = 0.95
        self.g = 9.81
        self.f = 0.0
        self.r = 0.0
        self.A = 1
        
        #self.h0 = np.ones((self.ny+2, self.nx+2), dtype=np.float32) * 60;
        self.waterHeight = 60
        self.eta0 = None
        self.u0 = None
        self.v0 = None
        self.Hi = None
        self.Bi = None

        self.ghosts = [2,2,2,2] # north, east, south, west
        self.validDomain = np.array([2,2,2,2])
        self.refRange = [-2, -2, 2, 2]
        self.dataRange = self.refRange
        self.boundaryConditions = None

        self.T = 50.0
        self.sim = None
        
    def tearDown(self):
        if self.sim != None:
            self.sim.cleanUp()
            self.sim = None
        self.eta0 = None
        self.u0 = None
        self.v0 = None
        self.Hi = None
        self.Bi = None

        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
        
        gc.collect() # Force run garbage collection to free up memory
        

    def allocData(self):
        dataShape = (self.ny + self.ghosts[0]+self.ghosts[2], 
                     self.nx + self.ghosts[1]+self.ghosts[3])
        self.eta0 = np.zeros(dataShape, dtype=np.float32);
        self.u0 = np.zeros(dataShape, dtype=np.float32)
        self.v0 = np.zeros(dataShape, dtype=np.float32)
        self.Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * self.waterHeight
        self.Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')


    def setBoundaryConditions(self, bcSettings=1):

        if (bcSettings == 1):
            self.boundaryConditions = Common.BoundaryConditions()
        elif (bcSettings == 2):
            self.boundaryConditions = Common.BoundaryConditions(2,2,2,2)            
        elif bcSettings == 3:
            # Periodic NS
            self.boundaryConditions = Common.BoundaryConditions(2,1,2,1)
        else:
            # Periodic EW
            self.boundaryConditions = Common.BoundaryConditions(1,2,1,2)

        
    def checkResults(self, eta1, u1, v1, etaRef, uRef, vRef, message=""):
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
                                self.refRange[3]:self.refRange[1]]) /np.max(np.abs(vRef))

        self.assertAlmostEqual(maxDiffEta, 0.0, places=4,
                               msg='Unexpected eta difference! Max diff: ' + str(maxDiffEta) + ', L2 rel diff: ' + str(diffEta) + message)
        self.assertAlmostEqual(maxDiffU, 0.0, places=4,
                               msg='Unexpected U difference: ' + str(maxDiffU) + ', L2 rel diff: ' + str(diffU) + message)
        self.assertAlmostEqual(maxDiffV, 0.0, places=4,
                               msg='Unexpected V difference: ' + str(maxDiffV) + ', L2 rel diff: ' + str(diffV) + message)

        
    def checkLakeAtRest(self, eta, u, v, message=""):
        etaMinMax = [np.min(eta), np.max(eta)]
        uMinMax = [np.min(u), np.max(u)]
        vMinMax = [np.min(v), np.max(v)]

        self.assertAlmostEqual(etaMinMax[0], 0.0, places=3,
                         msg='Non-constant water level: ' + str(etaMinMax) + message)
        self.assertAlmostEqual(etaMinMax[1], 0.0, places=3,
                         msg='Non-constant water level: ' + str(etaMinMax) + message)
        
        self.assertAlmostEqual(uMinMax[0], 0.0, places=3,
                         msg='Movement in water (u): ' + str(uMinMax) + message)
        self.assertAlmostEqual(uMinMax[1], 0.0, places=3,
                         msg='Movement in water (u): ' + str(uMinMax) + message)
        
        self.assertAlmostEqual(vMinMax[0], 0.0, places=3,
                         msg='Movement in water (v): ' + str(vMinMax) + message)
        self.assertAlmostEqual(vMinMax[1], 0.0, places=3,
                         msg='Movement in water (v): ' + str(vMinMax) + message)
                         
    ## Wall boundary conditions
    
    def test_wall_central(self):
        self.setBoundaryConditions()
        self.allocData()
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    # Skipping the following test:
    def notest_wall_central_with_nonzero_flat_bottom(self):
        self.setBoundaryConditions()
        self.allocData()
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        extraBottom = 10.0
        self.Hi = self.Hi - extraBottom
        self.eta0 = self.eta0 + extraBottom
        
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta1 = eta1 - extraBottom
        eta2, u2, v2 = loadResults("KP07", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2, message="\nKNOWN TO FAIL with value    eta difference! Max diff: 1.52587890625e-05, L2 diff: 0.000251422989705...")

 
    def test_wall_corner(self):
        self.setBoundaryConditions()
        self.allocData()
        addCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "wallBC", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_wall_upperCorner(self):
        self.setBoundaryConditions()
        self.allocData()
        addUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "wallBC", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

## Test lake at rest cases
    def test_lake_at_rest_flat_bottom(self):
        self.setBoundaryConditions()
        self.allocData()
        self.Hi = self.Hi+10.0
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta, u, v = self.sim.download()
        self.checkLakeAtRest(eta, u, v)


    def test_lake_at_rest_crater_bottom(self):
        self.setBoundaryConditions()
        self.allocData()
        makeBathymetryCrater(self.Bi, self.nx+1, self.ny+1, self.dx, self.dy, self.ghosts)
        self.Hi += self.Bi
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta, u, v = self.sim.download()
        self.checkLakeAtRest(eta, u, v)        


    def test_lake_at_rest_crazy_bottom(self):
        self.setBoundaryConditions()
        self.allocData()
        makeBathymetryCrazyness(self.Bi, self.nx+1, self.ny+1, self.dx, self.dy, self.ghosts)
        self.Hi += self.Bi
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta, u, v = self.sim.download()
        self.checkLakeAtRest(eta, u, v)
        
## Full periodic boundary conditions

    def test_periodic_central(self):
        self.setBoundaryConditions(bcSettings=2)
        self.allocData()
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "wallBC", "central")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)


    def test_periodic_corner(self):
        self.setBoundaryConditions(bcSettings=2)
        self.allocData()
        addCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "periodic", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_periodic_upperCorner(self):
        self.setBoundaryConditions(bcSettings=2)
        self.allocData()
        addUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "periodic", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


## North-south periodic boundary conditions

    def test_periodicNS_central(self):
        self.setBoundaryConditions(bcSettings=3)
        self.allocData()
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "wallBC", "central")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)

        
    def test_periodicNS_corner(self):
        self.setBoundaryConditions(bcSettings=3)
        self.allocData()
        addCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "periodicNS", "corner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)


        
    def test_periodicNS_upperCorner(self):
        self.setBoundaryConditions(bcSettings=3)
        self.allocData()
        addUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "periodicNS", "upperCorner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)

 ## East-west periodic boundary conditions

    def test_periodicEW_central(self):
        self.setBoundaryConditions(bcSettings=4)
        self.allocData()
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "wallBC", "central")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)       


    def test_periodicEW_corner(self):
        self.setBoundaryConditions(bcSettings=4)
        self.allocData()
        addCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "periodicEW", "corner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)       

    def test_periodicEW_upperCorner(self):
        self.setBoundaryConditions(bcSettings=4)
        self.allocData()
        addUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "periodicEW", "upperCorner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)       

  
    def test_coriolis_central(self):
        self.setBoundaryConditions()
        self.allocData()
        self.f = 0.01
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = KP07.KP07(self.gpu_ctx, \
                    self.eta0, self.Hi, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("KP07", "coriolis", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
