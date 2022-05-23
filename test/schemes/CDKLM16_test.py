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
import time
import numpy as np
import sys
import os
import gc

from testUtils import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')))

from SWESimulators import Common, CDKLM16


class CDKLM16test(unittest.TestCase):

    def setUp(self):
        self.gpu_ctx = Common.CUDAContext()

        self.nx = 50
        self.ny = 70
        
        self.dx = 200.0
        self.dy = 200.0
        
        self.dt = 0.9
        self.g = 9.81
        self.f = 0.0
        self.r = 0.0
        self.A = 1
        
        self.waterHeight = 60
        self.eta0 = None
        self.u0 = None
        self.v0 = None
        self.Hi = None
        
        self.ghosts = [2,2,2,2] # north, east, south, west
        self.validDomain = np.array([2,2,2,2])
        self.dataRange = [-2, -2, 2, 2]
        self.refRange = self.dataRange
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
                               msg='Unexpected U relative difference: ' + str(maxDiffU) + ', L2 rel diff: ' + str(diffU))
        self.assertAlmostEqual(maxDiffV, 0.0, places=3,
                               msg='Unexpected V relative difference: ' + str(maxDiffV) + ', L2 rel diff: ' + str(diffV))
    
        # Test maximal time step:
        self.sim.updateDt()
        dt_host = self.sim._getMaxTimestepHost()
        self.assertEqual(self.sim.dt, dt_host, msg="maximum time step")
    
    ## Wall boundary conditions
    
    def test_wall_central(self):
        self.setBoundaryConditions()
        self.allocData()
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


 
    def test_wall_corner(self):
        self.setBoundaryConditions()
        self.allocData()
        addCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_wall_upperCorner(self):
        self.setBoundaryConditions()
        self.allocData()
        addUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


## Full periodic boundary conditions

    def test_periodic_central(self):
        self.setBoundaryConditions(bcSettings=2)
        self.allocData()
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)


    def test_periodic_corner(self):
        self.setBoundaryConditions(bcSettings=2)
        self.allocData()
        addCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "periodic", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_periodic_upperCorner(self):
        self.setBoundaryConditions(bcSettings=2)
        self.allocData()
        addUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "periodic", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


## North-south periodic boundary conditions

    def test_periodicNS_central(self):
        self.setBoundaryConditions(bcSettings=3)
        self.allocData()
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)

        
    def test_periodicNS_corner(self):
        self.setBoundaryConditions(bcSettings=3)
        self.allocData()
        addCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "periodicNS", "corner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)
        #print "\nHvorfor gaar dette bra???"
        #print "self.refRange:  ", self.refRange
        #print "self.dataRange: ", self.dataRange
        

        
    def test_periodicNS_upperCorner(self):
        self.setBoundaryConditions(bcSettings=3)
        self.allocData()
        addUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "periodicNS", "upperCorner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)

 ## East-west periodic boundary conditions

    def test_periodicEW_central(self):
        self.setBoundaryConditions(bcSettings=4)
        self.allocData()
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)       


    def test_periodicEW_corner(self):
        self.setBoundaryConditions(bcSettings=4)
        self.allocData()
        addCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "periodicEW", "corner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)       

    def test_periodicEW_upperCorner(self):
        self.setBoundaryConditions(bcSettings=4)
        self.allocData()
        addUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "periodicEW", "upperCorner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)       

  
    def test_coriolis_central(self):
        self.setBoundaryConditions()
        self.allocData()
        self.f = 0.01
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "coriolis", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


    def test_betamodel_central(self):
        self.setBoundaryConditions()
        self.allocData()
        
        # Defining coriolis parameters and accounting for ghost cells:
        beta = 1e-6
        self.f = 0.01 - 2*beta*self.dy
        
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, coriolis_beta=beta)
    #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "betamodel", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

        
    def test_bathymetry_central(self):
        self.setBoundaryConditions()
        self.allocData() 
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        makeBottomTopography(self.Hi, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central", "bathymetry_")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
       
