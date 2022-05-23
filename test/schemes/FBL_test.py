# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital

This python module implements regression tests for the FBL scheme.

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
import os

from testUtils import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')))

from SWESimulators import Common, FBL

class FBLtest(unittest.TestCase):

    def setUp(self):
        self.gpu_ctx = Common.CUDAContext()

        self.nx = 50
        self.ny = 70
        
        self.dx = 200.0
        self.dy = 200.0
        
        self.dt = 1
        self.g = 9.81
        self.f = 0.0
        self.r = 0.0

        self.h0 = None #np.ones((self.ny, self.nx), dtype=np.float32) * 60;
        self.eta0 = None # np.zeros((self.ny, self.nx), dtype=np.float32);
        self.u0 = None # np.zeros((self.ny, self.nx+1), dtype=np.float32);
        self.v0 = None # np.zeros((self.ny+1, self.nx), dtype=np.float32);

        self.T = 50.0

        self.boundaryConditions = None #Common.BoundaryConditions()
        self.ghosts = [1, 1, 1, 1]
        self.arrayRange = None

        self.sim = None
        
    def tearDown(self):
        if self.sim != None:
            self.sim.cleanUp()
            self.sim = None
        self.h0 = None
        self.eta0 = None 
        self.u0 = None 
        self.v0 = None
        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
        gc.collect() # Force run garbage collection to free up memory
        
    def setBoundaryConditions(self, bcSettings=1):
        if (bcSettings == 1):
            self.boundaryConditions = Common.BoundaryConditions()
            #self.ghosts = [0,0,0,0] # north, east, south, west
            self.arrayRange = [None, None, 0, 0]
        elif (bcSettings == 2):
            self.boundaryConditions = Common.BoundaryConditions(2,2,2,2)
            #self.ghosts = [1,1,0,0] # Both periodic
            self.arrayRange = [-1, -1, 0, 0]
        elif bcSettings == 3:
            self.boundaryConditions = Common.BoundaryConditions(2,1,2,1)
            #self.ghosts = [1,0,0,0] # periodic north-south
            self.arrayRange = [-1, None, 0, 0]
        else:
            self.boundaryConditions = Common.BoundaryConditions(1,2,1,2)
            #self.ghosts = [0,1,0,0] # periodic east-west
            self.arrayRange = [None, -1, 0, 0]

    def createHostData(self):
        dataShape = (self.ny + self.ghosts[0] + self.ghosts[2], 
                     self.nx + self.ghosts[1] + self.ghosts[2])
    

        self.h0 = np.ones(dataShape, dtype=np.float32) * 60;
        self.eta0 = np.zeros(dataShape, dtype=np.float32);
        self.u0 = np.zeros((dataShape[0]  , dataShape[1]-1), dtype=np.float32, order='C');
        self.v0 = np.zeros((dataShape[0]+1, dataShape[1]  ), dtype=np.float32, order='C');
        
    def checkResults(self, eta1, u1, v1, etaRef, uRef, vRef, refRange=None):
        if refRange is None:
            diffEta = np.linalg.norm(eta1 - etaRef)
            diffU = np.linalg.norm(u1-uRef)
            diffV = np.linalg.norm(v1-vRef)
            maxDiffEta = np.max(eta1 - etaRef)
            maxDiffU = np.max(u1-uRef)
            maxDiffV = np.max(v1-vRef)
            
        else:
            diffEta = np.linalg.norm(eta1 -etaRef[refRange[2]:refRange[0],
                                                  refRange[3]:refRange[1]])
            diffU = np.linalg.norm(u1 - uRef[refRange[2]:refRange[0], :])
            diffV = np.linalg.norm(v1 - vRef[:, refRange[3]:refRange[1]])
            maxDiffEta = np.max(eta1 - etaRef[refRange[2]:refRange[0],
                                              refRange[3]:refRange[1]])
            maxDiffU = np.max(u1 - uRef[refRange[2]:refRange[0], :])
            maxDiffV = np.max(v1 - vRef[:, refRange[3]:refRange[1]])
            
        
        self.assertAlmostEqual(maxDiffEta, 0.0, places=5,
                               msg='Unexpected eta difference! Max diff: ' + str(maxDiffEta) + ', L2 diff: ' + str(diffEta))
        self.assertAlmostEqual(maxDiffU, 0.0, places=5,
                               msg='Unexpected U difference: ' + str(maxDiffU) + ', L2 diff: ' + str(diffU))
        self.assertAlmostEqual(maxDiffV, 0.0, places=5,
                               msg='Unexpected V difference: ' + str(maxDiffV) + ', L2 diff: ' + str(diffV))
        


    def test_wall_central(self):
        self.setBoundaryConditions(1)
        self.createHostData()
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy,
                        self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_wall_corner(self):
        self.setBoundaryConditions(1)
        self.createHostData()
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "wallBC", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
        
    def test_wall_upperCorner(self):
        self.setBoundaryConditions(1)
        self.createHostData()
        makeUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "wallBC", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


    def test_periodic_central(self):
        self.setBoundaryConditions(2)
        self.createHostData()
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, \
                           boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
        

    def test_periodic_corner(self):
        self.setBoundaryConditions(2)
        self.createHostData()
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, \
                           boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "periodicAll", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2, self.arrayRange)
        
    def test_periodic_upperCorner(self):
        self.setBoundaryConditions(2)
        self.createHostData()
        makeUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, \
                           boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "periodicAll", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
  
    def test_periodicNS_central(self):
        self.setBoundaryConditions(3)
        self.createHostData()
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, \
                           boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
        

    def test_periodicNS_corner(self):
        self.setBoundaryConditions(3)
        self.createHostData()
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, \
                           boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "periodicNS", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2, self.arrayRange)
        
        
    def test_periodicNS_upperCorner(self):
        self.setBoundaryConditions(3)
        self.createHostData()
        makeUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, \
                           boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "periodicNS", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_periodicEW_central(self):
        self.setBoundaryConditions(4)
        self.createHostData()
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, \
                           boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
        

    def test_periodicEW_corner(self):
        self.setBoundaryConditions(4)
        self.createHostData()
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, \
                           boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "periodicEW", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2, self.arrayRange)
        
    def test_periodicEW_upperCorner(self):
        self.setBoundaryConditions(4)
        self.createHostData()
        makeUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, \
                           boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "periodicEW", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
        
    def test_coriolis_central(self):
        self.setBoundaryConditions(1)
        self.createHostData()
        self.f = 0.01
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy,
                        self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "coriolis", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_betamodel_central(self):
        self.setBoundaryConditions(1)
        self.createHostData()
        self.f = 0.01
        beta = 1e-6
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy,
                        self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r, coriolis_beta=beta)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "betamodel", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


    def test_bathymetry_central(self):
        self.setBoundaryConditions(1)
        self.createHostData()
        makeBottomTopography(self.h0, self.nx, self.ny, self.dx, self.dy, self.ghosts, intersections=False)
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy,
                        self.ghosts)
        self.sim = FBL.FBL(self.gpu_ctx, \
                           self.h0, self.eta0, self.u0, self.v0, \
                           self.nx, self.ny, \
                           self.dx, self.dy, self.dt, \
                           self.g, self.f, self.r)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download(interior_domain_only=True)
        eta2, u2, v2 = loadResults("FBL", "wallBC", "central", "bathymetry_")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
