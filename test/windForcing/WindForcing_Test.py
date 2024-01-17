# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2022 SINTEF Digital

This python module implements tests related to wind stress onto drifters

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

from gpuocean.SWEsimulators import CDKLM16
from gpuocean.drifters import GPUDrifterCollection
from gpuocean.utils import Common, WindStress

class WindForcingTest(unittest.TestCase):
    def setUp(self):
        self.N_winds = 12

        self.gpu_ctx = Common.CUDAContext()

        self.gpu_ctx_winds = []
        for i in range(self.N_winds):
            # Generating new contextes without iPythonMagic requires to reset the kernel every time it crashes 
            self.gpu_ctx_winds.append( Common.CUDAContext() )


        self.nx = 100
        self.ny = 100

        self.dx = 100.0
        self.dy = 100.0

        self.dt = 0.0
        self.g = 9.81
        self.f = 0.0
        self.r = 0.0

        self.ghosts = [2,2,2,2] # north, east, south, west

        self.dataShape = (self.ny + self.ghosts[0] + self.ghosts[2], self.nx + self.ghosts[1] + self.ghosts[3])

        self.eta0 = np.zeros(self.dataShape, dtype=np.float32)
        self.hu0 = np.zeros(self.dataShape, dtype=np.float32)
        self.hv0 = np.zeros(self.dataShape, dtype=np.float32)
        self.Hi = 10 * np.ones((self.dataShape[0]+1, self.dataShape[1]+1), dtype=np.float32, order='C')

        self.boundary_conditions = Common.BoundaryConditions(2,2,2,2)

        self.T = 600

        wind_t = np.array([0])
        wind_u = [np.array([[30]])]
        wind_v = [np.array([[ 0]])]

        wind4sim = WindStress.WindStress(t=wind_t, wind_u=np.float32(wind_u), wind_v=np.float32(wind_v))

        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, \
                        self.eta0, self.hu0, self.hv0, self.Hi, \
                        self.nx, self.ny, \
                        self.dx, self.dy, self.dt, \
                        self.g, self.f, self.r, \
                        boundary_conditions=self.boundary_conditions,
                        wind=wind4sim)
    

    def tearDown(self) -> None:
        if self.sim != None:
            self.sim.cleanUp()
            self.sim = None

        self.gpu_ctx = None
        self.gpu_ctx_winds = self.N_winds*[None]

        gc.collect() # Force run garbage collection to free up memory


    ###########################################################################
    # Lake at rest - no atm pressure or wind

    def test_multiple_wind_fields(self):
        wind4drifters = []
        for i in range(self.N_winds):
            wind4drifters.append( WindStress.WindStress(t=np.array([0]), wind_u=np.float32([np.array([[10*np.sin(2*np.pi/self.N_winds*i)]])]), wind_v=np.float32([np.array([[10*np.cos(2*np.pi/self.N_winds*i)]])])) )

        drifterSets = []
        for i in range(self.N_winds):
            drifterSets.append( GPUDrifterCollection.GPUDrifterCollection( self.gpu_ctx_winds[i], 1, 
                                                        wind = wind4drifters[i], wind_drift_factor=0.02,
                                                        boundaryConditions = self.sim.boundary_conditions,
                                                        domain_size_x =  self.sim.nx*self.sim.dx,
                                                        domain_size_y =  self.sim.ny*self.sim.dy,
                                                        gpu_stream = self.sim.gpu_stream) )
            drifterSets[i].setDrifterPositions([[int(0.5*self.ny*self.dy),int(0.5*self.ny*self.dy)]])
        
        for min in range(self.T):
            dt = 1
            self.sim.step(dt)
            for i in range(self.N_winds):
                drifterSets[i].drift(self.sim.gpu_data.h0, self.sim.gpu_data.hu0, self.sim.gpu_data.hv0, \
                                self.sim.bathymetry.Bm, self.sim.nx, self.sim.ny, self.sim.t, self.sim.dx, self.sim.dy, \
                                dt, np.int32(2), np.int32(2))

        drifter_positions = []
        for i in range(self.N_winds):
            drifter_positions.append( list(drifterSets[i].getDrifterPositions()[0]) )

        # Reference calculated from commit 99fbe6ac78bc65a9f09e57a026c0b41cd528e62d
        ref = [[5046.37353515625, 5120.1171875],
            [5106.373046875, 5104.00390625],
            [5150.29638671875, 5060.05859375],
            [5166.3740234375, 5000.0],
            [5150.29638671875, 4939.94140625],
            [5106.373046875, 4895.99609375],
            [5046.37353515625, 4879.8828125],
            [4986.37353515625, 4895.99609375],
            [4942.4501953125, 4939.94140625],
            [4926.373046875, 5000.0],
            [4942.4501953125, 5060.05859375],
            [4986.37353515625, 5104.00390625]]

        self.assertAlmostEqual(np.max(np.abs(np.array(drifter_positions) - np.array(ref))), 0.0, places=4)