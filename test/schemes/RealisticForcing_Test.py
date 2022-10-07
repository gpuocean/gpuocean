# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2022 SINTEF Digital

This python module implements tests related to atmospheric pressure
implemented in the CDKLM scheme

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
from gpuocean.utils import Common, WindStress, AtmosphericPressure

class RealisticForcingTest(unittest.TestCase):
    def setUp(self):
        self.sim_args = {
            "gpu_ctx": Common.CUDAContext(),
            "nx": 500, "ny": 400, "dx": 800, "dy": 500,
            "g": 9.81, "dt": 0.0, "f": 0.0, "r": 0.0,
            "rho_o": 1015, 
            "wind_stress": WindStress.WindStress(),
            "atmospheric_pressure": AtmosphericPressure.AtmosphericPressure()
        }

        self.dataShape = (self.sim_args["ny"]+4, self.sim_args["nx"]+4)

        self.init_args = {
            "eta0": np.zeros(self.dataShape, dtype=np.float32),
            "hu0" : np.zeros(self.dataShape, dtype=np.float32),
            "hv0" : np.zeros(self.dataShape, dtype=np.float32),
            "H"  : np.ones((self.dataShape[0]+1, self.dataShape[1]+1), dtype=np.float32)*100
        }

        self.sim = None
        self.gradualT = 5*3600
        self.numPressures = 10
        self.p_atm_0 = np.float32(100000)
        self.p_atm = np.ones(self.dataShape, dtype=np.float32)*self.p_atm_0
    
    
    def setPressure(self, temporal_a=0, temporal_b=0):
        if temporal_a != temporal_b:
            self.setTemporalPressure(temporal_a, temporal_b)
        else:
            self.sim_args["atmospheric_pressure"] = AtmosphericPressure.AtmosphericPressure(P=[self.p_atm])

    def setTemporalPressure(self, a, b):
        P = [None]*self.numPressures
        t = np.arange(self.numPressures, dtype=np.float32)*3600*(b - a)/(self.numPressures-1)
        temporal_handle = self.makeTemporalFunctionHandle(a, b)
        for i in range(self.numPressures):
            P[i] = (self.p_atm - self.p_atm_0)*temporal_handle(t[i]) + self.p_atm_0
        self.sim_args["atmospheric_pressure"] = AtmosphericPressure.AtmosphericPressure(t=t, P=P)
    
    ###########################################################
    ## Setting up initial conditions and forcing
    def setBumpyP(self, balanced_eta=True, temporal_a=0, temporal_b=0):
        x = np.linspace(0, 2*np.pi, int((self.dataShape[1]-4)/2))
        y = np.linspace(0, 2*np.pi, self.dataShape[0]-4)

        for i in range(len(x)):
            for j in range(len(y)):
                self.p_atm[j, i                             ] += 0.5*(1 - np.cos(x[i]))*(1 - np.cos(y[j]))*2000
                self.p_atm[j, i + int(self.sim_args["nx"]/2)] += 0.5*(np.cos(x[i]) - 1)*(1 - np.cos(y[j]))*2000

                if balanced_eta:
                    self.init_args["eta0"][j, i                             ] = -0.5*(1 - np.cos(x[i]))*(1 - np.cos(y[j]))*2000 / (self.sim_args["g"]*self.sim_args["rho_o"])
                    self.init_args["eta0"][j, i + int(self.sim_args["nx"]/2)] = -0.5*(np.cos(x[i]) - 1)*(1 - np.cos(y[j]))*2000 / (self.sim_args["g"]*self.sim_args["rho_o"])
        self.setPressure(temporal_a, temporal_b)  

    def setLinearXP(self, balanced_eta=True, temporal_a=0, temporal_b=0):
        fullx = np.linspace(-1.5, self.sim_args["nx"]+1.5, self.sim_args["nx"]+4)
        fully = np.linspace(-1.5, self.sim_args["ny"]+1.5, self.sim_args["ny"]+4)
        for i in range(len(fullx)):
            for j in range(len(fully)):
                # Linear along x
                self.p_atm[j, i] += - 2000 + 4000*fullx[i]/ self.sim_args["nx"]
                if balanced_eta:
                    self.init_args["eta0"][j, i] = - 4000*fullx[i]/(self.sim_args["nx"] * (self.sim_args["g"]*self.sim_args["rho_o"]))
        self.setPressure(temporal_a, temporal_b)

    
    def setLinearXP_(self, balanced_eta=True):
        fullx = np.linspace(-1.5, self.sim_args["nx"]+1.5, self.sim_args["nx"]+4)
        fully = np.linspace(-1.5, self.sim_args["ny"]+1.5, self.sim_args["ny"]+4)
        for i in range(len(fullx)):
            for j in range(len(fully)):
                # Linear along x
                self.p_atm[j, i] += - 2000 + 4000*fullx[i]/ self.sim_args["nx"]
                if balanced_eta:
                    self.init_args["eta0"][j, i] = - 4000*fullx[i]/(self.sim_args["nx"] * (self.sim_args["g"]*self.sim_args["rho_o"]))

    def setLinearYP(self, balanced_eta=True, temporal_a=0, temporal_b=0):
        fullx = np.linspace(-1.5, self.sim_args["nx"]+1.5, self.sim_args["nx"]+4)
        fully = np.linspace(-1.5, self.sim_args["ny"]+1.5, self.sim_args["ny"]+4)
        for i in range(len(fullx)):
            for j in range(len(fully)):
                # Linear along y
                self.p_atm[j, i]  += - 2000 + 4000*fully[j]/ self.sim_args["ny"]
                if balanced_eta:
                    self.init_args["eta0"][j, i] = - 4000*fully[j]/(self.sim_args["ny"] * (self.sim_args["g"]*self.sim_args["rho_o"]))
        self.setPressure(temporal_a, temporal_b)


    def setLinearDiagP(self, balanced_eta=True, temporal_a=0, temporal_b=0):
        fullx = np.linspace(-1.5, self.sim_args["nx"]+1.5, self.sim_args["nx"]+4)
        fully = np.linspace(-1.5, self.sim_args["ny"]+1.5, self.sim_args["ny"]+4)
        for i in range(len(fullx)):
            for j in range(len(fully)):
                # Linear on the diagonal
                self.p_atm[j, i] += - 2000 + 4000*(fullx[i] + fully[j])/ (self.sim_args["nx"] + self.sim_args["ny"])
                if balanced_eta:
                    self.init_args["eta0"][j, i] = - 4000*(fullx[i] + fully[j])/((self.sim_args["nx"] + self.sim_args["ny"]) * (self.sim_args["g"]*self.sim_args["rho_o"]))
        self.setPressure(temporal_a, temporal_b)

    def setLinearWindY(self):
        t = [0, 10*360]
        X = [np.zeros((self.sim_args["ny"]+4,1), dtype=np.float32, order='C'),
             np.zeros((self.sim_args["ny"]+4,1), dtype=np.float32, order='C')]
        Y = [np.zeros((1,1),                     dtype=np.float32, order='C'),
             np.zeros((1,1),                     dtype=np.float32, order='C')]
        
        # Linear wind stress
        X[1][:,0] += np.linspace(0, 5e-4, self.sim_args["ny"]+4)

        self.sim_args["wind_stress"] = WindStress.WindStress(t=t, X=X, Y=Y)

    def makeGradualP(self, a, b):
        self.sim_args["p_atm_factor_handle"] = self.makeTemporalFunctionHandle(a, b)
        
    def makeTemporalFunctionHandle(self, a, b):
        return lambda t: min((t+a*3600)/(b*3600), 1.0)
        

    def tearDown(self) -> None:
        if self.sim != None:
            self.sim.cleanUp()
            self.sim = None

        self.init_args = None

        if self.sim_args["gpu_ctx"] is not None:
            self.assertEqual(sys.getrefcount(self.sim_args["gpu_ctx"]), 2)
            self.gpu_ctx = None
        
        gc.collect() # Force run garbage collection to free up memory


    def check_steady_state(self, places=0, eta_only=False, normalize_eta=False):
        eta1, hu1, hv1 = self.sim.download(interior_domain_only=True)
        etadiff = eta1 - self.init_args["eta0"][2:-2, 2:-2]
        if normalize_eta:
            etadiff = eta1 - (self.init_args["eta0"][2:-2, 2:-2] - np.mean(self.init_args["eta0"][2:-2, 2:-2]))

        if places == 0:
            if not eta_only:
                self.assertEqual(np.abs(hu1[1:-1, 1:-1]).max(), 0, msg='Not steady state. Got np.abs(hu1).max() = '+str(np.abs(hu1).max()))
                self.assertEqual(np.abs(hv1[1:-1, 1:-1]).max(), 0, msg='Not steady state. Got np.abs(hv1).max() = '+str(np.abs(hv1).max()))
            self.assertEqual(np.abs(etadiff[1:-1, 1:-1]).max(), 0, msg='Not steady state. Got np.abs(etadiff).max() = '+str(np.abs(etadiff).max()))
            
        else:
            if not eta_only:
                self.assertAlmostEqual(np.abs(hu1[1:-1, 1:-1]).max(), 0, places=places)
                self.assertAlmostEqual(np.abs(hv1[1:-1, 1:-1]).max(), 0, places=places)
            self.assertAlmostEqual(np.abs(etadiff[1:-1, 1:-1]).max(), 0, places=places)


    ###########################################################################
    # Lake at rest - no atm pressure or wind

    def test_steady_state_lake_at_rest(self):
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        self.sim.step(3600)
        self.check_steady_state()


    ###########################################################################
    # Test atmospheric pressure class
    def test_atmospheric_pressure_default_class(self):
        self.assertEqual(len(self.sim_args["atmospheric_pressure"].t), 1)
        self.assertEqual(len(self.sim_args["atmospheric_pressure"].P), 1)
        self.assertEqual(self.sim_args["atmospheric_pressure"].P[0].shape, (1,1))
        self.assertEqual(self.sim_args["atmospheric_pressure"].P[0][0,0], 0)
        self.assertEqual(self.sim_args["atmospheric_pressure"].t[0], 0)
        orig_P = self.sim_args["atmospheric_pressure"].getOriginalP()
        self.assertEqual(len(orig_P), 1)
        self.assertEqual(orig_P[0].shape, (1,1))
        self.assertEqual(orig_P[0][0,0], 0)
        
        
    
    def test_atmospheric_pressure_temporal_class(self):
        self.setLinearXP(temporal_a = 0 , temporal_b = 9)
        self.assertEqual(self.numPressures, len(self.sim_args["atmospheric_pressure"].t))
        self.assertEqual(self.numPressures, len(self.sim_args["atmospheric_pressure"].P))
        self.assertEqual(self.sim_args["atmospheric_pressure"].P[0].shape, self.dataShape)
        self.assertTrue(np.all(self.sim_args["atmospheric_pressure"].P[0] == 0))
        orig_P = self.sim_args["atmospheric_pressure"].getOriginalP()
        self.assertTrue(np.all(orig_P[0] == self.p_atm_0))
        self.assertTrue(np.all(orig_P[-1] == self.p_atm))
        self.assertTrue(np.all(self.sim_args["atmospheric_pressure"].P[-1] == self.p_atm - self.p_atm_0))
        for i in range(1,len(self.sim_args["atmospheric_pressure"].t)):
            self.assertLess(   self.sim_args["atmospheric_pressure"].P[i][ 10,  10], self.sim_args["atmospheric_pressure"].P[i-1][ 10,  10])
            self.assertGreater(self.sim_args["atmospheric_pressure"].P[i][-10, -10], self.sim_args["atmospheric_pressure"].P[i-1][-10, -10])

    ###########################################################################
    # Non-zero steady states caused by atmospheric pressure 

    def test_steady_state_linear_x(self):
        self.setLinearXP()
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        self.sim.step(3600)
        self.check_steady_state(places=3)

    def test_steady_state_linear_y(self):
        self.setLinearYP()
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        self.sim.step(3600)
        self.check_steady_state(places=3)

    def test_steady_state_linear_diag(self):
        self.setLinearDiagP()
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        self.sim.step(3600)
        self.check_steady_state(places=2)

    def test_steady_state_bumps(self):
        self.setBumpyP()
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        self.sim.step(3600)
        self.check_steady_state(places=2)

    ###########################################################################
    # Obtaining approximate non-zero steady states through slowly changing atm pressure 

    def test_create_steady_state_linear_x(self):
        self.setLinearXP(balanced_eta=False, temporal_a=0, temporal_b=12)
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        for i in range(3600):
            self.sim.step(24)
        self.setLinearXP(balanced_eta=True)
        self.check_steady_state(places=1, eta_only=True, normalize_eta=True)

    def test_create_steady_state_linear_y(self):
        self.setLinearYP(balanced_eta=False, temporal_a=0, temporal_b=12)
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        for i in range(3600):
            self.sim.step(24)
        self.setLinearYP(balanced_eta=True)
        self.check_steady_state(places=1, eta_only=True, normalize_eta=True)

    def test_create_steady_state_linear_diag(self):
        self.setLinearDiagP(balanced_eta=False, temporal_a=0, temporal_b=12)
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        for i in range(3600):
            self.sim.step(24)
        self.setLinearDiagP(balanced_eta=True)
        self.check_steady_state(places=1, eta_only=True, normalize_eta=True)


    def test_create_steady_state_bumps(self):
        self.numPressures = 20
        self.setBumpyP(temporal_a=-10, temporal_b=10)
        self.init_args["eta0"] *= -1
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        for i in range(3600):
            self.sim.step(24)
        self.setBumpyP()
        self.check_steady_state(places=1, eta_only=True, normalize_eta=True)

    ###########################################################################
    ### Compare interpolation between few and many textures when temporal change is linear
    def test_different_number_of_linear_textures(self):
        self.numPressures = 100
        self.setLinearDiagP(balanced_eta=False, temporal_a=0, temporal_b=12)
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        for i in range(360):
            self.sim.step(24)
        eta1, hu1, hv1 = self.sim.download(interior_domain_only=True)

        self.sim.cleanUp()
        self.numPressures = 2
        self.p_atm = np.ones(self.dataShape, dtype=np.float32)*self.p_atm_0
        self.setLinearDiagP(balanced_eta=False, temporal_a=0, temporal_b=12)
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        for i in range(360):
            self.sim.step(24)
        eta2, hu2, hv2 = self.sim.download(interior_domain_only=True)

        self.assertLess(np.max(np.abs(eta1 - eta2)), 1e-4)
        self.assertLess(np.max(np.abs(hu1 - hu2)), 1e-2)
        self.assertLess(np.max(np.abs(hv1 - hv2)), 1e-2)
        



    ###########################################################################
    # Check linear wind forcing
    def test_wind_linear_y(self):
        # To investigate bug, see following notebook:
        # https://github.com/havahol/miscGPUOcean/blob/360848153609c2aefe8997dcac72884ecfca2d5f/atmospheric-pressure/WindTextureCheck.ipynb
        self.setLinearWindY()
        self.sim = CDKLM16.CDKLM16(**self.sim_args, **self.init_args)
        for i in range(360):
            self.sim.step(24)
        eta, hu, hv = self.sim.download(interior_domain_only=True)
        for j in range(1, self.sim.ny):
            self.assertGreaterEqual(hu[j, 100], hu[j-1, 100], "The linear wind-driven velocity flattens out")
        