# -*- coding: utf-8 -*-
"""
This software is part of GPU Ocean. 

Copyright (C) 2019 SINTEF Digital

This python module implements regression tests for writing and reading
simulators to NetCDF.

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

from SWESimulators import Common, CDKLM16, SimWriter, DoubleJetCase, SimReader


class NetCDFtest(unittest.TestCase):

    def setUp(self):
        self.gpu_ctx = Common.CUDAContext()

        self.sim = None
        self.file_sim = None
        
        self.printall = True
        
    def tearDown(self):
        if self.sim is not None:
            self.sim.cleanUp()
            self.sim = None
        if self.file_sim is not None:
            self.file_sim.cleanUp()
            self.file_sim = None
        
        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
        
        gc.collect() # Force run garbage collection to free up memory
        


    def test_netcdf_cdklm(self):
        
        # Create simulator and write to file:
        doubleJetCase = DoubleJetCase.DoubleJetCase(self.gpu_ctx,
                                                    DoubleJetCase.DoubleJetPerturbationType.IEWPFPaperCase)

        doubleJetCase_args, doubleJetCase_init = doubleJetCase.getInitConditions()
        netcdf_args = {
            'write_netcdf': True,
            'netcdf_filename': 'netcdf_test/netcdf_test.nc'
        }
        self.sim = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init, **netcdf_args)
        self.sim.closeNetCDF()
        
        
        # Create new simulator from the newly created file
        self.file_sim = CDKLM16.CDKLM16.fromfilename(self.gpu_ctx, netcdf_args['netcdf_filename'], cont_write_netcdf=False)
        
        # Loop over object attributes and compare those that are scalars
        for attr in dir(self.sim):
            if not attr.startswith('__'):
                if np.isscalar(getattr(self.sim, attr)):
                    sim_attr      = getattr(self.sim,      attr)
                    file_sim_attr = getattr(self.file_sim, attr)

                    # Round the simulation time to make it comparable
                    if attr == 't':
                        file_sim_attr = round(file_sim_attr) 
                        sim_attr      = round(sim_attr)

                    # Create error message and compare values of attributes
                    assert_msg = "Discrepancy in attribute " + attr + ":\n" + \
                                 "         sim."+ attr + ": " + str(sim_attr) + \
                                 "    file_sim."+ attr + ": " + str(file_sim_attr)
                    self.assertEqual(sim_attr, file_sim_attr, msg=assert_msg)
                    
                    if self.printall:
                        print('file_sim.' + attr + ': ' + str(file_sim_attr))
                        print('     sim.' + attr + ': ' + str(sim_attr))
                        if np.isreal(file_sim_attr):
                            print('Diff: ' + str(file_sim_attr - sim_attr))
                        print('')

                    #if np.isreal(getattr(file_sim, attr)):
                    #    print('Diff: ' + str(getattr(file_sim, attr) - getattr(sim, attr)))

        # Loop over attributes in the model error
        for attr in dir(self.sim.small_scale_model_error):
            if not attr.startswith('__'):
                if np.isscalar(getattr(self.sim.small_scale_model_error, attr)):
                    
                    sim_attr      = getattr(self.sim.small_scale_model_error,      attr)
                    file_sim_attr = getattr(self.file_sim.small_scale_model_error, attr)
                    
                    # Create error message and compare values of attributes
                    assert_msg = "Discrepancy in attribute " + attr + ":\n" + \
                                 "         sim.small_scale_model_error"+ attr + ": " + str(sim_attr) + \
                                 "    file_sim.small_scale_model_error"+ attr + ": " + str(file_sim_attr)
                    self.assertEqual(sim_attr, file_sim_attr, msg=assert_msg)
                    
                    if self.printall:
                        print('file_sim.small_scale_model_error.' + attr + ': ' + str(file_sim_attr))
                        print('     sim.small_scale_model_error.' + attr + ': ' + str(sim_attr))
                        if np.isreal(file_sim_attr):
                            print('Diff: ' + str(file_sim_attr - sim_attr))
                        print('')

                        
        # Compare ocean state:
        s_eta0, s_hu0, s_hv0 = self.sim.download()
        f_eta0, f_hu0, f_hv0 = self.file_sim.download()
        self.checkResults(s_eta0, s_hu0, s_hv0, f_eta0, f_hu0, f_hv0)
        
        dt = self.sim.dt
        self.sim.step(100*dt, apply_stochastic_term=False)
        self.file_sim.step(100*dt, apply_stochastic_term=False)
        s_eta0, s_hu0, s_hv0 = self.sim.download()
        f_eta0, f_hu0, f_hv0 = self.file_sim.download()
        self.checkResults(s_eta0, s_hu0, s_hv0, f_eta0, f_hu0, f_hv0)
        
        
         
    def checkResults(self, sim_eta, sim_hu, sim_hv, file_eta, file_hu, file_hv):
        diffEta = np.linalg.norm(sim_eta - file_hu) / np.max(np.abs(file_eta))
        diffU = np.linalg.norm(sim_hu - file_hu) / np.max(np.abs(file_hu))
        diffV = np.linalg.norm(sim_hv - file_hv) / np.max(np.abs(file_hv))
        maxDiffEta = np.max(sim_eta - file_eta) / np.max(np.abs(file_eta))
        maxDiffU = np.max(sim_hu - file_hu) / np.max(np.abs(file_hu))
        maxDiffV = np.max(sim_hv - file_hv) / np.max(np.abs(file_hv))
        
        self.assertAlmostEqual(maxDiffEta, 0.0, places=3,
                               msg='Unexpected eta difference! Max rel diff: ' + str(maxDiffEta) + ', L2 rel diff: ' + str(diffEta))
        self.assertAlmostEqual(maxDiffU, 0.0, places=3,
                               msg='Unexpected hu relative difference: ' + str(maxDiffU) + ', L2 rel diff: ' + str(diffU))
        self.assertAlmostEqual(maxDiffV, 0.0, places=3,
                               msg='Unexpected hv relative difference: ' + str(maxDiffV) + ', L2 rel diff: ' + str(diffV))
    
        # Test maximal time step:
        self.sim.updateDt()
        dt_host = self.sim._getMaxTimestepHost()
        self.assertEqual(self.sim.dt, dt_host, msg="maximum time step")
        
        if self.printall:
            print("Test of state successful")
        
        
 