# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018-2019 SINTEF Digital
Copyright (C) 2018-2019 Norwegian Meteorological Institute

This python module contains helper functions for data assimilation
experiments based on the DoubleJetCase.

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

import sys, os, json, datetime, time
import numpy as np
from SWESimulators import CDKLM16, Common, DoubleJetCase, GPUDrifterCollection, Observation



def generateTruth(gpu_ctx, destination_dir, 
                  duration_in_days=13, duration_in_hours=0,
                  folder_name=None,
                  log_to_screen=False):
    """
    This funtion generates a truth simulation that will be the subject for 
    data assimilation experiments. It is based on the DoubleJetCase parameters 
    and initial conditions, and is (by default) spun up for 3 days before starting 
    to write its state to file. The generated data set should cover time range from
    day 3 to day 13.
    
    The end time of the truth is duration_in_days + duration_in_hours.
    """



            
    #--------------------------------------------------------------
    # PARAMETERS
    #--------------------------------------------------------------
    # This file takes no parameters, as it is should clearly define a specific truth.
    # If we come to a time where we need to specify a lot of different truths, we can introduce argparser again.

    # Time parameters
    start_time      =  3*24*60*60 #  3 days
    end_time        = (duration_in_days*24 + duration_in_hours)*60*60
    simulation_time = end_time - start_time
    observation_frequency = 5*60  # 5 minutes
    netcdf_frequency = 60*60      # every hour

    # Drifter parameters:
    num_drifters = 64

    assert(os.path.exists(destination_dir)), 'destination_dir does not exist: ' + str(destination_dir)

    # File parameters:
    folder = os.path.join(destination_dir, "truth_" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    if folder_name is not None:
        folder = os.path.join(destination_dir, folder_name)
    assert(not os.path.exists(folder)), 'The directory ' + folder + ' already exists!'
    os.makedirs(folder)
    
    netcdf_filename  = os.path.join(folder, "double_jet_case_truth.nc")
    drifter_filename = os.path.join(folder, "drifter_observations.pickle")

    if log_to_screen: print("------ Generating initial ensemble ---------------")
    if log_to_screen: print("Writing truth to file: " + netcdf_filename)


    # Create CUDA context
    device_name = gpu_ctx.cuda_device.name()


    #--------------------------------------------------------------
    # Creating the Case (including spin up)
    #--------------------------------------------------------------
    if log_to_screen: print("Initializing the truth")

    tic = time.time()
    sim = None
    doubleJetCase = None

    try:
        doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx,
                                                    DoubleJetCase.DoubleJetPerturbationType.IEWPFPaperCase)

        doubleJetCase_args, doubleJetCase_init = doubleJetCase.getInitConditions()

        if (np.any(np.isnan(doubleJetCase_init["eta0"]))):
            print(" `-> ERROR: Not a number in spinup, aborting!")
            raise RuntimeError('Not a number in spinup')


        toc = time.time()
        if log_to_screen: print("\n{:02.4f} s: ".format(toc-tic) + "Generated initial conditions at day 3")

        
        #--------------------------------------------------------------
        # Initialize the truth from the Case (spin up is done in the Case)
        #--------------------------------------------------------------

        toc = time.time()
        if log_to_screen: print("\n{:02.4f} s: ".format(toc-tic) + "Generated initial conditions for truth")

        #netcdf_args = {}
        netcdf_args = {'write_netcdf': True, 'netcdf_filename': netcdf_filename}

        tic = time.time()
        sim = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init, **netcdf_args)

        toc = time.time()
        if log_to_screen: print("\n{:02.4f} s: ".format(toc-tic) + "Truth simulator initiated")

        t = sim.t
        if log_to_screen: print("Three days = " + str(start_time) + " seconds, but sim.t = " + str(sim.t) + ". Same? " + str(start_time == sim.t))
        assert(round(start_time) == round(sim.t)), 'Spin up time seems to be wrong'


        #--------------------------------------------------------------
        # Create drifters at t = 3 days
        #--------------------------------------------------------------
        drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters,
                                                             boundaryConditions=doubleJetCase_args['boundary_conditions'],
                                                             domain_size_x=sim.nx*sim.dx, domain_size_y=sim.ny*sim.dy)
        sim.attachDrifters(drifters)



        #--------------------------------------------------------------
        # Create observation object and write the first drifter positions
        #--------------------------------------------------------------
        observations = Observation.Observation(domain_size_x = sim.nx*sim.dx,
                                               domain_size_y = sim.ny*sim.dy,
                                               nx=sim.nx, ny=sim.ny)

        # Configure observations to register static positions
        observations.setBuoyCellsByFrequency(25, 25)
        observations.add_observation_from_sim(sim)

        netcdf_iterations = int(simulation_time/netcdf_frequency)
        observation_sub_iterations = int(netcdf_frequency/observation_frequency)

        if log_to_screen: print("We will now make " + str(netcdf_iterations) + " iterations with writing netcdf, and  " + \
              str(observation_sub_iterations) + " subiterations with registering drifter positions.")

        theoretical_time = start_time


        ########################
        # Main simulation loop #
        ########################
        tic = time.time()

        time_error = 0

        for netcdf_it in range(netcdf_iterations):
            for observation_sub_it in range(observation_sub_iterations):

                next_obs_time = t + observation_frequency
                
                # Step until next observation 
                sim.dataAssimilationStep(next_obs_time, write_now=False)
                
                # Store observation
                observations.add_observation_from_sim(sim)
                
                t = sim.t
                
                time_error += t-next_obs_time
                theoretical_time += observation_frequency
              
            sim.writeState()
            
            if netcdf_it % 10 == 0:
                
                # Check that everything looks okay
                eta, hu, hv = sim.download()
                if (np.any(np.isnan(eta))):
                    raise RuntimeError('Not a number at time ' + str(sim.t))
                
                sub_t = time.time() - tic
                if log_to_screen: print("{:02.4f} s into loop: ".format(sub_t) + "Done with netcdf iteration " + str(netcdf_it) + " of " + str(netcdf_iterations))
               
                
        toc = time.time()
        if log_to_screen: print("{:02.4f} s: ".format(toc-tic) + "Done with generating thruth")

        if log_to_screen: print("sim.t:            " + str(sim.t))


        ########################
        # Simulation loop DONE #
        ########################
                
        # Dump drifter observations to file:
        tic = time.time()
        observations.to_pickle(drifter_filename)
        toc = time.time()
        if log_to_screen: print("\n{:02.4f} s: ".format(toc-tic) + "Drifter observations written to " + drifter_filename)
    
        return os.path.abspath(folder)
        
    except:
        
        raise
       
    finally:
        if sim is not None:
            sim.cleanUp()
        if doubleJetCase is not None:
            doubleJetCase.cleanUp()
        if log_to_screen: print("\n{:02.4f} s: ".format(toc-tic) + "Clean up simulator done.")




