# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements an the
Implicit Equal-Weight Particle Filter (IEWPF), for use on
simplified ocean models.
The following papers describe the original iEWPF scheme, though with mistakes and variations.
     - 'Implicit equal-weights particle filter' by Zhu, van Leeuwen and Amezcua, Quarterly
            Journal of the Royal Meteorological Society, 2016
     - 'State-of-the-art stochastic data assimilation methods for high-dimensional
            non-Gaussian problems' by Vetra-Carvalho et al, Tellus, 2018
The following paper describe the two-stage IEWPF scheme:
     - 'A revied Implicit Equal-Weights Particle Filter' by Skauvold et al, ???, 2018


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


from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import gc
import pycuda.driver as cuda
from scipy.special import lambertw, gammainc
from scipy.optimize import newton
import logging

from SWESimulators import Common, OceanStateNoise, config, EnsemblePlot

class IEWPFOcean:
    """
    This class implements the Implicit Equal-Weight Particle Filter for an ocean
    model with small scale ocean state perturbations as model errors.
    
    Input to constructor:
    ensemble: An object of super-type BaseOceanStateEnsemble.
            
    """
    def __init__(self, ensemble, debug=False, show_errors=False,
                 block_width=16, block_height=16, beta=None, write_betas=False):
        
        self.logger = logging.getLogger(__name__)
        self.logger_level = config.GPUOceanLoggerLevels.IEWPF_DEBUG
        
        self.gpu_ctx = ensemble.gpu_ctx
        self.master_stream = cuda.Stream()
        
        self.debug = debug
        self.show_errors = show_errors
        
        # Store information needed internally in the class
        self.dx = np.float32(ensemble.getDx()) 
        self.dy = np.float32(ensemble.getDy())
        self.nx = np.int32(ensemble.getNx())
        self.ny = np.int32(ensemble.getNy())
        
        self.interpolation_factor = np.int32(ensemble.particles[0].small_scale_model_error.interpolation_factor)
        
        # Check that the interpolation factor plays well with the grid size:
        assert ( self.interpolation_factor > 0 and self.interpolation_factor % 2 == 1), 'interpolation_factor must be a positive odd integer'
        assert (self.nx % self.interpolation_factor == 0), 'nx must be divisible by the interpolation factor'
        assert (self.ny % self.interpolation_factor == 0), 'ny must be divisible by the interpolation factor'
        
        # The size of the coarse grid 
        self.coarse_nx = np.int32(self.nx/self.interpolation_factor)
        self.coarse_ny = np.int32(self.ny/self.interpolation_factor)
        self.coarse_dx = np.float32(self.dx*self.interpolation_factor)
        self.coarse_dy = np.float32(self.dy*self.interpolation_factor)
        
        self.soar_q0 = np.float32(ensemble.particles[0].small_scale_model_error.soar_q0)
        self.soar_L  = np.float32(ensemble.particles[0].small_scale_model_error.soar_L)
        self.f = np.float32(ensemble.particles[0].f)
        self.g = np.float32(ensemble.particles[0].g)
        
        # Water depth is assumed constant, assumption is checked below.
        H = ensemble.particles[0].downloadBathymetry()[1][2:-2, 2:-2] # H in cell centers
        self.const_H = np.float32(H[0,0])

        self.boundaryConditions = ensemble.particles[0].boundary_conditions
        
        self.geoBalanceConst = np.float32(self.g*self.const_H/(2.0*self.f))

        self.Nx = np.int32(self.nx*self.ny*3)  # state dimension
        self.random_numbers_ratio = self.Nx/(self.coarse_nx*self.coarse_ny)
        
        self.numParticles = np.int32(ensemble.getNumParticles())
        self.numDrifters  = np.int32(ensemble.getNumDrifters())
        
        
        # The underlying assumptions are:
        # 1) that the equilibrium depth is constant:
        assert(np.max(H) == np.min(H)), 'IEWPF can not be used with a non-constant ocean depth'
        # 2) that both boundaries are periodic:
        assert(self.boundaryConditions.isPeriodicNorthSouth()), 'IEWPF requires periodic boundary conditions in north-south'
        assert(self.boundaryConditions.isPeriodicEastWest()),  'IEWPF requires periodic boundary conditions in east-west'
        # 3) that the Coriolis force is constant for the entire domain:
        assert (ensemble.particles[0].coriolis_beta == 0), 'IEWPF requires constant Coriolis forcing, but got beta = ' + str(ensemble.beta)
        # 4) that dx and dy are the same
        assert (self.dx == self.dy), 'IEWPF requires square grid cells, but got (dx, dy) = ' + str((self.dx, self.dy))
        
        
        # Note the we intentionally do not add the ensemble as a member variable.
        
        # Create constant matrix S = (HQH^T + R)^-1 and copy to the GPU
        # The matrix represents the combined "observed model error" and observation error.
        self.S_host, self.S_device = None, None
        self.S_host = self._createS(ensemble)
        self.S_device = Common.CUDAArray2D(self.master_stream, 2, 2, 0, 0, self.S_host)
        
        # Create constant localized SVD matrix and copy to the GPU.
        # This matrix is defined for the coarse grid, and ignores all use of the interpolation operator.
        self.localSVD_host, self.localSVD_device = None, None
        self.localSVD_host = self._generateLocaleSVDforP(ensemble)
        self.localSVD_device = Common.CUDAArray2D(self.master_stream, 49, 49, 0, 0, self.localSVD_host)
    
        
        self.iewpf_kernels = self.gpu_ctx.get_kernel("iewpf_kernels.cu", \
                                                     defines={'block_width': block_width, 'block_height': block_height})
        
        
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.setBufferToZeroKernel = self.iewpf_kernels.get_function("setBufferToZero")
        self.setBufferToZeroKernel.prepare("iiPi")
        
        self.blas_xaxpbyKernel = self.iewpf_kernels.get_function("blas_xaxpby")
        self.blas_xaxpbyKernel.prepare("iiPiPiff")
        
        self.halfTheKalmanGainKernel = self.iewpf_kernels.get_function("halfTheKalmanGain")
        self.halfTheKalmanGainKernel.prepare("iiffffiifffPi")
        
        self.localSVDOnGlobalXiKernel = self.iewpf_kernels.get_function("localSVDOnGlobalXi")
        self.localSVDOnGlobalXiKernel.prepare("iiiiPiPi")
        
        
        #Compute kernel launch parameters
        self.local_size_Kalman  = (7, 7, 1)
        self.global_size_Kalman = (1, 1)
        
        self.local_size_SVD  = (7, 7, 1)
        self.global_size_SVD = (1, 1)
        
        self.local_size_domain = (block_width, block_height, 1)
        self.global_size_domain = ( \
                                   int(np.ceil(self.nx / float(self.local_size_domain[0]))), \
                                   int(np.ceil(self.ny / float(self.local_size_domain[1]))) \
                                  ) 
        self.noise_buffer_domain = ( \
                                    int(np.ceil(self.coarse_nx / float(self.local_size_domain[0]))), \
                                    int(np.ceil(self.coarse_ny / float(self.local_size_domain[1]))) \
                                   ) 

        self.beta = beta

        self.write_betas = True 
        if self.write_betas:
            self.betas = np.zeros(0)
    
       
    def log(self, msg):
        self.logger.log(self.logger_level.value, msg)
    
    
    def __del__(self):
        self.cleanUp()
        
        
    def cleanUp(self):
        # All allocated data needs to be freed from here
        if self.S_device is not None:
            self.S_device.release()
        if self.localSVD_device is not None:
            self.localSVD_device.release()
        self.gpu_ctx = None
    
    
    
    ### Main two-stage IEWPF METHOD
    def iewpf_2stage(self, ensemble, infoPlots=None, it=None, perform_step=True):
        """
        The complete two-stage IEWPF algorithm implemented on the GPU.
        
        Input parameters:
            ensemble  - the ensemble on which the particle filter is appplied
            infoPlots (optional) - List of figures. New figure of ensemble is added
                before and after the particle filter
            it (optional) - The iteration number, used for logging and figure generation
            perform_step - Flag that indicates whether the ensemble and truth should perform the 
                final timestep to the observation, or if this has already been done.
        """
    
        # Step the truth and particles the final timestep:
        if perform_step:
            t = ensemble.step_truth(ensemble.getDt(), stochastic=True)
            t = ensemble.step_particles(ensemble.getDt(), stochastic=False)
        
        self.log('------------------------------------------------------')
        self.log('------ Two-stage IEWPF at t = ' + str(ensemble.t) + '   -------')
        self.log('------------------------------------------------------')
        
        mem_free, mem_available = cuda.mem_get_info()
        self.log("\n(free mem, avail mem, percentage free): " + str((mem_free, mem_available, 
                                                                100*mem_free/mem_available)))
        
        # All arrays are of size numParticles, even though the ensemble size is only 
        # numActiveParticles.
        
        # Obtain observations, innovations and the weight from previous timestep
        observed_drifter_positions = ensemble.observeTrueDrifters()
        innovations = ensemble.getInnovations()
        w_rest = -np.log(1.0/ensemble.getNumActiveParticles())*np.ones(ensemble.getNumParticles())
        
        self.log('observed drifter positions:\n' + str(observed_drifter_positions))
        self.log('observed true state:\n' + str(ensemble.observeTrueState()))
        self.log('observed particle states:\n' + str(ensemble.observeParticles()))
        self.log('(active particles, total number of particles): ' + str((ensemble.getNumActiveParticles(), ensemble.getNumParticles())))
        
        # save plot before
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 1)
            
        # The target weight depends on the values of phi, nu and gamma for all particles,
        # and these values are therefore required to be stored in arrays.
        phi_array     = np.zeros(ensemble.getNumParticles())
        nu_norm_array = np.zeros(ensemble.getNumParticles())
        gamma_array   = np.zeros(ensemble.getNumParticles())
        
        for p in range(ensemble.getNumParticles()):
            if ensemble.particlesActive [p]:
                # Pull particles towards observation by adding a Kalman gain term
                #     Also, we find phi within this function
                phi_array[p] = self.addKalmanGain(ensemble.particles[p], observed_drifter_positions, innovations[p], drifter_id=p)
                
                # Sample perpendicular xi and nu
                # Obtain gamma = xi^T * xi and nu^T * nu at the same time
                gamma_array[p], nu_norm_array[p] = self.samplePerpendicular(ensemble.particles[p])
                
                if np.isnan(phi_array[p]) or np.isnan(nu_norm_array[p]) or np.isnan(gamma_array[p]):
                    ensemble.deactivateParticle(p, msg='Failed with the Kalman gain, ' + \
                            '(phi, gamma, nu_norm): ' + str((phi_array[p], gamma_array[p], nu_norm_array[p])))
                
            else:
                phi_array[p], gamma_array[p], nu_norm_array[p] = np.nan, np.nan, np.nan
            
        c_array = phi_array + w_rest
        self.log('--------------------------------------')
        self.log('------ Half in two-stage IEWPF -------')
        self.log('--------------------------------------')
        self.log('phi_array:\n ' + str(phi_array))
        self.log("nu_norm_array:\n" + str(nu_norm_array))
        self.log("gamma_array:\n" + str(gamma_array))
        self.log("c_array:\n" + str(c_array))
        
        
        
        # Synchronize all particles in order to find the target weight and beta
        # based on the active particles only
        active_c_array = c_array[ensemble.particlesActive]
        active_nu_norm_array = nu_norm_array[ensemble.particlesActive]
        target_weight, beta = self.obtainTargetWeightTwoStage(active_c_array, active_nu_norm_array)
        
        self.log('target_weight: ' + str(target_weight))
        self.log('beta         : ' + str(beta))
        
        if beta < 0:
            print("------!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!------------------------")
            print("   Something went HORRIBLY WRONG!")
            print("   We got negative beta: " + str(beta))
            print("   Target weight: " + str(target_weight))
            print('phi_array:\n ' + str(phi_array))
            print("nu_norm_array:\n" + str(nu_norm_array))
            print("gamma_array:\n" + str(gamma_array))
            print("c_array:\n" + str(c_array))
            print("active_particles:\n" + str(ensemble.particlesActive))
            print("(mean(c) - c)/nu_norm:\n" + str((np.mean(active_c_array)-active_c_array)/active_nu_norm_array))
            print("\nSetting beta = 0 and continueing... ")
            print("------!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!------------------------")
            beta = 0

        if self.beta is not None:
            beta = self.beta
            self.log('beta (hard set): ' + str(beta))

        if self.write_betas:
            self.betas = np.append(self.betas, beta)
        
        for p in range(ensemble.getNumParticles()):
            if ensemble.particlesActive[p]:
                # Solve implicit equation
                c_star = target_weight - c_array[p] - (beta - 1)*nu_norm_array[p]
                try:
                    alpha = self.solveImplicitEquation(gamma_array[p], target_weight, w_rest[p], c_star, particle_id=p)
                
                    # Apply the SVD covariance structure at the drifter positions on scaled xi and nu
                    self.applySVDtoPerpendicular(ensemble.particles[p], observed_drifter_positions,
                                                 alpha, beta)
                    
                    # Add scaled sample from P to the state vector
                    ensemble.particles[p].small_scale_model_error.perturbSim(ensemble.particles[p],\
                                                                             update_random_field=False)
                    
                    # Update ghost cells to ensure that periodic boundary conditions are satisfied
                    ensemble.particles[p].applyBoundaryConditions()
                    
                except RuntimeError as re:
                    ensemble.deactivateParticle(p, msg='Failed solving implicit equation: ' + str(re))
                
        # save plot after
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 3)
            
            
    
    ### MAIN one-stage IEWPF METHOD
    def iewpf(self, ensemble, infoPlots=None, it=None, perform_step=True):
        """
        The complete IEWPF algorithm implemented on the GPU.
        
        Retrieves innovations and target weights from the ensemble, and updates 
        each particle according to the IEWPF method.
        """
        
        #
        # For definitions of phi, gamma, alpha, and description of the steps in this
        # algorithm, please take a look at the pseudocode in the paper
        # 'State-of-the-art stochastic data assimilation methods for high-dimensional
        # non-Gaussian problems' by Vetra-Carvalho et al, Tellus, 2018
        #
        
        # Step -1: Deterministic step
        if perform_step:
            t = ensemble.step_truth(ensemble.getDt(), stochastic=True)
            t = ensemble.step_particles(ensemble.getDt(), stochastic=False)
        
        # Step 0: Obtain innovations
        observed_drifter_positions = ensemble.observeTrueDrifters()
        innovations = ensemble.getInnovations()
        w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())
        
        # save plot before
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 1)
        
        # Step 1: Find maximum weight
        target_weight = self.obtainTargetWeight(ensemble, innovations)
        
        for p in range(ensemble.getNumParticles()):
                        
            # Loop step 1: Pull particles towards observation by adding a Kalman gain term
            #     Also, we find phi within this function
            phi = self.addKalmanGain(ensemble.particles[p], observed_drifter_positions, innovations[p], drifter_id=p)
            
            # Loop step 2: Sample xi \sim N(0, P), and get gamma in the process
            gamma = self.sampleFromP(ensemble.particles[p], observed_drifter_positions)
            
            # Loop step 3: Solve implicit equation
            c_star = target_weight - (phi + w_rest[p])
            alpha = self.solveImplicitEquation(gamma, target_weight, w_rest[p], c_star, particle_id=p)
            
            # Loop steps 4:Add scaled sample from P to the state vector
            ensemble.particles[p].small_scale_model_error.perturbSim(ensemble.particles[p],\
                                                                     update_random_field=False, \
                                                                     perturbation_scale=np.sqrt(alpha))   
            
            # Update ghost cells to ensure that periodic boundary conditions are satisfied
            ensemble.particles[p].applyBoundaryConditions()
            
            # TODO
            # Reset the drifter positions in each particle.
            # One key line woould be:
            # ensemble.particles[p].drifters.setDrifterPositions(newPos)

        
        # save plot after
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 3)
    
    
    
    def iewpf_timer(self, ensemble, infoPlots=None, it=None,  perform_step=True):
        """
        Same as the function iewpf(self, ...) but with lots of events so that
        various parts of the IEWPF algorithm can be timed.
        
        The complete IEWPF algorithm implemented on the GPU.
        
        Retrieves innovations and target weights from the ensemble, and updates 
        each particle according to the IEWPF method.
        """
        # Step -1: Deterministic step
        print ("----------")
        start_pre_loop = cuda.Event()
        start_pre_loop.record(self.master_stream)
        
        if perform_step:
            t = ensemble.step_truth(ensemble.getDt(), stochastic=True)
            t = ensemble.step_particles(ensemble.getDt(), stochastic=False)

        deterministic_step_event = cuda.Event()
        deterministic_step_event.record(self.master_stream)
        deterministic_step_event.synchronize()
        gpu_elapsed = deterministic_step_event.time_since(start_pre_loop)*1.0e-3
        print ("Deterministic timestep took: " + str(gpu_elapsed))
        
        # Step 0: Obtain innovations
        observed_drifter_positions = ensemble.observeTrueDrifters()
        
        observe_drifters_event = cuda.Event()
        observe_drifters_event.record(self.master_stream)
        observe_drifters_event.synchronize()
        gpu_elapsed = observe_drifters_event.time_since(deterministic_step_event)*1.0e-3
        print("Observing drifters took:     " + str(gpu_elapsed))
        
        
        innovations = ensemble.getInnovations()
        innovations_event = cuda.Event()
        innovations_event.record(self.master_stream)
        innovations_event.synchronize()
        gpu_elapsed = innovations_event.time_since(observe_drifters_event)*1.0e-3
        print("innovations_event took:      " + str(gpu_elapsed))
        
        w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())

        # Step 1: Find maximum weight
        target_weight = self.obtainTargetWeight(ensemble, innovations)
        
        dummy = self.download_reduction_buffer()
        target_weight_event = cuda.Event()
        target_weight_event.record(self.master_stream)
        target_weight_event.synchronize()
        gpu_elapsed = target_weight_event.time_since(innovations_event)*1.0e-3
        print("Finding target weight took:  " + str(gpu_elapsed))
        
        for p in range(ensemble.getNumParticles()):
            print ("----------")
            print ("Starting particle " + str(p))
            start_loop = cuda.Event()
            kalman_event = cuda.Event()
            p_event = cuda.Event()
            add_scaled_event = cuda.Event()
            
            start_loop.record(self.master_stream)
            
            # Loop step 1: Pull particles towards observation by adding a Kalman gain term
            #     Also, we find phi within this function
            phi = self.addKalmanGain(ensemble.particles[p], observed_drifter_positions, innovations[p], drifter_id=p)
            
            kalman_event.record(self.master_stream)
            kalman_event.synchronize()
            gpu_elapsed = kalman_event.time_since(start_loop)*1.0e-3
            print ("Kalman gain took:   " + str(gpu_elapsed))
            
            
            # Loop step 2: Sample xi \sim N(0, P), and get gamma in the process
            gamma = self.sampleFromP(ensemble.particles[p], observed_drifter_positions)
            
            p_event.record(self.master_stream)
            p_event.synchronize()
            gpu_elapsed = p_event.time_since(kalman_event)*1.0e-3
            print ("Sample from P took: " + str(gpu_elapsed) )
            
            # Loop step 3: Solve implicit equation
            c_star = target_weight - (phi + w_rest)
            alpha = self.solveImplicitEquation(gamma, target_weight, w_rest[p], c_star, particle_id=p)
            
            
            
            # Loop steps 4:Add scaled sample from P to the state vector
            ensemble.particles[p].small_scale_model_error.perturbSim(ensemble.particles[p],\
                                                                     update_random_field=False, \
                                                                     perturbation_scale=np.sqrt(alpha))
            
            # Update ghost cells to ensure that periodic boundary conditions are satisfied
            ensemble.particles[p].applyBoundaryConditions()
            
            add_scaled_event.record(self.master_stream)
            add_scaled_event.synchronize()
            gpu_elapsed = add_scaled_event.time_since(p_event)*1.0e-3
            print ("Add scaled xi took: " + str(gpu_elapsed) )
            
            print ("Done particle " + str(p))
            print ("----------")
            
        # save plot after
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 3)
    
    ###------------------------    
    ### GPU Methods
    ###------------------------
    # Functions needed for the GPU implementation of IEWPF
    
    def setNoiseBufferToZero(self, sim):
        """
        Reset the simulators random numbers buffer to zero, and thereby prepares 
        the generation of the Kalman gain.
        """
        self.setBufferToZeroKernel.prepared_async_call(self.noise_buffer_domain,
                                                       self.local_size_domain, 
                                                       sim.gpu_stream,
                                                       sim.small_scale_model_error.rand_nx, 
                                                       sim.small_scale_model_error.rand_ny,
                                                       sim.small_scale_model_error.random_numbers.data.gpudata,
                                                       sim.small_scale_model_error.random_numbers.pitch)
        
    def addBetaNuIntoAlphaXi(self, sim, alpha, beta):
        """
        The expression
        alpha^{1/2} P^{1/2} xi + beta^{1/2} P^{1/2} nu
        is simplified to
        P^{1/2} (alpha^{1/2} xi + beta^{1/2} nu),
        and this function is therefore used to obtain
        xi = alpha^{1/2} xi + beta^{1/2} nu
        """
        
        # x = a*x + b*y
        # x = xi, a = alpha, y = nu, b = beta
        self.blas_xaxpbyKernel.prepared_async_call(self.noise_buffer_domain,
                                                  self.local_size_domain, 
                                                  sim.gpu_stream,
                                                  sim.small_scale_model_error.rand_nx, 
                                                  sim.small_scale_model_error.rand_ny,
                                                  sim.small_scale_model_error.random_numbers.data.gpudata,
                                                  sim.small_scale_model_error.random_numbers.pitch,
                                                  sim.small_scale_model_error.perpendicular_random_numbers.data.gpudata,
                                                  sim.small_scale_model_error.perpendicular_random_numbers.pitch,
                                                  np.float32(np.sqrt(alpha)),
                                                  np.float32(np.sqrt(beta)))
        
        
        
    def addKalmanGain(self, sim, all_observed_drifter_positions, innovation, drifter_id=None):
        """
        Generates a Kalman gain type field according to the drifter positions and innovation,
        and adds it to the ocean state held på the simulator.
        """
        self.log("Innovations from drifter " + str(drifter_id) + ":\n" + str(innovation))
        
        # Find phi as we go: phi = d^T S d
        phi = 0.0
        
        # Loop over drifters to get half the Kalman gain for each innovation
        for drifter in range(self.numDrifters):
            
            # Reset the random numbers buffer for the given sim to zero:
            self.setNoiseBufferToZero(sim)
        
            local_innovation = innovation[drifter,:]
            observed_drifter_position = all_observed_drifter_positions[drifter,:]
            
            cell_id_x = np.int32(int(np.floor(observed_drifter_position[0]/self.dx)))
            cell_id_y = np.int32(int(np.floor(observed_drifter_position[1]/self.dy)))
            coarse_cell_id_x = np.int32(int(np.floor(observed_drifter_position[0]/self.coarse_dx)))
            coarse_cell_id_y = np.int32(int(np.floor(observed_drifter_position[1]/self.coarse_dy)))
            
            # 1) Solve linear problem
            e = np.dot(self.S_host, local_innovation)
            
            self.halfTheKalmanGainKernel.prepared_async_call(self.global_size_Kalman,
                                                             self.local_size_Kalman,
                                                             sim.gpu_stream,
                                                             self.coarse_nx, self.coarse_ny, 
                                                             self.coarse_dx, self.coarse_dy,
                                                             self.soar_q0, self.soar_L,
                                                             coarse_cell_id_x, coarse_cell_id_y,
                                                             self.geoBalanceConst,
                                                             np.float32(e[0,0]), np.float32(e[0,1]),
                                                             sim.small_scale_model_error.random_numbers.data.gpudata,
                                                             sim.small_scale_model_error.random_numbers.pitch)
            
            phi += local_innovation[0]*e[0,0] + local_innovation[1]*e[0,1]
            
            # The final step of the Kalman gain is to obtain geostrophic balance on the obtained field.
            sim.small_scale_model_error.perturbSim(sim, update_random_field=False,
                                                  align_with_cell_i=cell_id_x, align_with_cell_j=cell_id_y)
        return phi
        # end of addKalmanGain
        #----------------------------------
    
    
    def sampleFromP(self, sim, all_observed_drifter_positions, return_original_random_numbers=False):
        """
        Samples random numbers N(0,I) and applies the covariance structure defined by the
        precomputed SVD at the observation positions.
        The result is written to the random numbers buffer of sim, so that the final step 
        (applying SOAR + geostrophic balance, and scaling) can be done next.
        """
        # Sample from N(0,I)
        sim.small_scale_model_error.generateNormalDistribution()

        std_norm_host = None        
        if return_original_random_numbers:
            std_norm_host = sim.small_scale_model_error.getRandomNumbers()
        
        # Obtain gamma
        sim.gpu_stream.synchronize()
        gamma = sim.small_scale_model_error.getRandomNorm() * self.random_numbers_ratio
        sim.gpu_stream.synchronize()
            
        for drifter in range(self.numDrifters):
            observed_drifter_position = all_observed_drifter_positions[drifter,:]
            
            coarse_cell_id_x = int(np.floor(observed_drifter_position[0]/self.coarse_dx))
            coarse_cell_id_y = int(np.floor(observed_drifter_position[1]/self.coarse_dy))
        
            self.applyLocalSVDOnGlobalXi(sim, coarse_cell_id_x, coarse_cell_id_y)
        
        if return_original_random_numbers:
            return gamma, std_norm_host
        else:
            return gamma
    
    def applyLocalSVDOnGlobal(self, sim, 
                              drifter_coarse_cell_id_x, drifter_coarse_cell_id_y,
                              random_numbers):
        """
        Calls the kernel that applies the covariance structure of the precomputed SVD block 
        centered at the drifter position.
        Since this structure can be applied to the random numbers in both buffers, the buffer
        to use is sent as a reference through the random_numbers parameter.
        """
        # Assuming that the random numbers buffer for the given sim is filled with N(0,I) numbers
        self.localSVDOnGlobalXiKernel.prepared_async_call(self.global_size_SVD,
                                                          self.local_size_SVD,
                                                          sim.gpu_stream,
                                                          self.coarse_nx, self.coarse_ny,
                                                          np.int32(drifter_coarse_cell_id_x),
                                                          np.int32(drifter_coarse_cell_id_y),
                                                          self.localSVD_device.data.gpudata,
                                                          self.localSVD_device.pitch, 
                                                          random_numbers.data.gpudata,
                                                          random_numbers.pitch)

    def applyLocalSVDOnGlobalXi(self, sim, drifter_coarse_cell_id_x, drifter_coarse_cell_id_y):
        """
        Calling applyLocalSVDOnGlobal with the xi buffer
        """
        self.applyLocalSVDOnGlobal(sim, 
                                   drifter_coarse_cell_id_x, drifter_coarse_cell_id_y,
                                   sim.small_scale_model_error.random_numbers)
        
    def applyLocalSVDOnGlobalNu(self, sim, drifter_coarse_cell_id_x, drifter_coarse_cell_id_y):
        """
        Calling applyLocalSVDOnGlobal with the nu buffer
        """
        self.applyLocalSVDOnGlobal(sim, 
                                   drifter_coarse_cell_id_x, drifter_coarse_cell_id_y,
                                   sim.small_scale_model_error.perpendicular_random_numbers)
        
    
    
    def samplePerpendicular(self, sim, return_original_random_numbers=False):
        """
        Samples two perpendicular random vectors from N(0,I)
        """
        # Sample perpendicular xi and nu
        sim.small_scale_model_error.generatePerpendicularNormalDistributions()
        
        orig_xi_host, orig_nu_host = None, None       
        if return_original_random_numbers:
            orig_xi_host = sim.small_scale_model_error.getRandomNumbers()
            orig_nu_host = sim.small_scale_model_error.getPerpendicularRandomNumbers()
        
        # Obtain the norms of
        sim.gpu_stream.synchronize()
        reduction_buffer_host = sim.small_scale_model_error.getReductionBuffer()
        sim.gpu_stream.synchronize()
        
        gamma = reduction_buffer_host[0,0] * self.random_numbers_ratio
        nu_norm = reduction_buffer_host[0,1] * self.random_numbers_ratio
        
        if return_original_random_numbers:
            return gamma, nu_norm, orig_xi_host, orig_nu_host
        else:
            return gamma, nu_norm
        
    def applySVDtoPerpendicular(self, sim, all_observed_drifter_positions, alpha, beta):
        """
        Applies the covariance structure defined by the pre-computed SVD at 
        the drifter positions for both of the perpendicular random vectors.
        
        This operation is performed on the same coarse grid for all drifters,
        meaning that no offset is applied to the fine-coarse grid mapping.
        """
        
        # Update xi = \alpha^{1/2}*xi + \beta^{1/2}*\nu
        self.addBetaNuIntoAlphaXi(sim, alpha, beta)
        
        for drifter in range(self.numDrifters):
            observed_drifter_position = all_observed_drifter_positions[drifter,:]
            
            coarse_cell_id_x = int(np.floor(observed_drifter_position[0]/self.coarse_dx))
            coarse_cell_id_y = int(np.floor(observed_drifter_position[1]/self.coarse_dy))
        
            self.applyLocalSVDOnGlobalXi(sim, coarse_cell_id_x, coarse_cell_id_y)
        
    def applySVDtoPerpendicular_slow(self, sim, all_observed_drifter_positions):
        """
        Applies the covariance structure defined by the pre-computed SVD at 
        the drifter positions for both of the perpendicular random vectors.
        
        This operation is performed on the same coarse grid for all drifters,
        meaning that no offset is applied to the fine-coarse grid mapping.
        
        This slow version is kept for testing purposes.
        """
        
        for drifter in range(self.numDrifters):
            observed_drifter_position = all_observed_drifter_positions[drifter,:]
            
            coarse_cell_id_x = int(np.floor(observed_drifter_position[0]/self.coarse_dx))
            coarse_cell_id_y = int(np.floor(observed_drifter_position[1]/self.coarse_dy))
        
            self.applyLocalSVDOnGlobalXi(sim, coarse_cell_id_x, coarse_cell_id_y)
            self.applyLocalSVDOnGlobalNu(sim, coarse_cell_id_x, coarse_cell_id_y)
        
    
    
    ###------------------------    
    ### CPU Methods
    ###------------------------
    # Parts of the efficient IEWPF method that is solved on the CPU
    
    
    # As we have S = (HQH^T + R)^-1, we can do step 1 of the IEWPF algorithm
    def obtainTargetWeight(self, ensemble, d, w_rest=None):
        if w_rest is None:
            w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())
            
        Ne = ensemble.getNumParticles()
        c = np.zeros(Ne)
        for particle in range(Ne):
            # Obtain db = d^T S d
            db = 0.0
            for drifter in range(ensemble.getNumDrifters()):
                e = np.dot(self.S_host, d[particle,drifter,:])
                db += np.dot(e, d[particle, drifter, :])
            c[particle] = w_rest[particle] + db
            if self.debug: print( "c[" + str(particle) + "]: ", c[particle])
            if self.debug: print ("exp(-c[" + str(particle) + "]: ", np.exp(-c[particle]))
        target_weight = np.max(c)
        self.log('-------------------------------------')
        self.log('w_rest: ' + str(w_rest[0]))
        self.log("c values:\n" + str(c))
        self.log("w_target --> " + str(target_weight))
        return target_weight
    
    def obtainTargetWeightTwoStage(self, c_array, nu_norms):
        """
        Calculates the target weight and the beta parameter for the 
        two-stage IEWPF scheme.
        Returns w_target, beta
        """
        assert(len(c_array) == len(nu_norms))
        
        w_target = np.mean(c_array)
        beta = np.min( (w_target - c_array)/nu_norms) + 1.0
        return w_target, beta
        
    
    ### Solving the implicit equation on the CPU:
    
    def _old_old_implicitEquation(self, alpha, gamma, Nx, a):
        return (alpha-1.0)*gamma - Nx*np.log(alpha) + a
    
    def _old_implicitEquation(self, alpha, gamma, Nx, target_weight, c):
        """
        This is the equation that we now should have solved by using the lambert W function
        """
        return np.log(alpha*alpha*Nx/gamma) - (alpha*alpha*Nx/gamma) - ((target_weight - c)/Nx) + 1
        
    def _implicitEquation_no_limit(self, alpha, gamma, Nx, c_star):
        """
        The implicit equation that should be zero when we solve for alpha.
        This form does not assume that N_x is large.
        """
        lhs = gammainc(Nx/2, alpha*gamma/2)
        rhs = gammainc(Nx/2, gamma/2)
        expo = np.exp(-c_star/2)
        return lhs - expo*rhs
    
    def _implicitEquation_no_limit_derivative(self, alpha, gamma, Nx, c_star):
        """
        The derivative of _implicitEquation_no_limit
        """
        return (alpha*gamma/2)**(Nx/2 - 1) *np.exp(-alpha*gamma/2)*gamma/2
        
    def solveImplicitEquation(self, gamma, 
                              target_weight, w_rest, c_star,
                              particle_id=None):
        """
        Solving the scalar implicit equation using the Lambert W function, and 
        updating the buffers eta_a, hu_a, hv_a as:
        x_a = x_a + alpha*xi
        """
        try:
            self.log("")
            self.log("---- Implicit equation particle " + str(particle_id) + " ---------")

            params = {
                'gamma': gamma,
                'Nx': self.Nx,
                'w_rest': w_rest,
                'target_weight': target_weight,
                'c_star': c_star,
                'particle_id': particle_id
            }
            self.log("Input params:")
            self.log(params)

            #alpha_newton = newton(lambda x: self._implicitEquation_no_limit(x, gamma, self.Nx, c_star),
            #                      0.5, maxiter=2000, tol=1e-6)
            #                      #fprime=lambda x: self._implicitEquation_no_limit_derivative(x, gamma, self.Nx, c_star))
            #self.log("alpha_newton from Newton's method: " + str(alpha_newton))
            #self.log("Discrepancy with alpha_newton: "+ str(self._implicitEquation_no_limit(alpha_newton, gamma, self.Nx, c_star)))

            self.log("")
            self.log("Using the Lambert W:")
            lambert_arg = -(gamma/self.Nx)*np.exp(-gamma/self.Nx)*np.exp(-c_star/self.Nx)
            self.log("\tLambert W arg: " + str(lambert_arg))
            lambert_min1 = lambertw(lambert_arg, k=-1)
            lambert_zero = lambertw(lambert_arg, k=0)

            alpha_scale = -(self.Nx/gamma)
            alpha_min1 = alpha_scale*np.real(lambert_min1)
            alpha_zero = alpha_scale*np.real(lambert_zero)

            self.log("\tlambert_min1 = " + str(lambert_min1) + " --> alpha = " + str(alpha_min1))
            self.log("\tlambert_zero = " + str(lambert_zero) + " --> alpha = " + str(alpha_zero))
            self.log("Discrepancy with alpha_zero: "+ str(self._implicitEquation_no_limit(alpha_zero, gamma, self.Nx, c_star)))


            alpha = alpha_zero
            self.log("returning alpha = " + str(alpha))
            return alpha
        except RuntimeError as re:
            print('Got exception during solveImplicitEquation')
            print(str(re))
            print("Input params:")
            print(params)
            raise
        
    def _createS(self, ensemble):
        """
        Create the 2x2 matrix S = (HQH^T + R)^-1

        Constant as long as
         - the forcing on the drifters, and the drifters themselves, are independent,
         - H(x,y) = const, and
         - double periodic boundary conditions
        """

        # Local storage for x and y correlations:
        x_corr = np.zeros((7,7))
        y_corr = np.zeros((7,7))
        tmp_x = np.zeros((7,7))
        tmp_y = np.zeros((7,7))

        # Mid_coordinates:
        mid_i, mid_j = 3, 3

        # Fill the buffers with U_{GB}^T H^T
        # Spread information from central point to the neighbours according to 
        # geostrophic balance.
        x_corr[mid_j+1, mid_i] = -self.geoBalanceConst/self.coarse_dy
        x_corr[mid_j-1, mid_i] =  self.geoBalanceConst/self.coarse_dy
        y_corr[mid_j, mid_i+1] =  self.geoBalanceConst/self.coarse_dx
        y_corr[mid_j, mid_i-1] = -self.geoBalanceConst/self.coarse_dx
        if self.debug: self.showMatrices(x_corr, y_corr, "$U_{GB}^T  H^T$")
    
        # Apply the SOAR function to fill x and y with 7x5 and 5x7 respectively.
        # Each of the values above is spread according to the SOAR function
        # First for x:
        for j,i in [mid_j+1, mid_i], [mid_j-1, mid_i]:
            for b in range(j-2, j+3):
                for a in range(i-2, i+3):
                    tmp_x[b, a] += x_corr[j,i]*self._SOAR_Q_CPU(a, b, i, j)
        # Then for y:
        for j,i in [mid_j, mid_i+1], [mid_j, mid_i-1]:
            for b in range(j-2, j+3):
                for a in range(i-2, i+3):
                    tmp_y[b, a] += y_corr[j,i]*self._SOAR_Q_CPU(a, b, i, j)
        if self.debug: self.showMatrices(tmp_x, tmp_y, "$Q_{SOAR} U_{GB}^T H^T$")   
        
        
        # Apply the SOARfunction again to fill the points needed to find drift in (mid_i, mid_j)
        # In order to reuse the bicubic interpolation routine, we need to calculate the SOAR values
        # in the 5x5 area centered in the mid-cell.
        # The values outside of the 7x7 buffers are all zero, so we don't have to loop over them.
        for j in range(mid_j-2, mid_j+3):
            for i in range(mid_i-2, mid_i+3):
                x_corr[j,i] = 0
                y_corr[j,i] = 0
                for b in range(max(j-2,0), min(j+3,7)):
                    for a in range(max(i-2,0), min(i+3,7)):
                        SOAR_Q_res = self._SOAR_Q_CPU(a, b, i, j)
                        x_corr[j,i] += tmp_x[b, a]*SOAR_Q_res
                        y_corr[j,i] += tmp_y[b, a]*SOAR_Q_res
                if self.debug: print ("(j, i ,x_corr[j,i], y_corr[j,i]): ", (j, i ,x_corr[j,i], y_corr[j,i]))
        if self.debug: self.showMatrices(x_corr, y_corr, "$Q_{SOAR} Q_{SOAR} U_{GB}^T H^T$")
        
        
        # Obtaining the values required for finding geostrophic balanced values in the center cell.
        # Need only consider one cell in each direction.
        # We find these values by interpolation or direct reading of x_corr and y_corr values
        if self.interpolation_factor > 1:
            
            # Use a function alias to get more readable code
            # This function has input values: coarse_eta, coarse_i, coarse_j, rel_x, rel_y
            interpolation_alias = ensemble.particles[0].small_scale_model_error._bicubic_interpolation_inner
            
            # Relative offset of neighbouring fine grid point
            rel_offset = 1.0/self.interpolation_factor
            
            # north and east values are calculated from the surface in the north-east direction
            # south and west values are calculated from the surface in the south-west direction 
            
            x_north = interpolation_alias(x_corr, mid_i  , mid_j  ,            0.0,     rel_offset)
            x_east  = interpolation_alias(x_corr, mid_i  , mid_j  ,     rel_offset,            0.0)
            x_south = interpolation_alias(x_corr, mid_i-1, mid_j-1,            1.0, 1.0-rel_offset)
            x_west  = interpolation_alias(x_corr, mid_i-1, mid_j-1, 1.0-rel_offset,            1.0)

            y_north = interpolation_alias(y_corr, mid_i  , mid_j  ,            0.0,     rel_offset)
            y_east  = interpolation_alias(y_corr, mid_i  , mid_j  ,     rel_offset,            0.0)
            y_south = interpolation_alias(y_corr, mid_i-1, mid_j-1,            1.0, 1.0-rel_offset)
            y_west  = interpolation_alias(y_corr, mid_i-1, mid_j-1, 1.0-rel_offset,            1.0)        
        
        else:
            x_north = x_corr[mid_j+1, mid_i  ]
            x_south = x_corr[mid_j-1, mid_i  ]
            x_west  = x_corr[mid_j  , mid_i-1]
            x_east  = x_corr[mid_j  , mid_i+1]

            y_north = y_corr[mid_j+1, mid_i  ]
            y_south = y_corr[mid_j-1, mid_i  ]
            y_west  = y_corr[mid_j  , mid_i-1]
            y_east  = y_corr[mid_j  , mid_i+1]
        
        if self.debug: print("[x_north, x_east, x_south, x_west]: ", [x_north, x_east, x_south, x_west])
        if self.debug: print("[y_north, y_east, y_south, y_west]: ", [y_north, y_east, y_south, y_west])
            
        
        # geostrophic balance:
        x_hu = -self.geoBalanceConst*(x_north - x_south)/self.dy
        x_hv =  self.geoBalanceConst*(x_east  - x_west )/self.dx
        y_hu = -self.geoBalanceConst*(y_north - y_south)/self.dy
        y_hv =  self.geoBalanceConst*(y_east  - y_west )/self.dx 

        # Structure the information as a  
        HQHT = np.matrix([[x_hu, y_hu],[x_hv, y_hv]])    
        if self.debug: print ("HQHT\n", HQHT)
        if self.debug: print ("ensemble.observation_cov\n", ensemble.getObservationCov())
        S_inv = HQHT + ensemble.getObservationCov()
        if self.debug: print ("S_inv\n", S_inv)
        S = np.linalg.inv(S_inv)
        if self.debug: print( "S\n", S)
        return S.astype(np.float32, order='C')
    
    
    
        
    
    ###---------------------------
    ### Download GPU buffers
    ###---------------------------
    
    def download_S(self):
        """
        2x2 matrix: S = (H Q H^T + R)^-1
        """
        return self.S_device.download(self.master_stream)
    
    def download_localSVD(self):
        """
        The 49 x 49 matrix that holds the matrix square root of term that turns up in
        the middle of the P covariance matrix.
        """
        return self.localSVD_device.download(self.master_stream)
    
    def download_reduction_buffer(self):
        """
        This buffer holds only a single float, and is used to store the result
        from reduction operations.
        """
        return self.reduction_buffer.download(self.master_stream)
    
        
    def showMatrices(self, x, y, title, z = None):
        num_cols = 2
        if z is not None:
            num_cols = 3
        fig = plt.figure(figsize=(num_cols*2,2))
        plt.subplot(1,num_cols,1)
        plt.imshow(x.copy(), origin="lower", interpolation="None")
        plt.xlabel('(%.2E, %.2E)' % (np.min(x), np.max(x)))
        plt.subplot(1,num_cols,2)
        plt.imshow(y.copy(), origin="lower", interpolation="None")
        plt.xlabel('(%.2E, %.2E)' % (np.min(y), np.max(y)))
        if z is not None:
            plt.subplot(1, num_cols, 3)
            plt.imshow(z.copy(), origin="lower", interpolation="None")
            plt.xlabel('(%.2E, %.2E)' % (np.min(z), np.max(z)))
        plt.suptitle(title)

    ###-----------------------
    ### IEWPF CPU functions
    ###-----------------------
        
    def _SOAR_Q_CPU(self, a_x, a_y, b_x, b_y):
        """
        CPU implementation of a SOAR covariance function between grid points
        (a_x, a_y) and (b_x, b_y) with periodic boundaries
        """
        dist_x = min((a_x - b_x)**2, (a_x - (b_x + self.coarse_nx))**2, (a_x - (b_x - self.coarse_nx))**2)
        dist_y = min((a_y - b_y)**2, (a_y - (b_y + self.coarse_ny))**2, (a_y - (b_y - self.coarse_ny))**2)
        
        dist = np.sqrt( self.coarse_dx*self.coarse_dx*dist_x  +  self.coarse_dy*self.coarse_dy*dist_y)
        
        return self.soar_q0*(1.0 + dist/self.soar_L)*np.exp(-dist/self.soar_L)



    def _createCutoffSOARMatrixQ(self, ensemble, nx=None, ny=None, cutoff=2):
        """
        Creates a full matrix representing Q_{SOAR}^{1/2}.
        Should never be called with large (nx, ny), and should only be used when the
        grid size is really really small, and/or for debuging purposes
        
        Resulting matrix is (nx*ny \times nx*ny)
        """
        if nx is None:
            nx = self.coarse_nx
        if ny is None:
            ny = self.coarse_ny
        
        Q = np.zeros((ny*nx, ny*nx))
        for a_y in range(ny):
            for a_x in range(nx):
            
                # index on the "flattened" grid, and col index in the resulting matrix
                j = a_y*nx + a_x
                
                # Loop over SOAR correlation area
                for b_y in range(a_y-cutoff, a_y+cutoff+1):
                    if b_y < 0:    
                         b_y = b_y + ny
                    if b_y > ny-1: 
                        b_y = b_y - ny
                    for b_x in range(a_x-cutoff, a_x+cutoff+1):
                        if b_x < 0:
                            b_x = b_x + nx
                        if b_x > nx-1: 
                            b_x = b_x - nx
                        i = b_y*nx + b_x
                        
                        # In the SOAR function, we use the ensemble nx and ny, to correctly 
                        # account for periodic boundaries.
                        Q[j, i] = self._SOAR_Q_CPU(a_x, a_y, b_x, b_y)
        return Q


    def _createUGBmatrix(self, ensemble, nx=None, ny=None):
        """
        Creates a full matrix representing Q_{GB}^{1/2}.
        Should never be called with large (nx, ny), and should only be used when the
        grid size is really really small, and/or for debuging purposes.
        
        Resulting matrix is (3*nx*ny \times nx*ny)
        """
        if nx is None:
            nx = self.coarse_nx
        if ny is None:
            ny = self.coarse_ny
        
        I = np.eye(nx*ny)
        A_hu = np.zeros((ny*nx, ny*nx))
        A_hv = np.zeros((ny*nx, ny*nx))
        for a_y in range(ny):
            for a_x in range(nx):
                
                # index on the "flattened" grid, and col index in the resulting block matrix
                j = a_y*nx + a_x 
                
                # geo balance for hu:
                i = (a_y+1)*nx + a_x
                if a_y == ny-1:
                    i = 0*nx + a_x
                A_hu[j,i] = 1.0
                i = (a_y-1)*nx + a_x
                if a_y == 0:
                    i = (ny-1)*nx + a_x
                A_hu[j,i] = -1.0

                # geo balance for hv:
                i = a_y*nx + a_x + 1
                if a_x == nx-1:
                    i = a_y*nx + 0
                A_hv[j,i] = 1.0

                i = a_y*nx + a_x - 1
                if a_x == 0:
                    i = a_y*nx + nx - 1
                A_hv[j,i] = -1.0

        A_hu *= -self.geoBalanceConst/self.coarse_dy
        A_hv *=  self.geoBalanceConst/self.coarse_dx
            
        return np.bmat([[I], [A_hu], [A_hv]])

    def _createMatrixH(self, nx, ny, pos_x, pos_y):
        """
        Creates a full observation matrix H.
        Should never be called with large (nx, ny), and should only be used when the
        grid size is really really small, and/or for debuging purposes
        
        Resulting matrix size: [2, 3*nx*ny]
        """
        H = np.zeros((2, 3*nx*ny))
        index = pos_y*nx + pos_x
        H[0, 1*nx*ny + index] = 1
        H[1, 2*nx*ny + index] = 1
        return H

    def _generateLocaleSVDforP(self, ensemble, returnUSigV=False):
        """
        Generates the local square root of the SVD-block needed for P^1/2.
        This matrix is defined for the coarse grid, and ignores all use of the interpolation operator.

        Finding:   U*Sigma*V^H = I - Q*U_GB^T*H^T*S*H*U_GB*Q
        Returning: U*sqrt(Sigma)
        """

        # Since the structure of the SVD-block is the same for any drifter position, we build the block
        # on a 7x7 domain with the observation in the middle cell
        local_nx = 7
        local_ny = 7
        pos_x = 3
        pos_y = 3

        # Create the matrices needed
        H      = self._createMatrixH(local_nx, local_ny, pos_x, pos_y)
        Q_soar = self._createCutoffSOARMatrixQ(ensemble, nx=local_nx, ny=local_ny)
        U_GB   = self._createUGBmatrix(ensemble, nx=local_nx, ny=local_ny)

        UQ = np.dot(U_GB, Q_soar)
        HUQ = np.dot(H, UQ)
        SHUQ = np.dot(self.S_host, HUQ)
        HTSHUQ = np.dot(H.transpose(), SHUQ)
        UTHTSHUQ = np.dot(U_GB.transpose(), HTSHUQ)
        QUTHTSHUQ = np.dot(Q_soar, UTHTSHUQ)

        svd_input = np.eye(local_nx*local_nx) - QUTHTSHUQ
        
        u, s, vh = np.linalg.svd(svd_input, full_matrices=True)

        if self.debug:
            SVD_prod = np.dot(u, np.dot(np.diag(s), vh))
            fig = plt.figure(figsize=(4,4))
            plt.imshow(SVD_prod, interpolation="None")
            plt.title("SVD_prod")
            plt.colorbar()

            fig = plt.figure(figsize=(4,4))
            plt.imshow(SVD_prod - np.eye(49), interpolation="None")
            plt.title("SVD_prod - I")
            plt.colorbar()

            fig = plt.figure(figsize=(4,4))
            plt.imshow(u, interpolation="None")
            plt.title("u")
            plt.colorbar()

        if returnUSigV:
            return u, s, vh
        
        return np.dot(u, np.diag(np.sqrt(s))).astype(np.float32, order='C')
    
    
  
    
    
    def iewpf_CPU(self, ensemble, infoPlots=None, it=None, perform_step=True):
        """
        The complete IEWPF algorithm implemented on the CPU.
        
        Retrieves innovations and target weights from the ensemble, and updates 
        each particle according to the IEWPF method.
        """
        # Step -1: Deterministic step
        if perform_step:
            t = ensemble.step_truth(ensemble.getDt(), stochastic=True)
            t = ensemble.step_particles(ensemble.getDt(), stochastic=False)


        # Step 0: Obtain innovations
        observed_drifter_position = ensemble.observeTrueDrifters()
        innovations = ensemble.getInnovations()
        w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())

        # save plot halfway
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 1)

        # Step 1: Find maximum weight
        target_weight = self.obtainTargetWeight(ensemble, innovations)
        

        for p in range(ensemble.getNumParticles()):
            
            # Loop step 1: Pull particles towards observation by adding a Kalman gain term
            eta_a, hu_a, hv_a, phi = self.applyKalmanGain_CPU(ensemble.particles[p], \
                                                              observed_drifter_position,
                                                              innovations[p])
            
            

            # Loop step 2: Sample xi \sim N(0, P)
            p_eta, p_hu, p_hv, gamma = self.drawFromP_CPU(ensemble.particles[p], 
                                                          observed_drifter_position)
            xi = [p_eta, p_hu, p_hv] 

            

            # Loop steps 3 and 4: Solve implicit equation and add scaled sample from P to the state vector
            self.applyScaledPSample_CPU(ensemble.particles[p], eta_a, hu_a, hv_a, \
                                        phi, xi, gamma, 
                                        target_weight, w_rest[p])

            # CPU step: Fix boundaries and upload the resulting state to the GPU
            eta_a = self._expand_to_periodic_boundaries(eta_a, 2)
            hu_a  = self._expand_to_periodic_boundaries(hu_a,  2)
            hv_a  = self._expand_to_periodic_boundaries(hv_a,  2)
            ensemble.particles[p].upload(eta_a, hu_a, hv_a)

            
        # save plot after
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 3)
    
    
    def iewpf_original_operation_order_CPU(self, ensemble, infoPlots=None, it=None, perform_step=True):
        """
        The complete IEWPF algorithm (as described by the SotA-18 paper) implemented on the CPU.
        
        Retrieves innovations and target weights from the ensemble, and updates 
        each particle according to the IEWPF method.
        """
        # Step -1: Deterministic step
        if perform_step:
            t = ensemble.step_truth(ensemble.getDt(), stochastic=True)
            t = ensemble.step_particles(ensemble.getDt(), stochastic=False)


        # Step 0: Obtain innovations
        observed_drifter_position = ensemble.observeTrueDrifters()
        innovations = ensemble.getInnovations()
        w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())

        # save plot halfway
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 1)

        # Step 1: Find maximum weight
        target_weight = self.obtainTargetWeight(ensemble, innovations)
        

        for p in range(ensemble.getNumParticles()):
        
            # Step 2: Sample xi \sim N(0, P)
            p_eta, p_hu, p_hv, gamma = self.drawFromP_CPU(ensemble.particles[p], 
                                                          observed_drifter_position)
            xi = [p_eta, p_hu, p_hv] 

            # Step 3: Pull particles towards observation by adding a Kalman gain term
            eta_a, hu_a, hv_a, phi = self.applyKalmanGain_CPU(ensemble.particles[p], \
                                                              observed_drifter_position,
                                                              innovations[p])

            # Step 4: Solve implicit equation and add scaled sample from P
            self.applyScaledPSample_CPU(ensemble.particles[p], eta_a, hu_a, hv_a, \
                                        phi, xi, gamma, 
                                        target_weight, w_rest[p])

            eta_a = self._expand_to_periodic_boundaries(eta_a, 2)
            hu_a  = self._expand_to_periodic_boundaries(hu_a,  2)
            hv_a  = self._expand_to_periodic_boundaries(hv_a,  2)
            ensemble.particles[p].upload(eta_a, hu_a, hv_a)

        # save plot after
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 3)
    
    
    def _expand_to_periodic_boundaries(self, interior, ghostcells):
        """
        Expand a buffer with the requested number of ghost cells, and fills these
        ghost cells according to periodic boundary conditions.
        """
        if ghostcells == 0:
            return interior
        (ny, nx) = interior.shape

        nx_halo = nx + 2*ghostcells
        ny_halo = ny + 2*ghostcells
        newBuf = np.zeros((ny_halo, nx_halo))
        newBuf[ghostcells:-ghostcells, ghostcells:-ghostcells] = interior

        # Fill ghost cells with values according to periodic boundary conditions
        for g in range(ghostcells):
            newBuf[g, :] = newBuf[ny_halo - 2*ghostcells + g, :]
            newBuf[ny_halo - 1 - g, :] = newBuf[2*ghostcells - 1 - g, :]
        for g in range(ghostcells):
            newBuf[:, g] = newBuf[:, nx_halo - 2*ghostcells + g]
            newBuf[:, nx_halo - 1 - g] = newBuf[:, 2*ghostcells - 1 - g]
        return newBuf
    
    def _keepPlot(self, ensemble, infoPlots, it, stage):
        title = "it=" + str(it) + " before IEWPF"
        if stage == 2:
            title = "it=" + str(it) + " during IEWPF (with deterministic step)"
        elif stage == 3:
            title = "it=" + str(it) + " after IEWPF"
        infoFig = EnsemblePlot.plotDistanceInfo(ensemble, title=title, printInfo=False)
        plt.close(infoFig)
        infoPlots.append(infoFig)

        
    def _apply_periodic_boundary(self, index, dim_size):
        if index < 0:
            return index + dim_size
        elif index >= dim_size:
            return index - dim_size
        return index
    
    def _apply_local_SVD_to_global_xi_CPU(self, global_xi, pos_x, pos_y):
        """
        Despite the bad name, this is a good function!

        It takes as input:
         - the global_xi stored in a (ny, nx) buffer
         - the drifter cell position (pos_x, pos_y)
        
        It find the local sqrt (SVD) as U*sqrt(Sigma) in a (49, 49) buffer in self.
        

        The global_xi buffer is modified so that xi = U*sqrt(Sigma)*xi

        Note that we have to make a copy of xi so that we don't read already updated values.

        The function assumes periodic boundary conditions in both dimensions.
        """


        # Copy the result (representing the multiplication with I)
        read_global_xi = global_xi.copy()

        # Read the non-zero structure from tildeP to tildeP_block
        for loc_y_j in range(7):
            global_y_j = pos_y - 3 + loc_y_j
            global_y_j = self._apply_periodic_boundary(global_y_j, self.ny)
            for loc_x_j in range(7):
                global_x_j = pos_x - 3 + loc_x_j
                global_x_j = self._apply_periodic_boundary(global_x_j, self.nx)

                global_j = global_y_j*self.nx + global_x_j
                local_j = loc_y_j*7 + loc_x_j

                xi_j = 0.0
                for loc_y_i in range(7):
                    global_y_i = pos_y - 3 + loc_y_i
                    global_y_i = self._apply_periodic_boundary(global_y_i, self.ny)
                    for loc_x_i in range(7):
                        global_x_i = pos_x - 3 + loc_x_i
                        global_x_i = self._apply_periodic_boundary(global_x_i, self.nx)

                        global_i = global_y_i*self.nx + global_x_i
                        local_i = loc_y_i*7 + loc_x_i

                        xi_j += self.localSVD_host[local_j, local_i]*read_global_xi[global_y_i, global_x_i]

                global_xi[global_y_j, global_x_j] = xi_j

    def drawFromP_CPU(self, sim, observed_drifter_pos):

        # 1) Draw \tilde{\xi} \sim N(0, I)
        sim.small_scale_model_error.generateNormalDistributionCPU()
        if self.debug: print ("noise shape: ", sim.small_scale_model_error.random_numbers_host.shape)

        # Comment in these lines to see how the SVD structure affect the random numbers
        #sim.small_scale_model_error.random_numbers_host *= 0
        #sim.small_scale_model_error.random_numbers_host += 1
        original = None
        if self.debug: original = sim.small_scale_model_error.random_numbers_host.copy()

        # 1.5) Find gamma, which is needed by step 3
        gamma = np.sum(sim.small_scale_model_error.random_numbers_host **2)
        if self.debug: print ("Gamma obtained from standard gaussian: ", gamma)
        if self.debug: self.showMatrices(sim.small_scale_model_error.random_numbers_host, original,\
                                         "Std normal dist numbers")

        # 2) For each drifter, apply the local sqrt SVD-term
        for drifter in range(sim.drifters.getNumDrifters()):

            # 2.1) Find the cell index for the noise.random_numbers_host buffer.
            # This buffer has no ghost cells.
            cell_id_x = int(np.floor(observed_drifter_pos[drifter,0]/sim.dx))
            cell_id_y = int(np.floor(observed_drifter_pos[drifter,1]/sim.dy))

            # 2.2) Apply the local sqrt(SVD)-term
            self._apply_local_SVD_to_global_xi_CPU(sim.small_scale_model_error.random_numbers_host, \
                                                   cell_id_x, cell_id_y)
            if self.debug: self.showMatrices(sim.small_scale_model_error.random_numbers_host, \
                                             original - sim.small_scale_model_error.random_numbers_host, \
                                             "Std normal dist numbers after SVD from drifter " + str(drifter))

        # 3 and 4) Apply SOAR and geostrophic balance
        H_mid = sim.downloadBathymetry()[0]
        p_eta, p_hu, p_hv = sim.small_scale_model_error._obtainOceanPerturbations_CPU(H_mid, sim.f, sim.coriolis_beta, sim.g)

        if self.debug:
            self.showMatrices(p_eta[1:-1, 1:-1], p_hu, "Sample from P", p_hv)
            draw_from_q_maker = OceanStateNoise.OceanStateNoise(sim.gpu_ctx, sim.gpu_stream, \
                                                                sim.nx, sim.ny, sim.dx, sim.dy, \
                                                                sim.boundary_conditions, \
                                                                sim.small_scale_model_error.staggered, \
                                                                sim.small_scale_model_error.soar_q0, \
                                                                sim.small_scale_model_error.soar_L)
            draw_from_q_maker.random_numbers_host = original.copy()
            q_eta, q_hu, q_hv = draw_from_q_maker._obtainOceanPerturbations_CPU(H_mid, sim.f, sim.coriolis_beta, sim.g)
            self.showMatrices(q_eta[1:-1, 1:-1], q_hu, "Equivalent sample from Q", q_hv)
            self.showMatrices(p_eta[1:-1, 1:-1] - q_eta[1:-1, 1:-1], p_hu - q_hu, "diff sample from P and sample from Q", p_hv - q_hv)

        return p_eta[2:-2, 2:-2], p_hu, p_hv, gamma
    
    
    
    def applyKalmanGain_CPU(self, sim, \
                        all_observed_drifter_positions, innovation, \
                        returnKalmanGainTerm=False):
        """
        Creating a Kalman gain type field, K = QH^T S d
        Returning two different values: x_a = x + K, and also phi = d^T S d
        Return eta_a, hu_a, hv_a, phi
        """
        # Following the 3rd step of the IEWPF algorithm
        if self.debug: print ("(nx, ny, dx, dy): ", (sim.nx, sim.ny, sim.dx, sim.dy))
        if self.debug: print ("all_observed_drifter_positions: ", all_observed_drifter_positions)
        if self.debug: print ("innovation:  ", innovation)

        # 0.1) Allocate buffers
        total_K_eta = np.zeros((sim.ny, sim.nx))
        total_K_hu  = np.zeros((sim.ny, sim.nx))
        total_K_hv  = np.zeros((sim.ny, sim.nx))
        phi = 0.0

        # Assume drifters to be far appart, and make a Kalman gain factor for each drifter
        for drifter in range(sim.drifters.getNumDrifters()):

            local_innovation = innovation[drifter,:]
            observed_drifter_position = all_observed_drifter_positions[drifter,:]


            # 0.1) Find the cell index assuming no ghost cells
            cell_id_x = int(np.floor(observed_drifter_position[0]/sim.dx))
            cell_id_y = int(np.floor(observed_drifter_position[1]/sim.dy))
            if self.debug: print ("(cell_id_x, cell_id_y): ", (cell_id_x, cell_id_y))

            # 1) Solve linear problem
            e = np.dot(self.S_host, local_innovation)
            if self.debug: print("e: ", e)

            # 2) K = QH^T e = U_GB Q^{1/2} Q^{1/2} U_GB^T  H^T e
            #    Obtain the Kalman gain
            # 2.1) U_GB^T H^T e
            # 2.1.1) H^T: The transpose of the observation operator now maps the velocity at the 
            #        drifter position to the complete state vector:
            #        H^T [hu(posx, posy), hv(posx, posy)] = [zeros_eta, 0 0 hu(posx, posy) 0 0, 0 0 hv(posx, posy) 0 0]

            # 2.1.2) U_GB^T: map out to laplacian stencil
            local_huhv = np.zeros(4) # representing [north, east, south, west] positions from 
            north_east_south_west_index = [[4,3,0], [3,4,1], [2,3,2], [3,2,3]] #[[y-index-eta, x-index-eta, index-local_eta_soar]]
            # the x-component of the innovation spreads to north and south
            local_huhv[0] = -e[0,0]*self.geoBalanceConst/sim.dy # north 
            local_huhv[2] =  e[0,0]*self.geoBalanceConst/sim.dy # south
            # the y-component of the innovation spreads to east and west
            local_huhv[1] =  e[0,1]*self.geoBalanceConst/sim.dx # east
            local_huhv[3] = -e[0,1]*self.geoBalanceConst/sim.dx # west

            # 2.1.3) Q^{1/2}:
            local_eta = np.zeros((7,7))
            for j,i,soar_res_index in north_east_south_west_index:
                if self.debug: print (j,i), soar_res_index
                for b in range(j-2, j+3):
                    for a in range(i-2, i+3):
                        local_eta[b,a] += local_huhv[soar_res_index]*self._SOAR_Q_CPU(a, b, i, j)
            if self.debug: self.showMatrices(local_eta, local_eta, "local $\eta$ from global $\eta$")

            # 2.2) Apply U_GB Q^{1/2} to the result
            # 2.2.1)  Easiest way: map local_eta to a global K_eta_tmp buffer
            K_eta_tmp = np.zeros((sim.ny, sim.nx))
            if self.debug: print( "K_eta_tmp.shape",  K_eta_tmp.shape)
            for j in range(7):
                j_global = (cell_id_y-3+j+sim.ny)%sim.ny 
                for i in range(7):
                    i_global = (cell_id_x-3+i+sim.nx)%sim.nx 
                    K_eta_tmp[j_global, i_global] += local_eta[j,i]
            if self.debug: self.showMatrices(K_eta_tmp, local_eta, "global K_eta from local K_eta, halfway in the calc.")

            # 2.2.2) Use K_eta_tmp as the noise.random_numbers_host
            sim.small_scale_model_error.random_numbers_host = K_eta_tmp

            # 2.2.3) Apply soar + geo-balance
            H_mid = sim.downloadBathymetry()[0]
            K_eta , K_hu, K_hv = sim.small_scale_model_error._obtainOceanPerturbations_CPU(H_mid, sim.f, sim.coriolis_beta, sim.g)
            if self.debug: self.showMatrices(K_eta[1:-1, 1:-1], K_hu, "Kalman gain from drifter " + str(drifter), K_hv)

            total_K_eta += K_eta[2:-2, 2:-2]
            total_K_hu  += K_hu
            total_K_hv  += K_hv
            if self.debug: self.showMatrices(total_K_eta, total_K_hu, "Total Kalman gain after drifter " + str(drifter), total_K_hv)


            # 3) Obtain phi = d^T * e
            phi += local_innovation[0]*e[0,0] + local_innovation[1]*e[0,1]
            if self.debug: print( "phi after drifter " + str(drifter) + ": ", phi)
        if self.debug: print( "phi: ", phi)

        if returnKalmanGainTerm:
            return total_K_eta, total_K_hu, total_K_hv, phi

        # 4) Obtain x_a
        eta_a, hu_a, hv_a = sim.download(interior_domain_only=True)
        if self.debug: self.showMatrices(eta_a, hu_a, "$M(x)$", hv_a)
        eta_a += total_K_eta
        hu_a += total_K_hu
        hv_a += total_K_hv
        if self.debug: print( "Shapes of x_a: ", eta_a.shape, hu_a.shape, hv_a.shape)
        if self.debug: self.showMatrices(eta_a, hu_a, "$x_a = M(x) + K$", hv_a)
        
        return eta_a, hu_a, hv_a, phi
            
            
    def applyScaledPSample_CPU(self, sim, eta_a, hu_a, hv_a, \
                           phi, xi, gamma, 
                           target_weight, w_rest, particle_id=None):
        """
        Solving the scalar implicit equation using the Lambert W function, and 
        updating the buffers eta_a, hu_a, hv_a as:
        x_a = x_a + alpha*xi
        """


        # 5) obtain gamma
        if self.debug: print ("Shapes of xi: ", xi[0].shape, xi[1].shape, xi[2].shape)
        if self.debug: print ("gamma: ", gamma)
        if self.debug: print ("Nx: ", self.Nx)
        if self.debug: print ("w_rest: ", w_rest)
        if self.debug: print ("target_weight: ", target_weight)
        if self.debug: print ("phi: ", phi)

        # 6) Find a
        a = phi - w_rest + target_weight
        if self.debug: print( "a = phi - w_rest + target_weight: ", a)
            
        c = phi - w_rest
        if self.debug: print ("c = phi - w_rest: ", c)

        # 7) Solving the Lambert W function
        lambert_W_arg = -(gamma/self.Nx)*np.exp(a/self.Nx)*np.exp(-gamma/self.Nx)
        alpha_min1 = -(self.Nx/gamma)*np.real(lambertw(lambert_W_arg, k=-1))
        alpha_zero = -(self.Nx/gamma)*np.real(lambertw(lambert_W_arg))
        if self.debug: print ("Check a against the Lambert W requirement: ", a, " < ", - self.Nx + gamma - self.Nx*np.log(gamma/self.Nx), " = ", a <  - self.Nx + gamma - self.Nx*np.log(gamma/self.Nx))
        if self.debug: print ("-e^-1 < z < 0 : ", -1.0/np.exp(1), " < ", lambert_W_arg, " < ", 0, " = ", \
            (-1.0/np.exp(1) < lambert_W_arg, lambert_W_arg < 0))
        if self.debug: print ("Obtained (alpha k=-1, alpha k=0): ", (alpha_min1, alpha_zero))
        if self.debug: print ("The two branches from Lambert W: ", (lambertw(lambert_W_arg), lambertw(lambert_W_arg, k=-1)))
        if self.debug: print ("The two branches from Lambert W: ", (np.real(lambertw(lambert_W_arg)), np.real(lambertw(lambert_W_arg, k=-1))))

        alpha = alpha_zero
        if lambert_W_arg > (-1.0/np.exp(1)) :
            alpha_u = np.random.rand()
            if alpha_u < 0.5:
                alpha = alpha_min1
                if self.debug: print( "Drew alpha from -1-branch")
        elif self.show_errors:
            print ("!!!!!!!!!!!!")
            print ("BAD BAD ARGUMENT TO LAMBERT W")
            print ("Particle ID: ", particle_id)
            print ("Obtained (alpha k=0, alpha k=-1): ", (alpha_zero, alpha_min1))
            print ("The requirement is lamber_W_arg > (-1.0/exp(1)): " + str(lambert_W_arg) + " > " + str(-1.0/np.exp(1.0)))
            print ("gamma: ", gamma)
            print ("Nx: ", self.Nx)
            print ("w_rest: ", w_rest)
            print ("target_weight: ", target_weight)
            print ("phi: ", phi)
            print ("!!!!!!!!!!!!")
        
        # Comment in these lines to have a look specifically at the solution of
        # the implicit equation.
        #oldDebug = self.debug
        #self.debug = True
        if self.debug: print ("--------------------------------------")
        if self.debug: print ("Obtained (lambert_ans k=0, lambert_ans k=-1): ", (lambertw(lambert_W_arg), lambertw(lambert_W_arg, k=-1)))
        if self.debug: print ("Obtained (alpha k=0, alpha k=-1): ", (alpha_zero, alpha_min1))
        if self.debug: print ("Checking implicit equation with alpha (k=0, k=-1): ", \
            (self._old_implicitEquation(alpha_zero, gamma, self.Nx, a, c), 
             self._old_implicitEquation(alpha_min1, gamma, self.Nx, a, c)))
        
        alpha = np.sqrt(alpha)
        if self.debug: print ("alpha = np.sqrt(alpha) ->  ", alpha)
        if self.debug: print( "--------------------------------------")
        #debug = oldDebug

        # 7.2) alpha*xi
        if self.debug: self.showMatrices(alpha*xi[0], alpha*xi[1], "alpha * xi", alpha*xi[2])

            
        # 8) Final nudge!
        eta_a += alpha*xi[0][1:-1,1:-1]
        hu_a  += alpha*xi[1]
        hv_a  += alpha*xi[2]
        if self.debug: self.showMatrices(eta_a, hu_a, "final x = x_a + alpha*xi", hv_a)

