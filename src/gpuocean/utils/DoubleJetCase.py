# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2019 SINTEF Digital

This python class initialize a instable double jet case, inspired by 
the Galewsky test case:
  - Joseph Galewsky, Richard K. Scott & Lorenzo M. Polvani (2004) An initial-
    value problem for testing numerical models of the global shallow-water equations, Tellus A:
    Dynamic Meteorology and Oceanography, 56:5, 429-440, DOI: 10.3402/tellusa.v56i5.14436

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

import numpy as np
import scipy.integrate
import warnings

if scipy.__version__.startswith('1.4'):
    from scipy.integrate.quadrature import AccuracyWarning
else:
    from scipy.integrate._quadrature import AccuracyWarning
    
from SWESimulators import CDKLM16, Common

class DoubleJetPerturbationType:
    """
    An enum-type class for defining different types of initializations and 
    perturbations for the double jet simulation case.
    """
    
    # Steady state solution
    SteadyState = 1
    
    # Bump located at the standard position x = x_0 for both jets.
    StandardPerturbedState = 2
    
    # Bumps located randomly according to N(x_0, sigma), independent locations for the two jets.
    NormalPerturbedState = 3
    
    # Bumps located at random uniformly distributed along the x-axis, independent for the two jets.
    UniformPerturbedState = 4
    
    # A strong (20x) model error is added to the steady-state
    ModelErrorPerturbation = 5
    
    # The initial state consists of the steady-state after a given spin-up time.
    # Used in a DoubleJetEnsemble, each ensemble member is also given an individual spin up from the
    # already spun-up initial conditions.
    SpinUp = 6
    
    # Same as NormalPerturbedState, but in a DoubleJetEnsemble each ensemble member is spun up individually.
    NormalPerturbedSpinUp = 7
    
    # Similar to SpinUp, but the model error is only applied every 10'th timestep.
    LowFrequencySpinUp = 8

    # Standard deterministic perturbation of the steady-state, with common and individual spin up, and 
    # model errors only every 10th timestep.
    LowFrequencyStandardSpinUp = 9
    
    # IEWPF paper case!
    # Using the dataAssimilationStep with 1 min model time steps with model error and dynamic dt.
    # Initialize with 3 days spin up
    IEWPFPaperCase = 10
    
    @staticmethod
    def _assert_valid(pert_type):
        assert(pert_type == DoubleJetPerturbationType.SteadyState or \
               pert_type == DoubleJetPerturbationType.StandardPerturbedState or \
               pert_type == DoubleJetPerturbationType.NormalPerturbedState or \
               pert_type == DoubleJetPerturbationType.UniformPerturbedState or \
               pert_type == DoubleJetPerturbationType.ModelErrorPerturbation or \
               pert_type == DoubleJetPerturbationType.SpinUp or \
               pert_type == DoubleJetPerturbationType.NormalPerturbedSpinUp or \
               pert_type == DoubleJetPerturbationType.LowFrequencySpinUp or \
               pert_type == DoubleJetPerturbationType.LowFrequencyStandardSpinUp or \
               pert_type == DoubleJetPerturbationType.IEWPFPaperCase), \
        'Provided double jet perturbation type ' + str(pert_type) + ' is invalid'

class DoubleJetCase:
    """
    Class that generates initial conditions for a double jet case (both perturbed and unperturbed)
    """
    
    
    def __init__(self, gpu_ctx, 
                 perturbation_type=DoubleJetPerturbationType.SteadyState,
                 model_error = True, commonSpinUpTime = 200000):
        """
        Class that generates initial conditions for a double jet case (both perturbed and unperturbed).
        The use of initial perturbations/spin up periods are given by the perturbation_type argument,
        which should be a DoubleJetPerturbationType instance.
        """
        # The following parameters are the standard choices we have made for our double jet case.
        # If any of them are to be altered, they should be made optional input parameters to the
        # constructor, with the values below given as default parameters.
        
        # Check that the provided perturbation type is valid
        DoubleJetPerturbationType._assert_valid(perturbation_type)
        self.perturbation_type = perturbation_type
        
        # Domain-related parameters
        self.phi_0 =  72*np.pi/180.0
        self.phi_05 = 75*np.pi/180.0
        self.phi_1 =  78*np.pi/180.0

        self.midpoint_phi_pos = 73.5*np.pi/180
        self.midpoint_phi_neg = 76.5*np.pi/180

        self.phi_delta = 5.5*np.pi/180
        self.phi_pos_min = self.midpoint_phi_pos - self.phi_delta
        self.phi_pos_max = self.midpoint_phi_pos + self.phi_delta
        self.phi_neg_min = self.midpoint_phi_neg - self.phi_delta
        self.phi_neg_max = self.midpoint_phi_neg + self.phi_delta
        self.e_n = np.exp( -4/(self.phi_delta*2)**2)
    
        distance_between_latitudes = 111e3 # m
        degrees_0   = self.phi_0*180/np.pi
        degrees_1   = self.phi_1*180/np.pi
        y_south = degrees_0*distance_between_latitudes
        y_north = degrees_1*distance_between_latitudes
        degrees_mid = self.phi_05*180/np.pi
    
        self.ny = 300
        self.dy = (y_north - y_south)/self.ny
        self.dx = self.dy
        self.nx = 500
        
        self.ghosts = np.array([2,2,2,2]) # north, east, south, west
        self.dataShape = (self.ny+self.ghosts[0]+self.ghosts[2], self.nx+self.ghosts[1]+self.ghosts[3])
        
        # Physical parameters
        self.g = 9.80616              # m/s^2 - gravitational acceleration
        omega = 7.2722e-5              # 1/s  - Angular rotation speed of the earth
        self.earth_radius = 6.37122e6 # m - radius of the Earth
        
        self.u_max = 3 # m/s   - Gulf stream has "maximum speed typically about 2.5 m/s"
        self.h_0 = 230 # m     - It was found to be 230.03, but with a dobious calculation. 
                       #       - Better then just to set the depth to a constant :) 
        self.commonSpinUpTime     = commonSpinUpTime  # s - Because it just seems like a good measure.
        self.individualSpinUpTime = 100000  # s - Because it just seems like a good measure.
        
        self.f = 2*omega*np.sin(self.phi_05)
        self.tan = np.tan(self.phi_05)
        
        
        # Initial data
        sim_h_init, redef_hu_init = self._initSteadyState()
        sim_h_init_mean = sim_h_init.mean()
        
        self.delta_eta = np.max(sim_h_init) - np.min(sim_h_init)
        
        
        max_dt = 0.25*self.dx/(np.max(redef_hu_init/sim_h_init + np.sqrt(self.g*sim_h_init)))
        dt = 0.8*max_dt
        
        self.base_cpu_Hi = np.ones((self.dataShape[0]+1, self.dataShape[1]+1), dtype=np.float32) * sim_h_init_mean
        self.base_cpu_eta = -np.ones(self.dataShape, dtype=np.float32) * sim_h_init_mean
        self.base_cpu_hu = np.zeros(self.dataShape, dtype=np.float32)
        self.base_cpu_hv = np.zeros(self.dataShape, dtype=np.float32)

        for i in range(self.dataShape[1]):
            self.base_cpu_eta[:,i] += sim_h_init
            self.base_cpu_hu[:,i] = redef_hu_init
    
        self.sim_args = {
            "gpu_ctx": gpu_ctx,
            "nx": self.nx, "ny": self.ny,
            "dx": self.dy, "dy": self.dy,
            "dt": dt,
            "g": self.g,
            "f": self.f,
            "coriolis_beta": 0.0,
            "r": 0.0,
            "H": self.base_cpu_Hi, 
            "t": 0.0,
            "rk_order": 2,
            "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
            "small_scale_perturbation": model_error,
            "small_scale_perturbation_amplitude": 0.0003,
            "small_scale_perturbation_interpolation_factor": 5,
        }
        
        self.base_init = {
            "eta0": self.base_cpu_eta, 
            "hu0": self.base_cpu_hu,
            "hv0": self.base_cpu_hv
        }
        
        if self.perturbation_type == DoubleJetPerturbationType.SpinUp or \
           self.perturbation_type == DoubleJetPerturbationType.LowFrequencySpinUp or \
           self.perturbation_type == DoubleJetPerturbationType.LowFrequencyStandardSpinUp:
            if self.perturbation_type == DoubleJetPerturbationType.LowFrequencySpinUp:
                self.commonSpinUpTime = self.commonSpinUpTime
                self.individualSpinUpTime = self.individualSpinUpTime*1.5
            
            
            elif self.perturbation_type == DoubleJetPerturbationType.LowFrequencyStandardSpinUp:
                self.sim_args, self.base_init = self.getStandardPerturbedInitConditions()
                self.commonSpinUpTime = self.commonSpinUpTime*2
                
            tmp_sim = CDKLM16.CDKLM16(**self.sim_args, **self.base_init)
            tmp_t = tmp_sim.step(self.commonSpinUpTime)
            
            tmp_eta, tmp_hu, tmp_hv = tmp_sim.download(interior_domain_only=False)
            self.base_init['eta0'] = tmp_eta
            self.base_init['hu0']  = tmp_hu
            self.base_init['hv0']  = tmp_hv
            self.sim_args['t'] = tmp_sim.t
            tmp_sim.cleanUp()
            
            
        # The IEWPFPaperCase - isolated to give a better overview
        if self.perturbation_type == DoubleJetPerturbationType.IEWPFPaperCase:
            self.sim_args["small_scale_perturbation_amplitude"] = 0.00025
            self.sim_args["model_time_step"] = 60 # sec
            
            tmp_sim = CDKLM16.CDKLM16(**self.sim_args, **self.base_init)
            tmp_sim.updateDt()
            
            three_days = 3*24*60*60
            tmp_t = tmp_sim.dataAssimilationStep(three_days)
            
            tmp_eta, tmp_hu, tmp_hv = tmp_sim.download(interior_domain_only=False)
            self.base_init['eta0'] = tmp_eta
            self.base_init['hu0']  = tmp_hu
            self.base_init['hv0']  = tmp_hv
            self.sim_args['t'] = tmp_sim.t
            tmp_sim.cleanUp()
    
    def __del__(self):
        self.cleanUp()
        
        
    def cleanUp(self):
        # All allocated data needs to be freed from here
        self.gpu_ctx = None
        self.sim_args = None
        self.base_init = None
        
    def getInitConditions(self):
        """
        Provides dicts with initial conditions and constructor arguments suitable for a CDKLM simulator.
        """
        if self.perturbation_type == DoubleJetPerturbationType.StandardPerturbedState:
            return self.getStandardPerturbedInitConditions()
        elif self.perturbation_type == DoubleJetPerturbationType.NormalPerturbedState or \
             self.perturbation_type == DoubleJetPerturbationType.NormalPerturbedSpinUp:
            return self.getNormalPerturbedInitConditions()
        elif self.perturbation_type == DoubleJetPerturbationType.UniformPerturbedState:
            return self.getUniformPerturbedInitConditions()
        else:
            # perturbation type is SteadyState, ModelErrorPerturbation, SpinUp, LowFrequencySpinUp
            return self.getBaseInitConditions()
    
    def getBaseInitConditions(self):
        """
        Provides the unperturbed steady-state double jet initial conditions
        """
        return self.sim_args, self.base_init
    
    def getStandardPerturbedInitConditions(self):
        """
        Provides the standard perturbed double jet initial conditions, using two eta-bumps at x = nx/4
        """
        mid_cell_x_pos = int(self.nx/5) + self.ghosts[3]
        mid_cell_x_neg = int(self.nx/5) + self.ghosts[3]
        return self._create_perturbed_init(mid_cell_x_pos, mid_cell_x_neg)

    def getNormalPerturbedInitConditions(self):
        """
        Provides the standard perturbed double jet initial conditions, 
        using two eta-bumps at slightly perturbed 
        """
        mid_cell_x_pos = np.random.normal(self.nx/5, 10)
        mid_cell_x_neg = np.random.normal(self.nx/5, 10)
        return self._create_perturbed_init(mid_cell_x_pos, mid_cell_x_neg)
        
    def getUniformPerturbedInitConditions(self):
        """
        Provides the standard perturbed double jet initial conditions, using two eta-bumps at random x-positions
        """
        mid_cell_x_pos = int(np.random.rand()*self.nx + self.ghosts[3])
        mid_cell_x_neg = int(np.random.rand()*self.nx + self.ghosts[3])
        return self._create_perturbed_init(mid_cell_x_pos, mid_cell_x_neg)
        
    def _create_perturbed_init(self, mid_cell_x_pos, mid_cell_x_neg):
        """
        Creates initial conditions with perturbations in eta according to the indices given as input.
        """
        eta_pert = np.zeros(self.dataShape)
        
        distance_between_longitudes_75 = 28.7e3 # m 
        distance_between_latitudes = 111e3 # m
        radius_y_cells = distance_between_longitudes_75*180/self.dx

        mid_cell_y_pos = int(1*self.ny/4)
        mid_cell_y_neg = int(3*self.ny/4)
        pert_alpha = self.phi_delta #1/6 # 1/3
        pert_beta = self.phi_delta/5 # 1/30 # 1/15
        h_hat = 0.12*self.delta_eta

        for j in range(self.ny+self.ghosts[2]+self.ghosts[0]):
            for i in range(self.nx+self.ghosts[3]+self.ghosts[1]):

                cell_diff_x_pos = i-mid_cell_x_pos
                cell_diff_x_pos = min(abs(cell_diff_x_pos), abs(cell_diff_x_pos+self.nx), abs(cell_diff_x_pos-self.nx))
                cell_diff_x_neg = i-mid_cell_x_neg
                cell_diff_x_neg = min(abs(cell_diff_x_neg), abs(cell_diff_x_neg+self.nx), abs(cell_diff_x_neg-self.nx))


                squared_dist_y_pos = ((1/pert_beta)*(np.pi/180)*(j-mid_cell_y_pos)*self.dy/distance_between_latitudes)**2
                squared_dist_y_neg = ((1/pert_beta)*(np.pi/180)*(j-mid_cell_y_neg)*self.dy/distance_between_latitudes)**2
                squared_dist_x_pos = ((1/pert_alpha)*(np.pi/180)*(cell_diff_x_pos)*self.dx/(distance_between_longitudes_75))**2
                squared_dist_x_neg = ((1/pert_alpha)*(np.pi/180)*(cell_diff_x_neg)*self.dx/(distance_between_longitudes_75))**2

                lat = np.cos(75*np.pi/180) # approximation into the beta-plane


                eta_pert[j,i] += h_hat*lat*np.exp(-squared_dist_y_pos - squared_dist_x_pos) +\
                                 h_hat*lat*np.exp(-squared_dist_y_neg - squared_dist_x_neg)

        return self.sim_args, {"eta0": self.base_cpu_eta + eta_pert, "hu0": self.base_cpu_hu, "hv0": self.base_cpu_hv}
        
    ###-----------------------------------------------------------------
    ### Utility functions for creating the stable initial case
    ###-----------------------------------------------------------------
    def _initSteadyState(self):
        """
        Main function for creating the unperturbed steady-state initial conditions
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=AccuracyWarning)
            #warnings.simplefilter('ignore', scipy.integrate.quadrature.AccuracyWarning)
            #warnings.simplefilter('ignore', AccuracyWarning)
            #warnings.simplefilter('ignore',DeprecationWarning)
        
            # The initial conditions are created through four steps, here as a cross section along y
            # 1. Calculate $u_{temp}$ based on the expression for initial $u$ from the paper
            # 2. Calculate initial $h_{init}$ using the expression for $u_{temp}$.
            # 3. Re-calculate initial $u_{init}$ by using the expression for geostrophic balance on the initial $h_{init}$.
            # 4. Obtain $hu_{init} = h_{init} u_{init}$.

            dy_phi = (self.phi_1 - self.phi_0)/self.ny
            sim_phi = np.linspace(self.phi_0 - 2*dy_phi, self.phi_1 + 2*dy_phi, self.ny+4)
            
            # 1)
            sim_u_init = self._init_u(sim_phi)
            
            # 2)
            sim_h_init = self._generate_h0(sim_phi, self.phi_0)

            sim_h_init_mean = np.mean(sim_h_init)
            
            # Calculate hu which is in geotrophic balance wrt sim_h_init (it's slope is equal to the slope of eta)
            redef_hu_init = np.zeros_like(sim_h_init)
            for j in range(1, len(redef_hu_init)-1):
                redef_hu_init[j] = - (self.g*sim_h_init_mean/self.f)*(sim_h_init[j+1]-sim_h_init[j-1])/(2*self.dy)

            return sim_h_init, redef_hu_init

    
    def _init_u_scalar(self, lat):
        """
        The initialization function used by Galewsky
        """
        if lat < self.phi_05:
            return (self.u_max/self.e_n) *np.exp(1/((lat-self.phi_pos_min)*(lat-self.phi_pos_max)))
        else:
            return -(self.u_max/self.e_n) *np.exp(1/((lat-self.phi_neg_min)*(lat-self.phi_neg_max)))

    def _init_u(self, lat):
        """
        Initializing u according to Galewsky
        """
        steps = 1
        if np.isscalar(lat):
            return steps*self._init_u_scalar(lat)
        else:
            out = np.zeros_like(lat)
            for i in range(len(lat)):
                if lat[i] > self.phi_0 and lat[i] <= self.phi_1:
                    out[i] = self._init_u_scalar(lat[i])
                if out[i] == np.inf:
                    out[i] = 0.0
            return steps*out
    
    # Integrand for initialization of h
    def _init_h_integrand(self, lat):
        """
        Integrand in Galewsky's expression for initial h
        """
        return self.earth_radius*self._init_u(lat)*(self.f + (self.tan/self.earth_radius)*self._init_u(lat))
    
    def _generate_h0(self, lat, lat_0):
        """
        Initializing gh according to galewsky
        """
        gh0 = np.zeros_like(lat)

        for i in range(lat.size):
            gh0[i] = self.g*self.h_0 - scipy.integrate.quadrature(self._init_h_integrand, self.phi_0, lat[i])[0]
        return gh0/self.g
