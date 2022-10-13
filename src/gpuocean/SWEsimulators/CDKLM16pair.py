# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2016 SINTEF ICT, 
Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

This python module implements the finite-volume scheme proposed by
Alina Chertock, Michael Dudzinski, A. Kurganov & Maria Lukacova-Medvidova (2016)
Well-Balanced Schemes for the Shallow Water Equations with Coriolis Forces

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

#Import packages we need
import numpy as np


# Needed for the random perturbation of the wind forcing:
import pycuda.driver as cuda


class CDKLM16pair():
    """
    Class that solves the SW equations using the Coriolis well balanced reconstruction scheme, as given by the publication of Chertock, Dudzinski, Kurganov and Lukacova-Medvidova (CDFLM) in 2016.
    """

    def __init__(self, sim, slave_sim):
        """
        Initialization routine

        sim - CDKLM16 simulation
        """
        
        # Note CDKLM16 instances cannot be copied due to pycuda conflicts,
        # hence we need to initialise two independent simulations outside this class
        self.sim = sim
        self.slave_sim = slave_sim

        self.l = 0


    def step(self, t, apply_stochastic_term=False):
        self.sim.step(t)
        self.slave_sim.step(t)

        def child_level(sim, l):
            if len(sim.children) == 0:
                return l
            else:
                return max([child_level(child,l+1) for child in sim.children])
                
        l = child_level(self.sim, 0)
        if l > self.l:
            self.l = l 

        def scale_level(sim, s):
            if len(sim.children) == 0:
                return s
            else:
                return s * sim.children[0].rescale_factor
        scale_level(self.sim, 1.0)

        