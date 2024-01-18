# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2024 SINTEF Digital
Copyright (C) 2024 Norwegian Meteorological Institute

Utililty functions for fields represented by Lagrangian particles

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

def lagrangian2concentration(positions, nx, ny, dx, dy, 
                             total_concentration=1.0):
    """
    Maps a list of lagrangian positions to a np.array of shape (ny, nx) with relative concentration of particles within each cell 
    """
    c = np.zeros((ny, nx))
    n = len(positions)
    for i in range(n):
        idx = int(positions[i, 0] // dx)
        idy = int(positions[i, 1] // dy)
        c[idy, idx] += 1
    return (c/n)*total_concentration
    
def concentrationFromSim(sim, dx=None, dy=None, 
                         total_concentration=1.0):
    c =  lagrangian2concentration(sim.drifters.getDrifterPositions(), 
                                  sim.nx, sim.ny, sim.dx, sim.dy,
                                  total_concentration=total_concentration)
    eta, _, _ = sim.download(interior_domain_only=True)
    return np.ma.masked_array(c, eta.mask)