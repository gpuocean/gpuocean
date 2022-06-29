"""
This software is part of GPU Ocean. 

Copyright (C) 2018, 2019, 2022 SINTEF Digital
Copyright (C) 2018, 2019 Norwegian Meteorological Institute

This python module implements atmoshperic forcing, which is used onto 
shallow water models.

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

from asyncio import shield
from re import T
import numpy as np

class AtmosphericPressure():
    """
    Atmospheric pressure for forcing of simplified ocean models.

    Note that normal pressure is 1000 hPa
    High pressure systems might be at 1020 hPa, and low pressure 980 hPa.
    Since the systems are large and since we are only interested in the differences,
    we subtract the "normal" pressure (default: 1000 hPa), so that we get a range -20:20 hPa
    """

    def __init__(self, t=None, P=None, 
                 shiftValue = 100000 # 1000 hPa
                 ):
        self.t = [0]
        self.P = [np.zeros((1,1), dtype=np.float32, order='C')]

        self.numAtmPressures = 1

        if t is not None:
            assert(P is not None), "Missing atmospheric pressure P"

            assert(len(t) == len(P)), str(len(t)) + " vs " + str(len(P))

            self.numAtmPressures = len(t)

            for i in range(len(P)):
                assert(P[i].dtype == "float32"), "Atmospheric pressure needs to be of type np.float32"

            self.t = t
            self.P = P

        self.shiftValue = shiftValue
        if P is None:
            self.shiftValue = 0

        if not (self.shiftValue == 0): 
            for i in range(len(self.P)):
                self.P[i] -= self.shiftValue

    def getOriginalP(self):

        orig_P = [None]*len(self.P)
        for i in range(len(orig_P)):
            orig_P[i] = self.P[i] + self.shiftValue
        
        return orig_P