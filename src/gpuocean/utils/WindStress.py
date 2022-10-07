# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018, 2019, 2022 SINTEF Digital
Copyright (C) 2018, 2019 Norwegian Meteorological Institute

This python module implements wind forcing, which is used onto 
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

from ctypes import *
from distutils import dep_util
import numpy as np


import warnings
import functools

from abc import ABCMeta, abstractmethod
from gpuocean.utils.Common import deprecated



class WindStress():
    
    def __init__(self, source_filename=None, t=None, X=None, Y=None, u_wind=None, v_wind=None):
        
        self.source_filename = source_filename
        self.t = [0]
        self.X = [np.zeros((1,1), dtype=np.float32, order='C')]
        self.Y = [np.zeros((1,1), dtype=np.float32, order='C')]
        
        self.numWindSteps = 1
        
        if t is not None:
            if (X is None and Y is None) and (u_wind is not None or v_wind is not None):
                assert(u_wind is not None), "missing wind in x u_wind"
                assert(v_wind is not None), "missing wind in y v_wind"
            
                assert(len(t) == len(u_wind)), str(len(t)) + " vs " + str(len(u_wind))
                assert(len(t) == len(v_wind)), str(len(t)) + " vs " + str(len(v_wind))

                X, Y = self._compute_wind_stress_from_wind(u_wind, v_wind)

            assert(X is not None), "missing wind forcing X"
            assert(Y is not None), "missing wind forcing Y"
            
            assert(len(t) == len(X)), str(len(t)) + " vs " + str(len(X))
            assert(len(t) == len(Y)), str(len(t)) + " vs " + str(len(Y))

            self.numWindSteps = len(t)
            
            for i in range(len(X)):
                assert (X[i].dtype == 'float32'), "Wind data needs to be of type np.float32"
                assert (Y[i].dtype == 'float32'), "Wind data needs to be of type np.float32"
            
            self.t = t
            self.X = X
            self.Y = Y
            
    def _compute_wind_stress_from_wind(self, u_wind, v_wind):
 
        if type(u_wind) is list:
            u_wind = np.stack(u_wind, axis=0)
        if type(v_wind) is list:
            v_wind = np.stack(v_wind, axis=0)
        
        print(type(u_wind), type(v_wind))
        u_wind = u_wind.astype(np.float32)
        v_wind = v_wind.astype(np.float32)
        
        wind_speed = np.sqrt(np.power(u_wind, 2) + np.power(v_wind, 2))

        # C_drag as defined by Engedahl (1995)
        #(See "Documentation of simple ocean models for use in ensemble predictions. Part II: Benchmark cases"
        #at https://www.met.no/publikasjoner/met-report/met-report-2012 for details.) /
        def computeDrag(wind_speed):
            C_drag = np.where(wind_speed < 11, 0.0012, 0.00049 + 0.000065*wind_speed)
            return C_drag
        C_drag = computeDrag(wind_speed)

        rho_a = 1.225 # Density of air
        rho_w = 1025 # Density of water

        #Wind stress is then 
        # tau_s = rho_a * C_drag * |W|W
        wind_stress = C_drag * wind_speed * rho_a / rho_w
        wind_stress_u = wind_stress*u_wind
        wind_stress_v = wind_stress*v_wind
    
        return wind_stress_u, wind_stress_v
            

    
class WIND_STRESS_PARAMS(Structure):
    """Mapped to struct WindStressParams in common.cu
    DO NOT make changes here without changing common.cu accordingly!
    """
    
    _fields_ = [("wind_stress_type", c_int),
                ("tau0", c_float),
                ("rho", c_float),
                ("rho_air", c_float),
                ("alpha", c_float),
                ("xm", c_float),
                ("Rc", c_float),
                ("x0", c_float),
                ("y0", c_float),
                ("u0", c_float),
                ("v0", c_float),
                ("wind_speed", c_float),
                ("wind_direction", c_float)]

class BaseWindStress(object):
    """Superclass for wind stress params."""
    
    __metaclass__ = ABCMeta

    def __init__(self):
        pass
    
    @abstractmethod
    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        pass
    
    @abstractmethod
    def tostruct(self):
        """Return correct WindStressParams struct (defined above AND in common.cu)"""
        pass
    
    def csize(self):
        """Return size (in bytes) of WindStressParams struct (defined above AND in common.cu)"""
        return sizeof(WIND_STRESS_PARAMS)

class NoWindStress(BaseWindStress):
    """No wind stress."""

    def __init__(self):
        assert(False), "This is a deprecated wind stress definition. Please use the WindStress class!"
    
    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 0
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type())
        return wind_stress

class GenericUniformWindStress(BaseWindStress):
    """Generic uniform wind stress.
    
    rho_air: Density of air (approx. 1.3 kg / m^3 at 0 deg. C and 1013.25 mb)
    speed: Wind speed in m/s
    direction: Wind direction in degrees (clockwise, 0 being wind blowing from north towards south)
    """

    def __init__(self, \
                 rho_air=0, \
                 wind_speed=0, wind_direction=0):
        assert(False), "This is a deprecated wind stress definition. Please use the WindStress class!"
        self.rho_air = np.float32(rho_air)
        self.wind_speed = np.float32(wind_speed)
        self.wind_direction = np.float32(wind_direction)

    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 1
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type(), 
                                  rho_air=self.rho_air,
                                  wind_speed=self.wind_speed,
                                  wind_direction=self.wind_direction)
        return wind_stress

class UniformAlongShoreWindStress(BaseWindStress):
    """Uniform along shore wind stress.
    
    tau0: Amplitude of wind stress (Pa)
    rho: Density of sea water (1025.0 kg / m^3)
    alpha: Offshore e-folding length (1/(10*dx) = 5e-6 m^-1)
    """

    def __init__(self, \
                 tau0=0, rho=0, alpha=0):
        assert(False), "This is a deprecated wind stress definition. Please use the WindStress class!"
        self.tau0 = np.float32(tau0)
        self.rho = np.float32(rho)
        self.alpha = np.float32(alpha)

    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 2
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type(), 
                                  tau0=self.tau0,
                                  rho=self.rho,
                                  alpha=self.alpha)
        return wind_stress

class BellShapedAlongShoreWindStress(BaseWindStress):
    """Bell shaped along shore wind stress.
    
    xm: Maximum wind stress for bell shaped wind stress
    tau0: Amplitude of wind stress (Pa)
    rho: Density of sea water (1025.0 kg / m^3)
    alpha: Offshore e-folding length (1/(10*dx) = 5e-6 m^-1)
    """

    def __init__(self, \
                 xm=0, tau0=0, rho=0, alpha=0):
        assert(False), "This is a deprecated wind stress definition. Please use the WindStress class!"
        self.xm = np.float32(xm)
        self.tau0 = np.float32(tau0)
        self.rho = np.float32(rho)
        self.alpha = np.float32(alpha)

    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 3
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type(), 
                                  xm=self.xm,
                                  tau0=self.tau0,
                                  rho=self.rho,
                                  alpha=self.alpha)
        return wind_stress

class MovingCycloneWindStress(BaseWindStress):
    """Moving cyclone wind stress.
    
    Rc: Distance to max wind stress from center of cyclone (10dx = 200 000 m)
    x0: Initial x position of moving cyclone (dx*(nx/2) - u0*3600.0*48.0)
    y0: Initial y position of moving cyclone (dy*(ny/2) - v0*3600.0*48.0)
    u0: Translation speed along x for moving cyclone (30.0/sqrt(5.0))
    v0: Translation speed along y for moving cyclone (-0.5*u0)
    """

    def __init__(self, \
                 Rc=0, \
                 x0=0, y0=0, \
                 u0=0, v0=0):
        assert(False), "This is a deprecated wind stress definition. Please use the WindStress class!"
        self.Rc = np.float32(Rc)
        self.x0 = np.float32(x0)
        self.y0 = np.float32(y0)
        self.u0 = np.float32(u0)
        self.v0 = np.float32(v0)

    def type(self):
        """Mapping to wind_stress_type (defined in common.cu)"""
        return 4
    
    def tostruct(self):
        """Return correct WindStressParams struct (defined in common.cu)"""
        wind_stress = WIND_STRESS_PARAMS(wind_stress_type=self.type(), 
                                  Rc=self.Rc,
                                  x0=self.x0,
                                  y0=self.y0,
                                  u0=self.u0,
                                  v0=self.v0)
        return wind_stress

@deprecated
def WindStressParams(type=99, # "no wind" \
                 tau0=0, rho=0, alpha=0, xm=0, Rc=0, \
                 x0=0, y0=0, \
                 u0=0, v0=0, \
                 wind_speed=0, wind_direction=0):
    """
    Backward compatibility function to avoid rewriting old code and notebooks.
    
    SHOULD NOT BE USED IN NEW CODE! Make WindStress object directly instead.
    """
    
    type_ = np.int32(type)
    tau0_ = np.float32(tau0)
    rho_ = np.float32(rho)
    rho_air_ = np.float32(1.3) # new parameter
    alpha_ = np.float32(alpha)
    xm_ = np.float32(xm)
    Rc_ = np.float32(Rc)
    x0_ = np.float32(x0)
    y0_ = np.float32(y0)
    u0_ = np.float32(u0)
    v0_ = np.float32(v0)
    wind_speed_ = np.float32(wind_speed)
    wind_direction_ = np.float32(wind_direction)
    
    if type == 0:
        wind_stress = UniformAlongShoreWindStress( \
            tau0=tau0_, rho=rho_, alpha=alpha_)
    elif type == 1:
        wind_stress = BellShapedAlongShoreWindStress( \
            xm=xm_, tau0=tau0_, rho=rho_, alpha=alpha_)
    elif type == 2:
        wind_stress = MovingCycloneWindStress( \
            Rc=Rc_, x0=x0_, y0=y0_, u0=u0_, v0=v0_)
    elif type == 50:
        wind_stress = GenericUniformWindStress( \
            rho_air=rho_air_, wind_speed=wind_speed_, wind_direction=wind_direction_)
    elif type == 99:
        wind_stress = NoWindStress()
    else:
        raise RuntimeError('Invalid wind stress type!')
    
    return wind_stress
