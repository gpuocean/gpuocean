# -*- coding: utf-8 -*-

"""
This python module implements misc helper functions

Copyright (C) 2019 SINTEF ICT

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
from scipy import interpolate 


def fillMaskedValues(input, steps=5):

    def fillStep(a):
        valid = np.zeros(a.shape)
        valid[~a.mask] = 1
        valid[1:-1, 1:-1] += valid[:-2, 1:-1] + valid[2:, 1:-1] + valid[1:-1, :-2] + valid[1:-1, 2:]
        valid[valid == 0] = -1 #avoid divide by zero below

        values = np.copy(a.filled(0))
        values[1:-1, 1:-1] += values[:-2, 1:-1] + values[2:, 1:-1] + values[1:-1, :-2] + values[1:-1, 2:]
        values = values / valid
        values[~a.mask] = 0

        a[values > 0] = values[values > 0]
    
    retval = np.ma.copy(input)
    for i in range(steps):
        fillStep(retval)
        
    return retval



def intersectionsToMidpoints(a_i):
    """
    Converts values at cell intersections to values at midpoints. Use simple averaging
    Also respects masked values when computing the average
    """
    if np.ma.is_masked(a_i):
        valid = np.zeros(a_i.shape, dtype=np.int32)
        valid[~a_i.mask] = 1 
        valid = valid[:-1, :-1] + valid[:-1, 1:] + valid[1:, :-1] + valid[1:, 1:]
        mask = valid<4
        
        #Set values for all elements
        values = a_i.filled(0)
        all_values = values[:-1, :-1] + values[:-1, 1:] + values[1:, :-1] + values[1:, 1:]
        all_values[~mask] = all_values[~mask] / valid[~mask]
        all_values[mask] = a_i.fill_value
        
        return np.ma.array(all_values, mask=mask, fill_value=a_i.fill_value)
    else:
        values = 0.25*(a_i[:-1, :-1] + a_i[:-1, 1:] + a_i[1:, :-1] + a_i[1:, 1:])
   
    return values

    
def midpointsToIntersections(a_m, iterations=20, tolerance=5e-3, use_minmod=False, dt=0.125, land_value=0.0, compute_convergence=False):
    """
    Converts cell values at midpoints to cell values at midpoints using a cubic
    interpolating spline to generate first guess, followed by an iterative update. 
    """
    def genIntersections(midpoints, use_minmod):
        if (use_minmod):
            dx = minmodX(midpoints.data)
            dy = minmodY(midpoints.data)
        else:
            dx, dy = np.gradient(midpoints.data)
            
        mask = None
        if (np.ma.is_masked(midpoints)):
            mask = midpoints.mask
        else:
            mask = np.zeros(midpoints.shape, dtype=np.bool)
        
        #Set slope for masked cells to zero
        dx[mask] = 0.0
        dy[mask] = 0.0
        
        # d - c
        # | X |
        # a - b
        # Evaluate the piecewise planar surface in the four corners of cell X
        a_a = midpoints.data - 0.5*dx - 0.5*dy
        a_b = midpoints.data + 0.5*dx - 0.5*dy
        a_c = midpoints.data + 0.5*dx + 0.5*dy
        a_d = midpoints.data - 0.5*dx + 0.5*dy
        
        
        # Now take the average reconstructed value from the four cells which join
        # in a single cell.
        # d - c   d - c
        # | X |   | X |
        # a - b   a - b
        #       Y   
        # d - c   d - c
        # | X |   | X |
        # a - b   a - b
        a_a = a_a[1:, 1:] * (1-mask[1:, 1:])
        a_b = a_b[:-1, 1:] * (1-mask[:-1, 1:])
        a_c = a_c[:-1, :-1] * (1-mask[:-1, :-1])
        a_d = a_d[1:, :-1] * (1-mask[1:, :-1])
        
        # First count number of valid cells 
        # for each intersection
        count = 4 - (np.int32(mask[1:, 1:]) \
                + np.int32(mask[:-1, 1:]) \
                + np.int32(mask[:-1, :-1]) \
                + np.int32(mask[1:, :-1]))

        # Then set the average
        values = midpoints.data[1:, 1:] + midpoints.data[:-1, 1:] + midpoints.data[:-1, :-1] + midpoints.data[1:, :-1]
        values[count>0] = (a_a[count>0] + a_b[count>0] + a_c[count>0] + a_d[count>0]) / count[count>0]

        #Create mask
        out_mask = (count == 0)
        
        return np.ma.array(values, mask=out_mask, fill_value=midpoints.fill_value)
    
    vmax = a_m.max()
    vmin = a_m.min()
    
    # Generate initial guess
    a_i = genIntersections(a_m, use_minmod=use_minmod)
    a_i = np.clip(a_i, vmin, vmax)
    
    a_i_old = None
    if (compute_convergence):
        a_i_old = a_i.copy()
    
    # Iteratively refine intersections estimate
    #Use kind of a heat equation explisit solver with a source term from the error
    gauss_sigma = 1
    delta = np.zeros_like(a_m)
    u_mask = a_i.mask.copy() #binary_dilation(a_i.mask)
    
    convergence = {'l_1': [], 'l_2': [], 'l_inf': []}
    for i in range(2*iterations+1):        
        delta[1:-1,1:-1] = a_m.data[1:-1,1:-1] - intersectionsToMidpoints(a_i.data)
        
        if (np.ma.is_masked(a_m)):
            delta = np.ma.array(delta, mask=a_m.mask.copy())
        else:
            delta = np.ma.array(delta, mask=np.zeros(a_m.shape, dtype=np.bool))
        
        if (i%2 == 0):
            count = 4 - (np.int32(delta.mask[1:, 1:]) \
                    + np.int32(delta.mask[:-1, 1:]) \
                    + np.int32(delta.mask[:-1, :-1]) \
                    + np.int32(delta.mask[1:, :-1]))
            delta_sum = (delta[:-1, :-1] + delta[:-1, 1:] + delta[1:, 1:] + delta[1:, :-1])
            delta_i = np.zeros(a_i.shape)
            delta_i[count>2] = delta_sum[count>2] / count[count>2]
            a_i[~u_mask] += dt*delta_i[~u_mask]
            
            # Heat equation
            kappa = 1
            dx = 1
            dy = 1
            u = a_i.data.copy()
            
            #rand = (np.random.random_sample(a_i.shape) - 0.5)*l_inf*1e-2
            #u[~a_i.mask] += rand[~a_i.mask]
            
            u[1:-1, 1:-1] = u[1:-1, 1:-1] \
                            + kappa*dt/(dx*dx)*(u[1:-1, :-2] - 2*u[1:-1, 1:-1] + u[1:-1, 2:]) \
                            + kappa*dt/(dy*dy)*(u[:-2, 1:-1] - 2*u[1:-1, 1:-1] + u[2:, 1:-1])
            a_i[~u_mask] = u[~u_mask]
        else:
            # Intersections fix 
            a_i[~a_i.mask] = a_i[~a_i.mask] + genIntersections(delta, use_minmod=use_minmod)[~a_i.mask]
                
        a_i = np.clip(a_i, vmin, vmax)
        a_i.mask = u_mask
        
        #Stop criteria
        if (compute_convergence and i % 2 == 1):
            d = a_i - a_i_old
            a_i_old = a_i.copy()
            d[u_mask] = 0.0
            convergence['l_1'] += [np.sum(np.abs(d))/np.sum(~u_mask)]
            convergence['l_2'] += [np.sum(np.abs(d**2))**(1/2)/np.sum(~u_mask)]
            convergence['l_inf'] += [np.max(np.abs(d))]
            if (convergence['l_1'][0] / convergence['l_1'][0] < tolerance):
                break

    return a_i, convergence


    
    
def calcCoriolisParams(lat):
    #https://en.wikipedia.org/wiki/Coriolis_frequency
    #https://en.wikipedia.org/wiki/Beta_plane
    #Earths rotation rate: 7.2921 × 10−5 rad/s
    omega = 7.2921*1.0e-5
    
    #f is the angular velocity or frequency required to maintain a body at a fixed circle of latitude or zonal region
    f = 2.0*omega*np.sin(lat)
    
    #beta is df/dy where y points north
    #earth mean radius =  6,371.0088 km
    a = 6371008.8
    beta = 2.0*omega*np.cos(lat) / a
    
    return [f, beta]

def degToRad(deg):
    return deg*np.pi/180
    
    
    
def minmodY(eta, theta=1.3):
    backward = theta*(eta[1:-1,:] - eta[:-2,:])
    forward  = theta*(eta[2:,:] - eta[1:-1,:])
    central  = (eta[2:,:] - eta[:-2,:])/2
    
    if np.ma.is_masked(eta):
        backward[eta.mask[1:-1,:]] = 0.0
        forward[eta.mask[1:-1,:]] = 0.0
        central[eta.mask[1:-1,:]] = 0.0
    
    positive = np.minimum(np.minimum(forward * (forward > 0), backward * (backward > 0)), (central * (central > 0)))
    negative = np.maximum(np.maximum(forward * (forward < 0), backward * (backward < 0)), (central * (central < 0)))
    
    retval = np.zeros_like(eta)
    retval[1:-1,:] = positive + negative
    retval[0,:] = theta*(eta[1,:] - eta[0,:])
    retval[-1,:] = theta*(eta[-2,:] - eta[-1,:])
    
    if np.ma.is_masked(eta):
        retval = retval.filled(0.0)
        
    return retval

def minmodX(eta, theta=1.3):
    etaT = np.transpose(eta)
    D = minmodY(etaT)
    return np.transpose(D)

    
def rescaleMidpoints(data, nx1, ny1, **kwargs):
    ny0, nx0 = data.shape

    nx1 = int(nx1)
    ny1 = int(ny1)
    
    if (nx0 > nx1 and ny0 > ny1):
        # Subsample - non volume preserving        
        x0 = np.linspace(0.5, nx0-0.5, nx0)
        y0 = np.linspace(0.5, ny0-0.5, ny0)
        
        data_int = interpolate.interp2d(x0, y0, data, kind='linear', **kwargs)
        
        dx1 = nx0 / nx1
        dy1 = ny0 / ny1
        
        x1 = np.linspace(0.5*dx1, nx0-0.5*dx1, nx1)
        y1 = np.linspace(0.5*dy1, ny0-0.5*dy1, ny1)
        
        out_data = data_int(x1, y1)
        if np.ma.is_masked(data):
            x1, y1 = np.meshgrid(x1, y1)
            out_mask = data.mask[y1.round().astype(np.int32), x1.round().astype(np.int32)]
            out_data = np.ma.array(out_data, mask=out_mask)
            
        return nx0/nx1, ny0/ny1, out_data
        
    else:
        # Minmod reconstruction of slopes (can be volume preserving)
        dx = minmodX(data)
        dy = minmodY(data)
        
        dx1 = nx0 / nx1
        dy1 = ny0 / ny1
        
        x1 = np.linspace(0.5*dx1, nx0-0.5*dx1, nx1)
        y1 = np.linspace(0.5*dy1, ny0-0.5*dy1, ny1)

        x1, y1 = np.meshgrid(x1, y1)
        
        i = np.int32(x1)
        j = np.int32(y1)
        
        s = x1 - (i+0.5)
        t = y1 - (j+0.5)
        
        out_data = data[j, i] + s * dx[j, i] + t * dy[j, i]
        if np.ma.is_masked(data):
            out_mask = data.mask[y1.astype(np.int32), x1.astype(np.int32)]
            out_data = np.ma.array(out_data, mask=out_mask)
        
        return nx0/nx1, ny0/ny1, out_data

def rescaleIntersections(data, nx1, ny1, **kwargs):
    ny0, nx0 = data.shape

    nx1 = int(nx1)
    ny1 = int(ny1)

    if (nx0 > nx1 and ny0 > ny1):
        # Subsample - using linear interpolation
        x0 = np.linspace(0, nx0-1, nx0)
        y0 = np.linspace(0, ny0-1, ny0)
        
        data_int = interpolate.interp2d(x0, y0, data, kind='linear', **kwargs)
        
        x1 = np.linspace(0, nx0-1, nx1)
        y1 = np.linspace(0, ny0-1, ny1)
        
        out_data = data_int(x1, y1)
        if np.ma.is_masked(data):
            x1, y1 = np.meshgrid(x1, y1)
            out_mask = data.mask[y1.round().astype(np.int32), x1.round().astype(np.int32)]
            out_data = np.ma.array(out_data, mask=out_mask)
        
        return (nx0-1)/(nx1-1), (ny0-1)/(ny1-1), out_data
    else:
        # Rescales using bilinear interpolation        
        x1 = np.linspace(0, nx0-1, nx1)
        y1 = np.linspace(0, ny0-1, ny1)
        
        x1, y1 = np.meshgrid(x1, y1)
        
        #Get indices of four nearest neighbors
        i = np.int32(x1)
        j = np.int32(y1)
        k = np.minimum(i+1, nx0-1)
        l = np.minimum(j+1, ny0-1)
        
        #Get interpolation factors
        s = x1 - i
        t = y1 - j
        
        out_data = (1.0-t) * ((1.0-s)*data[j, i] + s*data[j, k]) \
                       + t * ((1.0-s)*data[l, i] + s*data[l, k])
        if np.ma.is_masked(data):
            out_mask = data.mask[y1.round().astype(np.int32), x1.round().astype(np.int32)]
            out_data = np.ma.array(out_data, mask=out_mask)
        
        return (nx0-1)/(nx1-1), (ny0-1)/(ny1-1), out_data
    
    
def calcGeostrophicBalance(eta, H_m, hu, hv, angle, f_beta, dx, dy, g=9.81, use_minmod=False, minmod_theta=1.3, norkyst=False):
    """
    Functionality for evaluating geostropic balance/imbalance in the system.
    If inputs hu and hv are set to None, then geostrophic balanced momenta are returned instead
    """


    #Calculate derivatives
    if (use_minmod):
        DetaDx = minmodX(eta, minmod_theta)/dx
        DetaDy = minmodY(eta, minmod_theta)/dy
    else:
        DetaDx, DetaDy = np.gradient(eta)
        # in NorKyst y is first axis and x is second axis
        if norkyst: 
            DetaDx, DetaDy = DetaDy, DetaDx
        DetaDx = DetaDx / dx
        DetaDy = DetaDy / dy
    
    #Get north and east vectors
    north = [np.sin(angle), np.cos(angle)]
    east = [np.cos(angle), -np.sin(angle)]

    #Calculat derivatives towards north and east for eta
    DetaDeast = east[0]*DetaDx + east[1]*DetaDy
    DetaDnorth = north[0]*DetaDx + north[1]*DetaDy
    
    #Calculate h
    h = H_m + eta
    
    if hu is None and hv is None:
        #Get northward and eastward momentums
        hu_east = -g/f_beta * h * DetaDnorth
        hv_north = g/f * h * DetaDeast

        # Analytic inversion of 2x2 linear system
        det = east[0]*north[1] - east[1]*north[0]

        hu = 1/det * ( north[1] * hu_east - east[1] * hv_north )
        hv = 1/det * ( -north[0] * hu_east + east[0] * hv_north)

        return hu, hv
        
    else: 
        #Get northward and eastward momentums
        hu_east = east[0]*hu + east[1]*hv
        hv_north = north[0]*hu + north[1]*hv
    
        #  f*hv - gh d\eta/dx = 0
        geos_x = f_beta*hv_north - g*h*DetaDeast

        #- f*hu - gh d\eta/dy = 0
        geos_y = -f_beta*hu_east - g*h*DetaDnorth
        
        return [geos_x, geos_y], [f_beta*hv_north, g*h*DetaDeast], [-f_beta*hu_east, g*h*DetaDnorth]

def desingularise(h, hu, eps):
    return hu / np.maximum(np.minimum(h*h/(2.0*eps)+0.5*eps, eps), np.abs(h))


