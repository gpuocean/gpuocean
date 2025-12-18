# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2019 Norwegian Meteorological Institute
Copyright (C) 2019, 2025 SINTEF Digital

This python module implements reading initial data from netcdf 
produced by the barents ensemble model

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



import time
import numpy as np
import xarray as xr

import datetime, os, copy
from netCDF4 import Dataset, MFDataset
import pyproj
from scipy.ndimage.morphology import binary_erosion, grey_dilation


import seawater as sw
from scipy.ndimage.filters import convolve, gaussian_filter

from gpuocean.utils import Common, WindStress, OceanographicUtilities


def getBarentsEnsembleMember(member, casename, timestep_indices=np.arange(24), return_info_only=False, **kwargs):
    """
    We use predefined ensemble members, since (at the time of writing) this is the only barotropic data we have
    """
    member_info = {
          '0': {'day': '13', 'T': '00', 'm': '05', 't_offset':  0},
          '1': {'day': '13', 'T': '00', 'm': '04', 't_offset':  0},
          '2': {'day': '13', 'T': '00', 'm': '03', 't_offset':  0},
          '3': {'day': '13', 'T': '00', 'm': '02', 't_offset':  0},
          '4': {'day': '13', 'T': '00', 'm': '01', 't_offset':  0},
          '5': {'day': '13', 'T': '00', 'm': '00', 't_offset':  0},
          '6': {'day': '12', 'T': '18', 'm': '23', 't_offset': 24},
          '7': {'day': '12', 'T': '18', 'm': '22', 't_offset': 24},
          '8': {'day': '12', 'T': '18', 'm': '21', 't_offset': 24},
          '9': {'day': '12', 'T': '18', 'm': '20', 't_offset': 24},
         '10': {'day': '12', 'T': '18', 'm': '19', 't_offset': 24},
         '11': {'day': '12', 'T': '18', 'm': '18', 't_offset': 24},
         '12': {'day': '12', 'T': '12', 'm': '17', 't_offset': 24},
         '13': {'day': '12', 'T': '12', 'm': '16', 't_offset': 24},
         '14': {'day': '12', 'T': '12', 'm': '15', 't_offset': 24},
         '15': {'day': '12', 'T': '12', 'm': '14', 't_offset': 24},
         '16': {'day': '12', 'T': '12', 'm': '13', 't_offset': 24},
         '17': {'day': '12', 'T': '12', 'm': '12', 't_offset': 24},
         '18': {'day': '12', 'T': '06', 'm': '11', 't_offset': 24},
         '19': {'day': '12', 'T': '06', 'm': '10', 't_offset': 24},
         '20': {'day': '12', 'T': '06', 'm': '09', 't_offset': 24},
         '21': {'day': '12', 'T': '06', 'm': '08', 't_offset': 24},
         '22': {'day': '12', 'T': '06', 'm': '07', 't_offset': 24},
         '23': {'day': '12', 'T': '06', 'm': '06', 't_offset': 24},
         '24': {'day': '12', 'T': '00', 'm': '05', 't_offset': 24},
         '25': {'day': '12', 'T': '00', 'm': '04', 't_offset': 24},
         '26': {'day': '12', 'T': '00', 'm': '03', 't_offset': 24},
         '27': {'day': '12', 'T': '00', 'm': '02', 't_offset': 24},
         '28': {'day': '12', 'T': '00', 'm': '01', 't_offset': 24},
         '29': {'day': '12', 'T': '00', 'm': '00', 't_offset': 24},
         '30': {'day': '11', 'T': '18', 'm': '23', 't_offset': 48},
         '31': {'day': '11', 'T': '18', 'm': '22', 't_offset': 48},
         '32': {'day': '11', 'T': '18', 'm': '21', 't_offset': 48},
         '33': {'day': '11', 'T': '18', 'm': '20', 't_offset': 48},
         '34': {'day': '11', 'T': '18', 'm': '19', 't_offset': 48},
         '35': {'day': '11', 'T': '18', 'm': '18', 't_offset': 48},
         '36': {'day': '11', 'T': '12', 'm': '17', 't_offset': 48},
         '37': {'day': '11', 'T': '12', 'm': '16', 't_offset': 48},
         '38': {'day': '11', 'T': '12', 'm': '15', 't_offset': 48},
         '39': {'day': '11', 'T': '12', 'm': '14', 't_offset': 48},
         '40': {'day': '11', 'T': '12', 'm': '13', 't_offset': 48},
         '41': {'day': '11', 'T': '12', 'm': '12', 't_offset': 48},
         '42': {'day': '11', 'T': '06', 'm': '11', 't_offset': 48},
         '43': {'day': '11', 'T': '06', 'm': '10', 't_offset': 48},
         '44': {'day': '11', 'T': '06', 'm': '09', 't_offset': 48},
         '45': {'day': '11', 'T': '06', 'm': '08', 't_offset': 48},
         '46': {'day': '11', 'T': '06', 'm': '07', 't_offset': 48},
         '47': {'day': '11', 'T': '06', 'm': '06', 't_offset': 48},
         '48': {'day': '11', 'T': '00', 'm': '05', 't_offset': 48},
         '49': {'day': '11', 'T': '00', 'm': '04', 't_offset': 48},
         '50': {'day': '11', 'T': '00', 'm': '03', 't_offset': 48},
         '51': {'day': '11', 'T': '00', 'm': '02', 't_offset': 48},
         '52': {'day': '11', 'T': '00', 'm': '01', 't_offset': 48},
         '53': {'day': '11', 'T': '00', 'm': '00', 't_offset': 48},
    }
    assert(member >= 0)
    assert(member <=53)
    member = str(member)
    d = member_info[member]['day']
    T = member_info[member]['T']
    m = member_info[member]['m']
    
    # Examples:
    # barents_url = "https://thredds.met.no/thredds/dodsC/fou-hi/barents_eps_surface/2024/05/12/T18Z/barents_sfc_20240512T18Zm23.nc" 
    # barotropic_url =         "https://thredds.met.no/thredds/dodsC/gpuocean/barents_eps/barents2500_barotropic_20240512T18Zm23.nc" 
    barents_url =   "https://thredds.met.no/thredds/dodsC/fou-hi/barents_eps_surface/2024/05/"+d+"/T"+T+"Z/barents_sfc_202405"+d+"T"+T+"Zm"+m+".nc" 
    barotropic_url ="https://thredds.met.no/thredds/dodsC/gpuocean/barents_eps/barents2500_barotropic_202405"+d+"T"+T+"Zm"+m+".nc" 

    if isinstance(timestep_indices, list):
        timestep_indices = np.array(timestep_indices)

    time_indices_offset = member_info[member]['t_offset']
    timestep_indices = timestep_indices + time_indices_offset

    if return_info_only:
        return barents_url, barotropic_url, timestep_indices

    return getInitialConditionsBarentsCases(barents_url, barotropic_url, casename, timestep_indices=timestep_indices, **kwargs)


def getBarentsSubdomains():
    """
    Lists (and defines) predefined subdomains for the Barents model.
    """
    return [
        {'name': 'covering_norkyst',   'x0': 379, 'x1': 712, 'y0':  1, 'y1': 496},
        {'name': 'covering_finnmark',  'x0': 420, 'x1': 672, 'y0':230, 'y1': 496},
        {'name': 'covering_north_cape','x0': 450, 'x1': 600, 'y0':270, 'y1': 456},
        {'name': 'complete_barents',   'x0':   5, 'x1': 734, 'y0':  5, 'y1': 943},
        {'name': 'finnmark',           'x0': 430, 'x1': 700, 'y0': 300, 'y1': 550},
    ]


def getInitialConditionsBarentsCases(barents_url, barotropic_url, casename, **kwargs):
    """
    Initial conditions for pre-defined areas within the Barents model domain. 
    """
    use_case = getCaseLocation(casename)
    return getInitialConditions(barents_url, barotropic_url, use_case['x0'], use_case['x1'], use_case['y0'], use_case['y1'], **kwargs)

def getCaseLocation(casename):
    """
    Domains for pre-defined areas within the Barents model domain. 
    """
    cases = getBarentsSubdomains()
    use_case = None
    for case in cases:
        if case['name'] == casename:
            use_case = case
            break

    assert(use_case is not None), 'Invalid case. Check BarentsInitializer.getBarentsSubdomains() to see valid case names'

    return use_case


def getInitialConditions(barents_url, barotropic_url, x0, x1, y0, y1, \
                         timestep_indices=np.arange(24), \
                         land_value=5.0, \
                         iterations=10, \
                         sponge_cells={'north':20, 'south': 20, 'east': 20, 'west': 20}, \
                         erode_land=0, 
                         download_data=False
                         ):
    """
    Constructing input arguments for CDKLM16 instances
    barents_url        - url to NetCDF-files representing one operational EPS member 
    barotropic_url     - url to NetCDF-files representing the GPU-ocean tailored EPS member 
    x0, x1, y0, y1     - subdomain
    timestep_indices   - list with timestep_indices "[12,13,14,15,16]"
    download_data      - downloading the source file for faster initialization next time - warning: downloaded files might be large
    """
    
    ic = {}
    
    #source_url_list[i] = checkCachedNetCDF(source_url_list[i], download_data=download_data)
    
    # Check that the urls have the same timestamp and member id 
    assert(barents_url[-18:] == barotropic_url[-18:])

    # Make dataset objects with xarray
    ds_barotropic = xr.open_dataset(barotropic_url)
    ds_barents = xr.open_dataset(barents_url)

    # Check that the files contain the same time steps:
    assert(np.all(ds_barents.time.data == ds_barotropic.ocean_time.data))

    t0_index = timestep_indices[0]
    timesteps_absolute = ds_barents.time.data[timestep_indices].astype('datetime64[s]')
    timesteps = timesteps_absolute - timesteps_absolute[0]
    timesteps = timesteps.astype(np.float32)
    assert(np.all(np.diff(timesteps) >= 0))

    # Read barotropic parameters:
    # add one frame of cells around eta temporarily
    eta0 = ds_barotropic.zeta.isel(ocean_time=t0_index, xi_rho=slice(x0-1, x1+1), eta_rho=slice(y0-1, y1+1)).data
    u0 = ds_barotropic.ubar.isel(ocean_time=t0_index, xi_u=slice(x0, x1+1), eta_u=slice(y0, y1  )).data
    v0 = ds_barotropic.vbar.isel(ocean_time=t0_index, xi_v=slice(x0, x1  ), eta_v=slice(y0, y1+1)).data
    
    # NOTE: xarray gives nan on the land mask, whereas Dataset gives masked arrays

    #Find u,v at cell centers
    # u0 = u0.filled(fill_value = 0.0)
    # v0 = v0.filled(fill_value = 0.0)
    u0[np.isnan(u0)] = 0.0
    v0[np.isnan(v0)] = 0.0

    u0 = (u0[:,1:] + u0[:, :-1]) * 0.5
    v0 = (v0[1:,:] + v0[:-1, :]) * 0.5

    # Read all other parameters from the standard Barents output file
    H_m = ds_barents.h.isel(X=slice(x0-1, x1+1), Y=slice(y0-1, y1+1)).data
    angle = ds_barents.angle.isel(X=slice(x0, x1), Y=slice(y0, y1)).data
    latitude = ds_barents.lat.isel(X=slice(x0, x1), Y=slice(y0, y1)).data
    longitude = ds_barents.lon.isel(X=slice(x0, x1), Y=slice(y0, y1)).data
    x = ds_barents.X.isel(X=slice(x0, x1)).data
    y = ds_barents.Y.isel(Y=slice(y0, y1)).data
    
       
    #Fallback if input quantities are not properly masked
    mask = np.isnan(eta0)
    eta0 = np.ma.MaskedArray(eta0, mask)
    if eta0.data.shape != eta0.mask.shape:
        mask = (H_m == land_value)

    #Generate intersections bathymetry
    H_m_mask = mask.copy()
    H_m = np.ma.array(H_m, mask=H_m_mask)
    for i in range(erode_land):
        new_water = H_m.mask ^ binary_erosion(H_m.mask)
        new_water[0,:]  = False # Avoid erosion along boundary
        new_water[-1,:] = False
        new_water[:,0]  = False
        new_water[:,-1] = False
        eps = 1.0e-5 #Make new Hm slighlyt different from land_value

        # Grey_dilation only works on positive numbers, so we add and subtract 10 to eta
        eta0_tmp = eta0 + 10
        eta0_dil = grey_dilation(eta0_tmp.filled(0.0), size=(3,3)) - 10        
        H_m[new_water] = land_value+eps
        eta0[new_water] = eta0_dil[new_water]
    
    H_i, _ = OceanographicUtilities.midpointsToIntersections(H_m, land_value=land_value, iterations=iterations)
    eta0 = eta0[1:-1, 1:-1]
    h0 = OceanographicUtilities.intersectionsToMidpoints(H_i).filled(land_value) + eta0.filled(0.0)

    # Some of the barents files has very high water levels (up to 50 m)
    # Hard code these out
    invalid_indices = np.abs(eta0) > 5
    if np.sum(invalid_indices) > 0:
        eta0[invalid_indices] = 0.0
        u0[invalid_indices] = 0.0
        v0[invalid_indices] = 0.0
    
    #Generate physical variables
    eta0 = np.ma.array(eta0.filled(0), mask=eta0.mask.copy())
    hu0 = np.ma.array(h0*u0, mask=eta0.mask.copy())
    hv0 = np.ma.array(h0*v0, mask=eta0.mask.copy())


    #Spong cells for e.g., flow relaxation boundary conditions
    ic['sponge_cells'] = sponge_cells
    
    #Number of cells
    ic['NX'] = x1 - x0
    ic['NY'] = y1 - y0
    
    # Domain size without ghost cells
    ic['nx'] = ic['NX']-4
    ic['ny'] = ic['NY']-4
    
    #Dx and dy
    #FIXME: Assumes equal for all.. .should check
    ic['dx'] = np.average(x[1:] - x[:-1])
    ic['dy'] = np.average(y[1:] - y[:-1])

    # Numerical time step
    # Set to zero so that the CFL condition is computed automatically
    ic['dt'] = 0.0

    #Gravity and friction
    #FIXME: Friction coeff from netcdf?
    ic['g'] = 9.81
    ic['r'] = 3.0e-3
    
    #Physical variables
    ic['H'] = H_i
    ic['Hm'] = H_m[1:-1, 1:-1]
    ic['eta0'] = eta0 #fill_coastal_data(eta0)
    ic['hu0'] = hu0
    ic['hv0'] = hv0
    
    #Coriolis angle and beta
    ic['angle'] = angle
    ic['latitude'] = OceanographicUtilities.degToRad(latitude)
    ic['lat'] = latitude
    ic['lon'] = longitude

    ic['f'] = 0.0 #Set using latitude instead
    # The beta plane of doing it:
    # ic['f'], ic['coriolis_beta'] = OceanographicUtilities.calcCoriolisParams(OceanographicUtilities.degToRad(latitude[0, 0]))
    
    #Boundary conditions
    bc_data, bc_entire_domain = getBoundaryConditionsData(ds_barotropic, timestep_indices, timesteps, x0, x1, y0, y1, H_m)
    ic['boundary_conditions_data'] = bc_data
    ic['boundary_conditions_entire_domain'] = bc_entire_domain
    ic['boundary_conditions'] = Common.BoundaryConditions(north=3, south=3, east=3, west=3, spongeCells=sponge_cells)
    ic['boundary_conditions_meta'] = None

    #wind (wind speed in m/s used for forcing on drifter)
    ic['wind'] = getWind(ds_barents, timestep_indices, timesteps, x0, x1, y0, y1, barents_url) 
    
    #Note
    ic['note'] = datetime.datetime.now().isoformat() + ": Generated from " + barents_url + " and " + barotropic_url
    
    #Initial reference time and all timesteps
    ic['t0'] = np.float64(timesteps_absolute[0])
    ic['timesteps'] = timesteps
    ic['timesteps_absolute'] = timesteps_absolute
    
    return ic




def rescaleInitialConditions(old_ic, scale):
    ic = copy.deepcopy(old_ic)
    
    ic['NX'] = int(old_ic['NX']*scale)
    ic['NY'] = int(old_ic['NY']*scale)
    gc_x = old_ic['NX'] - old_ic['nx']
    gc_y = old_ic['NY'] - old_ic['ny']
    ic['nx'] = ic['NX'] - gc_x
    ic['ny'] = ic['NY'] - gc_y
    ic['dx'] = old_ic['dx']/scale
    ic['dy'] = old_ic['dy']/scale
    _, _, ic['H'] = OceanographicUtilities.rescaleIntersections(old_ic['H'], ic['NX']+1, ic['NY']+1)
    _, _, ic['eta0'] = OceanographicUtilities.rescaleMidpoints(old_ic['eta0'], ic['NX'], ic['NY'])
    _, _, ic['hu0'] = OceanographicUtilities.rescaleMidpoints(old_ic['hu0'], ic['NX'], ic['NY'])
    _, _, ic['hv0'] = OceanographicUtilities.rescaleMidpoints(old_ic['hv0'], ic['NX'], ic['NY'])
    if (old_ic['angle'].shape == old_ic['eta0'].shape):
        _, _, ic['angle'] = OceanographicUtilities.rescaleMidpoints(old_ic['angle'], ic['NX'], ic['NY'])
    if (old_ic['latitude'].shape == old_ic['eta0'].shape):
        _, _, ic['latitude'] = OceanographicUtilities.rescaleMidpoints(old_ic['latitude'], ic['NX'], ic['NY'])
    
    #Scale number of sponge cells also
    for key in ic['boundary_conditions'].spongeCells.keys():
        ic['boundary_conditions'].spongeCells[key] = np.int32(ic['boundary_conditions'].spongeCells[key]*scale)
        
    #Not touched:
    #"boundary_conditions": 
    #"boundary_conditions_data": 
    #"wind_stress": 
    ic['note'] = old_ic['note'] + "\n" + datetime.datetime.now().isoformat() + ": Rescaled by factor " + str(scale)

    return ic


def getWindForcingOnlyBarentsCases(barents_url, casename, **kwargs):
    """
    Initial conditions for pre-defined areas within the Barents model domain, wind only.
    """
    use_case = getCaseLocation(casename)
    return getWindForcingOnly(barents_url, use_case['x0'], use_case['x1'], use_case['y0'], use_case['y1'], **kwargs)

def getWindForcingOnly(barents_url, x0, x1, y0, y1,
                       timestep_indices=np.arange(24)):
    """
    Constructing input arguments for CDKLM16 instances for the wind forcing only
    """
    forcing = {}
    
    # Make dataset objects with xarray
    ds_barents = xr.open_dataset(barents_url)

    t0_index = timestep_indices[0]
    timesteps_absolute = ds_barents.time.data[timestep_indices].astype('datetime64[s]')
    timesteps = timesteps_absolute - timesteps_absolute[0]
    timesteps = timesteps.astype(np.float32)
    assert(np.all(np.diff(timesteps) >= 0))

    # Need eta for the landmask
    eta0 = ds_barents.zeta.isel(time=t0_index, X=slice(x0, x1), Y=slice(y0, y1)).data
    mask = np.isnan(eta0)
    forcing["eta0"] = np.ma.MaskedArray(eta0, mask)
    

    # Read parameters to enable rotation and mapping into another domain
    forcing["angle"] = ds_barents.angle.isel(X=slice(x0, x1), Y=slice(y0, y1)).data
    forcing["lat"] = ds_barents.lat.isel(X=slice(x0, x1), Y=slice(y0, y1)).data
    forcing["lon"] = ds_barents.lon.isel(X=slice(x0, x1), Y=slice(y0, y1)).data
    
    forcing['wind'] = getWind(ds_barents, timestep_indices, timesteps, x0, x1, y0, y1, barents_url) 
    
    return forcing

### ------------------------------------------------
### Utility functions
### ------------------------------------------------

def getBoundaryConditionsData(ds_barotropic, timestep_indices, timesteps, x0, x1, y0, y1, H, store_entire_domain=True):
    """
    timestep_indices => index into netcdf-array, e.g. [1, 3, 5]
    timestep => time at timestep, e.g. [1800, 3600, 7200]
    """
    
    nt = len(timesteps)
    
    bc_eta = {}
    bc_eta['north'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_eta['south'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_eta['east'] = np.empty((nt, y1-y0), dtype=np.float32)
    bc_eta['west'] = np.empty((nt, y1-y0), dtype=np.float32)

    bc_hu = {}
    bc_hu['north'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_hu['south'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_hu['east'] = np.empty((nt, y1-y0), dtype=np.float32)
    bc_hu['west'] = np.empty((nt, y1-y0), dtype=np.float32)

    bc_hv = {}
    bc_hv['north'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_hv['south'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_hv['east'] = np.empty((nt, y1-y0), dtype=np.float32)
    bc_hv['west'] = np.empty((nt, y1-y0), dtype=np.float32)
    
    bc_entire_domain = None
    if store_entire_domain:
        bc_entire_domain = {
            'eta': np.empty((nt, y1-y0, x1-x0), dtype=np.float32),
            'hu':  np.empty((nt, y1-y0, x1-x0), dtype=np.float32),
            'hv':  np.empty((nt, y1-y0, x1-x0), dtype=np.float32),
        }        

    bc_index = 0

    # Only one file
    # loop time steps
    # input H to this function?
    #CONTINUE_HERE

    for timestep_index in timestep_indices:

        # add one frame of cells around eta temporarily
        eta = ds_barotropic.zeta.isel(ocean_time=timestep_index, xi_rho=slice(x0-1, x1+1), eta_rho=slice(y0-1, y1+1)).data
        u = ds_barotropic.ubar.isel(ocean_time=timestep_index, xi_u=slice(x0-1, x1+1+1), eta_u=slice(y0-1, y1+1  )).data
        v = ds_barotropic.vbar.isel(ocean_time=timestep_index, xi_v=slice(x0-1, x1+1  ), eta_v=slice(y0-1, y1+1+1)).data
        
        eta[np.isnan(eta)] = 0.0
        u[np.isnan(u)] = 0.0
        v[np.isnan(v)] = 0.0

        

        # Staggered grid to non-staggered
        u = (u[:,1:] + u[:, :-1]) * 0.5   
        v = (v[1:,:] + v[:-1, :]) * 0.5

        invalid_indices = np.abs(eta) > 5
        if np.sum(invalid_indices) > 0:
            eta[invalid_indices] = 0.0
            u[invalid_indices] = 0.0
            v[invalid_indices] = 0.0

        h = H + eta
        hu = h*u
        hv = h*v

        bc_eta['north'][bc_index] = eta[-1, 1:-1]
        bc_eta['south'][bc_index] = eta[0, 1:-1]
        bc_eta['east'][bc_index] = eta[1:-1, -1]
        bc_eta['west'][bc_index] = eta[ 1:-1, 0]

        bc_hu['north'][bc_index] = hu[-1, 1:-1]
        bc_hu['south'][bc_index] = hu[0, 1:-1]
        bc_hu['east'][bc_index] = hu[1:-1, -1]
        bc_hu['west'][bc_index] = hu[1:-1, 0]



        bc_hv['north'][bc_index] = hv[-1, 1:-1]
        bc_hv['south'][bc_index] = hv[0, 1:-1]
        bc_hv['east'][bc_index] = hv[1:-1, -1]
        bc_hv['west'][bc_index] = hv[1:-1, 0]

        if store_entire_domain:
            bc_entire_domain['eta'][bc_index] = eta[1:-1, 1:-1]
            bc_entire_domain['hu'][bc_index]  =  hu[1:-1, 1:-1]
            bc_entire_domain['hv'][bc_index]  =  hv[1:-1, 1:-1]

        bc_index = bc_index + 1


    bc_data = Common.BoundaryConditionsData(timesteps.copy(), 
        north=Common.SingleBoundaryConditionData(bc_eta['north'], bc_hu['north'], bc_hv['north']),
        south=Common.SingleBoundaryConditionData(bc_eta['south'], bc_hu['south'], bc_hv['south']),
        east=Common.SingleBoundaryConditionData(bc_eta['east'], bc_hu['east'], bc_hv['east']),
        west=Common.SingleBoundaryConditionData(bc_eta['west'], bc_hu['west'], bc_hv['west']))
    
    return bc_data, bc_entire_domain




def getWind(ds_barents, timestep_indices, timesteps, x0, x1, y0, y1, barents_url):
    """
    timestep_indices => index into netcdf-array, e.g. [1, 3, 5]
    timestep => time at timestep, e.g. [1800, 3600, 7200]
    """
    
    u_wind = ds_barents.Uwind.isel(time = timestep_indices, X=slice(x0, x1), Y=slice(y0, y1  )).data
    v_wind = ds_barents.Vwind.isel(time = timestep_indices, X=slice(x0, x1), Y=slice(y0, y1  )).data
    
    u_wind[np.isnan(u_wind)] = 0.0
    v_wind[np.isnan(v_wind)] = 0.0
    
    u_wind = u_wind.astype(np.float32)
    v_wind = v_wind.astype(np.float32)
    
    wind_source = WindStress.WindStress(t=timesteps.copy(), wind_u=u_wind, wind_v=v_wind, source_filename=barents_url)
    
    return wind_source


def makeEnsembleAnomalies(b_args_list):
    ## Subtracting the ensemble mean from eta, hu, hv for all the 

    ny, nx = b_args_list[0]['eta0'].shape
    Ne = len(b_args_list)

    for field in ["eta0", "hu0", "hv0"]:
        all_data  = np.empty((Ne, ny, nx))
        for i in range(Ne):
            all_data[i, :, :] = b_args_list[i][field].data

        all_data -= np.mean(all_data, axis=0)
        
        for i in range(Ne):
            b_args_list[i][field].data[:,:] = all_data[i, :, :]

    

## Utility functions
#---------------------




def fill_coastal_data(maarr):
    """
    Function manipulating the data of a masked array in the dry-zone.
    If a dry cell has one or more wet neighbors, the average data is filled (otherwise the dry data stays 0, what is the default)

    Input:  maarr - masked array
    Output: maarr - masked array (with same mask, but modified data)
    """

    for i in range(maarr.shape[1]):
        for j in range(maarr.shape[0]):
            if (maarr.mask[j,i]):
                N_wet_neighbors = 0
                sum = 0.0
                if i > 0:
                    if maarr.mask[j,i-1] == False:
                        sum += maarr.data[j,i-1]
                        N_wet_neighbors += 1 
                if i < maarr.shape[1]-1: 
                    if maarr.mask[j,i+1] == False:
                        sum += maarr.data[j,i-1]
                        N_wet_neighbors += 1 
                if j > 0: 
                    if maarr.mask[j-1,i] == False:
                        sum += maarr.data[j-1,i]
                        N_wet_neighbors += 1 
                if j < maarr.shape[0]-1: 
                    if maarr.mask[j+1,i] == False:
                        sum += maarr.data[j+1,i]
                        N_wet_neighbors += 1 
                if i > 0 and j > 0:
                    if maarr.mask[j-1,i-1] == False:
                        sum += maarr.data[j-1,i-1]
                        N_wet_neighbors += 1 
                if i < maarr.shape[1]-1 and j > 0:
                    if maarr.mask[j-1,i+1] == False:
                        sum += maarr.data[j-1,i+1]
                        N_wet_neighbors += 1 
                if i > 0 and j < maarr.shape[0]-1:
                    if maarr.mask[j+1,i-1] == False:
                        sum += maarr.data[j+1,i-1]
                        N_wet_neighbors += 1 
                if i < maarr.shape[1]-1 and j < maarr.shape[0]-1:
                    if maarr.mask[j+1,i+1] == False:
                        sum += maarr.data[j+1,i+1]
                        N_wet_neighbors += 1 
                if N_wet_neighbors > 0:
                    maarr.data[j,i] = sum/N_wet_neighbors
    return maarr



# Returns True if the current execution context is an IPython notebook, e.g. Jupyter.
# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def in_ipynb():
    try:
        cfg = get_ipython().config
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
        #if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            #print ('Running in ipython notebook env.')
            return True
        else:
            return False
    except NameError:
        #print ('NOT Running in ipython notebook env.')
        return False

def checkCachedNetCDF(source_url, download_data=True):
    """ 
    Checks if the file represented by source_url is available locally already.
    We search for the file in the working directory, or in a folder called 
    'netcdf_cache' in the working directory.
    If download_data is true, it will  download the netcfd file into 'netcdf_cache' 
    if it is not found locally already.
    """
    ### Check if local file exists:
    filename = os.path.abspath(os.path.basename(source_url))
    cache_folder='netcdf_cache'
    cache_filename = os.path.abspath(os.path.join(cache_folder,
                                                  os.path.basename(source_url)))
                                                  
    if (os.path.isfile(filename)):
        source_url = filename
        
    elif (os.path.isfile(cache_filename)):
        source_url = cache_filename
        
    elif (download_data):
        import requests
        download_url = source_url.replace("dodsC", "fileServer")

        req = requests.get(download_url, stream = True)
        filesize = int(req.headers.get('content-length'))

        is_notebook = False
        if(in_ipynb()):
            progress = Common.ProgressPrinter()
            pp = display(progress.getPrintString(0),display_id=True)
            is_notebook = True
        
        os.makedirs(cache_folder, exist_ok=True)

        print("Downloading data to local file (" + str(filesize // (1024*1024)) + " MB)")
        with open(cache_filename, "wb") as outfile:
            for chunk in req.iter_content(chunk_size = 10*1024*1024):
                if chunk:
                    outfile.write(chunk)
                    if(is_notebook):
                        pp.update(progress.getPrintString(outfile.tell() / filesize))

        source_url = cache_filename
    return source_url




def removeMetadata(old_ic):
    ic = old_ic.copy()
    
    ic.pop('note', None)
    ic.pop('NX', None)
    ic.pop('NY', None)
    ic.pop('sponge_cells', None)
    ic.pop('t0', None)
    ic.pop('timesteps', None)
    ic.pop('lat', None)
    ic.pop('lon', None)
    ic.pop('Hm', None)
    ic.pop('timesteps_absolute', None) 
    ic.pop('boundary_conditions_meta', None)
    ic.pop('boundary_conditions_entire_domain', None)
    
    return ic 




