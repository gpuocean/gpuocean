# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2019 Norwegian Meteorological Institute
Copyright (C) 2019 SINTEF Digital

This python module implements saving shallow water simulations to a
netcdf file.

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
import datetime, os, copy
from netCDF4 import Dataset, MFDataset
import pyproj
from scipy.ndimage.morphology import binary_erosion, grey_dilation

import seawater as sw
from scipy.ndimage.filters import convolve, gaussian_filter

from gpuocean.utils import Common, WindStress, OceanographicUtilities


def getNorkystSubdomains():
    """
    Lists (and defines) predefined subdomains for the NorKyst-800 model.
    """
    return [
        {'name': 'norwegian_sea',  'x0':  900, 'x1': 1400, 'y0':  600, 'y1':  875 },
        {'name': 'lofoten',        'x0': 1400, 'x1': 1900, 'y0':  450, 'y1':  750 },
        {'name': 'complete_coast', 'x0':   25, 'x1': 2575, 'y0':   25, 'y1':  875 },
        {'name': 'skagerrak',      'x0':   50, 'x1':  550, 'y0':   35, 'y1':  285 },
        {'name': 'oslo',           'x0':  500, 'x1':  550, 'y0':  160, 'y1':  210 },
        {'name': 'denmark',        'x0':    2, 'x1':  300, 'y0':    2, 'y1':  300 },
        {'name': 'lovese',         'x0': 1400, 'x1': 2034, 'y0':  450, 'y1':  769 },
        {'name': 'barents_sea',    'x0': 2150, 'x1': 2575, 'y0':  575, 'y1':  875 },
        {'name': 'finnmark',       'x0': 2000, 'x1': 2450, 'y0':  200, 'y1':  625 },
        {'name': 'north_cape',     'x0': 2080, 'x1': 2350, 'y0':  390, 'y1':  590 },
        {'name': 'north_sea',      'x0':   25, 'x1':  350, 'y0':  550, 'y1':  875 },
        {'name': 'vestlandskysten','x0':  350, 'x1':  850, 'y0':  550, 'y1':  850 },
        {'name': 'sorvestlandet',  'x0':  100, 'x1':  550, 'y0':  350, 'y1':  700 }
    ]


def getInitialConditionsNorKystCases(source_url, casename, **kwargs):
    """
    Initial conditions for pre-defined areas within the NorKyst-800 model domain. 
    """
    use_case = getCaseLocation(casename)
    return getInitialConditions(source_url, use_case['x0'], use_case['x1'], use_case['y0'], use_case['y1'], **kwargs)

def getCaseLocation(casename):
    """
    Domains for pre-defined areas within the NorKyst-800 model domain. 
    """
    cases = getNorkystSubdomains()
    use_case = None
    for case in cases:
        if case['name'] == casename:
            use_case = case
            break

    assert(use_case is not None), 'Invalid case. Check NetCDFInitialization.getNorkystSubdomains() to see valid case names'

    return use_case


def getInitialConditions(source_url_list, x0, x1, y0, y1, \
                         timestep_indices=None, \
                         norkyst_data = True,
                         land_value=5.0, \
                         iterations=10, \
                         sponge_cells={'north':20, 'south': 20, 'east': 20, 'west': 20}, \
                         erode_land=0, 
                         download_data=False
                         ):
    """
    Constructing input arguments for CDKLM16 instances
    source_url_list    - list with urls or paths to NetCDF-files (single files are allows)
    timestep_indices   - list with timestep_indices per file (for a single file, e.g., "[[12,13,14,15,16]]") 
    norkyst_data       - boolean whether source_url corresponds to a *Z-level* file from NorKyst800 model outputs
    download_data      - downloading the source file for faster initialization next time - warning: downloaded files might be large
    """
    
    ic = {}
    
    if type(source_url_list) is not list:
        source_url_list = [source_url_list]

    num_files = len(source_url_list)
    
    for i in range(len(source_url_list)):
        source_url_list[i] = checkCachedNetCDF(source_url_list[i], download_data=download_data)
    
        
    # Get time steps:
    if norkyst_data:
        time_str = 'time'
    else:
        time_str = 'ocean_time'

    if timestep_indices is None:
        timestep_indices = [None]*num_files
    elif type(timestep_indices) is not list:
        timestep_indices_tmp = [None]*num_files
        for i in range(num_files):
            timestep_indices_tmp[i] = timestep_indices
        timestep_indices = timestep_indices_tmp
    elif type(timestep_indices[0]) is not list:
        timestep_indices = [timestep_indices]
    
    timesteps = [None]*num_files

    t0_index = 0
    if timestep_indices is not None:
        if timestep_indices[0] is not None:
            t0_index = timestep_indices[0][0]
        

        
    for i in range(num_files):
        try:
            ncfile = Dataset(source_url_list[i])
            if (timestep_indices[i] is not None):
                timesteps[i] = ncfile.variables[time_str][timestep_indices[i][:]]
            else:
                if i == 0: 
                    timesteps[i] = ncfile.variables[time_str][t0_index:]
                    timestep_indices[i] = range(t0_index, len(ncfile.variables[time_str][:]))
                else:
                    timesteps[i] = ncfile.variables[time_str][:]
                    timestep_indices[i] = range(len(timesteps[i]))
        except Exception as e:
            print('exception in obtaining timestep for file '+str(i))
            raise e
        finally:
            ncfile.close()

    #Generate timesteps in reference to t0
    t0 = timesteps[0][0]
    for ts in timesteps:
        t0 = min(t0, min(ts))

    
    assert(np.all(np.diff([time_item for timesteps_sublist in timesteps for time_item in timesteps_sublist])>=0))
    # Obtain timesteps relative to initialization
    for i in range(num_files):
        timesteps[i] = timesteps[i] - t0 
    
    
    # Read constants and initial values from the first source url
    source_url = source_url_list[0]
    if norkyst_data:
        try:
            ncfile = Dataset(source_url)
            H_m = ncfile.variables['h'][y0-1:y1+1, x0-1:x1+1]
            eta0 = ncfile.variables['zeta'][t0_index, y0-1:y1+1, x0-1:x1+1]
            u0 = ncfile.variables['ubar'][t0_index, y0:y1, x0:x1]
            v0 = ncfile.variables['vbar'][t0_index, y0:y1, x0:x1]
            angle = ncfile.variables['angle'][y0:y1, x0:x1]
            latitude = ncfile.variables['lat'][y0:y1, x0:x1]
            x = ncfile.variables['X'][x0:x1]
            y = ncfile.variables['Y'][y0:y1]
        except Exception as e:
            raise e
        finally:
            ncfile.close()
        
        u0 = u0.filled(0.0)
        v0 = v0.filled(0.0)
        
        time_str = 'time'
    else:
        # Assuming staggered grid
        # First, try to read ubar and vbar. If it fails, integrate velocities vertically 
        try:
            ncfile = Dataset(source_url)
            H_m = ncfile.variables['h'][y0-1:y1+1, x0-1:x1+1]
            eta0 = ncfile.variables['zeta'][t0_index, y0-1:y1+1, x0-1:x1+1]

            try: 
                u0 = ncfile.variables['ubar'][t0_index, y0:y1, x0:x1+1]
                v0 = ncfile.variables['vbar'][t0_index, y0:y1+1, x0:x1]
                #Find u,v at cell centers
                u0 = u0.filled(fill_value = 0.0)
                v0 = v0.filled(fill_value = 0.0)
        
                u0 = (u0[:,1:] + u0[:, :-1]) * 0.5
                v0 = (v0[1:,:] + v0[:-1, :]) * 0.5
            except:
                u0 = ncfile.variables['u'][t0_index, :, y0:y1, x0:x1+1]
                v0 = ncfile.variables['v'][t0_index, :, y0:y1+1, x0:x1]
                #Find u,v at cell centers
                u0 = u0.filled(fill_value = 0.0)
                v0 = v0.filled(fill_value = 0.0)
        
                u0 = (u0[:, :,1:] + u0[:, :, :-1]) * 0.5
                v0 = (v0[:, 1:,:] + v0[:, :-1, :]) * 0.5
                
                integrator = vertical_integrator(source_url, H_m[1:-1,1:-1], x0=x0, x1=x1, y0=y0, y1=y1)
                u0 = np.sum(integrator * u0, axis=0)/(H_m[1:-1,1:-1])
                v0 = np.sum(integrator * v0, axis=0)/(H_m[1:-1,1:-1])

            angle = ncfile.variables['angle'][y0:y1, x0:x1]
            #lon, lat at cell centers:
            lat_rho = ncfile.variables['lat_rho'][y0:y1, x0:x1]
            lon_rho = ncfile.variables['lon_rho'][y0:y1, x0:x1]
        except Exception as e:
            raise e
        finally:
            ncfile.close()
        
        latitude = lat_rho
        
        #Find x, y (in Norkyst800 reference system, origin at norkyst800 origin)
        proj_str= '+proj=stere +ellps=WGS84 +lat_0=90.0 +lat_ts=60.0 +x_0=3192800 +y_0=1784000 +lon_0=70'
        #Check for other projection information in ncfile
        if np.any(np.array(list(map(lambda x: x.lower().startswith("projection"), ncfile.variables.keys())))):
            try: 
                ncfile = Dataset(source_url)
                proj_var_str = [key for key in ncfile.variables.keys() if key.lower().startswith("proj")][0]
                proj_str = ncfile.variables[proj_var_str].__getattr__('proj4')
            except Exception as e:
                raise e
            finally:
                ncfile.close()

        proj = pyproj.Proj(proj_str)
        
        x_rho, y_rho = proj(lon_rho, lat_rho, inverse = False)
        x, y = x_rho[0], y_rho[:,0]
        
       
    #Fallback if input quantities are not properly masked
    mask = eta0.mask.copy()
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
    ic['eta0'] = eta0 #fill_coastal_data(eta0)
    ic['hu0'] = hu0
    ic['hv0'] = hv0
    
    #Coriolis angle and beta
    ic['angle'] = angle
    ic['latitude'] = OceanographicUtilities.degToRad(latitude)
    ic['f'] = 0.0 #Set using latitude instead
    # The beta plane of doing it:
    # ic['f'], ic['coriolis_beta'] = OceanographicUtilities.calcCoriolisParams(OceanographicUtilities.degToRad(latitude[0, 0]))
    
    #Boundary conditions
    ic['boundary_conditions_data'] = getBoundaryConditionsData(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1, norkyst_data)
    ic['boundary_conditions'] = Common.BoundaryConditions(north=3, south=3, east=3, west=3, spongeCells=sponge_cells)
    
    #wind (wind speed in m/s used for forcing on drifter)
    ic['wind'] = getWind(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1) 
    
    #Note
    ic['note'] = datetime.datetime.now().isoformat() + ": Generated from " + str(source_url_list)
    
    #Initial reference time and all timesteps
    ic['t0'] = t0
    ic['timesteps'] = np.array([time_item for timesteps_sublist in timesteps for time_item in timesteps_sublist])
    
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



def getBoundaryConditionsData(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1, norkyst_data):
    """
    timestep_indices => index into netcdf-array, e.g. [1, 3, 5]
    timestep => time at timestep, e.g. [1800, 3600, 7200]
    """
    num_files = len(source_url_list)
    
    nt = 0
    for i in range(num_files):
        nt += len(timesteps[i])
    
    if (timestep_indices is None):
        timestep_indices = [None]*num_files
        for i in range(num_files):
            timestep_indices[i] = range(len(timesteps[i]))

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
    
    
    bc_index = 0
    for i in range(num_files):
        try:
            ncfile = Dataset(source_url_list[i])

            H = ncfile.variables['h'][y0-1:y1+1, x0-1:x1+1]
            
            for timestep_index in timestep_indices[i]:
                zeta = ncfile.variables['zeta'][timestep_index, y0-1:y1+1, x0-1:x1+1]
                zeta = zeta.filled(0)
                bc_eta['north'][bc_index] = zeta[-1, 1:-1]
                bc_eta['south'][bc_index] = zeta[0, 1:-1]
                bc_eta['east'][bc_index] = zeta[1:-1, -1]
                bc_eta['west'][bc_index] = zeta[ 1:-1, 0]

                h = H + zeta
                
                if norkyst_data:
                    u = ncfile.variables['ubar'][timestep_index, y0-1:y1+1, x0-1:x1+1]
                    u = u.filled(0) #zero on land
                else:
                    # Assuming staggered grid                    
                    try:
                        u = ncfile.variables['ubar'][timestep_index, y0-1:y1+1, x0-1:x1+2]
                        u = u.filled(0) #zero on land
                        u = (u[:,1:] + u[:, :-1]) * 0.5   
                    except:
                        # If ubar don't exist, integrate u vertically
                        u = ncfile.variables['u'][timestep_index, :, y0-1:y1+1, x0-1:x1+2]
                        u = u.filled(fill_value = 0.0)
                        u = (u[:, :,1:] + u[:, :, :-1]) * 0.5
                        
                        integrator = vertical_integrator(source_url_list[i], H, x0=x0-1, x1=x1+1, y0=y0-1, y1=y1+1)
                        u = np.sum(integrator * u, axis=0)/h
                hu = h*u


                bc_hu['north'][bc_index] = hu[-1, 1:-1]
                bc_hu['south'][bc_index] = hu[0, 1:-1]
                bc_hu['east'][bc_index] = hu[1:-1, -1]
                bc_hu['west'][bc_index] = hu[1:-1, 0]

                if norkyst_data:
                    v = ncfile.variables['vbar'][timestep_index, y0-1:y1+1, x0-1:x1+1]
                    v = v.filled(0) #zero on land
                else:
                    # Assuming staggered grid
                    try:
                        v = ncfile.variables['vbar'][timestep_index, y0-1:y1+2, x0-1:x1+1]
                        v = v.filled(0) #zero on land
                        v = (v[1:,:] + v[:-1, :]) * 0.5
                    except:
                        # If vbar don't exist, integrate v vertically
                        v = ncfile.variables['v'][timestep_index, :, y0-1:y1+2, x0-1:x1+1]
                        v = v.filled(fill_value = 0.0)
                        v = (v[:, 1:, :] + v[:, :-1, :]) * 0.5
                        
                        integrator = vertical_integrator(source_url_list[i], H, x0=x0-1, x1=x1+1, y0=y0-1, y1=y1+1)
                        v = np.sum(integrator * v, axis=0)/h 
                hv = h*v


                bc_hv['north'][bc_index] = hv[-1, 1:-1]
                bc_hv['south'][bc_index] = hv[0, 1:-1]
                bc_hv['east'][bc_index] = hv[1:-1, -1]
                bc_hv['west'][bc_index] = hv[1:-1, 0]

                bc_index = bc_index + 1
                

        except Exception as e:
            raise e
        finally:
            ncfile.close()

    all_timesteps = [time_item for timesteps_sublist in timesteps for time_item in timesteps_sublist] # from list of lists to a single list.

    bc_data = Common.BoundaryConditionsData(all_timesteps.copy(), 
        north=Common.SingleBoundaryConditionData(bc_eta['north'], bc_hu['north'], bc_hv['north']),
        south=Common.SingleBoundaryConditionData(bc_eta['south'], bc_hu['south'], bc_hv['south']),
        east=Common.SingleBoundaryConditionData(bc_eta['east'], bc_hu['east'], bc_hv['east']),
        west=Common.SingleBoundaryConditionData(bc_eta['west'], bc_hu['west'], bc_hv['west']))
    
    return bc_data




def getWind(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1):
    """
    timestep_indices => index into netcdf-array, e.g. [1, 3, 5]
    timestep => time at timestep, e.g. [1800, 3600, 7200]
    """
    
    if type(source_url_list) is not list:
        source_url_list = [source_url_list]
    
    num_files = len(source_url_list)
    
    assert(num_files == len(timesteps)), str(num_files) +' vs '+ str(len(timesteps))
    
    if (timestep_indices is None):
        timestep_indices = [None]*num_files
        for i in range(num_files):
            timestep_indices[i] = range(len(timesteps[i]))
        
    u_wind_list = [None]*num_files
    v_wind_list = [None]*num_files
    
    if "Uwind" in Dataset(source_url_list[0]).variables:
        for i in range(num_files):
            try:
                ncfile = Dataset(source_url_list[i])
                if i == 0 and len(ncfile.variables['Uwind'].shape) == 0:
                    return WindStress.WindStress()
                u_wind_list[i] = ncfile.variables['Uwind'][timestep_indices[i], y0:y1, x0:x1]
                v_wind_list[i] = ncfile.variables['Vwind'][timestep_indices[i], y0:y1, x0:x1]
            except Exception as e:
                raise e
            finally:
                ncfile.close()
    else:
        return WindStress.WindStress()

    u_wind = u_wind_list[0].filled(0)
    v_wind = v_wind_list[0].filled(0)
    for i in range(1, num_files):
        u_wind = np.concatenate((u_wind, u_wind_list[i].filled(0)))
        v_wind = np.concatenate((v_wind, v_wind_list[i].filled(0)))
    
    u_wind = u_wind.astype(np.float32)
    v_wind = v_wind.astype(np.float32)
    
    source_filename = ' and '.join([url for url in source_url_list])

    all_timesteps = [time_item for timesteps_sublist in timesteps for time_item in timesteps_sublist]

    wind_source = WindStress.WindStress(t=all_timesteps.copy(), wind_u=u_wind, wind_v=v_wind, source_filename=source_filename)
    
    return wind_source


## Utility functions
#---------------------



def depth_integration(source_url, interface_depth, x0, x1, y0, y1, var, timestep_index=0): 
    """
    depth integration of var (variable in source_url-netcdf file) 
    for depth 0 down to interface depth using trpeziodal rule 
    NB! the interface depth has to be a depth level in the netcdf file
    """
    if isinstance(source_url, str):
        nc = Dataset(source_url)
    else: 
        nc = source_url
    #prepartions 
    nx = x1 - x0
    ny = y1 - y0

    depth = nc["depth"][:].data
    level_idx = np.where(depth == interface_depth)[0][0]

    # trapeziod average of u or v 
    uv = nc[var][timestep_index,:,y0:y1, x0:x1]
    uv_hat = (uv[:level_idx] + uv[1:level_idx+1])/2

    # depth steps and restructuring to the same dimensions as uv_hat
    depth_diff = depth[1:level_idx+1] - depth[:level_idx]
    for dd_idx in range(len(depth_diff)):
        if dd_idx == 0:
            dd_xy = np.full((ny,nx),depth_diff[dd_idx])[np.newaxis,:,:]
        else:
            dd_xy = np.concatenate((dd_xy, np.full((ny,nx),depth_diff[dd_idx])[np.newaxis,:,:]), axis=0 )

    huv = np.ma.sum(dd_xy * uv_hat, axis=0)

    if isinstance(source_url, str):
        nc.close()
    
    return huv


def vertical_integrator(source_url, mld, t=0, x0=0, x1=-1, y0=0, y1=-1):

    s_nc = Dataset(source_url)

    # Collect information about s-levels
    w_lvls = s_nc["Cs_w"][:].data

    # Collect information about domain
    s_hs   = s_nc["h"][y0:y1,x0:x1] + s_nc["zeta"][t,y0:y1,x0:x1]

    mask = False
    if  isinstance(mld.mask, np.ndarray):
        mask = np.array((len(w_lvls)*[mld.mask.copy()]))
    w_depths = np.ma.array(np.multiply.outer(w_lvls,s_hs), mask=mask)

    ny, nx = s_hs.shape

    ## Construct integration weights 
    depths_diff = w_depths[1:] - w_depths[:-1]
    weights = np.ma.array(np.zeros_like(depths_diff), mask=depths_diff.mask.copy())

    lvl_idxs = (np.arange(len(w_lvls)-1)[:,np.newaxis,np.newaxis]*np.ones((len(w_lvls)-1,ny,nx)))
    mld_upper_idx = np.maximum(np.argmax(-w_depths < mld, axis=0), 1)

    # Full w-levels
    weights[lvl_idxs >= mld_upper_idx] = 1

    # Partial w-levels as fraction 
    mld_lower_idx = np.ma.maximum(mld_upper_idx - 1, 0)
    mld_upper_depth = -np.take_along_axis(w_depths, mld_upper_idx.reshape(1,ny,nx), axis=0)[0]
    mld_lower_depth = -np.take_along_axis(w_depths, mld_lower_idx.reshape(1,ny,nx), axis=0)[0]

    np.put_along_axis(weights, mld_lower_idx.reshape(1,ny,nx), ((mld - mld_upper_depth)/(mld_lower_depth - mld_upper_depth)), axis=0)
    # NOTE: If the mld is below the bathymetry the last level can have weight >1

    integrator = depths_diff * weights

    return integrator


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
    
    return ic
