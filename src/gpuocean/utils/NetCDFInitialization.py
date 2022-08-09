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


def getBoundaryConditionsData(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1, norkyst_data, reduced_gravity_interface=None):
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
                
                if reduced_gravity_interface == None:
                    if norkyst_data:
                        hu = ncfile.variables['ubar'][timestep_index, y0-1:y1+1, x0-1:x1+1]
                        hu = hu.filled(0) #zero on land
                    else: 
                        hu = ncfile.variables['ubar'][timestep_index, y0-1:y1+1, x0-1:x1+2]
                        hu = hu.filled(0) #zero on land
                        hu = (hu[:,1:] + hu[:, :-1]) * 0.5   
                    hu = h*hu

                else:
                    hu = depth_integration(ncfile, reduced_gravity_interface, x0-1, x1+1, y0-1, y1+1, "u", timestep_index)

                bc_hu['north'][bc_index] = hu[-1, 1:-1]
                bc_hu['south'][bc_index] = hu[0, 1:-1]
                bc_hu['east'][bc_index] = hu[1:-1, -1]
                bc_hu['west'][bc_index] = hu[1:-1, 0]

                if reduced_gravity_interface == None:
                    if norkyst_data:
                        hv = ncfile.variables['vbar'][timestep_index, y0-1:y1+1, x0-1:x1+1]
                        hv = hv.filled(0) #zero on land
                    else:
                        hv = ncfile.variables['vbar'][timestep_index, y0-1:y1+2, x0-1:x1+1]
                        hv = hv.filled(0) #zero on land
                        hv = (hv[1:,:] + hv[:-1, :]) * 0.5
                    hv = h*hv
                else:
                    hv = depth_integration(ncfile, reduced_gravity_interface, x0-1, x1+1, y0-1, y1+1, "v", timestep_index)

                bc_hv['north'][bc_index] = hv[-1, 1:-1]
                bc_hv['south'][bc_index] = hv[0, 1:-1]
                bc_hv['east'][bc_index] = hv[1:-1, -1]
                bc_hv['west'][bc_index] = hv[1:-1, 0]

                bc_index = bc_index + 1
                

        except Exception as e:
            raise e
        finally:
            ncfile.close()

    bc_data = Common.BoundaryConditionsData(np.ravel(timesteps).copy(), 
        north=Common.SingleBoundaryConditionData(bc_eta['north'], bc_hu['north'], bc_hv['north']),
        south=Common.SingleBoundaryConditionData(bc_eta['south'], bc_hu['south'], bc_hv['south']),
        east=Common.SingleBoundaryConditionData(bc_eta['east'], bc_hu['east'], bc_hv['east']),
        west=Common.SingleBoundaryConditionData(bc_eta['west'], bc_hu['west'], bc_hv['west']))
    
    return bc_data


def getWindSourceterm(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1):
    """
    timestep_indices => index into netcdf-array, e.g. [1, 3, 5]
    timestep => time at timestep, e.g. [1800, 3600, 7200]
    """
    
    if type(source_url_list) is not list:
        source_url_list = [source_url_list]
    
    num_files = len(source_url_list)
    
    source_url = source_url_list[0]
    
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
    
    wind_source = WindStress.WindStress(t=np.ravel(timesteps).copy(), X=wind_stress_u, Y=wind_stress_v)
    
    return wind_source

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
    cases = [
        {'name': 'norwegian_sea',  'x0':  900, 'x1': 1400, 'y0':  600, 'y1':  875 },
        {'name': 'lofoten',        'x0': 1400, 'x1': 1900, 'y0':  450, 'y1':  750 },
        {'name': 'complete_coast', 'x0':   25, 'x1': 2575, 'y0':   25, 'y1':  875 },
        {'name': 'skagerrak',      'x0':   50, 'x1':  550, 'y0':   35, 'y1':  285 },
        {'name': 'oslo',           'x0':  500, 'x1':  550, 'y0':  160, 'y1':  210 },
        {'name': 'denmark',        'x0':    2, 'x1':  300, 'y0':    2, 'y1':  300 },
        {'name': 'lovese',         'x0': 1400, 'x1': 2034, 'y0':  450, 'y1':  769 },
        {'name': 'barents_sea',    'x0': 2150, 'x1': 2575, 'y0':  575, 'y1':  875 },
        {'name': 'north_sea',      'x0':   25, 'x1':  350, 'y0':  550, 'y1':  875 },
        {'name': 'vestlandskysten','x0':  350, 'x1':  850, 'y0':  550, 'y1':  850 },
        {'name': 'sorvestlandet',  'x0':  100, 'x1':  550, 'y0':  350, 'y1':  700 }
    ]
    use_case = None
    for case in cases:
        if case['name'] == casename:
            use_case = case
            break

    assert(use_case is not None), 'Invalid case. Please choose between:\n'+str([case['name'] for case in cases])

    return use_case

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

def getInitialConditions(source_url_list, x0, x1, y0, y1, \
                         timestep_indices=None, \
                         norkyst_data = True,
                         land_value=5.0, \
                         iterations=10, \
                         sponge_cells={'north':20, 'south': 20, 'east': 20, 'west': 20}, \
                         erode_land=0, 
                         download_data=True,
                         reduced_gravity_interface=None):
    ic = {}
    
    if type(source_url_list) is not list:
        source_url_list = [source_url_list]

    num_files = len(source_url_list)
    
    for i in range(len(source_url_list)):
        source_url_list[i] = checkCachedNetCDF(source_url_list[i], download_data=download_data)
    
        
    # Read constants and initial values from the first source url
    source_url = source_url_list[0]
    if norkyst_data:
        try:
            ncfile = Dataset(source_url)
            H_m = ncfile.variables['h'][y0-1:y1+1, x0-1:x1+1]
            eta0 = ncfile.variables['zeta'][0, y0-1:y1+1, x0-1:x1+1]
            u0 = ncfile.variables['ubar'][0, y0:y1, x0:x1]
            v0 = ncfile.variables['vbar'][0, y0:y1, x0:x1]
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
        try:
            ncfile = Dataset(source_url)
            H_m = ncfile.variables['h'][y0-1:y1+1, x0-1:x1+1]
            eta0 = ncfile.variables['zeta'][0, y0-1:y1+1, x0-1:x1+1]
            u0 = ncfile.variables['ubar'][0, y0:y1, x0:x1+1]
            v0 = ncfile.variables['vbar'][0, y0:y1+1, x0:x1]
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
        proj = pyproj.Proj(proj_str)
        
        x_rho, y_rho = proj(lon_rho, lat_rho, inverse = False)
        x, y = x_rho[0], y_rho[:,0]
        
        #Find u,v at cell centers
        u0 = u0.filled(fill_value = 0.0)
        v0 = v0.filled(fill_value = 0.0)
   
        u0 = (u0[:,1:] + u0[:, :-1]) * 0.5
        v0 = (v0[1:,:] + v0[:-1, :]) * 0.5
        
        time_str = 'ocean_time'

        
    # Get time steps:
    if timestep_indices is None:
        timestep_indices = [None]*num_files
    elif type(timestep_indices) is not list:
        timestep_indices_tmp = [None]*num_files
        for i in range(num_files):
            timestep_indices_tmp[i] = timestep_indices
        timestep_indices = timestep_indices_tmp
    
    timesteps = [None]*num_files
        
    for i in range(num_files):
        try:
            ncfile = Dataset(source_url_list[i])
            if (timestep_indices[i] is not None):
                timesteps[i] = ncfile.variables[time_str][timestep_indices[i][:]]
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
    
    assert(np.all(np.diff(timesteps)>=0))
    for i in range(num_files):
        timesteps[i] = timesteps[i] - t0
    
    #Generate intersections bathymetry
    H_m_mask = eta0.mask.copy()
    H_m = np.ma.array(H_m, mask=H_m_mask)
    for i in range(erode_land):
        new_water = H_m.mask ^ binary_erosion(H_m.mask)
        eps = 1.0e-5 #Make new Hm slighlyt different from land_value
        eta0_dil = grey_dilation(eta0.filled(0.0), size=(3,3))
        H_m[new_water] = land_value+eps
        eta0[new_water] = eta0_dil[new_water]
    
    if reduced_gravity_interface == None:
        H_i, _ = OceanographicUtilities.midpointsToIntersections(H_m, land_value=land_value, iterations=iterations)
        eta0 = eta0[1:-1, 1:-1]
        h0 = OceanographicUtilities.intersectionsToMidpoints(H_i).filled(land_value) + eta0.filled(0.0)
    else: 
        H_i, _ = OceanographicUtilities.midpointsToIntersections(H_m, land_value=land_value, iterations=iterations)
        H_i = np.ma.minimum(H_i, reduced_gravity_interface)
        eta0 = eta0[1:-1, 1:-1]
        print("Cut the bathymetry: no reconstruction!")
    
    #Generate physical variables
    eta0 = np.ma.array(eta0.filled(0), mask=eta0.mask.copy())
    if reduced_gravity_interface == None:
        hu0 = np.ma.array(h0*u0, mask=eta0.mask.copy())
        hv0 = np.ma.array(h0*v0, mask=eta0.mask.copy())
    else:
        hu0 = depth_integration(source_url, reduced_gravity_interface, x0, x1, y0, y1, "u")
        hv0 = depth_integration(source_url, reduced_gravity_interface, x0, x1, y0, y1, "v")
        print("Depth integration with trapeziodal rule, ignoring eta")
    
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
    
    #Gravity and friction
    #FIXME: Friction coeff from netcdf?
    if reduced_gravity_interface == None:
        ic['g'] = 9.81
    else: 
        ic['g'] = 0.1
        print("Reduce gravity (fixed value used)")
    ic['r'] = 3.0e-3
    
    #Physical variables
    ic['H'] = H_i
    ic['eta0'] = fill_coastal_data(eta0)
    ic['hu0'] = hu0
    ic['hv0'] = hv0
    
    #Coriolis angle and beta
    ic['angle'] = angle
    ic['latitude'] = OceanographicUtilities.degToRad(latitude)
    ic['f'] = 0.0 #Set using latitude instead
    # The beta plane of doing it:
    # ic['f'], ic['coriolis_beta'] = OceanographicUtilities.calcCoriolisParams(OceanographicUtilities.degToRad(latitude[0, 0]))
    
    #Boundary conditions
    if reduced_gravity_interface == None:
        ic['boundary_conditions_data'] = getBoundaryConditionsData(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1, norkyst_data)
    else:
        ic['boundary_conditions_data'] = getBoundaryConditionsData(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1, norkyst_data, reduced_gravity_interface)
        print("Depth integration with trapeziodal rule, ignoring eta")
    ic['boundary_conditions'] = Common.BoundaryConditions(north=3, south=3, east=3, west=3, spongeCells=sponge_cells)
    
    #Wind stress (shear stress acting on the ocean surface)
    ic['wind_stress'] = getWindSourceterm(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1)
    
    #wind (wind speed in m/s used for forcing on drifter)
    ic['wind'] = getWind(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1) 
    
    #Note
    ic['note'] = datetime.datetime.now().isoformat() + ": Generated from " + str(source_url_list)
    
    #Initial reference time and all timesteps
    ic['t0'] = t0
    ic['timesteps'] = np.ravel(timesteps)
    
    return ic

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


def getWind(source_url_list, timestep_indices, timesteps, x0, x1, y0, y1):
    """
    timestep_indices => index into netcdf-array, e.g. [1, 3, 5]
    timestep => time at timestep, e.g. [1800, 3600, 7200]
    """
    
    if type(source_url_list) is not list:
        source_url_list = [source_url_list]
    
    num_files = len(source_url_list)
    
    source_url = source_url_list[0]
    
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
    
    wind_source = WindStress.WindStress(t=np.ravel(timesteps).copy(), X=u_wind, Y=v_wind)
    
    return wind_source


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


def getCombinedInitialConditions(source_url_list, x0, x1, y0, y1,
                         reduced_gravity_interface, \
                         timestep_indices=None, \
                         norkyst_data = True,
                         land_value=5.0, \
                         iterations=10, \
                         sponge_cells={'north':20, 'south': 20, 'east': 20, 'west': 20}, \
                         erode_land=0, 
                         download_data=True):
    """
    Returning two set of sim_args 
    - one for barotropic simulation (full-depth integrated variables: eta=eta_full, huv=huv_full)
    - one for baroclinic simulation (using upper-layer-integrated variables: eta=0, huv=h(uv_upper - uv_full))
    """

    full_IC = getInitialConditions(source_url_list, x0, x1, y0, y1, \
                         timestep_indices=timestep_indices, \
                         norkyst_data = norkyst_data,
                         land_value=land_value, \
                         iterations=iterations, \
                         sponge_cells=sponge_cells, \
                         erode_land=erode_land, 
                         download_data=download_data,
                         reduced_gravity_interface=None)

    upper_IC = getInitialConditions(source_url_list, x0, x1, y0, y1, \
                         timestep_indices=timestep_indices, \
                         norkyst_data = norkyst_data,
                         land_value=land_value, \
                         iterations=iterations, \
                         sponge_cells=sponge_cells, \
                         erode_land=erode_land, 
                         download_data=download_data,
                         reduced_gravity_interface=reduced_gravity_interface)

    barotropic_IC = copy.deepcopy(full_IC)
    baroclinic_IC = copy.deepcopy(upper_IC)

    # Set initial conditions
    # eta
    upper_eta0 = upper_IC["eta0"]
    full_eta0  = full_IC["eta0"]
    baroclinic_IC["eta0"] = upper_eta0 - full_eta0

    # H
    if type(source_url_list) is not list:
        source_url_list = [source_url_list]
    nc = MFDataset(source_url_list)
    #print("This download maybe avoided by interpolation (and using the right mask)")
    full_eta0 = np.ma.array(nc['zeta'][0,y0:y1, x0:x1])
    full_H  = np.ma.array(nc['h'][y0:y1, x0:x1], mask=full_eta0.mask.copy())
    upper_H = np.ma.minimum(full_H, 25.0)

    # hu
    upper_u0 = upper_IC["hu0"]/(upper_H + upper_IC["eta0"])
    full_u0  = full_IC["hu0"]/(full_H + full_IC["eta0"])
    baroclinic_IC["hu0"]  = upper_H*(upper_u0 - full_u0)

    #hv
    upper_v0 = upper_IC["hv0"]/(upper_H + upper_IC["eta0"])
    full_v0  = full_IC["hv0"]/(full_H + full_IC["eta0"])
    baroclinic_IC["hv0"]  = upper_H*(upper_v0 - full_v0)

    # Prepare boundary conditions
    # NOTE: The following download is repetitive but with the current code design likely not avoidable
    full_eta = np.ma.array(nc['zeta'][:, y0-1:y1+1, x0-1:x1+1])
    
    full_H  = np.ma.array(nc['h'][y0-1:y1+1, x0-1:x1+1], mask=full_eta[0].mask.copy())
    upper_H = np.ma.minimum(full_H, reduced_gravity_interface) #CAREFUL! Ensure to use same calculation as in getInitialConditions()!!!

    for cardinal in ["north", "east", "south", "west"]:
        upper_eta_bc = getattr(getattr(upper_IC["boundary_conditions_data"], cardinal), "h")
        full_eta_bc  = getattr(getattr(full_IC["boundary_conditions_data"], cardinal), "h")
        setattr(getattr(baroclinic_IC["boundary_conditions_data"], cardinal), "h", np.zeros_like(upper_eta_bc, dtype=np.float32))

        if cardinal == "north":
            full_H_bc = full_H[-1,1:-1]
            upper_H_bc = upper_H[-1,1:-1]
            mask = full_eta[:,-1,1:-1].mask
        elif cardinal == "south":
            full_H_bc = full_H[0,1:-1]
            upper_H_bc = upper_H[0,1:-1]
            mask = full_eta[:,0,1:-1].mask
        elif cardinal == "west":
            full_H_bc = full_H[1:-1,0]
            upper_H_bc = upper_H[1:-1,0]
            mask = full_eta[:,1:-1,0].mask
        elif cardinal == "east":
            full_H_bc = full_H[1:-1,-1]
            upper_H_bc = upper_H[1:-1,-1]
            mask = full_eta[:,1:-1,-1].mask
        full_h_bc  = full_H_bc  + np.ma.array(full_eta_bc, mask=mask)
        full_u_bc  = getattr(getattr(full_IC["boundary_conditions_data"], cardinal), "hu")/full_h_bc
        full_v_bc  = getattr(getattr(full_IC["boundary_conditions_data"], cardinal), "hv")/full_h_bc
        upper_h_bc = upper_H_bc + np.ma.array(upper_eta_bc, mask=mask)
        upper_u_bc = getattr(getattr(upper_IC["boundary_conditions_data"], cardinal), "hu")/upper_h_bc
        upper_v_bc = getattr(getattr(upper_IC["boundary_conditions_data"], cardinal), "hv")/upper_h_bc

        baroclinic_hu_bc = (upper_h_bc*(upper_u_bc - full_u_bc)).filled(0)
        baroclinic_hv_bc = (upper_h_bc*(upper_v_bc - full_v_bc)).filled(0)


        setattr(getattr(baroclinic_IC["boundary_conditions_data"], cardinal), "hu", np.float32(baroclinic_hu_bc))
        setattr(getattr(baroclinic_IC["boundary_conditions_data"], cardinal), "hv", np.float32(baroclinic_hv_bc))

    return barotropic_IC, baroclinic_IC


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


def removeMetadata(old_ic):
    ic = old_ic.copy()
    
    ic.pop('note', None)
    ic.pop('NX', None)
    ic.pop('NY', None)
    ic.pop('sponge_cells', None)
    ic.pop('t0', None)
    ic.pop('timesteps', None)
    ic.pop('wind', None)
    
    return ic


def removeCombinedMetadata(old_barotropic_IC, old_baroclinic_IC):
    barotropic_IC = removeMetadata(old_barotropic_IC)
    baroclinic_IC = removeMetadata(old_baroclinic_IC)

    IC = copy.deepcopy(barotropic_IC)
    for key in ["eta0", "hu0", "hv0", "H", "g", "boundary_conditions_data"]:
        IC["barotropic_"+key] = IC.pop(key)
        IC["baroclinic_"+key] = baroclinic_IC[key]

    return IC

def potentialDensities(source_url, t=0, x0=0, x1=-1, y0=0, y1=-1):

    s_nc = Dataset(source_url)

    # Collect information about s-levels
    s_lvls = s_nc["Cs_r"][:].data

    # Collect information about domain
    s_hs   = s_nc["h"][y0:y1,x0:x1] + s_nc["zeta"][t,y0:y1,x0:x1]
    s_lats = s_nc["lat_rho"][y0:y1,x0:x1]

    # Fetch temperature and salinity from nc-file 
    s_temps = s_nc["temp"][t,:,y0:y1,x0:x1]
    s_sals  = s_nc["salt"][t,:,y0:y1,x0:x1]

    # Transform depths to pressures 
    s_depths = np.ma.array(np.multiply.outer(s_lvls,s_hs), mask=s_temps.mask.copy())
    s_pressures = sw.eos80.pres(-s_depths,s_lats)

    # Calculate potential densities from salinity, temperature and pressure (depth)
    s_pot_densities = sw.eos80.pden(s_sals,s_temps,s_pressures)

    return s_pot_densities


def MLD(source_url, thres, min_mld, max_mld=None, t=0, x0=0, x1=-1, y0=0, y1=-1):
    """
    Calculates the mixed layer depth (MLD) 
    by finding smoothed isopynic line along "thres" 
    or "min_mld" if the entire water column is heavier than thres

    Input_
    source_url: url to s-level file on thredds.met.no
    t: time index 
    x0, x1, y0, y1: indices specifying a subset of the domain 

    Output_
    mld: masked array with mixed layer depth as positive value in meters 
    """

    s_nc = Dataset(source_url)

    # Collect information about s-levels
    s_lvls = s_nc["Cs_r"][:].data


    # Calculate potential densities 
    s_pot_densities = potentialDensities(source_url, t=t, x0=x0, x1=x1, y0=y0, y1=y1)

    # Collect information about domain
    s_hs   = s_nc["h"][y0:y1,x0:x1] + s_nc["zeta"][t,y0:y1,x0:x1]
    ny, nx = s_hs.shape 
    s_depths = np.ma.array(np.multiply.outer(s_lvls,s_hs), mask=s_pot_densities.mask.copy())

    ## Get MLD by interpolation between the two s-levels where one has lighter and the next heavier water than thres 
    # if all s-levels are heavier than the threshold, then set the upper layer (last index, while argmax fills with 0 in such cases)
    mld_base_idx = np.ma.maximum(np.argmax(s_pot_densities < thres, axis=0), (len(s_lvls)-1)*(s_pot_densities[-1] > thres))
    # ensure that it is not the base index is not the last level 
    mld_base_idx = np.ma.maximum(mld_base_idx, 1)

    mld_prog_idx = mld_base_idx-1

    mld_depth_base = -np.take_along_axis(s_depths, mld_base_idx.reshape(1,ny,nx), axis=0)[0]
    mld_dens_base = np.take_along_axis(s_pot_densities, mld_base_idx.reshape(1,ny,nx), axis=0)[0]

    mld_depth_prog = -np.take_along_axis(s_depths, mld_prog_idx.reshape(1,ny,nx), axis=0)[0]
    mld_dens_prog = np.take_along_axis(s_pot_densities, mld_prog_idx.reshape(1,ny,nx), axis=0)[0]

    # Solving linear interpolation equals thres
    mld = (thres - mld_dens_base)*(mld_depth_prog - mld_depth_base)/(mld_dens_prog - mld_dens_base) + mld_depth_base

    # Again special attention to all-heavier locations 
    mld[s_pot_densities[-1] > thres] = 0

    # Bounding MLD between the bathymetry and min_mld
    mld = np.ma.minimum(mld, s_hs)
    
    if max_mld is not None:
        mld = np.ma.minimum(mld, max_mld)
    mld = np.ma.maximum(mld, min_mld)

    # ## Smoothing of MLD to avoid shocks 
    # mld = np.ma.array(gaussian_filter(mld, [1,1]), mask=s_temps[0].mask.copy())
    # # Bounding MLD between the bathymetry and min_mld
    # mld = np.ma.minimum(mld, s_hs)
    # mld = np.ma.maximum(mld, min_mld)
    # if max_mld is not None:
    #     mld = np.ma.minimum(mld, max_mld)

    return mld 


def MLD_integrator(source_url, mld, t=0, x0=0, x1=-1, y0=0, y1=-1):

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

    integrator = depths_diff * weights

    return integrator
