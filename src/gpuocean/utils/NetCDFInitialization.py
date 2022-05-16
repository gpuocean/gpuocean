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



import numpy as np
import datetime, os, copy
from netCDF4 import Dataset
import pyproj
from scipy.ndimage.morphology import binary_erosion, grey_dilation

from SWESimulators import Common, WindStress, OceanographicUtilities


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
                    hu = ncfile.variables['ubar'][timestep_index, y0-1:y1+1, x0-1:x1+1]
                    hu = hu.filled(0) #zero on land
                else: 
                    hu = ncfile.variables['ubar'][timestep_index, y0-1:y1+1, x0-1:x1+2]
                    hu = hu.filled(0) #zero on land
                    hu = (hu[:,1:] + hu[:, :-1]) * 0.5
                    
                hu = h*hu

                bc_hu['north'][bc_index] = hu[-1, 1:-1]
                bc_hu['south'][bc_index] = hu[0, 1:-1]
                bc_hu['east'][bc_index] = hu[1:-1, -1]
                bc_hu['west'][bc_index] = hu[1:-1, 0]

                if norkyst_data:
                    hv = ncfile.variables['vbar'][timestep_index, y0-1:y1+1, x0-1:x1+1]
                    hv = hv.filled(0) #zero on land
                else:
                    hv = ncfile.variables['vbar'][timestep_index, y0-1:y1+2, x0-1:x1+1]
                    hv = hv.filled(0) #zero on land
                    hv = (hv[1:,:] + hv[:-1, :]) * 0.5
                hv = h*hv

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
    
    for i in range(num_files):
        try:
            ncfile = Dataset(source_url_list[i])
            u_wind_list[i] = ncfile.variables['Uwind'][timestep_indices[i], y0:y1, x0:x1]
            v_wind_list[i] = ncfile.variables['Vwind'][timestep_indices[i], y0:y1, x0:x1]
        except Exception as e:
            raise e
        finally:
            ncfile.close()

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
        {'name': 'skagerak',       'x0':  300, 'x1':  600, 'y0':   50, 'y1':  250 },
        {'name': 'oslo',           'x0':  500, 'x1':  550, 'y0':  160, 'y1':  210 },
        {'name': 'denmark',        'x0':    2, 'x1':  300, 'y0':    2, 'y1':  300 },
        {'name': 'lovese',         'x0': 1400, 'x1': 2034, 'y0':  450, 'y1':  769 }
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
                         download_data=True):
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
    
    #Gravity and friction
    #FIXME: Friction coeff from netcdf?
    ic['g'] = 9.81
    ic['r'] = 3.0e-3
    
    #Physical variables
    ic['H'] = H_i
    ic['eta0'] = eta0
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
    
    for i in range(num_files):
        try:
            ncfile = Dataset(source_url_list[i])
            u_wind_list[i] = ncfile.variables['Uwind'][timestep_indices[i], y0:y1, x0:x1]
            v_wind_list[i] = ncfile.variables['Vwind'][timestep_indices[i], y0:y1, x0:x1]
        except Exception as e:
            raise e
        finally:
            ncfile.close()

    u_wind = u_wind_list[0].filled(0)
    v_wind = v_wind_list[0].filled(0)
    for i in range(1, num_files):
        u_wind = np.concatenate((u_wind, u_wind_list[i].filled(0)))
        v_wind = np.concatenate((v_wind, v_wind_list[i].filled(0)))
    
    u_wind = u_wind.astype(np.float32)
    v_wind = v_wind.astype(np.float32)
    
    wind_source = WindStress.WindStress(t=np.ravel(timesteps).copy(), X=u_wind, Y=v_wind)
    
    return wind_source

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
