# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2023 Norwegian Meteorological Institute
Copyright (C) 2023 SINTEF Digital

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
from netCDF4 import Dataset, MFDataset

import seawater as sw
from scipy.ndimage.filters import convolve, gaussian_filter

from gpuocean.utils import NetCDFInitialization


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

    return NetCDFInitialization.vertical_integrator(source_url, mld, t, x0, x1, y0, y1)


def correct_coastal_MLD(mld, source_url, coords=[0,None,0,None], rel_tol=0.25, abs_tol=1, K=20, land_value=0.0):
    """
    The MLD in shallow coastal may be restricted by the bottom topography. 
    rel_tol - parameter between [0,1] for the tolerated relative thickness of the lower layer
    abs_tol - parameter in meter for the tolerated absolute thickness of the lower layer
    If a MLD value is not tolerated, it is replaced by the K-th nearest neighbor average of tolerated values

    FIXME: The iterated call does not necessarily result in 0 corrections, since corrected values loose relation to the checked condition 
    """

    x0, x1 = coords[0], coords[1]
    y0, y1 = coords[2], coords[3]

    s_nc = Dataset(source_url)
    s_hs = s_nc["h"][y0:y1,x0:x1]

    bad_yx = np.where(np.logical_and(np.logical_or(np.abs(mld - s_hs) < rel_tol*s_hs, np.abs(mld - s_hs) < abs_tol), s_hs!=land_value))
    bad_mask = np.where(np.logical_or((s_hs==land_value), np.logical_or(np.abs(mld - s_hs) < rel_tol*s_hs, np.abs(mld - s_hs) < abs_tol)),1,0)

    Xidx = np.arange(0, mld.shape[1])
    Yidx = np.arange(0, mld.shape[0])

    xx, yy = np.meshgrid(Xidx, Yidx)

    for i in range(len(bad_yx[0])):
        dists = (xx-bad_yx[1][i])**2 + (yy-bad_yx[0][i])**2 + 1e5*bad_mask
        sum = 0.0
        for k in range(K): 
            sum += mld[np.unravel_index(dists.argmin(), dists.shape)]
            dists[np.unravel_index(dists.argmin(), dists.shape)] = 1e5
        mld[bad_yx[0][i],bad_yx[1][i]] = sum/K

    print(str(len(bad_yx[0])) + " values corrected")

    return mld



def getCombinedInitialConditions(source_url, x0, x1, y0, y1, mld_dens,
                         timestep_indices=None, \
                         norkyst_data = True,
                         land_value=5.0, \
                         iterations=10, \
                         sponge_cells={'north':20, 'south': 20, 'east': 20, 'west': 20}, \
                         erode_land=0, 
                         download_data=True):
    """
    For details see "getInitiaiConditions"-function
    
    Returning two set of sim_args 
    - one for barotropic simulation (full-depth integrated variables: eta=eta_full, huv=huv_full)
    - one for baroclinic simulation (using upper-layer-integrated variables: eta=0, huv=h(uv_upper - uv_full))
    """

    full_IC = NetCDFInitialization.getInitialConditions(source_url, x0, x1, y0, y1, \
                         timestep_indices=timestep_indices, \
                         norkyst_data = norkyst_data,
                         land_value=land_value, \
                         iterations=iterations, \
                         sponge_cells=sponge_cells, \
                         erode_land=erode_land, 
                         download_data=download_data)


    barotropic_IC = copy.deepcopy(full_IC)
    baroclinic_IC = copy.deepcopy(full_IC)

    if timestep_indices is None:
        t0_idx = 0
    else:
        t0_idx = timestep_indices[0][0]
    mld = MLD(source_url, mld_dens, min_mld=1.5, max_mld=40, x0=x0, x1=x1, y0=y0, y1=y1, t=t0_idx)
    mld = NetCDFInitialization.fill_coastal_data(mld)
    ml_integrator = MLD_integrator(source_url, mld, x0=x0, x1=x1, y0=y0, y1=y1)

    # Set initial conditions
    # eta
    baroclinic_IC["eta0"] = mld 

    # currents
    nc = Dataset(source_url)
    u0 = nc.variables['u'][t0_idx, :, y0:y1, x0:x1+1]
    v0 = nc.variables['v'][t0_idx, :, y0:y1+1, x0:x1]
    
    u0 = u0.filled(fill_value = 0.0) 
    v0 = v0.filled(fill_value = 0.0) 
    u0 = (u0[:, :,1:] + u0[:, :, :-1]) * 0.5 #Find u,v at cell centers
    v0 = (v0[:, 1:,:] + v0[:, :-1, :]) * 0.5 #Find u,v at cell centers

    full_Hm  = np.ma.array(nc['h'][y0:y1, x0:x1], mask=full_IC["eta0"].mask.copy())

    baroclinic_IC["hu0"] = (np.sum(ml_integrator * u0, axis=0)/mld - full_IC["hu0"]/full_Hm)*mld
    baroclinic_IC["hv0"] = (np.sum(ml_integrator * v0, axis=0)/mld - full_IC["hv0"]/full_Hm)*mld

    # H
    baroclinic_IC["H"] = np.ma.array(np.zeros_like(full_IC["H"]), mask=full_IC["H"].mask)

    # Reduced gravity
    s_pot_densities = potentialDensities(source_url, t=t0_idx, x0=x0, x1=x1, y0=y0, y1=y1)
    ml_pot_density = np.average(np.sum(ml_integrator * s_pot_densities, axis=0)/np.sum(ml_integrator, axis=0)) #NOTE: np.sum(integrator, axis=0)) = mld

    inverse_integrator = np.ma.array(np.ones_like(ml_integrator), mask=ml_integrator.mask.copy()) - ml_integrator
    deep_pot_density  = np.average(np.sum(inverse_integrator * s_pot_densities, axis=0)/np.sum(inverse_integrator, axis=0))

    eps = (deep_pot_density - ml_pot_density)/deep_pot_density

    baroclinic_IC["g"] = eps * full_IC["g"]

    # Prepare boundary conditions
    if timestep_indices is not None:
        t_range = timestep_indices[0]
    else:
        t_range = np.arange(len(nc["ocean_time"][:]))

    # NOTE: The following download is repetitive but with the current code design likely not avoidable
    etas = np.ma.array(nc['zeta'][t_range, y0-1:y1+1, x0-1:x1+1])
    full_Hm  = np.ma.array(nc['h'][y0-1:y1+1, x0-1:x1+1], mask=etas[0].mask.copy())

    mlds = []
    hus = []
    hvs = []
    for t_idx in t_range:
        mld = NetCDFInitialization.fill_coastal_data(MLD(source_url, mld_dens, min_mld=1.5, max_mld=40, x0=x0-1, x1=x1+1, y0=y0-1, y1=y1+1, t=t_idx))
        ml_integrator = MLD_integrator(source_url, mld, t=t_idx, x0=x0-1, x1=x1+1, y0=y0-1, y1=y1+1)

        u = nc.variables['u'][t_idx, :, y0-1:y1+1, x0-1:x1+2].filled(fill_value = 0.0) 
        v = nc.variables['v'][t_idx, :, y0-1:y1+2, x0-1:x1+1].filled(fill_value = 0.0) 
        
        u = (u[:, :,1:] + u[:, :, :-1]) * 0.5 #Find u,v at cell centers
        v = (v[:, 1:,:] + v[:, :-1, :]) * 0.5 #Find u,v at cell centers

        mlds.append(mld)
        hus.append(np.sum(ml_integrator * u, axis=0))
        hvs.append(np.sum(ml_integrator * v, axis=0))

    mlds = np.ma.array(mlds)
    hus  = np.ma.array(hus)
    hvs  = np.ma.array(hvs)

    for cardinal in ["north", "east", "south", "west"]:

        full_eta_bc  = getattr(getattr(full_IC["boundary_conditions_data"], cardinal), "h")
        
        if cardinal == "north":
            full_H_bc = full_Hm[-1,1:-1]
            upper_h_bc = mlds[:,-1,1:-1]
            upper_hu_bc = hus[:,-1,1:-1]
            upper_hv_bc = hvs[:,-1,1:-1]
            mask = etas[:,-1,1:-1].mask
        elif cardinal == "south":
            full_H_bc = full_Hm[0,1:-1]
            upper_h_bc = mlds[:,0,1:-1]
            upper_hu_bc = hus[:,0,1:-1]
            upper_hv_bc = hvs[:,0,1:-1]
            mask = etas[:,0,1:-1].mask
        elif cardinal == "west":
            full_H_bc = full_Hm[1:-1,0]
            upper_h_bc = mlds[:,1:-1,0]
            upper_hu_bc = hus[:,1:-1,0]
            upper_hv_bc = hvs[:,1:-1,0]
            mask = etas[:,1:-1,0].mask
        elif cardinal == "east":
            full_H_bc = full_Hm[1:-1,-1]
            upper_h_bc = mlds[:,1:-1,-1]
            upper_hu_bc = hus[:,1:-1,-1]
            upper_hv_bc = hvs[:,1:-1,-1]
            mask = etas[:,1:-1,-1].mask
        full_h_bc  = full_H_bc  + np.ma.array(full_eta_bc, mask=mask)
        full_u_bc  = getattr(getattr(full_IC["boundary_conditions_data"], cardinal), "hu")/full_h_bc
        full_v_bc  = getattr(getattr(full_IC["boundary_conditions_data"], cardinal), "hv")/full_h_bc
        upper_u_bc = upper_hu_bc/upper_h_bc
        upper_v_bc = upper_hv_bc/upper_h_bc

        baroclinic_hu_bc = (upper_h_bc*(upper_u_bc - full_u_bc)).filled(0)
        baroclinic_hv_bc = (upper_h_bc*(upper_v_bc - full_v_bc)).filled(0)

        setattr(getattr(baroclinic_IC["boundary_conditions_data"], cardinal), "h",  np.float32(upper_h_bc))
        setattr(getattr(baroclinic_IC["boundary_conditions_data"], cardinal), "hu", np.float32(baroclinic_hu_bc))
        setattr(getattr(baroclinic_IC["boundary_conditions_data"], cardinal), "hv", np.float32(baroclinic_hv_bc))

    return barotropic_IC, baroclinic_IC




def removeCombinedMetadata(old_barotropic_IC, old_baroclinic_IC):
    barotropic_IC = NetCDFInitialization.removeMetadata(old_barotropic_IC)
    baroclinic_IC = NetCDFInitialization.removeMetadata(old_baroclinic_IC)

    IC = copy.deepcopy(barotropic_IC)
    for key in ["eta0", "hu0", "hv0", "H", "g", "boundary_conditions_data", "wind"]:
        IC["barotropic_"+key] = IC.pop(key)
        IC["baroclinic_"+key] = baroclinic_IC[key]

    return IC
