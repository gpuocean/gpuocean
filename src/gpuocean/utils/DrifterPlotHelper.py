# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2023 SINTEF Digital
Copyright (C) 2023 Norwegian Meteorological Institute

This python class aids in plotting results from the numerical 
simulations

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


from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import numpy as np
import time
import copy
import netCDF4

from gpuocean.utils import OceanographicUtilities


##################################################3
# BACKGROUND CANVASES

def background_from_netcdf(source_url, figsize=None, t_idx=0, domain=[0, None, 0, None], drifter_domain=[0, None, 0, None],
                          cmap=plt.cm.Oranges, vmax=1, cbar=True, lonlat_diff=None, **kwargs):
    """
    Creating a background canvas from netCDF files
    
    source_url  - link or path to netCDF file
    t_idx       - time index (int) in netCDF file
    domain      - [x0, x1, y0, y1] indices (int) spanning a frame in the grid of the netCDF file
    drifter_domain - [x0, x1, y0, y1] indices (int) spanning a frame inside of the plotting frame
    cmap        - plt.colormap for velocities
    vmax        - maximal velocity 
    cbar        - boolean for adding colorbar or not
    lonlat_diff - float with distance between longitudes and latitudes that are plotted on top of the domain

    Note: `domain` sets the x/y axis extent and is therefore different from `drifter_domain`
    """
    fig, ax = plt.subplots(1,1, figsize=figsize)    

    # Defining extent
    x0, x1, y0, y1 = domain

    # Loading ocean data from netCDF file
    try:
        ncfile = netCDF4.Dataset(source_url)
        H_m = ncfile.variables['h'][y0:y1, x0:x1]
        eta = ncfile.variables['zeta'][t_idx, y0:y1, x0:x1]
        hu = ncfile.variables['ubar'][t_idx, y0:y1, x0:x1]
        hv = ncfile.variables['vbar'][t_idx, y0:y1, x0:x1]
        
        hu = hu * (H_m + eta)
        hv = hv * (H_m + eta)
        
    except Exception as e:
        raise e
    finally:
        ncfile.close()

    # Loading grid data from netCDF file
    try:
        ncfile = netCDF4.Dataset(source_url)
        x = ncfile.variables['X'][x0:x1]
        y = ncfile.variables['Y'][y0:y1]

        dx = np.average(x[1:] - x[:-1])
        dy = np.average(y[1:] - y[:-1])

        ny, nx = H_m.shape
        
    except Exception as e:
        raise e
    finally:
        ncfile.close()

    extent = [0, nx*dx/1000, 0, ny*dy/1000]

    # Plotting velocity field
    u = OceanographicUtilities.desingularise(eta + H_m, hu, 0.00001)
    v = OceanographicUtilities.desingularise(eta + H_m, hv, 0.00001)
    velo = np.sqrt(u**2 + v**2)

    cmap = copy.copy(cmap)
    cmap.set_bad("grey", alpha=0.5)

    im = ax.imshow(velo, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, **kwargs)

    # Colorbar
    if cbar:
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        ax_divider = make_axes_locatable(ax)
        cbax = ax_divider.append_axes("right", size="5%", pad="2%")

        cb = plt.colorbar(im, cax=cbax)
        cb.set_label(label="velocity [$m/s^2$]")
        
    # Grid 
    if lonlat_diff is not None:
        try:
            nc = netCDF4.Dataset(source_url)
            lat = np.array(nc.variables["lat"])
            lon = np.array(nc.variables["lon"])

            spec_lat = lat[y0:y1, x0:x1]
            spec_lon = lon[y0:y1, x0:x1]

            cont_lon_case = ax.contour(spec_lon, levels = np.arange(0, 90, lonlat_diff), extent=extent, colors='k', alpha=0.2, linewidths=0.8, zorder=1)
            cont_lat_case = ax.contour(spec_lat, levels = np.arange(0, 90, lonlat_diff), extent=extent, colors='k', alpha=0.2, linewidths=0.8, zorder=1)

        except Exception as e:
            raise e
        
    set_drifter_zoom(ax, extent, drifter_domain, dx, dy)

    return ax


def background_from_sim(sim, figsize=None, domain=[0, None, 0, None], drifter_domain=[0, None, 0, None],
                          cmap=plt.cm.Oranges, vmax=1, cbar=True, **kwargs):
    """
    Creating a background canvas from sim
    
    sim - CDKLM simulator
    domain      - [x0, x1, y0, y1] indices (int) spanning a frame in the grid of the simulator
    drifter_domain - [x0, x1, y0, y1] indices (int) spanning a frame inside of the plotting frame
    cmap        - plt.colormap for velocities
    vmax        - maximal velocity 
    cbar        - boolean for adding colorbar or not

    Note: `domain` sets the x/y axis extent and is therefore different from `drifter_domain`
    """

    fig, ax = plt.subplots(1,1, figsize=figsize)    

    # Defining extent
    x0, x1, y0, y1 = domain

    # Loading ocean data from netCDF file
    eta, hu, hv = [state_var[y0:y1, x0:x1] for state_var in sim.download(interior_domain_only=True)]
    H_m = sim.bathymetry.download(gpu_stream=sim.gpu_stream)[1][2:-2,2:-2][y0:y1, x0:x1]

    dx, dy = sim.dx, sim.dy
    nx, ny = sim.nx, sim.ny

    extent = [0, nx*dx/1000, 0, ny*dy/1000]

    # Plotting velocity field
    from gpuocean.utils import OceanographicUtilities
    u = OceanographicUtilities.desingularise(eta + H_m, hu, 0.00001)
    v = OceanographicUtilities.desingularise(eta + H_m, hv, 0.00001)
    velo = np.sqrt(u**2 + v**2)

    import copy
    cmap = copy.copy(cmap)
    cmap.set_bad("grey", alpha=0.5)

    im = ax.imshow(velo, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, **kwargs)

    # Colorbar
    if cbar:
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        ax_divider = make_axes_locatable(ax)
        cbax = ax_divider.append_axes("right", size="5%", pad="2%")

        cb = plt.colorbar(im, cax=cbax)
        cb.set_label(label="velocity [$m/s^2$]")

    set_drifter_zoom(ax, extent, drifter_domain, dx, dy)

    return ax



def background_from_ensemble(ensemble, figsize=None, domain=[0, None, 0, None], drifter_domain=[0, None, 0, None],
                          cmap=plt.cm.Oranges, vmax=1, **kwargs):
    """
    Creating a background canvas from sim
    
    ensemble    - OceanModelEnsemble
    domain      - [x0, x1, y0, y1] indices (int) spanning a frame in the grid of the simulator
    drifter_domain - [x0, x1, y0, y1] indices (int) spanning a frame inside of the plotting frame
    cmap        - plt.colormap for velocities
    vmax        - maximal velocity 

    Note: `domain` sets the x/y axis extent and is therefore different from `drifter_domain`
    """

    fig, ax = plt.subplots(1,1, figsize=figsize)    

    # Defining extent
    x0, x1, y0, y1 = domain

    # Loading ocean data from netCDF file
    eta, hu, hv = [state_var[y0:y1, x0:x1] for state_var in ensemble.particles[0].download(interior_domain_only=True)]
    H_m = ensemble.particles[0].bathymetry.download(gpu_stream=ensemble.particles[0].gpu_stream)[1][2:-2,2:-2][y0:y1, x0:x1]

    dx, dy = ensemble.particles[0].dx, ensemble.particles[0].dy
    nx, ny = ensemble.particles[0].nx, ensemble.particles[0].ny

    extent = [0, nx*dx/1000, 0, ny*dy/1000]

    # Plotting velocity field
    # TODO: Add option for mean velo field
    cmap = copy.copy(cmap)
    cmap.set_bad("grey", alpha=0.5)

    im = ax.imshow(np.ma.array(np.zeros_like(eta), mask=copy.copy(ensemble.particles[0].getLandMask())), origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, **kwargs)
    
    set_drifter_zoom(ax, extent, drifter_domain, dx, dy)

    return ax



##################################################3
# DRIFTERS ON CANVAS

def add_drifter_on_background(ax, obs, drifter_id=0, color="blue", label=None,
                              start_t=None, end_t=None, **kwargs):

    if start_t is None:
        start_t = obs.obs_df["time"].iloc[0]
    if end_t is None:
        end_t = obs.obs_df["time"].iloc[-1]

    paths = obs.get_drifter_path(drifter_id=drifter_id, start_t=start_t, end_t=end_t, in_km=True)
    
    ax.plot(paths[0][:,0], paths[0][:,1], c=color, label=label,**kwargs)
    if len(paths) > 1:
        for path in paths[1:]:
            ax.plot(path[:,0], path[:,1], c=color, **kwargs)


def add_ensemble_drifter_on_background(ax, ensemble_obs, drifter_id=0, 
                                        color="blue", start_t=None, end_t=None, 
                                        **kwargs):

    if start_t is None:
        start_t = ensemble_obs[0].obs_df["time"].iloc[0]
    if end_t is None:
        end_t = ensemble_obs[0].obs_df["time"].iloc[-1]

    for obs in ensemble_obs:
        paths = obs.get_drifter_path(drifter_id=drifter_id, start_t=start_t, end_t=end_t, in_km=True)
        
        for path in paths:
            ax.plot(path[:,0], path[:,1], c=color,**kwargs)


##################################################3
# UTILS TO GET DRIFTER DOMAIN

def domain_around_drifter(obs, drifter_id, frame_in_km, domain=[0,None,0,None],
                          start_t=None, end_t=None):

    if start_t is None:
        start_t = obs.obs_df["time"].iloc[0]
    if end_t is None:
        end_t = obs.obs_df["time"].iloc[-1]

    dx = obs.domain_size_x/obs.nx
    dy = obs.domain_size_y/obs.ny

    paths = obs.get_drifter_path(drifter_id, start_t, end_t, in_km=True)

    x0 = np.min(np.array([np.min(path[:,0]) for path in paths]))
    x1 = np.max(np.array([np.max(path[:,0]) for path in paths]))

    y0 = np.min(np.array([np.min(path[:,1]) for path in paths]))
    y1 = np.max(np.array([np.max(path[:,1]) for path in paths]))


    x0, x1 = np.maximum(0, x0-frame_in_km), np.minimum(x1+frame_in_km, obs.domain_size_x/1000)
    y0, y1 = np.maximum(0, y0-frame_in_km), np.minimum(y1+frame_in_km, obs.domain_size_y/1000)

    x0 = domain[0]*dx + x0
    if domain[1] is not None:
        x1 = domain[1]*dx + x1
    y0 = domain[2]*dy + y0
    if domain[3] is not None:
        y1 = domain[3]*dy + y1 

    drifter_domain = [np.floor(x0/(dx/1000)).astype(int), np.ceil(x1/(dx/1000)).astype(int), 
                      np.floor(y0/(dy/1000)).astype(int), np.ceil(y1/(dy/1000)).astype(int)]

    return drifter_domain


def set_drifter_zoom(ax, extent, drifter_domain, dx, dy):
    x0 = drifter_domain[0]*dx/1000
    x1 = extent[1]
    if drifter_domain[1] is not None:
        x1 = drifter_domain[1]*dx/1000
    y0 = drifter_domain[2]*dy/1000
    y1 = extent[3]
    if drifter_domain[3] is not None:
        y1 = drifter_domain[3]*dy/1000
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
