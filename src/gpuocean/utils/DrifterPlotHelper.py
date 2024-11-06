# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2023, 2024 SINTEF Digital
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
import copy

from gpuocean.utils import OceanographicUtilities



##################################################3
# BACKGROUND CANVASES

def background_from_netcdf(source_url, ax=None, figsize=None, t_idx=0, domain=[0, None, 0, None], drifter_domain=[0, None, 0, None],
                           background_type="velocity", cmap=None, vmax=None, cbar=True, lonlat_diff=None, **kwargs):
    """
    Creating a background canvas from netCDF files
    
    source_url  - link or path to netCDF file
    t_idx       - time index (int) in netCDF file
    domain      - [x0, x1, y0, y1] indices (int) spanning a frame in the grid of the netCDF file
    drifter_domain - [x0, x1, y0, y1] indices (int) spanning a frame inside of the plotting frame
    background_type -  any of the following strings: [eta, velocity*, velocity_variance, velocity_stddev, landmask] (* default)
    cmap        - plt.colormap for velocities
    vmax        - maximal velocity 
    cbar        - boolean for adding colorbar or not
    lonlat_diff - float with distance between longitudes and latitudes that are plotted on top of the domain

    Note: `domain` sets the x/y axis extent and is therefore different from `drifter_domain`
    """
   
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

    ax, extent = make_generic_background(dx, dy, ax=ax,
                                         eta=eta, hu=hu, hv=hv, landmask=np.isnan(eta), H_m=H_m,
                                         u=None, v=None, u_var=None, v_var=None,
                                         figsize=figsize, cmap=cmap, vmax=vmax,
                                         background_type=background_type,
                                         return_extent=True,
                                         **kwargs) 

    set_drifter_zoom(ax, extent, drifter_domain, dx, dy)

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


def background_from_sim(sim, ax=None, figsize=None, domain=[0, None, 0, None], drifter_domain=[0, None, 0, None],
                        background_type="velocity", cmap=None, vmax=None, cbar=True, **kwargs):
    """
    Creating a background canvas from sim
    
    sim - CDKLM simulator
    domain      - [x0, x1, y0, y1] indices (int) spanning a frame in the grid of the simulator
    drifter_domain - [x0, x1, y0, y1] indices (int) spanning a frame inside of the plotting frame
    background_type -  any of the following strings: [eta, velocity*, velocity_variance, velocity_stddev, landmask] (* default)
    cmap        - plt.colormap for velocities
    vmax        - maximal velocity 
    cbar        - boolean for adding colorbar or not

    Note: `domain` sets the x/y axis extent and is therefore different from `drifter_domain`
    """

    # Defining extent
    x0, x1, y0, y1 = domain

    # Loading ocean data from sim
    eta, hu, hv = [state_var[y0:y1, x0:x1] for state_var in sim.download(interior_domain_only=True)]
    _, H_m = sim.downloadBathymetry(interior_domain_only=True)
    H_m = H_m[y0:y1, x0:x1]
    landmask = sim.getLandMask(interior_domain_only=True)
    if landmask is not None:
        landmask = landmask[y0:y1, x0:x1]

    dx, dy = sim.dx, sim.dy
 
    ax, extent = make_generic_background(dx, dy, ax=ax,
                                         eta=eta, hu=hu, hv=hv, landmask=landmask, H_m=H_m,
                                         u=None, v=None, u_var=None, v_var=None,
                                         figsize=figsize, cmap=cmap, vmax=vmax,
                                         background_type=background_type,
                                         return_extent=True,
                                         **kwargs) 

    set_drifter_zoom(ax, extent, drifter_domain, dx, dy)

    return ax



def background_from_ensemble(ensemble, ax=None, figsize=None, domain=[0, None, 0, None], drifter_domain=[0, None, 0, None],
                             background_type="landmask", cmap=None, vmax=None, **kwargs):
    """
    Creating a background canvas from ensemble
    
    ensemble    - OceanModelEnsemble
    domain      - [x0, x1, y0, y1] indices (int) spanning a frame in the grid of the simulator
    background_type -  any of the following strings: [eta, velocity, velocity_variance, velocity_stddev, landmask*] (* default)
    drifter_domain - [x0, x1, y0, y1] indices (int) spanning a frame inside of the plotting frame
    cmap        - plt.colormap for velocities
    vmax        - maximal velocity 

    Note: `domain` sets the x/y axis extent and is therefore different from `drifter_domain`
    """

    # Defining extent
    x0, x1, y0, y1 = domain

    dx, dy = ensemble.particles[0].dx, ensemble.particles[0].dy

    # Loading ocean data from ensemble
    ensemble_mean = ensemble.estimate(np.mean)
    eta, hu, hv = [ensemble_mean[i, y0:y1, x0:x1] for i in range(3)]
    _, H_m = ensemble.particles[0].downloadBathymetry(interior_domain_only=True)
    H_m = H_m[y0:y1, x0:x1]
    landmask = ensemble.particles[0].getLandMask(interior_domain_only=True)
    if landmask is not None:
        landmask = landmask[y0:y1, x0:x1]

    mean_velocity = ensemble.estimateVelocity(np.mean)
    var_velocity  = ensemble.estimateVelocity(np.var)
    u,     v     = [mean_velocity[i, y0:y1, x0:x1] for i in range(2)]
    u_var, v_var = [ var_velocity[i, y0:y1, x0:x1] for i in range(2)]

    ax, extent = make_generic_background(dx, dy, ax=ax,
                                         eta=eta, hu=hu, hv=hv, landmask=landmask, H_m=H_m,
                                         u=u, v=v, u_var=u_var, v_var=v_var,
                                         figsize=figsize, cmap=cmap, vmax=vmax,
                                         background_type=background_type,
                                         return_extent=True,
                                         **kwargs) 

    set_drifter_zoom(ax, extent, drifter_domain, dx, dy)

    return ax

def background_from_mlensemble(mlensemble, ax=None, figsize=None, domain=[0, None, 0, None], drifter_domain=[0, None, 0, None],
                               background_type="velocity_variance", cmap=None, vmax=None, **kwargs):
    """
    Creating a background canvas from multi-level ensemble
    
    ensemble    - MultiLevelOceanEnsemble
    domain      - [x0, x1, y0, y1] indices (int) spanning a frame in the grid of the simulator
    background_type -  any of the following strings: [eta, velocity, velocity_variance*, velocity_stddev, landmask] (* default)
    drifter_domain - [x0, x1, y0, y1] indices (int) spanning a frame inside of the plotting frame
    cmap        - plt.colormap for velocities
    vmax        - maximal velocity 

    Note: `domain` sets the x/y axis extent and is therefore different from `drifter_domain`
    """

    # Defining extent
    x0, x1, y0, y1 = domain

    mlmean = mlensemble.estimate(np.mean)
    eta, hu, hv = [mlmean[i, y0:y1, x0:x1] for i in range(3)]
    _, H_m = mlensemble.ML_ensemble[-1][0][0].downloadBathymetry(interior_domain_only=True)
    H_m = H_m[y0:y1, x0:x1]
    landmask = mlensemble.ML_ensemble[-1][0][0].getLandMask(interior_domain_only=True)
    if landmask is not None:
        landmask = landmask[y0:y1, x0:x1]

    mlmean_velocity = mlensemble.estimateVelocity(np.mean)
    mlvar_velocity  = mlensemble.estimateVelocity(np.var)
    u,     v     = [mlmean_velocity[i, y0:y1, x0:x1] for i in range(2)]
    u_var, v_var = [ mlvar_velocity[i, y0:y1, x0:x1] for i in range(2)]
    dx, dy = mlensemble.dxs[-1], mlensemble.dys[-1]

    ax, extent = make_generic_background(dx, dy, ax=ax, 
                                         eta=eta, hu=hu, hv=hv, landmask=landmask, H_m=H_m,
                                         u=u, v=v, u_var=u_var, v_var=v_var,
                                         figsize=figsize, cmap=cmap, vmax=vmax,
                                         background_type=background_type,
                                         return_extent=True,
                                         **kwargs)  
    set_drifter_zoom(ax, extent, drifter_domain, dx, dy)

    return ax

def background_from_grid_parameters(nx, ny, dx, dy, ax=None, figsize=None, drifter_domain=[0, None, 0, None], **kwargs):
    

    ax, extent = make_generic_background(dx, dy, ax=ax, nx=nx, ny=ny,
                                         figsize=figsize, return_extent=True,
                                         background_type="empty",
                                         **kwargs)
    
    set_drifter_zoom(ax, extent, drifter_domain, dx, dy)

    return ax

def make_generic_background(dx, dy, ax=None, nx=None, ny=None,
                            eta=None, hu=None, hv=None, landmask=None, H_m=None,
                            u=None, v=None, u_var=None, v_var=None,
                            figsize=None, cmap=None, vmax=None, cbar=True,
                            background_type='landmask',
                            return_extent=False,
                            **kwargs):
    """
    Creating a background canvas with values directly from np.arrays
    
    dx, dy      - grid cell sizes
    domain      - [x0, x1, y0, y1] indices (int) spanning a frame in the grid of the simulator
    background_type -  any of the following strings: [eta, velocity, velocity_variance, velocity_stddev, landmask*, empty] (* default)
    drifter_domain - [x0, x1, y0, y1] indices (int) spanning a frame inside of the plotting frame
    cmap        - plt.colormap for velocities
    vmax        - maximal velocity 

    Note: `domain` sets the x/y axis extent and is therefore different from `drifter_domain`
    """
    _check_background_type(background_type)
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)

    if cmap is None:
        if background_type == "eta":
            cmap = plt.cm.BrBG
        elif background_type in ["velocity_variance", "velocity_stddev"]:
            cmap = plt.cm.Reds
        else: # "landmask" or "velocity"
            cmap = plt.cm.Oranges
    cmap = copy.copy(cmap)
    cmap.set_bad("grey", alpha=0.5)
    
    nx, ny = nx, ny
    if eta is not None:
        ny, nx = eta.shape
    elif u is not None:
        ny, nx = u.shape
    elif u_var is not None:
        ny, nx = u_var.shape
    extent = [0, nx*dx/1000, 0, ny*dy/1000]

    # Make the different backgrounds
    if background_type == 'empty':
        ax.imshow(np.zeros((nx,ny)), origin="lower", cmap=cmap, vmin=0.0, vmax=0.0, extent=extent, **kwargs)
        cbar = False
        
    elif background_type == 'landmask':
        assert(eta is not None), "require eta to make landmask background"
        if vmax is None:
            vmax = 1
        im = ax.imshow(np.ma.array(np.zeros_like(eta), mask=copy.copy(landmask)), 
                       origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, **kwargs)
        
    elif background_type == 'eta':
        assert(eta is not None), "require eta to make eta background"
        if vmax is None:
            vmax = np.max(np.abs(eta))
        im = ax.imshow(eta, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, extent=extent, **kwargs)
        
    elif background_type == 'velocity':
        if u is None or v is None:
            assert(eta is not None), "require eta to make velocity background when u and v is not provided"
            assert(hu is not None), "require hu to make velocity background when u and v is not provided"
            assert(hv is not None), "require hv to make velocity background when u and v is not provided"
            assert(H_m is not None), "require H_m to make velocity background when u and v is not provided"
            u = OceanographicUtilities.desingularise(eta + H_m, hu, 0.00001)
            v = OceanographicUtilities.desingularise(eta + H_m, hv, 0.00001)
        velo = np.sqrt(u**2 + v**2)
        if vmax is None:
            vmax = np.max(velo)

        im = ax.imshow(velo, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, **kwargs)

    elif background_type in ['velocity_variance', "velocity_stddev"]:
        assert(u_var is not None), "require u_var to make 'velocity_variance' or 'velocity_stddev' background"
        assert(v_var is not None), "require v_var to make 'velocity_variance' or 'velocity_stddev' background"
        velo_var = np.sqrt(u_var**2 + v_var**2)    
        if background_type == 'velocity_variance':
            if vmax is None:
                vmax = np.max(velo_var)
            im = ax.imshow(velo_var, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, **kwargs)        
        else:
            velo_stddev = np.sqrt(velo_var)
            if vmax is None:
                vmax = np.max(velo_stddev)
            im = ax.imshow(velo_stddev, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, **kwargs)        
    else:
        raise KeyError("Got value "+str(background_type)+
                       " as input for 'background_type'. Acceptable values are 'landmask', 'eta', 'velocity', 'velocity_stddev', and 'velocity_variance'.")
    
    
    # Colorbar
    if cbar and not background_type == 'landmask':
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        ax_divider = make_axes_locatable(ax)
        cbax = ax_divider.append_axes("right", size="5%", pad="2%")

        cb = plt.colorbar(im, cax=cbax)
        if background_type == 'velocity':
            cb.set_label(label="velocity [$m/s^2$]")
        elif background_type == 'eta':
            cb.set_label(label="water level [$m$]")
        elif background_type == 'velocity_variance':
            cb.set_label(label="velocity variance [$(m/s^2)^2$]")
        elif background_type == 'velocity_stddev':
            cb.set_label(label="velocity std.dev. [$m/s^2$]")

    if return_extent:
        return ax, extent
    return ax

def _check_background_type(background_type):
    valid_background_types = ["eta", "velocity", "velocity_variance", "velocity_stddev", "landmask", "empty"]
    assert(background_type in valid_background_types), "'"+str(background_type)+"' is an invalid background_type. Valid background_type values are "+str(valid_background_types)
        
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

def add_drifter_positions_on_background(ax, drifter_pos, color="black", **kwargs):
    ax.scatter(drifter_pos[:,0]/1000.0, drifter_pos[:,1]/1000.0, c=color, **kwargs)

##################################################3
# UTILS TO GET DRIFTER DOMAIN

def domain_around_drifter(obs, drifter_id, padding_in_km=1, domain=[0,None,0,None],
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


    x0, x1 = np.maximum(0, x0-padding_in_km), np.minimum(x1+padding_in_km, obs.domain_size_x/1000)
    y0, y1 = np.maximum(0, y0-padding_in_km), np.minimum(y1+padding_in_km, obs.domain_size_y/1000)

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


##################################################3
# Kernel Density Estimation

def add_kde_on_background(ax, ensemble_obs, drifter_id=0, cmap="Greens", label=None, add_legend=True,
                              drifter_t=None, **kwargs):
    
    if drifter_t is None:
        drifter_t = ensemble_obs[0].obs_df["time"].iloc[-1]

    ## Get last postions
    numTrajectories = len(ensemble_obs)

    last_positions = np.zeros((numTrajectories,2))
    for d in range(numTrajectories):
        last_positions[d] = ensemble_obs[d].get_drifter_path(drifter_id, 0, drifter_t)[-1][-1]
    last_positions = last_positions[~np.isnan(last_positions)].reshape(-1,2)

    # Axes
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    nx = ensemble_obs[0].nx
    ny = ensemble_obs[0].ny

    dx = ensemble_obs[0].domain_size_x/nx
    dy = ensemble_obs[0].domain_size_y/ny

    x = np.linspace(xmin+0.5*dx/1000, xmax-0.5*dx/1000, nx)
    y = np.linspace(ymin+0.5*dy/1000, ymax-0.5*dy/1000, ny)

    xx, yy = np.meshgrid(x,y)
    ccs = np.vstack([xx.ravel(), yy.ravel()])
    
    ## Kernel density estimation 
    clp = (last_positions-np.average(last_positions,axis=0)).T
    clp[0][clp[0] > (x[-1]/2)] = clp[0][clp[0] > (x[-1]/2)] - x[-1]
    clp[0][clp[0] < (-x[-1]/2)] = clp[0][clp[0] < (-x[-1]/2)] + x[-1]
    clp[1][clp[1] > (y[-1]/2)] = clp[1][clp[1] > (y[-1]/2)] - y[-1]
    clp[1][clp[1] < (-y[-1]/2)] = clp[1][clp[1] < (-y[-1]/2)] + y[-1]

    raw_cov = np.cov(clp)

    bw = numTrajectories**(-1./(2+4))

    cov = raw_cov * bw
    covinv = np.linalg.inv(cov)
        
    f = np.zeros((ny,nx))
    for e in range(numTrajectories):
        d = (ccs.T-last_positions[e]).T
        d[0][d[0] > (x[-1]/2)] = d[0][d[0] > (x[-1]/2)] - x[-1]
        d[0][d[0] < (-x[-1]/2)] = d[0][d[0] < (-x[-1]/2)] + x[-1]
        d[1][d[1] > (y[-1]/2)] = d[1][d[1] > (y[-1]/2)] - y[-1]
        d[1][d[1] < (-y[-1]/2)] = d[1][d[1] < (-y[-1]/2)] + y[-1]
        f += np.exp(-1/2*np.sum((d*np.dot(covinv,d)), axis=0)).reshape(ny,nx)
        
    ## Levels for plotting
    fmass = np.sum(f)

    fmax = np.max(f)

    levels = np.linspace(0,fmax,100)
    level_probs = np.zeros(100)
    for l in range(len(levels)):
        level_probs[l] = np.sum(f[f>levels[l]])/fmass
    level_probs[-1] = 0.0

    desired_probs = [0.9,0.75,0.5,0.25,0.0] #descending! (ending with 0.0)
    desired_levels = np.zeros_like(desired_probs)
    for p in range(len(desired_probs)):
        desired_levels[p] = levels[np.abs(level_probs-desired_probs[p]).argmin()]
    desired_levels = np.unique(desired_levels)

    # plotting levels and areas
    cfset = ax.contourf(xx, yy, f, levels=desired_levels, cmap=cmap, alpha=0.5)
    cset = ax.contour(xx, yy, f, levels=desired_levels, colors='k', alpha=0.25, linewidths=1)
    
    # Legend
    if add_legend:
        proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cfset.collections]
        labels = []
        for p in desired_probs:
            labels.append(str(int(p*100))+"%")
        ax.legend(proxy, labels, 
                #prop={'size': 18}, 
                labelcolor="black", 
                framealpha=0.9,
                loc=0
                )
    