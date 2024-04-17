# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2016 SINTEF ICT, 
Copyright (C) 2017-2024 SINTEF Digital
Copyright (C) 2017-2024 Norwegian Meteorological Institute

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
from matplotlib import animation

import numpy as np
import time
import re
import sys
import copy
import tqdm
from datetime import timedelta
from netCDF4 import Dataset


from gpuocean.utils import OceanographicUtilities

def plotState(eta, hu, hv, h, 
              dx, dy, t,
              fig=None,
              plot_title="",
              eta_max=None, 
              huv_max=None, 
              add_extra=False, 
              ax=None, sp=None):
    """
    Utility function for plotting the simplified ocean state.
    The default is to plot the conserved variables (eta, hu, hv), but the 
    flag 'add_extra' also enables plotting of particle velocity and 
    vorticity magnitude.
    """

    if fig is None and add_extra:
        fig = plt.figure(figsize=(12, 5))  
    elif fig is None:
        fig = plt.figure(figsize=(12, 3))  

    fig.suptitle("{:s} (time: {:0>8})".format(plot_title, str(timedelta(seconds=int(t)))), 
                 fontsize=18, x=0.13,
                 horizontalalignment='left')
    
    ny, nx = eta.shape
    domain_extent = [0, nx*dx/1000, 0, ny*dy/1000]
    
    x_plots = 3
    y_plots = 1
    if (add_extra == True):
        x_plots=3
        y_plots=2

    if not eta_max:
        eta_max = np.max(np.abs(eta))
    if not huv_max:
        huv_max = max(np.max(np.abs(hu)), np.max(np.abs(hv)))
    
    V_max = 6 * huv_max / np.max(h)
    R_min = -V_max/2000
    R_max = V_max/2000
    
    # Fix colormaps

    eta_cmap = copy.copy(plt.cm.BrBG)
    eta_cmap.set_bad(color='grey')
    huv_cmap = copy.copy(plt.cm.coolwarm)
    huv_cmap.set_bad(color='grey')
    vel_cmap = copy.copy(plt.cm.Oranges)
    vel_cmap.set_bad(color='grey')
    vor_cmap = copy.copy(plt.cm.seismic)
    vor_cmap.set_bad(color='grey')

    
    if (ax is None):
        ax = [None]*x_plots*y_plots
        sp = [None]*x_plots*y_plots

        ax[0] = plt.subplot(y_plots, x_plots, 1)
        sp[0] = ax[0].imshow(eta, interpolation="none", origin='lower', 
                             cmap=eta_cmap, 
                             vmin=-eta_max, vmax=eta_max, 
                             extent=domain_extent)
        plt.colorbar(sp[0], shrink=0.9)
        plt.axis('image')
        plt.title("eta")
        
        ax[1] = plt.subplot(y_plots, x_plots, 2)
        sp[1] = ax[1].imshow(hu, interpolation="none", origin='lower', 
                             cmap=huv_cmap, 
                             vmin=-huv_max, vmax=huv_max, 
                             extent=domain_extent)
        plt.colorbar(sp[1], shrink=0.9)
        plt.axis('image')
        plt.title("hu")

        ax[2] = plt.subplot(y_plots, x_plots, 3)
        sp[2] = ax[2].imshow(hv, interpolation="none", origin='lower', 
                             cmap=huv_cmap, 
                             vmin=-huv_max, vmax=huv_max, 
                             extent=domain_extent)
        plt.colorbar(sp[2], shrink=0.9)
        plt.axis('image')
        plt.title("hv")
        
        if (add_extra == True):
            V = genVelocity(h, hu, hv)
            ax[3] = plt.subplot(y_plots, x_plots, 4)
            sp[3] = ax[3].imshow(V, interpolation="none", origin='lower', 
                               cmap=vel_cmap, 
                               vmin=0, vmax=V_max, 
                               extent=domain_extent)
            plt.colorbar(sp[3], shrink=0.9)
            plt.axis('image')
            plt.title("Particle velocity magnitude")

            R = genColors(h, hu/dy, hv/dx, vor_cmap, R_min, R_max)
            ax[4] = plt.subplot(y_plots, x_plots, 5)
            sp[4] = ax[4].imshow(R, interpolation="none", 
                               origin='lower',
                               cmap=vor_cmap,
                               extent=domain_extent)
            plt.colorbar(sp[4], shrink=0.9)
            plt.axis('image')
            plt.title("Vorticity magnitude")
            
    else:        
        #Update plots
        fig.sca(ax[0])
        sp[0].set_data(eta)
        
        fig.sca(ax[1])
        sp[1].set_data(hu)
        
        fig.sca(ax[2])
        sp[2].set_data(hv)
        
        if (add_extra == True):
            V = genVelocity(h, hu, hv)
            fig.sca(ax[3])
            sp[3].set_data(V)

            R = genColors(h, hu/dx, hv/dy, vor_cmap, R_min, R_max)
            fig.sca(ax[4])
            sp[4].set_data(R)
    
    return ax, sp



def plotSim(sim, **kwargs):
    """
    Plot the state of the sim.

    For kwargs: See plotState.
    """

    eta, hu, hv = sim.download(interior_domain_only=True)
    _, Hm = sim.downloadBathymetry(interior_domain_only=True)
    h = Hm + eta
     
    return plotState(eta, hu, hv, h, 
                     sim.dx, sim.dy, sim.t,
                     **kwargs)



def simAnimation(sim, T_end, anim_dt, huv_max=100.0, eta_max=1, 
                 plot_title="", add_extra=False, fig=None, create_movie=True):
    """
    Creates an animation of the simulator with frames anim_dt apart.
    Number of frames are therefore T_end//anim_dt.
    """

    num_frames = T_end // anim_dt
    
    #Create figure 
    if fig is None and add_extra:
        fig = plt.figure(figsize=(12, 5))  
    elif fig is None:
        fig = plt.figure(figsize=(12, 3))  

    if not create_movie:
        for i in range(num_frames):
            sim.step(anim_dt)
            sim.updateDt()
        if T_end > sim.t:
            sim.step(T_end - sim.t)
        
    # Plot initial conditions (or end state if not create_movie)
    ax, sp = plotSim(sim, fig=fig, plot_title=plot_title,
                    eta_max=eta_max, huv_max=huv_max, add_extra=add_extra)
    
    if not create_movie:
        return 
    
    #Helper function which simulates and plots the solution
    def animate(i):
        if (i>0):
            sim.step(anim_dt)
            sim.updateDt()
            print("."+str(i)+"/"+str(num_frames)+".", end='')

        plotSim(sim, fig=fig, plot_title=plot_title, 
                ax=ax, sp=sp,
                add_extra=add_extra)
        
    #Matplotlib for creating an animation
    anim = animation.FuncAnimation(fig, animate, range(num_frames), interval=100)
    plt.close(fig)
    return anim



def ncAnimation(filename, movie_frames=None, create_movie=True, fig=None,
                add_extra=False, **kwargs):
    """
    Make animation of the netcdf file. 
    If movie_frames is not None, the plotted states are interpolated between the available timesteps
    """
    #Create figure and plot initial conditions
    ncfile = None
    if fig is None:
        fig = _plotHelperInitFig(add_extra)

    try:
        ncfile = Dataset(filename)
        x = ncfile.variables['x'][:]
        y = ncfile.variables['y'][:]
        t = ncfile.variables['time'][:]

        H_m = ncfile.variables['Hm'][:,:]
        eta = ncfile.variables['eta'][:,:,:]
        hu = ncfile.variables['hu'][:,:,:]
        hv = ncfile.variables['hv'][:,:,:]
    except Exception as e:
        raise e
    finally:
        if ncfile is not None:
            ncfile.close()


    if movie_frames is None:
        movie_frames = len(t)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    if (create_movie):
        ax, sp = plotState( 
                            eta[0],
                            hu[0],
                            hv[0],
                            H_m+eta[0],
                            dx, dy, t[0], 
                            fig=fig,
                            plot_title=filename,
                            add_extra=add_extra,
                            **kwargs)
    else:
        ax, sp = plotState(
                            eta[-1],
                            hu[-1],
                            hv[-1],
                            H_m+eta[-1],
                            dx, dy, t[0], 
                            fig=fig,
                            plot_title=filename,
                            add_extra=add_extra,
                           **kwargs)
        return

    #Helper function which simulates and plots the solution    
    def animate(i):
        t_now = t[0] + (i / (movie_frames-1)) * (t[-1] - t[0]) 

        k = np.searchsorted(t, t_now)
        if (k >= eta.shape[0]):
            k = eta.shape[0] - 1
        j = max(0, k-1)
        if (j == k):
            k += 1
        s = (t_now - t[j]) / (t[k] - t[j])

        plotState(
                (1-s)*eta[j] + s*eta[k], 
                (1-s)*hu[j]  + s*hu[k], 
                (1-s)*hv[j]  + s*hv[k], 
                H_m+(1-s)*eta[j] + s*eta[k], 
                dx, dy, t_now, 
                fig=fig, 
                plot_title=filename, 
                add_extra=add_extra,
                **kwargs, ax=ax, sp=sp)

        print("."+str(i)+"/"+str(movie_frames)+".", end='')


    #Matplotlib for creating an animation
    anim = animation.FuncAnimation(fig, animate, range(movie_frames), interval=100)
    plt.close(fig)
    
    return anim

def norkystAnimation(source_url, x0, x1, y0, y1, fig=None, tsteps=None, movie_frames=None, create_movie=True, add_extra=False, **kwargs):
    """
    Norkyst-specific animation function
    tsteps can be a list of indices to 
    """
    #Create figure and plot initial conditions
    if fig is None:
        fig = _plotHelperInitFig(add_extra)

    # Read num timesteps
    try:
        ncfile = Dataset(source_url)
        time_var = ncfile.variables['time']

        if tsteps is None:
            t = time_var[:]
            tsteps = list(range(len(t)))
        else:
            t = time_var[tsteps]
    except Exception as e:
        raise e
    finally:
        ncfile.close()
    t = t - t[0]

    if (not create_movie):
        tsteps = [tsteps[0]] + [tsteps[-1]]
        
    if movie_frames is None:
        movie_frames = len(tsteps)
        
    ncfile = None
    try:
        ncfile = Dataset(source_url)
        H_m = ncfile.variables['h'][y0:y1, x0:x1]
        eta = ncfile.variables['zeta'][tsteps, y0:y1, x0:x1]
        hu = ncfile.variables['ubar'][tsteps, y0:y1, x0:x1]
        hv = ncfile.variables['vbar'][tsteps, y0:y1, x0:x1]
        
        for timestep in range(len(tsteps)):
            hu[timestep] = hu[timestep] * (H_m + eta[timestep])
            hv[timestep] = hv[timestep] * (H_m + eta[timestep])

        x = ncfile.variables['X'][x0:x1]
        y = ncfile.variables['Y'][y0:y1]
        dx = np.average(x[1:] - x[:-1])
        dy = np.average(y[1:] - y[:-1])

    except Exception as e:
        raise e
    finally:
        if ncfile is not None:
            ncfile.close()

    if (create_movie):
        ax, sp = plotState( eta[0],
                            hu[0],
                            hv[0],
                            H_m+eta[0],
                            dx, dy, t[0], 
                            fig=fig, plot_title="Reference solution",
                            add_extra=add_extra,
                            **kwargs)
    else:
        ax, sp = plotState(eta[-1],
                            hu[-1],
                            hv[-1],
                            H_m+eta[-1],
                            dx, dy, t[-1],
                            fig=fig, plot_title="Reference solution",
                            add_extra=add_extra,
                            **kwargs)
        return
        
    #Helper function which simulates and plots the solution    
    def animate(i):
        t_now = t[0] + (i / (movie_frames-1)) * (t[-1] - t[0]) 
        
        k = np.searchsorted(t, t_now)
        if (k >= eta.shape[0]):
            k = eta.shape[0] - 1
        j = max(0, k-1)
        if (j == k):
            k += 1
        s = (t_now - t[j]) / (t[k] - t[j])
        
        plotState((1-s)*eta[j] + s*eta[k], 
                    (1-s)*hu[j]  + s*hu[k], 
                    (1-s)*hv[j]  + s*hv[k], 
                    H_m+(1-s)*eta[j] + s*eta[k], 
                    dx, dy, t_now,
                    fig=fig, plot_title="Reference solution",
                    add_extra=add_extra,
                    **kwargs, ax=ax, sp=sp)

    #Matplotlib for creating an animation
    anim = animation.FuncAnimation(fig, animate, range(movie_frames), interval=100)
    plt.close(fig)
    return anim




def _plotHelperInitFig(add_extra):
    #Create figure 
    if add_extra:
        return plt.figure(figsize=(12, 5))  
    else:
        return plt.figure(figsize=(12, 3))  
    


def imshow3(eta, hu, hv, 
            eta_max=None, huv_max=None, figsize=(12, 3), 
            titles=["eta", "hu", "hv"], suptitle=None,
            cmaps=[plt.cm.BrBG, plt.cm.coolwarm, plt.cm.coolwarm],
            interpolation="none"):
   
    fig = plt.figure(figsize=figsize)

    if not eta_max:
        eta_max = np.max(np.abs(eta))
    if not huv_max:
        huv_max = max(np.max(np.abs(hv)), np.max(np.abs(hv)))

    maxvals = [eta_max, huv_max, huv_max]
    data = [eta, hu, hv]

    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        sp = ax.imshow(data[i], interpolation=interpolation, origin='lower', 
                                cmap=cmaps[i], vmin=-maxvals[i], vmax=maxvals[i])
        plt.colorbar(sp, shrink=0.9)
        plt.axis('image')
        plt.title(titles[i])

    if suptitle:
        plt.suptitle(suptitle)
            


##################################################
# Functions from before 2020. 
# Might be useful again for certain simulations...

"""
Class that makes plotting faster by caching the plots instead of recreating them
"""
class PlotHelper:

    def __init__(self, fig, x_coords, y_coords, radius, eta1, u1, v1, eta2=None, u2=None, v2=None, interpolation_type='spline36', plotRadial=False):
        self.ny, self.nx = eta1.shape
        self.fig = fig
        self.plotRadial = plotRadial
        
        if self.plotRadial:
            fig.set_figheight(15)
            fig.set_figwidth(15)
        else:
            fig.set_figheight(6)
            fig.set_figwidth(14)

            
        min_x = np.min(x_coords[:,0]);
        min_y = np.min(y_coords[0,:]);
        
        max_x = np.max(x_coords[0,:]);
        max_y = np.max(y_coords[:,0]);
        
        domain_extent = [ x_coords[0, 0], x_coords[0, -1], y_coords[0, 0], y_coords[-1, 0] ]
        
        if not self.plotRadial:
            self.gs = gridspec.GridSpec(1, 3)  
        elif (eta2 is None):
            self.gs = gridspec.GridSpec(2, 3)
        else:
            assert(u2 is not None)
            assert(v2 is not None)
            self.gs = gridspec.GridSpec(3, 3)
        
        ax = self.fig.add_subplot(self.gs[0, 0])
        self.sp_eta = plt.imshow(eta1, interpolation=interpolation_type, origin='lower', vmin=-0.05, vmax=0.05, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('Eta')
        plt.colorbar()
        self.drifters = plt.scatter(x=None, y=None, color='blue')
        self.observations = plt.scatter(x=None, y=None, color='red')
        self.driftersMean = plt.scatter(x=None, y=None, color='red', marker='+')
        
        ax = self.fig.add_subplot(self.gs[0, 1])
        self.sp_u = plt.imshow(u1, interpolation=interpolation_type, origin='lower', vmin=-1.5, vmax=1.5, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('U')
        plt.colorbar()
        
        ax = self.fig.add_subplot(self.gs[0, 2])
        self.sp_v = plt.imshow(v1, interpolation=interpolation_type, origin='lower', vmin=-1.5, vmax=1.5, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('V')
        plt.colorbar()
        
        if self.plotRadial:
            ax = self.fig.add_subplot(self.gs[1, 0])
            self.sp_radial1, = plt.plot(radius.ravel(), eta1.ravel(), '.')
            plt.axis([0, min(max_x, max_y), -1.5, 1])
            plt.title('Eta Radial plot')

            ax = self.fig.add_subplot(self.gs[1, 1])
            self.sp_x_axis1, = plt.plot(x_coords[self.ny/2,:], eta1[self.ny/2,:], 'k+--', label='x-axis')
            self.sp_y_axis1, = plt.plot(y_coords[:,self.nx/2], eta1[:,self.nx/2], 'kx:', label='y-axis')
            plt.axis([max(min_x, min_y), min(max_x, max_y), -1.5, 1])
            plt.title('Eta along axis')
            plt.legend()

            ax = self.fig.add_subplot(self.gs[1, 2])
            self.sp_x_diag1, = plt.plot(1.41*np.diagonal(x_coords, offset=-abs(self.nx-self.ny)/2), \
                                       np.diagonal(eta1, offset=-abs(self.nx-self.ny)/2), \
                                       'k+--', label='x = -y')
            self.sp_y_diag1, = plt.plot(1.41*np.diagonal(y_coords.T, offset=abs(self.nx-self.ny)/2), \
                                       np.diagonal(eta1.T, offset=abs(self.nx-self.ny)/2), \
                                       'kx:', label='x = y')
            plt.axis([max(min_x, min_y), min(max_x, max_y), -1.5, 1])
            plt.title('Eta along diagonal')
            plt.legend()
        
            if eta2 is not None:
                ax = self.fig.add_subplot(self.gs[2, 0])
                self.sp_radial2, = plt.plot(radius.ravel(), eta2.ravel(), '.')
                plt.axis([0, min(max_x, max_y), -1.5, 1])
                plt.title('Eta2 Radial plot')

                ax = self.fig.add_subplot(self.gs[2, 1])
                self.sp_x_axis2, = plt.plot(x_coords[self.ny/2,:], eta2[self.ny/2,:], 'k+--', label='x-axis')
                self.sp_y_axis2, = plt.plot(y_coords[:,self.nx/2], eta2[:,self.nx/2], 'kx:', label='y-axis')
                plt.axis([max(min_x, min_y), min(max_x, max_y), -1.5, 1])
                plt.title('Eta2 along axis')
                plt.legend()

                ax = self.fig.add_subplot(self.gs[2, 2])
                self.sp_x_diag2, = plt.plot(1.41*np.diagonal(x_coords, offset=-abs(self.nx-self.ny)/2), \
                                           np.diagonal(eta2, offset=-abs(self.nx-self.ny)/2), \
                                           'k+--', label='x = -y')
                self.sp_y_diag2, = plt.plot(1.41*np.diagonal(y_coords.T, offset=abs(self.nx-self.ny)/2), \
                                           np.diagonal(eta2.T, offset=abs(self.nx-self.ny)/2), \
                                           'kx:', label='x = y')
                plt.axis([max(min_x, min_y), min(max_x, max_y), -1.5, 1])
                plt.title('Eta2 along diagonal')
                plt.legend()
        
        
   

    @classmethod
    def fromsim(cls, sim, fig):
        x_center = sim.dx*(sim.nx)/2.0
        y_center = sim.dy*(sim.ny)/2.0
        y_coords, x_coords = np.mgrid[0:(sim.ny+20)*sim.dy:sim.dy, 0:(sim.nx+20)*sim.dx:sim.dx]
        x_coords = np.subtract(x_coords, x_center)
        y_coords = np.subtract(y_coords, y_center)
        radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))
        eta, hu, hv = sim.download(interior_domain_only=True)
        return cls( fig, x_coords, y_coords, radius, eta, hu, hv)
        
        
    def plot(self, eta1, u1, v1, eta2=None, u2=None, v2=None):
        self.fig.add_subplot(self.gs[0, 0])
        self.sp_eta.set_data(eta1)

        self.fig.add_subplot(self.gs[0, 1])
        self.sp_u.set_data(u1)

        self.fig.add_subplot(self.gs[0, 2])
        self.sp_v.set_data(v1)
            
        if self.plotRadial: 
            self.fig.add_subplot(self.gs[1, 0])
            self.sp_radial1.set_ydata(eta1.ravel());

            self.fig.add_subplot(self.gs[1, 1])
            self.sp_x_axis1.set_ydata(eta1[(self.ny+2)/2,:])
            self.sp_y_axis1.set_ydata(eta1[:,(self.nx+2)/2])

            self.fig.add_subplot(self.gs[1, 2])
            self.sp_x_diag1.set_ydata(np.diagonal(eta1, offset=-abs(self.nx-self.ny)/2))
            self.sp_y_diag1.set_ydata(np.diagonal(eta1.T, offset=abs(self.nx-self.ny)/2))

            if (eta2 is not None):
                self.fig.add_subplot(self.gs[2, 0])
                self.sp_radial2.set_ydata(eta2.ravel());

                self.fig.add_subplot(self.gs[2, 1])
                self.sp_x_axis2.set_ydata(eta2[(self.ny+2)/2,:])
                self.sp_y_axis2.set_ydata(eta2[:,(self.nx+2)/2])

                self.fig.add_subplot(self.gs[2, 2])
                self.sp_x_diag2.set_ydata(np.diagonal(eta2, offset=-abs(self.nx-self.ny)/2))
                self.sp_y_diag2.set_ydata(np.diagonal(eta2.T, offset=abs(self.nx-self.ny)/2))
        
        plt.draw()
        time.sleep(0.001)
        
    
    def showDrifters(self, drifters, showObservation=True, showMean=True):
        self.drifters.set_offsets(drifters.getDrifterPositions())
        if showMean:
            self.driftersMean.set_offsets(drifters.getCollectionMean())
        if showObservation:
            self.observations.set_offsets(drifters.getObservationPosition())
        plt.draw()
        
        
"""
For easily creating a plot of values on a 2D domain
"""        
class SinglePlot:
    
    def __init__(self, fig, x_coords, y_coords, data, interpolation_type='spline36', title='Data'):
        self.ny, self.nx = data.shape
        self.fig = fig;
        
        fig.set_figheight(5)
        fig.set_figwidth(5)
        
        min_x = np.min(x_coords[:,0]);
        min_y = np.min(y_coords[0,:]);
        
        max_x = np.max(x_coords[0,:]);
        max_y = np.max(y_coords[:,0]);
        
        domain_extent = [ x_coords[0, 0], x_coords[0, -1], y_coords[0, 0], y_coords[-1, 0] ]
        
        self.gs = gridspec.GridSpec(1,1)
        
        maxValue = np.max(data)
        minValue = np.min(data)
        
        ax = self.fig.add_subplot(self.gs[0, 0])
        self.sp_eta = plt.imshow(data, interpolation=interpolation_type, origin='lower', vmin=minValue, vmax=maxValue, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title(title)
        plt.colorbar()
        

        
        
class EnsembleAnimator:
    """
    For easily making animation of ensemble simulations.
    The ensemble is expected to be a OceanStateEnsemble-type object.
    """
    
    def __init__(self, fig, ensemble,  interpolation_type='spline36', \
                 eta_abs_lim=0.05, volume_transport_abs_lim=1.5, \
                 trueStateOnly=False):
        self.ny, self.nx = ensemble.ny, ensemble.nx
        self.domain_size_x = ensemble.getDomainSizeX()
        self.domain_size_y = ensemble.getDomainSizeY()
        self.fig = fig;
        
        self.trueStateOnly = trueStateOnly
        
        if self.trueStateOnly:
            fig.set_figheight(4)
        else:
            fig.set_figheight(16)
        fig.set_figwidth(12)
        
        # Obtain the following fields:
        eta_true, hu_true, hv_true = ensemble.downloadTrueOceanState()
        if not self.trueStateOnly:
            eta_mean, hu_mean, hv_mean, eta_rmse, hu_rmse, hv_rmse, eta_r, hu_r, hv_r = ensemble.downloadEnsembleStatisticalFields()
        
        r_deviation = 0.2
        r_min, r_max = 1.0-r_deviation, 1.0+r_deviation
        
        domain_extent = [ 0.0, self.domain_size_x, 0.0, self.domain_size_y ]
        
        self.gs = None
        if self.trueStateOnly:
            self.gs = gridspec.GridSpec(1,3)
        else:
            self.gs = gridspec.GridSpec(4, 3)
        
        ## TRUE STATE
        ax = self.fig.add_subplot(self.gs[0, 0])
        self.true_eta = plt.imshow(eta_true, interpolation=interpolation_type, origin='lower', vmin=-eta_abs_lim, vmax=eta_abs_lim, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('True eta')
        plt.colorbar()
        self.true_drifters = plt.scatter(x=None, y=None, color='blue')
        self.true_observations = plt.scatter(x=None, y=None, color='red')
        self.true_driftersMean = plt.scatter(x=None, y=None, color='red', marker='+')

        ax = self.fig.add_subplot(self.gs[0, 1])
        self.true_hu = plt.imshow(hu_true, interpolation=interpolation_type, origin='lower', vmin=-volume_transport_abs_lim, vmax=volume_transport_abs_lim, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('True hu')
        plt.colorbar()

        ax = self.fig.add_subplot(self.gs[0, 2])
        self.true_hv = plt.imshow(hv_true, interpolation=interpolation_type, origin='lower', vmin=-volume_transport_abs_lim, vmax=volume_transport_abs_lim, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('True hv')
        plt.colorbar()

        if not self.trueStateOnly:
            # ENSEMBLE MEANS 
            ax = self.fig.add_subplot(self.gs[1, 0])
            self.mean_eta = plt.imshow(eta_mean, interpolation=interpolation_type, origin='lower', vmin=-eta_abs_lim, vmax=eta_abs_lim, extent=domain_extent)
            plt.axis('tight')
            ax.set_aspect('equal')
            plt.title('Ensemble mean eta')
            plt.colorbar()
            self.mean_drifters = plt.scatter(x=None, y=None, color='blue')
            self.mean_observations = plt.scatter(x=None, y=None, color='red')
            self.mean_driftersMean = plt.scatter(x=None, y=None, color='red', marker='+')

            ax = self.fig.add_subplot(self.gs[1, 1])
            self.mean_hu = plt.imshow(hu_mean, interpolation=interpolation_type, origin='lower', vmin=-volume_transport_abs_lim, vmax=volume_transport_abs_lim, extent=domain_extent)
            plt.axis('tight')
            ax.set_aspect('equal')
            plt.title('ensemble mean hu')
            plt.colorbar()

            ax = self.fig.add_subplot(self.gs[1, 2])
            self.mean_hv = plt.imshow(hv_mean, interpolation=interpolation_type, origin='lower', vmin=-volume_transport_abs_lim, vmax=volume_transport_abs_lim, extent=domain_extent)
            plt.axis('tight')
            ax.set_aspect('equal')
            plt.title('ensemble mean hv')
            plt.colorbar()
            
            
            ## ROOT MEAN-SQUARE ERROR
            ax = self.fig.add_subplot(self.gs[2, 0])
            self.rmse_eta = plt.imshow(eta_rmse, interpolation=interpolation_type, origin='lower', vmin=-eta_abs_lim, vmax=eta_abs_lim, extent=domain_extent)
            plt.axis('tight')
            ax.set_aspect('equal')
            plt.title('RMSE eta')
            plt.colorbar()
            
            self.rmse_drifters = plt.scatter(x=None, y=None, color='blue')
            self.rmse_observations = plt.scatter(x=None, y=None, color='red')
            self.rmse_driftersMean = plt.scatter(x=None, y=None, color='red', marker='+')

            ax = self.fig.add_subplot(self.gs[2, 1])
            self.rmse_hu = plt.imshow(hu_rmse, interpolation=interpolation_type, origin='lower', vmin=-volume_transport_abs_lim, vmax=volume_transport_abs_lim, extent=domain_extent)
            plt.axis('tight')
            ax.set_aspect('equal')
            plt.title('RMSE hu')
            plt.colorbar()

            ax = self.fig.add_subplot(self.gs[2, 2])
            self.rmse_hv = plt.imshow(hv_rmse, interpolation=interpolation_type, origin='lower', vmin=-volume_transport_abs_lim, vmax=volume_transport_abs_lim, extent=domain_extent)
            plt.axis('tight')
            ax.set_aspect('equal')
            plt.title('RMSE hv')
            plt.colorbar()
            
            ## r = sigma / RMSE
            ax = self.fig.add_subplot(self.gs[3, 0])
            self.r_eta = plt.imshow(eta_r, interpolation=interpolation_type, origin='lower', vmin=r_min, vmax=r_max, extent=domain_extent)
            plt.axis('tight')
            ax.set_aspect('equal')
            plt.title('r = sigma/RMSE (eta)')
            plt.colorbar()
            
            self.r_drifters = plt.scatter(x=None, y=None, color='blue')
            self.r_observations = plt.scatter(x=None, y=None, color='red')
            self.r_driftersMean = plt.scatter(x=None, y=None, color='red', marker='+')

            ax = self.fig.add_subplot(self.gs[3, 1])
            self.r_hu = plt.imshow(hu_r, interpolation=interpolation_type, origin='lower', vmin=r_min, vmax=r_max, extent=domain_extent)
            plt.axis('tight')
            ax.set_aspect('equal')
            plt.title('r = sigma/RMSE (hu)')
            plt.colorbar()

            ax = self.fig.add_subplot(self.gs[3, 2])
            self.r_hv = plt.imshow(hv_r, interpolation=interpolation_type, origin='lower', vmin=r_min, vmax=r_max, extent=domain_extent)
            plt.axis('tight')
            ax.set_aspect('equal')
            plt.title('r = sigma/RMSE (hv)')
            plt.colorbar()

    
    def plot(self, ensemble):
        # Obtain the following fields:
            
        eta_true, hu_true, hv_true = ensemble.downloadTrueOceanState()
        
        if not self.trueStateOnly:
            eta_mean, hu_mean, hv_mean, eta_rmse, hu_rmse, hv_rmse, eta_r, hu_r, hv_r = ensemble.downloadEnsembleStatisticalFields()
        
        # TRUE STATE
        self.fig.add_subplot(self.gs[0, 0])
        self.true_eta.set_data(eta_true)

        self.fig.add_subplot(self.gs[0, 1])
        self.true_hu.set_data(hu_true)

        self.fig.add_subplot(self.gs[0, 2])
        self.true_hv.set_data(hv_true)
            
        if not self.trueStateOnly:
            # ENSEMBLE MEAN
            self.fig.add_subplot(self.gs[1, 0])
            self.mean_eta.set_data(eta_mean)

            self.fig.add_subplot(self.gs[1, 1])
            self.mean_hu.set_data(hu_mean)

            self.fig.add_subplot(self.gs[1, 2])
            self.mean_hv.set_data(hv_mean)
            
            # ROOT MEAN-SQUARE ERROR
            self.fig.add_subplot(self.gs[2, 0])
            self.rmse_eta.set_data(eta_rmse)

            self.fig.add_subplot(self.gs[2, 1])
            self.rmse_hu.set_data(hu_rmse)

            self.fig.add_subplot(self.gs[2, 2])
            self.rmse_hv.set_data(hv_rmse)

            # ROOT MEAN-SQUARE ERROR
            self.fig.add_subplot(self.gs[3, 0])
            self.r_eta.set_data(eta_r)

            self.fig.add_subplot(self.gs[3, 1])
            self.r_hu.set_data(hu_r)

            self.fig.add_subplot(self.gs[3, 2])
            self.r_hv.set_data(hv_r)
        
        # Drifters
        drifterPositions = ensemble.observeDrifters()
        trueDrifterPosition = ensemble.observeTrueDrifters()
        
        # TODO
        # These lines which updates the drifter positions to the animations
        # broke when updating from Python 2 to Python 3.
        # This should be fixed again... 
        
        #self.true_drifters.set_offsets(drifterPositions)
        #self.true_observations.set_offsets(trueDrifterPosition)
        
        if not self.trueStateOnly:
                       
            #self.mean_drifters.set_offsets(drifterPositions)
            #self.mean_observations.set_offsets(trueDrifterPosition)
            
            #self.rmse_drifters.set_offsets(drifterPositions)
            #self.rmse_observations.set_offsets(trueDrifterPosition)
            
            #self.r_drifters.set_offsets(drifterPositions)
            #self.r_observations.set_offsets(trueDrifterPosition)
            pass
        
        
        plt.draw()
        time.sleep(0.001)
        
        
        
        


##################################################3
# DRIFTERS ON CANVAS


def genVelocity(rho, rho_u, rho_v):
    mask = None
    if (np.ma.is_masked(rho)):
        mask = rho.mask
        rho = rho.filled(0.0)
    if (np.ma.is_masked(rho_u)):
        rho_u = rho_u.filled(0.0)
    if (np.ma.is_masked(rho_v)):
        rho_v = rho_v.filled(0.0)
    u = OceanographicUtilities.desingularise(rho, rho_u, 0.00001)
    v = OceanographicUtilities.desingularise(rho, rho_v, 0.00001)
    u = np.sqrt(u**2 + v**2)

    if (mask is not None):
        u = np.ma.array(u, mask=mask)

    return u
    

def genSchlieren(rho):
    #Compute length of z-component of normalized gradient vector 
    normal = np.gradient(rho) #[x, y, 1]
    length = 1.0 / np.sqrt(normal[0]**2 + normal[1]**2 + 1.0)
    schlieren = np.power(length, 128)
    return schlieren


def genVorticity(rho, rho_u, rho_v):
    mask = None
    if (np.ma.is_masked(rho)):
        mask = rho.mask
        rho = rho.filled(0.0)
    if (np.ma.is_masked(rho_u)):
        rho_u = rho_u.filled(0.0)
    if (np.ma.is_masked(rho_v)):
        rho_v = rho_v.filled(0.0)
    u = OceanographicUtilities.desingularise(rho, rho_u, 0.00001)
    v = OceanographicUtilities.desingularise(rho, rho_v, 0.00001)
    u_max = u.max()
    
    du_dy, _ = np.gradient(u)
    _, dv_dx = np.gradient(v)
    
    #Length of curl
    curl = dv_dx - du_dy
    if (mask is not None):
        curl = np.ma.array(curl, mask=mask)
    return curl


def genColors(rho, rho_u, rho_v, cmap, vmin, vmax, use_schlieren=False):
    curl = genVorticity(rho, rho_u, rho_v)

    colors = Normalize(vmin, vmax, clip=True)(curl)
    colors = cmap(colors)
    
    if (use_schlieren):
        schlieren = genSchlieren(rho)
        for k in range(3):
            colors[:,:,k] = colors[:,:,k]*schlieren

    return colors

    


def tex_escape(text):
    """Escape text for LaTeX processing of figures"""
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)
