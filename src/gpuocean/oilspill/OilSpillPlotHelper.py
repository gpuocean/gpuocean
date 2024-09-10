

from matplotlib import pyplot as plt
import numpy as np

import copy
import os

from gpuocean.utils import DrifterPlotHelper
from gpuocean.drifters import LagrangianUtils

def addConcentrationToBackground(ax, sim):
    cons = LagrangianUtils.concentrationFromSim(sim)
    cons[cons == 0] = np.NaN

    oil_cmap = copy.copy(plt.cm.ocean_r)
    oil_cmap.set_bad(alpha=0)

    oil_im = ax.imshow(cons, origin='lower', 
                      extent=[0, sim.nx*sim.dx/1000, 0, sim.ny*sim.dy/1000], 
                      cmap=oil_cmap)
    plt.colorbar(oil_im, ax=ax, 
                 orientation='horizontal', 
                 label="oil concentration [$g/m^2$]",
                 shrink=0.7, pad=0.1)
    # cbar_oil = fig.colorbar(oil_im, ax=ax, orientation='horizontal', label="oil concentration [$g/m^2$]")
    # cbar_oil.ax.set_position([0.0, 0.25, 1, 0.03])  # [left, bottom, width, height]

def getVerticalHistogram(drifters, bin_size=0.1, max_depth=None):
    zpos = drifters.getDrifterPositions()[:, 2]
    submerged = zpos[zpos < 0]
    # print(submerged)
    num_surfaced = len(zpos[zpos == 0.0])
    # print("num_surfaced: ", num_surfaced)
    # print("num submerged: ", len(submerged))
    # print("min/max zpos: ", np.min(zpos), np.max(zpos))
    # print(zpos.shape)
    if max_depth is None:
        max_depth = 2
    max_depth = max(max_depth, -np.min(zpos))
    num_bins = int(np.ceil(max_depth / bin_size))
    count, bins = np.histogram(submerged, bins=num_bins)
    fraction_submerged = len(submerged)/len(zpos)
    return count, bins, fraction_submerged
    
# count, bins = getVerticalHistogram(drifters)

def addConcentrationToBackground2(ax, sim, cb_shrink=0.7, cb_pad=0.2, clim=None):
    cons = LagrangianUtils.concentrationFromSim(sim)
    cons[cons == 0] = np.NaN

    oil_cmap = copy.copy(plt.cm.ocean_r)
    #oil_cmap = copy.copy(plt.cm.gist_stern_r)
    #oil_cmap = copy.copy(plt.cm.cubehelix_r)
    # oil_cmap = copy.copy(plt.cm.PuBuGn)
    # oil_cmap = copy.copy(plt.cm.bone_r)
    # oil_cmap.set_bad(alpha=0)

    extent = [0, sim.nx*sim.dx/1000, 0, sim.ny*sim.dy/1000]
    oil_im = ax.imshow(cons, origin='lower', 
                      extent=extent, 
                      cmap=oil_cmap, clim=clim)
    # ax.set_xlabel("[km]")
    ax.set_ylabel("[km]")

    plt.colorbar(oil_im, ax=ax, 
                 orientation='horizontal', 
                 label="oil concentration [particles$/m^2$]",
                 shrink=cb_shrink, pad=cb_pad)
                 #location='bottom')


def plotOilWithVerticalDistribution(sim, domain=[0, None, 0, None],
                                    do_save=False, dirname="oilspill", counter=0,
                                    save_filename=None,
                                    oilspill_clim=None, max_depth=None, vmax=None,
                                    submerged_particles_max=30000,
                                    include_stranding=None, classifier=None,
                                    stranding_start=0, midpoint_gs=7,
                                    close_plot=False):
    
    vertical_plots = 1
    if include_stranding:
        if np.sum(sim.drifters.is_stranded()) > 0:
            vertical_plots = 2
        else:
            include_stranding = False

        if classifier is None:
            classifier = lambda pos : np.zeros(pos.shape[0]), [None]

    fig = plt.figure(figsize=(12,5))
    gs = fig.add_gridspec(vertical_plots,10)
    ax1 = fig.add_subplot(gs[:, 0:midpoint_gs])
    DrifterPlotHelper.background_from_sim(sim, ax=ax1, drifter_domain=domain, vmax=vmax), #, background_type="eta")

    # # DrifterPlotHelper.add_drifter_positions_on_background(ax, drifters.getDrifterPositions(), s=0.05)
    addConcentrationToBackground2( ax1, sim, cb_pad=0.1, cb_shrink=0.6, clim=oilspill_clim)
    # ax1.set_title("After t = " +str(int(sim.t)/3600)+" h")
    ax1.set_title("Oil spill after {0:04.1f} hours".format(int(sim.t)/3600))

    ax2 = fig.add_subplot(gs[0, midpoint_gs:])
    plotVerticalHistogram(sim, ax=ax2, max_depth=max_depth, 
                          submerged_particles_max=submerged_particles_max)
    
    if include_stranding:
        
        ax3 = fig.add_subplot(gs[1, midpoint_gs:])
        stranded_pos, stranded_times = sim.drifters.get_stranded_particles()
        classification, hist_labels = classifier(stranded_pos)
        unique_classifications = np.unique(classification)
        print(unique_classifications)
        stranded_cmap = ["red", "blue", "forestgreen"]      

        stranded_times = stranded_times/3600  
        first_time = np.min(stranded_times)
        last_time = np.max(stranded_times)
        bins = np.linspace(first_time, last_time, 30)
        
        for key in unique_classifications:
            key = int(key)
            ax1.scatter(stranded_pos[classification==key, 0]/1000.0,
                        stranded_pos[classification==key, 1]/1000.0,
                        c=stranded_cmap[key], s=0.5, alpha=1)

            ax3.hist(stranded_times[classification==key], bins=bins, density=False, 
                     facecolor=stranded_cmap[key], alpha=0.8,
                     label=[hist_labels[key]])
        if hist_labels[0] is not None:
            ax3.legend()
        ax3.set_xlim([stranding_start, 24.5])
        frac_stranded = len(stranded_times)/sim.drifters.num_active_drifters
        ax3.set_title('Stranding times ('+str(int(frac_stranded*1000)/10)+"%)")
        ax3.set_ylabel("num particles")
        ax3.set_xlabel("time (hours)")
        ax3.set_yscale("log")

    plt.tight_layout()

    if do_save:
        if close_plot:
            plt.close()
        if save_filename is None:
            save_filename = "oilspill_"+str(counter).zfill(4)
        fig.savefig(os.path.join(dirname, save_filename+".png"))
        fig.savefig(os.path.join(dirname, save_filename+".pdf"))
        

def plotVerticalHistogram(sim, ax=None, max_depth=None, submerged_particles_max=30000, figsize=(4,6)):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)

    counts, bins, frac_sub = getVerticalHistogram(sim.drifters, max_depth=max_depth)
    plt.hist(bins[:-1], bins, weights=counts, orientation='horizontal')
    ax.set_ylabel('depth (m)')

    if max_depth is not None:
        ax.set_ylim(-max_depth, 1)
    ax.set_xlabel('num particles')
    ax.set_xscale('log')
    ax.set_xlim(0.7, submerged_particles_max)
    ax.set_title('Depth of submerged ('+str(int(frac_sub*1000)/10)+"%)")

def plotStranding(sim, domain=[0, None, 0, None], 
                  do_save=False, dirname="oilspill", 
                  counter=0, oilspill_clim=None,
                  clasifier=None):
    
    fig = plt.figure(figsize=(12,5))
    gs = fig.add_gridspec(1,10)
    ax1 = fig.add_subplot(gs[0, 0:7])
    DrifterPlotHelper.background_from_sim(sim, ax=ax1, drifter_domain=domain, vmax=vmax), #, background_type="eta")

    # # DrifterPlotHelper.add_drifter_positions_on_background(ax, drifters.getDrifterPositions(), s=0.05)
    addConcentrationToBackground2( ax1, sim, cb_pad=0.1, cb_shrink=0.6, clim=oilspill_clim)
    # ax1.set_title("After t = " +str(int(sim.t)/3600)+" h")
    ax1.set_title("Oil spill after {0:04.1f} hours".format(int(sim.t)/3600))

