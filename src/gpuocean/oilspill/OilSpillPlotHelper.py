

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
    oil_cmap.set_bad(alpha=0)

    extent = [0, sim.nx*sim.dx/1000, 0, sim.ny*sim.dy/1000]
    oil_im = ax.imshow(cons, origin='lower', 
                      extent=extent, 
                      cmap=oil_cmap, clim=clim)
    
    plt.colorbar(oil_im, ax=ax, 
                 orientation='horizontal', 
                 label="oil concentration [$g/m^2$]",
                 shrink=cb_shrink, pad=cb_pad)
                 #location='bottom')


def plotOilWithVerticalDistribution(sim, domain=[0, None, 0, None], do_save=False, dirname="oilspill", counter=0,
                                    oilspill_clim=None, max_depth=None, vmax=None,
                                    submerged_particles_max=30000):
    fig = plt.figure(figsize=(12,5))
    gs = fig.add_gridspec(1,10)
    ax1 = fig.add_subplot(gs[0, 0:7])
    DrifterPlotHelper.background_from_sim(sim, ax=ax1, drifter_domain=domain, vmax=vmax), #, background_type="eta")

    # # DrifterPlotHelper.add_drifter_positions_on_background(ax, drifters.getDrifterPositions(), s=0.05)
    addConcentrationToBackground2( ax1, sim, cb_pad=0.1, cb_shrink=0.6, clim=oilspill_clim)
    # ax1.set_title("After t = " +str(int(sim.t)/3600)+" h")
    ax1.set_title("Oil spill after {0:04.1f} hours".format(int(sim.t)/3600))

    ax2 = fig.add_subplot(gs[0, 7:])
    # ax2.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
    counts, bins, frac_sub = getVerticalHistogram(sim.drifters, max_depth=max_depth)
    plt.hist(bins[:-1], bins, weights=counts, orientation='horizontal')
    ax2.set_ylabel('depth (m)')

    if max_depth is not None:
        ax2.set_ylim(-max_depth, 1)
    ax2.set_xlabel('num particles')
    ax2.set_xscale('log')
    ax2.set_xlim(0, submerged_particles_max)
    ax2.set_title('Depth of submerged ('+str(int(frac_sub*1000)/10)+"%)")
    if do_save:
        plt.close()
        fig.savefig(os.path.join(dirname, "oilspill_"+str(counter).zfill(4)+".png"))