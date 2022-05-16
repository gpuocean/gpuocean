# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2019 SINTEF Digital

This module contains code for plotting various properties related to
an ensemble.

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

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time

from SWESimulators import Common
from SWESimulators import DataAssimilationUtils as dautils

def _updateMinMax(eta, hu, hv, fieldRanges):
    """
    Internal utility function for updating range of the fields eta, hu and hv.
    """
    fieldRanges[0] = min(fieldRanges[0], np.min(eta))
    fieldRanges[1] = max(fieldRanges[1], np.max(eta))
    fieldRanges[2] = min(fieldRanges[2], np.min(hu ))
    fieldRanges[3] = max(fieldRanges[3], np.max(hu ))
    fieldRanges[4] = min(fieldRanges[4], np.min(hv ))
    fieldRanges[5] = max(fieldRanges[5], np.max(hv ))

def _markDriftersInImshow(ax, ensemble, observed_drifter_positions):
    """
    Internal utility function for marking drifter positions in a imshow
    """
    for d in range(ensemble.getNumDrifters()):
        cell_id_x = int(np.floor(observed_drifter_positions[d,0]/ensemble.getDx()))
        cell_id_y = int(np.floor(observed_drifter_positions[d,1]/ensemble.getDy()))
        circ = matplotlib.patches.Circle((cell_id_x, cell_id_y), 1, fill=False)
        ax.add_patch(circ)    

def plotEnsemble(ensemble, num_particles=5, plotVelocityField=True):
    """
    Utility function to plot:
        - the true state
        - the ensemble mean
        - the state of up to 5 first ensemble members
    """
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    observed_drifter_positions = ensemble.observeTrueDrifters()

    numParticlePlots = min(ensemble.getNumParticles(), num_particles)
    numPlots = numParticlePlots + 3
    plotCols = 4
    fig_width = 16
    if not plotVelocityField:
        plotCols = 3
        fig_width = 12
    fig = plt.figure(figsize=(fig_width, 3*numPlots))

    eta_true, hu_true, hv_true = ensemble.downloadTrueOceanState()
    fieldRanges = np.zeros(6) # = [eta_min, eta_max, hu_min, hu_max, hv_min, hv_max]

    _updateMinMax(eta_true, hu_true, hv_true, fieldRanges)
    X,Y = np.meshgrid(np.arange(0, ensemble.getNx(), 1.0), np.arange(0, ensemble.getNy(), 1.0))

    eta_mean = np.zeros_like(eta_true)
    hu_mean = np.zeros_like(hu_true)
    hv_mean = np.zeros_like(hv_true)
    eta_mrse = np.zeros_like(eta_true)
    hu_mrse = np.zeros_like(hu_true)
    hv_mrse = np.zeros_like(hv_true)

    eta = [None]*numParticlePlots
    hu = [None]*numParticlePlots
    hv = [None]*numParticlePlots
    numNonNans = 0
    for p in range(ensemble.getNumParticles()):
        if p < numParticlePlots:
            eta[p], hu[p], hv[p] = ensemble.downloadParticleOceanState(p)
            if not np.isnan(eta[p].max()):
                eta_mean += eta[p]
                hu_mean += hu[p]
                hv_mean += hv[p]
                eta_mrse += (eta_true - eta[p])**2
                hu_mrse += (hu_true - hu[p])**2
                hv_mrse += (hv_true - hv[p])**2
                _updateMinMax(eta[p], hu[p], hv[p], fieldRanges)
                numNonNans += 1
        else:
            tmp_eta, tmp_hu, tmp_hv = ensemble.downloadParticleOceanState(p)
            if not np.isnan(tmp_eta.max()):
                eta_mean += tmp_eta
                hu_mean += tmp_hu
                hv_mean += tmp_hv
                eta_mrse += (eta_true - tmp_eta)**2
                hu_mrse += (hu_true - tmp_hu)**2
                hv_mrse += (hv_true - tmp_hv)**2
                _updateMinMax(tmp_eta, tmp_hu, tmp_hv, fieldRanges)
                numNonNans += 1

    eta_mean = eta_mean/ensemble.getNumParticles()
    hu_mean = hu_mean/ensemble.getNumParticles()
    hv_mean = hv_mean/ensemble.getNumParticles()
    eta_mrse = np.sqrt(eta_mrse/ensemble.getNumParticles())
    hu_mrse = np.sqrt(hu_mrse/ensemble.getNumParticles())
    hv_mrse = np.sqrt(hv_mrse/ensemble.getNumParticles())

    eta_lim = np.max(np.abs(fieldRanges[:2]))
    huv_lim = np.max(np.abs(fieldRanges[2:]))
    eta_mrse_max = eta_mrse.max()
    huv_mrse_max = max(hu_mrse.max(), hv_mrse.max())

    eta_levels = np.linspace(fieldRanges[0], fieldRanges[1], 10)
    hu_levels = np.linspace(fieldRanges[2], fieldRanges[3], 10)
    hv_levels = np.linspace(fieldRanges[4], fieldRanges[5], 10)
    eta_mrse_levels = np.linspace(0, eta_mrse_max, 5)
    huv_mrse_levels = np.linspace(0, huv_mrse_max, 5)

    fignum = 1
    ax = plt.subplot(numPlots, plotCols, fignum)
    plt.imshow(eta_true, origin='lower', vmin=-eta_lim, vmax=eta_lim)
    plt.colorbar()
    plt.contour(eta_true, levels=eta_levels, colors='black', alpha=0.5)
    plt.title("true eta")
    _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
    fignum += 1
    ax = plt.subplot(numPlots, plotCols, fignum)
    plt.imshow(hu_true, origin='lower', vmin=-huv_lim, vmax=huv_lim)
    plt.colorbar()
    plt.contour(hu_true, levels=hu_levels, colors='black', alpha=0.5)
    plt.title("true hu")
    _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
    fignum += 1
    ax = plt.subplot(numPlots, plotCols, fignum)
    plt.imshow(hv_true, origin='lower', vmin=-huv_lim, vmax=huv_lim)
    plt.colorbar()
    plt.contour(hv_true, levels=hv_levels, colors='black', alpha=0.5)
    plt.title("true hv")
    _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
    if plotVelocityField:        
        fignum += 1
        ax = plt.subplot(numPlots, plotCols, fignum)
        plt.quiver(X, Y, hu_true, hv_true)
        plt.title("velocity field")
        _markDriftersInImshow(ax, ensemble, observed_drifter_positions)

    fignum += 1
    ax = plt.subplot(numPlots, plotCols, fignum)
    plt.imshow(eta_mean, origin='lower', vmin=-eta_lim, vmax=eta_lim)
    plt.colorbar()
    plt.contour(eta_mean, levels=eta_levels, colors='black', alpha=0.5)
    plt.title("mean eta")
    _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
    fignum += 1
    ax = plt.subplot(numPlots, plotCols, fignum)
    plt.imshow(hu_mean, origin='lower', vmin=-huv_lim, vmax=huv_lim)
    plt.colorbar()
    plt.contour(hu_mean, levels=hu_levels, colors='black', alpha=0.5)
    plt.title("mean hu")
    _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
    fignum += 1
    ax = plt.subplot(numPlots, plotCols, fignum)
    plt.imshow(hv_mean, origin='lower', vmin=-huv_lim, vmax=huv_lim)
    plt.colorbar()
    plt.contour(hv_mean, levels=hv_levels, colors='black', alpha=0.5)
    plt.title("mean hv")
    _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
    if plotVelocityField:
        fignum += 1
        ax = plt.subplot(numPlots, plotCols, fignum)
        plt.quiver(X, Y, hu_mean, hv_mean)
        plt.title("velocity field")
        _markDriftersInImshow(ax, ensemble, observed_drifter_positions)


    fignum += 1
    ax = plt.subplot(numPlots, plotCols, fignum)
    plt.imshow(eta_mrse, origin='lower', vmin=0, vmax=eta_mrse_max)
    plt.colorbar()
    plt.contour(eta_mrse, levels=eta_mrse_levels, colors='black', alpha=0.5)
    plt.title("RMSE eta")
    _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
    fignum += 1
    ax = plt.subplot(numPlots, plotCols, fignum)
    plt.imshow(hu_mrse, origin='lower', vmin=0, vmax=huv_mrse_max)
    plt.colorbar()
    plt.contour(hu_mrse, levels=huv_mrse_levels, colors='black', alpha=0.5)
    plt.title("RMSE hu")
    _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
    fignum += 1
    ax = plt.subplot(numPlots, plotCols, fignum)
    plt.imshow(hv_mrse, origin='lower', vmin=0, vmax=huv_mrse_max)
    plt.title("RMSE hv")
    plt.colorbar()
    #plt.colorbar() # TODO: Find a nice way to include colorbar to this plot...
    plt.contour(hv_mrse, levels=huv_mrse_levels, colors='black', alpha=0.5)
    _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
    if plotVelocityField:
        fignum += 1

    for p in range(numParticlePlots):
        fignum += 1
        ax = plt.subplot(numPlots, plotCols, fignum)
        plt.imshow(eta[p], origin='lower', vmin=-eta_lim, vmax=eta_lim)
        plt.colorbar()
        plt.contour(eta[p], levels=eta_levels, colors='black', alpha=0.5)
        plt.title("particle eta")
        _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
        fignum += 1
        ax = plt.subplot(numPlots, plotCols, fignum)
        plt.imshow(hu[p], origin='lower', vmin=-huv_lim, vmax=huv_lim)
        plt.colorbar()
        plt.contour(hu[p], levels=hu_levels, colors='black', alpha=0.5)
        plt.title("particle hu")
        _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
        fignum += 1
        ax = plt.subplot(numPlots, plotCols, fignum)
        plt.imshow(hv[p], origin='lower', vmin=-huv_lim, vmax=huv_lim)
        plt.colorbar()
        plt.contour(hv[p], levels=hv_levels, colors='black', alpha=0.5)
        plt.title("particle hv")
        _markDriftersInImshow(ax, ensemble, observed_drifter_positions)
        if plotVelocityField:
            fignum += 1
            ax = plt.subplot(numPlots, plotCols, fignum)
            plt.quiver(X, Y, hu[p], hv[p])
            plt.title("velocity field")
            _markDriftersInImshow(ax, ensemble, observed_drifter_positions)

    plt.axis('tight')

def plotDistanceInfo(ensemble, title=None, printInfo=False):
    """
    Utility function for generating informative plots of the ensemble relative to the observation
    """
    if ensemble.observation_type == dautils.ObservationType.UnderlyingFlow or \
       ensemble.observation_type == dautils.ObservationType.DirectUnderlyingFlow or \
       ensemble.observation_type == dautils.ObservationType.StaticBuoys:
        return plotVelocityInfo(ensemble, title=title, printInfo=printInfo)

    fig = plt.figure(figsize=(10,6))
    gridspec.GridSpec(2, 3)

    # PLOT POSITIONS OF PARTICLES AND OBSERVATIONS
    ax0 = plt.subplot2grid((2,3), (0,0))
    plt.plot(ensemble.observeParticles()[:,:,0].flatten(), \
             ensemble.observeParticles()[:,:,1].flatten(), 'b.')
    plt.plot(ensemble.observeTrueState()[:,0], \
             ensemble.observeTrueState()[:,1], 'r.')
    ensembleMean = ensemble.getEnsembleMean()
    if ensembleMean is not None:
        plt.plot(ensembleMean[0], ensembleMean[1], 'r+')
    plt.xlim(0, ensemble.getDomainSizeX())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(0, ensemble.getDomainSizeY())
    plt.title("Particle positions")

    # PLOT DISCTRIBUTION OF PARTICLE DISTANCES AND THEORETIC OBSERVATION PDF
    ax0 = plt.subplot2grid((2,3), (0,1), colspan=2)
    innovations = ensemble.getInnovationNorms()
    obs_var = ensemble.getObservationVariance()
    plt.hist(innovations, bins=30, \
             range=(0, max(min(ensemble.getDomainSizeX(), ensemble.getDomainSizeY()), np.max(innovations))),\
             normed=True, label="particle innovations")

    # With observation 
    x = np.linspace(0, max(ensemble.getDomainSizeX(), ensemble.getDomainSizeY()), num=100)
    gauss_pdf = ensemble.getGaussianWeight(x, normalize=False)
    plt.plot(x, gauss_pdf, 'g', label="pdf directly from innovations")
    plt.legend()
    plt.title("Distribution of particle innovations")

    # PLOT SORTED DISTANCES FROM OBSERVATION
    ax0 = plt.subplot2grid((2,3), (1,0), colspan=3)
    gaussWeights = ensemble.getGaussianWeight()
    indices_sorted_by_observation = innovations.argsort()
    ax0.plot(gaussWeights[indices_sorted_by_observation]/np.max(gaussWeights), 'g', label="Gauss weight")
    ax0.set_ylabel('Relative weight')
    ax0.grid()
    ax0.set_ylim(0,1.4)
    plt.legend(loc=7)

    ax1 = ax0.twinx()
    ax1.plot(innovations[indices_sorted_by_observation], label="innovations")
    ax1.set_ylabel('Innovations from observation', color='b')

    plt.title("Sorted innovations from observation")

    if title is not None:
        plt.suptitle(title, fontsize=16)
    return fig

def _fillPolarPlot(ax, ensemble, observedParticles, drifter_id=0, printInfo=False):
    """
    Internal utility function for plotting current under a drifter as a polar plot.
    """
    max_r = 0
    observedParticlesSingleDrifter = observedParticles[:, drifter_id, :]
    if printInfo: print("observedParticlesSingleDrifter: \n" +str(observedParticlesSingleDrifter))
    for p in range(ensemble.getNumParticles()):
        u, v = observedParticlesSingleDrifter[p,0], observedParticlesSingleDrifter[p,1]
        r = np.sqrt(u**2 + v**2)
        max_r = max(max_r, r)
        theta = np.arctan(v/u)
        if (u < 0):
            theta += np.pi
        arr1 = plt.arrow(theta, 0, 0, r, alpha = 0.5, \
                         length_includes_head=True, \
                         edgecolor = 'green', facecolor = 'green', zorder = 5)

    obs_u = ensemble.observeTrueState()[drifter_id, 2]
    obs_v = ensemble.observeTrueState()[drifter_id, 3]
    if printInfo: print("observedTrueState: " + str((obs_u, obs_v)))
    obs_r = np.sqrt(obs_u**2 + obs_v**2)
    max_r = max(max_r, obs_r)
    obs_theta = np.arctan(obs_v/obs_u)
    if (obs_u < 0):
        obs_theta += np.pi
    arr1 = plt.arrow(obs_theta, 0, 0, obs_r, alpha = 0.5,\
                     length_includes_head=True, \
                     edgecolor = 'red', facecolor = 'red', zorder = 5)


    #ax.plot(theta, r, color='#ee8d18', lw=3)
    ax.set_rmax(max_r*1.2)
    plt.grid(True)
    plt.title("Observations from drifter " + str(drifter_id))


def plotVelocityInfo(ensemble, title=None, printInfo=False):
    """
    Utility function for generating informative plots of the ensemble relative to the observation
    """

    fig = None
    plotRows = 2
    if ensemble.getNumDrifters() == 1:
        fig = plt.figure(figsize=(10,6))
    else:
        fig = plt.figure(figsize=(10,9))
        plotRows = 3
    gridspec.GridSpec(plotRows, 3)

    observedParticles = ensemble.observeParticles()

    # PLOT POSITIONS OF PARTICLES AND OBSERVATIONS

    ax = plt.subplot2grid((plotRows,3), (0,0), polar=True)
    _fillPolarPlot(ax, ensemble, observedParticles, drifter_id=0, printInfo=printInfo)

    # PLOT DISCTRIBUTION OF PARTICLE DISTANCES AND THEORETIC OBSERVATION PDF
    ax0 = plt.subplot2grid((plotRows,3), (0,1), colspan=2)
    innovations = ensemble.getInnovationNorms()
    obs_var = ensemble.getObservationVariance()
    range_x = np.sqrt(obs_var)*20

    # With observation 
    x = np.linspace(0, range_x, num=100)
    gauss_pdf = ensemble.getGaussianWeight(x, normalize=False)
    plt.plot(x, gauss_pdf, 'g', label="pdf directly from innovations")
    plt.legend()
    plt.title("Distribution of particle innovations")

    #hisograms:
    ax1 = ax0.twinx()
    ax1.hist(innovations, bins=30, \
             range=(0, range_x),\
             density=True, label="particle innovations (norm)")

    # PLOT SORTED DISTANCES FROM OBSERVATION
    ax0 = plt.subplot2grid((plotRows,3), (1,0), colspan=3)
    gaussWeights = ensemble.getGaussianWeight()
    indices_sorted_by_observation = innovations.argsort()
    ax0.plot(gaussWeights[indices_sorted_by_observation]/np.max(gaussWeights),\
             'g', label="Weight directly from innovations")
    ax0.set_ylabel('Weights directly from innovations', color='g')
    ax0.grid()
    ax0.set_ylim(0,1.4)
    #plt.legend(loc=7)
    ax0.set_xlabel('Particle ID')

    ax1 = ax0.twinx()
    ax1.plot(innovations[indices_sorted_by_observation], label="innovations")
    ax1.set_ylabel('Innovations', color='b')

    plt.title("Sorted distances from observation")

    if ensemble.getNumDrifters() > 1:
        for drifter_id in range(1,min(4, ensemble.getNumDrifters())):
            ax = plt.subplot2grid((plotRows,3), (2,drifter_id-1), polar=True)
            _fillPolarPlot(ax, ensemble, observedParticles, drifter_id=drifter_id, printInfo=printInfo)

    if title is not None:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

def plotRMSE(ensemble):
    fig = plt.figure(figsize=(10,3))
    plt.plot(ensemble.tArray, ensemble.rmseUnderDrifter_eta, label='eta')
    plt.plot(ensemble.tArray, ensemble.rmseUnderDrifter_hu,  label='hu')
    plt.plot(ensemble.tArray, ensemble.rmseUnderDrifter_hv,  label='hv')
    #plt.plot(observation_iterations, 0.0*np.ones_like(observation_iterations), 'o')
    plt.title("RMSE under drifter")
    plt.legend(loc=0)
    plt.grid()
    #plt.ylim([0, 1])

    fig = plt.figure(figsize=(10,3))
    plt.plot(ensemble.tArray, ensemble.varianceUnderDrifter_eta, label='eta')
    plt.plot(ensemble.tArray, ensemble.varianceUnderDrifter_hu,  label='hu')
    plt.plot(ensemble.tArray, ensemble.varianceUnderDrifter_hv,  label='hv')
    #plt.plot(observation_iterations, 0.0*np.ones_like(observation_iterations), 'o')
    plt.title("Std.dev. under drifter")
    plt.legend(loc=0)
    plt.grid()
    #plt.ylim([0, 1])

    fig = plt.figure(figsize=(10,3))
    plt.plot(ensemble.tArray, ensemble.rUnderDrifter_eta, label='eta')
    plt.plot(ensemble.tArray, ensemble.rUnderDrifter_hu,  label='hu')
    plt.plot(ensemble.tArray, 1.0/np.array(ensemble.rUnderDrifter_hv),  label='hv')
    #plt.plot(observation_iterations, 0.0*np.ones_like(observation_iterations), 'o')
    plt.title("r = std.dev./rmse under drifter")
    plt.legend(loc=0)
    plt.grid()
    #plt.ylim([0, 5])