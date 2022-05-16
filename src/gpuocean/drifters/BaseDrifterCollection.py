# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class provides an abstract collection of Drifters. All 
actual drifters should inherit this class, and implement its abstract methods.

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
import numpy as np
import time
import abc

from SWESimulators import Common
from SWESimulators import DataAssimilationUtils as dautils

class BaseDrifterCollection(object):    
    """
    Abstract collection of Drifters. 
    All actual drifters should inherit this class, and implement its abstract methods.
    """ 
    
    __metaclass__ = abc.ABCMeta

    def __init__(self, numDrifters, observation_variance=0.1, \
                 boundaryConditions=Common.BoundaryConditions(), \
                 domain_size_x=1.0, domain_size_y=1.0):
        """
        Creates a GlobalParticles object for drift trajectory ensemble.

        numDrifters: number of drifers in the collection, not included the observation
        observation_variance: uncertainty of observation position
        boundaryConditions: BoundaryConditions object, relevant during re-initialization of particles.    
        """
        
        self.numDrifters = numDrifters
        
        # Observation index is the last particle
        self.obs_index = self.numDrifters 
        self.observation_variance = observation_variance
        
        self.positions = None # Needs to be allocated in the child class
        # Should represent all particles plus observation
        
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        
        # Boundary conditions are read from a BoundaryConditions object
        self.boundaryConditions = boundaryConditions
    
    def __del__(self):
        self.cleanUp()
    
    def copy(self):
        """
        Makes an independent indentical copy of the current object
        """
        pass
            
    
    ### Abstract GETs
    @abc.abstractmethod
    def getDrifterPositions(self):
        pass
    
    @abc.abstractmethod
    def getObservationPosition(self):
        pass
    
    ### Abstract SETs
    @abc.abstractmethod
    def setDrifterPositions(self, newDrifterPositions):
        pass
    
    @abc.abstractmethod
    def setObservationPosition(self, newObservationPosition):
        pass
    
    @abc.abstractmethod
    def cleanUp(self):
        pass
    
    
    ### GETs
    def getNumDrifters(self):
        #print "\n\nUsing DrifterCollection::getNumDrifters()\n"
        return self.numDrifters
    
    def getObservationVariance(self):
        return self.observation_variance
    
    def getBoundaryConditions(self):
        return self.boundaryConditions
    
    def getDomainSizeX(self):
        return self.domain_size_x
    
    def getDomainSizeY(self):
        return self.domain_size_y
        
        
    ### SETs
    def setBoundaryConditions(self, boundaryConditions):
        self.boundaryConditions = boundaryConditions
        
    def setDomainSize(self, size_x, size_y):
        self.domain_size_x = size_x
        self.domain_size_y = size_y
        
    
    ### Other abstract methods
    @abc.abstractmethod
    def enforceBoundaryConditions(self):
        """
        Enforces boundary conditions on all particles in the ensemble, and the observation.
        This function should be called whenever particles are moved, to enforce periodic boundary conditions for particles that have left the domain.
        """
        pass
        
        
    ### METHODS UNIQUELY DEFINED FOR ALL CHILD CLASSES
    
    def uniformly_distribute_drifters(self, initialization_cov_drifters=None):
        """
        Initializing/re-distributing drifters so that they are equally spread as much as possibly 
        within the given domain, by creating a regular grid and using the cell centers as drifter
        positions.
        """
        # Define mid-points for the different drifters k
        # Decompose the domain, so that we spread the drifters as much as possible
        sub_domains_y = np.int(np.round(np.sqrt(self.numDrifters)))
        sub_domains_x = np.int(np.ceil(1.0*self.numDrifters/sub_domains_y))
        drifterPositions = np.empty((self.numDrifters, 2))
       
        for sub_y in range(sub_domains_y):
            for sub_x in range(sub_domains_x):
                drifter_id = sub_y*sub_domains_x + sub_x
                if drifter_id >= self.numDrifters:
                    break
                drifterPositions[drifter_id, 0]  = (sub_x + 0.5)*self.domain_size_x/sub_domains_x
                drifterPositions[drifter_id, 1]  = (sub_y + 0.5)*self.domain_size_y/sub_domains_y
        
        # Perturb the drifter positions
        if initialization_cov_drifters is None:
            initialization_cov_drifters = np.zeros((2,2))
        elif np.isscalar(initialization_cov_drifters):
            initialization_cov_drifters = np.eye(2)*initialization_cov_drifters
            
        assert(initialization_cov_drifters.shape == (2,2)), \
            'initialization_cov_drifters has the wrong shape: ' + str(initialization_cov_drifters)
        
        for d in range(self.numDrifters):
            drifterPositions[d,:] = np.random.multivariate_normal(drifterPositions[d,:], initialization_cov_drifters)
            
        self.setDrifterPositions(drifterPositions)
    
    def _getClosestPositions(self, obs=None):
        """
        Returns a set of coordinates corresponding to each particles closest position to the observation,
        considering possible periodic boundary conditions
        """
        if not (self.boundaryConditions.isPeriodicNorthSouth() or self.boundaryConditions.isPeriodicEastWest()):
            #return self.positions 
            return self.getDrifterPositions()
        else:
            periodicPositions = self.getDrifterPositions().copy()
            obs_x, obs_y = self.getObservationPosition()
            if obs is not None:
                obs_x = obs[0]
                obs_y = obs[1]
            if self.boundaryConditions.isPeriodicEastWest():
                for i in range(self.getNumDrifters()):
                    x = periodicPositions[i,0]
                    
                    pos_x = np.array([x - self.getDomainSizeX(), x, \
                                      x + self.getDomainSizeX()])
                    dist_x = np.abs(pos_x - obs_x)
                    periodicPositions[i,0] = pos_x[np.argmin(dist_x)]

            if self.boundaryConditions.isPeriodicNorthSouth():
                for i in range(self.getNumDrifters()):
                    y = periodicPositions[i,1]
                    
                    pos_y = np.array([y - self.getDomainSizeY(), y, \
                                      y + self.getDomainSizeY()])
                    dist_y = np.abs(pos_y - obs_y)
                    periodicPositions[i,1] = pos_y[np.argmin(dist_y)]
        return periodicPositions
        
    
    def getDistances(self, obs=None):
        """
        Computes the distance between drifter and observation. Possible periodic boundary conditions are taken care of.
        obs can be sat to be different than the observation within this collection.
        """
        distances = np.zeros(self.getNumDrifters())
        innovations = self.getInnovations(obs)
        if obs is None:
            obs = self.getObservationPosition()
        for i in range(self.getNumDrifters()):
            distances[i] = np.sqrt( (innovations[i,0])**2 +
                                    (innovations[i,1])**2)
        return distances
        
    def getInnovations(self, obs=None):
        """
        Computes the innovation, meaning the distance vector between drifter
        and observation, as d = y - H(x).
        Possible periodic boundary conditions are taken care of.
        
        obs can be sat to be different than the observation within this collection.
        """
        if obs is None:
            obs = self.getObservationPosition()
            closestPositions = self._getClosestPositions()
        else:
            closestPositions = self._getClosestPositions(obs)
        for i in range(self.getNumDrifters()):
            closestPositions[i,0] = obs[0] - closestPositions[i,0]
            closestPositions[i,1] = obs[1] - closestPositions[i,1]

        return closestPositions

    def getGaussianWeight(self, distances=None, normalize=True):
        """
        Calculates a weight associated to every particle, based on the distances from the observation, using 
        Gaussian uncertainty for the observation.
        """

        if distances is None:
            distances = self.getDistances()
        observationVariance = self.getObservationVariance()
        
        weights = (1.0/np.sqrt(2*np.pi*observationVariance))* \
                   np.exp(- (distances**2/(2*observationVariance)))

        if normalize:
            return weights/np.sum(weights)
        return weights  
    
    def getCauchyWeight(self, distances=None, normalize=True):
        """
        Calculates a weight associated to every particle, based on its distance from the observation, 
        using Cauchy distribution based on the uncertainty of the position of the observation.
        This distribution should be used if wider tails of the distribution is beneficial.
        """
        
        if distances is None:
            distances = self.getDistances()
        observationVariance = self.getObservationVariance()
            
        weights = 1.0/(np.pi*np.sqrt(observationVariance)*(1 + (distances**2/observationVariance)))
        if normalize:
            return weights/np.sum(weights)
        return weights
    
    
    
    def _enforceBoundaryConditionsOnPosition(self, x, y):
        """
        Maps the given coordinate to a coordinate within the domain. This function assumes that periodic boundary conditions are used, and should be considered as a private function.
        """
        ### TODO: SWAP the if's with while's?
        # Check what we assume is periodic boundary conditions
        if x < 0:
            x = self.domain_size_x + x
        if y < 0:
            y = self.domain_size_x + y
        if x > self.domain_size_x:
            x = x - self.domain_size_x
        if y > self.domain_size_y:
            y = y - self.domain_size_y
        return x, y
    
    
    def getCollectionMean(self, obs=None):
        """
        Calculates the mean position of all the drifters in the collection.
        For cases with periodic boundary conditions, the every drifter position is considered in the direction which minimize the distance to the observation.
        """
        closestPositions = self._getClosestPositions(obs=obs)
        mean_x, mean_y = np.mean(closestPositions, axis=0)
        mean_x, mean_y = self._enforceBoundaryConditionsOnPosition(mean_x, mean_y)
        return np.array([mean_x, mean_y])
    
    
    
    def resample(self, newSampleIndices, reinitialization_variance):
        """
        Resamples the particle positions at the given indices. Duplicates are resampled from a gaussian distribution.

        newSampleIndices: particle indices selected for resampling
        reinitialization_variance: variance used when resampling duplicates
        """

        oldDrifterPositions = self.getDrifterPositions().copy()
        newNumberOfDrifters = len(newSampleIndices)
        newDrifterPositions = np.zeros((newNumberOfDrifters, 2))

        if self.getNumDrifters() != newNumberOfDrifters:
            raise RuntimeError("ERROR: The size of the new ensemble differs from the old size!\n" + \
                               "(old size, new size): " + str((self.getNumDrifters(), newNumberOfDrifters)) + \
                               "\nWe can fix this in the future by requiring a function resizeEnsemble")

        # We really do not the if. The random number with zero variance returns exactly the mean
        if reinitialization_variance == 0:
            # Simply copy the given positions
            newDrifterPositions[:,:] = oldDrifterPositions[newSampleIndices, :]
        else:
            # Make sure to make a clean copy of first resampled particle, and add a disturbance of the next ones.
            resampledOnce = np.full(self.getNumDrifters(), False, dtype=bool)
            var = np.eye(2)*reinitialization_variance
            for i in range(len(newSampleIndices)):
                index = newSampleIndices[i]
                if resampledOnce[index]:
                    newDrifterPositions[i,:] = np.random.multivariate_normal(oldDrifterPositions[index,:], var)
                else:
                    newDrifterPositions[i,:] = oldDrifterPositions[index,:]
                    resampledOnce[index] = True

        # Set particle positions to the ensemble:            
        self.setDrifterPositions(newDrifterPositions)

        # Enforce boundary conditions
        self.enforceBoundaryConditions()


    
    def initializeUniform(self):
        """
        Initialization of all particles (and observation) within the domain.
        """

        # Initialize in unit square
        positions = np.random.rand(self.getNumDrifters()+1, 2)  
        
        # Move observation to the middle 0.5x0.5 square
        positions[-1,:] = positions[-1,:]*0.5 + 0.25
        
        # Map to domain
        positions[:,0] = positions[:,0]*self.domain_size_x
        positions[:,1] = positions[:,1]*self.domain_size_y
        
        # Set positions
        self.setDrifterPositions(positions[:-1, :])
        self.setObservationPosition(positions[-1, :])
    
    def plotDistanceInfo(self, title=None):
        """
        Utility function for generating informative plots of the ensemble relative to the observation
        """    
        fig = plt.figure(figsize=(10,6))
        gridspec.GridSpec(2, 3)
        
        # PLOT POSITIONS OF PARTICLES AND OBSERVATIONS
        ax0 = plt.subplot2grid((2,3), (0,0))
        plt.plot(self.getDrifterPositions()[:,0], \
                 self.getDrifterPositions()[:,1], 'b.')
        plt.plot(self.getObservationPosition()[0], \
                 self.getObservationPosition()[1], 'r.')
        collectionMean = self.getCollectionMean()
        if collectionMean is not None:
            plt.plot(collectionMean[0], collectionMean[1], 'r+')
        plt.xlim(0, self.getDomainSizeX())
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(0, self.getDomainSizeY())
        plt.title("Particle positions")
        
        # PLOT DISCTRIBUTION OF PARTICLE DISTANCES AND THEORETIC OBSERVATION PDF
        ax0 = plt.subplot2grid((2,3), (0,1), colspan=2)
        distances = self.getDistances()
        obs_var = self.getObservationVariance()
        plt.hist(distances, bins=30, \
                 range=(0, max(min(self.getDomainSizeX(), self.getDomainSizeY()), np.max(distances))),\
                 normed=True, label="particle distances")
        
        # With observation 
        x = np.linspace(0, max(self.getDomainSizeX(), self.getDomainSizeY()), num=100)
        cauchy_pdf = self.getCauchyWeight(x, normalize=False)
        gauss_pdf = self.getGaussianWeight(x, normalize=False)
        plt.plot(x, cauchy_pdf, 'r', label="obs Cauchy pdf")
        plt.plot(x, gauss_pdf, 'g', label="obs Gauss pdf")
        plt.legend()
        plt.title("Distribution of particle distances from observation")
        
        # PLOT SORTED DISTANCES FROM OBSERVATION
        ax0 = plt.subplot2grid((2,3), (1,0), colspan=3)
        cauchyWeights = self.getCauchyWeight()
        gaussWeights = self.getGaussianWeight()
        indices_sorted_by_observation = distances.argsort()
        ax0.plot(cauchyWeights[indices_sorted_by_observation]/np.max(cauchyWeights), 'r', label="Cauchy weight")
        ax0.plot(gaussWeights[indices_sorted_by_observation]/np.max(gaussWeights), 'g', label="Gauss weight")
        ax0.set_ylabel('Relative weight')
        ax0.grid()
        ax0.set_ylim(0,1.4)
        plt.legend(loc=7)
        
        ax1 = ax0.twinx()
        ax1.plot(distances[indices_sorted_by_observation], label="distance")
        ax1.set_ylabel('Distance from observation', color='b')
        
        plt.title("Sorted distances from observation")

        if title is not None:
            plt.suptitle(title, fontsize=16)
            
            