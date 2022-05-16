# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018-2019 SINTEF Digital

This module implements a selection of resampling schemes used for 
particle filters, as described in 
van Leeuwen, P.J., 2009: Particle Filtering in Geophysical Systems. 
Mon. Wea. Rev., 137, 4089–4114, https://doi.org/10.1175/2009MWR2835.1 


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

from SWESimulators import Common

class ObservationType:
    """
    An enum-type class for defining different types of observation operators.
    """
    DrifterPosition = 1
    UnderlyingFlow = 2
    DirectUnderlyingFlow = 3
    StaticBuoys = 4
    
    @staticmethod
    def _assert_valid(obs_type):
        assert(obs_type == ObservationType.DrifterPosition or \
               obs_type == ObservationType.UnderlyingFlow or \
               obs_type == ObservationType.DirectUnderlyingFlow or \
               obs_type == ObservationType.StaticBuoys), \
        'Provided observation type ' + str(obs_type) + ' is invalid'


    
def probabilisticResampling(ensemble, reinitialization_variance=0):
    """
    Probabilistic resampling of the particles based on the attached observation.
    Particles are sampled directly from the discrete distribution given by their weights.

    ensemble: The ensemble to be resampled, holding the ensemble particles, the observation, and measures to compute the weight of particles based on this information.
    reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
    If reinitialization_variance is zero, exact duplications are generated.

    Implementation based on the description in van Leeuwen (2009) 'Particle Filtering in Geophysical Systems', Section 3a.1)
    """
    
    # Obtain weights:
    weights = ensemble.getGaussianWeight()
    #weights = getCauchyWeight(ensemble.getDistances(), \
    #                          ensemble.getObservationVariance())
    
    # Create array of possible indices to resample:
    allIndices = np.arange(ensemble.getNumParticles())
    
    # Draw new indices based from discrete distribution based on weights
    newSampleIndices = np.random.choice(allIndices, ensemble.getNumParticles(), p=weights)
        
    # Return a new set of particles
    ensemble.resample(newSampleIndices, reinitialization_variance)


def residualSampling(ensemble, reinitialization_variance=0, onlyDeterministic=False, onlyStochastic=False):
    """
    Residual resampling of particles based on the attached observation.
    Each particle is first resampled floor(N*w) times, which in total given M <= N ensemble. Afterwards, N-M particles are drawn from the discrete distribution given by weights N*w % 1.

   ensemble: The ensemble to be resampled, holding the ensemble particles, the observation, and measures to compute the weight of particles based on this information.
    reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
    If reinitialization_variance is zero, exact duplications are generated.

    Implementation based on the description in van Leeuwen (2009) 'Particle Filtering in Geophysical Systems', Section 3a.2)
    """
    
    # Obtain weights:
    #weights = getCauchyWeight(ensemble.getDistances(), \
    #                          ensemble.getObservationVariance())
    weights = ensemble.getGaussianWeight()
    
    # Create array of possible indices to resample:
    allIndices = np.arange(ensemble.getNumParticles())

    # Deterministic resampling based on the integer part of N*weights:
    weightsTimesN = weights*ensemble.getNumParticles()
    weightsTimesNInteger = np.int64(np.floor(weightsTimesN))
    deterministicResampleIndices = np.repeat(allIndices, weightsTimesNInteger)
    
    # Stochastic resampling based on the decimal parts of N*weights:
    decimalWeights = np.mod(weightsTimesN, 1)
    decimalWeights = decimalWeights/np.sum(decimalWeights)
    stochasticResampleIndices = np.random.choice(allIndices, 
                                                 ensemble.getNumParticles() - len(deterministicResampleIndices), 
                                                 p=decimalWeights)
    ### NOTE!
    # In numpy v >= 1.13, np.divmod can be used to get weightsTimesNInteger and decimalWeights from one function call.
    
    if onlyDeterministic:
        ensemble.resample(deterministicResampleIndices, reinitialization_variance)
    if onlyStochastic:
        ensemble.resample(stochasticResampleIndices, reinitialization_variance)
    
    ensemble.resample(np.concatenate((deterministicResampleIndices, stochasticResampleIndices)), \
                      reinitialization_variance)
    


def stochasticUniversalSampling(ensemble, reinitialization_variance=0):
    """
    Stochastic resampling of particles based on the attached observation.
    Consider all weights as line lengths, so that all particles represent segments completely covering the line [0, 1]. Draw u ~ U[0,1/N], and resample all particles representing points u + i/N, i = 0,...,N-1 on the line.

    ensemble: The ensemble to be resampled, holding the ensemble particles, the observation, and measures to compute the weight of particles based on this information.
    reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
    If reinitialization_variance is zero, exact duplications are generated.

    Implementation based on the description in van Leeuwen (2009) 'Particle Filtering in Geophysical Systems', Section 3a.3)
    """   
    
    # Obtain weights:
    #weights = getCauchyWeight(ensemble.getDistances(), \
    #                          ensemble.getObservationVariance())
    weights = ensemble.getGaussianWeight()

    # Create array of possible indices to resample:
    allIndices = np.array(range(ensemble.getNumParticles()))
    
    # Create histogram buckets based on the cumulative weights
    cumulativeWeights = np.concatenate(([0.0], np.cumsum(weights)))
    
    # Find first starting position:
    startPos = np.random.rand()/ensemble.getNumParticles()
    lengths = 1.0/ensemble.getNumParticles()
    #print startPos, lengths
    selectionValues = allIndices*lengths + startPos
    
    # Create a histogram of selectionValues within the cumulativeWeights buckets
    bucketValues, buckets = np.histogram(selectionValues, bins=cumulativeWeights)
    
    #newSampleIndices has now the number of times each index should be resampled
    # We need to go from [0, 0, 1, 4, 0] to [2,3,3,3,3]
    newSampleIndices = np.repeat(allIndices, bucketValues)
    
    # Return a new set of particles
    ensemble.resample(newSampleIndices, reinitialization_variance)


def metropolisHastingSampling(ensemble,  reinitialization_variance=0):
    """
    Resampling based on the Monte Carlo Metropolis-Hasting algorithm.
    The first particle, having weight w_1, is automatically resampled. The next particle, with weight w_2, is then resampled with the probability p = w_2/w_1, otherwise the first particle is sampled again. The latest resampled particle make the comparement basis for the next particle. 

    ensemble: The ensemble to be resampled, holding the ensemble particles, the observation, and measures to compute the weight of particles based on this information.
    reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
    If reinitialization_variance is zero, exact duplications are generated.

    Implementation based on the description in van Leeuwen (2009) 'Particle Filtering in Geophysical Systems', Section 3a.4)
    """
    
    # Obtain weights:
    #weights = getCauchyWeight(ensemble.getDistances(), \
    #                          ensemble.getObservationVariance())
    weights = ensemble.getGaussianWeight()
    
    # Create buffer for indices which should be in the new ensemble:
    newSampleIndices = np.zeros_like(weights, dtype=int)
    
    # The first member is automatically a member of the new ensemble
    newSampleIndices[0] = 0
    
    # Iterate through all weights, and apply the Metropolis-Hasting algorithm
    for i in range(1, ensemble.getNumParticles()):
        # Draw random number U[0,1]
        p = np.random.rand()
        if p < weights[i]/weights[newSampleIndices[i-1]]:
            newSampleIndices[i] = i
        else:
            newSampleIndices[i] = newSampleIndices[i-1]
    
    # Return a new set of particles
    ensemble.resample(newSampleIndices, reinitialization_variance)
