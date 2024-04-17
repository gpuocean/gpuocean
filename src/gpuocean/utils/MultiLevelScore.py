# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2023  SINTEF Digital

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
import os

class MultiLevelScore:
    """
    This class implements some ML scores
    """

    def __init__(self, args_list):
        """
        Preparing arrays to to write scores.
        
        Input:
        args_list   - dict with nx, ny and dx, dy information for a multi-level ensemble
        """

        self.args_list = args_list
        self.num_levels = len(args_list)

        # Times when scores evaluated
        self.ts = []
        # ML egliability 1: || Var[g(x^l)] ||_2
        self.scores = []
        # ML egliability 2: || Var[g(x^l+) - g(x^l-)] ||_2
        self.diff_scores = []
        # || MLmean - truth ||_2
        self.rmses = []
        # || MLstd ||_2
        self.stddevs = []
    

    @staticmethod
    def g_functionalVar(SL_state):
        """
        L_g functional as in notation of Kjetil's PhD thesis.
        This should be the functional that is under investigation for the variance level plot

        Input a ndarray of size (3, ny, nx, Ne)

        Returns a ndarray of same size as SL_state (3, ny, nx, Ne)
        """
        return (SL_state - np.mean(SL_state, axis=-1)[:,:,:,np.newaxis])**2
        
    @staticmethod
    def L2norm(field, lvl_grid_args):
        """
        integral_D(f dx)
        where D are uniform finite volumes

        Input:
        field           - ndarray of shape (3,ny,nx,..)
        lvl_grid_args   - dict with nx, ny and dx, dy information

        Output:
        L2norm          - ndarray of shape (3,...)
        """
        # assert field.shape[1:3] == (lvl_grid_args["ny"], lvl_grid_args["nx"]), "field has wrong resolution"
        return np.sqrt(np.sum((field)**2 * lvl_grid_args["dx"]*lvl_grid_args["dy"], axis=(1,2)))


    def assess_score(self, MLOceanEnsemble):
        ML_state = MLOceanEnsemble.download()

        score = np.zeros((self.num_levels,3))

        score[0] = MultiLevelScore.L2norm(np.var(MultiLevelScore.g_functionalVar(ML_state[0]),ddof=1, axis=-1), self.args_list[0])
        for l_idx in range(1,self.num_levels):
            score[l_idx] = MultiLevelScore.L2norm(np.var(MultiLevelScore.g_functionalVar(ML_state[l_idx][0]),ddof=1, axis=-1), self.args_list[l_idx])
        
        self.scores.append(score)


    def assess_diff_score(self, MLOceanEnsemble):
        ML_state = MLOceanEnsemble.download()

        diff_score = np.zeros((self.num_levels,3))

        for l_idx in range(1,self.num_levels):
            diff_score[l_idx] = MultiLevelScore.L2norm(np.var(MultiLevelScore.g_functionalVar(ML_state[l_idx][0]) 
                                                         - MultiLevelScore.g_functionalVar(ML_state[l_idx][1].repeat(2,1).repeat(2,2)),
                                                        ddof=1, axis=-1), 
                                                    self.args_list[l_idx])

        self.diff_scores.append(diff_score)


    def assess_rmse(self, MLOceanEnsemble, truth):
        if not isinstance(truth, str):
            true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
        else: 
            true_eta, true_hu, true_hv = np.load(truth)

        rmse = MultiLevelScore.L2norm(MLOceanEnsemble.estimate(np.mean) - [true_eta, true_hu, true_hv], self.args_list[-1])

        self.rmses.append(rmse)

    
    def assess_stddev(self, MLOceanEnsemble):
        stddev = MultiLevelScore.L2norm(MLOceanEnsemble.estimate(np.std, ddof=1), self.args_list[-1] )

        self.stddevs.append(stddev)


    def assess(self, MLOceanEnsemble, truth):
        """
        Assess all skills together

        MLOceanEnsemble - MultiLevelOceanEnsemble instance (with same properties as from `args_list` in `__init__`)
        truth           - 1) CDKLM instance (with same properties as `args_list[-1]`)
                          2) str with path to a `.npy` with state corresponding to `args_list[-1]`
        """
        self.ts.append(MLOceanEnsemble.t)
        self.assess_score(MLOceanEnsemble)
        self.assess_diff_score(MLOceanEnsemble)
        self.assess_rmse(MLOceanEnsemble, truth)
        self.assess_stddev(MLOceanEnsemble)


    def save2file(self, output_path, phrase=""):
        np.save(output_path+"/MLts"+phrase+".npy", self.ts)
        np.save(output_path+"/MLscores"+phrase+".npy", np.array(self.scores))
        np.save(output_path+"/MLdiff_scores"+phrase+".npy", np.array(self.diff_scores))
        np.save(output_path+"/MLrmses"+phrase+".npy", np.array(self.rmses))
        np.save(output_path+"/MLstddevs"+phrase+".npy", np.array(self.stddevs))