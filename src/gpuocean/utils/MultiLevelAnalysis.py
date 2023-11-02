import numpy as np
from matplotlib import pyplot as plt


class Analysis:

    def __init__(self, grid_args_list, vars, diff_vars, works=None, diff_works=None):
        """
        data_args_list: list of level information. Required keys: dx, dy, nx, ny
        vars_file: path to npz as result of `run_LvlVar.py`. ATTENTION: with same ls as given here
        diff_vars_file: path to npz as result of `run_LvlVar.py`. ATTENTION: with same ls as given here
        """
        
        self.dxs = [grid_args["dx"] for grid_args in grid_args_list]
        self.dys = [grid_args["dy"] for grid_args in grid_args_list]

        self.nxs = [grid_args["nx"] for grid_args in grid_args_list]
        self.nys = [grid_args["ny"] for grid_args in grid_args_list]


        self.vars = vars
        if isinstance(vars, str):
            self.vars = np.load(vars)

        assert len(grid_args_list) == len(self.vars), "Wrong number of levels"

        
        self.diff_vars = diff_vars
        if isinstance(diff_vars, str):
            self.diff_vars = np.load(diff_vars)

        if len(self.diff_vars) == len(self.vars):
            self.diff_vars = self.diff_vars[1:]

        
        self.works = works
        if works is not None:
            if isinstance(works, str):
                self.works = np.load(works)

            assert len(grid_args_list) == len(self.works), "Wrong number of levels"

        self.diff_works = diff_works
        if diff_works is not None:
            if isinstance(diff_works, str):
                self.diff_works = np.load(diff_works)

            if len(self.diff_works) == len(self.works):
                self.diff_works = self.diff_works[1:]



    def plotLvlVar(self, relative=False):

        if relative: 
            vars = self.vars/self.vars[-1]
            diff_vars = self.diff_vars/self.vars[-1]
        else:
            vars = self.vars
            diff_vars = self.diff_vars

        with plt.rc_context({'lines.color':'black', 
                        'text.color':'black', 
                        'axes.labelcolor':'black', 
                        'xtick.color':'black',
                        'ytick.color':'black'}):
            fig, axs = plt.subplots(1,3, figsize=(15,5))

            Nxs = [nx*ny for nx, ny in zip(self.nxs, self.nys)]
            variable_name = ["\eta", "hu", "hv"]
            for i in range(3):
                axs[i].loglog(Nxs, vars[:,i], label=r"$|| Var["+variable_name[i]+"^l] ||_{L^2}$", linewidth=3)
                axs[i].loglog(Nxs[1:], diff_vars[:,i], label=r"$|| Var["+variable_name[i]+"^l-"+variable_name[i]+"^{l-1}] ||_{L^2}$", linewidth=3)
                axs[i].set_xlabel("# grid cells", fontsize=15)
                axs[i].legend(labelcolor="black", loc=(0.2,0.5), fontsize=15)

                axs[i].set_xticks(Nxs)
                axs[i].xaxis.grid(True)
                axs[i].set_xticklabels(Nxs)



    def _level_work(self, l_idx):
        """
        Cubic work in terms of grid discretisation

        The dx should be in synv with `BasinInit.py`
        """
        if l_idx == 0:
            if self.works is None:
                dx = 1/2*(self.dxs[0] + self.dys[0])
                return dx**(-3)
            else:
                return self.works[0]
        else:
            if self.works is None:
                dx = 1/2*(self.dxs[l_idx] + self.dys[l_idx])
                coarse_dx = 1/2*(self.dxs[l_idx-1] + self.dys[l_idx-1])
                return dx**(-3) + coarse_dx**(-3)
            else:
                return self.diff_works[l_idx-1]


    def optimal_Ne(self, tau):
        """
        Evaluating the optimal ML ensemble size for a error level `tau`

        See Ch. 5 of Kjetils thesis for reference 
        """

        rel_vars = self.vars/self.vars[-1]
        rel_diff_vars = self.diff_vars/self.vars[-1]

        avg_vars = np.mean(rel_vars, axis=1)
        avg_diff_vars = np.mean(rel_diff_vars, axis=-1)


        allwork = 0
        for l_idx in range(len(self.dxs)):
            if l_idx == 0: 
                allwork += np.sqrt(avg_vars[l_idx] * self._level_work(l_idx))
            else:
                allwork += np.sqrt(avg_diff_vars[l_idx-1] * self._level_work(l_idx))

        optNe_ref = np.zeros(len(self.dxs))
        for l_idx in range(len(self.dxs)):
            if l_idx == 0: 
                optNe_ref[l_idx] = np.sqrt(avg_vars[l_idx]/self._level_work(l_idx)) * allwork
            else: 
                optNe_ref[l_idx] = np.sqrt(avg_diff_vars[l_idx-1]/self._level_work(l_idx)) * allwork

        return np.int32(np.ceil(1/(tau**2)*optNe_ref))

    
    def work(self, Nes):
        """
        Evaluating the work for an ML ensemble
        work(0 and + ensemble members) + work(- ensemble members)
        """
        assert len(Nes) == len(self.dxs), "Wrong number of levels"
        return np.sum([self._level_work(l_idx) for l_idx in range(len(self.dxs))] * np.array(Nes))


    def theoretical_error(self, Nes):
        """
        Evaluating the theoretical error for an ML ensemble
        Ref: PhD thesis of Kjetil, Thm 5.2.1 or PhD thesis Florian Muller, eq. (2.21)
        """
        assert len(Nes) == len(self.dxs), "Wrong number of levels"

        ## Kjetil:
        # theo_err = np.sqrt(self.vars[0])/np.sqrt(Nes[0])
        # for l_idx in range(1, len(self.dxs)):
        #     theo_err += np.sqrt(self.diff_vars[l_idx-1])/np.sqrt(Nes[l_idx])

        # return theo_err

        ## Florian
        theo_err = self.vars[0]/Nes[0]
        for l_idx in range(1, len(self.dxs)):
            theo_err += self.diff_vars[l_idx-1]/Nes[l_idx]

        return np.sqrt(theo_err)