import os
import numpy as np

from gpuocean.oilspill import OilDrift

class OilSpillIO:
    """
    File interaction for oil spill simulations 
    """

    def __init__(self, positions=None, diameters=None, t=None):
        self.positions = positions if positions is not None else []
        self.diameters = diameters if diameters is not None else []
        self.t = t if t is not None else []

    def save_from_sim(self, sim):
        self.t.append(sim.t)
        self.positions.append(sim.drifters.getDrifterPositions())
        self.diameters.append(sim.drifters.getDropletDiameters())

    def write_to_file(self, filename):
        if filename[-4:] != ".npz":
            filename = filename + ".npz"

        d = {
            "t": self.t,
            "positions": self.positions,
            "diameters": self.diameters,
            }
        np.savez(filename, **d)

    @classmethod
    def fromfilename(cls, filename):
        if filename[-4:] != ".npz":
            filename = filename + ".npz"
        
        with np.load(filename, allow_pickle=True) as data:
            t = data["t"]
            pos = data["positions"]
            dia = data["diameters"]
        return cls(positions=pos, diameters=dia, t=t)

    def create_drifter_object(self, gpu_ctx, t_index):
        assert(t_index < len(self.t) or t_index == -1)
        return OilDrift.OilDrift(gpu_ctx, self.positions[t_index])


    