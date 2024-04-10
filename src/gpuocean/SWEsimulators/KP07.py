# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2016 SINTEF ICT, 
Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

This python module implements the Kurganov-Petrova numerical scheme 
for the shallow water equations, described in 
A. Kurganov & Guergana Petrova
A Second-Order Well-Balanced Positivity Preserving Central-Upwind
Scheme for the Saint-Venant System Communications in Mathematical
Sciences, 5 (2007), 133-160. 

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

#Import packages we need
import numpy as np
import gc

from gpuocean.utils import Common, SimWriter, SimReader, WindStress, AtmosphericPressure
from gpuocean.SWEsimulators import Simulator

class KP07(Simulator.Simulator):
    """
    Class that solves the SW equations using the Forward-Backward linear scheme
    """

    def __init__(self, \
                 gpu_ctx, \
                 eta0, H, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f=0.0, r=0.0, \
                 t=0.0, \
                 theta=1.3, use_rk2=True,
                 coriolis_beta=0.0, \
                 y_zero_reference_cell = 0, \
                 wind=WindStress.WindStress(), \
                 atmospheric_pressure=AtmosphericPressure.AtmosphericPressure(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 write_netcdf=False, \
                 comm=None, \
                 ignore_ghostcells=False, \
                 offset_x=0, offset_y=0, \
                 flux_slope_eps = 1.0e-1, \
                 depth_cutoff = 1.0e-5, \
                 flux_balancer = 0.0, \
                 block_width=32, block_height=16):
        """
        Initialization routine
        eta0: Initial deviation from mean sea level incl ghost cells, (nx+2)*(ny+2) cells
        hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+2) cells
        hv0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+1) cells
        H: Depth from equilibrium defined on cell corners, (nx+5)*(ny+5) corners
        nx: Number of cells along x-axis
        ny: Number of cells along y-axis
        dx: Grid cell spacing along x-axis (20 000 m)
        dy: Grid cell spacing along y-axis (20 000 m)
        dt: Size of each timestep (90 s)
        g: Gravitational accelleration (9.81 m/s^2)
        f: Coriolis parameter (1.2e-4 s^1), effectively as f = f + beta*y
        r: Bottom friction coefficient (2.4e-3 m/s)
        t: Start simulation at time t
        theta: MINMOD theta used the reconstructions of the derivatives in the numerical scheme
        use_rk2: Boolean if to use 2nd order Runge-Kutta (false -> 1st order forward Euler)
        coriolis_beta: Coriolis linear factor -> f = f + beta*(y-y_0)
        y_zero_reference_cell: The cell representing y_0 in the above, defined as the lower face of the cell .
        wind: Wind stress parameters
        atmospheric_pressure: Object with values for atmospheric pressure
        boundary_conditions: Boundary condition object
        write_netcdf: Write the results after each superstep to a netCDF file
        comm: MPI communicator
        depth_cutoff: Used for defining dry cells
        flux_slope_eps: Used for desingularization with dry cells
        flux_balancer: linear combination of upwind flux (value 1.0) and central-upwind flux (value 0.0)
        """
       
        
        ghost_cells_x = 2
        ghost_cells_y = 2
        y_zero_reference_cell = 2.0 + y_zero_reference_cell
        
        # Boundary conditions
        self.boundary_conditions = boundary_conditions

        # Extend the computational domain if the boundary conditions
        # require it
        if (boundary_conditions.isSponge()):
            nx = nx + boundary_conditions.spongeCells[1] + boundary_conditions.spongeCells[3] - 2*ghost_cells_x
            ny = ny + boundary_conditions.spongeCells[0] + boundary_conditions.spongeCells[2] - 2*ghost_cells_y
            y_zero_reference_cell = boundary_conditions.spongeCells[2] + y_zero_reference_cell
            
        self.use_rk2 = use_rk2
        rk_order = np.int32(use_rk2 + 1)
        A = None
        super(KP07, self).__init__(gpu_ctx, \
                                   nx, ny, \
                                   ghost_cells_x, \
                                   ghost_cells_y, \
                                   dx, dy, dt, \
                                   g, f, r, A, \
                                   t, \
                                   theta, rk_order, \
                                   coriolis_beta, \
                                   y_zero_reference_cell, \
                                   wind, \
                                   atmospheric_pressure, \
                                   write_netcdf, \
                                   ignore_ghostcells, \
                                   offset_x, offset_y, \
                                   comm, \
                                   block_width, block_height)
            
        # Index range for interior domain (north, east, south, west)
        # so that interior domain of eta is
        # eta[self.interior_domain_indices[2]:self.interior_domain_indices[0], \
        #     self.interior_domain_indices[3]:self.interior_domain_indices[1] ]
        self.interior_domain_indices = np.array([-2,-2,2,2])
        self._set_interior_domain_from_sponge_cells()
        
        # The ocean simulators and the swashes cases are defined on
        # completely different scales. We therefore specify a different
        # desingularization parameter if we run a swashes case.
        # Typical values:
        #ifndef SWASHES
            #define KPSIMULATOR_FLUX_SLOPE_EPS   1e-1f
            #define KPSIMULATOR_FLUX_SLOPE_EPS_4 1.0e-4f
        #else
            #define KPSIMULATOR_FLUX_SLOPE_EPS   1.0e-4f
            #define KPSIMULATOR_FLUX_SLOPE_EPS_4 1.0e-16f
        #endif
        defines = {'block_width': block_width, 'block_height': block_height,
                   'KPSIMULATOR_FLUX_SLOPE_EPS': str(flux_slope_eps)+'f',
                   'KPSIMULATOR_FLUX_SLOPE_EPS_4': str(flux_slope_eps**4)+'f',
                   'KPSIMULATOR_DEPTH_CUTOFF': str(depth_cutoff)+'f',
                   'FLUX_BALANCER': "{:.12f}f".format(flux_balancer)
                }
     
        
        #Get kernels
        self.kp07_kernel = gpu_ctx.get_kernel("KP07_kernel.cu", 
                defines=defines,
                compile_args={                          # default, fast_math, optimal
                    'options' : ["--ftz=true",          # false,   true,      true
                                 "--prec-div=false",    # true,    false,     false,
                                 "--prec-sqrt=false",   # true,    false,     false
                                 "--fmad=false"]        # true,    true,      false
                    
                    #'options': ["--use_fast_math"]
                    #'options': ["--generate-line-info"], 
                    #nvcc_options=["--maxrregcount=39"],
                    #'arch': "compute_50", 
                    #'code': "sm_50"
                },
                jit_compile_args={
                    #jit_options=[(cuda.jit_option.MAX_REGISTERS, 39)]
                }
                )
        
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.swe_2D = self.kp07_kernel.get_function("swe_2D")
        self.swe_2D.prepare("iifffffffffiPiPiPiPiPiPiPiPiiiiif")
        self.update_wind_stress(self.kp07_kernel, self.swe_2D)
        
        
        # Upload Bathymetry
        self.bathymetry = Common.Bathymetry(self.gpu_ctx, self.gpu_stream, \
                                            nx, ny, ghost_cells_x, ghost_cells_y, H, boundary_conditions)
       
        # Adjust eta for possible dry states
        Hm = self.downloadBathymetry()[1]
        eta0 = np.maximum(eta0, -Hm)
        
        #Create data by uploading to device    
        self.gpu_data = Common.SWEDataArakawaA(self.gpu_stream, nx, ny, ghost_cells_x, ghost_cells_y, eta0, hu0, hv0)
        
         
        self.bc_kernel = Common.BoundaryConditionsArakawaA(gpu_ctx, \
                                                           self.nx, \
                                                           self.ny, \
                                                           ghost_cells_x, \
                                                           ghost_cells_y, \
                                                           self.boundary_conditions)
        
        if self.write_netcdf:
            self.sim_writer = SimWriter.SimNetCDFWriter(self, ignore_ghostcells=self.ignore_ghostcells, \
                                    offset_x=self.offset_x, offset_y=self.offset_y)
            
    @classmethod
    def fromfilename(cls, gpu_ctx, filename, cont_write_netcdf=True):
        """
        Initialize and hotstart simulation from nc-file.
        cont_write_netcdf: Continue to write the results after each superstep to a new netCDF file
        filename: Continue simulation based on parameters and last timestep in this file
        """
        # open nc-file
        sim_reader = SimReader.SimNetCDFReader(filename, ignore_ghostcells=False)
        sim_name = str(sim_reader.get('simulator_short'))
        assert sim_name == cls.__name__, \
               "Trying to initialize a " + \
               cls.__name__ + " simulator with netCDF file based on " \
               + sim_name + " results."
        
        # read parameters
        nx = sim_reader.get("nx")
        ny = sim_reader.get("ny")

        dx = sim_reader.get("dx")
        dy = sim_reader.get("dy")

        width = nx * dx
        height = ny * dy

        dt = sim_reader.get("dt")
        g = sim_reader.get("g")
        r = sim_reader.get("bottom_friction_r")
        f = sim_reader.get("coriolis_force")
        beta = sim_reader.get("coriolis_beta")
        
        minmodTheta = sim_reader.get("minmod_theta")
        timeIntegrator = sim_reader.get("time_integrator")
        if (timeIntegrator == 2):
            using_rk2 = True
        else:
            using_rk2 = False 
        y_zero_reference_cell = sim_reader.get("y_zero_reference_cell")        
        
        wind = WindStress.WindStress()

        boundaryConditions = sim_reader.getBC()

        H = sim_reader.getH()
        
        # get last timestep (including simulation time of last timestep)
        eta0, hu0, hv0, time0 = sim_reader.getLastTimeStep()
        
        return cls(gpu_ctx, \
                 eta0, H, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 t=time0, \
                 theta=minmodTheta, use_rk2=using_rk2, \
                 coriolis_beta=beta, \
                 y_zero_reference_cell = y_zero_reference_cell, \
                 wind=wind, \
                 boundary_conditions=boundaryConditions, \
                 write_netcdf=cont_write_netcdf)

    def cleanUp(self):
        """
        Clean up function
        """
        self.closeNetCDF()
        
        self.gpu_data.release()
        
        self.bathymetry.release()
        
        self.gpu_ctx = None
        gc.collect()
        
    def step(self, t_end=0.0):
        """
        Function which steps n timesteps
        """
        n = int(t_end / self.dt + 1)

        if self.t == 0:
            self.bc_kernel.boundaryCondition(self.gpu_stream, \
                    self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0)
        
        for i in range(0, n):        
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            

            wind_stress_t = np.float32(self.update_wind_stress(self.kp07_kernel, self.swe_2D))
            
            if (local_dt <= 0.0):
                break
        
            if (self.use_rk2):
                self.swe_2D.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.coriolis_beta, \
                        self.y_zero_reference_cell, \
                        self.r, \
                        np.int32(0), \
                        self.gpu_data.h0.data.gpudata,  self.gpu_data.h0.pitch,  \
                        self.gpu_data.hu0.data.gpudata, self.gpu_data.hu0.pitch, \
                        self.gpu_data.hv0.data.gpudata, self.gpu_data.hv0.pitch, \
                        self.gpu_data.h1.data.gpudata,  self.gpu_data.h1.pitch,  \
                        self.gpu_data.hu1.data.gpudata, self.gpu_data.hu1.pitch, \
                        self.gpu_data.hv1.data.gpudata, self.gpu_data.hv1.pitch, \
                        self.bathymetry.Bi.data.gpudata, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data.gpudata, self.bathymetry.Bm.pitch, \
                        self.boundary_conditions.north, self.boundary_conditions.east, self.boundary_conditions.south, self.boundary_conditions.west, \
                        wind_stress_t)
                
                self.bc_kernel.boundaryCondition(self.gpu_stream, \
                        self.gpu_data.h1, self.gpu_data.hu1, self.gpu_data.hv1)
                
                self.swe_2D.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.coriolis_beta, \
                        self.y_zero_reference_cell, \
                        self.r, \
                        np.int32(1), \
                        self.gpu_data.h1.data.gpudata,  self.gpu_data.h1.pitch,  \
                        self.gpu_data.hu1.data.gpudata, self.gpu_data.hu1.pitch, \
                        self.gpu_data.hv1.data.gpudata, self.gpu_data.hv1.pitch, \
                        self.gpu_data.h0.data.gpudata,  self.gpu_data.h0.pitch,  \
                        self.gpu_data.hu0.data.gpudata, self.gpu_data.hu0.pitch, \
                        self.gpu_data.hv0.data.gpudata, self.gpu_data.hv0.pitch, \
                        self.bathymetry.Bi.data.gpudata, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data.gpudata, self.bathymetry.Bm.pitch, \
                        self.boundary_conditions.north, self.boundary_conditions.east, self.boundary_conditions.south, self.boundary_conditions.west, \
                        wind_stress_t)
                
                self.bc_kernel.boundaryCondition(self.gpu_stream, \
                        self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0) 
            else:
                self.swe_2D.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.coriolis_beta, \
                        self.y_zero_reference_cell, \
                        self.r, \
                        np.int32(0), \
                        self.gpu_data.h0.data.gpudata,  self.gpu_data.h0.pitch,  \
                        self.gpu_data.hu0.data.gpudata, self.gpu_data.hu0.pitch, \
                        self.gpu_data.hv0.data.gpudata, self.gpu_data.hv0.pitch, \
                        self.gpu_data.h1.data.gpudata,  self.gpu_data.h1.pitch,  \
                        self.gpu_data.hu1.data.gpudata, self.gpu_data.hu1.pitch, \
                        self.gpu_data.hv1.data.gpudata, self.gpu_data.hv1.pitch, \
                        self.bathymetry.Bi.data.gpudata, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data.gpudata, self.bathymetry.Bm.pitch, \
                        self.boundary_conditions.north, self.boundary_conditions.east, self.boundary_conditions.south, self.boundary_conditions.west, \
                        wind_stress_t)
                self.gpu_data.swap()
                self.bc_kernel.boundaryCondition(self.gpu_stream, \
                        self.gpu_data.h0, self.gpu_data.hu0, self.gpu_data.hv0)
                
            self.t += np.float64(local_dt)
            self.num_iterations += 1
            
        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)
            
        return self.t
    
    
    def downloadBathymetry(self, interior_domain_only=False):
        Bi, Bm = self.bathymetry.download(self.gpu_stream)
        
        if interior_domain_only:
            Bi = Bi[self.interior_domain_indices[2]:self.interior_domain_indices[0]+1,  
               self.interior_domain_indices[3]:self.interior_domain_indices[1]+1] 
            Bm = Bm[self.interior_domain_indices[2]:self.interior_domain_indices[0],  
               self.interior_domain_indices[3]:self.interior_domain_indices[1]]
               
        return [Bi, Bm]