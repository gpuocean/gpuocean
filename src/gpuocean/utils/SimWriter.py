# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2016 SINTEF ICT, 
Copyright (C) 2017-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

This python module implements saving shallow water simulations to a
netcdf file.

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
import datetime
from netCDF4 import Dataset
import subprocess
import matplotlib.pyplot as plt
import os as os
import time

class SimNetCDFWriter:
    """Write simulator output to file in netCDF-format, following the CF convention.

    Args:
        sim: Simulator that will be generating netCDF output.
        filename: Use this filename. (If not defined, a filename will be generated.)
        num_layers: Number of layers in sim.
        staggerede_grid: Is simulator grid staggered.
        ignore_ghostcells: Ghost cells will not be written to file if set to True.
        offset_x: Offset simulator origo with offset_x*dx in x-dimension, before writing to netCDF. 
        offset_y: Offset simulator origo with offset_y*dy in y-dimension, before writing to netCDF.
    """
    def __init__(self, sim, super_dir_name=None, filename=None, num_layers=1, staggered_grid=False, \
                 ignore_ghostcells=False, \
                 offset_x=0, offset_y=0):

        # Parallel netCDF4 write?
        # TODO: Implement check/test for feature or take as an argument
        self.write_parallel = False
        
        # OpenCL queue:
        self.gpu_stream = sim.gpu_stream

        # Write options:
        self.ignore_ghostcells = ignore_ghostcells
        self.staggered_grid = staggered_grid
        self.num_layers = num_layers

        # Identification of solution
        self.timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.timestamp_short = datetime.datetime.now().strftime("%Y_%m_%d")
        try:
            self.git_hash = str.strip(str(subprocess.check_output(['git', 'rev-parse', 'HEAD'])))
        except:
            self.git_hash = "git info missing..."

        self.simulator_short = str(sim.__class__.__name__)

        if super_dir_name is not None:
            self.super_dir_name = super_dir_name
        else:
            self.super_dir_name = os.getcwd()
        
        if not filename:
            self.dir_name = "netcdf_" + self.timestamp_short
            self.dir_name = os.path.join(self.super_dir_name, self.dir_name)
            self.output_file_name = os.path.join(self.dir_name, self.simulator_short + "_" + self.timestamp + ".nc")
        elif os.path.split(filename)[0] == "":
            self.dir_name = "netcdf_" + self.timestamp_short
            self.dir_name = os.path.join(self.super_dir_name, self.dir_name)
            self.output_file_name = os.path.join(self.dir_name, self.simulator_short + "_" + filename + ".nc")
        else:
            # Input file name is given as 'folder/file' 
            self.output_file_name = filename + ".nc"
            self.dir_name = os.path.dirname(filename)
            
        # Avoid filenames such as "file.nc.nc"
        self.output_file_name = self.output_file_name.replace('.nc.nc', '.nc')
        
        self.current_directory =os.getcwd()
        self.textPos = -1
        
        # Simulator info
        self.boundary_conditions = str(sim.boundary_conditions)
        # (machine readable BC)
        self.boundary_conditions_sponge_mr = str(sim.boundary_conditions.getSponge())
        
        self.dt = sim.dt
        if self.staggered_grid:
            self.Hm = sim.H.download(self.gpu_stream)
        else:
            self.Hi, self.Hm = sim.bathymetry.download(self.gpu_stream) # Hi
        
        if (sim.__class__.__name__ == "KP07"):
            if (sim.use_rk2):
                self.time_integrator = 2
            else:
                self.time_integrator = 1
        else:
            self.time_integrator = sim.rk_order
        
        self.minmod_theta = sim.theta
        self.coriolis_force = sim.f
        self.coriolis_beta = sim.coriolis_beta
        self.y_zero_reference_cell = sim.y_zero_reference_cell-2
        self.wind_stress = sim.wind_stress.source_filename
        self.eddy_viscosity_coefficient = sim.A
        g = sim.g
        nx = sim.nx
        ny = sim.ny
        dx = sim.dx
        dy = sim.dy
        dt = sim.dt
        self.dx = dx
        self.dy = dy
        auto_dt = False
        self.ghost_cells_east = sim.ghost_cells_x
        self.ghost_cells_west = sim.ghost_cells_x
        self.ghost_cells_north = sim.ghost_cells_y
        self.ghost_cells_south = sim.ghost_cells_y
        self.bottom_friction_r = sim.r

        self.ghost_cells_tot_y = self.ghost_cells_north + self.ghost_cells_south
        self.ghost_cells_tot_x = self.ghost_cells_east  + self.ghost_cells_west 
            
            
        # Organize directory and create file:
        if(sim.comm):
            os.makedirs(self.dir_name, exist_ok=True)
            if(self.write_parallel):
                # FIXME: Needs to be updated to handle more than one member/particle per MPI process
                self.ncfile = Dataset(self.output_file_name,'w', clobber=True, parallel=True)
            else:
                self.output_file_name = self.output_file_name.replace('.nc', '_' + str(sim.comm.rank) + '_' + str(sim.local_particle_id) + '.nc')
                self.ncfile = Dataset(self.output_file_name,'w', clobber=True, parallel=False)
        else:
            os.makedirs(self.dir_name, exist_ok=True)
            self.ncfile = Dataset(self.output_file_name,'w', clobber=True, parallel=False)
        self.ncfile.Conventions = "CF-1.4"
        
        # Write global attributes
        self.ncfile.git_hash = self.git_hash
        self.ncfile.ignore_ghostcells = str(self.ignore_ghostcells)
        self.ncfile.num_layers = self.num_layers
        self.ncfile.staggered_grid = str(self.staggered_grid)
        self.ncfile.simulator_short = self.simulator_short
        self.ncfile.boundary_conditions = self.boundary_conditions
        self.ncfile.boundary_conditions_sponge_mr = self.boundary_conditions_sponge_mr
        self.ncfile.time_integrator = self.time_integrator
        self.ncfile.minmod_theta = self.minmod_theta
        self.ncfile.coriolis_force = self.coriolis_force
        self.ncfile.coriolis_beta = self.coriolis_beta
        self.ncfile.y_zero_reference_cell = self.y_zero_reference_cell
        if self.wind_stress is not None:
            self.ncfile.wind_stress_source = self.wind_stress
        self.ncfile.eddy_viscosity_coefficient = self.eddy_viscosity_coefficient
        self.ncfile.g = g  
        self.ncfile.nx = nx
        self.ncfile.ny = ny
        self.ncfile.dx = dx
        self.ncfile.dy = dy
        self.ncfile.dt = dt
        self.ncfile.auto_dt = str(auto_dt)
        self.ncfile.bottom_friction_r = self.bottom_friction_r
        self.ncfile.ghost_cells_north = self.ghost_cells_north
        self.ncfile.ghost_cells_east  = self.ghost_cells_east
        self.ncfile.ghost_cells_south = self.ghost_cells_south
        self.ncfile.ghost_cells_west  = self.ghost_cells_west
        
        
        # Write parameters related to stochastic model error and data assimilation
        # At the time of writing, such parameters are only available in the CDKLM simulator.
        if (sim.__class__.__name__ == "CDKLM16"):
            self.model_time_step = sim.model_time_step
            self.ncfile.model_time_step = self.model_time_step 
            
            self.small_scale_perturbation = sim.small_scale_perturbation
            self.ncfile.small_scale_perturbation = str(self.small_scale_perturbation)
            
            if self.small_scale_perturbation:

                self.small_scale_perturbation_amplitude = sim.small_scale_model_error.soar_q0
                self.ncfile.small_scale_perturbation_amplitude = self.small_scale_perturbation_amplitude

                self.small_scale_perturbation_interpolation_factor = sim.small_scale_model_error.interpolation_factor
                self.ncfile.small_scale_perturbation_interpolation_factor = self.small_scale_perturbation_interpolation_factor
            
            #y_zero_reference_cell
            

        
        #Create dimensions 
        self.ncfile.createDimension('time', None) #Unlimited time dimension
        if(not self.ignore_ghostcells):
            self.ncfile.createDimension('x', nx + self.ghost_cells_tot_x)
            self.ncfile.createDimension('y', ny + self.ghost_cells_tot_y)
        else:
            self.ncfile.createDimension('x', nx)
            self.ncfile.createDimension('y', ny)
        if (not self.ignore_ghostcells) and (self.staggered_grid):
            if (self.simulator_short == 'FBL'):
                # Adjusting global domain size according to FBL scheme stencil requirements
                self.ncfile.createDimension('x_hu',   nx + self.ghost_cells_tot_x - 1)
                self.ncfile.createDimension('y_hu',   ny + self.ghost_cells_tot_y)
                self.ncfile.createDimension('x_hv',   nx + self.ghost_cells_tot_x)
                self.ncfile.createDimension('y_hv',   ny + self.ghost_cells_tot_y + 1)
            else:
                self.ncfile.createDimension('x_hu',   nx + self.ghost_cells_tot_x + 1)
                self.ncfile.createDimension('y_hu',   ny + self.ghost_cells_tot_y)
                self.ncfile.createDimension('x_hv',   nx + self.ghost_cells_tot_x)
                self.ncfile.createDimension('y_hv',   ny + self.ghost_cells_tot_y + 1)
        if not self.staggered_grid: 
            self.ncfile.createDimension('x_Hi', nx + self.ghost_cells_tot_x + 1)
            self.ncfile.createDimension('y_Hi', ny + self.ghost_cells_tot_y + 1)
        if (sim.comm and self.write_parallel):
            # FIXME: Needs to be updated to handle more than one member/particle per MPI process
            self.ncfile.createDimension('ensemble_member', sim.comm.size)
        
        #Create axis
        self.nc_time = self.ncfile.createVariable('time', np.dtype('float32').char, 'time')
        x = self.ncfile.createVariable('x', np.dtype('float32').char, 'x')
        y = self.ncfile.createVariable('y', np.dtype('float32').char, 'y')

        x.standard_name = "projection_x_coordinate"
        y.standard_name = "projection_y_coordinate"
        x.axis = "X"
        y.axis = "Y"

        if (not self.ignore_ghostcells) and (self.staggered_grid):
            x_hu = self.ncfile.createVariable('x_hu', np.dtype('float32').char, 'x_hu')
            y_hu = self.ncfile.createVariable('y_hu', np.dtype('float32').char, 'y_hu')
            x_hu.standard_name = "projection_x_coordinate"
            y_hu.standard_name = "projection_y_coordinate"
            x_hu.axis = "X"
            y_hu.axis = "Y"
            x_hv = self.ncfile.createVariable('x_hv', np.dtype('float32').char, 'x_hv')
            y_hv = self.ncfile.createVariable('y_hv', np.dtype('float32').char, 'y_hv')
            x_hv.standard_name = "projection_x_coordinate"
            y_hv.standard_name = "projection_y_coordinate"
            x_hv.axis = "X"
            y_hv.axis = "Y"
        
        if not self.staggered_grid:
            x_Hi = self.ncfile.createVariable('x_Hi', np.dtype('float32').char, 'x_Hi')
            y_Hi = self.ncfile.createVariable('y_Hi', np.dtype('float32').char, 'y_Hi')
            x_Hi.standard_name = "projection_x_coordinate"
            y_Hi.standard_name = "projection_y_coordinate"
            x_Hi.axis = "X"
            y_Hi.axis = "Y"
            
        if (sim.comm and self.write_parallel):
            # FIXME: Needs to be updated to handle more than one member/particle per MPI process
            ensemble_member = self.ncfile.createVariable('ensemble_member', np.dtype('float32').char, 'ensemble_member')
            ensemble_member.long_name = "ensemble run number"
            ensemble_member.standard_name = "realization"
            ensemble_member._CoordinateAxisType = "Ensemble"
            ensemble_member[:] = range(0, sim.rank.size)

        #Create bogus projection variable
        self.nc_proj = self.ncfile.createVariable('projection_stere', np.dtype('int32').char)
        self.nc_proj.grid_mapping_name = 'polar_stereographic'
        self.nc_proj.scale_factor_at_projection_origin = 0.9330127018922193
        self.nc_proj.straight_vertical_longitude_from_pole = 70.0
        self.nc_proj.latitude_of_projection_origin = 90.0
        self.nc_proj.earth_radius = 6371000.0
        self.nc_proj.proj4 = '+proj=stere +lat_0=90 +lon_0=70 +lat_ts=60 +units=m +a=6.371e+06 +e=0 +no_defs'

        if(not self.ignore_ghostcells):
            x[:] = np.linspace(-self.ghost_cells_west*dx + dx/2.0, \
                               (nx + self.ghost_cells_east)*dx - dx/2.0, \
                               nx + self.ghost_cells_tot_x)
            y[:] = np.linspace(-self.ghost_cells_south*dy + dy/2.0, \
                               (ny + self.ghost_cells_north)*dy - dy/2.0, \
                               ny + self.ghost_cells_tot_y)
        else:
            x[:] = np.linspace(offset_x, nx*dx, nx)
            y[:] = np.linspace(offset_y, ny*dy, ny)
            
        if not self.ignore_ghostcells and self.staggered_grid:
            if (self.simulator_short == 'FBL'):
                # Adjusting global domain size according to FBL scheme stencil requirements
                x_hu[:] = np.linspace(-self.ghost_cells_west*dx, \
                                      (nx + self.ghost_cells_east)*dx, \
                                       nx + self.ghost_cells_tot_x - 1)
                y_hu[:] = np.linspace(-self.ghost_cells_south*dy + dy/2.0, \
                                      (ny + self.ghost_cells_north)*dy + dy/2.0, \
                                       ny + self.ghost_cells_tot_y)
                x_hv[:] = np.linspace(-self.ghost_cells_west*dx + dx/2.0, \
                                      (nx + self.ghost_cells_east)*dx + dx/2.0, \
                                       nx + self.ghost_cells_tot_x)
                y_hv[:] = np.linspace(-self.ghost_cells_south*dy, \
                                      (ny + self.ghost_cells_north)*dy, \
                                       ny + self.ghost_cells_tot_y + 1)
            else:
                x_hu[:] = np.linspace(-self.ghost_cells_west*dx, \
                                      (nx + self.ghost_cells_east)*dx, \
                                       nx + self.ghost_cells_tot_x + 1)
                y_hu[:] = np.linspace(-self.ghost_cells_south*dy + dy/2.0, \
                                      (ny + self.ghost_cells_north)*dy + dy/2.0, \
                                       ny + self.ghost_cells_tot_y)
                x_hv[:] = np.linspace(-self.ghost_cells_west*dx + dx/2.0, \
                                      (nx + self.ghost_cells_east)*dx + dx/2.0, \
                                       nx + self.ghost_cells_tot_x)
                y_hv[:] = np.linspace(-self.ghost_cells_south*dy, \
                                      (ny + self.ghost_cells_north)*dy, \
                                       ny + self.ghost_cells_tot_y + 1)
        
        if not self.staggered_grid:
            x_Hi[:] = np.linspace(-self.ghost_cells_west*dx, \
                                  (nx + self.ghost_cells_east)*dx, \
                                   nx + self.ghost_cells_tot_x + 1)
            y_Hi[:] = np.linspace(-self.ghost_cells_south*dy, \
                                  (ny + self.ghost_cells_north)*dy, \
                                   ny + self.ghost_cells_tot_y + 1)
            
        #Set units
        self.nc_time.units = 'seconds since 1970-01-01 00:00:00'
        x.units = 'meter'
        y.units = 'meter'
        if not self.ignore_ghostcells and self.staggered_grid:
            x_hu.units = 'meter'
            y_hu.units = 'meter'
            x_hv.units = 'meter'
            y_hv.units = 'meter'

        if not self.staggered_grid:
            x_Hi.units = 'meter'
            y_Hi.units = 'meter'
            
        #Create a land mask (with no land)
        self.nc_land = self.ncfile.createVariable('land_binary_mask', np.dtype('float32').char, ('y', 'x'))
        self.nc_land.standard_name = 'land_binary_mask'
        self.nc_land.units = '1'
        self.nc_land[:] = 0

        # Create info about bathymetry / equilibrium depth
        self.nc_Hm = self.ncfile.createVariable('Hm', np.dtype('float32').char, ('y', 'x'), zlib=True)
        self.nc_Hm.standard_name = 'water_surface_reference_datum_altitude'
        self.nc_Hm.grid_mapping = 'projection_stere'
        self.nc_Hm.coordinates = 'y x'
        self.nc_Hm.units = 'meter'
        self.nc_Hm[:] = self.Hm
        
        if not self.staggered_grid:
            self.nc_Hi = self.ncfile.createVariable('Hi', np.dtype('float32').char, ('y_Hi', 'x_Hi'), zlib=True)
            self.nc_Hi.standard_name = 'water_surface_reference_datum_altitude'
            self.nc_Hi.grid_mapping = 'projection_stere'
            self.nc_Hi.coordinates = 'y x'
            self.nc_Hi.units = 'meter'
            self.nc_Hi[:] = self.Hi
        
        # FIXME: Needs to be updated to handle more than one member/particle per MPI process
        if(sim.comm and self.write_parallel):
            self.nc_eta = self.ncfile.createVariable('eta', np.dtype('float32').char, ('time', 'ensemble_member', 'y', 'x'), zlib=True)
            self.nc_eta.set_collective(True)
            if not self.ignore_ghostcells and self.staggered_grid:
                self.nc_hu = self.ncfile.createVariable('hu', np.dtype('float32').char, ('time', 'ensemble_member', 'y_hu', 'x_hu'), zlib=True)
                self.nc_hu.set_collective(True)
                self.nc_hv = self.ncfile.createVariable('hv', np.dtype('float32').char, ('time', 'ensemble_member', 'y_hv', 'x_hv'), zlib=True)
                self.nc_hv.set_collective(True)
            else:
                self.nc_hu = self.ncfile.createVariable('hu', np.dtype('float32').char, ('time', 'ensemble_member', 'y', 'x'), zlib=True)
                self.nc_hu.set_collective(True)
                self.nc_hv = self.ncfile.createVariable('hv', np.dtype('float32').char, ('time', 'ensemble_member', 'y', 'x'), zlib=True)
                self.nc_hv.set_collective(True)
        else:
            if(sim.comm):
                self.ncfile.ensemble_member = sim.comm.rank
            self.nc_eta = self.ncfile.createVariable('eta', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
            if not self.ignore_ghostcells and self.staggered_grid:
                self.nc_hu = self.ncfile.createVariable('hu', np.dtype('float32').char, ('time', 'y_hu', 'x_hu'), zlib=True)
                self.nc_hv = self.ncfile.createVariable('hv', np.dtype('float32').char, ('time', 'y_hv', 'x_hv'), zlib=True)
            else:
                self.nc_hu = self.ncfile.createVariable('hu', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
                self.nc_hv = self.ncfile.createVariable('hv', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
            
        self.nc_eta.standard_name = 'water_surface_height_above_reference_datum'
        self.nc_hu.standard_name = 'x_sea_water_velocity'
        self.nc_hv.standard_name = 'y_sea_water_velocity'
        self.nc_eta.grid_mapping = 'projection_stere'
        self.nc_hu.grid_mapping = 'projection_stere'
        self.nc_hv.grid_mapping = 'projection_stere'
        self.nc_eta.coordinates = 'y x'
        self.nc_hu.coordinates = 'y_hu x_hu'
        self.nc_hv.coordinates = 'y_hv x_hv'

        #Set units
        self.nc_eta.units = 'meter'
        self.nc_hu.units = 'meter second-1'
        self.nc_hv.units = 'meter second-1'
 
        # Init conditions should be added as the first element in the above arrays!
        self.i = 0
        self.writeTimestep(sim)

       
    def __str__(self):
        msg = ""
        theMap = vars(self)
        for i,j in theMap.items():
            if str(type(j)) == '<type \'str\'>':
                msg += i + ":\t\t" + j
            elif str(type(j)) == '<type \'numpy.ndarray\'>':
                msg += i + ":\t\tnumpy array of shape " + str(j.shape)
            elif str(type(j)) == '<type \'numpy.int32\'>' or \
                 str(type(j)) == '<type \'numpy.float32\'>' or \
                 str(type(j)) == '<type \'int\'>' or \
                 str(type(j)) == '<type \'bool\'>':
                msg += i + ":\t\t" + str(j)
            else:
                msg += i + ":\t\t" + str(type(j)) 
            msg += '\n'
        return msg    
        
        
    def __enter__(self):
        return self
        
        
        
        
        
    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing file " + self.output_file_name + " ...")
        self.ncfile.close()
        
        

    def writeTimestep(self, sim):
        eta, hu, hv = sim.download()
        if (self.ignore_ghostcells):
            # FIXME: Needs to be updated to handle more than one member/particle per MPI process
            if(sim.comm and self.write_parallel):
                self.nc_time[self.i] = sim.t
                self.nc_eta[self.i, sim.ensemble_member, :] = eta[1:-1, 1:-1]
                self.nc_hu[self.i, sim.ensemble_member, :] = hu[1:-1, 1:-2]
                self.nc_hv[self.i, sim.ensemble_member, :] = hv[1:-2, 1:-1]
            else:
                self.nc_time[self.i] = sim.t
                self.nc_eta[self.i, :] = eta[1:-1, 1:-1]
                self.nc_hu[self.i, :] = hu[1:-1, 1:-2]
                self.nc_hv[self.i, :] = hv[1:-2, 1:-1]
        else:
            # FIXME: Needs to be updated to handle more than one member/particle per MPI process
            if(sim.comm and self.write_parallel):
                self.nc_time[self.i] = sim.t
                self.nc_eta[self.i, sim.ensemble_member, :] = eta
                self.nc_hu[self.i, sim.ensemble_member, :] = hu
                self.nc_hv[self.i, sim.ensemble_member, :] = hv
            else:
                self.nc_time[self.i] = sim.t
                self.nc_eta[self.i, :] = eta
                self.nc_hu[self.i, :] = hu
                self.nc_hv[self.i, :] = hv
                       
        self.i += 1

            
    def write(self, t, eta, hu, hv, eta2=None, hu2=None, hv2=None):
        if (self.ignore_ghostcells):
            self.nc_time[self.i] = t
            #self.nc_eta[i, :] = eta[1:-1, 1:-1]
            #self.nc_u[i, :] = u[1:-1, 1:-1]
            #self.nc_v[i, :] = v[1:-1, 1:-1]
            self.nc_hu[self.i, :] = hu[1:-2, 1:-1]
            self.nc_hv[self.i, :] = hv[1:-1, 1:-2]
        else:
            self.nc_time[self.i] = t
            self.nc_eta[self.i, :] = eta
            self.nc_hu[self.i, :] = hu
            self.nc_hv[self.i, :] = hv
            
        if(self.num_layers == 2):
            if (self.ignore_ghostcells):
                self.nc_eta2[self.i, :] = eta2[1:-1, 1:-1]
                self.nc_hu2[self.i, :] = hu2[1:-1, 1:-1]
                self.nc_hv2[self.i, :] = hv2[1:-1, 1:-1]
            else:
                self.nc_eta2[self.i, :] = eta2
                self.nc_hu2[self.i, :] = hu2
                self.nc_hv2[self.i, :] = hv2

        self.i += 1


    def _addText(self, ax, msg):
        bp = 70 # breakpoint
        if len(msg) > bp:
            rest = '     ' + msg[bp:]
            ax.text(0.1, self.textPos, msg[0:bp])
            self.textPos -= 0.2
            self._addText(ax, rest)
        else:
            ax.text(0.1, self.textPos, msg)
            #print len(msg)
            self.textPos -= 0.2
        
    def infoPlot(self, ax):
        self.textPos = 2.3
        # Ax is the subplot object
        ax.text(1, 2.8, 'NetCDF INFO')
        
        self._addText(ax, 'working directory: ' + self.current_directory)
        self._addText(ax, 'filename: ' + self.output_file_name)
        self._addText(ax, '')
        self._addText(ax, 'git hash: ' + self.git_hash)
        self._addText(ax, '')
        self._addText(ax, 'Simulator: ' + self.simulator_short)
        self._addText(ax, 'BC: ' + str(self.boundary_conditions))
        self._addText(ax, 'f:  ' + str(self.coriolis_force))
        self._addText(ax, 'dt: ' + str(self.dt) + ", dx: " + str(self.dx) + ", dy: " + str(self.dy))
        self._addText(ax, 'wind type: ' + str(self.wind_stress))
        
        ax.axis([0, 6, 0, 3])
