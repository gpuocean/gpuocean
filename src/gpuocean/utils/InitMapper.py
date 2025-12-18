
# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2024, 2025 SINTEF Digital

This python module implements mapping of initial conditions from
one domain to another. The domains can have different orientations and
resolutions.

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
import time, copy

from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import scipy.interpolate.ndgriddata as ndgriddata

from gpuocean.utils import Common



def mapArgsFromBarentsToNorkyst(barents_args, norkyst_args, 
                                fill_land_linear=True,
                                map_momentum=False, 
                                norkyst_weight=0.0, norkyst_weight_eta=None,
                                map_bc=True, verbose=True, barents_is_anomalies=False):
    
    assert(len(barents_args["wind"].t) == len(norkyst_args["wind"].t))
    start_time = time.time()

    if norkyst_weight_eta is None:
        norkyst_weight_eta = norkyst_weight

    new_args = copy.deepcopy(norkyst_args)

    # map eta0 
    if verbose: print("mapArgsFromBarentsToNorkyst: %.2f s --> mapping eta..." % (time.time() - start_time))
    barents_eta0_filled = fill_landmask(barents_args["eta0"], linear=fill_land_linear)    
    barents_eta0_in_norkyst = mapBarentsToNorkyst(barents_eta0_filled, barents_args, norkyst_args)
    barents_eta0_in_norkyst = np.ma.MaskedArray(barents_eta0_in_norkyst, norkyst_args["eta0"].mask)

    if barents_is_anomalies:
        new_args["eta0"] += barents_eta0_in_norkyst
    else:
        new_args["eta0"] = barents_eta0_in_norkyst*(1.0 - norkyst_weight_eta) + norkyst_args["eta0"]*norkyst_weight_eta

    # map field (hu0, hv0)
    # Map (u, v) by default, then adjust to norkyst depth to get (hu, hv)
    if verbose: print("mapArgsFromBarentsToNorkyst: %.2f s --> mapping (hu, hv)..." % (time.time() - start_time))
    if map_momentum:
        barents_hu_filled = fill_landmask(barents_args["hu0"], linear=fill_land_linear)
        barents_hv_filled = fill_landmask(barents_args["hv0"], linear=fill_land_linear)
        barents_hu_in_norkyst, barents_hv_in_norkyst = rotateBarentsIntoNorkyst(barents_hu_filled, barents_hv_filled,
                                                                                barents_args, norkyst_args)
        
    else:
        barents_u = barents_args["hu0"]/(barents_args["eta0"] + barents_args["Hm"])
        barents_v = barents_args["hv0"]/(barents_args["eta0"] + barents_args["Hm"])
        barents_u_filled = fill_landmask(barents_u, linear=fill_land_linear)
        barents_v_filled = fill_landmask(barents_v, linear=fill_land_linear)
        barents_u_in_norkyst, barents_v_in_norkyst = rotateBarentsIntoNorkyst(barents_u_filled, barents_v_filled,
                                                                            barents_args, norkyst_args)
        barents_hu_in_norkyst = barents_u_in_norkyst*(new_args["eta0"] + new_args["Hm"])
        barents_hv_in_norkyst = barents_v_in_norkyst*(new_args["eta0"] + new_args["Hm"])
    barents_hu_in_norkyst = np.ma.MaskedArray(barents_hu_in_norkyst, norkyst_args["hu0"].mask)
    barents_hv_in_norkyst = np.ma.MaskedArray(barents_hv_in_norkyst, norkyst_args["hv0"].mask)

    if barents_is_anomalies:
        new_args["hu0"] += barents_hu_in_norkyst 
        new_args["hv0"] += barents_hv_in_norkyst 
    else:
        new_args["hu0"] = barents_hu_in_norkyst*(1.0 - norkyst_weight) + norkyst_args["hu0"]*norkyst_weight
        new_args["hv0"] = barents_hv_in_norkyst*(1.0 - norkyst_weight) + norkyst_args["hv0"]*norkyst_weight

    # map wind
    if verbose: print("mapArgsFromBarentsToNorkyst: %.2f s --> mapping wind..." % (time.time() - start_time))

    for t in range(len(barents_args["wind"].t)):
        wind_u_filled = fill_landmask(barents_args["wind"].wind_u[t], 
                                      eta_mask=barents_args["eta0"].mask,
                                      linear=fill_land_linear)
        wind_v_filled = fill_landmask(barents_args["wind"].wind_v[t], 
                                      eta_mask=barents_args["eta0"].mask,
                                      linear=fill_land_linear)
        wind_u_mapped, wind_v_mapped = rotateBarentsIntoNorkyst(wind_u_filled, wind_v_filled,
                                                                barents_args, norkyst_args)
        new_args["wind"].wind_u[t] = wind_u_mapped
        new_args["wind"].wind_v[t] = wind_v_mapped
    
    # Map boundary conditions
    if map_bc:
        if verbose: print("mapArgsFromBarentsToNorkyst: %.2f s --> mapping boundary conditions..." % (time.time() - start_time))

        assert("boundary_conditions_meta" in norkyst_args.keys())
        assert("boundary_conditions_entire_domain" in barents_args.keys())
        assert(norkyst_args["boundary_conditions_meta"] is not None)
        assert(barents_args["boundary_conditions_entire_domain"] is not None)

        nt = len(norkyst_args['boundary_conditions_data'].t)

        bc_eta = {}
        bc_eta['north'] = np.copy(norkyst_args['boundary_conditions_data'].north.h)
        bc_eta['south'] = np.copy(norkyst_args['boundary_conditions_data'].south.h)
        bc_eta['east'] = np.copy(norkyst_args['boundary_conditions_data'].east.h)
        bc_eta['west'] = np.copy(norkyst_args['boundary_conditions_data'].west.h)

        bc_hu = {}
        bc_hu['north'] = np.copy(norkyst_args['boundary_conditions_data'].north.hu)
        bc_hu['south'] = np.copy(norkyst_args['boundary_conditions_data'].south.hu)
        bc_hu['east'] = np.copy(norkyst_args['boundary_conditions_data'].east.hu)
        bc_hu['west'] = np.copy(norkyst_args['boundary_conditions_data'].west.hu)

        bc_hv = {}
        bc_hv['north'] = np.copy(norkyst_args['boundary_conditions_data'].north.hv)
        bc_hv['south'] = np.copy(norkyst_args['boundary_conditions_data'].south.hv)
        bc_hv['east'] = np.copy(norkyst_args['boundary_conditions_data'].east.hv)
        bc_hv['west'] = np.copy(norkyst_args['boundary_conditions_data'].west.hv)


        angle_diffs = {}
        for direction in ['north', 'south', 'east', 'west']:
            angle_barents = mapBarentsToNorkyst_direct(barents_args["angle"], 
                                                       barents_args["lon"], barents_args["lat"],
                                                       norkyst_args["boundary_conditions_meta"]["lon"][direction], 
                                                       norkyst_args["boundary_conditions_meta"]["lat"][direction])
            angle_diffs[direction] = angle_barents - norkyst_args["boundary_conditions_meta"]["angle"][direction]

        for t in range(nt):
            if verbose: print("mapArgsFromBarentsToNorkyst: %.2f s --> bc it %i" % ((time.time() - start_time), t))

            barents_eta_filled = fill_landmask(barents_args["boundary_conditions_entire_domain"]["eta"][t], 
                                                eta_mask=barents_args["eta0"].mask,
                                                linear=fill_land_linear)    
            for direction in ['north', 'south', 'east', 'west']:
                barents_eta_in_norkyst =  mapBarentsToNorkyst_direct(barents_eta_filled, 
                                                                     barents_args["lon"], barents_args["lat"],
                                                                     norkyst_args["boundary_conditions_meta"]["lon"][direction],
                                                                     norkyst_args["boundary_conditions_meta"]["lat"][direction])
                
                # barents_eta0_in_norkyst[bc_eta[direction][t] == 0.0] = 0.0
                bc_eta[direction][t] = barents_eta_in_norkyst*(1.0 - norkyst_weight_eta) + bc_eta[direction][t]*norkyst_weight_eta

            # Same for direction...
            # if True: #map_momentum:
            barents_hu_filled = fill_landmask(barents_args["boundary_conditions_entire_domain"]["hu"][t], 
                                                eta_mask=barents_args["eta0"].mask,
                                                linear=fill_land_linear)
            barents_hv_filled = fill_landmask(barents_args["boundary_conditions_entire_domain"]["hv"][t], 
                                                eta_mask=barents_args["eta0"].mask,
                                                linear=fill_land_linear)
            
            for direction in ['north', 'south', 'east', 'west']:

                # Always map (hu, hv) directly here
                barents_hu_in_norkyst = mapBarentsToNorkyst_direct(barents_hu_filled,
                                                                   barents_args["lon"], barents_args["lat"], 
                                                                   norkyst_args["boundary_conditions_meta"]["lon"][direction], 
                                                                   norkyst_args["boundary_conditions_meta"]["lat"][direction])
                                                                                               
                barents_hv_in_norkyst = mapBarentsToNorkyst_direct(barents_hv_filled,
                                                                   barents_args["lon"], barents_args["lat"], 
                                                                   norkyst_args["boundary_conditions_meta"]["lon"][direction], 
                                                                   norkyst_args["boundary_conditions_meta"]["lat"][direction])
                barents_hu_in_norkyst_rotated = barents_hu_in_norkyst*np.cos(angle_diffs[direction]) - barents_hv_in_norkyst*np.sin(angle_diffs[direction])                                                  
                barents_hv_in_norkyst_rotated = barents_hu_in_norkyst*np.sin(angle_diffs[direction]) + barents_hv_in_norkyst*np.cos(angle_diffs[direction])                                                  

                bc_hu[direction][t] = barents_hu_in_norkyst_rotated*(1.0 - norkyst_weight) + bc_hu[direction][t]*norkyst_weight
                bc_hv[direction][t] = barents_hv_in_norkyst_rotated*(1.0 - norkyst_weight) + bc_hv[direction][t]*norkyst_weight

            # else:
                # barents_u = barents_args["hu0"]/(barents_args["eta0"] + barents_args["Hm"])
                # barents_v = barents_args["hv0"]/(barents_args["eta0"] + barents_args["Hm"])
                # barents_u_filled = fill_landmask(barents_u, linear=fill_land_linear)
                # barents_v_filled = fill_landmask(barents_v, linear=fill_land_linear)
                # barents_u_in_norkyst, barents_v_in_norkyst = rotateBarentsIntoNorkyst(barents_u_filled, barents_v_filled,
                #                                                                     barents_args, norkyst_args)
                # barents_hu_in_norkyst = barents_u_in_norkyst*(new_args["eta0"] + new_args["Hm"])
            #     # barents_hv_in_norkyst = barents_v_in_norkyst*(new_args["eta0"] + new_args["Hm"])
            # barents_hu_in_norkyst = np.ma.MaskedArray(barents_hu_in_norkyst, norkyst_args["hu0"].mask)
            # barents_hv_in_norkyst = np.ma.MaskedArray(barents_hv_in_norkyst, norkyst_args["hv0"].mask)

                
    
        new_args["boundary_conditions_data"] = Common.BoundaryConditionsData(
            norkyst_args['boundary_conditions_data'].t.copy(), 
            north=Common.SingleBoundaryConditionData(bc_eta['north'], bc_hu['north'], bc_hv['north']),
            south=Common.SingleBoundaryConditionData(bc_eta['south'], bc_hu['south'], bc_hv['south']),
            east=Common.SingleBoundaryConditionData(bc_eta['east'], bc_hu['east'], bc_hv['east']),
            west=Common.SingleBoundaryConditionData(bc_eta['west'], bc_hu['west'], bc_hv['west']))


    return new_args

def mapWindFromBarentsToNorkyst(barents_args, norkyst_args,
                                fill_land_linear=True,
                                verbose=False):
    
    assert(len(barents_args["wind"].t) == len(norkyst_args["wind"].t))
    start_time = time.time()

    new_args = copy.deepcopy(norkyst_args)

    
    for t in range(len(barents_args["wind"].t)):
        wind_u_filled = fill_landmask(barents_args["wind"].wind_u[t], 
                                      eta_mask=barents_args["eta0"].mask,
                                      linear=fill_land_linear)
        wind_v_filled = fill_landmask(barents_args["wind"].wind_v[t], 
                                      eta_mask=barents_args["eta0"].mask,
                                      linear=fill_land_linear)
        wind_u_mapped, wind_v_mapped = rotateBarentsIntoNorkyst(wind_u_filled, wind_v_filled,
                                                                barents_args, norkyst_args)
        new_args["wind"].wind_u[t] = wind_u_mapped
        new_args["wind"].wind_v[t] = wind_v_mapped

    if verbose: print("mapArgsFromBarentsToNorkyst: %.2f s --> Done mapping wind..." % (time.time() - start_time))

    return new_args

### Utility functions


def fill_landmask(data, eta_mask=None, linear=True):
    if eta_mask is None:
        mask = np.where(~data.mask)
    else:
        mask = np.where(~eta_mask)
    
    if linear:
        interpolator = LinearNDInterpolator(np.transpose(mask), data[mask])
    else:
        interpolator = NearestNDInterpolator(np.transpose(mask), data[mask])
    
    data_filled = interpolator(*np.indices(data.shape))
    return data_filled 

def mapBarentsToNorkyst(data, barents_args, norkyst_args):
    return ndgriddata.griddata((barents_args["lon"].flatten(), barents_args["lat"].flatten()), 
                               data.flatten(), 
                               (norkyst_args["lon"], norkyst_args["lat"]), 
                               method="linear")

def mapBarentsToNorkyst_direct(data, barents_lon, barents_lat, norkyst_lon, norkyst_lat):
    return ndgriddata.griddata((barents_lon.flatten(), barents_lat.flatten()), 
                               data.flatten(), 
                               (norkyst_lon, norkyst_lat), 
                               method="linear")



def rotateBarentsIntoNorkyst(u_data, v_data, barents_args, norkyst_args):
    u = mapBarentsToNorkyst(u_data, barents_args, norkyst_args)
    v = mapBarentsToNorkyst(v_data, barents_args, norkyst_args)
    angle_barents = mapBarentsToNorkyst(barents_args["angle"], barents_args, norkyst_args)
    angle_diff = angle_barents - norkyst_args["angle"]

    u_corrected = u*np.cos(angle_diff) - v*np.sin(angle_diff)
    v_corrected = u*np.sin(angle_diff) + v*np.cos(angle_diff)
    return u_corrected, v_corrected

def rotateBarentsIntoNorkyst_direct(u_data, v_data, barents_lon, barents_lat, norkyst_lon, norkyst_lat, barents_angle, norkyst_angle):
    u = mapBarentsToNorkyst_direct(u_data, barents_lon, barents_lat, norkyst_lon, norkyst_lat)
    v = mapBarentsToNorkyst_direct(v_data, barents_lon, barents_lat, norkyst_lon, norkyst_lat)
    angle_barents = mapBarentsToNorkyst_direct(barents_angle, barents_lon, barents_lat, norkyst_lon, norkyst_lat)
    angle_diff = angle_barents - norkyst_angle

    u_corrected = u*np.cos(angle_diff) - v*np.sin(angle_diff)
    v_corrected = u*np.sin(angle_diff) + v*np.cos(angle_diff)
    return u_corrected, v_corrected

