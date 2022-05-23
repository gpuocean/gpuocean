/*
This software is part of GPU Ocean. 

Copyright (C) 2018-2019  SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

These CUDA kernels implement boundary conditions for 
the Centered in Time, Centered in Space(leapfrog)
numerical scheme for the shallow water equations, described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5.


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
*/

#include "common.cu"

// Boundary conditions are defined as
// 1: Closed wall
// 2: Periodic
// 3: Open (Flow Relaxation Scheme)
// 4: Open (Linear interpolation)
        
// Fix north-south boundary before east-west (to get the corners right)
extern "C" {
__global__ void boundaryEtaKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,
        int bc_east_, int bc_west_,

        // Data
        float* eta_ptr_, int eta_pitch_) {
    // Global thread sizes:
    // ti = {0, 3} 
    // thread 0 is index 0, thread 1 is index ny_+1, thread 2 and 3 idle
    // tj = {0, ny_+1},
    
    
    // Index of cell within domain
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;
    const int ti = (thread_id == 1) ? nx_ + 1 : thread_id;

    int opposite_col_index = nx_;
    if ( (ti == nx_+1 && bc_east_ == 2) || (ti == 0 && bc_west_ == 1) ) {
        opposite_col_index = 1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if (((ti == 0     && bc_west_ < 3) ||
         (ti == nx_+1 && bc_east_ < 3)    ) &&  tj < ny_+2) {
        float* eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj);
        eta_row[ti] = eta_row[opposite_col_index];
    }
}
} // extern "C"



extern "C" {
__global__ void boundaryUKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
        int bc_east_, int bc_west_,
    
        // Data
        float* U_ptr_, int U_pitch_) {
    // Global thread sizes:
    // ti = {0, 3} 
    // thread 0 is index 0, thread 1 is index 1,
    // thread 2 is nx+1 and thread 3 is nx+2
    // tj = {0, ny_+1},
    
    // Index of cell within domain
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;
    const int ti = (thread_id > 1) ? nx_ - 1 + thread_id : thread_id;

    
    // Check if thread is in the domain:
    if (ti <= nx_+2 && tj <= ny_+1) {   
        float* u_row = (float*) ((char*) U_ptr_ + U_pitch_*tj);

        if ( (ti < 2 && bc_west_ == 1) || (ti > nx_ && bc_east_ == 1) ) {
            if (ti==0) {
                u_row[ti] = -u_row[2];
            }
            else if (ti==nx_+2) {
                u_row[ti] = -u_row[nx_];
            }
            else{
                u_row[ti] = 0;
            }
        }
        else if (bc_west_ == 2) { // bc_east is then automatically also 2
            // Periodic
            int opposite_col_index = nx_;
            if (ti > nx_) {
                opposite_col_index = ti - nx_;
            }
            
            // We should have computed both u_row[1] and u_row[nx_+1],
            // and these two should already have the same values.
            if ( ti == 0 || ti > nx_) {
                u_row[ti] = u_row[opposite_col_index];
            }
        }
    }
}
} // extern "C"


extern "C" {
__global__ void boundaryVKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
        int bc_east_, int bc_west_,

        // Data
        float* V_ptr_, int V_pitch_) {
    // Global thread sizes:
    // ti = {0, 3} 
    // thread 0 is index 0, thread 1 is index ny_+1, thread 2 and 3 idle
    // tj = {0, ny_+1},
    
    // Index of cell within domain
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;
    const int ti = (thread_id == 1) ? nx_ + 1 : thread_id;

    // Check if thread is in the domain:
    if (ti <= nx_+1 && tj <= ny_+2) {   
        float* v_row = (float*) ((char*) V_ptr_ + V_pitch_*tj);

        int opposite_col_index = nx_;
        if ( (ti == nx_+1 && bc_east_ == 2) || (ti == 0 && bc_west_ == 1) ) {
             opposite_col_index = 1;
        }
        
        // We should have computed both u_row[1] and u_row[nx_+1],
        // and these two should already have the same values.
         if ( (ti == 0 && bc_west_ < 3) || (ti == nx_+1 && bc_east_ < 3) )   {

            v_row[ti] = v_row[opposite_col_index];
        }
    }
}
} // extern "C" 
