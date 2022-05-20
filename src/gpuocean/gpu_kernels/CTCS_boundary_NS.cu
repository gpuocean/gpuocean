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
__global__ void boundaryEtaKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,
        int bc_north_, int bc_south_,
    
        // Data
        float* eta_ptr_, int eta_pitch_) {
    // Global thread sizes:
    // ti = {0, nx_+1}
    // tj = {0, 3},
    // thread 0 is index 0, thread 1 is index ny_+1, thread 2 and 3 idle
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int tj = (thread_id == 1) ? ny_+1 : thread_id;
    
    int opposite_row_index = ny_;
    //if (tj == ny_+1) {
    if ( (tj == ny_+1 && bc_north_ == 2) || (tj == 0 && bc_south_ == 1) ) {
        opposite_row_index = 1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if (((tj == 0     && bc_south_ < 3)  ||
         (tj == ny_+1 && bc_north_ < 3)) &&
        ti > -1 && ti < nx_+2) {
        float* ghost_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj);
        float* opposite_row = (float*) ((char*) eta_ptr_ + eta_pitch_*opposite_row_index);
        ghost_row[ti] = opposite_row[ti];
    }
}
} // extern "C"


// NS need to be called before EW!
extern "C" {
__global__ void boundaryUKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
        int bc_north_, int bc_south_,

        // Data
        float* U_ptr_, int U_pitch_) {
    // Global thread sizes:
    // ti = {0, nx_+1}
    // tj = {0, 3},
    // thread 0 is index 0, thread 1 is index ny_+1, thread 2 and 3 idle
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int tj = (thread_id == 1) ? ny_+1 : thread_id;
    
    // Check if thread is in the domain:
    if (ti <= nx_+2 && tj <= ny_+1) {
        // The thread's row:
        float* u_row = (float*) ((char*) U_ptr_ + U_pitch_*tj);

        int opposite_row_index = ny_;
        if ( (tj == ny_+1 && bc_north_ == 2) || (tj == 0 && bc_south_ == 1) ) {
            opposite_row_index = 1;
        }
        
        if ( ((tj == 0     && bc_south_ < 3)  ||
              (tj == ny_+1 && bc_north_ < 3)) &&
              ti < nx_+3) {
              //ti > 0 && ti < nx_+1 ) {
            float* u_opposite_row = (float*) ((char*) U_ptr_ + U_pitch_*opposite_row_index);
            u_row[ti] = u_opposite_row[ti];
        }
    } 
}
} // extern "C"


// NS need to be called before EW!
extern "C" {
__global__ void boundaryVKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
        int bc_north_, int bc_south_,

        // Data
        float* V_ptr_, int V_pitch_) {
    // Global thread sizes:
    // tj = {0, 3} 
    // thread 0 is index 0, thread 1 is index 1,
    // thread 2 is ny+1 and thread 3 is ny+2
    // ti = {0, nx_+1},
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int tj = (thread_id > 1) ? ny_ - 1 + thread_id : thread_id;


    // Check if thread is in the domain:
    if (ti <= nx_+1 && tj <= ny_+2) {   
        float* v_row = (float*) ((char*) V_ptr_ + V_pitch_*tj);

        
        if ( (tj < 2 && bc_south_ == 1 ) || (tj > ny_ && bc_north_ == 1) ) {
            if (tj == 0) {
                float* v_row_inner = (float*) ((char*) V_ptr_ + V_pitch_*2);
                v_row[ti] = -v_row_inner[ti];
            }
            else if (tj == ny_+2) {
                float* v_row_inner = (float*) ((char*) V_ptr_ + V_pitch_*ny_);
                v_row[ti] = -v_row_inner[ti];
            }
            else{
                v_row[ti] = 0.0f;
            }
        }
        else if (bc_north_ == 2) { // implicit bc_south_ == 2
            // Periodic
            int opposite_row_index = ny_;
            if (tj == ny_+2) {
                opposite_row_index = 2; // 2 and ny_+1 -> 1
            }
            if (tj == ny_+1) {
                opposite_row_index = 1;
            }
            
            if ( (tj == 0 || tj > ny_) && ti < nx_+2 ) {
                float* v_opposite_row = (float*) ((char*) V_ptr_ + V_pitch_*opposite_row_index);
                v_row[ti] = v_opposite_row[ti];
            }
        }
    }
}
} // extern "C"

