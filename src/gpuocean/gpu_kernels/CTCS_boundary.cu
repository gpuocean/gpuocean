/*
These OpenCL kernels implement boundary conditions for 
the Centered in Time, Centered in Space(leapfrog)
numerical scheme for the shallow water equations, described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .

Copyright (C) 2016  SINTEF ICT

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


extern "C" {
__global__ void boundary_linearInterpol_NS(
    // Discretization parameters
    int nx_, int ny_,
    int nx_halo_, int ny_halo_,
    int staggered_x_, int staggered_y_, // U->(1,0), V->(0,1), eta->(0,0)
    
    // Boundary condition parameters
    int sponge_cells_north_, int sponge_cells_south_,
    int bc_north_, int bc_south_,
        
    // Data
    float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if (( ((bc_south_ == 4) &&
           (tj < sponge_cells_south_) && (tj > 0)) ||
          ((bc_north_ == 4) &&
           (tj > ny_ + 2*ny_halo_ + staggered_y_ - sponge_cells_north_ - 1) &&
           (tj < ny_ + 2*ny_halo_ + staggered_y_ - 1)) ) &&
        (ti > 0) && (ti < nx_ + 2*ny_halo_ + staggered_x_ -1))
    {
        // Identify inner and outer row
        int inner_row = sponge_cells_south_;
        int outer_row = 0;
        if (tj > sponge_cells_south_) {
            inner_row = ny_ + 2*ny_halo_ + staggered_y_ - sponge_cells_north_ - 1;
            outer_row = ny_ + 2*ny_halo_ + staggered_y_ - 1;
        }

        // Get outer value
        float* outer_row_ptr = (float*) ((char*) data_ptr_ + data_pitch_*outer_row);
        float outer_value = outer_row_ptr[ti];

        // Get inner value
        float* inner_row_ptr = (float*) ((char*) data_ptr_ + data_pitch_*inner_row);
        float inner_value = inner_row_ptr[ti];

        // Find target cell
        float* target_row_ptr = (float*) ((char*) data_ptr_ + data_pitch_*tj);

        // Interpolate:
        float ratio = ((float)(tj - outer_row))/(inner_row - outer_row);
        target_row_ptr[ti] = outer_value + ratio*(inner_value - outer_value);
    }
}
} // extern "C"


extern "C" {
__global__ void boundary_linearInterpol_EW(
    // Discretization parameters
    int nx_, int ny_,
    int nx_halo_, int ny_halo_,
    int staggered_x_, int staggered_y_, // U->(1,0), V->(0,1), eta->(0,0)
    
    // Boundary condition parameters
    int sponge_cells_east_, int sponge_cells_west_,
    int bc_east_, int bc_west_,
        
    // Data
    float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if (( ((bc_west_ == 4) &&
           (ti < sponge_cells_west_) && (ti > 0)) ||
          ((bc_east_ == 4) &&
           (ti > nx_ + 2*nx_halo_ + staggered_x_ - sponge_cells_east_ - 1) &&
           (ti < nx_ + 2*nx_halo_ + staggered_x_ - 1)) ) &&
        (tj > 0) && (tj < ny_ + 2*ny_halo_ + staggered_y_-1))
    {

        // Identify inner and outer row
        int inner_col = sponge_cells_west_;
        int outer_col = 0;
        if (ti > sponge_cells_west_) {
            inner_col = nx_ + 2*nx_halo_ + staggered_x_ - sponge_cells_east_ - 1;
            outer_col = nx_ + 2*nx_halo_ + staggered_x_ - 1;
        }

        // Get row:
        float* data_row = (float*) ((char*) data_ptr_ + data_pitch_*tj);

        // Get inner value
        float inner_value = data_row[inner_col];

        // Get outer value
        float outer_value = data_row[outer_col];

        // Interpolate:
        float ratio = ((float)(ti - outer_col))/(inner_col - outer_col);
        data_row[ti] = outer_value + ratio*(inner_value - outer_value);
    }
}
} // extern "C"


extern "C" {
__global__ void boundary_flowRelaxationScheme_NS(
    // Discretization parameters
    int nx_, int ny_,
    int nx_halo_, int ny_halo_,
    int staggered_x_, int staggered_y_, // U->(1,0), V->(0,1), eta->(0,0)
    
    // Boundary condition parameters
    int sponge_cells_north_, int sponge_cells_south_,
    int bc_north_, int bc_south_,
    
    // Data
    float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if (( ((bc_south_ == 3) &&
           (tj < sponge_cells_south_) && (tj > 0)) ||
          ((bc_north_ == 3) &&
           (tj > ny_ + 2*ny_halo_ + staggered_y_ - sponge_cells_north_ - 1) &&
           (tj < ny_ + 2*ny_halo_ + staggered_y_ - 1)) ) &&
        (ti > 0) && (ti < nx_ + 2*ny_halo_ + staggered_x_-1))
    {
        // Identify the exterior and current row
        int exterior_row = 0;
        int j = tj;
        if (tj > sponge_cells_south_) {
            exterior_row = ny_ + 2*ny_halo_ + staggered_y_ - 1;
            j = exterior_row - tj;
        }
        float alpha = 1.0f - tanh((j-1.0f)/2.0f);

        // Get exterior value
        float* exterior_row_ptr = (float*) ((char*) data_ptr_ + data_pitch_*exterior_row);
        float exterior_value = exterior_row_ptr[ti];

        // Find target cell
        float* target_row_ptr = (float*) ((char*) data_ptr_ + data_pitch_*tj);

        // Interpolate:
        target_row_ptr[ti] = (1.0f - alpha)*target_row_ptr[ti] + alpha*exterior_value;
    }
}
} // extern "C" 


extern "C" {
__global__ void boundary_flowRelaxationScheme_EW(
    // Discretization parameters
    int nx_, int ny_,
    int nx_halo_, int ny_halo_,
    int staggered_x_, int staggered_y_, // U->(1,0), V->(0,1), eta->(0,0)
    
    // Boundary condition parameters
    int sponge_cells_east_, int sponge_cells_west_,
    int bc_east_, int bc_west_,
        
    // Data
    float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if (( ((bc_west_ == 3) &&
           (ti < sponge_cells_west_) && (ti > 0)) ||
          ((bc_east_ == 3) &&
           (ti > nx_ + 2*nx_halo_ + staggered_x_ - sponge_cells_east_ - 1) &&
           (ti < nx_ + 2*nx_halo_ + staggered_x_ - 1)) ) &&
        (tj > 0) && (tj < ny_ + 2*ny_halo_ + staggered_y_-1))
    {

        int exterior_col = 0;
        int j = ti;
        if (ti > sponge_cells_west_) {
            exterior_col = nx_ + 2*nx_halo_ + staggered_x_ - 1;
            j = exterior_col - ti;
        }
        float alpha = 1.0f - tanh((j-1.0f)/2.0f);

        // Get row:
        float* data_row = (float*) ((char*) data_ptr_ + data_pitch_*tj);

        // Get exterior value
        float exterior_value = data_row[exterior_col];

        // Interpolate:
        data_row[ti] = (1.0f - alpha)*data_row[ti] + alpha*exterior_value;
    }
}
} // extern "C"
