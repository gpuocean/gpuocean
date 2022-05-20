/*
This software is part of GPU Ocean.

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This CUDA kernel initializes bottom topography in cell centers,
based on bottom topography given on cell intersections.

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

//WARNING: Must match CDKLM16_kernel.cu and max_dt.cu
//WARNING: This is error prone - as comparison with floating point numbers is not accurate
#define CDKLM_DRY_FLAG 1.0e-30f
#define CDKLM_DRY_EPS 1.0e-3f

/**
 * Initializing bottom bathymetry on cell mid-points (Bm) based on 
 * given bathymetry on cell intersections (Bi).
 * Grid (including ghost cells) of nx by ny cells, and 
 * nx+1 by ny+1 intersections.
 */
extern "C" {
__global__ void initBm(const int nx_, const int ny_,
        float* Bi_ptr_, int Bi_pitch_,
        float land_value_,
        float* Bm_ptr_, int Bm_pitch_) {
            
           
    //const float land_value_ = 1.0e20;

    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;
    
    //Shared memory variables
    __shared__ float Bi[block_height+1][block_width+1];
    
    // Read Bi into shared memory
    for (int j=ty; j<block_height+1; j+=blockDim.y) {
        const int glob_j = by + j;
        if (glob_j < ny_+1) {
            float* const Bi_row = (float*) ((char*) Bi_ptr_ + Bi_pitch_*glob_j);
            for (int i=tx; i<block_width+1; i+=blockDim.x) {
                const int glob_i = bx + i;
                if (glob_i < nx_+1) {
                    Bi[j][i] = Bi_row[glob_i];
                    
                    // Set dry intersections to "zero"
                    if (fabsf(Bi[j][i] - land_value_) <= CDKLM_DRY_EPS) {
                        Bi[j][i] = CDKLM_DRY_FLAG;
                    }
                }
            }
        }
    }

    // Sync so that all threads have written to shared memory
    __syncthreads();

    // Calculate Bm and write to main memory
    if (ti < nx_ && tj < ny_) {
        float* Bm_row = (float*) ((char*) Bm_ptr_ + Bm_pitch_*tj);
        
        const float a = Bi[ty][tx];
        const float b = Bi[ty+1][tx];
        const float c = Bi[ty][tx+1];
        const float d = Bi[ty+1][tx+1];
        
        const float wet_count = 4.0f - 
                            (a == CDKLM_DRY_FLAG) -
                            (b == CDKLM_DRY_FLAG) -
                            (c == CDKLM_DRY_FLAG) -
                            (d == CDKLM_DRY_FLAG);
        
        // Set completely dry cells to land_value_
        if (wet_count == 0.0) {
            Bm_row[ti] = land_value_;
        }
        else {
            Bm_row[ti] = (a+b+c+d) / (wet_count);
        }
    }
}
} // extern "C"


/**
 *  Kernel for changing water elevation to water depth (e.g. for CDKLM16)
 *  (nx, ny) is the size of domain **including** ghost cells.
 */
extern "C" {
__global__ void waterElevationToDepth(const int nx_, const int ny_,
                    float* h_ptr_, int h_pitch_,
                    float* Bm_ptr_, int Bm_pitch_ ) {
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if (ti < nx_ && tj < ny_) {
        float* const h_row = (float*) ((char*) h_ptr_ + h_pitch_*tj);
        float* const Bm_row = (float*) ((char*) Bm_ptr_ + Bm_pitch_*tj);
        h_row[ti] -= Bm_row[ti];
    }
    
}
} // extern "C"


/**
 *  Kernel for changing water elevation to water depth (e.g. for CDKLM16)
 *  (nx, ny) is the size of domain **including** ghost cells.
 */
extern "C" {
__global__ void waterDepthToElevation(const int nx_, const int ny_,
                    float* w_ptr_, int w_pitch_,
                    float* h_ptr_, int h_pitch_,
                    float* Bm_ptr_, int Bm_pitch_ ) {
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if (ti < nx_ && tj < ny_) {
        float* const h_row = (float*) ((char*) h_ptr_ + h_pitch_*tj);
        float* const Bm_row = (float*) ((char*) Bm_ptr_ + Bm_pitch_*tj);
        float* const w_row = (float*) ((char*) w_ptr_ + w_pitch_*tj);
        
        w_row[ti] = h_row[ti] + Bm_row[ti];
    }
    
}
} // extern "C" 
