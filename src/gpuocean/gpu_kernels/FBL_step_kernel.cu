/*
This software is part of GPU Ocean.

Copyright (C) 2018, 2019 SINTEF Digital
Copyright (C) 2018, 2019 Norwegian Meteorological Institute

This CUDA kernel implements the Forward Backward Linear 
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

// Finds the coriolis term based on the linear Coriolis force
// f = \tilde{f} + beta*(y-y0)
__device__ float linear_coriolis_term(const float f, const float beta,
                                      const float tj, const float dy,
                                      const float y_zero_reference_cell) {
    // tj is number of grid cells north the south face of first southern ghost cell
    // y_0 is at the southern face of the row y_zero_reference_cell.
    float y = (tj-y_zero_reference_cell)*dy;
    return f + beta * y;
}


/**
  * Kernel that evolves U, V and eta one step in time.
  */
extern "C" {
__global__ void fblStepKernel(
        //Discretization parameters
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
    
        //Physical parameters
        float g_, //< Gravitational constant
        float f_, //< Coriolis coefficient
        float beta_, //< Coriolis force f_ + beta_*(y-y0)
        float y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)
        float r_, //< Bottom friction coefficient
    
        //Data
        float* H_ptr_, int H_pitch_,
        float* U_ptr_, int U_pitch_,
        float* V_ptr_, int V_pitch_,
        float* eta_ptr_, int eta_pitch_,
        
        // Wall boundary conditions packed as bit-wise boolean
        int wall_bc_,
        
        // Wind stress parameters
        float wind_stress_t_) {
        
    __shared__ float H_shared[block_height+2][block_width+2];
    __shared__ float U_shared[block_height+2][block_width+1];
    __shared__ float V_shared[block_height+3][block_width+2];
    __shared__ float eta_shared[block_height+2][block_width+2];

    //Index of thread within block
    const int tx_u = threadIdx.x;   // U has no ghost cells in x-dir
    const int tx = threadIdx.x + 1; // Including ghost cell
    const int ty = threadIdx.y + 1; // Including ghost cell

    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti_u = bx + tx_u;
    const int ti = bx + tx;
    const int tj = by + ty;
    
    //Read H and eta into shared memory [block_height+2][block_width+2]
    for (int j=threadIdx.y; j<block_height+2; j+=blockDim.y) {
        const int l = by + j;
        
        //Compute the pointer to current row in the H and eta arrays
        float* const H_row = (float*) ((char*) H_ptr_ + H_pitch_*l);
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*l);
        
        for (int i=threadIdx.x; i<block_width+2; i+=blockDim.x) {
            const int k = bx + i;
            if (k < nx_+2 && l < ny_+2) {
                H_shared[j][i] = H_row[k];
                eta_shared[j][i] = eta_row[k];
            }
            else {
                H_shared[j][i] = 0.0f;
                eta_shared[j][i] = 0.0f;
            }
        }
    }
    
    //Read V into shared memory [block_height+3][block_width+2]
    for (int j=threadIdx.y; j<block_height+3; j+=blockDim.y) {
        const unsigned int l = by + j;
        
        //Compute the pointer to current row in the V array
        float* const V_row = (float*) ((char*) V_ptr_ + V_pitch_*l);
        
        for (int i=threadIdx.x; i<block_width+2; i+=blockDim.x) {
            const unsigned int k = bx + i;
            
            if (k < nx_+2 && l < ny_+3) {
                V_shared[j][i] = V_row[k];
            }
            else {
                V_shared[j][i] = 0.0f;
            }
        }
    }

    //Make sure all threads have read into shared mem
    __syncthreads();
    
    // Calculate U and write U to shared memory [block_height+2][block_width+1]
    for (int j=threadIdx.y; j<block_height+2; j+=blockDim.y) {
        const int l = by + j;
        
        //Compute the pointer to current row in the V array
        float* const U_row = (float*) ((char*) U_ptr_ + U_pitch_*l);
        //float U_current = 0.0f;
        //if (ti_u < nx_ && tj > 0 && tj < ny_+1) {
        //    U_current = U_row[ti_u];
        //}
    
        
        for (int i=threadIdx.x; i<block_width+1; i+=blockDim.x) {
            const int k = bx + i;
            float U_current = 0.0f;
            if (k < nx_+1 && l < ny_+2) {
                U_current = U_row[k];
            }
            
            //Reconstruct H at the U position
            float H_m = 0.5f*(H_shared[j][i] + H_shared[j][i+1]);
            
            // Coriolis forces at U position and V positions
            float f_u =   linear_coriolis_term(f_, beta_, l+0.5f, dy_, y_zero_reference_cell_);
            float f_v_p = linear_coriolis_term(f_, beta_, l+1.0f, dy_, y_zero_reference_cell_);
            float f_v_m = linear_coriolis_term(f_, beta_, l     , dy_, y_zero_reference_cell_);
   
            //Reconstruct f*V at the U position
            float fV_m = 0.25f*( f_v_m*(V_shared[j  ][i] + V_shared[j  ][i+1])
                               + f_v_p*(V_shared[j+1][i] + V_shared[j+1][i+1]) );

            //Calculate the friction coefficient
            float B = H_m/(H_m + r_*dt_);

            //Calculate the gravitational effect
            float P = g_*H_m*(eta_shared[j][i] - eta_shared[j][i+1])/dx_;
            
            //FIXME Check coordinates (ti_, tj_) here!!!
            //TODO Check coordinates (ti_, tj_) here!!!
            //WARNING Check coordinates (ti_, tj_) here!!!
            float X = windStressX(wind_stress_t_, k, l+0.5, nx_, ny_);

            //Compute the U at the next timestep
            float U_next = B*(U_current + dt_*(fV_m + P + X) );
            
            // Checking wall boundary conditions west and east
            if ( ((k == 0) && (wall_bc_ & 0x08)) || ((k == nx_) && (wall_bc_ & 0x02)) ) {
                U_next = 0.0f;
            }
            
            U_shared[j][i] = U_next;
            //U_shared[j][i] = 0.5f*(eta_shared[j][i] + eta_shared[j][i+1]);
            
        }
    }
    __syncthreads();
    
    // Calculate V and write V to shared memory 
    // Write to [block_height+1][block_width] within [block_height+3][block_width+2]
    for (int j=threadIdx.y+1; j<block_height+2; j+=blockDim.y) {
        const unsigned int l = by + j;
            
        for (int i=threadIdx.x+1; i<block_width+1; i+=blockDim.x) {
            const unsigned int k = bx + i;
            
            float V_current = V_shared[j][i];
            
            //Reconstruct H at the V position
            float H_m = 0.5f*(H_shared[j-1][i] + H_shared[j][i]);

            // Coriolis forces at V position and U positions
            float f_v   = linear_coriolis_term(f_, beta_, l,      dy_, y_zero_reference_cell_);
            float f_u_p = linear_coriolis_term(f_, beta_, l+0.5f, dy_, y_zero_reference_cell_);
            float f_u_m = linear_coriolis_term(f_, beta_, l-0.5f, dy_, y_zero_reference_cell_); 
    
            //Reconstruct f*U at the V position
            float fU_m = 0.25f*( f_u_m*(U_shared[j-1][i-1] + U_shared[j-1][i])
                               + f_u_p*(U_shared[j  ][i-1] + U_shared[j  ][i]) );

            //Calculate the friction coefficient
            float B = H_m/(H_m + r_*dt_);

            //Calculate the gravitational effect
            float P = g_*H_m*(eta_shared[j-1][i] - eta_shared[j][i])/dy_;

            //FIXME Check coordinates (k, l) here!!!
            //TODO Check coordinates (k, l) here!!!
            //WARNING Check coordinates (k, l) here!!!
            float Y = windStressY(wind_stress_t_, k+0.5, l, nx_, ny_);

            //Compute the V at the next timestep
            float V_next = B*(V_current + dt_*(-fU_m + P + Y) );
            
            // Checking wall boundary conditions
            if ( ((l < 2) && (wall_bc_ & 0x04)) || ((l > ny_) && (wall_bc_ & 0x01)) ) {
                V_next = 0.0f;
            }
            
            V_shared[j][i] = V_next;
        }
    }
    __syncthreads();
    
    
    // Calculate eta for internal local threads and write all results
    float eta_next = eta_shared[ty][tx] - dt_/dx_ * (U_shared[ty][tx] - U_shared[ty][tx-1])
                                        - dt_/dy_ * (V_shared[ty+1][tx] - V_shared[ty][tx]);
    
    
    //Write to main memory for internal cells
    if (ti_u < nx_ && tj > 0 && tj < ny_+1) {
        float* const U_row = (float*) ((char*) U_ptr_ + U_pitch_*tj);
        U_row[ti_u] = U_shared[ty][tx_u];
    }
    
    //Write to V and eta to main memory
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        
        float* const V_row = (float*) ((char*) V_ptr_ + V_pitch_*tj);
        V_row[ti] = V_shared[ty][tx];
        
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj);
        eta_row[ti] = eta_next;
    }
}
} // extern "C" 