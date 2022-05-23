/**
This software is part of GPU Ocean. 

Copyright (C) 2016-2018  SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This CUDA kernel implements part of the Centered in Time, Centered 
in Space (leapfrog) numerical scheme for the shallow water equations, 
described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .


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


/**
  * Kernel that evolves U, V and eta one step in time.
  */
extern "C" {
__global__ void ctcsStepKernel(
        //Discretization parameters
        const int nx_, const int ny_,
        const int wall_bc_,
        const float dx_, const float dy_, const float dt_,
    
        //Physical parameters
        const float g_, //< Gravitational constant
        const float f_, //< Coriolis coefficient
        const float beta_, //< Coriolis force f_ + beta_*(y-y0)
        const float y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)
        const float r_, //< Bottom friction coefficient
    
        //Numerical diffusion
        const float A_,
    
        // Read and write Data
        float* eta0_ptr_, const int eta0_pitch_, //eta^n-1 (also used as output, that is eta^n+1)
        float* U0_ptr_, const int U0_pitch_, // U^n-1, also output, U^n+1
        float* V0_ptr_, const int V0_pitch_, // V^n-1, also output V^n+1

        // Read data
        float* H_ptr_, const int H_pitch_,
        float* eta1_ptr_, const int eta1_pitch_, // eta^n
        float* U1_ptr_, const int U1_pitch_, // U^n
        float* V1_ptr_, const int V1_pitch_, // V^n 
    
        // Wind stress parameters
        const float wind_stress_t_) {
        
    __shared__ float H_shared[block_height+2][block_width+2];
    __shared__ float eta1_shared[block_height+2][block_width+2];
    __shared__ float U1_shared[block_height+2][block_width+2];
    __shared__ float V1_shared[block_height+2][block_width+2];
    
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

   
    //Start of block within domain
    const int bx = blockDim.x * blockIdx.x + 1; //Skip global ghost cells
    const int by = blockDim.y * blockIdx.y + 1; //Skip global ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;
    
    
    //Compute pointer to current row in the U array and r ead current U
    float* const U0_row = (float*) ((char*) U0_ptr_ + U0_pitch_*tj);
    float U0 = 0.0f;
    if (ti > 0 && ti < nx_+2 && tj > 0 && tj < ny_+1) {
        U0 = U0_row[ti];
    }

    //Compute pointer to current row in the V array and read current V
    float* const V0_row = (float*) ((char*) V0_ptr_ + V0_pitch_*tj);
    float V0 = 0.0f;
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+2) {        
        V0 = V0_row[ti];
    }
    
    //Compute pointer to current row in the U array and read current eta
    float* eta0_row = (float*) ((char*) eta0_ptr_ + eta0_pitch_*tj);
    float eta0 = 0.0f;
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        eta0 = eta0_row[ti];
    }

    //Read H and eta into shared memory: (nx+2)*(ny+2) cells
    for (int j=ty; j<block_height+2; j+=blockDim.y) {
        // "fake" global ghost cells by clamping
        // const int l = clamp(by + j - 1, 1, ny_);
        
        // H and eta are filled with proper ghost cell values
        const int l = by + j - 1;
        if (l >= 0 && l <= ny_+1) {
        
            //Compute the pointer to current row in the H and eta arrays
            float* const H_row = (float*) ((char*) H_ptr_ + H_pitch_*l);
            float* const eta1_row = (float*) ((char*) eta1_ptr_ + eta1_pitch_*l);

            for (int i=tx; i<block_width+2; i+=blockDim.x) {
                // "fake" global ghost cells by clamping
                //const int k = clamp(bx + i, 1, nx_);
                
                const int k = bx + i - 1;
                if ( k >= 0 && k <= nx_+1) {
                    H_shared[j][i] = H_row[k];
                    eta1_shared[j][i] = eta1_row[k];
                }
            }
        }
    }

    //Read U into shared memory: (nx+2)*(ny+2) cells
    for (int j=ty; j<block_height+2; j+=blockDim.y) {
        // "fake" ghost cells by clamping
        // const int l = clamp(by + j - 1, 1, ny_);
        
        const int l = by + j - 1;
        if (l >= 0 && l <= ny_+1) {

            //Compute the pointer to current row in the U array
            float* const U1_row = (float*) ((char*) U1_ptr_ + U1_pitch_*l);

            for (int i=tx; i<block_width+2; i+=blockDim.x) {
                // Prevent out-of-bounds
                // const int k = clamp(bx + i - 1, halo_x_, nx_+halo_x_);
                
                const int k = bx + i - 1;
                if ( k >= 0 && k <= nx_+2) {
                    U1_shared[j][i] = U1_row[k];
                }
            }
        }
    }
    

    //Read V into shared memory: (nx+2)*(ny+2) cells
    for (int j=ty; j<block_height+2; j+=blockDim.y) {
        // Prevent out-of-bounds
        // const int l = clamp(by + j - 1, 0, ny_);
        
        const int l = by + j - 1;
        if (l >= 0 && l <= ny_+2) {

            //Compute the pointer to current row in the U array
            float* const V1_row = (float*) ((char*) V1_ptr_ + V1_pitch_*l);

            for (int i=tx; i<block_width+2; i+=blockDim.x) {
                // "fake" ghost cells by clamping
                // const int k = clamp(bx + i, halo_x_, nx_+halo_x_);

                const int k = bx + i - 1;
                if (k >= 0 && k <= nx_+1) {                
                    V1_shared[j][i] = V1_row[k];
                }
            }
        }
    }
    
    //Make sure all threads have read into shared mem
    __syncthreads();
    
    
    
    // Find next U
    {
        /**
          * Now get all our required variables as short-hands
          * here we use the notation of
          *  Var_00 as var_i,j
          *  Var_p0 as var_i+1,j
          *  Var_0m as var_i,j-1
          * etc
          */
        const float U_00 = U1_shared[ty+1][tx+1]; //U at "center"
        const float U_0p = U1_shared[ty+2][tx+1]; //U at "north"
        const float U_0m = U1_shared[ty  ][tx+1]; //U at "south"
        const float U_p0 = U1_shared[ty+1][tx+2]; //U at "east"
        const float U_m0 = U1_shared[ty+1][tx  ]; //U at "west"
        
        // First row of V1_shared is intentionally never used
        const float V_00 = V1_shared[ty+1+1][tx  ];
        const float V_p0 = V1_shared[ty+1+1][tx+1];
        const float V_0m = V1_shared[ty+1  ][tx  ];
        const float V_pm = V1_shared[ty+1  ][tx+1];
        
        const float H_0m = H_shared[ty  ][tx  ]; 
        const float H_00 = H_shared[ty+1][tx  ]; 
        const float H_0p = H_shared[ty+2][tx  ];
        const float H_pm = H_shared[ty  ][tx+1];
        const float H_p0 = H_shared[ty+1][tx+1]; 
        const float H_pp = H_shared[ty+2][tx+1];
        
        const float eta_0m = eta1_shared[ty  ][tx  ]; 
        const float eta_00 = eta1_shared[ty+1][tx  ]; 
        const float eta_0p = eta1_shared[ty+2][tx  ];
        const float eta_pm = eta1_shared[ty  ][tx+1];
        const float eta_p0 = eta1_shared[ty+1][tx+1]; 
        const float eta_pp = eta1_shared[ty+2][tx+1];

        // Coriolis at U positions:
        const float f_v_0 = f_ + beta_ * ((blockIdx.y * blockDim.y + threadIdx.y)+0.5f-y_zero_reference_cell_ + 0.5f)*dy_;
        const float f_v_m = f_ + beta_ * ((blockIdx.y * blockDim.y + threadIdx.y)-0.5f-y_zero_reference_cell_ + 0.5f)*dy_;

        //Reconstruct H_bar and H_x (at the U position)
        const float H_bar_0m = 0.25f*(H_0m + H_pm + H_00 + H_p0);
        const float H_bar_00 = 0.25f*(H_00 + H_p0 + H_0p + H_pp);
        const float H_x = 0.5f*(H_00 + H_p0);
        
        //Reconstruct Eta_bar at the V position
        const float eta_bar_0m = 0.25f*(eta_0m + eta_pm + eta_00 + eta_p0);
        const float eta_bar_00 = 0.25f*(eta_00 + eta_p0 + eta_0p + eta_pp);
        
        //Reconstruct fV at the U position
        const float fV_bar = 0.25f*( f_v_m*(V_0m + V_pm) + f_v_0*(V_00 + V_p0) );

        //Calculate the friction coefficient
        const float C = 1.0f + 2.0f*r_*dt_/H_x + 2.0f*A_*dt_*(dx_*dx_ + dy_*dy_)/(dx_*dx_*dy_*dy_);

        //Calculate the pressure/gravitational effect
        const float h_p0 = H_p0 + eta_p0;
        const float h_00 = H_00 + eta_00;
        const float h_x = 0.5f*(h_00 + h_p0); //Could possibly use h for pressure terms instead of H
        
        // Different options for discretizing the pressure term:
        
        // Option 1: The version we have used all along:
        //const float P_x_hat = -0.5f*g_*(eta_p0*eta_p0 - eta_00*eta_00);
        //const float P_x = -g_*h_x*(eta_p0 - eta_00) + P_x_hat;
        
        // Option 2: The version that is closer (but no equal to) the summary in part II of the MET report
        const float P_x_hat = -0.5f*g_*(eta_p0*eta_p0 - eta_00*eta_00);
        const float P_x = -g_*H_x*(eta_p0 - eta_00) + P_x_hat;
        
        // Option 3: Corresponding to Lars Petter's compendium GEF4510 2014 (if we discard varying bathymetry)
        //const float P_x = -0.5f*g_*(h_p0*h_p0 - h_00*h_00);
        
        // Option 4: According to Lars Petter's fortran code
        //const float P_x = -0.5f*g_*((H_p0 + H_00)*(eta_p0 - eta_00));
        //const float P_x = -g_*H_x*(eta_p0 - eta_00) + P_x_hat;
        
        // It seems that options 2 and 3 gives the same results (at least for Kelvin waves)
        
        //Calculate nonlinear effects
        const float N_a = (U_p0 + U_00)*(U_p0 + U_00) / (H_p0 + eta_p0);
        const float N_b = (U_00 + U_m0)*(U_00 + U_m0) / (H_00 + eta_00);
        const float N_c = (U_0p + U_00)*(V_p0 + V_00) / (H_bar_00 + eta_bar_00);
        const float N_d = (U_00 + U_0m)*(V_pm + V_0m) / (H_bar_0m + eta_bar_0m);
        const float N = 0.25f*( N_a - N_b + (dx_/dy_)*(N_c - N_d) );
        
        //Calculate eddy viscosity term
        const float E = (U_p0 - U0 + U_m0)/(dx_*dx_) + (U_0p - U0 + U_0m)/(dy_*dy_);
        
        //Calculate the wind shear stress
        //FIXME Check coordinates (ti_, tj_) here!!!
        //TODO Check coordinates (ti_, tj_) here!!!
        //WARNING Check coordinates (ti_, tj_) here!!!
        const float X = windStressX(wind_stress_t_, ti, tj+0.5, nx_, ny_);

        //Compute the V at the next timestep
        float U2 = (U0 + 2.0f*dt_*(fV_bar + (N + P_x)/dx_ + X + A_*E) ) / C;

        // Set wall boundary conditions
        if ( (ti == 1 && (wall_bc_ & 0x08)) || ((ti == nx_ +1) && (wall_bc_ & 0x02)) ) {
            U2 = 0.0f;
        }
        
        //Write to main memory for internal cells
        if (ti > 0 && ti < nx_+2 && tj > 0 && tj < ny_+1) {
            U0_row[ti] = U2;
        }
    }
    
    // Find next V:
    {
        /**
          * Now get all our required variables as short-hands
          * here we use the notation of
          *  Var_00 as var_i,j
          *  Var_p0 as var_i+1,j
          *  Var_0m as var_i,j-1
          * etc
          */
        const float V_00 = V1_shared[ty+1][tx+1]; //V at "center"
        const float V_0p = V1_shared[ty+2][tx+1]; //V at "north"
        const float V_0m = V1_shared[ty  ][tx+1]; //V at "south"
        const float V_p0 = V1_shared[ty+1][tx+2]; //V at "east"
        const float V_m0 = V1_shared[ty+1][tx  ]; //V at "west"
        
        // First col of U1_shared is intentionally never used
        const float U_00 = U1_shared[ty  ][tx+1+1];
        const float U_0p = U1_shared[ty+1][tx+1+1];
        const float U_m0 = U1_shared[ty  ][tx+1  ];
        const float U_mp = U1_shared[ty+1][tx+1  ];
        
        const float H_m0 = H_shared[ty  ][tx  ]; 
        const float H_00 = H_shared[ty  ][tx+1]; 
        const float H_p0 = H_shared[ty  ][tx+2];
        const float H_mp = H_shared[ty+1][tx  ];
        const float H_0p = H_shared[ty+1][tx+1]; 
        const float H_pp = H_shared[ty+1][tx+2];
        
        const float eta_m0 = eta1_shared[ty  ][tx  ]; 
        const float eta_00 = eta1_shared[ty  ][tx+1]; 
        const float eta_p0 = eta1_shared[ty  ][tx+2];
        const float eta_mp = eta1_shared[ty+1][tx  ]; 
        const float eta_0p = eta1_shared[ty+1][tx+1]; 
        const float eta_pp = eta1_shared[ty+1][tx+2];

        // Coriolis at U positions:
        const float f_u_0 = f_ + beta_ * ((blockIdx.y * blockDim.y + threadIdx.y+1)-0.5f-y_zero_reference_cell_)*dy_;
        const float f_u_p = f_ + beta_ * ((blockIdx.y * blockDim.y + threadIdx.y+1)+0.5f-y_zero_reference_cell_)*dy_;   
        
        //Reconstruct H_bar and H_y (at the V position)
        const float H_bar_m0 = 0.25f*(H_m0 + H_mp + H_00 + H_0p);
        const float H_bar_00 = 0.25f*(H_00 + H_0p + H_p0 + H_pp);
        const float H_y = 0.5f*(H_00 + H_0p);
        
        //Reconstruct Eta_bar at the V position
        const float eta_bar_m0 = 0.25f*(eta_m0 + eta_mp + eta_00 + eta_0p);
        const float eta_bar_00 = 0.25f*(eta_00 + eta_0p + eta_p0 + eta_pp);

        //Reconstruct fU at the V position
        const float fU_bar = 0.25f*( f_u_0*(U_m0 + U_00) + f_u_p*(U_mp + U_0p) );

        //Calculate the friction coefficient
        const float C = 1.0f + 2.0f*r_*dt_/H_y + 2.0f*A_*dt_*(dx_*dx_ + dy_*dy_)/(dx_*dx_*dy_*dy_);

        //Calculate the pressure/gravitational effect
        const float h_0p = H_0p + eta_0p;
        const float h_00 = H_00 + eta_00;
        const float h_y = 0.5f*(h_00 + h_0p); //Could possibly use h for pressure terms instead of H
        
        // The version we have used all along:
        //const float P_y_hat = -0.5f*g_*(eta_0p*eta_0p - eta_00*eta_00);
        //const float P_y = -g_*h_y*(eta_0p - eta_00) + P_y_hat;
        
        // The version that is closer (but no equal to) the summary in part II of the MET report
        const float P_y_hat = -0.5f*g_*(eta_0p*eta_0p - eta_00*eta_00);
        const float P_y = -g_*H_y*(eta_0p - eta_00) + P_y_hat;
        
        // Corresponding to Lars Petter's compendium GEF4510 2014 (if we discard varying bathymetry)
        //const float P_y = -0.5f*g_*(h_0p*h_0p - h_00*h_00);

        // Option 4: According to Lars Petter's fortran code
        //const float P_y = -0.5f*g_*((H_0p + H_00)*(eta_0p - eta_00));
        //const float P_y = -g_*H_y*(eta_0p - eta_00) + P_y_hat;
        
        //Calculate nonlinear effects
        const float N_a = (V_0p + V_00)*(V_0p + V_00) / (H_0p + eta_0p);
        const float N_b = (V_00 + V_0m)*(V_00 + V_0m) / (H_00 + eta_00);
        const float N_c = (U_0p + U_00)*(V_p0 + V_00) / (H_bar_00 + eta_bar_00);
        const float N_d = (U_mp + U_m0)*(V_00 + V_m0) / (H_bar_m0 + eta_bar_m0);
        const float N = 0.25f*( N_a - N_b + (dy_/dx_)*(N_c - N_d) );
        
        //Calculate eddy viscosity term
        const float E = (V_p0 - V0 + V_m0)/(dx_*dx_) + (V_0p - V0 + V_0m)/(dy_*dy_);

        //Calculate the wind shear stress
        //FIXME Check coordinates (ti_, tj_) here!!!
        //TODO Check coordinates (ti_, tj_) here!!!
        //WARNING Check coordinates (ti_, tj_) here!!!
        const float Y = windStressY(wind_stress_t_, ti+0.5, tj, nx_, ny_);
        
        //Compute the V at the next timestep
        float V2 = (V0 + 2.0f*dt_*(-fU_bar + (N + P_y)/dy_ + Y + A_*E) ) / C;
        
        if ( ((tj == 1) && (wall_bc_ & 0x04)) || ((tj == ny_+1) && (wall_bc_ & 0x01)) ) {
            V2 = 0.0f;
        }
        
        //Write to main memory for internal cells
        if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+2) {
            V0_row[ti] = V2;
        }
    }
    
    //Compute the eta at the next timestep
    //We intentionally do not read first row and first col of U1_shared and V1_shared
    const float eta2 = eta0 - 2.0f*dt_/dx_ * (U1_shared[ty+1][tx+1+1] - U1_shared[ty+1][tx+1])
                            - 2.0f*dt_/dy_ * (V1_shared[ty+1+1][tx+1] - V1_shared[ty+1][tx+1]);
    
    //Write to main memory for internal cells
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        eta0_row[ti] = eta2;
    }
    
    
}
} //extern "C" 