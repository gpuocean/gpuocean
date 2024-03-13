/*
This software is part of GPU Ocean. 

Copyright (C) 2024 SINTEF Digital

Super simple drift kernel.

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

#include "random_number_generators.cu"

__device__ float waterVelocity(
        float* eta_ptr, const int eta_pitch,
        float* momentum_ptr, const int momentum_pitch,
        float* Hm_ptr, const int Hm_pitch,
        const int cell_id_x, const int cell_id_y) {
    
    // Read the water velocity from global memory
    float* const eta_row_y = (float*) ((char*) eta_ptr + eta_pitch*cell_id_y);
    float* const Hm_row_y = (float*) ((char*) Hm_ptr + Hm_pitch*cell_id_y);
    float const h = Hm_row_y[cell_id_x] + eta_row_y[cell_id_x];

    float* const momentum_row = (float*) ((char*) momentum_ptr + momentum_pitch*cell_id_y);
    float const velocity = momentum_row[cell_id_x]/h;
    
    return velocity;
}

extern "C" {
__global__ void superSimpleDrift(
        const int nx, const int ny,
        const float dx, const float dy, const float dt,

        float* eta_ptr, const int eta_pitch,
        float* hu_ptr, const int hu_pitch,
        float* hv_ptr, const int hv_pitch,
        float* Hm_ptr, const int Hm_pitch,

        const int num_drifters,
        float* drifters_positions, const int drifters_pitch,
        float* random_numbers, const int rand_pitch,
        unsigned long long* seed_ptr, int seed_pitch, 
        const int rng_type)
    {
        // Each thread will be responsible for one drifter only 
        // Local index of thread within block (only needed in one dim)
        const int tx = threadIdx.x;
        // Index of start of block 
        const int bx = blockDim.x * blockIdx.x;
        // Global index of thread 
        const int ti = bx + tx;

        // We might have launched more threads than we have drifters
        if (ti < num_drifters ) {

            // Obtain pointer to our drifter:
            float* drifter = (float*) ((char*) drifters_positions + drifters_pitch*ti);
            float drifter_pos_x = drifter[0];
            float drifter_pos_y = drifter[1];

            // Find indices for the cell this thread's particle is in
            // Note that we compensate for 2 ghost cells in each direction 
            int const cell_id_x = (int)(floor(drifter_pos_x/dx) + 2);
            int const cell_id_y = (int)(floor(drifter_pos_y/dy) + 2);
            
            // Read and compute water velocity within cell
            float const u = waterVelocity(eta_ptr, eta_pitch,
                                          hu_ptr, hu_pitch,
                                          Hm_ptr, Hm_pitch, 
                                          cell_id_x, cell_id_y);
            float const v = waterVelocity(eta_ptr, eta_pitch,
                                          hv_ptr, hv_pitch,
                                          Hm_ptr, Hm_pitch, 
                                          cell_id_x, cell_id_y);
        
            // Move drifter with a simple forward Euler
            drifter_pos_x += u*dt;
            drifter_pos_y += v*dt;
            
            if (rng_type < 5) {
                // Diffusion in x and y
                float* rand_drifter = (float*)((char*) random_numbers + rand_pitch*ti);
                if (rng_type == 4) {
                    unsigned long long* const seed_row = (unsigned long long*) ((char*) seed_ptr + seed_pitch*ti);
                    unsigned long long seed = seed_row[0];
                    
                    // Generating 5 normal distributed random numbers
                    for (int i = 0; i < 3; i++) {
                        float2 rand_n = rand_normal(&seed);
                        rand_drifter[i*2] = rand_n.x;
                        if (i < 2) {
                            rand_drifter[i*2+1]= rand_n.y;
                        }
                    }
                    
                    // Write seed back to global memory
                    seed_row[0] = seed;
                }
                drifter_pos_x += rand_drifter[0]*sqrt(dt);
                drifter_pos_y += rand_drifter[1]*sqrt(dt);
            }
            else if(rng_type == 5) {
                // Avoid touching the random_numbers array
                unsigned long long* const seed_row = (unsigned long long*) ((char*) seed_ptr + seed_pitch*ti);
                unsigned long long seed = seed_row[0];
                
                // Generating 5 normal distributed random numbers
                float rand_numbers [5];
                for (int i = 0; i < 3; i++) {
                    float2 rand_n = rand_normal(&seed);
                    rand_numbers[i*2] = rand_n.x;
                    if (i < 2) {
                        rand_numbers[i*2+1] = rand_n.y;
                    }
                }
                
                // Write seed back to global memory
                seed_row[0] = seed;
                
                drifter_pos_x += rand_numbers[0]*sqrt(dt);
                drifter_pos_y += rand_numbers[1]*sqrt(dt);
            }

            // Assuming periodic boundary conditions
            drifter_pos_x -= floor(drifter_pos_x / (nx*dx))*(nx*dx);
            drifter_pos_y -= floor(drifter_pos_y / (ny*dy))*(ny*dy) + 0.1;

            // Write to global memory
            drifter[0] = drifter_pos_x;
            drifter[1] = drifter_pos_y;
        }
    }
} // extern "C"