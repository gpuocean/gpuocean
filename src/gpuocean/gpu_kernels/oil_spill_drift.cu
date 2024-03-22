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

__device__ float water_velocity(
        const float* eta_ptr, const int eta_pitch,
        const float* momentum_ptr, const int momentum_pitch,
        const float* Hm_ptr, const int Hm_pitch,
        const int cell_id_x, const int cell_id_y) {
    
    // Read the water velocity from global memory
    const float* eta_row_y = (float*) ((char*) eta_ptr + eta_pitch*cell_id_y);
    const float* Hm_row_y = (float*) ((char*) Hm_ptr + Hm_pitch*cell_id_y);
    const float h = Hm_row_y[cell_id_x] + eta_row_y[cell_id_x];

    const float* momentum_row = (float*) ((char*) momentum_ptr + momentum_pitch*cell_id_y);
    const float velocity = momentum_row[cell_id_x]/h;
    
    return velocity;
}

__device__ float water_velocity_no_interpolation(
        const float* eta_ptr, const int eta_pitch,
        const float* momentum_ptr, const int momentum_pitch,
        const float* Hm_ptr, const int Hm_pitch,
        const float drifter_pos_x, const float drifter_pos_y, 
        const float dx, const float dy) {
    
    // Find indices for the cell this thread's particle is in
    // Note that we compensate for 2 ghost cells in each direction 
    const int cell_id_x = (int)(floor(drifter_pos_x/dx) + 2);
    const int cell_id_y = (int)(floor(drifter_pos_y/dy) + 2);
    
    // Read and compute water velocity within cell
    return water_velocity(eta_ptr, eta_pitch,
                          momentum_ptr, momentum_pitch,
                          Hm_ptr, Hm_pitch, 
                          cell_id_x, cell_id_y);
}

__device__ float rise_velocity(
        float droplet_depth,
        const float droplet_diameter,
        const float water_density,
        const float oil_density,
        const float water_viscosity,
        const float g) {
    // Calculate the rise velocity of a droplet in m/s.

    const float g_delro = g * (water_density - oil_density) / water_density;
    const float w1 = pow(droplet_diameter, 2) * g_delro / (18. * water_viscosity);
    const float w2 = copysignf(w1, g_delro);
    const float rise_velocity = w1 * w2 / (w1 + w2); // in m/s
    
    return rise_velocity;
}

__device__ void fill_randn(
        float* rand_numbers, const int n, 
        unsigned long long* seed_ptr, const int seed_pitch,
        const int ti) { 
    // Read seed
    unsigned long long* const seed_row = (unsigned long long*) ((char*) seed_ptr + seed_pitch*ti);
    unsigned long long seed = seed_row[0];
    
    for (int i = 0; i < 3; i++) {
        float2 rand_n = rand_normal(&seed);
        rand_numbers[i*2] = rand_n.x;
        if (i < (int)(floor(n/2.0))) {
            rand_numbers[i*2+1] = rand_n.y;
        }
    }
    // Write seed back to global memory
    seed_row[0] = seed;
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
        unsigned long long* seed_ptr, int seed_pitch, 
        const float horisontal_diffusivity,
        const float droplet_diameter,
        const float oil_density, const float water_density,
        const float water_viscosity, const float g)
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

            // Generate 5 random numbers sampled from N(0, 1) 
            float rand_numbers [5];
            fill_randn(rand_numbers, 5, seed_ptr, seed_pitch, ti);

            // Obtain pointer to our drifter:
            float* drifter = (float*) ((char*) drifters_positions + drifters_pitch*ti);
            float drifter_pos_x = drifter[0];
            float drifter_pos_y = drifter[1];
            float drifter_depth = drifter[2];
            
            // Read and compute water velocity within cell
            const float u = water_velocity_no_interpolation(eta_ptr, eta_pitch,
                                                            hu_ptr, hu_pitch,
                                                            Hm_ptr, Hm_pitch, 
                                                            drifter_pos_x, drifter_pos_y,
                                                            dx, dy);
            const float v = water_velocity_no_interpolation(eta_ptr, eta_pitch,
                                                            hv_ptr, hv_pitch,
                                                            Hm_ptr, Hm_pitch, 
                                                            drifter_pos_x, drifter_pos_y,
                                                            dx, dy);
        
            // Move drifter with a simple forward Euler
            drifter_pos_x += u*dt;
            drifter_pos_y += v*dt;
            
            // Add horizontal diffusion
            drifter_pos_x += horisontal_diffusivity*rand_numbers[0]*sqrt(dt);
            drifter_pos_y += horisontal_diffusivity*rand_numbers[1]*sqrt(dt);
           
            // Assuming periodic boundary conditions
            drifter_pos_x -= floor(drifter_pos_x / (nx*dx))*(nx*dx);
            drifter_pos_y -= floor(drifter_pos_y / (ny*dy))*(ny*dy);

            // Move drifter vertically.
            const float rise_vel = rise_velocity(drifter_depth, droplet_diameter, water_density, oil_density, water_viscosity, g);

            // Update drifter depth
            drifter_depth += rise_vel;
            drifter_depth = min(drifter_depth, 0.0);

            // Write to global memory
            drifter[0] = drifter_pos_x;
            drifter[1] = drifter_pos_y;
            drifter[2] = drifter_depth;
        }
    }
} // extern "C"
