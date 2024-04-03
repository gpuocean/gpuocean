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

__device__ float water_velocity_bilinear_interpolation(
        const float* eta_ptr, const int eta_pitch,
        const float* momentum_ptr, const int momentum_pitch,
        const float* Hm_ptr, const int Hm_pitch,
        const float drifter_pos_x, const float drifter_pos_y, 
        const float dx, const float dy) {
    
    // Find indices for the cell this thread's particle is in
    // Note that we compensate for 2 ghost cells in each direction 
    const int cell_id_x = (int)(floor(drifter_pos_x/dx) + 2);
    const int cell_id_y = (int)(floor(drifter_pos_y/dy) + 2);

    // Find neighbouring cells and relative position between cell centers
    float const frac_x = drifter_pos_x / dx - floor(drifter_pos_x / dx);
    float const frac_y = drifter_pos_y / dy - floor(drifter_pos_y / dy);
    
    const int cell_id_x0 = frac_x < 0.5f ? cell_id_x - 1 : cell_id_x;
    const float x_factor = frac_x < 0.5f ? frac_x + 0.5f : frac_x - 0.5f; 
    const int cell_id_x1 = cell_id_x0 + 1;

    const int cell_id_y0 = frac_y < 0.5f ? cell_id_y - 1 : cell_id_y;
    const float y_factor = frac_y < 0.5f ? frac_y + 0.5f : frac_y - 0.5f; 
    const int cell_id_y1 = cell_id_y0 + 1;
        
    float const vel_x0y0 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x0, cell_id_y0);
    float const vel_x1y0 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x1, cell_id_y0);
    float const vel_x0y1 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x0, cell_id_y1);
    float const vel_x1y1 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x1, cell_id_y1);
    
    float const vel_y0 = (1-x_factor)*vel_x0y0 + x_factor * vel_x1y0; 
    float const vel_y1 = (1-x_factor)*vel_x0y1 + x_factor * vel_x1y1; 

    // Read and compute water velocity within cell
    return (1-y_factor)*vel_y0 + y_factor*vel_y1;
}

__device__ float rise_velocity(
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

__device__ float is_submerged(float* drifter) {
    return drifter[2] < 0.0;
}

__device__ float euler_maruyama_scheme(
    const float ksi,
    const float vertical_diffusivity) {
    // Euler-Maruyama scheme assuming constant diffusivity. Return vertical displacement due to diffusivity.

    return ksi * sqrt(2 * vertical_diffusivity);
}

__device__ float vertical_transport(
        float& droplet_depth,
        const float droplet_diameter,
        const float water_density,
        const float oil_density,
        const float water_viscosity,
        const float g,
        const float dt,
        const float ksi,
        const float water_depth,
        const float vertical_diffusivity) {
    // Move the drifter vertically (advection + diffusion)

    // Vertical diffusion step (m)
    const float diffusion_step = euler_maruyama_scheme(ksi, vertical_diffusivity);

    droplet_depth = -abs(droplet_depth + diffusion_step); // Reflect off surface
    droplet_depth = min(2 * water_depth - droplet_depth, droplet_depth); // Reflect off bottom
    if (droplet_depth > 0.0) {
        droplet_depth = water_depth * 0.5f;
    }

    // Calculate the rise velocity due to buoyancy
    const float rise_vel = rise_velocity(droplet_diameter, water_density, oil_density, water_viscosity, g);
    
    // Vertical advection step in m
    const float advection_step = rise_vel * dt;
    droplet_depth += advection_step;
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
        const float horizontal_diffusivity,
        const float vertical_diffusivity,
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

            // Random number for vertical diffusion (normal distribution with mean 0 and variance dt)
            const float ksi_z = rand_numbers[0] * sqrt(dt);
            
            // Obtain pointer to our drifter:
            float* drifter = (float*) ((char*) drifters_positions + drifters_pitch*ti);
            float drifter_pos_x = drifter[0];
            float drifter_pos_y = drifter[1];
            float drifter_depth = drifter[2];
            
            // Read and compute water velocity within cell
            const float u = water_velocity_bilinear_interpolation(eta_ptr, eta_pitch,
                                                            hu_ptr, hu_pitch,
                                                            Hm_ptr, Hm_pitch, 
                                                            drifter_pos_x, drifter_pos_y,
                                                            dx, dy);
            const float v = water_velocity_bilinear_interpolation(eta_ptr, eta_pitch,
                                                            hv_ptr, hv_pitch,
                                                            Hm_ptr, Hm_pitch, 
                                                            drifter_pos_x, drifter_pos_y,
                                                            dx, dy);
        
            // Move drifter with a simple forward Euler
            drifter_pos_x += u*dt;
            drifter_pos_y += v*dt;
            
            // Add horizontal diffusion
            drifter_pos_x += horizontal_diffusivity*rand_numbers[0]*sqrt(dt);
            drifter_pos_y += horizontal_diffusivity*rand_numbers[1]*sqrt(dt);
           
            // Assuming periodic boundary conditions
            drifter_pos_x -= floor(drifter_pos_x / (nx*dx))*(nx*dx);
            drifter_pos_y -= floor(drifter_pos_y / (ny*dy))*(ny*dy);

            // Move drifter vertically.
            if (is_submerged(drifter)) {
                // Find the local water depth
                const int cell_id_x = (int)(floor(drifter_pos_x/dx) + 2);
                const int cell_id_y = (int)(floor(drifter_pos_y/dy) + 2);
                const float* Hm_row = (float*) ((char*) Hm_ptr + Hm_pitch*cell_id_y);
                const float water_depth = Hm_row[cell_id_x];

                vertical_transport(drifter_depth, droplet_diameter, water_density,
                                   oil_density, water_viscosity, g, dt, ksi_z, water_depth,
                                   vertical_diffusivity);
            }


            // Write to global memory
            drifter[0] = drifter_pos_x;
            drifter[1] = drifter_pos_y;
            drifter[2] = drifter_depth;
        }
    }
} // extern "C"
