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


__device__ float array_lookup_2d(
        const float* data_ptr, const int data_pitch,
        const int cell_id_x, const int cell_id_y) {

    const float* data_row = (float*)((char*) data_ptr + data_pitch*cell_id_y);
    return data_row[cell_id_x];
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
        const float dx, const float dy,
        const bool momentum_is_wind) {
    
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
    
    float vel_x0y0; 
    float vel_x1y0; 
    float vel_x0y1; 
    float vel_x1y1; 

    if (momentum_is_wind) {
        vel_x0y0 = array_lookup_2d(momentum_ptr, momentum_pitch, cell_id_x0, cell_id_y0);
        vel_x1y0 = array_lookup_2d(momentum_ptr, momentum_pitch, cell_id_x1, cell_id_y0);
        vel_x0y1 = array_lookup_2d(momentum_ptr, momentum_pitch, cell_id_x0, cell_id_y1);
        vel_x1y1 = array_lookup_2d(momentum_ptr, momentum_pitch, cell_id_x1, cell_id_y1);
    }
    else {
        vel_x0y0 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x0, cell_id_y0);
        vel_x1y0 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x1, cell_id_y0);
        vel_x0y1 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x0, cell_id_y1);
        vel_x1y1 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x1, cell_id_y1);
    }    
    const float vel_y0 = (1-x_factor)*vel_x0y0 + x_factor * vel_x1y0; 
    const float vel_y1 = (1-x_factor)*vel_x0y1 + x_factor * vel_x1y1; 

    // Read and compute water velocity within cell
    return (1-y_factor)*vel_y0 + y_factor*vel_y1;
}

__device__ void boundary_conditions(
        float& rel_pos_x, float& rel_pos_y,
        const float ref_pos_x, const float ref_pos_y,
        const int nx, const int ny,
        const float dx, const float dy) {
    // Deal with domain-related boundary conditions 

    const float abs_pos_x = rel_pos_x + ref_pos_x;
    const float abs_pos_y = rel_pos_y + ref_pos_y;
    // TODO: What do we do with "open" boundary conditions
    // Assuming periodic boundary conditions
    if (abs_pos_x < 0.0f) {
        rel_pos_x += nx*dx;
    } else if (abs_pos_x > nx*dx) {
        rel_pos_x -= nx*dx;
    }
    if (abs_pos_y < 0.0f) {
        rel_pos_y += ny*dy;
    } else if (abs_pos_y > ny*dy) {
        rel_pos_y -= ny*dy;
    }
}

__device__ float rise_velocity(
        const float droplet_diameter,
        const float water_density,
        const float oil_density,
        const float water_viscosity,
        const float g) {
    // Calculate the rise velocity of a droplet in m/s.

    float rise_velocity = 0.0f;
    if (droplet_diameter > 0.0f) {
        const float g_delro = g * (water_density - oil_density) / water_density;
        if (abs(g_delro) > 0) {
            const float w1 = pow(droplet_diameter, 2) * g_delro / (18.0f * water_viscosity);
            float w2 = 1.054f * sqrt(droplet_diameter * abs(g_delro));
            w2 = copysignf(w2, g_delro);
            rise_velocity = w1 * w2 / (w1 + w2); // in m/s
        }
    }

    return rise_velocity;
}

__device__ float is_submerged(const float drifter_depth) {
    return drifter_depth < 0.0f;
}

__device__ float euler_maruyama_scheme(
    const float ksi,
    const float vertical_diffusivity) {
    // Euler-Maruyama scheme assuming constant diffusivity. Return vertical displacement due to diffusivity.

    return ksi * sqrt(2 * vertical_diffusivity);
}

__device__ void vertical_transport(
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

    // Conventions:
    // droplet_depth is 0 at the surface and negative downwards
    // water_depth is a positive number
    droplet_depth = -abs(droplet_depth + diffusion_step); // Reflect off surface
    // Handling different sign conventions here
    droplet_depth = max(-(2 * water_depth + droplet_depth), droplet_depth); // Reflect off bottom
    // Handle droplets above surface after reflection
    // by putting them in the middle of the water column
    // (should only happen very rarely, and only in very shallow water)
    if (droplet_depth > 0.0f) {
	// minus sign due to different conventions
        droplet_depth = -water_depth * 0.5f;
    }

    // Calculate the rise velocity due to buoyancy
    const float rise_vel = rise_velocity(droplet_diameter, water_density, oil_density, water_viscosity, g);
    // Vertical advection step in m
    const float advection_step = rise_vel * dt;
    droplet_depth += advection_step;
}

__device__ float white_cap_coverage(const float wind_speed) {
    // White cap coverage (fraction) for a wind speed (measured at 10m height).
    // This model is valid for wind speeds < 23.1 m/s
    if (wind_speed < 3.7f) {
        return 0.0f;
    }
    if (wind_speed < 10.187f ) {
        // Dividing by 100 to convert percent to fraction
        return 3.18f * 10e-3 * powf(wind_speed - 3.7f, 3) / 100;
    } else {
        // Dividing by 100 to convert percent to fraction
        return 4.82f * 10e-4 * powf(wind_speed + 1.98f, 3) / 100;
    }
}


__device__ float mean_wave_period(const float wind_speed, const float g) {
    // Mean wave period calculate from the wind speed.
    const float period = 0.812f * 3.14f * wind_speed / g;
    return period;
}

__device__ float entrainment_rate(const float wind_speed, const float g) {
    // Entrainment rate [s**-1]
    float rate = 0.0f;
    const float wave_period = mean_wave_period(wind_speed, g);

    if (wave_period > 0.0f) {
        const float white_cap_cov = white_cap_coverage(wind_speed);
        rate = white_cap_cov / wave_period;
    }

    return rate;
}

__device__ float entrainment_probability(const float wind_speed, const float g, const float dt) {
    // Probability of entrainment of a surface particle.
    const float rate = entrainment_rate(wind_speed, g);
    return 1 - exp(-rate * dt);
}

__device__ float significant_wave_height(const float wind_speed, const float fetch, const float g) {
    // Calculate the significant wave height [m]
    // Based on the JONSWAP model and associated empirical relations.
    // See Carter (1982) for details.
    // windspeed: windspeed [m/s]
    // fetch: fetch [m]
    float wave_height = 0.0f;

    // Avoid division by 0
    if (wind_speed > 0.0f) {
        // Constants for the JONSWAP model:
        const float h_max   = 0.243f;   // Nondimensional height maximum.
        const float h_const = 0.0016f;  // Nondimensional height constant.

        // Calculate wave height
        const float h_nodim = h_const * sqrt(g * fetch / pow(wind_speed, 2));
        wave_height = min(h_max, h_nodim) * pow(wind_speed, 2) / g;
    }

    return wave_height;
}

__device__ float weber_number(
        const float oil_density,
        const float oil_film_thickness,
        const float oil_water_ift,
        const float wave_height,
        const float g) {
    // Calculate Weber number (Johansen 2015) 
    return 2 * g * wave_height * oil_density * oil_film_thickness / oil_water_ift;
}

__device__ float reynolds_number(
        const float oil_density,
        const float oil_film_thickness,
        const float oil_viscosity,
        const float wave_height,
        const float g) {
    // Calculate Reynold number (Johansen 2015)            
    return sqrt(2 * g * wave_height) * oil_density * oil_film_thickness / oil_viscosity;
}

__device__ float weber_natural_dispersion_d50(
        const float oil_density,
        const float oil_viscosity,
        const float oil_water_ift,
        const float oil_film_thickness,
        const float wave_height,
        const float g) {
    // Weber natural dispersion model (Johansen 2015). Predicts median droplet size D50 in m.
    // oil_density: [kg/m**3]
    // oil_viscosity: [kg/m/s]
    // oil_water_ift: oil-water interfacial tension [N/m]
    // oil_film_thickness: thickness of oil film on the surface [mm]
    // wave_height: significant wave height [m]
    // g: acceleration of gravity [m/s**2]
    const float We = weber_number(oil_density, oil_film_thickness, oil_water_ift, wave_height, g);
    const float Re = reynolds_number(oil_density, oil_film_thickness, oil_viscosity, wave_height, g);

    const float A = 2.251f;
    const float B = 0.027f;
    const float alpha = 0.6f;

    return A * pow(We, -alpha)*(1 + B*pow((We / Re), alpha)) * oil_film_thickness; 
}

__device__ void entrain(
    float &drifter_depth,
    float &d50,
    const float wind_speed,
    const float g,
    const float dt,
    const float2 random_numbers_uniform,
    const float random_number_normal,
    const float oil_density,
    const float oil_viscosity,
    const float oil_water_ift, 
    const float oil_film_thickness
    )
{
    // Entrainment of particles by breaking waves.
    // drifter_depth: [m]
    // d50: volume median droplet diameter [m]
    // wind_speed: absolute wind speed [m/s]
    // g: acceleration of gravity [m/s**2]
    // dt: timestep [s]
    // random_numbers_uniform: 2 random numbers from a uniform distribution on [0, 1]
    // random_number_normal: random number from a normal distribution with variance=1 and mean=0
    // oil_density: [kg/m**3]
    // oil_viscosity: [kg/m/s]
    // oil_water_ift: oil-water interfacial tension [N/m]
    // oil_film_thickness: thickness of oil film on the surface [mm]
    const float random_number_1 = random_numbers_uniform.x;
    const float random_number_2 = random_numbers_uniform.y;
    if (random_number_1 < entrainment_probability(wind_speed, g, dt)) {
        const float Hs = significant_wave_height(wind_speed, 100000, g);
        const float low = Hs * (1.5f-0.35f);
        const float high = Hs * (1.5f+0.35f);
        // Postition the entrained particle at a random depth in the range [-Hs * (1.5-0.35), -Hs * (1.5+0.35)]
        drifter_depth = -(random_number_2 * (high - low) + low);
        const float d50n = weber_natural_dispersion_d50(oil_density, oil_viscosity, oil_water_ift, oil_film_thickness, Hs, g);

        // From number size distribution to volume size distribution.
        const float sigma = 0.921034f;
        const float d50v = exp(log(d50n) + 3.0f*pow(sigma, 2));

        // Log normal distribution
        d50 = exp(random_number_normal * sigma + log(d50v));
    }
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
        if (i < (int)(floor(n/2.0f))) {
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
        float* relative_positions, const int relative_positions_pitch,
        const float* reference_positions, const int reference_positions_pitch,
        unsigned long long* seed_ptr, int seed_pitch, 
        const float horizontal_diffusivity,
        const float vertical_diffusivity,
        float* droplet_diameters, const int droplet_diameter_pitch,
        const float oil_density, const float water_density,
        const float oil_viscosity, const float water_viscosity,
        const float oil_film_thickness, const float oil_water_ift,
        const float g,
        float* wind_u_ptr, const int wind_u_pitch,
        float* wind_v_ptr, const int wind_v_pitch,
        const float windage)

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
            
            // Obtain pointer to our Lagrangian drifter:
            float* relative_position = (float*) ((char*) relative_positions + relative_positions_pitch*ti);
            float rel_pos_x = relative_position[0];
            float rel_pos_y = relative_position[1];
            float drifter_depth = relative_position[2];

            float* reference_position = (float*) ((char*) reference_positions + reference_positions_pitch*ti);
            float ref_pos_x = reference_position[0];
            float ref_pos_y = reference_position[1];
            
            float abs_pos_x = rel_pos_x + ref_pos_x;
            float abs_pos_y = rel_pos_y + ref_pos_y;
            
            // Pointer to droplet diameter
            float* droplet_diameter = (float*) ((char*) droplet_diameters + droplet_diameter_pitch*ti);
            float d50 = droplet_diameter[0];

            // Read and compute water velocity within cell
            const float u = water_velocity_bilinear_interpolation(eta_ptr, eta_pitch,
                                                            hu_ptr, hu_pitch,
                                                            Hm_ptr, Hm_pitch, 
                                                            abs_pos_x, abs_pos_y,
                                                            dx, dy, false);
            const float v = water_velocity_bilinear_interpolation(eta_ptr, eta_pitch,
                                                            hv_ptr, hv_pitch,
                                                            Hm_ptr, Hm_pitch, 
                                                            abs_pos_x, abs_pos_y,
                                                            dx, dy, false);
            
            // Move drifter with a simple forward Euler
            rel_pos_x += u*dt;
            rel_pos_y += v*dt;
            
            // Add horizontal diffusion
            rel_pos_x += rand_numbers[0]*sqrt(2*horizontal_diffusivity*dt);
            rel_pos_y += rand_numbers[1]*sqrt(2*horizontal_diffusivity*dt);
           
            
            // Move drifter vertically.
            if (is_submerged(drifter_depth)) {
                // Find the local water depth
                const int cell_id_x = (int)(floor(abs_pos_x/dx) + 2);
                const int cell_id_y = (int)(floor(abs_pos_y/dy) + 2);
                const float* Hm_row = (float*) ((char*) Hm_ptr + Hm_pitch*cell_id_y);
                const float water_depth = Hm_row[cell_id_x];

                // Random number for vertical diffusion (normal distribution with mean 0 and variance dt)
                const float ksi_z = rand_numbers[2] * sqrt(dt);
                vertical_transport(drifter_depth, d50, water_density,
                                   oil_density, water_viscosity, g, dt, ksi_z, water_depth,
                                   vertical_diffusivity);
            }
            else {
                // Influence from wind for surface drifters
                const float wind_u = water_velocity_bilinear_interpolation(nullptr, 0,
                                                            wind_u_ptr, wind_u_pitch,
                                                            nullptr, 0, 
                                                            abs_pos_x, abs_pos_y,
                                                            dx, dy, true);
                const float wind_v = water_velocity_bilinear_interpolation(nullptr, 0,
                                                            wind_v_ptr, wind_v_pitch,
                                                            nullptr, 0, 
                                                            abs_pos_x, abs_pos_y,
                                                            dx, dy, true);

                // Advection
                rel_pos_x += windage*wind_u*dt;
                rel_pos_y += windage*wind_v*dt;
            
                // Create 2 random numbers from a uniform distribution
                unsigned long long* const seed_row = (unsigned long long*) ((char*) seed_ptr + seed_pitch*ti);
                unsigned long long seed = seed_row[0];
                const float2 rand_u = rand_uniform(&seed);
                // Write seed back to global memory
                seed_row[0] = seed;

                // Entrainment of surface drifter
                if (ENABLE_ENTRAINMENT) {
                    const float wind_speed = sqrt(pow(wind_u, 2) + pow(wind_v, 2));
                    entrain(drifter_depth, d50, wind_speed, g, dt, rand_u, rand_numbers[3], oil_density, oil_viscosity, oil_water_ift, oil_film_thickness);
                }
            }

            // Assuming periodic boundary conditions
            boundary_conditions(rel_pos_x, rel_pos_y, 
                                ref_pos_x, ref_pos_y, 
                                nx, ny, dx, dy);
            
            // Write to global memory
            relative_position[0] = rel_pos_x;
            relative_position[1] = rel_pos_y;
            relative_position[2] = drifter_depth;
            droplet_diameter[0] = d50;
        }
    }
} // extern "C"
