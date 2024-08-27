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
#include "interpolation.cu"

#define DRY_EPS 1.0e-3f
#define LAND_VALUE 1.0e20f

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
    const int cell_id_x = (int)(floorf(drifter_pos_x/dx) + 2);
    const int cell_id_y = (int)(floorf(drifter_pos_y/dy) + 2);
    
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
    const int cell_id_x = (int)(floorf(drifter_pos_x/dx) + 2);
    const int cell_id_y = (int)(floorf(drifter_pos_y/dy) + 2);

    // Find neighbouring cells and relative position between cell centers
    float const frac_x = drifter_pos_x / dx - floorf(drifter_pos_x / dx);
    float const frac_y = drifter_pos_y / dy - floorf(drifter_pos_y / dy);
    
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

    vel_x0y0 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x0, cell_id_y0);
    vel_x1y0 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x1, cell_id_y0);
    vel_x0y1 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x0, cell_id_y1);
    vel_x1y1 = water_velocity(eta_ptr, eta_pitch, momentum_ptr, momentum_pitch, Hm_ptr, Hm_pitch, cell_id_x1, cell_id_y1);

    const float vel_y0 = (1-x_factor)*vel_x0y0 + x_factor * vel_x1y0; 
    const float vel_y1 = (1-x_factor)*vel_x0y1 + x_factor * vel_x1y1; 

    // Read and compute water velocity within cell
    return (1-y_factor)*vel_y0 + y_factor*vel_y1;
}

__device__ float wind(const float* wind_current_arr, const float* wind_next_arr,
                      const float wind_t_,
                      const float drifter_pos_x_, const float drifter_pos_y_,
                      const float domain_size_x_, const float domain_size_y_,
                      const int data_nx, const int data_ny) {

    //Normalize coordinates (to [0, 1])
    const float s = drifter_pos_x_ / domain_size_x_;
    const float t = drifter_pos_y_ / domain_size_y_;

    //Look up current and next timestep (using bilinear texture interpolation)
    const float current = bilinear_interpolation(wind_current_arr, data_nx, data_ny, s, t);
    const float next = bilinear_interpolation(wind_next_arr, data_nx, data_ny, s, t);

    //Interpolate in time
    return wind_t_*next + (1.0f - wind_t_)*current;
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
        if (fabsf(g_delro) > 0) {
            const float w1 = droplet_diameter*droplet_diameter * g_delro / (18.0f * water_viscosity);
            float w2 = 1.054f * sqrtf(droplet_diameter * fabsf(g_delro));
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

    return ksi * sqrtf(2 * vertical_diffusivity);
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
    droplet_depth = -fabsf(droplet_depth + diffusion_step); // Reflect off surface
    // Handling different sign conventions here
    droplet_depth = fmaxf(-(2 * water_depth + droplet_depth), droplet_depth); // Reflect off bottom
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
    droplet_depth = fminf(droplet_depth, 0.0f);
}

__device__ float white_cap_coverage(const float wind_speed) {
    // White cap coverage (fraction) for a wind speed (measured at 10m height).
    // This model is valid for wind speeds < 23.1 m/s
    if (wind_speed < 3.7f) {
        return 0.0f;
    }
    if (wind_speed < 10.187f ) {
        const float wind_diff = wind_speed - 3.7f;
        // 10e-3 / 100 = 10e-5, where dividing by 100 to convert percent to fraction
        return 3.18f * 10e-5f * wind_diff*wind_diff*wind_diff;
    } else {
        const float wind_diff = wind_speed + 1.98f;
        // 10e-4 / 100 = 10e-6, where dividing by 100 to convert percent to fraction
        return 4.82f * 10e-6f * wind_diff*wind_diff*wind_diff;
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
    return 1 - expf(-rate * dt);
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
        const float h_nodim = h_const * sqrtf(g * fetch) / wind_speed;
        wave_height = fminf(h_max, h_nodim) * wind_speed*wind_speed / g;
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
    return sqrtf(2 * g * wave_height) * oil_density * oil_film_thickness / oil_viscosity;
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

    return A * powf(We, -alpha)*(1 + B*powf((We / Re), alpha)) * oil_film_thickness; 
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
        const float d50v = expf(logf(d50n) + 3.0f*sigma*sigma);

        // Log normal distribution
        d50 = expf(random_number_normal * sigma + logf(d50v));
    }
}

__device__ bool is_dry_cell(float water_depth) {
    // Function to check if a cell is dry (land)
    return (fabsf(water_depth - LAND_VALUE) <= DRY_EPS);
}

__device__ float get_water_depth(float* Hm_ptr, int Hm_pitch, float abs_pos_x, float abs_pos_y, float dx, float dy) {
    // Function to get water depth at a given position
    const int cell_id_x = (int)(floorf(abs_pos_x/dx) + 2);
    const int cell_id_y = (int)(floorf(abs_pos_y/dy) + 2);
    const float* Hm_row = (float*) ((char*) Hm_ptr + Hm_pitch*cell_id_y);
    return Hm_row[cell_id_x];
}

extern "C" {
__global__ void superSimpleDrift(
        const int nx, const int ny,
        const float dx, const float dy, const float dt,

        float* eta_ptr, const int eta_pitch,
        float* hu_ptr, const int hu_pitch,
        float* hv_ptr, const int hv_pitch,
        float* Hm_ptr, const int Hm_pitch,
        float* Bi_ptr, const int Bi_pitch,

        const int num_drifters,
        const int num_active_drifters,
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
        const float* wind_x_current_arr,
        const float* wind_y_current_arr,
        const float* wind_x_next_arr,
        const float* wind_y_next_arr,
        const float wind_interpolation_t,
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
        if ((ti < num_drifters) && (ti < num_active_drifters)) {


            // Obtain pointer to our Lagrangian drifter:
            float* relative_position = (float*) ((char*) relative_positions + relative_positions_pitch*ti);
            float rel_pos_x = relative_position[0];
            float rel_pos_y = relative_position[1];
            float drifter_depth = relative_position[2];

            // stranded
            if (drifter_depth == 999) {
                return;
            }

            // Generate random numbers
            unsigned long long* const seed_row = (unsigned long long*) ((char*) seed_ptr + seed_pitch * ti);
            unsigned long long seed = seed_row[0];
            float2 rand_n1 = rand_normal(&seed);
            float2 rand_n2 = rand_normal(&seed);


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
                                                            dx, dy);
            const float v = water_velocity_bilinear_interpolation(eta_ptr, eta_pitch,
                                                            hv_ptr, hv_pitch,
                                                            Hm_ptr, Hm_pitch, 
                                                            abs_pos_x, abs_pos_y,
                                                            dx, dy);

            // Move drifter with a simple forward Euler
            rel_pos_x += u*dt;
            rel_pos_y += v*dt;

            // Move drifter vertically.
            if (is_submerged(drifter_depth)) {
                // Random number for vertical diffusion (normal distribution with mean 0 and variance dt)
                float water_depth = get_water_depth(Hm_ptr, Hm_pitch, abs_pos_x, abs_pos_y, dx, dy); // should this pos be after drift update?
                const float ksi_z = rand_n2.x * sqrtf(dt);
                vertical_transport(drifter_depth, d50, water_density,
                                   oil_density, water_viscosity, g, dt, ksi_z, water_depth,
                                   vertical_diffusivity);
            }
            else {
                const float wind_u = wind(wind_x_current_arr, wind_x_next_arr, 
                                          wind_interpolation_t, 
                                          abs_pos_x, abs_pos_y, 
                                          nx*dx, ny*dy, 
                                          WIND_X_NX, WIND_X_NY);
                const float wind_v = wind(wind_y_current_arr, wind_y_next_arr, 
                                          wind_interpolation_t, 
                                          abs_pos_x, abs_pos_y, 
                                          nx*dx, ny*dy, 
                                          WIND_Y_NX, WIND_Y_NY);

                // Advection
                rel_pos_x += windage*wind_u*dt;
                rel_pos_y += windage*wind_v*dt;

                // Entrainment of surface drifter
                if (ENABLE_ENTRAINMENT) {
                    const float wind_speed = sqrtf(wind_u*wind_u + wind_v*wind_v);
                    // Create 2 random numbers from a uniform distribution
                    const float2 rand_u = rand_uniform(&seed);
                    entrain(drifter_depth, d50, wind_speed, g, dt, rand_u, rand_n2.y, oil_density, oil_viscosity, oil_water_ift, oil_film_thickness);
                }
            }

            // after drift and advection we need to check if the particle has stranded
            abs_pos_x = rel_pos_x + ref_pos_x;
            abs_pos_y = rel_pos_y + ref_pos_y;
            float water_depth = get_water_depth(Bi_ptr, Bi_pitch, abs_pos_x, abs_pos_y, dx, dy);

            if (is_dry_cell(water_depth)) {
                // Stranded
                drifter_depth = 999;
            }
            
            // Add horizontal diffusion
            // If a particle is randomly displaced onto land, put it back where it came from
            auto new_rel_pos_x = rel_pos_x + rand_n1.x*sqrtf(2*horizontal_diffusivity*dt);
            auto new_rel_pos_y = rel_pos_y + rand_n1.y*sqrtf(2*horizontal_diffusivity*dt);

            abs_pos_x = new_rel_pos_x + ref_pos_x;
            abs_pos_y = new_rel_pos_y + ref_pos_y;

            water_depth = get_water_depth(Bi_ptr, Bi_pitch, abs_pos_x, abs_pos_y, dx, dy);
            // Only move particle if new cell is not land
            if (!is_dry_cell(water_depth)) {
                rel_pos_x = new_rel_pos_x;
                rel_pos_y = new_rel_pos_y;
            }
            
            // Assuming periodic boundary conditions
            boundary_conditions(rel_pos_x, rel_pos_y, 
                                ref_pos_x, ref_pos_y, 
                                nx, ny, dx, dy);

            // Write to global memory
            // Write seed back to global memory
            seed_row[0] = seed;
            relative_position[0] = rel_pos_x;
            relative_position[1] = rel_pos_y;
            relative_position[2] = drifter_depth;
            droplet_diameter[0] = d50;
        }
    }
} // extern "C"
