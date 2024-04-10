/*
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This CUDA kernel implements a selection of drift trajectory algorithms.

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



/**
  * Kernel that evolves drifter positions along u and v.
  */
  
//Code relating to wind-data

texture<float, cudaTextureType2D> wind_X_current;
texture<float, cudaTextureType2D> wind_X_next;

texture<float, cudaTextureType2D> wind_Y_current;
texture<float, cudaTextureType2D> wind_Y_next;

__device__ float windX(const float wind_t_, 
                       const float drifter_pos_x_, const float drifter_pos_y_, 
                       const float domain_size_x_, const float domain_size_y_) {
    
    //Normalize coordinates (to [0, 1])
    const float s = drifter_pos_x_ / domain_size_x_;
    const float t = drifter_pos_y_ / domain_size_y_;
    
    //Look up current and next timestep (using bilinear texture interpolation)
    const float current = tex2D(wind_X_current, s, t);
    const float next = tex2D(wind_X_next, s, t);
    
    //Interpolate in time
    return wind_t_*next + (1.0f - wind_t_)*current;
}

__device__ float windY(const float wind_t_, 
                       const float drifter_pos_x_, const float drifter_pos_y_, 
                       const float domain_size_x_, const float domain_size_y_) {
    
    //Normalize coordinates (to [0, 1])
    const float s = drifter_pos_x_ / domain_size_x_;
    const float t = drifter_pos_y_ / domain_size_y_;
    
    //Look up current and next timestep (using bilinear texture interpolation)
    const float current = tex2D(wind_Y_current, s, t);
    const float next = tex2D(wind_Y_next, s, t);
    
    //Interpolate in time
    return wind_t_*next + (1.0f - wind_t_)*current;
}




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
__global__ void passiveDrifterKernel(
        //Discretization parameters
        const int nx_, const int ny_,
        const float dx_, const float dy_, const float dt_,

        const int x_zero_reference_cell_, // the cell column representing x0 (x0 at western face)
        const int y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)
        
        // Data
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_, const int hu_pitch_,
        float* hv_ptr_, const int hv_pitch_,
        float* Hm_ptr_, const int Hm_pitch_,

        const int periodic_north_south_,
        const int periodic_east_west_,
        
        const int num_drifters_,
        float* drifters_positions_, const int drifters_pitch_,
        const float sensitivity_,
        const float wind_t_, 
        const float wind_drift_factor_) 
        {

    //Index of thread within block (only needed in one dim)
    const int tx = threadIdx.x;
        
    //Index of block within domain (only needed in one dim)
    const int bx = blockDim.x * blockIdx.x;
        
    //Index of cell within domain (only needed in one dim)
    const int ti = bx + tx;
    
    if (ti < num_drifters_ + 1) {
        // Obtain pointer to our particle:
        float* drifter = (float*) ((char*) drifters_positions_ + drifters_pitch_*ti);
        float drifter_pos_x = drifter[0];
        float drifter_pos_y = drifter[1];
        
        // Find cell ID for the cell in which our particle is
        int const cell_id_x = (int)(floor(drifter_pos_x/dx_) + x_zero_reference_cell_);
        int const cell_id_y = (int)(floor(drifter_pos_y/dy_) + y_zero_reference_cell_);
        
        float const frac_x = drifter_pos_x / dx_ - floor(drifter_pos_x / dx_);
        float const frac_y = drifter_pos_y / dy_ - floor(drifter_pos_y / dy_);
        
        const int cell_id_x0 = frac_x < 0.5f ? cell_id_x - 1 : cell_id_x;
        const float x_factor = frac_x < 0.5f ? frac_x + 0.5f : frac_x - 0.5f; 
        const int cell_id_x1 = cell_id_x0 + 1;

        const int cell_id_y0 = frac_y < 0.5f ? cell_id_y - 1 : cell_id_y;
        const float y_factor = frac_y < 0.5f ? frac_y + 0.5f : frac_y - 0.5f; 
        const int cell_id_y1 = cell_id_y0 + 1;
                
        float const u_x0y0 = waterVelocity(eta_ptr_, eta_pitch_,hu_ptr_, hu_pitch_,Hm_ptr_, Hm_pitch_, cell_id_x0, cell_id_y0);
        float const u_x1y0 = waterVelocity(eta_ptr_, eta_pitch_,hu_ptr_, hu_pitch_,Hm_ptr_, Hm_pitch_, cell_id_x1, cell_id_y0);
        float const u_x0y1 = waterVelocity(eta_ptr_, eta_pitch_,hu_ptr_, hu_pitch_,Hm_ptr_, Hm_pitch_, cell_id_x0, cell_id_y1);
        float const u_x1y1 = waterVelocity(eta_ptr_, eta_pitch_,hu_ptr_, hu_pitch_,Hm_ptr_, Hm_pitch_, cell_id_x1, cell_id_y1);
        
        float const v_x0y0 = waterVelocity(eta_ptr_, eta_pitch_,hv_ptr_, hv_pitch_,Hm_ptr_, Hm_pitch_, cell_id_x0, cell_id_y0);
        float const v_x1y0 = waterVelocity(eta_ptr_, eta_pitch_,hv_ptr_, hv_pitch_,Hm_ptr_, Hm_pitch_, cell_id_x1, cell_id_y0);
        float const v_x0y1 = waterVelocity(eta_ptr_, eta_pitch_,hv_ptr_, hv_pitch_,Hm_ptr_, Hm_pitch_, cell_id_x0, cell_id_y1);
        float const v_x1y1 = waterVelocity(eta_ptr_, eta_pitch_,hv_ptr_, hv_pitch_,Hm_ptr_, Hm_pitch_, cell_id_x1, cell_id_y1);
        
        float const u_y0 = (1-x_factor)*u_x0y0 + x_factor * u_x1y0; 
        float const u_y1 = (1-x_factor)*u_x0y1 + x_factor * u_x1y1; 
        
        float const v_y0 = (1-x_factor)*v_x0y0 + x_factor * v_x1y0; 
        float const v_y1 = (1-x_factor)*v_x0y1 + x_factor * v_x1y1;
        
        float u = (1-y_factor)*u_y0 + y_factor *u_y1;
        float v = (1-y_factor)*v_y0 + y_factor *v_y1;
        
        if (wind_drift_factor_) {
            float* const Hm_row_y = (float*) ((char*) Hm_ptr_ + Hm_pitch_*cell_id_y);
            float const Hm = Hm_row_y[cell_id_x];
            if (Hm < 1e20){ //using mask_value from Common.Bathymetry-class 
                u = u + (windX(wind_t_, drifter_pos_x, drifter_pos_y, nx_*dx_, ny_*dy_) - u) * wind_drift_factor_;
                v = v + (windY(wind_t_, drifter_pos_x, drifter_pos_y, nx_*dx_, ny_*dy_) - v) * wind_drift_factor_;
            }
        }
        
        // Move drifter
        drifter_pos_x += sensitivity_*u*dt_;
        drifter_pos_y += sensitivity_*v*dt_;
            
        // Ensure boundary conditions
        if (periodic_east_west_ && (drifter_pos_x < 0)) {
            drifter_pos_x += + nx_*dx_;
        }
        if (periodic_east_west_ && (drifter_pos_x > nx_*dx_)) {
            drifter_pos_x -= nx_*dx_;
        }
        if (periodic_north_south_ && (drifter_pos_y < 0)) {
            drifter_pos_y += ny_*dy_;
        }
        if (periodic_north_south_ && (drifter_pos_y > ny_*dy_)) {
            drifter_pos_y -= ny_*dy_;
        }

        // Write to global memory
        drifter[0] = drifter_pos_x;
        drifter[1] = drifter_pos_y;
    }
}
} // extern "C"
    

extern "C" {
__global__ void enforceBoundaryConditions(
        //domain parameters
        float domain_size_x_, float domain_size_y_,

        int periodic_north_south_,
        int periodic_east_west_,
        
        int num_drifters_,
        float* drifters_positions_, int drifters_pitch_) {
    
    //Index of drifter (only needed in one dimension)
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ti < num_drifters_ + 1) {
        // Obtain pointer to our particle:
        float* drifter = (float*) ((char*) drifters_positions_ + drifters_pitch_*ti);
        float drifter_pos_x = drifter[0];
        float drifter_pos_y = drifter[1];

        // Ensure boundary conditions
        if (periodic_east_west_ && (drifter_pos_x < 0)) {
            drifter_pos_x += + domain_size_x_;
        }
        if (periodic_east_west_ && (drifter_pos_x > domain_size_x_)) {
            drifter_pos_x -= domain_size_x_;
        }
        if (periodic_north_south_ && (drifter_pos_y < 0)) {
            drifter_pos_y += domain_size_y_;
        }
        if (periodic_north_south_ && (drifter_pos_y > domain_size_y_)) {
            drifter_pos_y -= domain_size_y_;
        }

        // Write to global memory
        drifter[0] = drifter_pos_x;
        drifter[1] = drifter_pos_y;
    }
}
} // extern "C"
