/*
This software is part of GPU Ocean

Copyright (C) 2023 SINTEF Digital
Copyright (C) 2023 Norwegian Meteorological Institute

These CUDA kernels generate random fields based on Karhunen-Loeve
basis functions.

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


texture<float, cudaTextureType2D> coriolis_f_tex;
texture<float, cudaTextureType2D> angle_tex;


/**
  * Returns the coriolis parameter f from the coriolis texture. 
  * @param i Cell number along x-axis, starting from (0, 0) corresponding to first cell in domain after global ghost cells
  * @param j Cell number along y-axis
  * @param nx_ Number of cells in internal domain (excluding the four ghost cells)
  * @param ny_ Number of cells in internal domain (excluding the four ghost cells)
  * The texture is assumed to also cover the ghost cells (same shape/extent as eta)
  */
__device__
inline float coriolisF(const int i, const int j, const int nx_, const int ny_) {
    //nx+4 to account for ghost cells
    //+0.5f to go to center of texel
    const float s = (i+0.5f) / (nx_+4.0f); 
    const float t = (j+0.5f) / (ny_+4.0f);
    //FIXME: Should implement so that subsampling does not get border issues, see
    //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-lookup
    return tex2D(coriolis_f_tex, s, t);
}



/**
  * Decompose the north vector to x and y coordinates
  * @param i Cell number along x-axis, starting from (0, 0) corresponding to first cell in domain after global ghost cells
  * @param j Cell number along y-axis
  * @param nx_ Number of cells in internal domain (excluding the four ghost cells)
  * @param ny_ Number of cells in internal domain (excluding the four ghost cells)
  */
__device__
inline float2 getNorth(const int i, const int j, const int nx_, const int ny_) {
    //nx+4 to account for ghost cells
    //+0.5f to go to center of texel
    const float s = (i+0.5f) / (nx_+4.0f);
    const float t = (j+0.5f) / (ny_+4.0f);
    const float angle = tex2D(angle_tex, s, t);
    //FIXME: Should implement so that subsampling does not get border issues, see
    //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-lookup
    return make_float2(sinf(angle), cosf(angle));
}
// FIXME: coriolisF and getNorth are both copied from cdklm_kernel.cu. These should
//        be implemented only once and included - but be careful to ensure that 
//        the textures are defined to avoid compilation errors!




/**
  * Kernel that adds a perturbation to the input field eta.
  * The kernel use Karhunen-Loeve type basis functions with rolling to perturb the eta fields.
  */
extern "C" {
__global__ void kl_sample_eta(
        // Size of computational data
        const int nx_, const int ny_,

        // Parameters related to the KL basis functions
        const int basis_x_start_, const int basis_x_end_,
        const int basis_y_start_, const int basis_y_end_,
        const int include_cos_, const int include_sin_,
        const float kl_decay_, const float kl_scaling_,
        const float roll_x_sin_, const float roll_y_sin_,
        const float roll_x_cos_, const float roll_y_cos_,

        // Normal distributed random numbers of size
        // [basis_x_end - basis_x_start + 1, 2*(basis_y_end - basis_y_start + 1)]
        // (one per KL basis function)
        float* random_ptr_, const int random_pitch_,

        // Ocean data variables - size [nx + 4, ny + 4]
        // Write to interior cells only,  [2:nx+2, 2:ny+2]
        float* eta_ptr_, const int eta_pitch_
    ) 
    {
        // Each thread is responsible for one grid point in the computational grid.
        
        //Index of cell within block
        const int tx = threadIdx.x; 
        const int ty = threadIdx.y;

        //Index of start of block within domain
        const int bx = blockDim.x * blockIdx.x; // Compansating for ghost cells
        const int by = blockDim.y * blockIdx.y; // Compensating for ghost cells

        //Index of cell within domain
        const int ti = bx + tx;
        const int tj = by + ty;

        // relative location on the unit square
        const float x = (ti + 0.5)/nx_;
        const float y = (tj + 0.5)/ny_;

        const float x_sin = x + roll_x_sin_;
        const float y_sin = y + roll_y_sin_;
        const float x_cos = x + roll_x_cos_;
        const float y_cos = y + roll_y_cos_;
        
        // Shared memory for random numbers
        __shared__ float rns[rand_ny][rand_nx];

        // Load random numbers into shmem
        for (int j = ty; j < rand_ny; j += blockDim.y) {
            float* const random_row_ = (float*) ((char*) random_ptr_ + random_pitch_*j);
            for (int i = tx; i < rand_nx; i += blockDim.x) {
                rns[j][i] = random_row_[i];
            }
        }
        __syncthreads();

        const int num_basis_x = basis_x_end_ - basis_x_start_ + 1;
        const int num_basis_y = basis_y_end_ - basis_y_start_ + 1;

        // Sample from the KL basis functions
        float d_eta = 0.0f;

        if (include_sin_) {
            for (int j = 0; j < num_basis_y; j++) {
                const int m = basis_y_start_ + j;
                for (int i = 0; i < num_basis_x; i++) {
                    const int n = basis_x_start_ + i;

                    d_eta += kl_scaling_ * rns[j][i] *
                             powf(m, -kl_decay_) * powf(n, -kl_decay_) *
                             sinpif(2*m*y_sin) * sinpif(2*n*x_sin);
                }
            }
        }

        if (include_cos_) {
            for (int j = 0; j < num_basis_y; j++) {
                const int m = basis_y_start_ + j;
                for (int i = 0; i < num_basis_x; i++) {
                    const int n = basis_x_start_ + i;

                    d_eta += kl_scaling_ * rns[num_basis_y + j][i] *
                             powf(m, -kl_decay_) * powf(n, -kl_decay_) *
                             cospif(2*m*y_cos) * cospif(2*n*x_cos);
                }
            }
        }

        if (ti < nx_ && tj < ny_ ) {
            float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*(tj + 2));
            eta_row[ti + 2] += d_eta;
        }
    }

} // extern "C"



/**
  * Kernel that adds a perturbation to the input field eta.
  * The kernel use Karhunen-Loeve type basis functions with rolling to perturb the eta fields.
  */
extern "C" {
__global__ void kl_sample_ocean_state(
        // Size of computational data
        const int nx_, const int ny_,
        const float dx_, const float dy_,

        // physical parameters
        const float g_, const float f_, const float beta_,

        // Parameters related to the KL basis functions
        const int basis_x_start_, const int basis_x_end_,
        const int basis_y_start_, const int basis_y_end_,
        const int include_cos_, const int include_sin_,
        const float kl_decay_, const float kl_scaling_,
        const float roll_x_sin_, const float roll_y_sin_,
        const float roll_x_cos_, const float roll_y_cos_,

        // Normal distributed random numbers of size
        // [basis_x_end - basis_x_start + 1, 2*(basis_y_end - basis_y_start + 1)]
        // (one per KL basis function)
        float* random_ptr_, const int random_pitch_,

        // Ocean data variables - size [nx + 4, ny + 4]
        // Write to interior cells only,  [2:nx+2, 2:ny+2]
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_, const int hu_pitch_,
        float* hv_ptr_, const int hv_pitch_,

        // Ocean data parameter - size [nx + 5, ny + 5]
        float* Hi_ptr_, const int Hi_pitch_,
        const float land_value_
    ) 
    {
        // Each thread is responsible for one grid point in the computational grid.
        
        //Index of cell within block
        const int tx = threadIdx.x; 
        const int ty = threadIdx.y;

        //Index of start of block within domain
        const int bx = (blockDim.x-2) * blockIdx.x + 1; // Compansating for ghost cells
        const int by = (blockDim.y-2) * blockIdx.y + 1; // Compensating for ghost cells

        //Index of cell within domain
        const int ti = bx + tx;
        const int tj = by + ty;

        // relative location on the unit square
        const float x = (ti + 0.5 - 2)/nx_;
        const float y = (tj + 0.5 - 2)/ny_;

        const float x_sin = x + roll_x_sin_;
        const float y_sin = y + roll_y_sin_;
        const float x_cos = x + roll_x_cos_;
        const float y_cos = y + roll_y_cos_;
        
        // Shared memory for random numbers
        __shared__ float rns[rand_ny][rand_nx];

        // Shared memory for eta perturbation
        __shared__ float d_eta_shmem[block_height][block_width];

        // Shared memory for Hi
        __shared__ float Hi_shmem[block_height+1][block_width+1];
        
        // Load random numbers into shmem
        for (int j = ty; j < rand_ny; j += blockDim.y) {
            float* const random_row_ = (float*) ((char*) random_ptr_ + random_pitch_*j);
            for (int i = tx; i < rand_nx; i += blockDim.x) {
                rns[j][i] = random_row_[i];
            }
        }
        __syncthreads();

        const int num_basis_x = basis_x_end_ - basis_x_start_ + 1;
        const int num_basis_y = basis_y_end_ - basis_y_start_ + 1;

        // Sample from the KL basis functions
        float d_eta = 0.0f;

        if (include_sin_) {
            for (int j = 0; j < num_basis_y; j++) {
                const int m = basis_y_start_ + j;
                for (int i = 0; i < num_basis_x; i++) {
                    const int n = basis_x_start_ + i;

                    //d_eta += 1.0f; 
                    d_eta += kl_scaling_ * rns[j][i] *
                             powf(m, -kl_decay_) * powf(n, -kl_decay_) *
                             sinpif(2*m*y_sin) * sinpif(2*n*x_sin);

                }
            }
        }

        if (include_cos_) {
            for (int j = 0; j < num_basis_y; j++) {
                const int m = basis_y_start_ + j;
                for (int i = 0; i < num_basis_x; i++) {
                    const int n = basis_x_start_ + i;

                    //d_eta += 1.0f;
                    d_eta += kl_scaling_ * rns[num_basis_y + j][i] *
                             powf(m, -kl_decay_) * powf(n, -kl_decay_) *
                             cospif(2*m*y_cos) * cospif(2*n*x_cos);
                }
            }
        }

        // Compute geostrophic balance using the perturbations in shmem

        // Write to shared memory
        d_eta_shmem[ty][tx] = d_eta;

        // Read Hi into shareed memory for given thread id
        for (int j = ty; j < block_height+1; j += blockDim.y) {
            const int global_j = clamp(by+j, 0, ny_+4);
            float* const Hi_row = (float*) ((char*) Hi_ptr_ + Hi_pitch_*(global_j));
            for (int i = tx; i < block_width+1; i += blockDim.x) {
                const int global_i = clamp(bx+i, 0, nx_+4);
                Hi_shmem[j][i] = Hi_row[global_i];
            }
        }

        __syncthreads();
        
        if ((tx > 0) && (tx < block_width - 1) && (ty > 0) && (ty < block_height - 1)) {


            // Check if cell is zero
            const bool dry_cell = (Hi_shmem[ty  ][tx  ] == land_value_) ||
                                  (Hi_shmem[ty  ][tx+1] == land_value_) ||
                                  (Hi_shmem[ty+1][tx  ] == land_value_) ||
                                  (Hi_shmem[ty+1][tx+1] == land_value_);
            
            // reconstruct H at cell center
            const float H_mid = 0.25f*(Hi_shmem[ty  ][tx] + Hi_shmem[ty  ][tx+1] +
                                       Hi_shmem[ty+1][tx] + Hi_shmem[ty+1][tx+1]   );

            // Get vector towards north.
            const float2 north = getNorth(ti, tj, nx_, ny_);
            
            // FIXME: Read from correct texture always
            float coriolis = f_ + beta_ * ((ti+0.5f)*dx_*north.x + (tj+0.5f)*dy_*north.y);
            if (f_ == 0) {
                coriolis = coriolisF(ti, tj, nx_, ny_);
            }
            
            // Slope of perturbation of eta
            const float eta_diff_x = (d_eta_shmem[ty  ][tx+1] - d_eta_shmem[ty  ][tx-1]) / (2.0f*dx_);
            const float eta_diff_y = (d_eta_shmem[ty+1][tx  ] - d_eta_shmem[ty-1][tx  ]) / (2.0f*dy_);

            // perturbation of hu and hv
            float d_hu = -(g_/coriolis)*(H_mid + d_eta_shmem[ty][tx])*eta_diff_y;
            float d_hv =  (g_/coriolis)*(H_mid + d_eta_shmem[ty][tx])*eta_diff_x;        
            // FIXME: Do we need to multiply with north.x and north.y here? Check CDKLM kernel.        
            

            if ((ti > 1) && (ti < nx_ + 2) && (tj > 1) && (tj < ny_ + 2) ) {
                float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*(tj));
                float* const  hu_row = (float*) ((char*)  hu_ptr_ +  hu_pitch_*(tj));
                float* const  hv_row = (float*) ((char*)  hv_ptr_ +  hv_pitch_*(tj));

                eta_row[ti] += d_eta;
                 hu_row[ti] += d_hu;
                 hv_row[ti] += d_hv;
                
            }
        }
    }

} // extern "C"