/*
This software is part of GPU Ocean

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

These CUDA kernels generates random numbers for creating model error 
based on given covariance relations.

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
#include "random_number_generators.cu"

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
  * Kernel that takes two random vectors (xi, nu) as input, and make nu perpendicular to xi 
  */
extern "C" {
__global__ void makePerpendicular(
        // Size of data
        const int nx_, const int ny_,
        
        //Data
        float* xi_ptr_, const int xi_pitch_,           // size [nx, ny]
        float* nu_ptr_, const int nu_pitch_,           // size [nx, ny]  
        float* reduction_buffer                        // [xi*xi, nu*nu, xi*nu]
    ) {
    
    //Index of cell within domain
    const int ti = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int tj = (blockDim.y * blockIdx.y) + threadIdx.y;

    // Each thread computes and writes two uniform numbers.

    if ((ti < nx_) && (tj < ny_)) {
        
        // Read the values from the reduction buffer
        const float xi_norm = reduction_buffer[0];
        const float nu_norm = reduction_buffer[1];
        const float dot_product = reduction_buffer[2];
        
        const float parallel_factor = dot_product/xi_norm;
        const float perpendicular_norm = nu_norm - parallel_factor*dot_product;
                                              
        //Compute pointer to current row in the xi and nu arrays
        float* xi_row = (float*) ((char*) xi_ptr_ + xi_pitch_*tj);
        float* nu_row = (float*) ((char*) nu_ptr_ + nu_pitch_*tj);
        
        
        
        nu_row[ti] = sqrt(nu_norm/perpendicular_norm)*(nu_row[ti] - (parallel_factor*xi_row[ti]));
    }
}
} // extern "C"




/**
  * Local function calculating the SOAR function given two grid locations
  */
__device__ float soar_covariance(int a_x, int a_y, int b_x, int b_y,
                      float dx, float dy, float soar_q0, float soar_L) {
    const float dist = sqrt( dx*dx*(a_x - b_x)*(a_x - b_x) +
                             dy*dy*(a_y - b_y)*(a_y - b_y) );
    return soar_q0*( 1.0f + dist/soar_L)*exp(-dist/soar_L);
}



/**
  * Kernel that generates a perturbation of the eta field on the coarse grid.
  * The perturbation is based on a SOAR covariance function using a cut-off value of 2.
  * The result is according to the use of periodic boundary conditions
  */
extern "C" {
__global__ void SOAR(
        // Size of data
        const int nx_, const int ny_,
        const float dx_, const float dy_,
                
        // Parameter for the SOAR function
        float soar_q0_, float soar_L_, 

        // Scale parameter for the final stochastic field
        float perturbation_scale_,
        
        // Periodic domain
        const int periodic_north_south_, const int periodic_east_west_,
        
        // random data (input) - if periodic BC: size [nx, ny]
        //                       else:           size [nx + 8, ny + 8]
        float* random_ptr_, const int random_pitch_,

        // Coarse grid data variable (output) - size [nx+4, ny+4]
        // Write to all cells
        float* coarse_ptr_, const int coarse_pitch_,
        const int additive_ // Interpreted as a bool
    ) {

    // Find this thread's indices
    // Note that we write to the both internal cells and ghost cells
    // Total number of threads for this kernel is (nx+4, ny+4).
    
    //Index of cell within block
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;

    //Index of start of block in the coarse buffer
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell in the coarse buffer
    const int ti = bx + tx;
    const int tj = by + ty;

    const int cutoff = 2;

    // Local storage for xi (the random numbers)
    // Two ghost cells required for the SOAR stencil
    __shared__ float xi[block_height+4][block_width+4];

    // Read random numbers into local memory, using periodic BCs if needed:
    // In order to globally write (nx+4, ny+4) values, we need to read (nx+8, ny+8) values.
    for (int j = ty; j < block_height+4; j += blockDim.y) {
        int global_j = 0;
        if (periodic_north_south_) {
            global_j = (by + j - cutoff - 2 + ny_) % ny_;
        } else {
            global_j = clamp(by + j, 0, ny_+8);
        }
        
        float* const random_row = (float*) ((char*) random_ptr_ + random_pitch_*global_j);
        
        for (int i = tx; i < block_width+4; i += blockDim.x) {
        
            int global_i = 0;
            
            if (periodic_east_west_) {
                global_i = (bx + i - cutoff - 2 + nx_) % nx_;
            } else {
                global_i = clamp(bx + i, 0, nx_+8);
            }
            
            xi[j][i] = random_row[global_i];
        }
    }

    __syncthreads();
    
    
    // Compute d_eta using the SOAR covariance function, and store in global memory.
    // All reads are from local memory, and each thread loops over all cells within the cutoff area.
    const int a_x = tx + cutoff;
    const int a_y = ty + cutoff;
    float Qxi = 0.0f;
    for (int b_y = ty; b_y < a_y + cutoff + 1; b_y++) {
        for (int b_x = tx; b_x < a_x + cutoff + 1; b_x++) {
            const float Q = soar_covariance(a_x, a_y, b_x, b_y,
                                            dx_, dy_, soar_q0_, soar_L_);
            Qxi += Q*xi[b_y][b_x];
        }
    }
    
    // Write eta to global memory
    if ((ti < nx_+4) && (tj < ny_+4)) {

        //Compute pointer to current row in the coarse array
        float* coarse_row = (float*) ((char*) coarse_ptr_ + coarse_pitch_*(tj));
        if (additive_ > 0) {
            coarse_row[ti] += perturbation_scale_ * Qxi;
        }
        else {
            coarse_row[ti] = perturbation_scale_ * Qxi;
        }
    }
}
} // extern "C"





/**
  * Kernel that adds a perturbation to the input fields eta, hu and hv.
  * The kernel assumes that the coarse grid is equal to the computational grid, and the 
  * values from the coarse grid can therefore be added to eta directly.
  * In order to avoid non-physical states and shock solutions, perturbations for hu and hv
  * are generated according to the geostrophic balance.
  */
extern "C" {
__global__ void geostrophicBalance(
        // Size of data
        const int nx_, const int ny_,
        const float dx_, const float dy_,
        const int ghost_cells_x_, const int ghost_cells_y_,

        // physical parameters
        const float g_, const float f_, const float beta_, const float y0_reference_cell_,
        
        // d_eta values (coarse grid) - size [nx + 4, ny + 4]
        float* coarse_ptr_, const int coarse_pitch_,
    
        // Ocean data variables - size [nx + 4, ny + 4]
        // Write to interior cells only,  [2:nx+2, 2:ny+2]
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_, const int hu_pitch_,
        float* hv_ptr_, const int hv_pitch_,

        // Ocean data parameter - size [nx + 5, ny + 5]
        float* Hi_ptr_, const int Hi_pitch_,
        const float land_value_
    ) {

    //Index of cell within block
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;

    //Index of start of block within domain
    const int bx = blockDim.x * blockIdx.x + ghost_cells_x_; // Compansating for ghost cells
    const int by = blockDim.y * blockIdx.y + ghost_cells_y_; // Compensating for ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;

    // Shared memory for d_eta (also used for H)
    __shared__ float d_eta[block_height+2][block_width+2];

    // Use shared memory for d_eta to compute H_mid for given thread id
    // Read H on cell intersections 
    for (int j = ty; j < block_height+1; j += blockDim.y) {
        const int global_j = clamp(by+j, 0, ny_+4);
        float* const Hi_row = (float*) ((char*) Hi_ptr_ + Hi_pitch_*(global_j));
        for (int i = tx; i < block_width+1; i += blockDim.x) {
            const int global_i = clamp(bx+i, 0, nx_+4);
            d_eta[j][i] = Hi_row[global_i];
        }
    }
    
    __syncthreads();
    
    // Check if cell is zero
    const bool dry_cell = (d_eta[ty  ][tx  ] == land_value_) ||
                          (d_eta[ty  ][tx+1] == land_value_) ||
                          (d_eta[ty+1][tx  ] == land_value_) ||
                          (d_eta[ty+1][tx+1] == land_value_);
                          
    // Reconstruct H in cell center
    const float H_mid = 0.25f*(d_eta[ty  ][tx] + d_eta[ty  ][tx+1] +
                               d_eta[ty+1][tx] + d_eta[ty+1][tx+1]   );
    
    __syncthreads();
    
    // Read d_eta from coarse_buffer into shared memory
    // One layer of ghost cells are required for central differences leading to hu and hv 
    for (int j = ty; j < block_height+2; j += blockDim.y) {
        
        const int global_j = clamp(by+j-1, 0, ny_+3);
        float* const coarse_row = (float*) ((char*) coarse_ptr_ + coarse_pitch_*(global_j));
        
        for (int i = tx; i < block_width+2; i += blockDim.x) {
            const int global_i = clamp(bx+i-1, 0, nx_+3);
            d_eta[j][i] = coarse_row[global_i];
        }
    }
    
    __syncthreads();

    // Evaluate geostrophic balance and write eta, hu and hv to global memory
    if ( (ti > 1) && (tj > 1) && (ti < nx_+2) && (tj < ny_+2)) {

        //Compute pointer to current row in the U array
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*(tj));
        float* const hu_row  = (float*) ((char*) hu_ptr_  + hu_pitch_*(tj));
        float* const hv_row = (float*) ((char*)  hv_ptr_ + hv_pitch_*(tj));
        
        // Indices within the eta shared memory
        const int eta_tx = tx+1;
        const int eta_ty = ty+1;

        // Get vector towards north.
        const float2 north = getNorth(ti, tj, nx_, ny_);
        
        // FIXME: Read from correct texture always
        float coriolis = f_ + beta_ * ((ti+0.5f)*dx_*north.x + (tj+0.5f)*dy_*north.y);
        if (f_ == 0) {
            coriolis = coriolisF(ti, tj, nx_, ny_);
        }
        
        
        // Total water depth in the given cell (H + eta + d_eta)
        const float h_mid = d_eta[eta_ty][eta_tx] + H_mid + eta_row[ti];

        // Slope of perturbation of eta
        const float eta_diff_x = (d_eta[eta_ty  ][eta_tx+1] - d_eta[eta_ty  ][eta_tx-1]) / (2.0f*dx_);
        const float eta_diff_y = (d_eta[eta_ty+1][eta_tx  ] - d_eta[eta_ty-1][eta_tx  ]) / (2.0f*dy_);

        // perturbation of hu and hv
        const float d_hu = -(g_/coriolis)*h_mid*eta_diff_y;
        const float d_hv =  (g_/coriolis)*h_mid*eta_diff_x;        

        if (true) {
            eta_row[ti] += d_eta[eta_ty][eta_tx];
             hu_row[ti] += d_hu;
             hv_row[ti] += d_hv;
        } else {
            eta_row[ti] = d_eta[eta_ty][eta_tx];
            hu_row[ti] = d_hu;
            hv_row[ti] = d_hv;
        }
        
        if (dry_cell) {
            eta_row[ti] = 0.0f;
            hu_row[ti]  = 0.0f;
            hv_row[ti]  = 0.0f;
            
        }
    }
}
} // extern "C"




//texture<float, cudaTextureType2D> angle_tex;

/**
  * Kernel that adds a perturbation to the input fields eta, hu and hv.
  * The kernel use a bicubic interpolation to transfer values from the coarse grid to the
  * computational grid. Since the derivatives of the eta field are known during the interpolation,
  * hu and hv are assigned their appropriate geostrophically balanced values directly.
  */
extern "C" {
__global__ void bicubicInterpolation(
        // Size of computational data
        const int nx_, const int ny_,
        const int ghost_cells_x_, const int ghost_cells_y_,
        const float dx_, const float dy_,
    
        // Size of coarse data
        const int coarse_nx_, const int coarse_ny_,
        const int coarse_ghost_cells_x_, const int coarse_ghost_cells_y_,
        const float coarse_dx_, const float coarse_dy_,
        const int offset_i_, const int offset_j_,
    
        // physical parameters
        const float g_, const float f_, const float beta_, const float y0_reference_cell_,
        
        // d_eta values (coarse grid) - size [nx + 4, ny + 4]
        float* coarse_ptr_, const int coarse_pitch_,
    
        // Ocean data variables - size [nx + 4, ny + 4]
        // Write to interior cells only,  [2:nx+2, 2:ny+2]
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_, const int hu_pitch_,
        float* hv_ptr_, const int hv_pitch_,

        // Ocean data parameter - size [nx + 5, ny + 5]
        float* Hi_ptr_, const int Hi_pitch_,
        const float land_value_
    ) {
    
    // Each thread is responsible for one grid point in the computational grid.
    // Due to central differences of d_eta needed for geostrophic balance, we 
    // interpolate with a halo of 1 cell in all directions for eta.
    
    //Index of cell within block
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;

    //Index of start of block within domain
    const int bx = blockDim.x * blockIdx.x + ghost_cells_x_; // Compansating for ghost cells
    const int by = blockDim.y * blockIdx.y + ghost_cells_y_; // Compensating for ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;

    // Shared memory for H and the coarse values
    __shared__ float coarse[block_height+6][block_width+6];
    __shared__ float d_eta[block_height+2][block_width+2];
    

    // Use shared memory for coarse to read H_i for given thread id
    for (int j = ty; j < block_height+1; j += blockDim.y) {
        const int global_j = clamp(by+j, 0, ny_+4);
        float* const Hi_row = (float*) ((char*) Hi_ptr_ + Hi_pitch_*(global_j));
        for (int i = tx; i < block_width+1; i += blockDim.x) {
            const int global_i = clamp(bx+i, 0, nx_+4);
            coarse[j][i] = Hi_row[global_i];
        }
    }
    
    __syncthreads();
    
    // Check if cell is zero
    const bool dry_cell = (coarse[ty  ][tx  ] == land_value_) ||
                          (coarse[ty  ][tx+1] == land_value_) ||
                          (coarse[ty+1][tx  ] == land_value_) ||
                          (coarse[ty+1][tx+1] == land_value_);
    
    // reconstruct H at cell center
    const float H_mid = 0.25f*(coarse[ty  ][tx] + coarse[ty  ][tx+1] +
                               coarse[ty+1][tx] + coarse[ty+1][tx+1]   );
    
    __syncthreads();
    
    
    // Find coarse index for thread (0,0). All threads need to know this in order to read
    // coarse data correctly into coarse shmem.
    const int bx_x = (bx - ghost_cells_x_ + 0.5 + offset_i_)*dx_;
    const int by_y = (by - ghost_cells_y_ + 0.5 + offset_j_)*dy_;

    // The start of the coarse buffer which needs to be read into shared memory.
    // The coarse buffer has two layers of ghost cells.
    const int coarse_bx = (int)(floorf((bx_x/coarse_dx_) + coarse_ghost_cells_x_ - 2.5f ));
    const int coarse_by = (int)(floorf((by_y/coarse_dy_) + coarse_ghost_cells_y_ - 2.5f ));

    // Read d_eta from coarse_buffer into shared memory.
    // For very small blocks which is particularly bad alligned with the coarse grid,
    // we need the following amount of data from the coarse grid:
    for (int j = ty; j < block_height+6; j += blockDim.y) {
        
        const int global_j = clamp(coarse_by+j, 0, coarse_ny_+3);
        float* const coarse_row = (float*) ((char*) coarse_ptr_ + coarse_pitch_*(global_j));
        
        for (int i = tx; i < block_width+6; i += blockDim.x) {
            
            const int global_i = clamp(coarse_bx+i, 0, coarse_nx_+3);
            coarse[j][i] = coarse_row[global_i];
        }
    }
    __syncthreads();    
        
    // Carry out bicubic interpolation and write d_eta to shmem.
    // We calulate d_eta for all threads within the block plus one layer of ghost cells, so that
    // geostrophic balance can be found for the cell in the block.
    for (int j = ty; j < block_height+2; j += blockDim.y) {
        for (int i = tx; i < block_width+2; i += blockDim.x) {
            
            // Thread index in the global fine domain
            const int loop_ti = bx + i - 1;
            const int loop_tj = by + j - 1;

            // Find coarse index for this thread
            const float x = (loop_ti - ghost_cells_x_ + 0.5f + offset_i_)*dx_;
            const float y = (loop_tj - ghost_cells_y_ + 0.5f + offset_j_)*dy_;
            
            // Location in the coarse grid:
            int coarse_i = (int)(floorf((x/coarse_dx_) + coarse_ghost_cells_x_ - 0.5f));
            int coarse_j = (int)(floorf((y/coarse_dy_) + coarse_ghost_cells_y_ - 0.5f));

            // When interpolating onto the first ghostcell to the right and top of the domain,
            // there exist a special case that results in reading outside of the coarse buffer.
            // These if-statements fix that issue, while still resulting in valid code.
            // This should give rel_x and/or rel_y equal to 1.0.
            if (coarse_i > coarse_nx_ + 1 ) {
                coarse_i -= 1;
            }
            if (coarse_j > coarse_ny_ + 1 ) {
                coarse_j -= 1;
            }

            // Location of the grid point on the coarse grid.
            const float coarse_x = (coarse_i - coarse_ghost_cells_x_ + 0.5f)*coarse_dx_;
            const float coarse_y = (coarse_j - coarse_ghost_cells_y_ + 0.5f)*coarse_dy_;

            // Location in coarse shmem:
            const int loc_i = coarse_i - coarse_bx; // coarse_bx accounts for ghost cell already.
            const int loc_j = coarse_j - coarse_by; 
            
            //min_val = min(min_val, min(loc_i-1, loc_j-1));
            //max_val = max(max_val, max(loc_i-1, loc_j-1));
           
            //---------------------------------
            // Read values for interpolation
            //---------------------------------
            Matrix4x4_d f_matrix;
            // [[ f00,  f01,  fy00,  fy01 ]
            //  [ f10,  f11,  fy10,  fy11 ]
            //  [fx00, fx01, fxy00, fxy01 ]
            //  [fx10, fx11, fxy10, fxy11 ]]
            {
                // We use the following notation:
                //     function values (f) or derivatives (fx, fy), or cross derivatives (fxy)
                //     Evaluated in the unit square at given positions
                //     e.g., f01   = function value at (0, 1)
                //           fy_10 = value of derivative wrt y at (-1,0)
                //           fy21  = value of derivative wrt y at ( 2,1)
                
                // Corner point values
                f_matrix.m_row0.x = coarse[loc_j  ][loc_i  ]; // f00
                f_matrix.m_row0.y = coarse[loc_j+1][loc_i  ]; // f01
                f_matrix.m_row1.x = coarse[loc_j  ][loc_i+1]; // f10
                f_matrix.m_row1.y = coarse[loc_j+1][loc_i+1]; // f11

                // Derivatives in x-direction in the corner points
                f_matrix.m_row2.x = (coarse[loc_j  ][loc_i+1] - coarse[loc_j  ][loc_i-1])/2.0f; // fx00
                f_matrix.m_row2.y = (coarse[loc_j+1][loc_i+1] - coarse[loc_j+1][loc_i-1])/2.0f; // fx01
                f_matrix.m_row3.x = (coarse[loc_j  ][loc_i+2] - coarse[loc_j  ][loc_i  ])/2.0f; // fx10
                f_matrix.m_row3.y = (coarse[loc_j+1][loc_i+2] - coarse[loc_j+1][loc_i  ])/2.0f; // fx11

                // Derivatives in y-direction in the corner points
                f_matrix.m_row0.z = (coarse[loc_j+1][loc_i  ] - coarse[loc_j-1][loc_i  ])/2.0f; // fy00
                f_matrix.m_row0.w = (coarse[loc_j+2][loc_i  ] - coarse[loc_j  ][loc_i  ])/2.0f; // fy01
                f_matrix.m_row1.z = (coarse[loc_j+1][loc_i+1] - coarse[loc_j-1][loc_i+1])/2.0f; // fy10
                f_matrix.m_row1.w = (coarse[loc_j+2][loc_i+1] - coarse[loc_j  ][loc_i+1])/2.0f; // fy11

                // Derivatives in y-direction required for obtaining the cross derivatives
                const float fy_10 = (coarse[loc_j+1][loc_i-1] - coarse[loc_j-1][loc_i-1])/2.0f; // fy_10
                const float fy_11 = (coarse[loc_j+2][loc_i-1] - coarse[loc_j  ][loc_i-1])/2.0f; // fy_11
                const float fy20  = (coarse[loc_j+1][loc_i+2] - coarse[loc_j-1][loc_i+2])/2.0f; // fy20 
                const float fy21  = (coarse[loc_j+2][loc_i+2] - coarse[loc_j  ][loc_i+2])/2.0f; // fy21 

                // Cross-derivatives in the corner points = x-derivatives of wide y-derivatives
                f_matrix.m_row2.z = (f_matrix.m_row1.z - fy_10)/2.0f; // fxy00
                f_matrix.m_row2.w = (f_matrix.m_row1.w - fy_11)/2.0f; // fxy01
                f_matrix.m_row3.z = (fy20 -  f_matrix.m_row0.z)/2.0f; // fxy10
                f_matrix.m_row3.w = (fy21 -  f_matrix.m_row0.w)/2.0f; // fxy11
            }

            // Map cell center position (x,y) onto the unit square
            const float rel_x = (x - coarse_x)/coarse_dx_;
            const float rel_y = (y - coarse_y)/coarse_dy_;

            //Bilinear interpolation (kept for future reference)
            //const float bi_linear_eta = f00*(1.0f-rel_x)*(1.0f-rel_y) + f10*rel_x*(1.0f-rel_y) + f01*(1.0f-rel_x)*rel_y + f11*rel_x*rel_y;

            // Obtain the coefficients for the bicubic surface
            Matrix4x4_d a_matrix = bicubic_interpolation_coefficients(f_matrix);

            const float4 x_vec = make_float4(1.0f, rel_x, rel_x*rel_x, rel_x*rel_x*rel_x);
            const float4 y_vec = make_float4(1.0f, rel_y, rel_y*rel_y, rel_y*rel_y*rel_y);
            
            d_eta[j][i] = bicubic_evaluation(x_vec, y_vec, a_matrix);
        }
    }
    
    __syncthreads();
    
    // Obtain geostrophic balance and add results to global memory
    if ( (ti > 1) && (tj > 1) && (ti < nx_+2) && (tj < ny_+2)) {

        // Indices within the d_eta shared memory
        const int eta_tx = tx + 1;
        const int eta_ty = ty + 1;
        
        // Get vector towards north.
        const float2 north = getNorth(ti, tj, nx_, ny_);
        
        // FIXME: Read from correct texture always
        float coriolis = f_ + beta_ * ((ti+0.5f)*dx_*north.x + (tj+0.5f)*dy_*north.y);
        if (f_ == 0) {
            coriolis = coriolisF(ti, tj, nx_, ny_);
        }
        
        // Slope of perturbation of eta
        const float eta_diff_x = (d_eta[eta_ty  ][eta_tx+1] - d_eta[eta_ty  ][eta_tx-1]) / (2.0f*dx_);
        const float eta_diff_y = (d_eta[eta_ty+1][eta_tx  ] - d_eta[eta_ty-1][eta_tx  ]) / (2.0f*dy_);

        // perturbation of hu and hv
        float d_hu = -(g_/coriolis)*(H_mid + d_eta[eta_ty][eta_tx])*eta_diff_y;
        float d_hv =  (g_/coriolis)*(H_mid + d_eta[eta_ty][eta_tx])*eta_diff_x;        
        // FIXME: Do we need to multiply with north.x and north.y here? Check CDKLM kernel.        
        
        //Compute pointer to current row in the U array
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*(tj));
        float* const hu_row  = (float*) ((char*) hu_ptr_  + hu_pitch_*(tj));
        float* const hv_row = (float*) ((char*)  hv_ptr_ + hv_pitch_*(tj));
        
        // Set the perturbation to zero if the cell is on land
        if (dry_cell) {
            d_eta[eta_ty][eta_tx] = 0.0f;
            d_hu = 0.0f;
            d_hv = 0.0f;
        }
        
        eta_row[ti] += d_eta[eta_ty][eta_tx];
         hu_row[ti] += d_hu; 
         hv_row[ti] += d_hv;
              
    }
}
} // extern "C"


// ------------------------------------------------------------------------
//   Perturbations using Karhunen-Loeve basis functions
// ------------------------------------------------------------------------




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
        __shared__ float rns[kl_rand_ny][kl_rand_nx];

        // Load random numbers into shmem
        for (int j = ty; j < kl_rand_ny; j += blockDim.y) {
            float* const random_row_ = (float*) ((char*) random_ptr_ + random_pitch_*j);
            for (int i = tx; i < kl_rand_nx; i += blockDim.x) {
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
        __shared__ float rns[kl_rand_ny][kl_rand_nx];

        // Shared memory for eta perturbation
        __shared__ float d_eta_shmem[block_height][block_width];

        // Shared memory for Hi
        __shared__ float Hi_shmem[block_height+1][block_width+1];
        
        // Load random numbers into shmem
        for (int j = ty; j < kl_rand_ny; j += blockDim.y) {
            float* const random_row_ = (float*) ((char*) random_ptr_ + random_pitch_*j);
            for (int i = tx; i < kl_rand_nx; i += blockDim.x) {
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