/*
This software is part of GPU Ocean.

Copyright (C) 2018  SINTEF ICT

These CUDA kernels implements specific parts of the Implicit Equal-Weights 
Particle Filter (IEWPF) algorithm.
The following papers describe the method, though with mistakes and variations.
     - 'Implicit equal-weights particle filter' by Zhu, van Leeuwen and Amezcua, Quarterly
            Journal of the Royal Meteorological Society, 2016
     - 'State-of-the-art stochastic data assimilation methods for high-dimensional
            non-Gaussian problems' by Vetra-Carvalho et al, Tellus, 2018
     - 'A revied Implicit Equal-Weights Particle Filter' by Skauvold et al, ???, 2018

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
  * Kernel that sets the given buffer to zero
  */
extern "C" {
    __global__ void setBufferToZero(const int nx_, const int ny_,
                                    float* data_ptr_, const int data_pitch_)
    {
        const int ti = blockDim.x*blockIdx.x + threadIdx.x;
        const int tj = blockDim.y*blockIdx.y + threadIdx.y;

        if ((ti < nx_) && (tj < ny_)) {
            float* data_row = (float*) ((char*) data_ptr_ + data_pitch_*tj);
            data_row[ti] = 0.0f;
        }            
    }
} // extern "C"


/**
  * Kernel that implements x = a*x + b*y on a 2D buffer.
  * Buffers x and y is required to be of the same size.
  */
extern "C" {
    __global__ void blas_xaxpby(const int nx_, const int ny_,
                                float* x_ptr_, const int x_pitch_,
                                float* y_ptr_, const int y_pitch_,
                                const float a, const float b)
    {
        const int ti = blockDim.x*blockIdx.x + threadIdx.x;
        const int tj = blockDim.y*blockIdx.y + threadIdx.y;

        if ((ti < nx_) && (tj < ny_)) {
            float* x_row = (float*) ((char*) x_ptr_ + x_pitch_*tj);
            float* y_row = (float*) ((char*) y_ptr_ + y_pitch_*tj);
            x_row[ti] = a*x_row[ti] + b*y_row[ti];
        }            
    }
} // extern "C"





/**
  * Local function calculating the SOAR function given two grid locations
  */
__device__ float soar_covariance(const int a_x, const int a_y,
                                 const int b_x, const int b_y,
                                 const float dx, const float dy,
                                 const int nx, const int ny,
                                 const float soar_q0, const float soar_L) {
    
    const float dist_x = min(min((a_x - (b_x - nx))*(a_x - (b_x - nx)),
                                 (a_x - (b_x + nx))*(a_x - (b_x + nx))),
                             (a_x - b_x)*(a_x - b_x));
    const float dist_y = min(min((a_y - (b_y - ny))*(a_y - (b_y - ny)),
                                 (a_y - (b_y + ny))*(a_y - (b_y + ny))),
                             (a_y - b_y)*(a_y - b_y));
                                
    const float dist = sqrt( dx*dx*dist_x + dy*dy*dist_y );
    return soar_q0*( 1.0f + dist/soar_L)*exp(-dist/soar_L);
}


/**
  * Kernel that obtains the first half of the Kalman gain based on the
  * innovation from a single drifter.
  *
  * nx, ny: domain size
  * dx, dx: cell size
  * soar_q0, soar_L: amplitude and correlation radius, respectively, of the model error covariance
  * drifter_cell_x, drifter_cell_y: Cell in which the drifter is observed
  * geo_balance_const: constant appearing in the geostrophic balance, = g*H/(2*f)
  * e_x, e_y: Scaled innovation for given drifter, = S*d
  * K_ptr, K_pitch: Output buffer to write the Kalman gain contribution, size [ny, nx].
  */
extern "C" {
    __global__ void halfTheKalmanGain(const int nx_, const int ny_,
                                      const float dx_, const float dy_,
                                      const float soar_q0_, const float soar_L_,

                                      const int drifter_cell_x_, const int drifter_cell_y_,

                                      const float geoBalanceConst_, // g*H/(2*f)
                                      const float e_x_, const float e_y_, // S*innovation

                                      float* K_ptr_, const int K_pitch_)
    {
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        
        //const int drifter_id = blockIdx.x; // Currently assumed to be 1
        
        __shared__ float shared_huhv[4];
        
        float K_tmp = 0.0f;

        
        // U_GB^T: map out to laplacian stencil
        if ((tx == 0) && (ty == 0)) {
            // the x-component of the innovation spreads to north and south
            shared_huhv[0] = -e_x_*geoBalanceConst_/dy_; // north 
            shared_huhv[2] =  e_x_*geoBalanceConst_/dy_; // south
            // the y-component of the innovation spreads to east and west
            shared_huhv[1] =  e_y_*geoBalanceConst_/dx_; // east
            shared_huhv[3] = -e_y_*geoBalanceConst_/dx_; // west
        }
        __syncthreads();
        
        // Q^{1/2}: map each shared huhv out to SOAR stencil:
        if ((tx < 7) && (ty < 7)) {
            
            // north laplacian:
            if ((tx > 0) && (tx < 6) && (ty > 1) && (ty < 7)) {
                K_tmp += shared_huhv[0]*soar_covariance(tx, ty, 3 , 4,
                                                        dx_, dy_, nx_, ny_, soar_q0_, soar_L_);
            }
            // south laplacian:
            if ((tx > 0) && (tx < 6) && (ty > -1) && (ty < 5)) {
                K_tmp += shared_huhv[2]*soar_covariance(tx, ty, 3 , 2,
                                                        dx_, dy_, nx_, ny_, soar_q0_, soar_L_);
            }
            // east laplacian:
            if ((tx > 1) && (tx < 7) && (ty > 0) && (ty < 6)) {
                K_tmp += shared_huhv[1]*soar_covariance(tx, ty, 4 , 3,
                                                        dx_, dy_, nx_, ny_, soar_q0_, soar_L_);
            }
            // west laplacian:
            if ((tx > -1) && (tx < 5) && (ty > 0) && (ty < 6)) {
                K_tmp += shared_huhv[3]*soar_covariance(tx, ty, 2 , 3,
                                                        dx_, dy_, nx_, ny_, soar_q0_, soar_L_);
            }
            
            // Index in the K buffer (7x7 area centered  in drifter_cell_x and y)
            const int global_id_x = (drifter_cell_x_ - 3 + tx + nx_) % nx_;
            const int global_id_y = (drifter_cell_y_ - 3 + ty + ny_) % ny_;

            float* K_row = (float*) ((char*) K_ptr_ + K_pitch_*(global_id_y));
            K_row[global_id_x] += K_tmp;
            
        }
    }
} // extern "C"



/**
  * Local function ensuring that periodic boundary conditions are applied to indices
  */
__device__ float apply_periodic_boundary(const int index, const int dim_size)
{
    if (index < 0) {
        return index + dim_size;
    }
    else if (index >= dim_size) {
        return index - dim_size;
    }
    return index; 
}


/**
  * Kernel that multiplies the globally defined xi with the local 49 by 49 SVD block.
  */
extern "C" {
    __global__ void localSVDOnGlobalXi(const int nx_, const int ny_,
                                       const int pos_x_, const int pos_y_,

                                       const float* svd_ptr_, const int svd_pitch_,
                                       float* xi_ptr_, const int xi_pitch_)
    {
        __shared__ float shared_xi[7][7];
        __shared__ float shared_svd[49][49];
                                        
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        const int global_x_j = apply_periodic_boundary(pos_x_ - 3 + tx, nx_);
        const int global_y_j = apply_periodic_boundary(pos_y_ - 3 + ty, ny_);

        // Read the relevant xi field into shared memory
        float* xi_row =  (float*) ((char*) xi_ptr_ + xi_pitch_*(global_y_j));
        if ((tx < 7) && (ty < 7)) {
            shared_xi[ty][tx] = xi_row[global_x_j];
        }
        
        // Read the SVD block into shared memory
        for (int j=ty; j<49; j+=blockDim.y) {
            float* const svd_row = (float*) ((char*) svd_ptr_ + svd_pitch_*j);
            for (int i=tx; i<49; i+=blockDim.x) {
                shared_svd[j][i] = svd_row[i];
            }
        }
        
        __syncthreads();

        if ((tx < 7) && (ty < 7)) {
            float xi = 0.0f;
            const int local_j = ty*7 + tx; // thread row index of the SVD block
            int local_i;
            for (int loc_y_i = 0; loc_y_i < 7; loc_y_i++) {
                for (int loc_x_i = 0; loc_x_i < 7; loc_x_i++) {
                    local_i = loc_y_i*7 + loc_x_i;
                    xi += shared_svd[local_j][local_i] * shared_xi[loc_y_i][loc_x_i];
                }
            }
            
            // Write to global memory:
            xi_row[global_x_j] = xi;
        }
    }
            
} // extern "C"


