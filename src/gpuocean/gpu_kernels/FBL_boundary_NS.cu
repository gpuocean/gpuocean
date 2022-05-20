/*
This software is part of GPU Ocean. 

Copyright (C) 2018, 2019 SINTEF Digital
Copyright (C) 2018, 2019 Norwegian Meteorological Institute

This CUDA kernel implements part of the Forward Backward Linear 
numerical scheme for the shallow water equations, described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "common.cu"


extern "C" {
__global__ void closedBoundaryUKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

        // Data
        float* U_ptr_, int U_pitch_) {
            
    // Global thread sizes:
    // ti = [0, nx_+1]
    // tj = [0, 3],
    // thread 0 is index 0, thread 1 is index ny_+1, 
    // thread 2 and 3 idle
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int tj = (thread_id == 1) ? ny_ + 1 : thread_id;

    //Compute pointer to current row in the U array
    float* const U_row = (float*) ((char*) U_ptr_ + U_pitch_*tj);   
    
    if (ti < nx_+1) {
        if ((tj == 0    ) && (bc_south_ == 1)) {
            float* const U_row_inner = (float*) ((char*) U_ptr_ + U_pitch_*1);
            U_row[ti] = U_row_inner[ti];
        }
        if ((tj == ny_+1) && (bc_north_ == 1)) {
            float* const U_row_inner = (float*) ((char*) U_ptr_ + U_pitch_*ny_);
            U_row[ti] = U_row_inner[ti];
        }
    }
}
} // extern "C" 



 // Fix north-south boundary before east-west (to get the corners right)
 extern "C" {
__global__ void periodicBoundaryUKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
    
        // Data
        float* U_ptr_, int U_pitch_) {

    // Global thread sizes:
    // ti = [0, nx_+1]
    // tj = [0, 3],
    // thread 0 is index 0, thread 1 is index ny_+1, 
    // thread 2 and 3 idle
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int tj = (thread_id == 1) ? ny_ + 1 : thread_id;

    int opposite_row_index = ny_;
    if (tj == ny_+1) {
        opposite_row_index = 1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if (((tj == 0) || (tj == ny_ + 1)) && (ti > 0) && (ti < nx_))  {
        float* ghost_row = (float*) ((char*) U_ptr_ + U_pitch_*tj);
        float* opposite_row = (float*) ((char*) U_ptr_ + U_pitch_*opposite_row_index);
        ghost_row[ti] = opposite_row[ti];
    }
}
} // extern "C"




// Fix north-south boundary before east-west (to get the corners right)
extern "C" {
__global__ void closedBoundaryVKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

        // Data
        float* V_ptr_, int V_pitch_) {
            
    // Global thread sizes:
    // ti = [0, nx_+1]
    // tj = [0, 3],
    // thread 0 is index 0,    thread 1 is 1, 
    // thread 2 is index ny+1, thread 3 is ny+2, 
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id = blockIdx.y * blockDim.y + threadIdx.y;

    const int tj = (thread_id > 1) ? ny_ + thread_id - 1 : thread_id;

    //Compute pointer to current row in the V array
    float* const V_row = (float*) ((char*) V_ptr_ + V_pitch_*tj);   
    
    if (ti < nx_ + 2) {
        
        if ((tj == 1 ) && (bc_south_ == 1)) {
            V_row[ti] == 0.0f;
        }
        
       if ((tj == ny_+1) && (bc_north_ == 1)) {
           V_row[ti] = 0.0;
       }
        
        if ((tj == 0) && (bc_south_ == 1)) {
            float* const V_row_inner = (float*) ((char*) V_ptr_ + V_pitch_*2);   
            V_row[ti] = -V_row_inner[ti];
        }
        
        if ((tj == ny_+2) && (bc_north_ == 1)) {
            float* const V_row_inner = (float*) ((char*) V_ptr_ + V_pitch_*ny_);   
            V_row[ti] = -V_row_inner[ti];
        }
    }
}
} // extern "C"

 // Fix north-south boundary before east-west (to get the corners right)
 extern "C" {
__global__ void periodicBoundaryVKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
    
        // Data
        float* V_ptr_, int V_pitch_) {
    // One row of ghost values must be updated with the opposite 
    // interior cells' values.
    // The northern boundary must be given the value from the southern boundary

    // Global thread sizes:
    // ti = [0, nx_+1]
    // tj = [0, 3],
    // thread 0 is index 0,    thread 1 is 1, 
    // thread 2 is index ny+1, thread 3 is ny+2, 
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id = blockIdx.y * blockDim.y + threadIdx.y;

    const int tj = (thread_id > 1) ? ny_ + thread_id - 1 : thread_id;

    int opposite_row_index = ny_;
    if (tj == ny_+ 1) {
        opposite_row_index = 1;
    }
    if (tj == ny_+2) {
        opposite_row_index = 2;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ( ((tj == 0) || (tj == ny_ + 1) || (tj == ny_ + 2))
          && (ti > 0) && (ti < nx_+1) )  {
        float* ghost_row = (float*) ((char*) V_ptr_ + V_pitch_*tj);
        float* opposite_row = (float*) ((char*) V_ptr_ + V_pitch_*opposite_row_index);
        ghost_row[ti] = opposite_row[ti]; 
    }
}
} // extern "C"


// Fix north-south boundary before east-west (to get the corners right)
extern "C" {
__global__ void closedBoundaryEtaKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

        // Data
        float* eta_ptr_, int eta_pitch_) {
            
    // All eta values living outside of a closed boundary should
    // be ignored by the step-kernel. Anyway, we but them to zero to
    // make sure they are well defined, but this kernel should not need to
    // be called between time-steps.
    
    // Global thread sizes:
    // ti = [0, nx_+1]
    // tj = [0, 3],
    // thread 0 is index 0, thread 1 is index ny_+1, 
    // thread 2 and 3 idle
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id = blockIdx.y * blockDim.y + threadIdx.y;

    const int tj = (thread_id == 1) ? ny_+1 : thread_id;

    //Compute pointer to current row in the eta array
    float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj); 
    
    if (ti < nx_+2) {

        if ( (tj == 0  ) && (bc_south_ == 1)) {
            float* const eta_row_inner = (float*) ((char*) eta_ptr_ + eta_pitch_*1);
            eta_row[ti] = eta_row_inner[ti];
        }

        if ((tj == ny_+1)  && (bc_north_ == 1)) {
            float* const eta_row_inner = (float*) ((char*) eta_ptr_ + eta_pitch_*ny_);
            eta_row[ti] = eta_row_inner[ti];
        }
    }

}
} // extern "C"



// Fix north-south boundary before east-west (to get the corners right)
extern "C" {
__global__ void periodicBoundaryEtaKernel_NS(
        // Discretization parameters
        int nx_, int ny_,

        // Data
        float* eta_ptr_, int eta_pitch_) {
    
    // Global thread sizes:
    // ti = [0, nx_+1]
    // tj = [0, 3],
    // thread 0 is index 0, thread 1 is index ny_+1, 
    // thread 2 and 3 idle
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int tj = (thread_id == 1) ? ny_+1 : thread_id;
    
    int opposite_row_index = ny_;
    if (tj == ny_+1) {
        opposite_row_index = 1;
    }
    
    // Set northern ghost cells
    if ( ((tj == 0) || (tj == ny_+1)) &&
          (ti > 0) && (ti < nx_+1) ) {
        float* ghost_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj);
        float* opposite_row = (float*) ((char*) eta_ptr_ + eta_pitch_*opposite_row_index);
        ghost_row[ti] = opposite_row[ti]; 
    }
}
} // extern "C"
