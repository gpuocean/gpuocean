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

// Fix  east-west boundary before north-south (to get the corners right)
// This is contrary to all the other BC kernels.
extern "C" {
__global__ void closedBoundaryUKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

        // Data
        float* U_ptr_, int U_pitch_) {
            
    // Global tread sizes:
    // ti = {0, 1}
    // thread 0 is index 0
    // thread 1 is index nx
    // tj = [0, ny_+1]
    
    // Index of cell within domain
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int ti = (thread_id == 0) ? 0 : nx_;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    //Compute pointer to current row in the U array
    float* const U_row = (float*) ((char*) U_ptr_ + U_pitch_*tj);   
    
    if ( ( ((ti == 0  ) && (bc_west_ == 1)) || 
           ((ti == nx_) && (bc_east_ == 1))    ) && tj < ny_ + 2) {
        U_row[ti] = 0.0f;
    }
}
} // extern "C" 



extern "C" {
__global__ void periodicBoundaryUKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
    
        // Data
        float* U_ptr_, int U_pitch_) {
    // U has no ghost cells in x-directions, but the values 
    // *on* the boundary need to match. 
    // The compute_U_kernel fixes the western (ti = 0) boundary,
    // and the eastern boundary (ti = nx) needs to be sat the same.

    // Global tread sizes:
    // ti = {0, 1}
    // thread 0 is index 0
    // thread 1 is index nx
    // tj = [0, ny_+1]
    
    // Index of cell within domain
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int ti = (thread_id == 0) ? 0 : nx_;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is in the domain:
    if ((ti == nx_) && (tj <  ny_+2)) {
        float* U_row = (float*) ((char*) U_ptr_ + U_pitch_*tj);
        U_row[ti] = U_row[0];
    }
}
} // extern "C"




extern "C" {
__global__ void closedBoundaryVKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

        // Data
        float* V_ptr_, int V_pitch_) {
      
    // Global tread sizes:
    // ti = {0, 1}
    // thread 0 is index 0
    // thread 1 is index nx+1
    // tj = [0, ny_+1]
      
    // Index of cell within domain
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int ti = (thread_id == 1) ? nx_ + 1 : thread_id ;

    //Compute pointer to current row in the V array
    float* const V_row = (float*) ((char*) V_ptr_ + V_pitch_*tj);   
    
    if ( tj < ny_+3 ) {
        if ((ti == 0    ) && (bc_west_ == 1)) {
            V_row[ti] = V_row[1];
        }
        if ((ti == nx_+1) && (bc_east_ == 1)) {
            V_row[ti] = V_row[nx_];
        }
    }
}
} // extern "C"'


extern "C" {
__global__ void periodicBoundaryVKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
    
        // Data
        float* V_ptr_, int V_pitch_) {
    // Straight forward with one ghost column on each side
    
    // Global tread sizes:
    // ti = {0, 1}
    // thread 0 is index 0
    // thread 1 is index nx+1
    // tj = [0, ny_+1]
      
    // Index of cell within domain
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int ti = (thread_id == 1) ? nx_ + 1 : thread_id ;

    int opposite_col_index = nx_;
    if (ti == nx_+1) {
        opposite_col_index = 1;
    }
    
    // Check if thread is in the domain:
    if ( ((ti == 0) || (ti == nx_+1)) && (tj <  ny_+3) ) {
        float* V_row = (float*) ((char*) V_ptr_ + V_pitch_*tj);
        V_row[ti] = V_row[opposite_col_index];
    }
}
} // extern "C"



extern "C" {
__global__ void closedBoundaryEtaKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

        // Data
        float* eta_ptr_, int eta_pitch_) {
            
    // All eta values living outside of a closed boundary should
    // be ignored by the step-kernel. Anyway, we but them to zero to
    // make sure they are well defined, but this kernel should not need to
    // be called between time-steps.
    
    // Global tread sizes:
    // ti = {0, 1}
    // thread 0 is index 0
    // thread 1 is index nx+1
    // tj = [0, ny_+1]
    
    // Index of cell within domain
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;
    const int ti = (thread_id == 1) ? nx_+1 : thread_id; 

    //Compute pointer to current row in the eta array
    float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj); 
    
    if (tj < ny_+2) {
        
        if ((ti == 0    ) && (bc_west_ == 1)) {
            eta_row[0] = eta_row[1];
        }
        
        if ((ti == nx_+1) && (bc_east_ == 1)) {
            eta_row[nx_+1] = eta_row[nx_];
        }
        
    }
}
} // extern "C"




extern "C" {
__global__ void periodicBoundaryEtaKernel_EW(
        // Discretization parameters
        int nx_, int ny_,

        // Data
        float* eta_ptr_, int eta_pitch_) {
    
    // Global tread sizes:
    // ti = {0, 1}
    // thread 0 is index 0
    // thread 1 is index nx+1
    // tj = [0, ny_+1]
    
    // Index of cell within domain
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int ti = (thread_id == 1) ? nx_+1 : thread_id;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;
    
    float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj);

    int opposite_col_index = nx_;
    if (ti == nx_+1) {
        opposite_col_index = 1;
    }
    
    // Set northern ghost cells
    if ( ((ti == 0) || (ti == nx_+1)) &&
          (tj < ny_+2) ) {
        eta_row[ti] = eta_row[opposite_col_index];
    }
}
} // extern "C"
