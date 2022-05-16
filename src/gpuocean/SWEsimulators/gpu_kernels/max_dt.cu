/*
This software is part of GPU Ocean.

Copyright (C) 2019  SINTEF Digital

Implements kernels for obtaining maximum step size for finite-volume
schemes implemented in GPU Ocean.

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


// Requires that the following names are defined when the kernel is built:
// NUM_THREADS 
// block_height
// block_width


#define FLT_MAX 100000.0f

//WARNING: Must match CDKLM16_kernel.cu and initBm_kernel.cu
//WARNING: This is error prone - as comparison with floating point numbers is not accurate
#define CDKLM_DRY_FLAG 1.0e-30f
#define CDKLM_DRY_EPS 1.0e-3f


extern "C" {
/*
 * Find the maximum dt allowed within each block based on the current ocean state.
 */
__global__ void per_block_max_dt(
        const int nx_, const int ny_,
        const float dx_, const float dy_,
        const float g_,
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_,  const int hu_pitch_,
        float* hv_ptr_,  const int hv_pitch_,
        float* Hm_ptr_,  const int Hm_pitch_,
        float land_value_,
        float* dt_ptr_,  const int dt_pitch_)
{
    //const float land_value_ = 1.0e20;
    
    // Block ID for output
    const int block_id_x = blockIdx.x;
    const int block_id_y = blockIdx.y;
    
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 2;
    
    __shared__ float shared_dt[block_height][block_width];
    volatile float* shared_dt_volatile = shared_dt[0];



    if ((ti < nx_+2) && (tj < ny_+2)) {
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj);
        float* const hu_row  = (float*) ((char*) hu_ptr_  +  hu_pitch_*tj);
        float* const hv_row  = (float*) ((char*) hv_ptr_  +  hv_pitch_*tj);
        float* const Hm_row  = (float*) ((char*) Hm_ptr_  +  Hm_pitch_*tj);
        
        float const eta = eta_row[ti];
        const float H = Hm_row[ti];
        float const h   = eta + H;
        
        //Ignore cells which are dry or masked as land
        if (h <= 0.0 || fabsf(H - land_value_) < CDKLM_DRY_EPS) {
            shared_dt[ty][tx] = FLT_MAX;
        }
        else {
            float const u   = hu_row[ti]/h;
            float const v   = hv_row[ti]/h;
            
            float const gravity_wave = sqrt(g_*h);
            
            float dt_thread =          0.25f*dx_/abs(u + gravity_wave);
            dt_thread = min(dt_thread, 0.25f*dx_/abs(u - gravity_wave));
            dt_thread = min(dt_thread, 0.25f*dy_/abs(v + gravity_wave));
            dt_thread = min(dt_thread, 0.25f*dy_/abs(v - gravity_wave));
            
            shared_dt[ty][tx] = dt_thread;
        }
    }
    else {
        shared_dt[ty][tx] = FLT_MAX;
    }
    
    __syncthreads();
    
    
    // Now, apply min across the shared memory according to a reduction tree pattern
    const int elements_in_shared = block_height*block_width;
    const int tid = tx + blockDim.x*ty;
    
    // Now, apply minimization all elements into a single element
    
    if (elements_in_shared >= 512) {
        // Assume that we never use more than 32x32=1024 threads per block.
        if (tid < 256) shared_dt[0][tid] = min(shared_dt[0][tid], shared_dt[0][tid + 256]);
        __syncthreads();
    }
    if (elements_in_shared >= 256) {
        if (tid < 128) shared_dt[0][tid] = min(shared_dt[0][tid], shared_dt[0][tid + 128]);
        __syncthreads();
    }
    if (elements_in_shared >= 128) {
        if (tid < 64) shared_dt[0][tid] = min(shared_dt[0][tid], shared_dt[0][tid + 64]);
        __syncthreads();
    }
    if (tid < 32) {
        if (elements_in_shared >= 64) shared_dt_volatile[tid] = min(shared_dt_volatile[tid], shared_dt_volatile[tid + 32]);
        if (tid < 16) {
            if (elements_in_shared >= 32) shared_dt_volatile[tid] = min(shared_dt_volatile[tid], shared_dt_volatile[tid + 16]);
            if (elements_in_shared >= 16) shared_dt_volatile[tid] = min(shared_dt_volatile[tid], shared_dt_volatile[tid +  8]);
            if (elements_in_shared >=  8) shared_dt_volatile[tid] = min(shared_dt_volatile[tid], shared_dt_volatile[tid +  4]);
            if (elements_in_shared >=  4) shared_dt_volatile[tid] = min(shared_dt_volatile[tid], shared_dt_volatile[tid +  2]);
            if (elements_in_shared >=  2) shared_dt_volatile[tid] = min(shared_dt_volatile[tid], shared_dt_volatile[tid +  1]);
        }
        
        if (tid == 0) {
            float* const dt_row = (float*) ((char*) dt_ptr_ + dt_pitch_*block_id_y);
            dt_row[block_id_x] = shared_dt_volatile[tid];
        }
    }
}
} // extern "C"
   


extern "C" {
__global__ void max_dt_reduction(
        //Discretization parameters
        const int num_elements,         
        float* dt_buffer,               // per block max dt, with num_elements elements
        float* max_dt_buffer)           // a buffer of size 1 to put the result in.
{
    
    __shared__ float sdata[NUM_THREADS];
    volatile float* sdata_volatile = sdata;
    
    unsigned int tid = threadIdx.x;

    // Square each elements and reduce to "NUM_THREADS" elements
    float thread_dt = FLT_MAX;
    for (unsigned int i = tid; i < num_elements; i += NUM_THREADS) {
        thread_dt = min(thread_dt, dt_buffer[i]);
    }
    sdata[tid] = thread_dt;
    __syncthreads();

    //Now, sum all elements into a single element
    if (NUM_THREADS >= 512) {
        if (tid < 256) sdata[tid] = min(sdata[tid], sdata[tid + 256]);
        __syncthreads();
    }
    if (NUM_THREADS >= 256) {
        if (tid < 128) sdata[tid] = min(sdata[tid], sdata[tid + 128]);
        __syncthreads();
    }
    if (NUM_THREADS >= 128) {
        if (tid < 64) sdata[tid] = min(sdata[tid], sdata[tid + 64]);
        __syncthreads();
    }
    if (tid < 32) {
        if (NUM_THREADS >= 64) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid + 32]);
        if (tid < 16) {
            if (NUM_THREADS >= 32) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid + 16]);
            if (NUM_THREADS >= 16) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  8]);
            if (NUM_THREADS >=  8) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  4]);
            if (NUM_THREADS >=  4) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  2]);
            if (NUM_THREADS >=  2) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  1]);
        }
        
        if (tid == 0) {
            max_dt_buffer[tid] = sdata_volatile[tid];
        }
    }
}
} // extern "C"





