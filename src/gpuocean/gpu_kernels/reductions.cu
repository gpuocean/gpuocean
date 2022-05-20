/*
This software is part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

Implements reduction kernels needed for GPU Ocean.

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

// Define total number of threads that should work together to 
// carry out the reduction. Value 128 chosen arbitrarily.
#define NUM_THREADS 128

extern "C" {
__global__ void squareSum(
        //Discretization parameters
        const int nx_, const int ny_,
        float* data_ptr_,
        float* result)
{

    __shared__ float sdata[NUM_THREADS];
    volatile float* sdata_volatile = sdata;
    
    unsigned int tid = threadIdx.x;
    const int numElements = nx_*ny_;

    // Square each elements and reduce to "NUM_THREADS" elements
    float threadSum = 0.0f;
    for (unsigned int i = tid; i < numElements; i += NUM_THREADS) {
        threadSum += data_ptr_[i]*data_ptr_[i];
    }
    sdata[tid] = threadSum;
    __syncthreads();

    //Now, sum all elements into a single element
    if (NUM_THREADS >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (NUM_THREADS >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (NUM_THREADS >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    if (tid < 32) {
        if (NUM_THREADS >= 64) sdata_volatile[tid] += sdata_volatile[tid + 32];
        if (tid < 16) {
            if (NUM_THREADS >= 32) sdata_volatile[tid] += sdata_volatile[tid + 16];
            if (NUM_THREADS >= 16) sdata_volatile[tid] += sdata_volatile[tid +  8];
            if (NUM_THREADS >=  8) sdata_volatile[tid] += sdata_volatile[tid +  4];
            if (NUM_THREADS >=  4) sdata_volatile[tid] += sdata_volatile[tid +  2];
            if (NUM_THREADS >=  2) sdata_volatile[tid] += sdata_volatile[tid +  1];
        }
        
        if (tid == 0) {
            result[tid] = sdata_volatile[tid];
        }
    }
}
} // extern "C"



extern "C" {
__global__ void squareSumDouble(
        //Discretization parameters
        const int nx_, const int ny_,
        float* data_ptr_1,
        float* data_ptr_2,
        float* result)
{

    __shared__ float snorm1[NUM_THREADS];
    __shared__ float snorm2[NUM_THREADS];
    __shared__ float sdot[NUM_THREADS];
    volatile float* snorm1_volatile = snorm1;
    volatile float* snorm2_volatile = snorm2;
    volatile float* sdotpr_volatile = sdot;
    
    // The input data is interpreted as one dimensional buffers
    unsigned int tid = threadIdx.x;
    const int numElements = nx_*ny_;

    // Square each elements and reduce to "NUM_THREADS" elements
    float threadNorm1 = 0.0f;
    float threadNorm2 = 0.0f;
    float threadDot   = 0.0f;
    float f1, f2;
    for (unsigned int i = tid; i < numElements; i += NUM_THREADS) {
        f1 = data_ptr_1[i];
        f2 = data_ptr_2[i];
        threadNorm1 += f1*f1;
        threadNorm2 += f2*f2;
        threadDot   += f1*f2;
    }
    snorm1[tid] = threadNorm1;
    snorm2[tid] = threadNorm2;
    sdot[tid] = threadDot;
    __syncthreads();

    //Now, sum all elements into a single element
    if (NUM_THREADS >= 512) {
        if (tid < 256) {
            snorm1[tid] += snorm1[tid + 256];
            snorm2[tid] += snorm2[tid + 256];
            sdot[tid] += sdot[tid + 256];
        }
        __syncthreads();
    }
    if (NUM_THREADS >= 256) {
        if (tid < 128){
            snorm1[tid] += snorm1[tid + 128];
            snorm2[tid] += snorm2[tid + 128];
            sdot[tid] += sdot[tid + 128];
        }
        __syncthreads();
    }
    if (NUM_THREADS >= 128) {
        if (tid < 64) {
            snorm1[tid] += snorm1[tid + 64];
            snorm2[tid] += snorm2[tid + 64];
            sdot[tid] += sdot[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        if (NUM_THREADS >= 64) {
            snorm1_volatile[tid] += snorm1_volatile[tid + 32];
            snorm2_volatile[tid] += snorm2_volatile[tid + 32];
            sdotpr_volatile[tid] += sdotpr_volatile[tid + 32];
        }
        if (tid < 16) {
            if (NUM_THREADS >= 32) {
                snorm1_volatile[tid] += snorm1_volatile[tid + 16];
                snorm2_volatile[tid] += snorm2_volatile[tid + 16];
                sdotpr_volatile[tid] += sdotpr_volatile[tid + 16];
            }
            if (NUM_THREADS >= 16) {
                snorm1_volatile[tid] += snorm1_volatile[tid +  8];
                snorm2_volatile[tid] += snorm2_volatile[tid +  8];
                sdotpr_volatile[tid] += sdotpr_volatile[tid +  8];
            }
            if (NUM_THREADS >=  8) {
                snorm1_volatile[tid] += snorm1_volatile[tid +  4];
                snorm2_volatile[tid] += snorm2_volatile[tid +  4];
                sdotpr_volatile[tid] += sdotpr_volatile[tid +  4];
            }
            if (NUM_THREADS >=  4) {
                snorm1_volatile[tid] += snorm1_volatile[tid +  2];
                snorm2_volatile[tid] += snorm2_volatile[tid +  2];
                sdotpr_volatile[tid] += sdotpr_volatile[tid +  2];
            }
            if (NUM_THREADS >=  2) {
                snorm1_volatile[tid] += snorm1_volatile[tid +  1];
                snorm2_volatile[tid] += snorm2_volatile[tid +  1];
                sdotpr_volatile[tid] += sdotpr_volatile[tid +  1];
            }
        }
        
        if (tid == 0) {
            result[0] = snorm1_volatile[tid];
            result[1] = snorm2_volatile[tid];
            result[2] = sdotpr_volatile[tid];
        }
    }
}
} // extern "C"


