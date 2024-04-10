

/**
  *  Generates two uniform random numbers based on the ANSIC Linear Congruential 
  *  Generator.
  */
__device__ float2 ansic_lcg(unsigned long long* seed_ptr) {
    unsigned long long seed = (*seed_ptr);
    double denum = 2147483648.0;
    unsigned long long modulo = 2147483647;

    seed = ((seed * 1103515245) + 12345) % modulo; //% 0x7fffffff;
    float u1 = seed / denum;

    seed = ((seed * 1103515245) + 12345) % modulo; //0x7fffffff;
    float u2 = seed / denum;

    (*seed_ptr) = seed;

    float2 out;
    out.x = u1;
    out.y = u2;

    return out;
    //return make_float2(u1, u2);
}
__device__ float2 rand_uniform(unsigned long long* seed_ptr) {
    return ansic_lcg(seed_ptr);
}

/**
  *  Generates two random numbers, drawn from a normal distribtion with mean 0 and
  *  variance 1. Based on the Box Muller transform.
  */
__device__ float2 boxMuller(float2 u) {
    float r = sqrt(-2.0f*log(u.x));
    float n1 = r*cospi(2*u.y);
    float n2 = r*sinpi(2*u.y);

    float2 out;
    out.x = n1;
    out.y = n2;
    return out;
}
__device__ float2 rand_normal(unsigned long long* seed_ptr) {
    float2 u = ansic_lcg(seed_ptr);
    return boxMuller(u);
}

/**
  * Kernel that generates uniform random numbers.
  */
extern "C" {
__global__ void uniformDistribution(
        // Size of data
        int seed_nx_, int seed_ny_,        
        int random_nx_,
        
        //Data
        unsigned long long* seed_ptr_, int seed_pitch_,
        float* random_ptr_, int random_pitch_
    ) {

    //Index of cell within domain
    const int ti = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int tj = (blockDim.y * blockIdx.y) + threadIdx.y;

    // Each thread computes and writes two uniform numbers.

    if ((ti < seed_nx_) && (tj < seed_ny_)) {
    
        //Compute pointer to current row in the U array
        unsigned long long* const seed_row = (unsigned long long*) ((char*) seed_ptr_ + seed_pitch_*tj);
        float* const random_row = (float*) ((char*) random_ptr_ + random_pitch_*tj);
        
        unsigned long long seed = seed_row[ti];
        float2 u = ansic_lcg(&seed);

        seed_row[ti] = seed;

        if (2*ti + 1 < random_nx_) {
            random_row[2*ti    ] = u.x;
            random_row[2*ti + 1] = u.y;
        }
        else if (2*ti == random_nx_) {
            random_row[2*ti    ] = u.x;
        }
    }
}
} // extern "C"

/**
  * Kernel that generates normal distributed random numbers with mean 0 and variance 1.
  */
extern "C" {
__global__ void normalDistribution(
        // Size of data
        int seed_nx_, int seed_ny_,
        int random_nx_,               // random_ny_ is equal to seed_ny_
        
        //Data
        unsigned long long* seed_ptr_, int seed_pitch_, // size [seed_nx, seed_ny]
        float* random_ptr_, int random_pitch_           // size [random_nx, seed_ny]
    ) {
    
    //Index of cell within domain
    const int ti = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int tj = (blockDim.y * blockIdx.y) + threadIdx.y;

    // Each thread computes and writes two uniform numbers.

    if ((ti < seed_nx_) && (tj < seed_ny_)) {
    
        //Compute pointer to current row in the U array
        unsigned long long* const seed_row = (unsigned long long*) ((char*) seed_ptr_ + seed_pitch_*tj);
        float* const random_row = (float*) ((char*) random_ptr_ + random_pitch_*tj);
        
        unsigned long long seed = seed_row[ti];
        float2 r = ansic_lcg(&seed);
        float2 u = boxMuller(r);

        seed_row[ti] = seed;

        if (2*ti + 1 < random_nx_) {
            random_row[2*ti    ] = u.x;
            random_row[2*ti + 1] = u.y;
        }
        else if (2*ti == random_nx_) {
            random_row[2*ti    ] = u.x;
        }
    }
}
} // extern "C"