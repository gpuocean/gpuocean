#ifndef COMMON_CU
#define COMMON_CU

/*
This software is part of GPU Ocean. 

Copyright (C) 2016-2019 SINTEF Digital
Copyright (C) 2017-2019 Norwegian Meteorological Institute

These CUDA kernels implement common functionality that is shared 
between multiple numerical schemes for solving the shallow water 
equations.

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

#define _180_OVER_PI 57.29578f
#define PI_OVER_180 0.01745329f
#define SQRT_OF_TWO 1.41421356237309504880f


inline __device__ float3 operator*(const float &a, const float3 &b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __device__ float3 operator/(const float3 &a, const float &b) {
    return make_float3(a.x/b, a.y/b, a.z/b);
}

inline __device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ __host__ float clamp(float f, float a, float b) {
    return fmaxf(a, fminf(f, b));
}

/**
  * Reads a block of data  with one ghost cell for the shallow water equations
  */
__device__ void readBlock1(float* h_ptr_, int h_pitch_,
                float* hu_ptr_, int hu_pitch_,
                float* hv_ptr_, int hv_pitch_,
                float Q[3][block_height+2][block_width+2], 
                const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of block within domain
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    
    //Read into shared memory
    for (int j=ty; j<block_height+2; j+=blockDim.y) {
        const int l = clamp(by + j, 0, ny_+1); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        float* const h_row = (float*) ((char*) h_ptr_ + h_pitch_*l);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*l);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*l);
        
        for (int i=tx; i<block_width+2; i+=blockDim.x) {
            const int k = clamp(bx + i, 0, nx_+1); // Out of bounds
            
            Q[0][j][i] = h_row[k];
            Q[1][j][i] = hu_row[k];
            Q[2][j][i] = hv_row[k];
        }
    }
}





/**
  * Reads a block of data  with two ghost cells for the shallow water equations
  */
__device__ void readBlock2(float* h_ptr_, int h_pitch_,
                float* hu_ptr_, int hu_pitch_,
                float* hv_ptr_, int hv_pitch_,
                float Q[3][block_height+4][block_width+4], 
                const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    //Index of block within domain
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    
    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=blockDim.y) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        float* const h_row = (float*) ((char*) h_ptr_ + h_pitch_*l);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*l);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*l);
        
        for (int i=tx; i<block_width+4; i+=blockDim.x) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds
            
            Q[0][j][i] = h_row[k];
            Q[1][j][i] = hu_row[k];
            Q[2][j][i] = hv_row[k];
        }
    }
}

/**
  * Reads a block of data  with two ghost cells for the shallow water equations,
  * while compensating for dry states
  */
__device__ void readBlock2DryStates(float* eta_ptr_, int eta_pitch_,
                float* hu_ptr_, int hu_pitch_,
                float* hv_ptr_, int hv_pitch_,
                float* Hm_ptr_, int Hm_pitch_,
                float Q[3][block_height+4][block_width+4], 
                const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    //Index of block within domain
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    
    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=blockDim.y) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*l);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*l);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*l);
        float* const Hm_row = (float*) ((char*) Hm_ptr_ + Hm_pitch_*l);
        
        for (int i=tx; i<block_width+4; i+=blockDim.x) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds
            
            Q[0][j][i] = max(eta_row[k], -Hm_row[k]);
            Q[1][j][i] = hu_row[k];
            Q[2][j][i] = hv_row[k];
        }
    }
}

/**
  * Reads a block of data  with two ghost cells for the shallow water equations
  */
__device__ void readBlock2single(float* data_ptr_, int data_pitch_,
		      float shmem[block_height+4][block_width+4],
		      const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    //Index of block within domain
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    
    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=blockDim.y) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        float* const data_row = (float*) ((char*) data_ptr_ + data_pitch_*l);
        
        for (int i=tx; i<block_width+4; i+=blockDim.x) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds
            shmem[j][i] = data_row[k];
        }
    }
}



/**
  * Writes a block of data to global memory for the shallow water equations.
  */
__device__ void writeBlock1(float* h_ptr_, int h_pitch_,
                 float* hu_ptr_, int hu_pitch_,
                 float* hv_ptr_, int hv_pitch_,
                 float Q[3][block_height+2][block_width+2],
                 const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 1; //Skip global ghost cells, i.e., +1
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    //Only write internal cells
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;

        float* const h_row  = (float*) ((char*) h_ptr_ + h_pitch_*tj);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*tj);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*tj);
        
        h_row[ti]  = Q[0][j][i];
        hu_row[ti] = Q[1][j][i];
        hv_row[ti] = Q[2][j][i];
    }
}





/**
  * Writes a block of data to global memory for the shallow water equations.
  */
__device__ void writeBlock2(float* h_ptr_, int h_pitch_,
                 float* hu_ptr_, int hu_pitch_,
                 float* hv_ptr_, int hv_pitch_,
                 float Q[3][block_height+4][block_width+4], 
                 const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 2;
    
    //Only write internal cells
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;

        float* const h_row  = (float*) ((char*) h_ptr_ + h_pitch_*tj);
        float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*tj);
        float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*tj);
        
        h_row[ti]  = Q[0][j][i];
        hu_row[ti] = Q[1][j][i];
        hv_row[ti] = Q[2][j][i];
    }
}






/**
  * No flow boundary conditions for the shallow water equations
  * with one ghost cell in each direction
  */
__device__ void noFlowBoundary1(float Q[3][block_height+2][block_width+2], const int nx_, const int ny_) {
    //Global index
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 1; //Skip global ghost cells, i.e., +1
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int i = tx + 1; //Skip local ghost cells, i.e., +1
    const int j = ty + 1;
    
    //Fix boundary conditions
    if (ti == 1) {
        Q[0][j][i-1] =  Q[0][j][i];
        Q[1][j][i-1] = -Q[1][j][i];
        Q[2][j][i-1] =  Q[2][j][i];
    }
    if (ti == nx_) {
        Q[0][j][i+1] =  Q[0][j][i];
        Q[1][j][i+1] = -Q[1][j][i];
        Q[2][j][i+1] =  Q[2][j][i];
    }
    if (tj == 1) {
        Q[0][j-1][i] =  Q[0][j][i];
        Q[1][j-1][i] =  Q[1][j][i];
        Q[2][j-1][i] = -Q[2][j][i];
    }
    if (tj == ny_) {
        Q[0][j+1][i] =  Q[0][j][i];
        Q[1][j+1][i] =  Q[1][j][i];
        Q[2][j+1][i] = -Q[2][j][i];
    }
}




/**
  * No flow boundary conditions for the shallow water equations
  * with two ghost cells in each direction
  *
  * BC values are defined as follows: 
  * 1: Wall boundary condition
  * 2: Periodic boundary condition
  * 3: Open boundary (numerical sponge)
  */
__device__ void noFlowBoundary2Mix(float Q[3][block_height+4][block_width+4],
			const int nx_, const int ny_,
			const int bc_north_, const int bc_east_,
			const int bc_south_, const int bc_west_) {
    
    //Global index
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 2;
    
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int i = tx + 2; //Skip local ghost cells, i.e., +2
    const int j = ty + 2;
    
    if (ti == 2 && bc_west_ == 1) {
	// Wall boundary on west
	Q[0][j][i-1] =  Q[0][j][i];
	Q[1][j][i-1] = -Q[1][j][i];
	Q[2][j][i-1] =  Q[2][j][i];
        
	Q[0][j][i-2] =  Q[0][j][i+1];
	Q[1][j][i-2] = -Q[1][j][i+1];
	Q[2][j][i-2] =  Q[2][j][i+1];
    }
    if (ti == nx_+1 && bc_east_ == 1) {
	// Wall boundary on east
	Q[0][j][i+1] =  Q[0][j][i];
	Q[1][j][i+1] = -Q[1][j][i];
	Q[2][j][i+1] =  Q[2][j][i];
        
	Q[0][j][i+2] =  Q[0][j][i-1];
	Q[1][j][i+2] = -Q[1][j][i-1];
	Q[2][j][i+2] =  Q[2][j][i-1];
    }
    if (tj == 2 && bc_south_ == 1) {
	// Wall boundary on south
	Q[0][j-1][i] =  Q[0][j][i];
	Q[1][j-1][i] =  Q[1][j][i];
	Q[2][j-1][i] = -Q[2][j][i];
        
	Q[0][j-2][i] =  Q[0][j+1][i];
	Q[1][j-2][i] =  Q[1][j+1][i];
	Q[2][j-2][i] = -Q[2][j+1][i];
    }
    if (tj == ny_+1 && bc_north_ == 1) {
	// Wall boundary on north
	Q[0][j+1][i] =  Q[0][j][i];
	Q[1][j+1][i] =  Q[1][j][i];
	Q[2][j+1][i] = -Q[2][j][i];
        
	Q[0][j+2][i] =  Q[0][j-1][i];
	Q[1][j+2][i] =  Q[1][j-1][i];
	Q[2][j+2][i] = -Q[2][j-1][i];
    }
}


/**
  * No flow boundary conditions for the shallow water equations
  * with two ghost cells in each direction
  */
__device__ void noFlowBoundary2(float Q[3][block_height+4][block_width+4], const int nx_, const int ny_, const int boundary_conditions_type_) {
    if (boundary_conditions_type_ == 2) {
	return;
    }
    int bc_north = 1;
    int bc_east = 1;
    int bc_south = 1;
    int bc_west = 1;
    if (boundary_conditions_type_ == 3) {
	bc_north = 2;
	bc_south = 2;
    }
    else if (boundary_conditions_type_ == 4) {
	bc_east = 2;
	bc_west = 2;
    }

    noFlowBoundary2Mix(Q, nx_, ny_, bc_north, bc_east, bc_south, bc_west);
}




/**
  * Evolves the solution in time along the x axis (dimensional splitting)
  */
__device__ void evolveF1(float Q[3][block_height+2][block_width+2],
              float F[3][block_height+1][block_width+1],
              const int nx_, const int ny_,
              const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 1; //Skip global ghost cells, i.e., +1
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;
        
        Q[0][j][i] = Q[0][j][i] + (F[0][ty][tx] - F[0][ty][tx+1]) * dt_ / dx_;
        Q[1][j][i] = Q[1][j][i] + (F[1][ty][tx] - F[1][ty][tx+1]) * dt_ / dx_;
        Q[2][j][i] = Q[2][j][i] + (F[2][ty][tx] - F[2][ty][tx+1]) * dt_ / dx_;
    }
}






/**
  * Evolves the solution in time along the x axis (dimensional splitting)
  */
__device__ void evolveF2(float Q[3][block_height+4][block_width+4],
              float F[3][block_height+1][block_width+1],
              const int nx_, const int ny_,
              const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;    
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 2;
    
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +1
        const int j = ty + 2;
        
        Q[0][j][i] = Q[0][j][i] + (F[0][ty][tx] - F[0][ty][tx+1]) * dt_ / dx_;
        Q[1][j][i] = Q[1][j][i] + (F[1][ty][tx] - F[1][ty][tx+1]) * dt_ / dx_;
        Q[2][j][i] = Q[2][j][i] + (F[2][ty][tx] - F[2][ty][tx+1]) * dt_ / dx_;
    }
}






/**
  * Evolves the solution in time along the y axis (dimensional splitting)
  */
__device__ void evolveG1(float Q[3][block_height+2][block_width+2],
              float G[3][block_height+1][block_width+1],
              const int nx_, const int ny_,
              const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;    
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 1; //Skip global ghost cells, i.e., +1
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;
        
        Q[0][j][i] = Q[0][j][i] + (G[0][ty][tx] - G[0][ty+1][tx]) * dt_ / dy_;
        Q[1][j][i] = Q[1][j][i] + (G[1][ty][tx] - G[1][ty+1][tx]) * dt_ / dy_;
        Q[2][j][i] = Q[2][j][i] + (G[2][ty][tx] - G[2][ty+1][tx]) * dt_ / dy_;
    }
}







/**
  * Evolves the solution in time along the y axis (dimensional splitting)
  */
__device__ void evolveG2(float Q[3][block_height+4][block_width+4],
              float G[3][block_height+1][block_width+1],
              const int nx_, const int ny_,
              const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 2;
    
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;
        
        Q[0][j][i] = Q[0][j][i] + (G[0][ty][tx] - G[0][ty+1][tx]) * dt_ / dy_;
        Q[1][j][i] = Q[1][j][i] + (G[1][ty][tx] - G[1][ty+1][tx]) * dt_ / dy_;
        Q[2][j][i] = Q[2][j][i] + (G[2][ty][tx] - G[2][ty+1][tx]) * dt_ / dy_;
    }
}










/**
  * Reconstructs a slope using the minmod limiter based on three 
  * consecutive values
  */
__device__ float minmodSlope(float left, float center, float right, float theta) {
    const float backward = (center - left) * theta;
    const float central = (right - left) * 0.5f;
    const float forward = (right - center) * theta;
    
	return 0.25f
		*copysign(1.0f, backward)
		*(copysign(1.0f, backward) + copysign(1.0f, central))
		*(copysign(1.0f, central) + copysign(1.0f, forward))
		*min( min(fabs(backward), fabs(central)), fabs(forward) );
}

__device__ float minmodRaw(float backward, float central, float forward) {

    return 0.25f
	*copysign(1.0f, backward)
	*(copysign(1.0f, backward) + copysign(1.0f, central))
	*(copysign(1.0f, central) + copysign(1.0f, forward))
	*min( min(fabs(backward), fabs(central)), fabs(forward) );
}


/**
  * Reconstructs a minmod slope for a whole block along x
  */
__device__ void minmodSlopeX(float  Q[3][block_height+4][block_width+4],
                  float Qx[3][block_height+2][block_width+2],
                  const float theta_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Reconstruct slopes along x axis
    for (int j=ty; j<block_height; j+=blockDim.y) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+2; i+=blockDim.x) {
            const int k = i + 1;
            for (int p=0; p<3; ++p) {
                Qx[p][j][i] = 0.5f * minmodSlope(Q[p][l][k-1], Q[p][l][k], Q[p][l][k+1], theta_);
            }
        }
    }
}



/**
  * Reconstructs a minmod slope for a whole block along y
  */
__device__ void minmodSlopeY(float  Q[3][block_height+4][block_width+4],
                  float Qy[3][block_height+2][block_width+2],
                  const float theta_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    for (int j=ty; j<block_height+2; j+=blockDim.y) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=blockDim.x) {            
            const int k = i + 2; //Skip ghost cells
            for (int p=0; p<3; ++p) {
                Qy[p][j][i] = 0.5f * minmodSlope(Q[p][l-1][k], Q[p][l][k], Q[p][l+1][k], theta_);
            }
        }
    }
}



texture<float, cudaTextureType2D> windstress_X_current;
texture<float, cudaTextureType2D> windstress_X_next;

texture<float, cudaTextureType2D> windstress_Y_current;
texture<float, cudaTextureType2D> windstress_Y_next;


/**
  * Returns the wind stress, trilinearly interpolated in space and time
  * @param wind_stress_t_ \in [0, 1] determines the temporal interpolation (0=current, 1=next)
  * @param ti_ Location of this thread along the x-axis in number of cells (NOTE: half indices)
  * @param tj_ Location of this thread along the y-axis in number of cells (NOTE: half indices)
  * @param nx_ Number of cells along x axis
  * @param ny_ Number of cells along y axis
  */
__device__ float windStressX(float wind_stress_t_, float ti_, float tj_, int nx_, int ny_) {
    
    //Normalize coordinates (to [0, 1])
    const float s = ti_ / float(nx_);
    const float t = tj_ / float(ny_);
    
    //Look up current and next timestep (using bilinear texture interpolation)
    float current = tex2D(windstress_X_current, s, t);
    float next = tex2D(windstress_X_next, s, t);
    
    //Interpolate in time
    return wind_stress_t_*next + (1.0f - wind_stress_t_)*current;
}

/**
  * Returns the wind stress, trilinearly interpolated in space and time
  * @param wind_stress_t_ \in [0, 1] determines the temporal interpolation (0=current, 1=next)
  * @param ti_ Location of this thread along the x-axis in number of cells (NOTE: half indices)
  * @param tj_ Location of this thread along the y-axis in number of cells (NOTE: half indices)
  * @param nx_ Number of cells along x axis
  * @param ny_ Number of cells along y axis
  */
__device__ float windStressY(float wind_stress_t_, float ti_, float tj_, int nx_, int ny_) {
    
    //Normalize coordinates (to [0, 1])
    const float s = ti_ / float(nx_);
    const float t = tj_ / float(ny_);
    
    //Look up current and next timestep (using bilinear texture interpolation)
    float current = tex2D(windstress_Y_current, s, t);
    float next = tex2D(windstress_Y_next, s, t);
    
    //Interpolate in time
    return wind_stress_t_*next + (1.0f - wind_stress_t_)*current;
}


__device__ float3 F_func(const float3 Q, const float g) {
    float3 F;

    F.x = Q.y;                              //hu
    F.y = Q.y*Q.y / Q.x + 0.5f*g*Q.x*Q.x;   //hu*hu/h + 0.5f*g*h*h;
    F.z = Q.y*Q.z / Q.x;                    //hu*hv/h;

    return F;
}


/**
  * Central upwind flux function
  * Takes Q = [h, hu, hv] as input, not [w, hu, hv].
  */
__device__ float3 CentralUpwindFlux(const float3 Qm, float3 Qp, const float g) {
    const float3 Fp = F_func(Qp, g);
    const float up = Qp.y / Qp.x;   // hu / h
    const float cp = sqrt(g*Qp.x); // sqrt(g*h)

    const float3 Fm = F_func(Qm, g);
    const float um = Qm.y / Qm.x;   // hu / h
    const float cm = sqrt(g*Qm.x); // sqrt(g*h)
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed
    
    return ((ap*Fm - am*Fp) + ap*am*(Qp-Qm))/(ap-am);
}


__device__ float3 F_func_bottom(const float3 Q, const float h, const float u, const float g) {
    float3 F;

    F.x = Q.y;                       //hu
    F.y = Q.y*u + 0.5f*g*(h*h);      //hu*u + 0.5f*g*h*h;
    F.z = Q.z*u;                     //hv*u;

    return F;
}


/**
  *  Struct and functions for doing bicubic interpolation.
  *  We use four float4 variables instead of an array in order to 
  *  keep the variables in registers rather than L1 cache.
  */

typedef struct
{
    float4 m_row0;
    float4 m_row1;
    float4 m_row2;
    float4 m_row3;
    
}Matrix4x4_d;

__device__ inline float dotProduct(const float4 v1, const float4 v2) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w;
}

/**
  *  Calculating the coefficient matrix for bicubic interpolation.
  *  Input matrix is evaluation of function values and derivatives in the corners
  *  of a unit square:
  *  f = [[ f00,  f01,  fy00,  fy01],
  *       [ f10,  f11,  fy10,  fy11],
  *       [fx00, fx01, fxy00, fxy01],
  *       [fx10, fx11, fxy10, fxy11] ]
  */
__device__ Matrix4x4_d bicubic_interpolation_coefficients(const Matrix4x4_d f) {
    Matrix4x4_d b;
    // [[ 1.0f,  0.0f,  0.0f,  0.0f)],
    //  [ 0.0f,  0.0f,  1.0f,  0.0f)],
    //  [-3.0f,  3.0f, -2.0f, -1.0f)],
    //  [ 2.0f, -2.0f,  1.0f,  1.0f)]]
    b.m_row0.x= 1.0f; b.m_row0.y= 0.0f; b.m_row0.z= 0.0f; b.m_row0.w= 0.0f;
    b.m_row1.x= 0.0f; b.m_row1.y= 0.0f; b.m_row1.z= 1.0f; b.m_row1.w= 0.0f;
    b.m_row2.x=-3.0f; b.m_row2.y= 3.0f; b.m_row2.z=-2.0f; b.m_row2.w=-1.0f;
    b.m_row3.x= 2.0f; b.m_row3.y=-2.0f; b.m_row3.z= 1.0f; b.m_row3.w= 1.0f;
    
    // Obtain fb = f * b^T, but store the result as its transpose:
    // fb[row i, col j]   = f[row i] dot b^T[col j] 
    //                    = f[row i] dot b[row j]
    // fb^T[row i, col j] = f[row j] dot b[row i]
    Matrix4x4_d fb_transpose;
    /*
    Below is the loop unrolled version of the following loop:
    for (int i = 0; i < 4; i++) {
        fb_transpose.m_row[i] = make_float4(dotProduct(f.m_row0, b.m_row[i]),
                                            dotProduct(f.m_row1, b.m_row[i]),
                                            dotProduct(f.m_row2, b.m_row[i]),
                                            dotProduct(f.m_row3, b.m_row[i]));
    }*/
    fb_transpose.m_row0.x = dotProduct(f.m_row0, b.m_row0);
    fb_transpose.m_row0.y = dotProduct(f.m_row1, b.m_row0);
    fb_transpose.m_row0.z = dotProduct(f.m_row2, b.m_row0);
    fb_transpose.m_row0.w = dotProduct(f.m_row3, b.m_row0);
    
    fb_transpose.m_row1.x = dotProduct(f.m_row0, b.m_row1);
    fb_transpose.m_row1.y = dotProduct(f.m_row1, b.m_row1);
    fb_transpose.m_row1.z = dotProduct(f.m_row2, b.m_row1);
    fb_transpose.m_row1.w = dotProduct(f.m_row3, b.m_row1);
    
    fb_transpose.m_row2.x = dotProduct(f.m_row0, b.m_row2);
    fb_transpose.m_row2.y = dotProduct(f.m_row1, b.m_row2);
    fb_transpose.m_row2.z = dotProduct(f.m_row2, b.m_row2);
    fb_transpose.m_row2.w = dotProduct(f.m_row3, b.m_row2);
    
    fb_transpose.m_row3.x = dotProduct(f.m_row0, b.m_row3);
    fb_transpose.m_row3.y = dotProduct(f.m_row1, b.m_row3);
    fb_transpose.m_row3.z = dotProduct(f.m_row2, b.m_row3);
    fb_transpose.m_row3.w = dotProduct(f.m_row3, b.m_row3);
    
    // Obtain out = b * f * b^T = b * fb
    // out[row i, col j] = b[row i] dot fb[col j]
    //                   = b[row i] dot fb^T[row j]
    Matrix4x4_d out;
    /*
    Below is the loop unrolled version of the following loop:
    for (int i = 0; i < 4; i++) {
        out.m_row[i] = make_float4(dotProduct(b.m_row[i], fb_transpose.m_row[0]),
                                   dotProduct(b.m_row[i], fb_transpose.m_row[1]),
                                   dotProduct(b.m_row[i], fb_transpose.m_row[2]),
                                   dotProduct(b.m_row[i], fb_transpose.m_row[3]));
    }*/
    
    out.m_row0.x = dotProduct(b.m_row0, fb_transpose.m_row0);
    out.m_row0.y = dotProduct(b.m_row0, fb_transpose.m_row1);
    out.m_row0.z = dotProduct(b.m_row0, fb_transpose.m_row2);
    out.m_row0.w = dotProduct(b.m_row0, fb_transpose.m_row3);
    
    out.m_row1.x = dotProduct(b.m_row1, fb_transpose.m_row0);
    out.m_row1.y = dotProduct(b.m_row1, fb_transpose.m_row1);
    out.m_row1.z = dotProduct(b.m_row1, fb_transpose.m_row2);
    out.m_row1.w = dotProduct(b.m_row1, fb_transpose.m_row3);
    
    out.m_row2.x = dotProduct(b.m_row2, fb_transpose.m_row0);
    out.m_row2.y = dotProduct(b.m_row2, fb_transpose.m_row1);
    out.m_row2.z = dotProduct(b.m_row2, fb_transpose.m_row2);
    out.m_row2.w = dotProduct(b.m_row2, fb_transpose.m_row3);
    
    out.m_row3.x = dotProduct(b.m_row3, fb_transpose.m_row0);
    out.m_row3.y = dotProduct(b.m_row3, fb_transpose.m_row1);
    out.m_row3.z = dotProduct(b.m_row3, fb_transpose.m_row2);
    out.m_row3.w = dotProduct(b.m_row3, fb_transpose.m_row3);
    
    return out;
}

__device__ float bicubic_evaluation(const float4 x, 
                                    const float4 y, 
                                    const Matrix4x4_d coeff) 
{
    // out = x^T * coeff * y
    // out = x^T * temp
    
    // tmp[i] = coeff[row i] * y
    float4 tmp;
    tmp.x = dotProduct(coeff.m_row0, y);
    tmp.y = dotProduct(coeff.m_row1, y);
    tmp.z = dotProduct(coeff.m_row2, y);
    tmp.w = dotProduct(coeff.m_row3, y);
                                   
    return dotProduct(x, tmp);
}



__device__ float desingularize(const float& h, const float& hu, const float& eps) {
    return hu / fmaxf(fminf(h*h/(2.0f*eps)+0.5*eps, eps), fabsf(h));
}


#endif // COMMON_CU
